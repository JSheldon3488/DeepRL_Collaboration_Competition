from networks import Actor, Critic
from utils import OUNoise, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import os
PATH = os.path.abspath(os.getcwd())

class MADDPG():
    """ Multi Agent Deep Deterministic Policy Gradients Agent used to interaction with and learn from an environment.
        This current iteration is not actually MADDPG. It is two individual DDPG agents with a wrapper class MADDPG
        that handles using two seperate agents. (Could not get true MADDPG to work)
     """

    def __init__(self, state_size: int, action_size: int, num_agents: int, epsilon: float, seed: int):
        """ Initialize a MADDPG Agent Object
        :param state_size: dimension of state per agent (input)
        :param action_size: dimension of action per agent (output)
        :param num_agents: number of concurrent agents in the environment
        :param epsilon: initial value of epsilon for exploration
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed
        self.t_step = 0

        # Hyperparameters
        self.update_every = 1
        self.num_updates = 1
        self.epsilon = epsilon
        self.epsilon_decay = .998
        self.epsilon_min = 0

        # Setup up all Agents
        self.agents = [DDPG(self.state_size, self.action_size, self.seed) for _ in range(self.num_agents)]

    def __str__(self):
        return "MADDPG_Agent"

    def train(self, env, brain_name, num_episodes=1000, print_every=25):
        """ Interacts with and learns from a given Unity Environment
        :param env: Unity Environment the agents is trying to learn
        :param brain_name: Brain for Environment
        :param num_episodes: Number of episodes to train
        :param print_every: How often in episodes to print a running average
        :return: Returns episodes scores and 100 episode averages as lists
        """
        # --------- Set Everything up --------#
        scores = []
        avg_scores = []
        scores_deque = deque(maxlen=100)

        # -------- Simulation Loop --------#
        for episode_num in range(1, num_episodes + 1):
            # Reset everything
            episode_scores = np.zeros(self.num_agents)
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations #[2,24]
            self.reset_noise()

            # Run the episode
            while True:
                actions = self.act(states, self.epsilon)
                env_info = env.step(actions)[brain_name]
                next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
                self.step(states, actions, rewards, next_states, dones)
                episode_scores += rewards
                states = next_states
                if np.any(dones):
                    break

            # -------- Episode Finished ---------#
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            scores.append(np.max(episode_scores))
            scores_deque.append(np.max(episode_scores))
            avg_scores.append(np.mean(scores_deque))
            max_score_last_100 = max(scores_deque)
            if episode_num % print_every == 0:
                print(f'Episode: {episode_num} \tAverage Score: {round(np.mean(scores_deque), 3)}, Max Last 100: {round(max_score_last_100,3)}')
                for i,agent in enumerate(self.agents):
                    torch.save(agent.actor_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_{i}_Actor.pth')
                    torch.save(agent.critic_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_{i}_Critic.pth')

        # -------- All Episodes finished Save parameters and scores --------#
        # Save Model Parameters
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_{i}_Actor.pth')
            torch.save(agent.critic_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_{i}_Critic.pth')
        # Save Scores to file
        f = open(f'{PATH}\scores\{self.__str__()}_Scores.txt', 'w')
        scores_string = "\n".join([str(score) for score in scores])
        f.write(scores_string)
        f.close()
        # Save average scores for 100 window average
        f = open(f'{PATH}\scores\{self.__str__()}_AvgScores.txt', 'w')
        avgScores_string = "\n".join([str(score) for score in avg_scores])
        f.write(avgScores_string)
        f.close()
        return scores, avg_scores

    def step(self, states, actions, rewards, next_states, dones):
        """ parse and store the experience for each individual agent and call individual agent step every update_every timesteps

        :param states: array of states agents used to select actions
        :param actions: array of actions taken by agents
        :param rewards: array of rewards from the environment for the last actions taken by the agents
        :param next_states: array of next states after actions were taken by the agents
        :param dones: array of bools representing if environment is finished or not
        """
        self.t_step += 1
        for i,agent in enumerate(self.agents):
            agent.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        if self.t_step%self.update_every == 0:
            self.t_step = 0
            for agent in self.agents:
                agent.step()

    def act(self, states, epsilon):
        """ Returns actions for each agent given states as per current policy. Policy comes from the agents actor network.
        :param states: array of states from the environment for each agent
        :param epsilon: probability of exploration
        :return: clipped actions for all agents from the local networks
        """
        actions = [agent.act(local_obs, epsilon) for agent, local_obs in zip(self.agents,states)]
        return np.asarray(actions)

    def reset_noise(self):
        """ resets to noise parameters for all agents """
        for agent in self.agents:
            agent.reset_noise()

class DDPG():
    """ This is an Individual DDPG Agent """

    def __init__(self, state_size, action_size, seed):
        """ Initialize a DDPG Agent Object
        :param state_size: dimension of state (input) for this decentralized actor
        :param action_size: dimension of action (output) for this decentralized actor
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.buffer_size = 100000
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.01
        self.lr_actor = 0.0001
        self.lr_critic = 0.001

        # Setup Networks (Actor: State -> Action, Critic: (States for all agents, Actions for all agents) -> Value)
        self.actor_local = Actor(self.state_size, self.action_size,  self.seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = self.lr_actor)
        self.critic_local = Critic(self.state_size, self.action_size, self.seed).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size, self.seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = self.lr_critic)

        # Initialize local and taret networks to start with same parameters
        self.soft_update(self.actor_local, self.actor_target, tau=1)
        self.soft_update(self.critic_local, self.critic_target, tau=1)

        # Noise Setup
        self.noise = OUNoise(self.action_size, self.seed)

        # Replay Buffer Setup
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def __str__(self):
        return "DDPG_Agent"

    def reset_noise(self):
        """ resets to noise parameters """
        self.noise.reset()

    def act(self, state, epsilon, add_noise=True):
        """ Returns actions for given states as per current policy. Policy comes from the actor network.
        :param state: observations for this individual agent
        :param epsilon: probability of exploration
        :param add_noise: bool on whether or not to potentially have exploration for action
        :return: clipped actions
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise and epsilon > np.random.random():
            actions += self.noise.sample()
        return np.clip(actions, -1,1)

    def step(self):
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """ Update actor and critic networks using a given batch of experiences
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(states) -> actions
            critic_target(states, actions) -> Q-value
        :param experiences: tuple of arrays (states, actions, rewards, next_states, dones)  sampled from the replay buffer
        """

        states, actions, rewards, next_states, dones = experiences
        # -------------------- Update Critic -------------------- #
        # Use target networks for getting next actions and q values and calculate q_targets
        next_actions = self.actor_target(next_states)
        next_q_targets = self.critic_target(next_states, next_actions)
        q_targets = rewards + (self.gamma * next_q_targets * (1 - dones))
        # Compute critic loss (Same as DQN Loss)
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------- Update Actor --------------------- #
        # Computer actor loss (maximize mean of Q(states,actions))
        action_preds = self.actor_local(states)
        # Optimizer minimizes and we want to maximize so multiply by -1
        actor_loss = -1 * self.critic_local(states, action_preds).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------- Update Target Networks ---------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_network, target_network, tau):
        """ soft update newtwork parametes
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_network: PyTorch Network that is always up to date
        :param target_network: PyTorch Network that is not up to date
        :param tau: update (interpolation) parameter
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)