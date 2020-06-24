from networks import Actor, Critic
from utils import OUNoise, ReplayBuffer

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from collections import deque
PATH = os.path.abspath(os.getcwd())

class MADDPG():
    """ Multi Agent Deep Deterministic Policy Gradients Agent used to interaction with and learn from an environment """

    def __init__(self, state_size: int, action_size: int, num_agents: int, epsilon: float, seed: int):
        """ Initialize a MADDPG Agent Object
        :param state_size: dimension of state (input)
        :param action_size: dimension of action (output)
        :param num_agents: number of concurrent agents in the environment
        :param epsilon: initial value of epsilon for exploration
        :param random_seed: random seed
        """
        # TODO: What parameters do we not need for MADDPG
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.full_state_size = self.state_size*self.num_agents
        self.full_action_size = self.action_size*self.num_agents
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.t_step = 0

        # Hyperparameters
        self.buffer_size = 100000
        self.batch_size = 256
        self.update_every = 2
        self.num_updates = 5
        self.gamma = 0.95
        self.tau = 0.02
        self.lr_actor = 0.0001
        self.lr_critic = 0.0004
        self.weight_decay = 0
        self.epsilon = epsilon
        self.epsilon_decay = 0.995 # around 1100 episodes exploring will hit min
        self.epsilon_min = 0.005

        # Setup up all Agents
        # decentralized actors
        self.actors = [MADDPG_Actor(state_size, action_size, self.seed) for _ in range(num_agents)]
        # centralized critic
        self.critic_local = Critic(self.full_state_size, self.full_action_size, self.seed).to(self.device)
        self.critic_target = Critic(self.full_state_size, self.full_action_size, self.seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = self.lr_critic, weight_decay = self.weight_decay)

        # Replay Buffer Setup
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def __str__(self):
        return "MADDPG_Agent"

    # TODO: Rewrite train
    def train(self, env, brain_name, num_episodes=2500, print_every=10):
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
            states = env_info.vector_observations
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
            if episode_num % print_every == 0:
                print(f'Episode: {episode_num} \tAverage Score: {round(np.mean(scores_deque), 3)}')
                for i,actor in enumerate(self.actors):
                    torch.save(actor.actor_local.state_dict(), f'{PATH}\checkpoints\{actor.__str__()}_{i}_Actor.pth')
                torch.save(self.critic_local.state_dict(), f'{PATH}\checkpoints\{self.__str__()}_Critic.pth')

        # -------- All Episodes finished Save parameters and scores --------#
        # Save Model Parameters
        for i, actor in enumerate(self.actors):
            torch.save(actor.actor_local.state_dict(), f'{PATH}\checkpoints\{actor.__str__()}_{i}_Actor.pth')
        torch.save(self.critic_local.state_dict(), f'{PATH}\checkpoints\{self.__str__()}_Critic.pth')
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
        """ what the agent needs to do for every time step that occurs in the environment. Takes
        in a (s,a,r,s',d) tuple and saves it to memory and learns from experiences. Note: this is not
        the same as a step in the environment. Step is only called once per environment time step.
        :param states: array of states agents used to select actions
        :param actions: array of actions taken by agents
        :param rewards: array of rewards from the environment for the last actions taken by the agents
        :param next_states: array of next states after actions were taken by the agents
        :param dones: array of bools representing if environment is finished or not
        """
        self.memory.add(states, actions, rewards, next_states, dones) # Note these contain info for both agents

        # Learn "num_updates" times every "update_every" time step
        self.t_step += 1
        if len(self.memory) > self.batch_size and self.t_step%self.update_every == 0:
            self.t_step = 0
            for _ in range(self.num_updates):
                self.learn(self.memory.sample())

    def act(self, states, epsilon, add_noise=True):
        """ Returns actions for each agent given states as per current policy. Policy comes from the actor network.
        :param states: array of states from the environment for each agent
        :param epsilon: probability of exploration
        :param add_noise: bool on whether or not to potentially have exploration for action
        :return: clipped actions for all agents from the local networks
        """
        actions = [actor.act(local_obs, epsilon) for actor, local_obs in zip(self.actors,states)]
        return actions

    def target_act(self, obs_full):
        """ Returns actions for each agent given states as per current policy. Policy comes from the actor network.
        :param states: array of states from the environment for each agent
        :param epsilon: probability of exploration
        :param add_noise: bool on whether or not to potentially have exploration for action
        :return: clipped actions for all agents from the local networks
        """
        actions = [actor.target_act(local_obs) for actor, local_obs in zip(self.actors,obs_full)]
        return actions

    def reset_noise(self):
        """ resets to noise parameters for all actors """
        for actor in self.actors:
            actor.reset_noise()

    # TODO: Rewrite learn
    def learn(self, experiences):
        """ Update actor and critic networks for agent using a given batch of experiences """

        full_states, full_actions, rewards, full_next_states, dones = experiences

        # -------------------- Update Critic -------------------- #
        # Setup Data
        critic_obs = full_states.view(-1, self.full_state_size)
        critic_next_obs = full_next_states.view(-1, self.full_state_size)
        critic_actions = full_actions.reshape(-1,self.full_action_size)

        print(full_next_states.shape)
        print(np.array(self.target_act(full_next_states)).shape)
        time.sleep(3)


        # Use target networks for getting next actions and q values and calculate q_targets
        critic_next_actions = torch.from_numpy(np.array(self.target_act(full_next_states))).reshape(-1,self.full_action_size).to(self.device)
        # Calculate the q_targets
        next_q_targets = self.critic_target(critic_next_obs, critic_next_actions)
        q_targets = rewards + self.gamma*next_q_targets*(1-dones)

       # TODO: Understand this better
        # Compute critic loss
        q_expected = self.critic_local(critic_obs,critic_actions)
        critic_loss = 0
        for i in range(self.num_agents):
            critic_loss += F.mse_loss(q_expected, q_targets[:,i].detach().reshape(-1,1)) / self.num_agents
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------- Update Actor --------------------- #
        # Computer actor loss (maximize mean of Q(states,actions))
        # Optimizer minimizes and we want to maximize so multiply by -1
        actor_loss = -1*self.critic_local(full_states, full_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #---------------- Update Target Networks ---------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        for actor in self.actors:
            actor.soft_update(actor.actor_local, actor.actor_target, self.tau)


    def soft_update(self, local_network, target_network, tau):
        """ soft update newtwork parametes
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_network: PyTorch Network that is always up to date
        :param target_network: PyTorch Network that is not up to date
        :param tau: update (interpolation) parameter
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class MADDPG_Actor():
    """ This is a decentralized actor used by the MADDPG Agent """

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
        self.tau = 0.001
        self.lr_actor = 0.0001

        # Setup Networks (Actor: State -> Action, Critic: (States for all agents, Actions for all agents) -> Value)
        self.actor_local = Actor(self.state_size, self.action_size,  self.seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = self.lr_actor)

        # Initialize local and taret networks to start with same parameters
        self.soft_update(self.actor_local, self.actor_target, tau=1)

        # Noise Setup
        self.noise = OUNoise(self.action_size, self.seed)

    def __str__(self):
        return "DDPG_Agent"

    def reset_noise(self):
        """ resets to noise parameters """
        self.noise.reset()

    def act(self, obs, epsilon, add_noise=True):
        """ Returns actions for given states as per current policy. Policy comes from the actor network.
        :param obs: observations for this individual agent
        :param epsilon: probability of exploration
        :param add_noise: bool on whether or not to potentially have exploration for action
        :return: clipped actions
        """
        state = torch.from_numpy(obs).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise and epsilon > np.random.random():
            actions += self.noise.sample()
        return np.clip(actions, -1,1)

    def target_act(self, obs):
        """ Used for selectiong actions for next_states. No exploration needed

        :param obs: observations for this indiviual actoin (next_state)
        :return: numpy array of clipped actions for this agent
        """
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(obs).cpu().data.numpy()
        self.actor_target.train()
        return np.clip(action, -1, 1)


    def soft_update(self, local_network, target_network, tau):
        """ soft update newtwork parametes
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_network: PyTorch Network that is always up to date
        :param target_network: PyTorch Network that is not up to date
        :param tau: update (interpolation) parameter
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)