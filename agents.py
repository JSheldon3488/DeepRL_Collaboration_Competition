from networks import Actor, Critic
from utils import OUNoise, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from collections import deque
PATH = os.path.abspath(os.getcwd()) + '\\Udacity_DeepRL_Collaboration_Competition'

class MADDPG():
    """ Multi Agent Deep Deterministic Policy Gradients Agent used to interaction with and learn from an environment """

    def __init__(self, state_size: int, action_size: int, num_agents: int, epsilon, random_seed: int):
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
        self.seed = random.seed(random_seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.t_step = 0

        # Hyperparameters
        self.buffer_size = 1000000
        self.batch_size = 256
        self.update_every = 5
        self.num_updates = 5
        self.gamma = 0.99
        self.tau = 0.001
        self.lr_actor = 0.0001
        self.lr_critic = 0.001
        self.weight_decay = 0
        self.epsilon = epsilon
        self.epsilon_decay = 0.997 # around 1750 episodes exploring will hit min
        self.epsilon_min = 0.005

        # Setup up all Agents
        # TODO: Does each agent need its own epsilon?
        self.agents = [Individual_DDPG(state_size, action_size, num_agents, agent_num, epsilon, random_seed) for agent_num in range(num_agents)]

        # Noise Setup
        self.noise = OUNoise(self.action_size, random_seed)

        # TODO: Make sure you are using a centralized Replay Buffer
        # Replay Buffer Setup
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def __str__(self):
        return "MADDPG_Agent"

    def train(self, env, brain_name, num_episodes=2500, max_time=1000, print_every=10):
        """ Interacts with and learns from a given Unity Environment
        :param env: Unity Environment the agents is trying to learn
        :param brain_name: Brain for Environment
        :param num_episodes: Number of episodes to train
        :param max_time: How long each episode runs for
        :param print_every: How often in episodes to print a running average
        :return: Returns episodes scores and 100 episode averages as lists
        """
        # --------- Set Everything up --------#
        scores = []
        avg_scores = []
        scores_deque = deque(maxlen=print_every)

        # -------- Simulation Loop --------#
        for episode_num in range(1, num_episodes + 1):
            # Reset everything
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            episode_scores = np.zeros(self.num_agents)
            self.reset_noise()
            # Run the episode
            for t in range(max_time):
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
                print(f'Episode: {episode_num} \tAverage Score: {round(np.mean(scores_deque), 2)}')
                for agent in self.agents:
                    torch.save(agent.actor_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_Agent{agent.agent_num}_Actor.pth')
                    torch.save(agent.critic_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_Agent{agent.agent_num}_Critic.pth')

        # -------- All Episodes finished Save parameters and scores --------#
        # Save Model Parameters
        torch.save(agent.actor_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_Agent{agent.agent_num}_Actor.pth')
        torch.save(agent.critic_local.state_dict(), f'{PATH}\checkpoints\{agent.__str__()}_Agent{agent.agent_num}_Critic.pth')
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
        # Save single array experiences in replay memory
        # NOTE: Will have to index correctly later to get the correct states and actions
        full_states = states.reshape(-1)
        full_actions = actions.reshape(-1)
        full_next_states = next_states.reshape(-1)
        self.memory.add(full_states, full_actions, rewards, full_next_states, dones)

        # Learn "num_updates" times every "update_every" time step
        self.t_step += 1
        if len(self.memory) > self.batch_size and self.t_step%self.update_every == 0:
            self.t_step = 0
            for _ in range(self.num_updates):
                for agent in self.agents:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent)

    def act(self, states, epsilon, add_noise=True):
        """ Returns actions for each agent given states as per current policy. Policy comes from the actor network.
        :param states: array of states from the environment for each agent
        :param epsilon: probability of exploration
        :param add_noise: bool on whether or not to potentially have exploration for action
        :return: clipped actions
        """
        all_actions = []
        for agent in self.agents:
            agent_state = torch.from_numpy(states[agent.agent_num,:]).float().to(self.device)
            agent.actor_local.eval() # Sets to eval mode (no gradients)
            with torch.no_grad():
                agent_actions = agent.actor_local(agent_state).cpu().data.numpy()
            agent.actor_local.train() # Sets to train mode (gradients back on)
            if add_noise and epsilon > np.random.random():
                agent_actions += agent.noise.sample()
            agent_actions = np.clip(agent_actions, -1,1)
            all_actions.append(agent_actions)
        return np.asarray(all_actions)

    def reset_noise(self):
        """ resets to noise parameters for all agents """
        for agent in self.agents:
            agent.noise.reset()

    def learn(self, experiences, update_agent):
        """ Update actor and critic networks for agent using a given batch of experiences """

        full_states, full_actions, rewards, full_next_states, dones = experiences

        # -------------------- Update Critic -------------------- #
        # Use target networks for getting next actions and q values and calculate q_targets
        # Get next actions for ecah agent based on their individual observation
        next_actions = []
        for agent in self.agents:
            next_actions.append((agent.actor_target(full_next_states[:, agent.agent_num*agent.state_size: agent.agent_num*agent.state_size+agent.state_size])))
        next_actions = np.asarray(next_actions).reshape(-1)
        # Calculate the q_targets
        next_q_targets = update_agent.critic_target(full_next_states, next_actions)
        q_targets = rewards[:, update_agent.agent_num] + (self.gamma*next_q_targets*(1-dones[:, update_agent.agent_num]))

        # Compute critic loss
        q_expected = update_agent.critic_local(full_states,full_actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize loss
        update_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        update_agent.critic_optimizer.step()

        # -------------------- Update Actor --------------------- #
        # Computer actor loss (maximize mean of Q(states,actions))
        # Optimizer minimizes and we want to maximize so multiply by -1
        actor_loss = -1*update_agent.critic_local(full_states, full_actions).mean()
        # Minimize the loss
        update_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        update_agent.actor_optimizer.step()

        #---------------- Update Target Networks ---------------- #
        update_agent.soft_update(update_agent.critic_local, update_agent.critic_target, update_agent.tau)
        update_agent.soft_update(update_agent.actor_local, update_agent.actor_target, update_agent.tau)


class Individual_DDPG():
    """ Deep Deterministic Policy Gradients Agent used to interaction with and learn from an environment """

    def __init__(self, state_size: int, action_size: int, num_agents: int, agent_num: int, epsilon, random_seed: int):
        """ Initialize a DDPG Agent Object
        :param state_size: dimension of state (input)
        :param action_size: dimension of action (output)
        :param num_agents: number of concurrent agents in the environment
        :param epsilon: initial value of epsilon for exploration
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.t_step = 0
        self.agent_num = agent_num

        # Hyperparameters
        self.buffer_size = 1000000
        self.batch_size = 128
        self.update_every = 10
        self.num_updates = 10
        self.gamma = 0.99
        self.tau = 0.001 #Need
        self.lr_actor = 0.0001
        self.lr_critic = 0.001
        self.weight_decay = 0
        self.epsilon = epsilon
        self.epsilon_decay = 0.97
        self.epsilon_min = 0.005

        # Setup Networks (Actor: State -> Action, Critic: (States for all agents, Actions for all agents) -> Value)
        self.actor_local = Actor(self.state_size, self.action_size,  self.seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = self.lr_actor)

        self.critic_local = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.seed).to(self.device)
        self.critic_target = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = self.lr_critic, weight_decay = self.weight_decay)

        # Initialize actor and critic networks to start with same parameters
        self.soft_update(self.actor_local, self.actor_target, tau=1)
        self.soft_update(self.critic_local, self.critic_target, tau=1)

        # TODO: Does each agent needs its own noise attribute? What is the return shape of noise.sample?
        # Noise Setup
        self.noise = OUNoise(self.action_size, random_seed)

        # Replay Buffer Setup
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def __str__(self):
        return "DDPG_Agent"

    def reset_noise(self):
        """ resets to noise parameters """
        self.noise.reset()

    def act(self, states, epsilon, add_noise=True):
        """ Returns actions for given states as per current policy. Policy comes from the actor network.
        :param states: array of states from the environment
        :param epsilon: probability of exploration
        :param add_noise: bool on whether or not to potentially have exploration for action
        :return: clipped actions
        """
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval() # Sets to eval mode (no gradients)
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train() # Sets to train mode (gradients back on)
        if add_noise and epsilon > np.random.random():
            actions += [self.noise.sample() for _ in range(self.num_agents)]
        return np.clip(actions, -1,1)


    def soft_update(self, local_network, target_network, tau):
        """ soft update newtwork parametes
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_network: PyTorch Network that is always up to date
        :param target_network: PyTorch Network that is not up to date
        :param tau: update (interpolation) parameter
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)