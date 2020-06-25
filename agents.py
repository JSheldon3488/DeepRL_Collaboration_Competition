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
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.t_step = 0

        # Hyperparameters
        self.buffer_size = 100000
        self.batch_size = 256
        self.update_every = 1
        self.num_updates = 1
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0 # Turned off noise scaling
        self.epsilon_min = 1

        # Setup up all Agents
        self.agents = [DDPG(self.state_size, self.action_size, self.num_agents, self.seed) for _ in range(self.num_agents)]

        # Replay Buffer Setup
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def __str__(self):
        return "MADDPG_Agent"

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
            # print(f'Original Shape of States: Expected = [2,24], Actual = {states.shape}') PASSED
            self.reset_noise()
            # print(f'Test of noise sample: Expected = [2,2] with numbers from standard normal, Actual = {[agent.noise.sample() for agent in self.agents]}') PASSED

            # Run the episode
            while True:
                actions = self.act(states, self.epsilon)
                # print(f'Actions Expected Shape: [2,2] Actual Shape: {actions.shape}') PASSED
                env_info = env.step(actions)[brain_name]
                next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done

                '''print("--------- Checking Replay Buffer Input ------------")
                print(f'States: {states.shape}')
                print(f'Actions: {actions.shape}')
                print(f'Rewards: {rewards}')                                    PASSED
                print(f'Next_States: {next_states.shape}')
                print(f'Dones: {dones}')
                print("----------------------------------------------------") '''

                self.step(states, actions, rewards, next_states, dones)
                episode_scores += rewards
                states = next_states
                if np.any(dones):
                    break

            # -------- Episode Finished ---------#
            self.epsilon -= self.epsilon_decay
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
            for agent_num in range(self.num_updates):
                self.learn(self.memory.sample(), agent_num)

    def act(self, states, epsilon, add_noise=True):
        """ Returns actions for each agent given states as per current policy. Policy comes from the actor network.
        :param states: array of states from the environment for each agent
        :param epsilon: probability of exploration
        :param add_noise: bool on whether or not to potentially have exploration for action
        :return: clipped actions for all agents from the local networks
        """
        actions = [agent.act(local_obs, epsilon) for agent, local_obs in zip(self.agents,states)]
        return np.asarray(actions)

    def target_act(self, obs_full):
        """ Returns actions for each agent given states as per current policy. Policy comes from the actor network.
        :param states: array of states from the environment for each agent
        :return: clipped actions for all agents from the local networks
        """
        actions = [agent.target_act(local_obs) for agent, local_obs in zip(self.agents,obs_full)]
        return np.asarray(actions)

    def reset_noise(self):
        """ resets to noise parameters for all actors """
        for agent in self.agents:
            agent.reset_noise()

    def learn(self, experiences, agent_num):
        """ Update actor and critic networks for agent using a given batch of experiences """

        full_states, full_actions, rewards, full_next_states, dones = experiences

        '''
        print('---------------Check Experience Output -------------------')
        print(f'full_states type: {type(full_states)}, shape: {full_states.shape}') # [256,2,24]
        print(f'full_actions type: {type(full_actions)}, shape: {full_actions.shape}') # [256,2,2]
        print(f'rewards type: {type(rewards)}, shape: {rewards.shape}') # [256,2]                   PASSED
        print(f'full_next_states type: {type(full_next_states)}, shape: {full_next_states.shape}') # [256,2,24]
        print(f'dones type: {type(dones)}, shape: {dones.shape}') # [256,2]
        print("")
        '''

        # -------------------- Update Critic --------------------
        ''' We need the target critic values for each agent where the input to the critic network for that agent is full next state information (both agents info)
            and each agents target action selection that would be selected based on that info. So for a single example from the full_states mini batch
            you need to get the actions for both agents based on their target actor network and their chunk of the observation. Then you combine
            the full observation and the actinos and feed them through the target critic for that agent (will happen for both), multiply it by the discount rate
            and add it to the reward to get y = two different values one for each agent.
        '''
        # Get next actions for all examples in the mini batch
        next_actions = torch.from_numpy(np.asarray([self.target_act(full_states[batch_num]) for batch_num in range(self.memory.batch_size)])).float().to(self.device)
        # print(f'Next Action type: {type(next_actions)}, Shape: {next_actions.shape}') #[256,2,2] PASSED

        # Get Q_values from target network for next_state and next_actions (Need to change next_states into [256,48] and next_actions into [256,4] for each agent
        full_next_states = full_next_states.view(self.memory.batch_size,-1)
        next_actions = next_actions.view(self.memory.batch_size,-1)
        # print(f'Full_Next_States type: {type(full_next_states)}, shape: {full_next_states.shape}') PASSED
        # print(f'Next_Actions type: {type(next_actions)}, shape: {next_actions.shape}') PASSED
        next_q_targets = []
        for agent in self.agents:
            next_q_targets.append(agent.critic_target(full_next_states, next_actions))
        # print(f'Next_q_targets type: {type(next_q_targets)}, shape: {next_q_targets[0].shape}') # List of Tensors that are [256,1]

        # Calclate y which is the q_targets = rewards for agent i + discount*q_targets for agent i (calculated for both agents)
        q_targets = [rewards[:,agent_num] + ((self.gamma*next_q_targets[agent_num]).view(-1))*(1-dones[:,agent_num]) for agent_num in range(self.num_agents)]
        # print(f'q_targets type: {type(q_targets[0])}, shape: {len(q_targets)},{q_targets[0].shape}') # 2 tensors of size [256] PASSED
        # Get Q_expected
        q_expected = []
        for agent in self.agents:
            q_expected.append(agent.critic_local(full_states.view(self.memory.batch_size,-1), full_actions.view(self.memory.batch_size,-1)).view(-1))
        # print(f'Q_Expected type: {type(q_expected[0])}, shape: {len(q_expected)},{q_expected[0].shape} ') 2 tensors of size [256] PASSED

        # Compute critic loss
        critic_loss = []
        for i, agent in enumerate(self.agents):
            critic_loss.append(F.mse_loss(q_targets[i],q_expected[i]))
            agent.critic_optimizer.zero_grad()
            critic_loss[i].backward()
            agent.critic_optimizer.step()


        # -------------------- Update Actor --------------------- #
        # Computer actor loss (maximize mean of Q(states,actions))
        # Optimizer minimizes and we want to maximize so multiply by -1
        for i,agent in enumerate(self.agents):
            actor_loss = -1*agent.critic_local(full_states.view(self.memory.batch_size,-1), full_actions.view(self.memory.batch_size,-1)).mean()
            # Minimize the negative loss (maximize q_value)
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        # TODO: Make sure soft updates correct
        #---------------- Update Target Networks ---------------- #
        for agent in self.agents:
            agent.soft_update(agent.actor_local, agent.actor_target, agent.tau)


class DDPG():
    """ This is a decentralized actor used by the MADDPG Agent """

    def __init__(self, state_size, action_size, num_agents, seed):
        """ Initialize a DDPG Agent Object
        :param state_size: dimension of state (input) for this decentralized actor
        :param action_size: dimension of action (output) for this decentralized actor
        :param random_seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.tau = 0.05
        self.lr_actor = 0.001
        self.lr_critic = 0.001

        # Setup Networks (Actor: State -> Action, Critic: (States for all agents, Actions for all agents) -> Value)
        self.actor_local = Actor(self.state_size, self.action_size,  self.seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = self.lr_actor)

        self.critic_local = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.seed).to(self.device)
        self.critic_target = Critic(self.state_size*self.num_agents, self.action_size*self.num_agents, self.seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = self.lr_critic)

        # Initialize local and taret networks to start with same parameters
        self.soft_update(self.actor_local, self.actor_target, tau=1)
        self.soft_update(self.critic_local, self.critic_target, tau=1)

        # TODO: Make sure Noise is returning correct size and values to add noise to actions
        # Noise Setup
        self.noise = OUNoise(self.action_size, self.seed)

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
        # print(f'Action before noise: Shape: {actions.shape}, Value: {actions}') PASSED
        if add_noise:
            actions += self.noise.sample()*epsilon
        # print(f'Action after noise: Shape: {actions.shape}, Value: {actions}') PASSED
        return np.clip(actions, -1,1)

    def target_act(self, state):
        """ Used for selectiong actions for next_states. No exploration needed

        :param state: observations for this indiviual actoin (next_state)
        :return: numpy array of clipped actions for this agent
        """
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
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