""" This file contains all the utility classes needed for this project """
import copy
from collections import deque, namedtuple
import numpy as np
import random
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """ Fixed Sized Replay buffer for storing and sampling experience tuples """

    def __init__(self, buffer_size, batch_size):
        """ Initialize Replay Buffer

        :param buffer_size: size of replay buffer
        :param batch_size: size of training sample
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen = self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        
    def add(self, states, actions, rewards, next_states, dones):
        """ Add an experience to memory """
        self.memory.append(self.experience(states, actions, rewards, next_states, dones))

    def sample(self):
        """ Sample batch_size of experiences randomly from memory. """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        " Return the current size of internal memory "
        return len(self.memory)


class OUNoise:
    """ Ornstein-Uhlenbeck process for generating noise for exploration """
    def __init__(self, action_size, seed, mu=0., theta=0.15, sigma=0.2):
        """ Initialize parameters and noise process """
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = None
        self.size = action_size
        self.reset()

    def reset(self):
        """ Reset the internal state noise and mean """
        self.state = copy.copy(self.mu)

    def sample(self):
        """ Update internal state and return it as a noise sample """
        x = self.state
        dx = self.theta*(self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

