import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """ Takes in a network layer and returns a range to use for parameter initialization """
    fan_in = layer.weight.data.size()[0]
    lim = 1./ np.sqrt(fan_in)
    return (-lim,lim)


class Actor(nn.Module):
    """ Actor (Policy) network for action selection """
    def __init__(self, input_size, action_size, seed, fc1_size=512, fc2_size=256, leak=0.01):
        """ Initialize parameters and build actor network
        :param input_size: Dimension of the input
        :param action_size: Dimension of the output
        :param seed: random seed
        """
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.leak = leak
        self.bn_input = nn.BatchNorm1d(input_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size,fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        self.reset_parameters()

    def forward(self, states):
        """ policy network that maps states -> actions """
        ''' 
        Forward pass with Normalization and leaky relu activations
        x = self.bn_input(states)
        x = F.leaky_relu(self.fc1(states), negative_slope=self.leak)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        return torch.tanh(self.fc3(x))
        '''
        # Forward pass with no Normalization or leaky relu
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

    def reset_parameters(self):
        """ Initialize or reset parameters for the network """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)


class Critic(nn.Module):
    " Critic (Value) network for evaluating actions "

    def __init__(self, input_size, action_size, seed, fc1_size=512, fc2_size=256, leak=0.01):
        """ Initialize the parameters and set up the network
        :param input_size: Dimension of input state for all agents (all agents observations combined)
        :param action_size: Dimension of actions for all agents (all agents actions combined)
        :param seed: random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.leak = leak
        self.bn_input = nn.BatchNorm1d(input_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size+action_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.reset_parameters()

    def forward(self, states, actions):
        """ Critic Network that maps (states,actions) pairs -> Q-Values """
        ''' 
        Forward pass with Normalization and leaky relu activations
        x = self.bn_input(states)
        x = F.leaky_relu(self.bn1(self.fc1(states)), negative_slope=self.leak)
        x_a = torch.cat((x,actions), dim=1)
        x_a = F.leaky_relu(self.fc2(x_a), negative_slope=self.leak)
        return self.fc3(x_a)
        '''
        # Forward Pass Without Normalization and Leaky Relu
        x = F.relu(self.fc1(states))
        x_a =  torch.cat((x, actions), dim=1)
        x_a = F.relu(self.fc2(x_a))
        return self.fc3(x_a)

    def reset_parameters(self):
        """ Initialize or reset parameters for the network """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)