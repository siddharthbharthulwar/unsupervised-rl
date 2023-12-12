import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.functional import softmax
from reinforce import Policy_Network


import gymnasium as gym

class Discriminator_Network(nn.Module):
    '''Discriminator Network'''

    def __init__(self, obs_space_dims : int, num_skills : int):
        """Initializes a neural network that estimates the probability of a skill given a state
        
        Args:
            obs_space_dims: Dimension of the observation space
            num_skills: Number of skills
        """

        super().__init__()

        #dimensions of each hidden layer
        hidden_dims = [128, 64, 32]

        #constructing shared net from hidden layer dimensions
        sequential_input = []
        prev_space = obs_space_dims
        for dim in hidden_dims:
            sequential_input.append(
                nn.Linear(prev_space, dim)
            )
            sequential_input.append(nn.Tanh())
            prev_space = dim

        self.shared_net = nn.Sequential(*sequential_input)
        self.output = nn.Linear(hidden_dims[-1], num_skills)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        hidden_features = self.shared_net(x.float())
        output = self.output(hidden_features)
        return softmax(output, dim=1)

class DIAYN:
    '''DIAYN algorithm'''

    def __init__(self, num_skills : int, obs_space_dims : int, action_space_dims : int):

        self.discriminator = Discriminator_Network(obs_space_dims, num_skills) #discriminator state -> skill
        self.policy = Policy_Network(obs_space_dims + num_skills, action_space_dims) #policy concatenated state and skill -> action

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.01)

        self.alpha = 0.1 #empirically found to be good in DIAYN
    
    def update(self, states, actions, z):

        return None