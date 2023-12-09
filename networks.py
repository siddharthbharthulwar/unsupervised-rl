import gymnasium as gym
import numpy as np
import math
import random as random
from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
from torch import distributions as pyd
import matplotlib.pyplot as plt

"""
Inspired by Gymnasium implementation of policy network.
Policy network for gym environments with continuous action spaces. 
Utilize network in policy gradient methods.
"""

class PolicyNet(nn.Module):
    def __init__(self, obs_space_D: int, action_space_D: int):
        """
        This neural network outputs the parameters of 
        Gaussian distributions from which actions are sampled.
        """

        super(PolicyNet, self).__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        self.net = nn.Sequential(
            nn.Linear(obs_space_D, hidden_space1),
            nn.Tanh(),
            nn.Linear(action_space_D, hidden_space2),
            nn.Tanh(),
        )

        # mean net
        self.mean_net = nn.Linear(hidden_space2, action_space_D)

        # std net
        self.std_net = nn.Linear(hidden_space2, action_space_D)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        """

        features = self.net(x.float())

        means = self.mean_net(features)
        stds = torch.log(1 + torch.exp(self.std_net(features)))

        return means, stds