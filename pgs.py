from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import distributions as pyd
from networks import PolicyNet

import gymnasium as gym

# plt.rcParams["figure.figsize"] = (10, 5)

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_D: int, action_space_D: int):
        """
        Initializes an agent.
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.log_probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = PolicyNet(obs_space_D, action_space_D)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """
        Samples an action.
        """

        # Obtain mean, std from net to construct a Gaussian. Sample.

        mean, std = self.net(torch.from_numpy(state))
        print(mean, std)
        normal_dist = pyd.Normal(mean + self.eps, std + self.eps)
        action = normal_dist.sample()
        log_prob = normal_dist.log_prob(action)

        self.log_probs.append(log_prob)

        return action.numpy()

    def compute_returns(self) -> list[float]:
        G = []
        returns = 0
        for reward in reversed(self.rewards):
            self.returns = reward + self.gamma * returns
            G.insert(0, returns)
        return G
    
    def update(self):
        """Updates the policy network's weights."""

        returns = torch.tensor(self.compute_returns())

        loss = 0
        
        for log_prob, return_val in zip(self.probs, returns):
            loss -= log_prob.mean() * return_val

        # Backpropagate and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []