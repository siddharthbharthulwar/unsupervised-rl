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

class PolicyNetwork(nn.Module):
    
    def __init__(self, state_space, action_space, param_file=None):

        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 8)
        self.fc2 = nn.Linear(8, action_space)

        if param_file is not None:
            self.load_state_dict(torch.load(param_file))

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

class PolicyNetworkContinuous(PolicyNetwork):

    def __init__(self, state_space, action_space, param_file=None):
        super(PolicyNetworkContinuous, self).__init__(state_space, action_space, param_file)  # Pass arguments to the parent constructor

        self.fc2 = nn.Linear(8, 2 * action_space)  # Double the output size for mean and std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Split the output into mean and standard deviation
        mean, log_std = torch.chunk(x, 2, dim=-1)
        std = torch.exp(log_std)  # Ensure std is positive

        return mean, std