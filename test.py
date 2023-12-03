import gymnasium as gym
import numpy as np
import random as random
import torch
from networks import PolicyNetwork
from torch import distributions as pyd

# this file is for testing the saved environment.

#HYPERPARAMS:
NUM_ITERATIONS = 5 #number of episodes to show to human
ACTION_SPACE = 4
STATE_SPACE = 8
PARAMS_PATH = 'policyNetwork.pt'

env = gym.make('LunarLander-v2', render_mode='human')
env.reset()


policyNetwork = PolicyNetwork(STATE_SPACE, ACTION_SPACE, param_file=PARAMS_PATH)
for it in range(5):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:

        env.render()
        action_logits = policyNetwork.forward(torch.from_numpy(obs))
        categorical = pyd.Categorical(action_logits)
        action = categorical.sample()
        next_obs, reward, terminated, truncated, info = env.step(action.numpy())
        obs = next_obs