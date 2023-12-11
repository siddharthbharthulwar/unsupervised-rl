import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
from reinforce import REINFORCE

"""Hyperparameters of testing procedure"""

ENV = "Pusher-v4"
SEED = 2
USE_MASTER = True
SEEDS = [1, 2, 3, 5, 8]
if (USE_MASTER):
    PARAMS_PATH = "state_dicts/" + ENV + "netMASTER.pt"
else:
    PARAMS_PATH = "state_dicts/" + ENV + "net" + str(SEED) + ".pt"

env = gym.make(ENV, render_mode="human")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
agent = REINFORCE(obs_space_dims, action_space_dims)
agent.net.load_state_dict(torch.load(PARAMS_PATH))


for seed in SEEDS:

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    obs, info = wrapped_env.reset(seed=seed)
    done = False
    rewards = []
    while not done:
        action = agent.sample_action(obs)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    print("Seed: ", seed, " with reward: ", np.mean(rewards))
