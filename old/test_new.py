from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
from reinforce_invpend_gym_v26 import REINFORCE, Policy_Network

plt.rcParams["figure.figsize"] = (10, 5)

env = gym.make("InvertedPendulum-v4", render_mode="human")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
agent = REINFORCE(obs_space_dims, action_space_dims)

agent.render_state_dict(wrapped_env, f"state_dicts/{env.unwrapped.spec.id}net8.pt")

# Not sure if this is the right way to do this