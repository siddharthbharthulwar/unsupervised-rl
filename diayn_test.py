import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from diayn import DIAYN

import gymnasium as gym

NUM_SKILLS = 14
ENV = "HalfCheetah-v4"
PARAMS_PATH = "state_dicts/" + ENV + "DIAYN.pt"

env = gym.make(ENV, render_mode="human")
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
agent = DIAYN(NUM_SKILLS, obs_space_dims, action_space_dims)
agent.policy.load_state_dict(torch.load(PARAMS_PATH))


for skill in range(NUM_SKILLS):

    obs, info = env.reset()
    done = False
    rewards = []
    while not done:
        action = agent.sample_action(obs, skill)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    print("Skill: ", skill, " with reward: ", np.mean(rewards))