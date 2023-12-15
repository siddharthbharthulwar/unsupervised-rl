import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from diayn import DIAYN
from PIL import Image

from env2d import Env

import gymnasium as gym

NUM_SKILLS = 5
ENV = "2dbox"
PARAMS_PATH = "state_dicts/" + ENV + "DIAYN.pt"

DISCRIMINATOR_ARCH = [32, 32]
POLICY_ARCH = [32, 32]

GIF = False

XBOUND = [-100, 100]
YBOUND = [-100, 100]

# seedx = random.uniform(XBOUND[0], XBOUND[1])
# seedy = random.uniform(YBOUND[0], YBOUND[1])

seedx = 0
seedy = 0


env = Env(seedx, seedy, XBOUND, YBOUND)
obs_space_dims = 2
action_space_dims = 2
agent = DIAYN(NUM_SKILLS, obs_space_dims, action_space_dims, DISCRIMINATOR_ARCH, POLICY_ARCH)
agent.policy.load_state_dict(torch.load(PARAMS_PATH))


for skill in range(NUM_SKILLS):

    xs = []
    ys = []

    obs, info = env.reset()
    done = False
    counter = 0
    while not done:
        action = agent.sample_action(obs, skill)
        obs, reward, terminated, truncated, info = env.step(action)

        xs.append(obs[0])
        ys.append(obs[1])
        done = terminated or truncated
        counter +=1

    plt.plot(xs, ys)
    print("Skill: ", skill)

plt.show()