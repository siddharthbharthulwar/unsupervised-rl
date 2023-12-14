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

import gymnasium as gym

NUM_SKILLS = 10
ENV = "HalfCheetah-v4"
PARAMS_PATH = "state_dicts/" + ENV + "DIAYN.pt"
DISC_ARCH = [32, 32]
POL_ARCH = [32, 32]

GIF = False

if GIF:
    env = gym.make(ENV, render_mode="rgb_array")
else:
    env = gym.make(ENV, render_mode = "human")
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
agent = DIAYN(NUM_SKILLS, obs_space_dims, action_space_dims, DISC_ARCH, POL_ARCH)
agent.policy.load_state_dict(torch.load(PARAMS_PATH))


for skill in range(NUM_SKILLS):

    obs, info = env.reset()
    done = False
    rewards = []
    counter = 0
    pil_images = []
    while not done:
        action = agent.sample_action(obs, skill)
        rend = env.render()
        if rend is not None and GIF:
            pil_image = Image.fromarray(rend)
            pil_images.append(pil_image)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        counter +=1

    print("Skill: ", skill, " with reward: ", np.mean(rewards))
    if GIF:
        pil_images[0].save("figures/" + ENV + "/" + str(skill) + ".gif", save_all=True, append_images=pil_images[1:], duration=100, loop=0)