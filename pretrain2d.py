import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from diayn import DIAYN
import gymnasium as gym
import random

from env2d import Env

ENV = "2dbox"

#hyperparameters: total # of skills, 

EPOCHS = 50000
NUM_SKILLS = 8

XBOUND = [-100, 100]
YBOUND = [-100, 100]

seedx = random.uniform(XBOUND[0], XBOUND[1])
seedy = random.uniform(YBOUND[0], YBOUND[1])

def sample_z():
    return np.random.randint(NUM_SKILLS)

env = Env(seedx, seedy, XBOUND, YBOUND)
last_path_return = 0
max_path_return = -1 * np.inf
num_episodes = 0
obs_space_dims = 2
action_space_dims = 2

agent = DIAYN(NUM_SKILLS, obs_space_dims, action_space_dims)
discriminator_losses = []
policy_losses = []

#basic training loop
for epoch in range(EPOCHS):
    if (epoch % 100 == 0):
        print("Epoch: ", epoch)
    z = sample_z()
    print(epoch)
    state, info = env.reset()
    #concat state with one-hot action
    done = False
    while not done:
        action = agent.sample_action(state, z)
        next_state, _, terminated, truncated, info = env.step(action)
        # agent.rewards.append(-info["reward_ctrl"])

        state = next_state
        done = terminated or truncated
    discriminator_loss, policy_loss = agent.update()
    discriminator_losses.append(discriminator_loss.detach().item())
    policy_losses.append(policy_loss.detach().item())


plt.plot(discriminator_losses)  
plt.savefig("diayn_plots/" + ENV + "disc.png")
plt.clf()

plt.plot(policy_losses)
plt.savefig("diayn_plots/" + ENV + "pol.png")

agent.save_state_dict(ENV)

