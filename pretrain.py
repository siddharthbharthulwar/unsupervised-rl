import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from diayn import DIAYN
import gymnasium as gym

ENV = "HalfCheetah-v4"

#hyperparameters: total # of skills, 

EPOCHS = 1000
NUM_SKILLS = 5

def sample_z():
    return np.random.randint(NUM_SKILLS)

env = gym.make(ENV)
last_path_return = 0
max_path_return = -1 * np.inf
num_episodes = 0
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

agent = DIAYN(NUM_SKILLS, obs_space_dims, action_space_dims)
discriminator_losses = []
policy_losses = []

#basic training loop
for epoch in range(EPOCHS):
    if (epoch % 100 == 0):
        print("Epoch: ", epoch)
    z = sample_z()
    state, info = env.reset()
    #concat state with one-hot action
    done = False
    while not done:
        action = agent.sample_action(state, z)
        next_state, _, terminated, truncated, info = env.step(action)
        agent.rewards.append(-info["reward_ctrl"])

        state = next_state
        done = terminated or truncated
    discriminator_loss, policy_loss = agent.update()
    discriminator_losses.append(discriminator_loss.detach().item())
    policy_losses.append(policy_loss.detach().item())


plt.plot(discriminator_losses)  
plt.ylim(-1000, 1000)
plt.savefig("diayn_plots/" + ENV + "disc.png")
plt.clf()

plt.plot(policy_losses)
plt.plot(-1000, 1000)
plt.savefig("diayn_plots/" + ENV + "pol.png")

agent.save_state_dict(ENV)

