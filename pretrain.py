import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from diayn import DIAYN
import gymnasium as gym
from env_wrapper import EnvWrapper

ENV = "2dbox"
INFO = { #this only matters for box2d env
    "xbounds": [-100, 100],
    "ybounds": [-100, 100]
} 
EPOCHS = 2000
NUM_SKILLS = 10

envwrapper = EnvWrapper(ENV, INFO)
env = envwrapper.env
last_path_return = 0
max_path_return = -1 * np.inf
num_episodes = 0

agent = DIAYN(NUM_SKILLS, envwrapper.obs_space_dims, envwrapper.action_space_dims, [8, 8], [8, 8])
discriminator_losses = []
policy_losses = []

#basic training loop
for epoch in range(EPOCHS):
    print(epoch)
    z = np.random.randint(NUM_SKILLS)
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
np.save("diayn_plots/" + ENV + "disc.npy", discriminator_losses)
plt.clf()

plt.plot(policy_losses)
plt.savefig("diayn_plots/" + ENV + "pol.png")
np.save("diayn_plots/" + ENV + "pol.npy", policy_losses)

agent.save_state_dict(ENV)

