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
from datetime import datetime


def add_training_run(d_arch, p_arch, env_name, epochs, num_skills):

    now = datetime.now()
    current_datetime = now.strftime("%m/%d/%Y, %H:%M:%S")

    file = open("training_runs.txt", "a+")
    file.seek(0)

    data = file.read(100)
    if len(data) > 0:
        file.write("\n")

    serialized_run = str(current_datetime) + ", "+ str(d_arch) + ", " + str(p_arch) + ", " + env_name + ", " + str(epochs) + ", " + str(num_skills)

    file.write(serialized_run)


ENV = "HalfCheetah-v4"
INFO = { #this only matters for box2d env
    "xbounds": [-100, 100],
    "ybounds": [-100, 100]
} 
EPOCHS = 40000
NUM_SKILLS = 15

DISCRIMINATOR_ARCH = [64, 64]
POLICY_ARCH = [64, 64]

envwrapper = EnvWrapper(ENV, INFO)
env = envwrapper.env
last_path_return = 0
max_path_return = -1 * np.inf
num_episodes = 0

agent = DIAYN(NUM_SKILLS, envwrapper.obs_space_dims, envwrapper.action_space_dims, DISCRIMINATOR_ARCH, POLICY_ARCH)
discriminator_losses = []
policy_losses = []
entropies = []
percentage_correct = []

#basic training loop
for epoch in range(EPOCHS):
    print(epoch)
    z = np.random.randint(NUM_SKILLS)
    state, info = env.reset()
    done = False
    counter = 0
    while not done:
        action = agent.sample_action(state, z)
        next_state, _, terminated, truncated, info = env.step(action)
        # agent.rewards.append(-info["reward_ctrl"])

        state = next_state
        done = terminated or truncated
        counter +=1

    percentage_correct.append(agent.correct / counter)
    discriminator_loss, policy_loss, entropy = agent.update()
    discriminator_losses.append(discriminator_loss.detach().item())
    policy_losses.append(policy_loss.detach().item())
    entropies.append(entropy)

print(len(agent.discrim_predicted_debug))

plt.plot(discriminator_losses)  
plt.savefig("diayn_plots/" + ENV + "disc.png")
np.save("diayn_plots/" + ENV + "disc.npy", discriminator_losses)
plt.clf()

plt.plot(policy_losses)
plt.savefig("diayn_plots/" + ENV + "pol.png")
np.save("diayn_plots/" + ENV + "pol.npy", policy_losses)

plt.clf()
plt.plot(entropies)
plt.savefig("diayn_plots/" + ENV + "ent.png")
np.save("diayn_plots/" + ENV + "ent.npy", entropies)

plt.clf()
plt.hist(agent.discrim_predicted_debug, bins=NUM_SKILLS)
plt.savefig("diayn_plots/" + ENV + "hist.png")


agent.save_state_dict(ENV)
add_training_run(DISCRIMINATOR_ARCH, POLICY_ARCH, ENV, EPOCHS, NUM_SKILLS)



