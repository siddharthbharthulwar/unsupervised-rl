import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

ENV = "Pusher-v4"

#hyperparameters: total # of skills, 

EPOCHS = 1000
NUM_SKILLS = 6

#need to be able to sample from z, such that z ~ p(z) where p(z) is uniform
#one-hot encode z


def sample_z():

    return None


def get_action(obs):

    return None

#process per each training loop:

'''
- sample z from p(z)
- concatenate one-hot-encoded z with observation vector -> let this be c
- get action from policy network as policyNetwork(c)
- if we want to learn p(z):
    - some complicated logic with discriminator here, todo figure out
    - turns out that this is an extension, and they actually go over it in the paper
    - learning p(z) reduces the effective number of skills, but we want to maximize for diversity of skills so leaving it uniform is prolly fine

- next_ob, reward, terminal, info = env.step(action)

- if terminated or truncated:
    - reset env, reset policy (what does reset policy mean?)
    - record max path return (max of current max path return and new path return)
- else:
    - state = next_state
- some fuckery with pool sizes (todo figure out)
- 

'''

env = gym.make(ENV)
last_path_return = 0
max_path_return = -1 * np.inf
num_episodes = 0

# for epoch in EPOCHS:

#     path_length_list = []
#     z = sample_z()
    
#     done = False
#     aug_obs = None
#     while not done:

#         action = get_action(aug_obs)

#         obs, reward, terminated, truncated, info = env.step(action)
#         aug_next_obs = None #todo: concatenate skill and next ob
#         done = terminated or truncated

#         if done:
#             path_length_list.append()

'''
agent.train pseudocode:

states, zs, dones, actions, next_states = 

reparam_actions, log_probs = policyNetwork(states)

logits = discriminator() (logits of which action it most likely was, so this is NUM_SKILLS-dimensional)

logq_z_ns = log_softmax(logits)

rewards = log q psi (z | s) - log p(z) -> note that log p(z) should be constantly uniform at first, and hence we dont need this term

backprop on policy network loss
backprop on discriminator loss

key insight: to make DIAYN work with reinforce, we need to use the intrinsic reward from the discriminator, not the extrinsic reward from the environment. intuition: if the discriminator can easily predict the correct skill from the state, the policy is generating distinct behaviors for each skill. 

r(s, z) = log q psi (z | s), where q psi (z | s) is discriminator estimate of probability of skill z given state s


'''

#basic training loop
for epoch in EPOCHS:


    z = sample_z()
    state = env.reset()
    #concat state with one-hot action
    episode_reward = 0
    done = False
    logq_zs = []

    while not done:
        action = get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        #concat next_state, one-hot action
        #store in agent??

        #logq_z = agent.train()
        logq_z = None

        if (logq_z is None):
            logq_zs.append(logq_z)
        else:
            logq_zs.append(logq_z)

        episode_reward += reward
        state = next_state
        done = terminated or truncated
