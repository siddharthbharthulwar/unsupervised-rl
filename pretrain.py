import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from diayn import DIAYN
import gymnasium as gym

ENV = "BipedalWalker-v3"

#hyperparameters: total # of skills, 

EPOCHS = 2000
NUM_SKILLS = 8

def sample_z():
    return np.random.randint(NUM_SKILLS)

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
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

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

we can either update policy/discriminator at every time step, or at every episode. open question : which one is better?? -> answer is that we do after every episode, since this is how reinforce fundamentally works. can experiment w/ different methods (SAC, PPO) if time permits. 
'''

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
        #note that we don't have access to external reward here

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

