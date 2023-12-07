import gymnasium as gym
import numpy as np
import math
import random as random
from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
from torch import distributions as pyd
import matplotlib.pyplot as plt
from networks import PolicyNetwork, PolicyNetworkContinuous

#HYPERPARAMS:
env = gym.make("BipedalWalker-v3")

NUM_ITERATIONS = 50
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.02

if isinstance(env.action_space, gym.spaces.box.Box):
    ACTION_SPACE = env.action_space.shape[0]
    STATE_SPACE = env.observation_space.shape[0]

    policyNetwork = PolicyNetworkContinuous(STATE_SPACE, ACTION_SPACE)

elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
    ACTION_SPACE = env.action_space.n
    STATE_SPACE = env.observation_space.n
    policyNetwork = PolicyNetwork(STATE_SPACE, ACTION_SPACE)

optimizer = optim.Adam(policyNetwork.parameters(), lr=LEARNING_RATE)
losses = []

def compute_returns(rewards, gamma=DISCOUNT_FACTOR):
    G = []
    returns = 0
    for reward in reversed(rewards):
        returns = reward + gamma * returns
        G.insert(0, returns)
    return G

def compute_loss(states, actions, G, policy):

    if policy.__class__.__name__ == "PolicyNetworkContinuous":
        
        loss = 0
        for state, action, return_ in zip(states, actions, G):
            mean, std = policy(torch.from_numpy(state))
            normal_dist = pyd.Normal(mean, std)
            log_prob = normal_dist.log_prob(action.clone().detach().requires_grad_(True))
            loss += -log_prob * return_  # negative because we want to maximize the return
        
        loss = torch.mean(loss)
        return loss

    else:

        loss = 0
        for state, action, return_ in zip(states, actions, G):
            action_logits = policy(torch.from_numpy(state))
            log_prob = torch.log(action_logits[action])
            loss += -log_prob * return_ #negative because we want to maximize the return
        return loss

for it in range(NUM_ITERATIONS):

    if (it % 100 == 0):
        print("Iteration: ", it)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    states, actions, rewards = [], [] ,[]

    #individual episode
    while not terminated and not truncated:
        
        if policyNetwork.__class__.__name__ == "PolicyNetworkContinuous":
            action_means, action_stds = policyNetwork.forward(torch.from_numpy(obs).float())
            normal = pyd.Normal(action_means, action_stds)
            action = normal.sample()
        else:
            action_logits = policyNetwork.forward(torch.from_numpy(obs))
            categorical = pyd.Categorical(action_logits)
            action = categorical.sample()
        # action = np.random.choice(len(action_logits), p=action_logits)
        next_obs, reward, terminated, truncated, info = env.step(action.numpy())

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs

    optimizer.zero_grad()
    G = compute_returns(rewards)
    loss = compute_loss(states, actions, G, policyNetwork)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

env.close()

plt.plot(losses)
plt.savefig("losses.png")
# plt.show()

#save the policy

print(policyNetwork.state_dict())
print(policyNetwork.__class__.__name__)

torch.save(policyNetwork.state_dict(), "policyNetworks/bipedalWalkerPolicyNetwork.pt")

# #deploy the policy

# env = gym.make("LunarLander-v2", render_mode="human")
# for it in range(5):
#     obs, _ = env.reset()
#     terminated = False
#     truncated = False
#     while not terminated and not truncated:

#         env.render()
#         action_logits = policyNetwork.forward(torch.from_numpy(obs))
#         categorical = pyd.Categorical(action_logits)
#         action = categorical.sample()
#         next_obs, reward, terminated, truncated, info = env.step(action.numpy())
#         obs = next_obs
