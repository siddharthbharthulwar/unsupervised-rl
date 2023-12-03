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


class PolicyNetwork(nn.Module):
    
    def __init__(self, state_space, action_space):

        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 8)
        self.fc2 = nn.Linear(8, action_space)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

#HYPERPARAMS:

NUM_ITERATIONS = 100
DISCOUNT_FACTOR = 0.99
ACTION_SPACE = 4
STATE_SPACE = 8
LEARNING_RATE = 0.01


policyNetwork = PolicyNetwork(STATE_SPACE, ACTION_SPACE)
env = gym.make("LunarLander-v2")
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

    loss = 0
    for state, action, return_ in zip(states, actions, G):
        action_logits = policy(torch.from_numpy(state))
        log_prob = torch.log(action_logits[action])
        loss += -log_prob * return_ #negative because we want to maximize the return
    return loss

for it in range(NUM_ITERATIONS):

    print("Iteration: ", it)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    states, actions, rewards = [], [] ,[]

    #individual episode
    while not terminated and not truncated:

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
plt.show()



