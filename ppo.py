import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from networks import PolicyNetwork  # Assume you have a PolicyNetwork implementation

# Hyperparameters
NUM_ITERATIONS = 3000
GAMMA = 0.99
LR = 0.002
EPSILON = 0.2
KL_TARGET = 0.01

# Create environment
env = gym.make("CartPole-v1")
ACTION_SPACE = env.action_space.n
STATE_SPACE = env.observation_space.n

# Policy network
policy_network = PolicyNetwork(STATE_SPACE, ACTION_SPACE)
optimizer = optim.SGD(policy_network.parameters(), lr=LR)

# Function to compute KL divergence
def kl_divergence(old_policy, new_policy, states):
    old_logits = old_policy(states)
    new_logits = new_policy(states)

    old_dist = Categorical(logits=old_logits)
    new_dist = Categorical(logits=new_logits)

    return torch.distributions.kl.kl_divergence(old_dist, new_dist)

# PPO loss function
def ppo_loss(old_policy, new_policy, states, actions, advantages):
    new_logits = new_policy(states)
    new_dist = Categorical(logits=new_logits)
    new_probs = new_dist.probs[range(len(actions)), actions]

    old_logits = old_policy(states)
    old_dist = Categorical(logits=old_logits)
    old_probs = old_dist.probs[range(len(actions)), actions]

    ratio = new_probs / old_probs
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages

    return -torch.min(surr1, surr2).mean()

# Training loop
for iteration in range(NUM_ITERATIONS):
    # Collect data
    states, actions, rewards, next_states, dones = [], [], [], [], []
    state = env.reset()

    while True:
        action_probs = policy_network(torch.tensor(state, dtype=torch.float32))
        action_dist = Categorical(probs=action_probs)
        action = action_dist.sample().item()

        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        state = next_state

        if done:
            break

    # Compute advantages
    returns = []
    advantage = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        advantage = reward + GAMMA * (1 - int(done)) * advantage
        returns.insert(0, advantage)

    # Convert to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    advantages = torch.tensor(returns, dtype=torch.float32)

    # Policy update
    old_policy = PolicyNetwork(STATE_SPACE, ACTION_SPACE)
    old_policy.load_state_dict(policy_network.state_dict())

    # Optimize policy using PPO loss with KL constraint
    for _ in range(10):  # You may need to adjust the number of optimization steps
        optimizer.zero_grad()
        loss = ppo_loss(old_policy, policy_network, states, actions, advantages)
        kl = kl_divergence(old_policy, policy_network, states).mean()
        loss = loss + KL_TARGET * kl  # Add KL penalty to the loss
        loss.backward()
        optimizer.step()

    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}, KL: {kl.item()}")

env.close()
