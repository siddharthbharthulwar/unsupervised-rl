import gymnasium as gym
import numpy as np
import random as random
import torch
from networks import PolicyNetwork, PolicyNetworkContinuous
from torch import distributions as pyd

# this file is for testing the saved environment.

#HYPERPARAMS:
NUM_ITERATIONS = 5 #number of episodes to show to human
PARAMS_PATH = 'policyNetworks/bipedalWalkerPolicyNetwork.pt'

state_dict = torch.load(PARAMS_PATH)


env = gym.make('BipedalWalker-v3', render_mode='human')

ACTION_SPACE = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.box.Box) else env.action_space.n
STATE_SPACE = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.box.Box) else env.observation_space.n

env.reset()

print(isinstance(env.action_space, gym.spaces.box.Box))
if isinstance(env.action_space, gym.spaces.box.Box):

    policyNetwork = PolicyNetworkContinuous(STATE_SPACE, ACTION_SPACE)
    policyNetwork.load_state_dict(state_dict)

    for it in range(5):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:

            env.render()
            action_means, action_stds = policyNetwork.forward(torch.from_numpy(obs).float())
            normal = pyd.Normal(action_means, action_stds)
            action = normal.sample()
            next_obs, reward, terminated, truncated, info = env.step(action.numpy())
            obs = next_obs

else:

    policyNetwork = PolicyNetwork(STATE_SPACE, ACTION_SPACE, param_file=PARAMS_PATH)
    for it in range(5):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:

            env.render()
            action_logits = policyNetwork.forward(torch.from_numpy(obs))
            categorical = pyd.Categorical(action_logits)
            action = categorical.sample()
            next_obs, reward, terminated, truncated, info = env.step(action.numpy())
            obs = next_obs
