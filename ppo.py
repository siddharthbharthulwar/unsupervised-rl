import gymnasium as gym
import numpy as np
import math
import random as random
from torch import nn
import torch
import torch.nn.functional as F

def normalize(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = math.sqrt(variance)
    normalized_data = [(x - mean) / (std_dev + 1e-8) for x in data]
    return normalized_data

# def calculate_advantages_and_returns()

class PPO:

    def __init__(self):

        self.policy = 0
        self.value = torch.ones(1)
    
    def select_action(self, state):

        return self.policy
    
    def get_value(self, state):

        return self.value


#HYPERPARAMS:
NUM_ITERATIONS = 10
NUM_ENV_STEPS = 100

# env = gym.make('InvertedPendulum-v4', render_mode='human')
env = gym.make("LunarLander-v2", render_mode="human")
#for lunar lander, the discrete action space is: [do nothing, fire left engine, fire main engine, fire right engine]. one action at a time

#observation space: 8-dimensional vector.

ppo = PPO()

for it in range(NUM_ITERATIONS):

    obs = env.reset()
    terminated = False
    truncated = False
    data = []

    for _ in range(NUM_ENV_STEPS): #rollout

        old_obs = obs

        if truncated or terminated:
            
            break

        env.render()
        action = np.array(ppo.select_action(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        data.append((old_obs, action, reward, obs, terminated or truncated))

    print(len(data))

env.close()