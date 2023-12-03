import gymnasium as gym
import numpy as np
import random as random
import torch

env = gym.make('InvertedPendulum-v4', render_mode='human')
env.reset()

prev_state = np.array([0, 0, 0, 0])
for _ in range(1000):
    env.render()
    if (prev_state[0] < 0):
        action = np.array([1])
    else:
        action = np.array([-1])
    prev_state, reward, terminated, truncated, info = env.step(action)
env.close()