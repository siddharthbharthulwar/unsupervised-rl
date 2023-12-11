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

#need to be able to sample from z, such that z ~ p(z) where p(z) is uniform
#one-hot encode z




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