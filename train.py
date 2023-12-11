import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from reinforce import REINFORCE

import gymnasium as gym

"""Hyperparameters of training procedure"""
ENV = "HalfCheetah-v4" #name of gymnasium environment
SAVE_BEST_SEED = False #whether we save the model with the best reward across seeds (true), or save every seed's model (false)
TOTAL_NUM_EPISODES = int(100000)
SEEDS = [1, 2, 3, 5, 8]

# Create and wrap the environment
env = gym.make(ENV)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = TOTAL_NUM_EPISODES  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []
best_model_seed = -1
best_model_state_dict = None
best_model_reward = float('-inf')

for seed in SEEDS:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []
    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)
        done = False
        while not done:
            action = agent.sample_action(obs)
            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

    if SAVE_BEST_SEED:

        if np.mean(reward_over_episodes) > best_model_reward:

            best_model_reward = np.mean(reward_over_episodes)
            best_model_seed = seed
            best_model_state_dict = agent.save_state_dict()

    else:

        agent.serialize_state_dict(f"state_dicts/{env.unwrapped.spec.id}net{seed}.pt")   

if SAVE_BEST_SEED:

    save_path = torch.save(best_model_state_dict, f"state_dicts/{env.unwrapped.spec.id}netMASTER.pt")

# Plot learning curve
# ~~~~~~~~~~~~~~~~~~~
#

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for" + env.unwrapped.spec.id
)
plt.savefig(f"rewards/{env.unwrapped.spec.id}-rewards.png")
