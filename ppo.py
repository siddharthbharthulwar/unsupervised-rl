import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dim, alpha,
            fc1_dims=16, fc2_dims=32):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.actor_mean = nn.Sequential(
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh()
        )

        # Policy Std Dev specific Linear Layer
        self.actor_std = nn.Sequential(
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        features = self.actor(state)
        mean, std = self.actor_mean(features), self.actor_std(features)
        std = T.log(T.exp(std) + 1)
        dist = Normal(mean, std)
        
        return dist

    # def save_state_dict(self, path):
    #     T.save(self.actor.state_dict(), path)

    # def load_state_dict(self, path):
    #     self.load_state_dict(T.load(path))

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, alpha, fc1_dims=16, fc2_dims=32):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(input_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        value = self.critic(state)

        return value

    # def save_state_dict(self, path):
    #     T.save(self.net.state_dict(), path)

    # def load_state_dict(self, path):
    #     self.load_state_dict(T.load(path))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    # def save_models(self, path):
    #     print('... saving models ...')
    #     self.actor.save_state_dict(path)
    #     self.critic.save_state_dict(path)

    # def load_models(self, path):
    #     print('... loading models ...')
    #     self.actor.load_state_dict(path)
    #     self.critic.load_state_dict(path)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float)
        state = (state - state.mean()) / (state.std() + 1e-8)  # Normalize the observation
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                # Generalized Advantage Estimation
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            # Normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = T.tensor(advantage, dtype=T.float)  # Convert advantage to a PyTorch tensor

            values = T.tensor(values)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float)
                old_probs = T.tensor(old_prob_arr[batch])
                actions = T.tensor(action_arr[batch])

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]

                # Entropy regularization
                entropy = dist.entropy().mean()

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() - 0.001 * entropy

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    
    env = gym.make('InvertedPendulum-v4')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape[0])
    n_games = int(5e3)

    figure_file = f'rewards/ppo_{env.unwrapped.spec.id}.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(np.array(action).reshape(1))
            n_steps += 1
            score += reward
            agent.remember(obs, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            obs = obs_
            done = terminated or truncated
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models("state_dicts")

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)