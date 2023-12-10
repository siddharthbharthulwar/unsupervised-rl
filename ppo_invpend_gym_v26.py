import torch
import torch.nn as nn
from torch.distributions import Normal
import gym, os
from itertools import count
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Model, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(n_latent_var, action_dim)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(n_latent_var, action_dim)
        )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
        # Memory:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    def forward(self, state, action=None, evaluate=False):
        # if evaluate is True then we also need to pass an action for evaluation
        # else we return a new action from distribution
        if not evaluate:
            state = torch.from_numpy(state).float().to(device)
        
        state_value = self.value_layer(state.float())
        
        action_probs = self.action_layer(state)
        action_means, action_stds = self.policy_mean_net(action_probs), self.policy_stddev_net(action_probs)
        action_stds = torch.log(torch.exp(action_stds) + 1)
        action_distribution = Normal(action_means, action_stds)
        
        if not evaluate:
            action = action_distribution.sample()
            self.actions.append(action)
            
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        if evaluate:
            return action_distribution.entropy().mean()
        
        if not evaluate:
            return action.item()
        
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = Model(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=lr, betas=betas)
        self.policy_old = Model(state_dim, action_dim, n_latent_var).to(device)
        
        self.MseLoss = nn.MSELoss()
        
    def update(self):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.policy_old.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list in tensor
        old_states = torch.tensor(self.policy_old.states).to(device).detach()
        old_actions = torch.tensor(self.policy_old.actions).to(device).detach()
        old_logprobs = torch.tensor(self.policy_old.logprobs).to(device).detach()
        

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            dist_entropy = self.policy(old_states.float(), old_actions.float(), evaluate=True)
            # Finding the ratio (pi_theta / pi_theta__old):
            logprobs = self.policy.logprobs[0].to(device)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            

            # Finding Surrogate Loss:
            state_values = self.policy.state_values[0].to(device)
            advantages = rewards - state_values.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.policy.clearMemory()
            
        self.policy_old.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

############## Hyperparameters ##############
env_name = "InvertedPendulum-v4"
#env_name = "CartPole-v1"
# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

render = False
log_interval = 10
n_latent_var = 64           # number of variables in hidden layer
n_update = 2              # update policy every n episodes
lr = 0.0007
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 5                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = None
#############################################

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
print(lr,betas)
# Plot duration curve: 
# From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
running_reward = 0
avg_length = 0
for i_episode in range(1, 1001):
    state = env.reset()
    for t in range(10000): #10000
        # Running policy_old:
        action = ppo.policy_old(state)
        state_n, reward, done, _ = env.step([action])

        # Saving state and reward:
        ppo.policy_old.states.append(state)
        ppo.policy_old.rewards.append(reward)
        
        state = state_n

        running_reward += reward
        if render:
            env.render()
        if done:
            #print(i_episode, t)
            episode_durations.append(t + 1)
            plot_durations()
            break
    
    avg_length += t
    # update after n episodes
    if i_episode % n_update == 0:

        ppo.update()

    # log
    if running_reward > (log_interval*200):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), 
                   './LunarLander_{}_{}_{}.pth'.format(
                    lr, betas[0], betas[1]))
        break

    if i_episode % log_interval == 0:
        avg_length = int(avg_length/log_interval)
        running_reward = int((running_reward/log_interval))

        print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0