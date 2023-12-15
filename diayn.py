import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
from reinforce import Policy_Network
from random import random

class Discriminator_Network(nn.Module):
    '''Discriminator Network'''

    def __init__(self, obs_space_dims : int, num_skills : int, hidden_dims : list):
        """Initializes a neural network that estimates the probability of a skill given a state
        
        Args:
            obs_space_dims: Dimension of the observation space
            num_skills: Number of skills
        """

        super().__init__()

        #dimensions of each hidden layer
        # hidden_dims = [512, 64, 32]
        # hidden_dims = [16, 8] #2dbox

        self.sequential_input = [None] * len(hidden_dims)
        prev_space = obs_space_dims

        for i, dim in enumerate(hidden_dims):
            self.sequential_input[i] = nn.Linear(prev_space, dim)
            nn.init.xavier_uniform_(self.sequential_input[i].weight)
            self.sequential_input[i].bias.data.fill_(0.01)
            prev_space = dim

        self.output = nn.Linear(hidden_dims[-1], num_skills)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = F.relu(self.sequential_input[0](x.float()))
        for net in self.sequential_input[1:]:
            x = F.relu(net(x))
        return self.output(x)

class DIAYN:
    '''DIAYN algorithm'''

    def __init__(self, num_skills : int, obs_space_dims : int, action_space_dims : int, disc_dims : list, pol_dims : list):

        self.num_skills = num_skills
        self.obs_space_dism = obs_space_dims
        self.action_space_dims = action_space_dims
        self.eps = 1e-6  # small number for mathematical stability
        self.learning_rate = 1e-4 # learning rate for optimizer (both discriminator and policy)

        self.discriminator = Discriminator_Network(obs_space_dims, num_skills, disc_dims) #discriminator state -> skill
        self.policy = Policy_Network(obs_space_dims + num_skills, action_space_dims, pol_dims) #policy concatenated state and skill -> action

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        self.alpha = 25 #empirically found to be good in DIAYN
        self.gamma = 0.99 #discount factor
        self.rewards = [] #intrinsic rewards from discriminator

        self.actions = []
        self.states = []
        self.z = None
        self.probs = []
        self.entropies = []

        self.correct = 0
        self.discrim_predicted_debug = []
        self.action_means_debug = []
        self.action_stddevs_debug = []

    def sample_action(self, state : np.ndarray, skill : int) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment
            skill: Skill to condition on

        Returns:
            action: Action to be performed, or a_t ~ pi_theta (a_t | s_t, z)
        """

        logits = self.discriminator(torch.from_numpy(state))
        self.discrim_predicted_debug.append(torch.argmax(F.softmax(logits)).item())

        self.states.append(torch.from_numpy(state))
        one_hot = np.zeros(self.num_skills)
        one_hot[skill] = 1

        state = np.concatenate((state, one_hot))
        state = torch.from_numpy(state).float()
        action_means, action_stddevs = self.policy(state)
        action_stddevs = action_stddevs + self.eps
        self.action_means_debug.append(action_means.detach().numpy())
        self.action_stddevs_debug.append(action_stddevs.detach().numpy())
        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means + self.eps, action_stddevs + self.eps)
        action = distrib.sample()
        # print("a1", action_stddevs)
        # print("a2", action_stddevs **2)
        e = torch.exp(torch.tensor(1.0))
        entropy = torch.log(2 * torch.tensor(np.pi) * e * action_stddevs**2 + 1)
        if (entropy < 0).any():
            print("entropy:", entropy)
            print("log entropy", torch.log(entropy))

        # if (entropy < 0).any():
        #     print("entropy:", entropy)
        #     print("means:", action_means)
        #     print("std:", action_stddevs)
        entropy = entropy.sum(axis=-1)
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)
        self.entropies.append(entropy)
        self.actions.append(action)
        self.z = skill
        return action
    
    def update(self):

        running_g = 0
        gs = []
        discriminator_loss = 0
        policy_loss = 0
        crossentropy = nn.CrossEntropyLoss()
        #extracting intrinsic reward for each state traversed (also calculating loss for discriminator)
        # for (state, ctrl_cost) in zip(self.states[::-1], self.rewards[::-1]):

        #     logits = self.discriminator(state)
        #     R = crossentropy(logits.unsqueeze(0), torch.tensor([self.z]))
        #     R = torch.log(R)
        #     discriminator_loss += R
        #     running_g = self.gamma * running_g + ctrl_cost + R
        #     gs.insert(0, running_g)

        # deltas = torch.tensor(gs)

        for state in self.states[::-1]:
            logits = self.discriminator(state)
            R = crossentropy(logits.unsqueeze(0), torch.tensor([self.z]))
            discriminator_loss += R
            running_g = self.gamma * running_g + R
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        #calculating loss for policy networkå
        entropies = []
        for log_prob, delta, entropy in zip(self.probs, deltas, self.entropies):
            entropies.append(entropy.detach().item())
            policy_loss += -1 * (self.alpha * entropy - log_prob.mean() * delta)
            # if random() < 0.5:
            #     policy_loss += log_prob.mean() * delta
            # else:
            # policy_loss += -1 * (self.alpha * entropy)

        # Update the policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy_optimizer.step()

        # Update the discriminator network
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
        self.discriminator_optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.actions = []
        self.states = []
        self.entropies = []
        self.correct = 0
        return (discriminator_loss, policy_loss, np.mean(entropies))
    
    def save_state_dict(self, env_name : str):
        torch.save(self.policy.state_dict(), "state_dicts/" + env_name + "DIAYN.pt")
        # torch.save(self.discriminator.state_dict(), "state_dicts/DIAYN_discriminator.pt")