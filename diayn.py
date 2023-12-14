import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.functional import softmax
from reinforce import Policy_Network

class Discriminator_Network(nn.Module):
    '''Discriminator Network'''

    def __init__(self, obs_space_dims : int, num_skills : int):
        """Initializes a neural network that estimates the probability of a skill given a state
        
        Args:
            obs_space_dims: Dimension of the observation space
            num_skills: Number of skills
        """

        super().__init__()

        #dimensions of each hidden layer
        # hidden_dims = [512, 64, 32]
        hidden_dims = [16, 8]


        #constructing shared net from hidden layer dimensions
        sequential_input = []
        prev_space = obs_space_dims
        for dim in hidden_dims:
            sequential_input.append(
                nn.Linear(prev_space, dim)
            )
            sequential_input.append(nn.Tanh())
            prev_space = dim

        self.shared_net = nn.Sequential(*sequential_input)
        self.output = nn.Linear(hidden_dims[-1], num_skills)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        hidden_features = self.shared_net(x.float())
        output = self.output(hidden_features)
        return output #don't need to use softmax because CrossEntropyLoss does it for us

class DIAYN:
    '''DIAYN algorithm'''

    def __init__(self, num_skills : int, obs_space_dims : int, action_space_dims : int):

        self.num_skills = num_skills
        self.obs_space_dism = obs_space_dims
        self.action_space_dims = action_space_dims
        self.eps = 1e-6  # small number for mathematical stability
        self.learning_rate = 1e-4 # learning rate for optimizer (both discriminator and policy)

        self.discriminator = Discriminator_Network(obs_space_dims, num_skills) #discriminator state -> skill
        self.policy = Policy_Network(obs_space_dims + num_skills, action_space_dims) #policy concatenated state and skill -> action

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        self.alpha = 0.1 #empirically found to be good in DIAYN
        self.gamma = 0.99 #discount factor
        self.rewards = [] #intrinsic rewards from discriminator

        self.actions = []
        self.states = []
        self.z = None
        self.probs = []
        self.entropies = []



    def sample_action(self, state : np.ndarray, skill : int) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment
            skill: Skill to condition on

        Returns:
            action: Action to be performed, or a_t ~ pi_theta (a_t | s_t, z)
        """

        self.states.append(torch.from_numpy(state))
        one_hot = np.zeros(self.num_skills)
        one_hot[skill] = 1

        state = np.concatenate((state, one_hot))
        state = torch.from_numpy(state).float()
        action_means, action_stddevs = self.policy(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means + self.eps, action_stddevs + self.eps)
        action = distrib.sample()
        entropy = distrib.entropy().sum(axis=-1)
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
        #TODO: implement w/ broadcasting instead of one-by-one


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
            R = torch.log(R)
            discriminator_loss += R
            running_g = self.gamma * running_g + R
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        #calculating loss for policy network
        for log_prob, delta, entropy in zip(self.probs, deltas, self.entropies):
            policy_loss += -1 * (log_prob.mean() * delta + self.alpha * entropy)

        # Update the policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update the discriminator network
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.actions = []
        self.states = []
        self.entropies = []

        return (discriminator_loss, policy_loss)
    
    def save_state_dict(self, env_name : str):
        torch.save(self.policy.state_dict(), "state_dicts/" + env_name + "DIAYN.pt")
        # torch.save(self.discriminator.state_dict(), "state_dicts/DIAYN_discriminator.pt")