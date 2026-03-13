import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-06
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
ActionSpace = namedtuple('ActionSpace', ['low', 'high'])

class GaussianPolicy(nn.Module):

    def __init__(self, state_dim, action_size, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.fc_in = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_size)
        self.log_std_linear = nn.Linear(hidden_dim, action_size)
        self.apply(weights_init_)
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.0)

    def forward(self, state):
        x = F.relu(self.fc_in(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return (mean, log_std)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return (action, mean, log_prob)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class Critic(nn.Module):

    def __init__(self, state_size, hidden_dim=64, action_size=1):
        super(Critic, self).__init__()
        self.fc_in = nn.Linear(state_size + action_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_continuous):
        x = torch.cat([state, action_continuous], dim=1)
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_out(x)

class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return (x1, x2)
