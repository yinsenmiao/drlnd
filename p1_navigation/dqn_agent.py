import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as function
import torch.optim as optim

# global constant
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 1e-3  # learning rate
BUFFER_SIZE = int(1e6)  # buffer size for experience replay
UPDATE_CYCLE = 8
BATCH_SIZE = 64
GAMMA = 0.99  # discount factor
TAU = 1e-3 # for soft update of target parameters


class QNetwork(nn.Module):
    # define QNetwork by inherit from torch's network class
    def __init__(self, state_dim, action_dim, seed, fc1_dim=128, fc2_dim=64, dueling=False):
        """
        :param state_dim: dimension of the state space
        :param action_dim: dimension of the action space
        :param seed: randomization seed for reproducibility
        :param fc1_dim: dimension of the L1 layer
        :param fc2_dim: dimensioan of the L2 layer
        :param dueling: if dueling is enabled or not
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) # set randomization seed for reproducibility
        self.dueling = dueling
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        if self.dueling:
            self.action_dim = action_dim
            self.value_fc = nn.Linear(fc2_dim, 1)
            self.advantage_fc = nn.Linear(fc2_dim, action_dim)
        else:
            self.fc3 = nn.Linear(fc2_dim, action_dim)

    def forward(self, state):
        """
        :param state: give states (state_dim, batch_size)
        :return: generate actions (action_dim, batch_size)
        """
        x = function.relu(self.fc1(state))
        x = function.relu(self.fc2(x))
        if self.dueling:
            value = self.value_fc(x).expand(x.size(0), self.action_dim)
            advantage = self.advantage_fc(x)
            x = value + advantage - advantage.mean(1).unsqueeze(1).expand(x.size(0), self.action_dim)
        else:
            x = self.fc3(x)
        return x


class ReplayBuffer:
    """
        Buffer to store experiences tuples
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        :param action_size: dimension of the action space
        :param buffer_size: size of the buffer
        :param batch_size: size of the batch
        :param seed: randomization seed for reproducibility
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = seed
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        # add experience to memory deque
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # random sample a batch of experiences
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        # return current size of the memory
        return len(self.memory)


class Agent():
    """
        agent interacts with environment and learn
    """

    def __init__(self, state_size, action_size, seed, filename=None, dueling=False, double=False):
        """
        :param state_size: dimension of the state space
        :param action_size: dimension of the action space
        :param seed: randomization seed for reproducibility
        :param filename: save model
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double = double  # turn on double DQN or not

        # initialize QNetwork
        self.qnn_local = QNetwork(state_size, action_size, seed, dueling=dueling).to(device)
        self.qnn_target = QNetwork(state_size, action_size, seed, dueling=dueling).to(device)
        self.optimizer = optim.Adam(self.qnn_local.parameters(), lr=LR)

        # load pre-trained parameters if existing
        if filename != None:
            try:
                weights = torch.load(filename)
                self.qnn_local.load_state_dict(weights)
                self.qnn_target.load_state_dict(weights)
                print("Weight loaded successful from the file: %s." % filename)
            except:
                print("No available weight file.")

        # replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # time step
        self.t_step = 0

    def soft_update(self, local_model, target_model, tau):
        """
            Soft update model parameters (alpha -constant)
            theta_target = tau * theta_local + (1 - tau) * theta_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def learn(self, experiences, gamma):
        # update Q network parameters given a batch of experiences
        states, actions, rewards, next_states, dones = experiences

        if not self.double:
            q_target_next = self.qnn_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            best_local_actions = self.qnn_local(states).detach().max(1)[1].unsqueeze(1)
            double_dqn_targets = self.qnn_target(next_states).detach()
            q_target_next = torch.gather(double_dqn_targets, 1, best_local_actions.long())

        # compute q target for current states
        q_target = rewards + (gamma * q_target_next * (1 - dones))

        # compute q values from local model
        q_local = self.qnn_local(states).gather(1, actions.long())
        loss = function.mse_loss(q_local, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnn_local, self.qnn_target, TAU)

    def step(self, state, action, reward, next_state, done):
        # save experiences into memory and learn later
        self.memory.add(state, action, reward, next_state, done)  # add to memory

        # learn from experiences every UPDATE_CYCLE steps
        self.t_step = (self.t_step + 1) % UPDATE_CYCLE
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0):
        # Returns actions for given state as per current policy.
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnn_local.eval()
        with torch.no_grad():
            action_values = self.qnn_local(state)
        self.qnn_local.train()

        # epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)  # cast to int32
        else:
            return random.choice(np.arange(self.action_size)).astype(int)  # cast to int32