import numpy as np
import random
from collections import namedtuple, deque
from policy_network import PolicyNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Our minibatch size
GAMMA = 0.90            # Discount factor
TAU = 1e-3              # For soft update of target parameters
LR = 5e-2               # Learning rate
UPDATE_EVERY = 4        # How often to update the network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Our policy network
        self.local_network = PolicyNetwork(
                state_size, action_size, seed).to(device)
        self.target_network = PolicyNetwork(
                state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize timestep
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience to replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Set the network to training mode
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()
        # Epsilon-Greedy calculation
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get best Q values for the target model
        Q_targets_next = self.target_network(next_states). \
            detach().max(1)[0].unsqueeze(1)
        # Compute the Q targets for the current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.local_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_network, self.target_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''
        Soft update model parameters
        theta_target = tau * theta_local + (1 - tau) * theta_target
        '''
        for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
                'Experience',
                field_names=[
                    'state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([
            e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([
            e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([
            e.reward for e in experiences if e is not None]))\
            .float().to(device)
        next_states = torch.from_numpy(np.vstack([
            e.next_state
            for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([
            e.done
            for e in experiences if e is not None])
                    .astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
