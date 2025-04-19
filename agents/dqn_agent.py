import numpy as np
import torch
import torch.nn.functional as F
from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size, seed=0,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99,
                 lr=1e-3, tau=1e-3, update_every=4):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, fc1_dims=64, fc2_dims=64).to("cpu")
        self.qnetwork_target = QNetwork(state_size, action_size, fc1_dims=64, fc2_dims=64).to("cpu")
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if np.random.rand() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
