import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, hidden1_dims, hidden2_dims, n_actions, device=DEVICE):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden1_dims = hidden1_dims
        self.hidden2_dims = hidden2_dims
        self.n_actions = n_actions
        # The star is for multiple elements of the observation vector
        self.fc1 = nn.Linear(*self.input_dims, self.hidden1_dims)
        self.fc2 = nn.Linear(self.hidden1_dims, self.hidden2_dims)
        self.fc3 = nn.Linear(self.hidden2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 memory_size=100000, eps_end=0.05, eps_dec=5e-4, device=DEVICE):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = memory_size
        self.batch_size = batch_size
        self.device = device
        # memory counter to track the first available position of the first memory
        self.mem_counter = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.val_net = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    hidden1_dims=256, hidden2_dims=256, device=DEVICE)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.val_net.device)
            actions = self.val_net.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return

        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        state_batch = torch.tensor(self.state_memory[batch]).to(self.val_net.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.val_net.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.val_net.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.val_net.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_eval = self.val_net.forward(state_batch)[batch_index, action_batch]

        q_next = self.val_net.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        self.val_net.optimizer.zero_grad()
        loss = self.val_net.loss(q_target, q_eval).to(self.val_net.device)
        loss.backward()
        self.val_net.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
