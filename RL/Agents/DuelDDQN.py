import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dims, filename, chkpt_dir, device=DEVICE):
        super(DuelingDeepQNetwork, self).__init__()

        # Initialize the network
        self.fc = nn.Linear(*input_dims, hidden_dims)
        self.val_net = nn.Linear(hidden_dims, 1)
        self.adv_net = nn.Linear(hidden_dims, n_actions)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, filename)

    def forward(self, state):
        x = F.relu(self.fc(state))
        val = self.val_net(x)
        advs = self.adv_net(x)

        return val, advs

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ReplayBuffer:
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool8)

    def store_data(self, state, action, reward, state_, terminal):
        index = self.mem_counter % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.reward_mem[index] = reward
        self.action_mem[index] = action
        self.terminal_mem[index] = terminal

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        terminals = self.terminal_mem[batch]

        return states, actions, rewards, states_, terminals


class Agent:
    def __init__(self, env, input_dims, n_actions, gamma, epsilon, update_time, n_episodes, lr=1e-3,
                 batch_size=64, hidden_dims=256, mem_size=100000, eps_min=0.01, eps_dec=5e-4,
                 chkpt_dir='./tmp/duelddqn'):
        self.env = env
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.update_time = update_time
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.mem_size = mem_size

        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_net = DuelingDeepQNetwork(input_dims, n_actions, hidden_dims,
                                         'duelddqn1_lunarlander', chkpt_dir)
        self.q_trg = DuelingDeepQNetwork(input_dims, n_actions, hidden_dims,
                                         'duelddqn2_lunarlander', chkpt_dir)

        self.learner_step = 0
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            self.q_net.eval()
            state = torch.tensor(observation, dtype=torch.float).to(self.q_net.device)
            _, advantage = self.q_net.forward(state)
            self.q_net.train()
            return torch.argmax(advantage).item()
        else:
            return np.random.choice(self.action_space)

    def update_target_network(self):
        if self.learner_step % self.update_time == 0:
            self.q_trg.load_state_dict(self.q_net.state_dict())

    def store_data(self, state, action, reward, state_, terminal):
        self.memory.store_data(state, action, reward, state_, terminal)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_net.save_checkpoint()
        self.q_trg.save_checkpoint()

    def load_model(self):
        self.q_net.load_checkpoint()
        self.q_trg.load_checkpoint()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # Update the target network
        self.update_target_network()

        # Sample memory and convert it to tensors
        states, actions, rewards, new_states, terminals = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states).to(self.q_net.device)
        actions = torch.tensor(actions).to(self.q_net.device)
        rewards = torch.tensor(rewards).to(self.q_net.device)
        states_ = torch.tensor(new_states).to(self.q_net.device)
        terminals = torch.tensor(terminals).to(self.q_net.device)

        # index the batch elements
        indices = np.arange(self.batch_size)

        # Get the sampled Q-values for the next sampled states and choose the best action
        vals_qnet_, advs_qnet_ = self.q_net.forward(states_)
        q_preds_ = torch.add(vals_qnet_, (advs_qnet_ - advs_qnet_.mean(dim=1, keepdim=True)))
        actions_ = torch.argmax(q_preds_, dim=1)
        # Compute the target Q-value
        vals_qtrg_, advs_qtrg_ = self.q_trg.forward(states_)
        q_trg_ = torch.add(vals_qtrg_, (advs_qtrg_ - advs_qtrg_.mean(dim=1, keepdim=True)))
        q_trg_[terminals] = 0.0
        q_targets = rewards + self.gamma * q_trg_[indices, actions_]

        # Get the Q-values for the current states
        vals_qnet, advs_qnet = self.q_net.forward(states)
        q_preds = torch.add(vals_qnet,
                            (advs_qnet - advs_qnet.mean(dim=1, keepdim=True)))[indices, actions]

        # Compute the loss and backpropagate it through the network
        loss = self.loss(q_preds, q_targets).to(self.q_net.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increase the episode counter
        self.learner_step += 1

        # Decrease the epsilon if possible
        self.decrement_epsilon()

    def train(self):
        scores, eps_history = [], []
        for i in range(self.n_episodes):
            score = 0
            terminal = False
            observation = self.env.reset()[0]
            while not terminal:
                action = self.choose_action(observation)
                observation_, reward, done, truncated, info = self.env.step(action)
                score += reward
                terminal = done or truncated
                self. store_data(observation, action, reward, observation_, terminal)
                self.learn()
                observation = observation_

            scores.append(score)
            eps_history.append(self.epsilon)
            avg_score = np.mean(scores[-100:])

            print(
                'episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
                               'epsilon %.2f' % self.epsilon
            )

        return scores, eps_history
