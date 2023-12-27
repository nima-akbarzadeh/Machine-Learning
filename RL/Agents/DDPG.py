import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden1_dims, hidden2_dims, lr, filename, chkpt_dir,
                 device=DEVICE):
        super(ActorNetwork, self).__init__()

        # Initialize the network
        self.fc1 = nn.Linear(*input_dims, hidden1_dims)
        self.bn1 = nn.LayerNorm(hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.bn2 = nn.LayerNorm(hidden2_dims)
        self.fc3 = nn.Linear(hidden2_dims, n_actions)

        # Initialize the weights (according to the paper)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        f3 = 0.003
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))

        # The action is bounded between [-1, 1] which can be generalized to any space centered
        # around 0 if the output is multiplied by a scalar.
        return torch.tanh(self.fc3(x))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden1_dims, hidden2_dims, lr, filename, chkpt_dir,
                 device=DEVICE):
        super(CriticNetwork, self).__init__()

        # Ecode the state
        self.state_fc1 = nn.Linear(*input_dims, hidden1_dims)
        self.state_bn1 = nn.LayerNorm(hidden1_dims)
        self.state_fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.state_bn2 = nn.LayerNorm(hidden2_dims)

        # Encode the action
        self.action_fc1 = nn.Linear(n_actions, hidden2_dims)

        # Initialize the last layer
        self.combined_fc1 = nn.Linear(hidden2_dims, 1)

        # Initialize the weights (according to the paper)
        f1 = 1./np.sqrt(self.state_fc1.weight.data.size()[0])
        nn.init.uniform_(self.state_fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.state_fc1.bias.data, -f1, f1)
        f2 = 1./np.sqrt(self.state_fc2.weight.data.size()[0])
        nn.init.uniform_(self.state_fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.state_fc2.bias.data, -f2, f2)
        f3 = 0.003
        nn.init.uniform_(self.combined_fc1.weight.data, -f3, f3)
        nn.init.uniform_(self.combined_fc1.bias.data, -f3, f3)

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)


    def forward(self, state, action):
        s = F.relu(self.state_bn1(self.state_fc1(state)))
        s = self.state_bn2(self.state_fc2(s))

        a = F.relu(self.action_fc1(action))

        return self.combined_fc1(F.relu(torch.add(s, a)))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev \
            + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer:
    def __init__(self, mem_size, input_dims, n_actions):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_mem = np.zeros((self.mem_size, n_actions), dtype=np.float32)
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


class Agent(object):
    def __init__(self, env, input_dims, n_actions, gamma, update_factor, n_episodes,
                 lr_actor=0.000025, lr_critic=0.00025, batch_size=64, hidden1_dims=256,
                 hidden2_dims=256, mem_size=100000, chkpt_dir='./tmp/ddpg'):
        self.env = env
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.rho = update_factor

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.noise = ActionNoise(mu=np.zeros(n_actions))

        self.act_net = ActorNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_actor,
                                    'ddpg_actor1_landlunar', chkpt_dir)
        self.act_trg = ActorNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_actor,
                                    'ddpg_actor2_landlunar', chkpt_dir)

        self.qval_net = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                      'ddpg_critic1_landlunar', chkpt_dir)
        self.qval_trg = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                      'ddpg_critic2_landlunar', chkpt_dir)

    def choose_action(self, observation):
        self.act_net.eval()
        observation = torch.tensor(observation, dtype=torch.float).to(self.act_net.device)
        action = self.act_net.forward(observation).to(self.act_net.device)
        noicy_action = action \
                       + torch.tensor(self.noise(), dtype=torch.float).to(self.act_net.device)
        self.act_net.train()
        return noicy_action.cpu().detach().numpy()

    def store_data(self, state, action, reward, new_state, done):
        self.memory.store_data(state, action, reward, new_state, done)

    def save_model(self):
        self.act_net.save_checkpoint()
        self.act_trg.save_checkpoint()
        self.qval_net.save_checkpoint()
        self.qval_trg.save_checkpoint()

    def load_model(self):
        self.act_net.load_checkpoint()
        self.act_trg.load_checkpoint()
        self.qval_net.load_checkpoint()
        self.qval_trg.load_checkpoint()

    def update_target_network(self):
        act_net_dict = dict(self.act_net.named_parameters())
        act_trg_dict = dict(self.act_trg.named_parameters())
        val_net_dict = dict(self.qval_net.named_parameters())
        val_trg_dict = dict(self.qval_trg.named_parameters())

        for name in act_net_dict:
            act_net_dict[name] = self.rho * act_net_dict[name].clone() \
                                 + (1 - self.rho) * act_trg_dict[name].clone()
        for name in val_net_dict:
            val_net_dict[name] = self.rho * val_net_dict[name].clone() \
                                 + (1 - self.rho) * val_trg_dict[name].clone()

        self.act_trg.load_state_dict(act_net_dict)
        self.qval_trg.load_state_dict(val_net_dict)


    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # Update the target network
        self.update_target_network()

        # Sample memory and convert it to tensors
        states, actions, rewards, new_states, terminals = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states).to(self.qval_net.device)
        actions = torch.tensor(actions).to(self.qval_net.device)
        rewards = torch.tensor(rewards).to(self.qval_net.device)
        states_ = torch.tensor(new_states).to(self.qval_net.device)
        terminals = torch.tensor(terminals).to(self.qval_net.device)

        # Compute the target values to be used in loss function
        self.act_trg.eval()
        self.qval_trg.eval()
        target_actions_ = self.act_trg.forward(states_)
        target_values_ = self.qval_trg.forward(states_, target_actions_)
        targets = []
        for j in range(self.batch_size):
            targets.append(rewards[j] + self.gamma * target_values_[j] * terminals[j])
        targets = torch.tensor(targets).to(self.qval_net.device)
        targets = targets.view(self.batch_size, 1)

        # Compute the loss and backpropagate it through the network
        self.qval_net.train()
        val_net_values = self.qval_net.forward(states, actions)
        val_net_loss = F.mse_loss(targets, val_net_values)
        self.qval_net.optimizer.zero_grad()
        val_net_loss.backward()
        self.qval_net.optimizer.step()

        self.qval_net.eval()
        self.act_net.train()
        mu = self.act_net.forward(states)
        actor_loss = torch.mean(-self.qval_net.forward(states, mu))
        self.act_net.optimizer.zero_grad()
        actor_loss.backward()
        self.act_net.optimizer.step()

        self.update_target_network()

    def train(self):
        scores = []
        for i in range(self.n_episodes):
            score = 0
            terminal = False
            observation = self.env.reset()[0]
            while not terminal:
                action = self.choose_action(observation)
                observation_, reward, done, truncated, info = self.env.step(action)
                score += reward
                terminal = done or truncated
                self.store_data(observation, action, reward, observation_, terminal)
                self.learn()
                observation = observation_

            scores.append(score)
            avg_score = np.mean(scores[-100:])

            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        return scores
