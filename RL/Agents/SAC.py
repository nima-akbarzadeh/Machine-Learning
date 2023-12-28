import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, reparam_noise, hidden1_dims, hidden2_dims, lr,
                 filename, chkpt_dir, device=DEVICE):
        super(ActorNetwork, self).__init__()

        # Initialize the network
        self.fc1 = nn.Linear(*input_dims, hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.fc3 = nn.Linear(hidden2_dims, n_actions)
        self.fc4 = nn.Linear(hidden2_dims, n_actions)

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.reparam_noise = reparam_noise
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        sigma = torch.clamp(self.fc4(x), min=self.reparam_noise, max=1)
        # clamp is faster than sigmoid

        return mu, sigma

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

        # Initialize the network
        self.fc1 = nn.Linear(input_dims[0] + n_actions, hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.fc3 = nn.Linear(hidden2_dims, 1)

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, input_dims, hidden1_dims, hidden2_dims, lr, filename, chkpt_dir,
                 device=DEVICE):
        super(ValueNetwork, self).__init__()

        # Initialize the network
        self.fc1 = nn.Linear(*input_dims, hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.fc3 = nn.Linear(hidden2_dims, 1)

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ReplayBuffer:
    def __init__(self, mem_size, input_dims, n_actions):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_mem = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool_)

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

    def __init__(self, env, input_dims, n_actions, gamma, update_factor, reward_scale,
                 reparam_noise, n_episodes, lr_actor=0.0003, lr_critic=0.0003, batch_size=64,
                 hidden1_dims=256, hidden2_dims=256, mem_size=100000, chkpt_dir='./tmp/sac'):
        self.env = env
        self.n_actions = n_actions
        self.max_action = env.action_space.high
        self.gamma = gamma
        self.rho = update_factor
        self.reward_scale = reward_scale
        self.reparam_noise = reparam_noise
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.learner_step = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.act_net = ActorNetwork(input_dims, n_actions, reparam_noise, hidden1_dims,
                                    hidden2_dims, lr_actor, 'sac_actor_landlunar', chkpt_dir)

        self.qval1_net = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                       'sac_critic1_landlunar', chkpt_dir)
        self.qval2_net = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                       'sac_critic2_landlunar', chkpt_dir)

        self.val_net = ValueNetwork(input_dims, hidden1_dims, hidden2_dims, lr_critic,
                                    'sac_value_landlunar', chkpt_dir)
        self.val_trg = ValueNetwork(input_dims, hidden1_dims, hidden2_dims, lr_critic,
                                    'sac_value_landlunar', chkpt_dir)

        # Update the target network
        self.update_target_network()

    def sample_normal(self, states, reparameterize=True):
        mu, sigma = self.act_net.forward(states)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()  # sample with noise
        else:
            actions = probabilities.sample()

        scaled_actions = torch.tanh(actions) * torch.tensor(self.max_action).to(self.act_net.device)
        log_probs = probabilities.log_prob(actions) \
                    - torch.log(1 - scaled_actions.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return scaled_actions, log_probs.view(-1)

    def choose_action(self, observation):
        self.act_net.eval()
        state = torch.tensor(observation).to(self.act_net.device)
        actions, _ = self.sample_normal(state, reparameterize=False)
        self.act_net.train()
        return actions.cpu().detach().numpy()[0]

    def store_data(self, state, action, reward, new_state, done):
        self.memory.store_data(state, action, reward, new_state, done)

    def update_target_network(self):
        val_net_dict = dict(self.val_net.named_parameters())
        val_trg_dict = dict(self.val_trg.named_parameters())
        for name in val_net_dict:
            val_net_dict[name] = self.rho * val_net_dict[name].clone() \
                                 + (1 - self.rho) * val_trg_dict[name].clone()
        self.val_trg.load_state_dict(val_net_dict)

    def save_model(self):
        self.act_net.save_checkpoint()
        self.qval1_net.save_checkpoint()
        self.qval2_net.save_checkpoint()
        self.val_net.save_checkpoint()
        self.val_trg.save_checkpoint()

    def load_model(self):
        self.act_net.load_checkpoint()
        self.qval1_net.load_checkpoint()
        self.qval2_net.load_checkpoint()
        self.val_net.load_checkpoint()
        self.val_trg.load_checkpoint()

    def get_targets(self, rewards, states_, terminals):
        self.val_trg.eval()
        value_ = self.val_trg(states_).view(-1)
        value_[terminals] = 0.0

        return self.reward_scale * rewards + self.gamma * value_

    def learn(self):
        # Off-policy learning
        if self.memory.mem_counter < self.batch_size:
            return

        # Sample memory and convert it to tensors
        states, actions, rewards, new_states, terminals = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states).to(self.qval1_net.device)
        actions = torch.tensor(actions).to(self.qval1_net.device)
        rewards = torch.tensor(rewards).to(self.qval1_net.device)
        states_ = torch.tensor(new_states).to(self.qval1_net.device)
        terminals = torch.tensor(terminals).to(self.qval1_net.device)

        # Compute the current q-values
        self.qval1_net.train()
        self.qval2_net.train()
        q_preds1 = self.qval1_net.forward(states, actions).view(-1)
        q_preds2 = self.qval2_net.forward(states, actions).view(-1)

        # Compute the target values
        q_targets = self.get_targets(rewards, states_, terminals)

        # Compute the critic loss
        critic_1_loss = 0.5 * F.mse_loss(q_targets, q_preds1)
        critic_2_loss = 0.5 * F.mse_loss(q_targets, q_preds2)
        critic_loss = critic_1_loss + critic_2_loss

        # Backpropagate
        self.qval1_net.optimizer.zero_grad()
        self.qval2_net.optimizer.zero_grad()
        critic_loss.backward()
        self.qval1_net.optimizer.step()
        self.qval2_net.optimizer.step()

        # Compute the current values
        value = self.val_net(states).view(-1)

        # Compute the target values
        self.act_net.eval()
        self.qval1_net.eval()
        self.qval2_net.eval()
        actions, log_probs = self.sample_normal(states, reparameterize=False)
        q_preds_new1 = self.qval1_net.forward(states, actions).view(-1)
        q_preds_new2 = self.qval2_net.forward(states, actions).view(-1)
        value_target = torch.min(q_preds_new1, q_preds_new2).view(-1) - log_probs

        # Compute the value loss
        value_loss = 0.5 * F.mse_loss(value, value_target)

        # Backpropagate
        self.val_net.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.val_net.optimizer.step()

        # Compute the actor loss
        self.act_net.train()
        self.qval1_net.eval()
        self.qval2_net.eval()
        actions, log_probs = self.sample_normal(states, reparameterize=True)
        q_preds_new1 = self.qval1_net.forward(states, actions)
        q_preds_new2 = self.qval2_net.forward(states, actions)
        actor_loss = torch.mean(log_probs - torch.min(q_preds_new1, q_preds_new2).view(-1))

        # Backpropagate
        self.act_net.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.act_net.optimizer.step()

        # Update the target network
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
