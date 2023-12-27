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
        self.fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.fc3 = nn.Linear(hidden2_dims, n_actions)

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

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
    def __init__(self, env, input_dims, n_actions, gamma, update_factor, update_actor_time, warmup,
                 noise, n_episodes, lr_actor=0.001, lr_critic=0.001, batch_size=64,
                 hidden1_dims=256, hidden2_dims=256, mem_size=100000, chkpt_dir='./tmp/td3'):
        self.env = env
        self.n_actions = n_actions
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.gamma = gamma
        self.rho = update_factor
        self.update_actor_time = update_actor_time
        self.warmup = warmup
        self.noise = noise
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.learner_step = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.act_net = ActorNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_actor,
                                    'td3_actor1_landlunar', chkpt_dir)
        self.act_trg = ActorNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_actor,
                                    'td3_actor2_landlunar', chkpt_dir)

        self.qval1_net = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                       'td3_critic11_landlunar', chkpt_dir)
        self.qval1_trg = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                       'td3_critic21_landlunar', chkpt_dir)

        self.qval2_net = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                       'td3_critic21_landlunar', chkpt_dir)
        self.qval2_trg = CriticNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr_critic,
                                       'td3_critic22_landlunar', chkpt_dir)

    def choose_action(self, observation):
        if self.learner_step < self.warmup:
            action = torch.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            state = torch.tensor(observation, dtype=torch.float).to(self.act_net.device)
            action = self.act_net.forward(state).to(self.act_net.device)
        noisy_action = torch.clamp(
            action + torch.tensor(np.random.normal(scale=self.noise),
                                  dtype=torch.float).to(self.act_net.device),
            self.min_action[0], self.max_action[0]
        )

        return noisy_action.cpu().detach().numpy()

    def store_data(self, state, action, reward, new_state, done):
        self.memory.store_data(state, action, reward, new_state, done)

    def save_model(self):
        self.act_net.save_checkpoint()
        self.act_trg.save_checkpoint()
        self.qval1_net.save_checkpoint()
        self.qval2_net.save_checkpoint()
        self.qval1_trg.save_checkpoint()
        self.qval2_trg.save_checkpoint()

    def load_model(self):
        self.act_net.load_checkpoint()
        self.act_trg.load_checkpoint()
        self.qval1_net.load_checkpoint()
        self.qval2_net.load_checkpoint()
        self.qval1_trg.load_checkpoint()
        self.qval2_trg.load_checkpoint()

    def update_target_network(self):
        act_net_dict = dict(self.act_net.named_parameters())
        act_trg_dict = dict(self.act_trg.named_parameters())
        val1_net_dict = dict(self.qval1_net.named_parameters())
        val1_trg_dict = dict(self.qval1_trg.named_parameters())
        val2_net_dict = dict(self.qval2_net.named_parameters())
        val2_trg_dict = dict(self.qval2_trg.named_parameters())

        for name in act_net_dict:
            act_net_dict[name] = self.rho * act_net_dict[name].clone() \
                                 + (1 - self.rho) * act_trg_dict[name].clone()
        for name in val1_net_dict:
            val1_net_dict[name] = self.rho * val1_net_dict[name].clone() \
                                  + (1 - self.rho) * val1_trg_dict[name].clone()
        for name in val2_net_dict:
            val2_net_dict[name] = self.rho * val2_net_dict[name].clone() \
                                  + (1 - self.rho) * val2_trg_dict[name].clone()

        self.act_trg.load_state_dict(act_net_dict)
        self.qval1_trg.load_state_dict(val1_net_dict)
        self.qval2_trg.load_state_dict(val2_net_dict)

    def get_targets(self, rewards, states_, terminals):
        # Compute the target actions
        # The outer clamp is for taking the action in the feasible range
        self.act_trg.eval()
        actions_ = self.act_trg.forward(states_)
        actions_ = torch.clamp(
            actions_ + torch.clamp(
                torch.tensor(np.random.normal(scale=0.2)),
                -0.5, 0.5
            ),
            self.min_action[0], self.max_action[0]
        )

        # Compute the target values
        self.qval1_trg.eval()
        self.qval2_trg.eval()
        q_trg1_ = self.qval1_trg.forward(states_, actions_)
        q_trg2_ = self.qval1_trg.forward(states_, actions_)
        q_trg1_[terminals] = 0.0
        q_trg2_[terminals] = 0.0
        q_trg1_ = q_trg1_.view(-1)
        q_trg2_ = q_trg2_.view(-1)
        q_trg_ = torch.min(q_trg1_, q_trg2_)
        q_targets = rewards + self.gamma * q_trg_

        return q_targets.view(self.batch_size, 1)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # Update the target network
        self.update_target_network()

        # Sample memory and convert it to tensors
        states, actions, rewards, new_states, terminals = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states).to(self.qval1_net.device)
        actions = torch.tensor(actions).to(self.qval1_net.device)
        rewards = torch.tensor(rewards).to(self.qval1_net.device)
        states_ = torch.tensor(new_states).to(self.qval1_net.device)
        terminals = torch.tensor(terminals).to(self.qval1_net.device)

        # Compute the current q-values
        q_preds1 = self.qval1_net.forward(states, actions)
        q_preds2 = self.qval2_net.forward(states, actions)

        # Compute the target values
        q_targets = self.get_targets(rewards, states_, terminals)

        # Compute the critic loss
        q1_loss = F.mse_loss(q_targets, q_preds1)
        q2_loss = F.mse_loss(q_targets, q_preds2)
        critic_loss = q1_loss + q2_loss

        # Backpropagate
        self.qval1_net.optimizer.zero_grad()
        self.qval2_net.optimizer.zero_grad()
        critic_loss.backward()
        self.qval1_net.optimizer.step()
        self.qval2_net.optimizer.step()

        # Check if the update_actor_time is arrived
        if self.learner_step % self.update_actor_time != 0:
            return
        else:
            # Compute the actor loss
            actor_q1_loss = self.qval1_net.forward(states, self.act_net.forward(states))
            actor_loss = -torch.mean(actor_q1_loss)

            # Backpropagate
            self.act_net.optimizer.zero_grad()
            actor_loss.backward()
            self.act_net.optimizer.step()

        # Increase the episode counter
        self.learner_step += 1

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
