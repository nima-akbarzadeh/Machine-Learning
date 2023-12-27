import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden1_dims, hidden2_dims, lr, filename, chkpt_dir,
                 device=DEVICE):
        super(ActorNetwork, self).__init__()

        # Initialize the network
        self.act_net = nn.Sequential(
            nn.Linear(*input_dims, hidden1_dims),
            nn.ReLU(),
            nn.Linear(hidden1_dims, hidden2_dims),
            nn.ReLU(),
            nn.Linear(hidden2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)

    def forward(self, state):
        return Categorical(self.act_net(state))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden1_dims, hidden2_dims, lr, filename, chkpt_dir,
                 device=DEVICE):
        super(CriticNetwork, self).__init__()

        self.val_net = nn.Sequential(
            nn.Linear(*input_dims, hidden1_dims),
            nn.ReLU(),
            nn.Linear(hidden1_dims, hidden2_dims),
            nn.ReLU(),
            nn.Linear(hidden2_dims, 1)
        )

        # Instantiate the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, filename)

    def forward(self, state):
        return self.val_net(state)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.vals = []
        self.terminals = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.rewards), np.array(self.vals), np.array(self.terminals), batches

    def store_data(self, state, action, prob, reward, val, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.vals.append(val)
        self.terminals.append(terminal)

    def clear_data(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.vals = []
        self.terminals = []


class Agent:
    def __init__(self, env, input_dims, n_actions, gamma, gae_lambda,
                 policy_clip, policy_horizon, n_epochs, n_episodes, lr=0.0003, batch_size=5,
                 hidden1_dims=256, hidden2_dims=256, chkpt_dir='./tmp/ppo'):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.policy_horizon = policy_horizon
        self.n_epochs = n_epochs
        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.memory = PPOMemory(batch_size)

        self.act_net = ActorNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims, lr,
                                    'ppo_actor_landlunar', chkpt_dir)
        self.val_net = CriticNetwork(input_dims, hidden1_dims, hidden2_dims, lr,
                                     'ppo_critic_landlunar', chkpt_dir)

    def choose_action(self, observation):
        self.act_net.eval()
        self.val_net.eval()
        state = torch.tensor([observation], dtype=torch.float).to(self.act_net.device)

        dist = self.act_net(state)
        action = dist.sample()
        logprobs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        value = self.val_net(state)
        value = torch.squeeze(value).item()

        self.act_net.train()
        self.val_net.train()
        return action, logprobs, value

    def store_data(self, state, action, probs, vals, reward, terminal):
        self.memory.store_data(state, action, probs, vals, reward, terminal)

    def save_model(self):
        self.act_net.save_checkpoint()
        self.val_net.save_checkpoint()

    def load_model(self):
        self.act_net.load_checkpoint()
        self.val_net.load_checkpoint()

    def learn(self):
        for _ in range(self.n_epochs):
            states_arr, actions_arr, old_probs_arr, vals_arr, rewards_arr, terminals_arr, batches \
                = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)

            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount * (rewards_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(terminals_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.act_net.device)

            values = torch.tensor(values).to(self.act_net.device)
            for batch in batches:
                states = torch.tensor(states_arr[batch], dtype=torch.float).to(self.act_net.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.act_net.device)
                actions = torch.tensor(actions_arr[batch]).to(self.act_net.device)

                dist = self.act_net(states)
                critic_value = self.val_net(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.act_net.optimizer.zero_grad()
                self.val_net.optimizer.zero_grad()
                total_loss.backward()
                self.act_net.optimizer.step()
                self.val_net.optimizer.step()

        self.memory.clear_data()

    def train(self):
        time_step = 0
        learner_step = 0
        # best_score = self.env.reward_range[0]
        scores = []
        for i in range(self.n_episodes):
            score = 0
            terminal = False
            observation = self.env.reset()[0]
            while not terminal:
                action, logprob, val = self.choose_action(observation)
                observation_, reward, done, truncated, info = self.env.step(action)
                score += reward
                terminal = done or truncated
                self.store_data(observation, action, logprob, reward, val, terminal)
                time_step += 1
                if time_step % self.policy_horizon == 0:
                    self.learn()
                    learner_step += 1
                observation = observation_

            scores.append(score)
            avg_score = np.mean(scores[-100:])

            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

            # if avg_score > best_score:
            #     best_score = avg_score
            #     self.save_model()

        return scores
