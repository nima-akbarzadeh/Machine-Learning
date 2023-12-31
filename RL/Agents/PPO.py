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
        n_data = len(self.states)
        batch_start = np.arange(0, n_data, self.batch_size)
        # Shuffle the data
        indices = np.arange(n_data, dtype=np.int64)
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
    def __init__(self, env, input_dims, n_actions, gamma, smoothing_lambda,
                 policy_clip, policy_horizon, n_epochs, n_episodes, lr=0.0003, batch_size=5,
                 hidden1_dims=256, hidden2_dims=256, chkpt_dir='./tmp/ppo'):
        self.env = env
        self.gamma = gamma
        self.smoothing_lambda = smoothing_lambda
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

        state = torch.tensor(observation, dtype=torch.float).to(self.act_net.device)
        dist = self.act_net(state)
        action = dist.sample()
        logprobs = dist.log_prob(action).item()
        action = action.item()

        value = self.val_net(state)
        value = value.item()

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

    def get_advantages(self, rewards, vals, terminals):
        # Compute the advantage for each step in the epoch
        # Note the sequence of rewards_np is not shuffled
        advantages = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (
                        rewards[k] + self.gamma * vals[k + 1] * (1 - int(terminals[k])) - vals[k]
                )
                discount *= self.gamma * self.smoothing_lambda
            advantages[t] = a_t

        return torch.tensor(advantages).to(self.act_net.device)

    def learn(self):
        # On-policy learning
        for _ in range(self.n_epochs):
            states_np, actions_np, logprobs_np, vals_np, rewards_np, terminals_np, batches \
                = self.memory.generate_batches()

            advantages = self.get_advantages(rewards_np, vals_np, terminals_np)
            values = torch.tensor(vals_np).to(self.act_net.device)
            for batch in batches:
                states = torch.tensor(states_np[batch], dtype=torch.float).to(self.act_net.device)
                actions = torch.tensor(actions_np[batch]).to(self.act_net.device)
                old_logprobs = torch.tensor(logprobs_np[batch]).to(self.act_net.device)

                # Compute the critic target values
                v_targets = advantages[batch] + values[batch]

                # Compute the critic current values
                v_preds = self.val_net(states)
                v_preds = torch.squeeze(v_preds)

                # Compute the critic loss
                critic_loss = (v_targets - v_preds) ** 2
                critic_loss = critic_loss.mean()

                # Compute the actor loss
                actor_dist = self.act_net(states)
                new_logprobs = actor_dist.log_prob(actions)
                prob_ratio = (new_logprobs - old_logprobs).exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantages[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Compute the total loss
                total_loss = actor_loss + 0.5 * critic_loss

                # Backpropagate
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
