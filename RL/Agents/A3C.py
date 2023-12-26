import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from abc import ABC

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dims, filename, chkpt_dir, device=DEVICE):
        super(ActorCritic, self).__init__()

        self.fc_v = nn.Linear(*input_dims, hidden_dims)
        self.out_v = nn.Linear(hidden_dims, 1)
        self.fc_p = nn.Linear(*input_dims, hidden_dims)
        self.out_p = nn.Linear(hidden_dims, n_actions)

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, filename)
        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, state):
        v = F.relu(self.fc_v(state))
        v = self.out_v(v)

        p = F.relu(self.fc_p(state))
        p = self.out_p(p)

        return v, p

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


# Share optimizer parameters among a pool of threads
class SharedOptim(torch.optim.Adam, ABC):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99)):
        super(SharedOptim, self).__init__(params, lr=lr, betas=betas)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Agent(mp.Process):
    def __init__(self, env, input_dims, n_actions, gamma, global_actor_critic, optimizer, worker_id,
                 update_time, n_episodes, chkpt_dir='./tmp/a3c', lr=1e-3, hidden_dims=128,
                 device=DEVICE):
        super(Agent, self).__init__()
        self.env = env
        self.gamma = gamma
        self.global_actor_critic = global_actor_critic
        self.optimizer = optimizer
        self.name = 'w%02i' % worker_id
        self.update_time = update_time
        self.n_episodes = n_episodes
        self.chkpt_dir = chkpt_dir
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.device = device

        self.episode_idx = mp.Value('i', 0)

        self.local_ac = ActorCritic(input_dims, n_actions, hidden_dims, f'a3c{worker_id}_cartepole',
                                    chkpt_dir)
        self.learner_step = 0

        self.states = []
        self.actions = []
        self.rewards = []

    def clear_trajectory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store_trajectory(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # Compute the returns over every step of the trajectory
    def get_return(self, terminal):
        states = torch.tensor(self.states, dtype=torch.float)
        v, _ = self.local_ac.forward(states)

        returns = []
        discounted_sum = v[-1] * (1 - int(terminal))
        for reward in self.rewards[::-1]:
            discounted_sum = reward + self.gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()

        return torch.tensor(returns, dtype=torch.float)

    def get_loss(self, terminal):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.get_return(terminal)
        val, pol = self.local_ac.forward(states)

        values = val.squeeze()
        critic_loss = (returns - values) ** 2

        prob_dist = Categorical(torch.softmax(pol, dim=1))
        actor_loss = -prob_dist.log_prob(actions) * (returns - values)

        return (critic_loss + actor_loss).mean()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float)
        val, pol = self.local_ac.forward(state)
        prob_dist = Categorical(torch.softmax(pol, dim=1))

        return prob_dist.sample().numpy()[0]

    def save_model(self):
        self.local_ac.save_checkpoint()

    def load_model(self):
        self.local_ac.load_checkpoint()

    def learn(self, terminal):
        if self.learner_step % self.update_time == 0 or terminal:
            # Compute the loss and backpropagate it through the network
            self.optimizer.zero_grad()
            loss = self.get_loss(terminal)
            loss.backward()
            for local_param, global_param in \
                    zip(self.local_ac.parameters(), self.global_actor_critic.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()

            # load the global architecture into the local one and start from scratch
            self.local_ac.load_state_dict(self.global_actor_critic.state_dict())
            self.clear_trajectory()

        # Increase the episode counter
        self.learner_step += 1

    def run(self):
        while self.episode_idx.value < self.n_episodes:
            terminal = False
            observation = self.env.reset()[0]
            score = 0
            self.clear_trajectory()
            while not terminal:
                action = self.choose_action(observation)
                observation_, reward, done, truncated, info = self.env.step(action)
                score += reward
                terminal = done or truncated
                self.store_trajectory(observation, action, reward)
                self.learn(terminal)
                observation = observation_

            with self.episode_idx.get_lock():
                self.episode_idx.value += 1

            print(self.name, 'episode ', self.episode_idx.value, 'score %.1f' % score)
