import gym
from RL.Agents.SAC import Agent
from RL.utils import plot_learning_scores

if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v1')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    gamma = 0.99

    n_episodes = 1000
    update_factor = 0.005
    reward_scale = 2
    reparam_noise = 1e-6   # helps with log of 0
    load_agent = False

    agent = Agent(env, input_dims, n_actions, gamma, update_factor, reward_scale, reparam_noise,
                  n_episodes)
    if load_agent:
        agent.load_model()

    scores = agent.train()

    filename = './RL/Plots/ddpg_lunarlander.png'
    plot_learning_scores([i + 1 for i in range(n_episodes)], scores, filename)
