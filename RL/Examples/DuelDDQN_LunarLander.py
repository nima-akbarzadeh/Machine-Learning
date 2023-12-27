import gym
from RL.Agents.DuelDDQN import Agent
from RL.utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    gamma = 0.99

    epsilon = 1.0
    n_episodes = 50
    update_time = 0.1 * n_episodes
    load_agent = False

    agent = Agent(env, input_dims, n_actions, gamma, update_time, epsilon, n_episodes)
    if load_agent:
        agent.load_model()

    scores, eps_history = agent.train()

    filename = './RL/Plots/duelddqn_lunarlander.png'
    plot_learning_curve([i + 1 for i in range(n_episodes)], scores, eps_history, filename)
