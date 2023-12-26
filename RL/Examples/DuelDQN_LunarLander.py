import gym
from RL.Agents.DuelDQN import Agent
from RL.utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    input_dims = [8]
    n_actions = 4
    gamma = 0.99
    epsilon = 1.0
    n_episodes = 50
    load_agent = False

    agent = Agent(env, input_dims, n_actions, gamma, epsilon, n_episodes)
    if load_agent:
        agent.load_models()

    scores, eps_history = agent.train(env, n_episodes)

    filename = './RL/Plots/dueldqn_lunarlander.png'
    plot_learning_curve([i + 1 for i in range(n_episodes)], scores, eps_history, filename)
