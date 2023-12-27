import gym
from RL.Agents.DDPG import Agent
from RL.utils import plot_learning_scores

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    gamma = 0.99

    update_factor = 0.001

    n_episodes = 5
    load_agent = False

    agent = Agent(env, input_dims, n_actions, gamma, update_factor, n_episodes)
    if load_agent:
        agent.load_model()

    scores = agent.train()

    filename = './RL/Plots/ddpg_lunarlander.png'
    plot_learning_scores([i + 1 for i in range(n_episodes)], scores, filename)
