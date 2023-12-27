import gym
from RL.Agents.PPO import Agent
from RL.utils import plot_learning_scores

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    gamma = 0.99

    gae_lambda = 0.95
    policy_clip = 0.2
    policy_horizon = 20
    n_epochs = 4

    n_episodes = 50
    load_agent = False

    agent = Agent(env, input_dims, n_actions, gamma, gae_lambda,
                  policy_clip, policy_horizon, n_epochs, n_episodes)

    if load_agent:
        agent.load_model()

    scores = agent.train()

    filename = './RL/Plots/ddpg_lunarlander.png'
    plot_learning_scores([i + 1 for i in range(n_episodes)], scores, filename)

