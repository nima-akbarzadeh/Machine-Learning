import gym
from RL.Agents.TD3 import Agent
from RL.utils import plot_learning_scores

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    gamma = 0.99

    n_episodes = 100
    update_factor = 0.005
    update_actor_time = 2
    warmup = int(0.1 * n_episodes)
    noise = 0.2
    load_agent = False

    agent = Agent(env, input_dims, n_actions, gamma, update_factor, update_actor_time, warmup,
                  noise, n_episodes)
    if load_agent:
        agent.load_model()

    scores = agent.train()

    filename = './RL/Plots/ddpg_lunarlander.png'
    plot_learning_scores([i + 1 for i in range(n_episodes)], scores, filename)
