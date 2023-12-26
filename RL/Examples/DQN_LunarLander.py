import gym
from RL.Agents.DQN import Agent
from RL.utils import plot_learning_curve
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_episodes = 50
    load_agent = False
    agent = Agent(input_dims=[8], n_actions=4, gamma=0.99, epsilon=1.0)

    if load_agent:
        agent.load_models()

    scores, eps_history = [], []
    for i in range(n_episodes):
        score = 0
        terminal = False
        observation = env.reset()[0]
        while not terminal:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            terminal = done or truncated
            agent.store_transition(observation, action, reward, observation_, terminal)
            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print(
            'episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon
        )

    x = [i + 1 for i in range(n_episodes)]
    filename = './RL/Plots/dqn_lunarlander.png'
    plot_learning_curve(x, scores, eps_history, filename)
