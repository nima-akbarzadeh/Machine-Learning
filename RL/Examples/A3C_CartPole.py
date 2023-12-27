import gym
import torch.multiprocessing as mp
from RL.Agents.A3C import ActorCritic, SharedOptim, Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_episodes = 1000
    input_dims = [4]
    n_actions = 2
    gamma = 0.99
    lr = 1e-4
    hidden_dims = 128
    update_time = 5
    load_agent = False

    global_actor_critic = ActorCritic(input_dims, n_actions, hidden_dims, f'a3c_cartepole',
                                      chkpt_dir='./tmp/a3c')
    global_actor_critic.share_memory()
    optimizer = SharedOptim(global_actor_critic.parameters(), lr, betas=(0.92, 0.999))

    workers = [Agent(env, input_dims, n_actions, gamma, global_actor_critic, optimizer, i,
                     update_time, n_episodes, f'tmp/a3c_{i}')
               for i in range(mp.cpu_count())]

    [w.start() for w in workers]
    [w.join() for w in workers]


    # env = gym.make('LunarLander-v2')
    # n_episodes = 50
    # load_agent = False
    # agent = Agent(input_dims=[8], n_actions=4, gamma=0.99, epsilon=1.0)
    #
    # if load_agent:
    #     agent.load_models()
    #
    # scores, eps_history = agent.train(env, n_episodes)
    #
    # filename = './RL/Plots/dqn_lunarlander.png'
    # plot_learning_curve([i + 1 for i in range(n_episodes)], scores, eps_history, filename)
