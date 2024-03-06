import itertools
import matplotlib.pyplot as plt


def get_best_moves(agent, env):
    """Print estimated best available moves."""
    policy = []
    x, y = env.shape
    states = list(itertools.product(range(x), range(y)))
    for state in states:
        available_actions = agent.get_max_actions(state)
        policy.append(''.join([env.action_symbol[a] for a in available_actions]))
    n_rows, n_cols = env.shape
    print("Estimated optimal moves in each state:")
    for i in range(0, n_rows*n_cols, n_cols):
        print(policy[i:i+n_cols])


def plot(t_steps):
    """"
    Plot number of steps taken within each episode before goal was reached.
    """
    plt.plot([i for i in range(len(t_steps))], t_steps)
    plt.xlabel("Episode")
    plt.ylabel("Time steps to goal")
    plt.show()


def experiment(env, agent, n_episodes, t_max=100):
    """
    :param env: the environment
    :param agent: the agent
    :param n_episodes: number of episodes to run
    :param t_max: max number of steps to take within an episode, terminate if goal is not reached by this point
    :return: list with number of steps taken until goal was reached in each episode
    """
    t_steps = []
    for i in range(n_episodes):
        terminal = False
        t = 0
        env.reset()
        total_reward = []
        while not terminal and t <= t_max:
            action = agent.choose()
            reward, new_state, terminal = env.step(action)
            total_reward.append(reward)
            agent.update(reward, new_state, terminal)
            t += 1
        t_steps.append(t)
        print(F"TOTAL RWARD:{sum(total_reward)}")
    return t_steps
