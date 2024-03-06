import numpy as np
from collections import defaultdict

class Agent(object):
    """
    Scaffold for agents solving gridworld problems.
    Initialize all Q(state, action) values at 0.
    Scaffold is missing algorithm specific methods:
        - choose() for selecting actions
        - update() for updating Q estimates

    :param start: tuple, the starting state of the agent.
    :param epsilon: real, determines level of exploration ( 0 < epsilon <= 1).
    :params alpha: real, the learning rate (0 < alpha <= 1)
    :param gamma: real, discount factor of future rewards (0 < gamma <= 1)
    """
    def __init__(self, start, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = 4
        self.last_action = None
        self.last_state = tuple(start)
        # to get state action values: q_table[state][action]
        self.q_table = defaultdict(lambda: {n:0 for n in range(self.n_actions)})

    def get_action_values(self, state):
        actions = self.q_table[state]
        return np.asarray(list(actions.values()))

    def get_max_actions(self, state):
        # return all actions with max q value
        action_values = self.get_action_values(state)
        action = np.argmax(action_values)
        return np.where(action_values == action_values[action])[0]

    def select_max_action(self, state):
        # select action with max q value, break ties if multiple
        available_actions = self.get_max_actions(state)
        return np.random.choice(available_actions)

    def epsilon_greedy(self, state):
        # choose action according to epsilon-greedy policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = self.select_max_action(self.last_state)
        return action


class Qlearning_Agent(Agent):
    """Q Learning agent (page 131)"""
    def choose(self):
        action = self.epsilon_greedy(self.last_state)
        self.last_action = action
        return action

    def update(self, reward, new_state, terminal):
        current_q = self.q_table[self.last_state][self.last_action]
        if terminal:
            next_q = 0
        else:
            max_action = self.select_max_action(new_state)
            next_q = self.q_table[new_state][max_action]
        new_q = current_q + self.alpha*(reward + self.gamma*next_q - current_q)
        self.q_table[self.last_state][self.last_action] = new_q
        self.last_state = new_state


class Sarsa_Agent(Agent):
    """Sarsa agent (page 130)"""
    def choose(self, state=None):
        if self.last_action == None:
            # first step taken
            action = self.epsilon_greedy(self.last_state)
            self.last_action = action
        else:
            # action was selected during update
            action = self.last_action
        return action

    def update(self, reward, new_state, terminal):
        current_q = self.q_table[self.last_state][self.last_action]
        if terminal:
            next_q = 0
        else:
            next_action = self.epsilon_greedy(new_state)
            next_q = self.q_table[new_state][next_action]
        new_q = current_q + self.alpha*(reward + self.gamma*next_q - current_q)
        self.q_table[self.last_state][self.last_action] = new_q
        self.last_state = new_state
        if not terminal:
            self.last_action = next_action
