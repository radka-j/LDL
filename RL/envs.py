import numpy as np

class Gridworld(object):
    """
    A Gridworld:
        - each state is a (x, y) position tuple
        - each state has 4 possible actions {up, down, left, right}
    The reward is -1 in all states except 0 in the terminal/goal state.

    :param shape: tuple, shape of the grid world (at least 4x4).
    :param start: tuple, starting position of agent.
    :param goal: tuple, location of the terminal/goal state.
    """
    def __init__(self, shape=(4, 4), start=(0, 0), goal=(3, 3)):
        self.shape = shape
        self.goal = np.array(goal)
        self.start = np.array(start)
        self.agent_pos = self.start
        self.actions = {
            0:np.array((0, 1)),
            1:np.array((0, -1)),
            2:np.array((-1, 0)),
            3:np.array((1, 0))
        }
        self.action_symbol = {0:"→", 1:"←", 2:"↑", 3:"↓"}
        self.create_values_array()

    def create_values_array(self):
        self.state_values = np.full(self.shape, -1)
        self.state_values[self.goal[0], self.goal[1]] = 0

    def reset(self):
        # return agent to the start
        self.agent_pos = self.start

    def get_reward(self, state):
        return self.state_values[state[0], state[1]]

    def is_terminal(self, state):
        # indicate whether state is terminal
        return np.array_equal(state, self.goal)

    def step(self, action):
        # get next state and validate whether move is allowed
        # update agent pos or stay
        # return reward, next state, whether state is terminal
        new_state = self.agent_pos + self.actions[action]
        if self.is_allowed(new_state):
            self.agent_pos = new_state
        else:
            new_state = self.agent_pos
        return self.get_reward(new_state), tuple(new_state), self.is_terminal(new_state)

    def is_allowed(self, state):
        # check whether this is a valid state (within Gridworld boundaries)
        x, y = state
        x_max, y_max = self.shape
        return (x >= 0 and x < x_max and y >= 0 and y < y_max)


class WindyGridworld(Gridworld):
    """
    Windy Gridworld example (page 130).

    A 7x10 Gridworld with upward wind through the middle of the grid.
    The wind strength varies by column (1 or 2 in center, 0 at the edges).
    The agent is shifted up by wind strength.

    Termination is not guaranteed for all policies so set limit on episode length.
    """
    def __init__(self):
        super().__init__(shape=(7, 10), start=(3, 0), goal=(3, 7))
        self.create_wind_array()

    def create_wind_array(self):
        # In certain columns, player is shifted up by 1 or 2 steps
        self.wind_array = np.full(self.shape, -1)
        self.wind_array[:, 0:3] = 0
        self.wind_array[:, -1] = 0
        self.wind_array[:, 6:8] = -2

    def get_new_state(self, action):
        # takes into account effect of wind
        # make sure can't pass grid boundaries
        current_x, current_y = self.agent_pos
        proposed_x, proposed_y = self.agent_pos + self.actions[action] + np.array((self.wind_array[current_x, current_y], 0))
        x_max, y_max = self.shape
        if proposed_x < 0:
            x = 0
        elif proposed_x >= x_max:
            x = x_max - 1
        else:
            x = proposed_x
        if proposed_y < 0:
            y = 0
        elif proposed_y >= y_max:
            y = y_max - 1
        else:
            y = proposed_y
        return (x, y)

    def step(self, action):
        # go to next state
        # return reward, next state, whether state is terminal
        new_state = self.get_new_state(action)
        self.agent_pos = new_state
        return self.get_reward(new_state), tuple(new_state), self.is_terminal(new_state)


class Cliff(Gridworld):
    """
    Cliff example (page 132).

    A 4x12 Gridworld which includes states with reward -100 (the cliff).
    The agent is returned to the start state if step off the cliff.
    """
    def __init__(self):
        super().__init__(shape=(4, 12), start=(3, 0), goal=(3, 11))
        self.add_cliff()

    def add_cliff(self):
        # the cliff are states in the grid where reward is -100
        for i in range(1, 11):
            self.state_values[3, i] = -100

    def step(self, action):
        # if go over cliff, return agent to start
        new_state = self.agent_pos + self.actions[action]
        if self.is_allowed(new_state):
            self.agent_pos = new_state
        else:
            new_state = self.agent_pos
        reward = self.get_reward(new_state)
        if reward == -100:
            new_state = self.start
            self.agent_pos = self.start
        return reward, tuple(new_state), self.is_terminal(new_state)
