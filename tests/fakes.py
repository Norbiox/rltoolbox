import numpy as np

from rltoolbox.abstract import Algorithm, Approximator, Environment, Model
from rltoolbox.algorithm.abstract import (
    ClassicAlgorithm,
    CMACAlgorithm,
    FuzzyAlgorithm
)
from rltoolbox.environment.grid import GridEnvironment


class FakeModel(Model):

    def __init__(self, var1=0.0, var2=1.0, var3=2.0, timestep=0.1,
                 *args, **kwargs):
        super().__init__(timestep, *args, **kwargs)
        self.init_var1 = var1
        self.init_var2 = var2
        self.init_var3 = var3
        self.reset()

    @property
    def observation(self):
        return (self.var1, self.var2)

    def render(self):
        print("FakeModel rendering")

    def reset(self):
        self.var1 = self.init_var1
        self.var2 = self.init_var2
        self.var3 = self.init_var3
        self.current_step = 0

    def step(self, control=None):
        if control is not None:
            self.var3 = control
        self.var1 += self.var2
        self.var2 += self.var3
        return self.observation


class FakeContinuousEnvironment(Environment):
    model = FakeModel
    actions = [-1.0, 0.0, 1.0]
    max_steps = 100
    state_variables_ranges = [
        [-1.0, 0.0, 1.0],
        [-1.0, 1.0]
    ]
    
    @property
    def reward(self):
        return 1.0

    def is_state_absorbing(self):
        return False


class FakeGridNoWallsEnvironment(GridEnvironment):
    grid = np.zeros((5, 5))
    grid[0:5, 0] = -1
    grid[0:5, 4] = -1
    grid[0, 0:5] = -1
    grid[4, 0:5] = -1
    starting_position = (2, 2)


class FakeGridWithWallsEnvironment(GridEnvironment):
    grid = np.zeros((4, 4))
    grid[1, 2] = -1
    grid[2, 3] = -1
    grid[0, 3] = 1
    walls_mark = -1
    starting_position = (0, 0)


class FakeAlgorithm(Algorithm):

    def get_greedy_actions(self):
        return [self.actions[0]]

    def run_learning_episode(self, render=False):
        pass


class FakeClassicalAlgorithm(ClassicAlgorithm):

    def get_greedy_actions(self):
        return [self.actions[0]]

    def run_learning_episode(self, render=False):
        pass


class FakeCMACAlgorithm(CMACAlgorithm):

    def get_greedy_actions(self):
        return [self.actions[0]]

    def run_learning_episode(self, render=False):
        pass


class FakeFuzzyAlgorithm(FuzzyAlgorithm):

    def get_greedy_actions(self):
        return [self.actions[0]]

    def run_learning_episode(self, render=False):
        pass


class FakeApproximator(Approximator):

    @property
    def possible_states(self):
        return self.n_state_variables

    def approximate_state(self, observation):
        return 0 if sum(observation) < 0 else 1
