import numpy as np

from .models import Grid
from ..abstract import Environment


__all__ = [
    'GRID66',
    'GRID69',
    'GRID2436',
    'GRID1010',
    'GRID2525'
]


class GridEnvironment(Environment):
    grid = None
    walls_mark = None
    starting_position = None
    model = Grid
    actions = Grid.actions
    max_steps = 1000

    def __init__(self, max_steps=None, *args, **kwargs):
        max_steps = max_steps or self.max_steps
        super().__init__(max_steps, init_agent_position=self.starting_position,
                         grid=self.grid, walls_mark=self.walls_mark)

    @property
    def reward(self):
        return self.model.grid[self.model.observation]

    @property
    def states(self):
        return list(range(self.grid.size))

    def get_state(self):
        position = self.model.observation
        return position[0] * self.grid.shape[1] + position[1]

    def is_state_absorbing(self):
        return self.reward != 0.0


class GRID66(GridEnvironment):
    grid = np.zeros((6, 6))
    grid[1:5, 1] = -1
    grid[1:5, 4] = -1
    grid[1, 2] = -1
    grid[4, 3] = -1
    grid[0, 5] = 1
    grid[5, 5] = 0.5
    walls_mark = -1
    starting_position = (1, 3)


class GRID69(GridEnvironment):
    grid = np.zeros((6, 9))
    grid[1:4, 2] = -1
    grid[0:3, 7] = -1
    grid[4, 5] = -1
    grid[0, 8] = 1
    walls_mark = -1
    starting_position = (5, 0)


class GRID2436(GridEnvironment):
    grid = np.zeros((24, 36), dtype=int)
    grid[5:17, 8:12] = -1
    grid[17:21, 20:24] = -1
    grid[0:12, 18:32] = -1
    grid[0, 35] = 1
    walls_mark = -1
    starting_position = (1, 3)


class GRID1010(GridEnvironment):
    grid = np.zeros((10, 10))
    grid[0:10, 0] = -1
    grid[0:10, 9] = -1
    grid[0, 0:10] = -1
    grid[9, 0:10] = -1
    starting_position = (5, 5)


class GRID2525(GridEnvironment):
    grid = np.zeros((25, 25))
    grid[0:25, 0] = -1
    grid[0:25, 24] = -1
    grid[0, 0:25] = -1
    grid[24, 0:25] = -1
    starting_position = (12, 12)
