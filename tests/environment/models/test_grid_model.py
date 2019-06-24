import numpy as np
import pytest

from rltoolbox.environment.models import Grid


mini_grid = np.array([[0, 0], [0, 0]])
medium_grid = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])
maxi_grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])


def test_grid_observation():
    grid = Grid(init_agent_position=((3, 2)))
    assert grid.observation == (3, 2)


def test_grid_render_without_fail():
    grid = Grid()
    grid.render()
    grid.close()


def test_grid_reset():
    grid = Grid(
        init_agent_position=(1,1), timestep=0.3,
        grid=medium_grid
    )
    grid.agent_position = (2, 3)
    grid.agent_direction = 'up'
    grid.current_step = 10
    grid.reset()
    assert grid.agent_position == (1, 1)
    assert grid.current_step == 0
    assert grid.timestep == 0.3


@pytest.mark.parametrize('grid,position,move',[
    (np.array([[0, 0], [0, 0]]), (0, 0), 'left'),
    (np.array([[0, 0], [0, 0]]), (1, 0), 'down'),
    (np.array([[0, 0], [0, 0]]), (0, 0), 'up'),
    (np.array([[0, 0], [0, 0]]), (0, 1), 'right'),
])
def test_impossible_moves(grid, position, move):
    grid = Grid(init_agent_position=position, grid=grid)
    new_position = grid.step(move)
    assert new_position == position


@pytest.mark.parametrize('grid,position,moves,expected_observation',[
    (maxi_grid, (2, 3), ['left', 'left', 'right'], (2, 2)),
    (maxi_grid, (1, 1), ['down', 'down', 'right', 'up'], (2, 2))
])
def test_grid_step(grid, position, moves, expected_observation):
    grid = Grid(init_agent_position=position, grid=grid)
    for move in moves:
        observation = grid.step(move)
    assert grid.current_step == len(moves)
    assert grid.agent_direction == moves[-1]
    assert observation == expected_observation
