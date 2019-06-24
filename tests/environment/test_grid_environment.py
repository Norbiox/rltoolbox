import pytest

from rltoolbox.environment.grid import *


environments = (GRID66, GRID69, GRID2436, GRID1010, GRID2525)
shapes = (env.grid.shape for env in environments)
walls_marks = (env.walls_mark for env in environments)


@pytest.mark.parametrize('env,shape',zip(environments, shapes))
def test_model_grid(env,shape):
    e = env()
    assert e.model.grid.shape == shape


@pytest.mark.parametrize('env,mark',zip(environments, walls_marks))
def test_wall_marks(env,mark):
    e = env()
    assert e.model.walls_mark == mark
