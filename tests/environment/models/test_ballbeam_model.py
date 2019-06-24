import numpy as np
import pytest

from rltoolbox.environment.models import BallBeam


def test_ball_beam_observation():
    bb = BallBeam(init_ball_position=0.3, init_ball_speed=-0.2)
    assert bb.observation == (0.3, -0.2)


def test_ball_beam_render_without_fail():
    bb = BallBeam()
    bb.render()
    bb.close()


def test_ball_beam_reset():
    bb = BallBeam(init_ball_position=0.2, init_ball_speed=-0.1,
                  init_beam_theta=np.pi / 4, timestep=0.05)
    bb.ball_position += 0.2
    bb.ball_speed -= 0.1
    bb.beam_theta += 1
    bb.current_step = 10
    bb.reset()
    assert bb.ball_position == 0.2
    assert bb.ball_speed == -0.1
    assert bb.beam_theta == np.pi / 4
    assert bb.current_step == 0
    assert bb.timestep == 0.05


@pytest.mark.parametrize('init_params,expected_observation_less_and_greater',[
    ([0.0, 0.0, np.pi / 8, 0.01], [0.0, np.inf, 0.0, np.inf]),
    ([0.0, -0.2, np.pi / 8, 0.01], [-np.inf, 0.0, -0.2, 0.0]),
    ([0.0, 0.0, -np.pi / 8, 0.01], [-np.inf, 0.0, -np.inf, 0.0]),
    ([0.0, 0.2, -np.pi / 8, 0.01], [0.0, np.inf, 0.0, 0.2])
])
def test_ball_beam_step(init_params, expected_observation_less_and_greater):
    exp = expected_observation_less_and_greater
    bb = BallBeam(*init_params)
    observation = bb.step()
    observation = bb.step()
    assert bb.current_step == 2
    assert bb.beam_theta == init_params[2]
    assert exp[0] < observation[0] < exp[1]
    assert exp[2] < observation[1] < exp[3]
