import numpy as np
import pytest

from rltoolbox.environment.models import CartPole


inf = np.inf


def test_cart_pole_observation():
    cp = CartPole(init_cart_position=0.1, init_cart_speed=0.2,
                  init_pole_angle=0.3, init_pole_speed=0.4)
    assert cp.observation == (0.1, 0.2, 0.3, 0.4)


def test_cart_pole_render_without_fail():
    cp = CartPole()
    cp.render()
    cp.close()


def test_cart_pole_reset():
    cp = CartPole(init_cart_position=0.1, init_cart_speed=0.2,
                  init_pole_angle=0.3, init_pole_speed=0.4, timestep=0.1)
    cp.cart_position += 0.1
    cp.cart_speed -= 0.02
    cp.pole_angle += 0.3
    cp.pole_speed -= 0.4
    cp.current_step = 10
    cp.reset()
    assert cp.cart_position == 0.1
    assert cp.cart_speed == 0.2
    assert cp.pole_angle == 0.3
    assert cp.pole_speed == 0.4
    assert cp.current_step == 0
    assert cp.timestep == 0.1


@pytest.mark.parametrize('init_params,expected_observation_less_and_greater',[
    ([0.0, 0.0, 0.0, 0.0, 10.0], [0.0, inf, 0.0, inf, -inf, 0.0, -inf, 0.0]),
    ([0.0, 0.0, 0.0, 0.0, -10.0], [-inf, 0.0, -inf, 0.0, 0.0, inf, 0.0, inf])
])
def test_cart_pole_step(init_params, expected_observation_less_and_greater):
    exp = expected_observation_less_and_greater
    cp = CartPole(*init_params)
    observation = cp.step()
    observation = cp.step()
    assert cp.current_step == 2
    assert cp.force == init_params[4]
    assert exp[0] < observation[0] < exp[1]
    assert exp[2] < observation[1] < exp[3]
    assert exp[4] < observation[2] < exp[5]
    assert exp[6] < observation[3] < exp[7]
