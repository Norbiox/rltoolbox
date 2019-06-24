import numpy as np
import pytest

from rltoolbox.environment.models import MountainCar


def test_mountain_car_observation():
    mc = MountainCar(init_car_position=0.3, init_car_speed=-0.02)
    assert mc.observation == (0.3, -0.02)


def test_mountain_car_render_without_fail():
    mc = MountainCar()
    mc.render()
    mc.close()


def test_mountain_car_reset():
    mc = MountainCar(init_car_position=0.1, init_car_speed=0.02,
                     init_car_acceleration=0.3, timestep=0.04)
    mc.car_position += 0.1
    mc.car_speed -= 0.02
    mc.car_acceleration -= 0.3
    mc.current_step = 10
    mc.reset()
    assert mc.car_position == 0.1
    assert mc.car_speed == 0.02
    assert mc.car_acceleration == 0.3
    assert mc.current_step == 0
    assert mc.timestep == 0.04


@pytest.mark.parametrize('init_params,expected_observation_less_and_greater',[
    ([0.2, 0.0, 0.0], [-1.21, 0.2, -0.071, 0.0]),
    ([0.2, 0.07, 0.0], [0.2, 0.51, -0.071, 0.07]),
    ([0.1, 0.06, 1.0], [0.1, 0.51, 0.05, 0.071]),
    ([-1.0, 0.0, 0.0], [-1.0, 0.51, 0.0, 0.071]),
])
def test_mountain_car_step(init_params, expected_observation_less_and_greater):
    exp = expected_observation_less_and_greater
    mc = MountainCar(*init_params)
    observation = mc.step()
    observation = mc.step()
    assert mc.current_step == 2
    assert mc.car_acceleration == init_params[2]
    assert exp[0] < observation[0] < exp[1]
    assert exp[2] < observation[1] < exp[3]
