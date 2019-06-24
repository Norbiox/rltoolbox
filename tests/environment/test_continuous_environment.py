import pytest
import numpy as np

from rltoolbox.environment.continuous import *


@pytest.mark.parametrize('ball_position,absorbing,reward',[
    (-0.7, True, -1.0),
    (-0.5, True, -1.0),
    (-0.3, False, 0.0),
    (0.0, False, 0.0),
    (0.3, False, 0.0),
    (0.5, True, -1.0),
    (0.7, True, -1.0)
])
def test_ball_beam_reward_and_state_absorbing(ball_position,
                                              absorbing,
                                              reward):
    env = BallBeam()
    env.model.ball_position = ball_position
    assert env.is_state_absorbing() == absorbing
    assert env.reward == reward


@pytest.mark.parametrize('car_position,absorbing,reward',[
    (-1.3, False, 0.0),
    (-1.2, False, 0.0),
    (-0.5, False, 0.0),
    (0.4, False, 0.0),
    (0.5, True, 1.0)
])
def test_mountain_car_reward_and_state_absorbing(car_position,
                                                 absorbing,
                                                 reward):
    env = MountainCar()
    env.model.car_position = car_position
    assert env.is_state_absorbing() == absorbing
    assert env.reward == reward


@pytest.mark.parametrize('pole_angle,absorbing,reward',[
    (np.radians(-13.0), True, -1.0),
    (np.radians(-12.0), True, -1.0),
    (np.radians(-8.0), False, 0.0),
    (np.radians(6.0), False, 0.0),
    (np.radians(12.0), True, -1.0),
    (np.radians(14.0), True, -1.0)
])
def test_art_pole_reward_and_state_absorbing(pole_angle,
                                             absorbing,
                                             reward):
    env = CartPole()
    env.model.pole_angle = pole_angle
    assert env.is_state_absorbing() == absorbing
    assert env.reward == reward


@pytest.mark.parametrize('env,action_index',[
    (BallBeam, 0),
    (MountainCar, 0),
    (CartPole, 0)
])
def test_non_approximated_environments_doing_action(env, action_index):
    e = env()
    state = e.state
    next_state = e.do_action(action_index)
    assert next_state == e.state
    assert next_state != state
