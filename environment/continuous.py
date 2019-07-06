from numpy import pi, radians

from . import models
from ..abstract import Environment


__all__ = ['BallBeam', 'MountainCar', 'CartPole']


class BallBeam(Environment):
    model = models.BallBeam
    actions = [-pi / 4, -pi / 8, pi / 8, pi / 4]
    state_variables = ['ball_position', 'ball_speed']
    state_variables_ranges = [
        [-0.2, 0.2],
        [-0.2, 0.2]
    ]
    max_steps = 100000

    @property
    def reward(self):
        if self.is_state_absorbing():
            return -1.0
        return 0.0

    def is_state_absorbing(self):
        return abs(self.model.ball_position) >= self.model.beam_length / 2


class MountainCar(Environment):
    model = models.MountainCar
    actions = [-1.0, 0.0, 1.0]
    state_variables = ['car_position', 'car_speed']
    state_variables_ranges = [
        [-0.86, -0.52, -0.18, 0.16],
        [-0.042, -0.014, 0.014, 0.042]
    ]
    max_steps = 100000

    @property
    def reward(self):
        if self.is_state_absorbing():
            return 1.0
        return 0.0

    def is_state_absorbing(self):
        return self.model.car_position == 0.5


class CartPole(Environment):
    model = models.CartPole
    actions = [-10.0, 10.0]
    state_variables = ['cart_position', 'cart_speed', 'pole_angle', 'pole_speed']
    state_variables_ranges = [
        [-1.44, -0.48, 0.48, 1.44],
        [-2/3, 2/3],
        [radians(rang) for rang in [-7.2, -2.4, 2.4, 7.2]],
        [radians(rang) for rang in [-400/3, 400/3]]
    ]
    max_steps = 100000

    @property
    def reward(self):
        if self.is_state_absorbing():
            return -1.0
        return 0.0

    def is_state_absorbing(self):
        return abs(self.model.pole_angle) > (12.0 / 180 * pi)
