import numpy as np
from matplotlib import pyplot as plt

from ..abstract import Model


g = 9.81


class Grid(Model):
    grid = np.zeros((10, 10))
    actions = ['up', 'right', 'down', 'left']
    walls_mark = None

    def __init__(self, init_agent_position=(0, 0), grid=None, walls_mark=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_agent_position = init_agent_position
        self.grid = grid if grid is not None else self.grid
        self.walls_mark = walls_mark or self.walls_mark
        self.reset()

    @property
    def agent_position(self):
        return self._agent_position

    @agent_position.setter
    def agent_position(self, position):
        if self.is_move_possible(position):
            self._agent_position = position
        else:
            raise ValueError(f"agent cannot be placed in position {position}")

    @property
    def observation(self):
        return self.agent_position

    def is_move_possible(self, new_position):
        borders_hit = any([
            new_position[0] < 0,
            new_position[0] >= self.grid.shape[0],
            new_position[1] < 0,
            new_position[1] >= self.grid.shape[1]
        ])
        if borders_hit:
            return False
        if self.walls_mark is not None and self.grid[new_position] == self.walls_mark:
            return False
        return True

    def render(self):
        string = '\n' * 100
        position_value = self.grid[self.agent_position]
        self.grid[self.agent_position] = np.nan
        string += str(self.grid)
        self.grid[self.agent_position] = position_value
        print(string)

    def reset(self):
        self.agent_position = self.init_agent_position
        self.agent_direction = 'up'
        self.current_step = 0

    def step(self, control):
        assert control in self.actions, \
            f"impossible move {control}, possible moves: {self.actions}"
        self.agent_direction = control
        current_position = list(self.agent_position)
        if control == "up":
            current_position[0] -= 1
        elif control == "right":
            current_position[1] += 1
        elif control == "down":
            current_position[0] += 1
        elif control == "left":
            current_position[1] -= 1
        try:
            self.agent_position = tuple(current_position)
        except ValueError:
            pass
        self.current_step += 1
        return self.observation


class BallBeam(Model):
    """BallBeam - model of ball balancing on beam"""
    beam_length = 2

    def __init__(self, init_ball_position=0, init_ball_speed=0,
                 init_beam_theta=np.pi / 8, timestep=0.02,
                 *args, **kwargs):
        super().__init__(timestep, *args, **kwargs)
        self.init_ball_position = init_ball_position
        self.init_ball_speed = init_ball_speed
        self.init_beam_theta = init_beam_theta
        self.reset()

    @property
    def observation(self):
        return (self.ball_position, self.ball_speed)

    def render(self):
        beam_x = [
            - self.beam_length / 2 * np.cos(self.beam_theta),
            self.beam_length / 2 * np.cos(self.beam_theta)
        ]
        beam_y = [
            self.beam_length / 2 * np.sin(self.beam_theta),
            - self.beam_length / 2 * np.sin(self.beam_theta)
        ]
        ball_r = 0.01
        ball_xy = (
            self.ball_position * np.cos(self.beam_theta),
            self.ball_position * (- np.sin(self.beam_theta)) + ball_r
        )
        if not self.viewer:
            self.viewer = plt.figure()
            self.viewer.ax = self.viewer.add_subplot(111)
            self.viewer.ax.axis('off')
            plt.xlim((- self.beam_length / 2, self.beam_length / 2))
            plt.ylim((- self.beam_length / 2, self.beam_length / 2))
            self.viewer.ax.axis('equal')
            self.viewer.beam_line, = self.viewer.ax.plot(beam_x, beam_y,
                                                         linewidth=3)
            self.viewer.ball = plt.Circle(ball_xy, ball_r, color='r')
            self.viewer.ax.add_artist(self.viewer.ball)
            self.viewer.show()
        self.viewer.ball.center = ball_xy
        self.viewer.beam_line.set_xdata(beam_x)
        self.viewer.beam_line.set_ydata(beam_y)
        self.viewer.canvas.draw()
        self.viewer.canvas.flush_events()

    def reset(self):
        self.beam_theta = self.init_beam_theta
        self.ball_position = self.init_ball_position
        self.ball_speed = self.init_ball_speed
        self.current_step = 0

    def step(self, control=None):
        if control is not None:
            self.beam_theta = control
        self.ball_position += self.timestep * self.ball_speed
        self.ball_speed += self.timestep * g * np.sin(self.beam_theta)
        self.current_step += 1
        return self.observation


class MountainCar(Model):
    """MountainCar - model of car climbing a hill (1990 Moore)"""
    hill_range = np.arange(-1.2, 0.6, 0.01)
    hill_line = np.diff(- 0.0025 * np.cos(3 * hill_range)) * 10000
    flag_r = 0.02
    flag_xy = (0.5, np.interp(0.5, hill_range[1::], hill_line))

    def __init__(self, init_car_position=-0.5, init_car_speed=0.0,
                 init_car_acceleration=0.0, timestep=0.01, *args, **kwargs):
        super().__init__(timestep, *args, **kwargs)
        self.init_car_position = init_car_position
        self.init_car_speed = init_car_speed
        self.init_car_acceleration = init_car_acceleration
        self.reset()

    @property
    def car_position(self):
        return self._car_position

    @car_position.setter
    def car_position(self, value):
        if value < -1.2:
            value = -1.2
            self.car_speed = 0
        elif value > 0.5:
            value = 0.5
            self.car_speed = 0
        self._car_position = value

    @property
    def car_speed(self):
        return self._car_speed

    @car_speed.setter
    def car_speed(self, value):
        if value < -0.07:
            value = -0.07
        elif value > 0.07:
            value = 0.07
        self._car_speed = value

    @property
    def observation(self):
        return (self.car_position, self.car_speed)

    def render(self):
        car_r = 0.02
        car_xy = (
            self.car_position,
            np.interp(
                self.car_position, self.hill_range[1::], self.hill_line
            ) + car_r
        )
        if not self.viewer:
            self.viewer = plt.figure()
            self.viewer.ax = self.viewer.add_subplot(111)
            self.viewer.ax.axis('off')
            self.viewer.ax.axis('equal')
            self.viewer.hill_line, = self.viewer.ax.plot(
                self.hill_range[1::], self.hill_line, linewidth=3
            )
            self.viewer.flag = plt.Circle(self.flag_xy, self.flag_r, color='g')
            self.viewer.ax.add_artist(self.viewer.flag)
            self.viewer.car = plt.Circle(car_xy, car_r, color='r')
            self.viewer.ax.add_artist(self.viewer.car)
            self.viewer.show()
        self.viewer.car.center = car_xy
        self.viewer.canvas.draw()
        self.viewer.canvas.flush_events()

    def reset(self):
        self.car_acceleration = self.init_car_acceleration
        self.car_position = self.init_car_position
        self.car_speed = self.init_car_speed
        self.current_step = 0

    def step(self, control=None):
        if control is not None:
            self.car_acceleration = control
        self.car_speed += 0.001 * self.car_acceleration - 0.0025 * \
            np.cos(3 * self.car_position)
        self.car_position += self.car_speed
        self.current_step += 1
        return self.observation


class CartPole(Model):
    mc = 1.0
    m = 0.1
    mpc = mc + m
    l = 0.5
    track_length = 2.4
    # for render function
    cart_size = (0.2, 0.08)
    bound_width = 0.2

    def __init__(self, init_cart_position=0.0, init_cart_speed=0.0,
                 init_pole_angle=0.0, init_pole_speed=0.0,
                 init_force=10.0, timestep=0.02, *args, **kwargs):
        super().__init__(timestep, *args, **kwargs)
        self.init_cart_position = init_cart_position
        self.init_cart_speed = init_cart_speed
        self.init_pole_angle = init_pole_angle
        self.init_pole_speed = init_pole_speed
        self.init_force = init_force
        self.reset()

    @property
    def cart_position(self):
        return self._cart_position

    @cart_position.setter
    def cart_position(self, value):
        if value <= -self.track_length / 2:
            value = -self.track_length / 2
            self.cart_speed = 0
        elif value >= self.track_length / 2:
            value = self.track_length / 2
            self.cart_speed = 0
        self._cart_position = value

    @property
    def observation(self):
        return (
            self.cart_position,
            self.cart_speed,
            self.pole_angle,
            self.pole_speed
        )

    def render(self):
        pole_x = [self.cart_position,
                  self.cart_position + self.l * 2 * np.sin(self.pole_angle)]
        pole_y = [0 + self.cart_size[1],
                  self.l * 2 * np.cos(self.pole_angle) + self.cart_size[1]]
        if not self.viewer:
            self.viewer = plt.figure()
            self.viewer.ax = self.viewer.add_subplot(111)
            self.viewer.ax.axis('off')
            self.viewer.ax.axis('equal')
            self.viewer.track_line, = self.viewer.ax.plot(
                [-self.track_length / 2 - self.cart_size[0] / 2 - self.bound_width,
                 self.track_length / 2 + self.cart_size[0] / 2 + self.bound_width],
                [0, 0],
                linewidth=3, color='black'
            )
            self.track_center_mark, = self.viewer.ax.plot(
                [0, 0], [-0.05, 0.05], color='green'
            )
            self.viewer.track_left_bound = plt.Rectangle(
                xy=(-self.track_length / 2 - self.cart_size[0] / 2 - self.bound_width, 0),
                width=self.bound_width, height=self.cart_size[1],
                color='r'
            )
            self.viewer.track_right_bound = plt.Rectangle(
                xy=(self.track_length / 2 + self.cart_size[0] / 2, 0),
                width=self.bound_width, height=self.cart_size[1],
                color='r'
            )
            self.viewer.cart = plt.Rectangle(
                xy=(self.cart_position - self.cart_size[0] / 2, 0),
                width=self.cart_size[0], height=self.cart_size[1],
                color='blue'
            )
            self.viewer.pole, = self.viewer.ax.plot(
                pole_x, pole_y, linewidth=2, color='red'
            )
            self.viewer.ax.add_artist(self.viewer.track_left_bound)
            self.viewer.ax.add_artist(self.viewer.track_right_bound)
            self.viewer.ax.add_artist(self.viewer.cart)
            self.viewer.show()
        self.viewer.cart.set_x(self.cart_position - self.cart_size[0] / 2)
        self.viewer.pole.set_xdata(pole_x)
        self.viewer.pole.set_ydata(pole_y)
        self.viewer.canvas.draw()
        self.viewer.canvas.flush_events()

    def reset(self):
        self.cart_position = self.init_cart_position
        self.cart_speed = self.init_cart_speed
        self.pole_angle = self.init_pole_angle
        self.pole_speed = self.init_pole_speed
        self.force = self.init_force
        self.current_step = 0

    def step(self, control=None):
        if control is not None:
            self.force = control
        theta = [self.pole_angle, self.pole_speed, 0]
        x = [self.cart_position, self.cart_speed, 0]

        theta2nominator = g * np.sin(theta[0]) + np.cos(theta[0]) * \
            (-self.force - self.m * self.l * theta[1] ** 2 * np.sin(theta[0])) / self.mpc
        theta2denominator = self.l * \
            (4 / 3 - (self.m * np.cos(theta[0]) ** 2) / self.mpc)
        theta[2] = theta2nominator / theta2denominator

        x[2] = (self.force + self.m * self.l * (
            theta[1] ** 2 * np.sin(theta[0]) - theta[2] * np.cos(theta[0])
        )) / self.mpc

        self.pole_angle += theta[1] * self.timestep
        self.pole_speed += theta[2] * self.timestep
        self.cart_position += x[1] * self.timestep
        self.cart_speed += x[2] * self.timestep
        self.current_step += 1
        return self.observation
