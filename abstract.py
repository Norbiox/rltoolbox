from abc import ABC, abstractmethod, abstractproperty
from matplotlib import pyplot as plt
from numpy import inf
from random import choice, random


class Algorithm(ABC):
    """Algorithm - abstraction for reinforced learning algorithms."""

    def __init__(self, environment, lambd: float, epsilon: float, gamma: float,
                 alpha: float, *args, **kwargs):
        self.environment = environment
        self.actions = list(range(len(self.environment.actions)))
        self.lambd = lambd
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.steps_per_episode = []

    @abstractmethod
    def get_greedy_actions(self, environment_state) -> list:
        pass

    @abstractmethod
    def run_learning_episode(self, render=False):
        pass

    @property
    def episodes(self):
        return len(self.steps_per_episode)

    @property
    def name(self) -> str:
        if self.lambd > 0.0:
            return self.__class__.__name__ + '(lambda)'
        return self.__class__.__name__ + '(0)'

    def get_action(self, epsilon_greedy=True) -> int:
        if epsilon_greedy and random() < self.epsilon:
            return choice(self.actions)
        else:
            return choice(self.get_greedy_actions())

    def learn(self, n_episodes=1, stop_when_learned=False, spe_lte=0,
              spe_gte=inf, wsize=1, print_status=True, render=False):
        for i in range(n_episodes):
            self.environment.clear()
            if print_status:
                print(f"environment: {self.environment.__class__.__name__}\n" +
                      f"algorithm:   {self.name}\n" +
                      f"episode:     {self.episodes + 1}\n")
            self.run_learning_episode(render=render)
            self.steps_per_episode.append(len(self.environment.steps))
            if stop_when_learned and self.is_learned(spe_lte, spe_gte, wsize):
                break
        return self.steps_per_episode, self.environment

    def is_learned(self, steps_per_episode_lte=0, steps_per_episode_gte=inf,
                   window_size=1):
        if len(self.steps_per_episode) < window_size:
            return False
        return all([
            steps_per_episode_gte <= n_steps or n_steps <= steps_per_episode_lte
            for n_steps in self.steps_per_episode[-window_size:]
        ])


class Approximator(ABC):
    """Approximator - state approximator for environments with continuous
        state variables."""

    def __init__(self, n_state_variables: int, state_variables_ranges: list,
                 *args, **kwargs):
        self.n_state_variables = n_state_variables
        self.state_variables_ranges = state_variables_ranges

    @abstractproperty
    def possible_states(self):
        pass

    @abstractmethod
    def approximate_state(self, observation: tuple):
        pass

    @property
    def n_state_variables(self):
        return self._n_state_variables

    @n_state_variables.setter
    def n_state_variables(self, value: int):
        if value <= 0:
            raise ValueError("cannot approximate when there's no variables" +
                             " to approximate")
        self._n_state_variables = value

    @property
    def state_variables_ranges(self):
        return self._state_variables_ranges

    @state_variables_ranges.setter
    def state_variables_ranges(self, value: list):
        if len(value) != self.n_state_variables or \
                any(map(lambda r: not isinstance(r, list), value)):
            raise ValueError("You must specify ranges list for each of state" +
                             " variables. Use empty list, if you don't want" +
                             " to take into account specific state variable")
        self._state_variables_ranges = value


class Model(ABC):
    """Model - abstraction of object (or set of objects) being a base of
        environment for training AI algorithms."""

    def __init__(self, timestep=0.01, *args, **kwargs):
        self.timestep = timestep
        self.viewer = None

    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None

    @abstractproperty
    def observation(self) -> tuple:
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, control: float or None) -> tuple:
        pass


class Environment(ABC):
    """Environment - abstraction of ready-to-use learning environment"""
    model = None
    state_variables_ranges = []
    max_steps = 100000

    def __init__(self, max_steps=None, state_variables_ranges=None,
                 *args, **kwargs):
        self.approximator = None
        self.model = self.model(*args, **kwargs)
        self.max_steps = max_steps or self.max_steps
        self.state_variables_ranges = state_variables_ranges or \
            self.state_variables_ranges
        self.clear()

    @abstractproperty
    def actions(self) -> list:
        pass

    @abstractproperty
    def reward(self) -> float:
        pass

    @abstractmethod
    def is_state_absorbing(self) -> bool:
        pass

    @property
    def done(self) -> bool:
        return self.is_state_absorbing() or len(self.steps) >= self.max_steps

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def states(self):
        if self.approximator is not None:
            return self.approximator.possible_states
        raise AttributeError("It's impossible to specify list of possible" +
                             " states for continuous non-approximated" +
                             " environment")

    def approximate_with(self, approximator, *args, **kwargs):
        self.approximator = approximator(
            len(self.model.observation),
            self.state_variables_ranges,
            *args, **kwargs
        )
        self.state = self.get_state()
        return self

    def clear(self):
        self.model.reset()
        self.state = self.get_state()
        self.steps = []

    def close(self):
        self.model.close()

    def do_action(self, action_index):
        action = self.actions[action_index]
        self.model.step(action)
        self.state = self.get_state()
        self.steps.append(self.state)
        return self.state

    def get_state(self):
        if self.approximator is not None:
            return self.approximator.approximate_state(self.model.observation)
        return self.model.observation

    def render(self):
        self.model.render()
