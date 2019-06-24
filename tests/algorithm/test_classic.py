import pytest
import numpy as np

from rltoolbox.abstract import Environment
from rltoolbox.algorithm import classic
from rltoolbox.approximator import TableApproximator
from rltoolbox.tests.fakes import (
    FakeContinuousEnvironment,
    FakeGridNoWallsEnvironment
)


def test_AHC_continuous_environment_get_greedy_actions():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(TableApproximator)
    algorithm = classic.AHC(environment)
    environment_state = 2
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions)
    for i in range(algorithm.mi.shape[0]):
        algorithm.mi[i][0] = 1
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions[:1])

def test_AHC_grid_environment_get_greedy_actions():
    environment = FakeGridNoWallsEnvironment()
    algorithm = classic.AHC(environment)
    environment_state = 2
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions)
    for i in range(algorithm.mi.shape[0]):
        algorithm.mi[i][0] = 1
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions[:1])

def test_AHC_0_grid_environment_run_learning_episode():
    environment = FakeGridNoWallsEnvironment()
    algorithm = classic.AHC(environment)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) != 0.0

def test_AHC_0_continuous_environment_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(TableApproximator)
    algorithm = classic.AHC(environment)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) != 0.0

def test_AHC_lambda_grid_environment_run_learning_episode():
    environment = FakeGridNoWallsEnvironment()
    algorithm = classic.AHC(environment, lambd=0.2)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) != 0.0

def test_AHC_lambda_continuous_environment_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(TableApproximator)
    algorithm = classic.AHC(environment, lambd=0.2)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.mi[:, i].sum() for i in algorithm.actions]) != 0.0


@pytest.mark.parametrize('algorithm', [
    classic.Q,
    classic.SARSA,
    classic.R
])
def test_Q_SARSA_R_continuous_environment_get_greedy_actions(algorithm):
    environment = FakeContinuousEnvironment()
    environment.approximate_with(TableApproximator)
    algorithm = algorithm(environment)
    environment_state = 2
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions)
    for i in range(algorithm.Q.shape[0]):
        algorithm.Q[i][0] = 1
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions[:1])


@pytest.mark.parametrize('algorithm', [
    classic.Q,
    classic.SARSA,
    classic.R
])
def test_Q_SARSA_R_grid_environment_get_greedy_actions(algorithm):
    environment = FakeGridNoWallsEnvironment()
    algorithm = algorithm(environment)
    environment_state = 2
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions)
    for i in range(algorithm.Q.shape[0]):
        algorithm.Q[i][0] = 1
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions[:1])


@pytest.mark.parametrize('algorithm,lambd', [
    (classic.Q, 0.0),
    (classic.Q, 0.2),
    (classic.SARSA, 0.0),
    (classic.SARSA, 0.2),
    (classic.R, 0.0),
    (classic.R, 0.2)
])
def test_Q_SARSA_R_continuous_environment_run_learning_episode(algorithm, lambd):
    environment = FakeContinuousEnvironment()
    environment.approximate_with(TableApproximator)
    algorithm = algorithm(environment, lambd=lambd)
    assert sum([algorithm.Q[:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.Q[:, i].sum() for i in algorithm.actions]) != 0.0


@pytest.mark.parametrize('algorithm,lambd', [
    (classic.Q, 0.0),
    (classic.Q, 0.2),
    (classic.SARSA, 0.0),
    (classic.SARSA, 0.2),
    (classic.R, 0.0),
    (classic.R, 0.2)
])
def test_Q_SARSA_R_grid_environment_run_learning_episode(algorithm, lambd):
    environment = FakeGridNoWallsEnvironment()
    algorithm = algorithm(environment, lambd=lambd)
    assert sum([algorithm.Q[:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.Q[:, i].sum() for i in algorithm.actions]) != 0.0
