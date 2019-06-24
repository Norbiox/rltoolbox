import pytest
import numpy as np

from rltoolbox.algorithm import cmac
from rltoolbox.abstract import Environment
from rltoolbox.approximator import CMACApproximator
from rltoolbox.tests.fakes import FakeContinuousEnvironment


def test_CMACAHC_get_greedy_actions():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(CMACApproximator)
    algorithm = cmac.CMACAHC(environment)
    environment_state = (0, 0)
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions)
    for l in range(algorithm.n_layers):
        for i in range(algorithm.mi[l].shape[0]):
            algorithm.mi[l][i][0] = 1
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions[:1])


def test_CMACAHC_0_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(CMACApproximator)
    algorithm = cmac.CMACAHC(environment)
    for l in range(algorithm.n_layers):
        assert sum([algorithm.mi[l][:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    for l in range(algorithm.n_layers):
        assert sum([algorithm.mi[l][:, i].sum() for i in algorithm.actions]) != 0.0


def test_CMACAHC_lambda_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(CMACApproximator)
    algorithm = cmac.CMACAHC(environment, lambd=0.1)
    for l in range(algorithm.n_layers):
        assert sum([algorithm.mi[l][:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    for l in range(algorithm.n_layers):
        assert sum([algorithm.mi[l][:, i].sum() for i in algorithm.actions]) != 0.0


@pytest.mark.parametrize('algorithm,lambd', [
    (cmac.CMACQ, 0.0),
    (cmac.CMACQ, 0.2),
    (cmac.CMACSARSA, 0.0),
    (cmac.CMACSARSA, 0.2),
    (cmac.CMACR, 0.0),
    (cmac.CMACR, 0.2)
])

def test_CMACQ_CMACSARSA_CMACR_get_greedy_actions(algorithm, lambd):
    environment = FakeContinuousEnvironment()
    environment.approximate_with(CMACApproximator)
    algorithm = algorithm(environment, lmbd=lambd)
    environment_state = (0, 0)
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions)
    for l in range(algorithm.n_layers):
        for i in range(algorithm.q[l].shape[0]):
            algorithm.q[l][i][0] = 1
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert np.array_equal(greedy_actions, algorithm.actions[:1])


@pytest.mark.parametrize('algorithm,lambd', [
    (cmac.CMACQ, 0.0),
    (cmac.CMACQ, 0.2),
    (cmac.CMACSARSA, 0.0),
    (cmac.CMACSARSA, 0.2),
    (cmac.CMACR, 0.0),
    (cmac.CMACR, 0.2)
])
def test_CMACQ_CMACSARSA_CMACR_lambda_run_learning_episode(algorithm, lambd):
    environment = FakeContinuousEnvironment()
    environment.approximate_with(CMACApproximator)
    algorithm = algorithm(environment, lambd=lambd)
    for l in range(algorithm.n_layers):
        assert sum([algorithm.q[l][:, i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    for l in range(algorithm.n_layers):
        assert sum([algorithm.q[l][:, i].sum() for i in algorithm.actions]) != 0.0
