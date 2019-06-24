import pytest
import numpy as np

from rltoolbox.abstract import Environment
from rltoolbox.algorithm import fuzzy
from rltoolbox.approximator import FuzzyApproximator
from rltoolbox.tests.fakes import FakeContinuousEnvironment
from rltoolbox.fuzzy import (
    FuzzySet,
    TrapezoidalMembershipFunction,
    TriangularMembershipFunction
)


fuzzy_sets=[
    FuzzySet.from_membership_functions_ranges([
        (-np.inf, -np.inf, -1.0, 0.0),
        (-1.0, 0.0, 1.0),
        (0.0, 1.0, np.inf, np.inf)
    ]),
    FuzzySet.from_membership_functions_ranges([
        (-np.inf, -np.inf, 0.0, 0.0),
        (0.0, 0.0, np.inf, np.inf)
    ])
]


@pytest.mark.parametrize('state,expected_phi', [
    (
        [np.array([1, 0])],
        np.array([1, 0])
    ),
    (
        [np.array([1, 0]), np.array([0, 1])],
        np.array([[0, 1], [0, 0]])
    ),
    (
        [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.5, 0.5])],
        np.ones((4, 3, 2)) / 8
    )
])
def test_phi(state, expected_phi):
    phi = fuzzy.phi(state)
    assert np.array_equal(phi, expected_phi)


def test_FQ_get_greedy_actions_invalid_environment_state_given():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FQ(environment)
    with pytest.raises(ValueError):
        algorithm.get_greedy_actions(np.ones((2,2)))
        
        
def test_FQ_get_greedy_actions():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FQ(environment)
    # environment state approximed for variables = (-0.5, -0.5)
    environment_state = [np.array([0.5, 0.5, 0.0]), np.array([1.0, 0.0])]
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    # all of possible actions because all q ale zeros
    assert list(greedy_actions) == algorithm.actions
    # now second actions q is modified to be greedy
    algorithm.q[1][1][0] = 1.0
    greedy_actions = algorithm.get_greedy_actions(environment_state)
    assert greedy_actions == algorithm.actions[1]


def test_FQ_0_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FQ(environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) != 0.0


def test_FQ_lambda_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FQ(environment, lambd=0.1)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) != 0.0


def test_FSARSA_0_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FSARSA(environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) != 0.0


def test_FSARSA_lambda_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FSARSA(environment, lambd=0.1)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) != 0.0

     
def test_FR_0_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FR(environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) != 0.0


def test_FR_lambda_run_learning_episode():
    environment = FakeContinuousEnvironment()
    environment.approximate_with(FuzzyApproximator, fuzzy_sets=fuzzy_sets)
    algorithm = fuzzy.FR(environment, lambd=0.1)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) == 0.0
    env = algorithm.run_learning_episode()
    assert isinstance(env, Environment)
    assert sum([algorithm.q[i].sum() for i in algorithm.actions]) != 0.0
