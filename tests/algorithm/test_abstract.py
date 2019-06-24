import pytest

from rltoolbox.tests.fakes import (
    FakeClassicalAlgorithm,
    FakeCMACAlgorithm,
    FakeFuzzyAlgorithm,
    FakeContinuousEnvironment,
    FakeGridNoWallsEnvironment,
    FakeGridWithWallsEnvironment
)
from rltoolbox.approximator import (
    TableApproximator,
    CMACApproximator,
    FuzzyApproximator
)


@pytest.mark.parametrize('env', [
    FakeGridNoWallsEnvironment(),
    FakeGridWithWallsEnvironment(),
    FakeContinuousEnvironment().approximate_with(TableApproximator)
])
def test_classical_algorithm_setting_proper_environment(env):
    a = FakeClassicalAlgorithm(env, lambd=0.0, epsilon=0.0, gamma=0.0, alpha=0.0)


@pytest.mark.parametrize('env', [
    FakeContinuousEnvironment().approximate_with(CMACApproximator)
])
def test_classical_algorithm_setting_wrong_environment(env):
    with pytest.raises(TypeError):
        a = FakeClassicalAlgorithm(env, lambd=0.0, epsilon=0.0, gamma=0.0, alpha=0.0)


@pytest.mark.parametrize('env', [
    FakeGridNoWallsEnvironment(),
    FakeGridWithWallsEnvironment(),
    FakeContinuousEnvironment().approximate_with(TableApproximator)
])
def test_CMAC_algorithm_setting_wrong_environment(env):
    with pytest.raises(TypeError):
        a = FakeCMACAlgorithm(env, lambd=0.0, epsilon=0.0, gamma=0.0, alpha=0.0)


@pytest.mark.parametrize('env', [
    FakeContinuousEnvironment().approximate_with(CMACApproximator)
])
def test_CMAC_algorithm_setting_proper_environment(env):
    a = FakeCMACAlgorithm(env, lambd=0.0, epsilon=0.0, gamma=0.0, alpha=0.0)


@pytest.mark.parametrize('env', [
    FakeGridNoWallsEnvironment(),
    FakeGridWithWallsEnvironment(),
    FakeContinuousEnvironment().approximate_with(TableApproximator),
    FakeContinuousEnvironment().approximate_with(CMACApproximator)
])
def test_fuzzy_algorithm_setting_wrong_environment(env):
    with pytest.raises(TypeError):
        a = FakeFuzzyAlgorithm(env, lambd=0.0, epsilon=0.0, gamma=0.0, alpha=0.0)


@pytest.mark.parametrize('env', [
    FakeContinuousEnvironment().approximate_with(FuzzyApproximator)
])
def test_CMAC_algorithm_setting_proper_environment(env):
    a = FakeFuzzyAlgorithm(env, lambd=0.0, epsilon=0.0, gamma=0.0, alpha=0.0)
