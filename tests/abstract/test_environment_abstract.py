import pytest

from rltoolbox.tests.fakes import FakeContinuousEnvironment, FakeApproximator
from rltoolbox.tests.fakes import FakeModel


def test_done():
    env = FakeContinuousEnvironment(101, [[1.0, 2.0], []])
    assert env.done is False
    for i in range(101):
        env.do_action(0)
    assert env.done is True


def test_states():
    env = FakeContinuousEnvironment(101)
    with pytest.raises(AttributeError):
        env.states
    env.approximate_with(FakeApproximator)
    assert env.states == 2


def test_state():
    env = FakeContinuousEnvironment(101)
    assert env.state == (0.0, 1.0)
    env.approximate_with(FakeApproximator)
    assert env.state == 1

