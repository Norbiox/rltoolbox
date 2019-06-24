import pytest

from rltoolbox.tests.fakes import FakeAlgorithm, FakeContinuousEnvironment


def test_init_algorithm():
    env = FakeContinuousEnvironment()
    alg = FakeAlgorithm(env, 0.1, 0.2, 0.3, 0.4, 0.5)
    assert alg.environment == env
    assert len(alg.actions) == len(env.actions)
    assert alg.lambd == 0.1
    assert alg.epsilon == 0.2
    assert alg.gamma == 0.3
    assert alg.alpha == 0.4


@pytest.mark.parametrize('lambd,name',[
    (0.0, 'FakeAlgorithm(0)'),
    (0.1, 'FakeAlgorithm(lambda)')
])
def test_algorithm_name(lambd, name):
    env = FakeContinuousEnvironment()
    alg = FakeAlgorithm(env, lambd, 0.1, 0.2, 0.3)
    assert alg.name == name


def test_get_action():
    env = FakeContinuousEnvironment()
    alg = FakeAlgorithm(env, 0.1, 0.2, 0.3, 0.4)
    for i in range(100):
        assert alg.get_action() in alg.actions


def test_learning():
    env = FakeContinuousEnvironment()
    alg = FakeAlgorithm(env, 0.1, 0.2, 0.3, 0.4)
    steps_per_episode, e = alg.learn()
    assert len(steps_per_episode) == alg.episodes == 1
    assert steps_per_episode[0] == len(e.steps)
    steps_per_episode, e = alg.learn(23)
    assert len(steps_per_episode) == alg.episodes == 24
