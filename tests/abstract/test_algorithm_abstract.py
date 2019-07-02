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


@pytest.mark.parametrize('steps_per_episode,spe_lte,spe_gte,wsize,expected',[
    ([0, 1, 2, 3, 4, 4, 4], 0, 3, 5, False),
    ([0, 1, 2, 3, 4, 4, 4], 0, 3, 4, True),
    ([0, 1, 2, 3, 4, 4, 4], 0, 5, 20, False),
    ([40, 35, 30, 33, 25, 20, 19, 20, 18, 17, 19, 20], 19, 10000, 5, False),
    ([40, 35, 30, 33, 25, 20, 19, 20, 18, 17, 19, 20], 20, 10000, 4, True),
    ([40, 35, 30, 33, 25, 20, 19, 20, 18, 17, 19], 19, 10000, 3, True),
    ([40, 40, 40], 50, 10000, 5, False),
])
def test_is_learned(steps_per_episode, spe_lte, spe_gte, wsize, expected):
    env = FakeContinuousEnvironment()
    alg = FakeAlgorithm(env, 0.1, 0.2, 0.3, 0.4)
    alg.steps_per_episode = steps_per_episode
    assert alg.is_learned(spe_lte, spe_gte, wsize) is expected


def test_learning():
    env = FakeContinuousEnvironment()
    alg = FakeAlgorithm(env, 0.1, 0.2, 0.3, 0.4)
    steps_per_episode, e = alg.learn()
    assert len(steps_per_episode) == alg.episodes == 1
    assert steps_per_episode[0] == len(e.steps)
    steps_per_episode, e = alg.learn(23)
    assert len(steps_per_episode) == alg.episodes == 24


def test_learning_with_stopping_when_learned():
    env = FakeContinuousEnvironment()
    alg = FakeAlgorithm(env, 0.1, 0.2, 0.3, 0.4)
    steps_per_episode, e = alg.learn(100, True, spe_gte=0, wsize=13)
    assert alg.is_learned(0, 0, 13)
    assert len(steps_per_episode) == 13
