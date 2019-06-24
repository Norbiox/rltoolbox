from numpy import inf
import pytest

from rltoolbox import fuzzy
from rltoolbox.approximator import FuzzyApproximator


@pytest.mark.parametrize('svr,expected_mfr', [
    ([1.0], [(-inf, -inf, 1.0, 1.0), (1.0, 1.0, inf, inf)]),
    ([-0.2, 0.2], [
        (-inf, -inf, -0.4, 0.0),
        (-0.4, 0.0, 0.4),
        (0.0, 0.4, inf, inf)
    ]),
    ([0.0, 2.0, 5.0], [
        (-inf, -inf, -1.0, 1.0),
        (-1.0, 1.0, 3.0),
        (0.5, 3.5, 6.5),
        (3.5, 6.5, inf, inf)
    ]),
    ([-0.86, -0.52, -0.18, 0.16], [
        (-inf, -inf, -1.03, -0.69),
        (-1.03, -0.69, -0.35),
        (-0.69, -0.35, -0.01),
        (-0.35, -0.01, 0.33),
        (-0.01, 0.33, inf, inf)
    ])
])
def test_state_variables_ranges_to_membership_functions_ranges(svr, expected_mfr):
    mfr = FuzzyApproximator.svr2mfr(svr)
    assert mfr == expected_mfr


@pytest.mark.parametrize('n_vars, svr, expected_state_shape', [
    (1, [[1.0]], (2, )),
    (2, [[0.0], [1.0, 2.0]], (2, 3)),
    (3, [[0.0, 1.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 8.0]], (3, 4, 5)),
    (4, [[0.0], [1.0], [], [2.0]], (2, 2, 0, 2))
])
def test_from_state_variables_ranges(n_vars, svr, expected_state_shape):
    fa = FuzzyApproximator(n_vars, svr)
    assert len(fa.fuzzy_sets) == n_vars
    for i in range(n_vars):
        if len(svr[i]) == 0:
            assert len(fa.fuzzy_sets[i].membership_functions) == 0
        elif len(svr[i]) == 1:
            assert len(fa.fuzzy_sets[i].membership_functions) == 2
        else:
            assert len(fa.fuzzy_sets[i].membership_functions) == len(svr[i]) + 1
    assert fa.state_shape == expected_state_shape


@pytest.mark.parametrize('fuzzy_sets', [
    fuzzy.FuzzySet([]),
    (fuzzy.FuzzySet([]), fuzzy.FuzzySet([])),
])
def test_from_fuzzy_sets_non_fuzzy_sets_list_error(fuzzy_sets):
    with pytest.raises(TypeError):
        FuzzyApproximator(2, [[], []], fuzzy_sets=fuzzy_sets)


@pytest.mark.parametrize('n_vars,fuzzy_sets', [
    (2, [fuzzy.FuzzySet([])]),
    (2, [fuzzy.FuzzySet([]) for i in range(3)])
])
def test_from_fuzzy_sets_missing_fuzzy_sets(n_vars, fuzzy_sets):
    with pytest.raises(ValueError):
        FuzzyApproximator(n_vars, [[] for i in range(n_vars)], fuzzy_sets)


@pytest.mark.parametrize('fuzzy_sets', [
    ['asdf'],
    [(0,)],
    [[1,2,3]],
    [fuzzy.TriangularMembershipFunction(1, 2, 3)]
])
def test_from_fuzzy_sets_non_fuzzy_set_objects(fuzzy_sets):
    with pytest.raises(TypeError):
        FuzzyApproximator(1, [[]], fuzzy_sets)


def test_approximate_state():
    fa = FuzzyApproximator(4, [[0.0], [1.0], [], [2.0]])
    state_variables = (0.0, 1.0, 2.0, 3.0)
    state = fa.approximate_state(state_variables)
    for i, s in enumerate(state):
        assert s.all() == fa.fuzzy_sets[i].membership_grades(state_variables[i]).all()
