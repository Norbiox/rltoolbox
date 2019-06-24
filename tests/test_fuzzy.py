import pytest
from numpy import inf
from random import uniform

from rltoolbox.fuzzy import *


@pytest.mark.parametrize('xs', [
    (0, -1, 1),
    (0, 0, -1),
    (0, 1, 0)
])
def test_triangular_membership_function_wrong_xs(xs):
    with pytest.raises(ValueError):
        TriangularMembershipFunction(*xs)


@pytest.mark.parametrize('xs, x, expected_grade', [
    ((-1, 0, 1), -1.0, 0.0),
    ((-1, 0, 1), 1.0, 0.0),
    ((-1, 0, 1), 0.0, 1.0),
    ((-1, 0, 1), -0.5, 0.5),
    ((-1, 0, 1), 0.5, 0.5),
    ((-5, -2, 1), -5.0, 0.0),
    ((-5, -2, 1), 1.0, 0.0),
    ((-5, -2, 1), -2.0, 1.0),
    ((-5, -2, 1), -3.5, 0.5),
    ((-5, -2, 1), -0.5, 0.5),
    ((-3, 1, 2), -3.0, 0.0),
    ((-3, 1, 2), 2.0, 0.0),
    ((-3, 1, 2), 1.0, 1.0),
    ((-3, 1, 2), -1.0, 0.5),
    ((-3, 1, 2), 1.5, 0.5),
])
def test_triangular_membership_function_calc_grade(xs, x, expected_grade):
    tmf = TriangularMembershipFunction(*xs)
    grade = tmf.membership_grade(x)
    assert round(grade, 1) == expected_grade


@pytest.mark.parametrize('xs', [
    (0, -1, 1, 2),
    (0, 0, -1, 2),
    (0, 0, 0, -1)
])
def test_trapezoidal_membership_function_wrong_xs(xs):
    with pytest.raises(ValueError):
        TrapezoidalMembershipFunction(*xs)


@pytest.mark.parametrize('xs, x, expected_grade', [
    ((-5, -2, 1, 9), -5.0, 0.0),
    ((-5, -2, 1, 9), -3.5, 0.5),
    ((-5, -2, 1, 9), -2.0, 1.0),
    ((-5, -2, 1, 9), -1.0, 1.0),
    ((-5, -2, 1, 9), 1.0, 1.0),
    ((-5, -2, 1, 9), 5.0, 0.5)
])
def test_trapezoid_membership_function_calc_grade(xs, x, expected_grade):
    tpmf = TrapezoidalMembershipFunction(*xs)
    grade = tpmf.membership_grade(x)
    assert round(grade, 1) == expected_grade


def test_fuzzy_set_error_given_list_has_nonmembershipfunction_element():
    with pytest.raises(TypeError):
        fs = FuzzySet([TrapezoidalMembershipFunction(1,2,3), 'sdf'])


def test_fuzzy_get_membership_grades():
    mf1 = TrapezoidalMembershipFunction(0.0, 2.0, 4.0, 6.0)
    mf2 = TriangularMembershipFunction(-2.0, 0.0, 2.0)
    mf3 = TriangularMembershipFunction(4.0, 10.0, 12.0)
    fs = FuzzySet([mf1, mf2, mf3])
    for i in range(10):
        x = uniform(-2.0, 12.0)
        assert fs.membership_grades(x).all() == np.array([
            mf1.membership_grade(x),
            mf2.membership_grade(x),
            mf3.membership_grade(x)
        ]).all()


@pytest.mark.parametrize('membership_function_ranges', [
    ((), ),
    ((1), ),
    ((1, 2), ),
    ((3, 4, 5, 6, 7), ),
])
def test_create_membership_function_from_range_tuple_value_errors(
        membership_function_ranges
):
    with pytest.raises(ValueError):
        mf = create_membership_function_from_range_tuple(membership_function_ranges)


def test_create_membership_function_from_not_tuple_error():
    with pytest.raises(TypeError):
        mf = create_membership_function_from_range_tuple('asdf')


@pytest.mark.parametrize(
    'membership_function_ranges,n_membership_functions,membership_functions_types',
    [
        ([(0.0, 1.0, 2.0)], 1, [TriangularMembershipFunction]),
        ([(0.0, 1.0, 2.0, 3.0)], 1, [TrapezoidalMembershipFunction]),
        ([(0.0, 1.0, 2.0), (3.0, 4.0, 5.0, 6.0)], 2, [
            TriangularMembershipFunction, TrapezoidalMembershipFunction
        ])
    ]
)
def test_fuzzy_set_from_membership_function_ranges(
        membership_function_ranges,
        n_membership_functions,
        membership_functions_types
):
    fs = FuzzySet.from_membership_functions_ranges(membership_function_ranges)
    assert len(fs.membership_functions) == n_membership_functions
    for mf, mftype in zip(fs.membership_functions, membership_functions_types):
        assert isinstance(mf, mftype)
