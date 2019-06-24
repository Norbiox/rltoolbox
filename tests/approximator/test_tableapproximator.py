import numpy as np
import pytest

from rltoolbox.approximator import TableApproximator


@pytest.mark.parametrize('init_parameters',[
    (0, []),
    (2, []),
    (2, [[]]),
    (2, [[], [], []]),
    (2, [[1,2,3], None])
])
def test_init_errors(init_parameters):
    with pytest.raises(ValueError):
        approximator = TableApproximator(*init_parameters)

@pytest.mark.parametrize('init_parameters,expected_state_shape',[
    ((2, [[], []]), (1, 1)),
    ((2, [[0], []]), (2, 1)),
    ((2, [[0], [0, 1]]), (2, 3)),
    ((4, [[1, 2, 3], [4, 5], [6], []]), (4, 3, 2, 1))
])
def test_state_shape(init_parameters, expected_state_shape):
    apx = TableApproximator(*init_parameters)
    assert apx.state_shape == expected_state_shape


@pytest.mark.parametrize('init_parameters,expected_possible_states',[
    ((2, [[], []]), [0]),
    ((2, [[0], []]), [0, 1]),
    ((2, [[0], [0, 1]]), [0, 1, 2, 3, 4, 5]),
    ((4, [[1, 2, 3], [4, 5], [6], []]), list(range(24)))
])
def test_state_shape(init_parameters, expected_possible_states):
    apx = TableApproximator(*init_parameters)
    assert apx.possible_states == expected_possible_states


@pytest.mark.parametrize('value,ranges,expected_state',[
    (2.0, [], 0),
    (2.0, [2.5], 0),
    (2.0, [1.5], 1),
    (2.0, [1.0, 2.0], 2)
])
def test_approximate_state_variable(value, ranges, expected_state):
    state = TableApproximator.approximate_state_variable(value, ranges)
    assert state == expected_state


@pytest.mark.parametrize('approximated_state_variables,state_shape',[
    ([1], [2, 4]),
    ([2, 4], [10])
])
def test_encoding_state_parameters_lengths_errors(approximated_state_variables, state_shape):
    with pytest.raises(ValueError):
        TableApproximator.encode_state(approximated_state_variables, state_shape)


@pytest.mark.parametrize('approximated_state_variables,state_shape',[
    ([1], [1]),
    ([2, 4], [4, 3])
])
def test_encoding_states_state_values_errors(approximated_state_variables, state_shape):
    with pytest.raises(ValueError):
        TableApproximator.encode_state(approximated_state_variables, state_shape)


@pytest.mark.parametrize('approximated_state_variables,state_shape,expected_state',[
    ([2, 3], [3, 4], 11),
    ([1, 2], [4, 5], 7),
    ([1, 3, 5], [2, 5, 10], 50 + 30 + 5)
])
def test_encode_state(approximated_state_variables, state_shape, expected_state):
    state = TableApproximator.encode_state(approximated_state_variables, state_shape)
    assert state == expected_state


@pytest.mark.parametrize('n_state_vars,state_vars_ranges,state_vars,expected_state',[
    (1, [[0.0, 1.0, 2.0, 3.0]], [2.0], 3),
    (2, [[1.0], [2.0, 4.0]], [2.0, 3.0], 1 * 3 + 1),
    (2, [[-1.0, 0.0, 1.0], [-1.0, -0.9, -0.5]], [0.5, -0.99], 2 * 4 + 1),
    (3, [[0.0], [-1.0, 0.0, 2.0], [1.1, 1.2, 2.3, 2.4]], [-1.0, 1.0, 2.2],
     0 * 4 * 5 + 2 * 5 + 2)
])
def test_approximate_state(n_state_vars, state_vars_ranges, state_vars, expected_state):
    apx = TableApproximator(n_state_vars, state_vars_ranges)
    state = apx.approximate_state(state_vars)
    assert state == expected_state
