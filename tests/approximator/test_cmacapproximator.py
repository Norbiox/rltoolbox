import numpy as np
import pytest

from rltoolbox.approximator import CMACApproximator


@pytest.mark.parametrize('init_parameters',[
    (0, []),
    (2, []),
    (2, [[]]),
    (2, [[], [], []]),
    (2, [[1,2,3], None]),
    (2, [[], []], 1)
])
def test_init_errors(init_parameters):
    with pytest.raises(ValueError):
        approximator = CMACApproximator(*init_parameters)


@pytest.mark.parametrize('variable_ranges,n_layers,expected_layers',[
    ([], 2, [[], []]),
    ([0.0], 3, [[0.0], [0.0], [0.0]]),
    ([1.0, 2.0, 3.0], 2, [
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5, 3.5]
    ]
     ),
    ([-0.9, 0.0, 0.9], 3, [
        [-0.9, 0.0, 0.9],
        [-1.5, -0.6, 0.3, 1.2],
        [-1.2, -0.3, 0.6, 1.5]
    ]
     )
])
def test_divide_variable_range_by_layers(variable_ranges, n_layers,
                                          expected_layers):
    layers = CMACApproximator.divide_variable_range_by_layers(
        variable_ranges, n_layers
    )
    assert layers == expected_layers


@pytest.mark.parametrize(
    'n_state_variables,state_variables_ranges,n_layers,expected_layers_ranges,expected_possible_states',
    [
        (1, [[1.0, 2.0, 3.0]], 2, [
            [[1.0, 2.0, 3.0]],
            [[0.5, 1.5, 2.5, 3.5]]
        ], (
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4]
        )
         ),
        (1, [[-0.9, 0.0, 0.9]], 3, [
            [[-0.9, 0.0, 0.9]],
            [[-1.5, -0.6, 0.3, 1.2]],
            [[-1.2, -0.3, 0.6, 1.5]]
        ], (
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]
        )
         ),
        (2, [[0.0], []], 3, [
            [[0.0], []],
            [[0.0], []],
            [[0.0], []]
        ], (
            [0, 1],
            [0, 1],
            [0, 1]
        )
         ),
        (2, [[-1.0, 5.0], [1.0, 4.0, 7.0]], 3, [
            [[-1.0, 5.0], [1.0, 4.0, 7.0]],
            [[-5.0, 1.0, 7.0], [-1.0, 2.0, 5.0, 8.0]],
            [[-3.0, 3.0, 9.0], [0.0, 3.0, 6.0, 9.0]]
        ], (
            list(range(3 * 4)),
            list(range(4 * 5)),
            list(range(4 * 5))
        )
         )
    ]
)
def test_generate_layers_ranges(n_state_variables, state_variables_ranges, n_layers,
                         expected_layers_ranges, expected_possible_states):
    approximator = CMACApproximator(n_state_variables, state_variables_ranges,
                                    n_layers)
    layers_ranges = approximator.generate_layers_ranges()
    assert layers_ranges == expected_layers_ranges
    assert approximator.possible_states == expected_possible_states


@pytest.mark.parametrize(
    'n_state_variables,state_variables_ranges,n_layers,state_variables,expected_state', [
        (2, [[0.0], []], 3, (1.0, 5.0), (1, 1, 1)),
        (2, [[0.0], [0.0]], 3, (1.0, 1.0), (3, 3, 3)),
        (1, [[1.0, 2.0, 3.0]], 2, (2.4,), (2, 2)),
        (2, [[0.0], [1.0, 2.0, 3.0]], 2, (0.5, 2.6), (6, 8)),
        (2, [[-1.0, 5.0], [1.0, 4.0, 7.0]], 3, (2.0, 3.7), (5, 12, 7))
    ]
)
def test_approximate_state(n_state_variables, state_variables_ranges, n_layers,
                           state_variables, expected_state):
    approximator = CMACApproximator(n_state_variables, state_variables_ranges,
                                    n_layers)
    state = approximator.approximate_state(state_variables)
    assert state == expected_state
