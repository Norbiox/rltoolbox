from functools import reduce
from numpy import inf

from .abstract import Approximator
from .fuzzy import FuzzySet
from .misc import window


class TableApproximator(Approximator):

    def __init__(self, n_state_variables: int, state_variables_ranges: list,
                 *args, **kwargs):
        super().__init__(n_state_variables, state_variables_ranges)
        self._state_shape = tuple(
            len(vr) + 1 for vr in self.state_variables_ranges
        )
        self._possible_states = [i for i in range(reduce(
            lambda x, y: x * y, self.state_shape
        ))]

    @property
    def possible_states(self):
        return self._possible_states

    @property
    def state_shape(self):
        return self._state_shape

    @staticmethod
    def approximate_state_variable(value: float, ranges: list) -> int:
        return sum([value >= rang for rang in ranges])

    @staticmethod
    def encode_state(approximated_state_variables: list,
                     state_shape: list) -> int:
        if len(approximated_state_variables) != len(state_shape):
            raise ValueError("number of state variables and number of state" +
                             " shapes must be equal")
        if any(map(lambda i: approximated_state_variables[i] >= state_shape[i],
                   range(len(state_shape)))):
            raise ValueError("one or more variable state(s) exceed(s) state" +
                             " shape")
        return sum([
            approximated_state_variables[i] * reduce(
                lambda x, y: x * y, state_shape[i+1::] or [1]
            )
            for i in range(len(approximated_state_variables))
        ])

    def approximate_state(self, state_variables: tuple) -> int:
        asv = [
            self.approximate_state_variable(var, shape)
            for var, shape
            in zip(state_variables, self.state_variables_ranges)
        ]
        return self.encode_state(asv, self.state_shape)


class CMACApproximator(Approximator):

    def __init__(self, n_state_variables: int, state_variables_ranges: list,
                 n_layers=2, *args, **kwargs):
        super().__init__(n_state_variables, state_variables_ranges)
        self.n_layers = n_layers
        self.layers_ranges = self.generate_layers_ranges()
        self.layers = [
            TableApproximator(self.n_state_variables, ranges)
            for ranges in self.layers_ranges
        ]
        self._possible_states = tuple(l.possible_states for l in self.layers)

    @property
    def n_layers(self):
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value):
        if value < 2:
            raise ValueError("number of layers must be at least 2")
        self._n_layers = value

    @property
    def possible_states(self):
        return self._possible_states

    @classmethod
    def divide_variable_range_by_layers(cls, rang: list, n_layers: int) -> list:
        if n_layers < 2:
            raise ValueError("number of layers must be at least 2")
        if len(rang) < 2:
            return [rang for i in range(n_layers)]
        distances = [round(x - y, 3) for x, y in zip(rang[1:], rang[:-1])]
        if len(set(distances)) > 1:
            raise ValueError("each state variable space must be divided" +
                             " into equal rang, e.g [-2.0, 0.0, 2.0]")
        distance = distances[0]
        left_border = rang[0] - distance
        range_step = distance / n_layers
        layers_ranges = [rang]
        for i in range(1, n_layers):
            layers_ranges.append([
                round(left_border + i * range_step + j * distance, 7)
                for j in range(len(rang) + 1)
            ])
        return layers_ranges

    def approximate_state(self, state_variables: tuple) -> tuple:
        return tuple(l.approximate_state(state_variables) for l in self.layers)

    def generate_layers_ranges(self) -> list:
        state_variables_ranges_divided = [
            self.divide_variable_range_by_layers(rang, self.n_layers)
            for rang in self.state_variables_ranges
        ]
        layers_ranges = [
            [rang[i] for rang in state_variables_ranges_divided]
            for i in range(self.n_layers)
        ]
        return layers_ranges


class FuzzyApproximator(Approximator):

    def __init__(self, n_state_variables: int, state_variables_ranges: list,
                 fuzzy_sets: list = None, *args, **kwargs):
        super().__init__(n_state_variables, state_variables_ranges)
        if fuzzy_sets is None:
            self.fuzzy_sets = [
                FuzzySet.from_membership_functions_ranges(self.svr2mfr(svr))
                for svr in state_variables_ranges
            ]
        else:
            self.fuzzy_sets = fuzzy_sets
        self.state_shape = tuple(
            len(fuzzy_set.membership_functions) for fuzzy_set in self.fuzzy_sets
        )

    @property
    def fuzzy_sets(self):
        return self._fuzzy_sets

    @fuzzy_sets.setter
    def fuzzy_sets(self, value: list):
        if not isinstance(value, list):
            raise TypeError(f"fuzzy_sets must be a list, given {type(value)}")
        if len(value) != self.n_state_variables:
            raise ValueError(f"fuzzy_sets must define at least empty" +
                             f" FuzzySet for each of {self.n_state_variables}" +
                             f" environment state variables,"
                             f" given {len(value)}")
        for obj in value:
            if not isinstance(obj, FuzzySet):
                raise TypeError(f"each object in fuzzy_sets list must be a" +
                                f" FuzzySet, given {type(obj)}")
        self._fuzzy_sets = value

    @property
    def possible_states(self):
        raise NotImplementedError("FuzzyApproximator has no possible states" +
                                  " defined, use state_shape instead to get" +
                                  " info about fuzzy sets configuration")

    @classmethod
    def svr2mfr(cls, state_variable_range: list) -> list:
        svr = state_variable_range
        mfr = []
        if len(svr) == 0:
            pass
        elif len(svr) == 1:
            mfr = [(-inf, -inf, svr[0], svr[0]), (svr[0], svr[0], inf, inf)]
        else:
            for i, pair in enumerate(window(svr)):
                delta = (pair[1] - pair[0]) / 2
                if i == 0:
                    mfr.append((
                        -inf,
                        -inf,
                        round(pair[0] - delta, 3),
                        round(pair[1] - delta, 3)
                    ))
                mfr.append((
                    round(pair[0] - delta, 3),
                    round(pair[0] + delta, 3),
                    round(pair[1] + delta, 3)
                ))
                if i == len(svr) - 2:
                    mfr.append((
                        round(pair[0] + delta, 3),
                        round(pair[1] + delta, 3),
                        inf,
                        inf
                    ))
        return mfr

    def approximate_state(self, state_variables):
        return [
            self.fuzzy_sets[i].membership_grades(state_variable)
            for i, state_variable in enumerate(state_variables)
        ]
