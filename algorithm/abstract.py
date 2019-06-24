from ..abstract import Algorithm
from ..approximator import (
    TableApproximator,
    CMACApproximator,
    FuzzyApproximator
)
from ..environment.grid import GridEnvironment


class ClassicAlgorithm(Algorithm):

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        is_grid = isinstance(value, GridEnvironment)
        if not is_grid:
            if not isinstance(value.approximator, TableApproximator):
                raise TypeError(f"{self} can work only with grid or table" +
                                " approximated environments")
        self._environment = value


class CMACAlgorithm(Algorithm):

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        if not isinstance(value.approximator, CMACApproximator):
            raise TypeError(f"{self} can work only with CMAC" +
                            " approximated environments")
        self._environment = value


class FuzzyAlgorithm(Algorithm):

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        if not isinstance(value.approximator, FuzzyApproximator):
            raise TypeError(f"{self} can work only with fuzzy" +
                            " approximated environments")
        self._environment = value
