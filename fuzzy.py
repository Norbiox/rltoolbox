from abc import ABC, abstractmethod
import numpy as np


class MembershipFunction(ABC):

    @abstractmethod
    def membership_grade(self, x):
        pass

    def __repr__(self):
        return self.__str__()


class TriangularMembershipFunction(MembershipFunction):

    def __init__(self, a: float, b: float, c: float):
        if not (a <= b and b <= c):
            raise ValueError("'b' must be greater or equal 'a' and 'c' must" +
                             " be greater or equal 'b'")
        self.a = a
        self.b = b
        self.c = c

    def membership_grade(self, x: float) -> float:
        if self.a <= x <= self.b:
            try:
                return (x - self.a) / (self.b - self.a)
            except ZeroDivisionError:
                return 0.0
        elif self.b <= x <= self.c:
            try:
                return (self.c - x) / (self.c - self.b)
            except ZeroDivisionError:
                return 0.0
        return 0.0

    def __str__(self):
        return self.__class__.__name__ + f"({self.a}, {self.b}, {self.c})"

    def __repr__(self):
        return self.__str__()


class TrapezoidalMembershipFunction(MembershipFunction):

    def __init__(self, a: float, b: float, c: float, d: float):
        if not (a <= b and b <= c and c <= d):
            raise ValueError("each next x must be greater or equal previous x")
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def membership_grade(self, x: float) -> float:
        if self.a <= x <= self.b:
            try:
                return (x - self.a) / (self.b - self.a)
            except ZeroDivisionError:
                return 0.0
        elif self.b <= x <= self.c:
            return 1.0
        elif self.c <= x <= self.d:
            try:
                return (self.d - x) / (self.d - self.c)
            except ZeroDivisionError:
                return 0.0
        return 0.0

    def __str__(self):
        return self.__class__.__name__ + \
            f"({self.a}, {self.b}, {self.c}, {self.d})"

    def __repr__(self):
        return self.__str__()


def create_membership_function_from_range_tuple(range_tuple: tuple):
    if not isinstance(range_tuple, tuple):
        raise TypeError("range_tuple parameter must be a tuple of numbers")
    if len(range_tuple) == 3:
        return TriangularMembershipFunction(*range_tuple)
    elif len(range_tuple) == 4:
        return TrapezoidalMembershipFunction(*range_tuple)
    else:
        raise ValueError(f"cannot create membership function with" +
                         f" {len(range_tuple)} range values, must be 3 or 4")


class FuzzySet:

    def __init__(self, membership_functions: list):
        self.membership_functions = membership_functions

    @property
    def membership_functions(self):
        return self._membership_functions

    @membership_functions.setter
    def membership_functions(self, value: list):
        for mf in value:
            if not isinstance(mf, MembershipFunction):
                raise TypeError("all elements of 'membership_functions' " +
                                "param must be MembershipFunction instances")
        self._membership_functions = value

    @classmethod
    def from_membership_functions_ranges(cls,
                                         membership_functions_ranges: list):
        return cls([
            create_membership_function_from_range_tuple(mfr)
            for mfr in membership_functions_ranges
        ])

    def membership_grades(self, x: float):
        return np.array([
            mf.membership_grade(x) for mf in self.membership_functions
        ])

    def __str__(self):
        return self.__class__.__name__ + "(" + \
            ', '.join([str(mf) for mf in self.membership_functions]) + ")"

    def __repr__(self):
        return self.__str__()
