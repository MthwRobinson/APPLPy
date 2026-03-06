"""ChiSquareRV distribution."""

from random import random

from sympy import (
    Rational,
    Symbol,
    atan,
    cos,
    cot,
    exp,
    factorial,
    floor,
    gamma,
    integrate,
    ln,
    log,
    oo,
    pi,
    simplify,
    sqrt,
    symbols,
)

from ...rv import IDF, RV, RVError
from .param_check import param_check

x, y, z, t, v = symbols("x y z t v")


class ChiSquareRV(RV):
    """
    Procedure Name: ChiSquareRV
    Purpose: Creates an instance of the chi square distribution
    Arguments:  1. N: a positive integer parameter
    Output:     1. A chi squared random variable
    """

    def __init__(self, N=Symbol("N", positive=True, integer=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(N, Symbol):
            if N <= 0 or not isinstance(N, int):
                err_string = "N must be a positive integer"
                raise RVError(err_string)
        X_dummy = RV((x ** ((N / 2) - 1) * exp(-x / 2)) / (2 ** (N / 2) * gamma(N / 2)), [0, oo])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["ChiSquareRV"]
