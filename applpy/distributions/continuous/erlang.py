"""ErlangRV distribution."""

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


class ErlangRV(RV):
    """
    Procedure Name: ErlangRV
    Purpose: Creates an instance of the Erlang distribution
    Arguments:  1. theta: a strictly positive parameter
                2. N: a positive integer parameter
    Output:     1. An erlang random variable
    """

    def __init__(
        self, theta=Symbol("theta", positive=True), N=Symbol("N", positive=True, integer=True)
    ):
        # x = Symbol('x', positive = True)
        if not isinstance(N, Symbol):
            if N <= 0 or not isinstance(N, int):
                err_string = "N must be a positive integer"
                raise RVError(err_string)
        if not isinstance(theta, Symbol):
            if theta <= 0:
                err_string = "theta must be positive"
                raise RVError(err_string)
        if theta in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            (theta * (theta * x) ** (N - 1) * exp(-theta * x)) / (factorial(N - 1)), [0, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["ErlangRV"]
