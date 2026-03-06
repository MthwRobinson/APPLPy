"""RayleighRV distribution."""

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


class RayleighRV(RV):
    """
    Procedure Name: RayleighRV
    Purpose: Creates an instance of the Rayleigh distribution
    Arguments:  1. theta: a strictly positive parameter
    Output:     1. A log normal random variable
    """

    def __init__(self, theta=Symbol("theta", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(theta, Symbol):
            if theta <= 0:
                err_string = "both parameters must be positive"
                raise RVError(err_string)
        if theta in [-oo, oo]:
            err_string = "both parameters must be finite"
        X_dummy = RV([2 * theta ** (2) * x * exp(-(theta ** (2)) * x**2)], [0, oo])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["RayleighRV"]
