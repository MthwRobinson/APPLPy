"""ErrorRV distribution."""

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


class ErrorRV(RV):
    """
    Procedure Name: ErrorRV
    Purpose: Creates an instance of the error distribution
    Arguments:  1. mu: a strictly positive parameter
                2. alpha: a real valued parameter
                3. d: a real valued parameter
    Output:     1. An error random variable
    """

    def __init__(self, mu=Symbol("mu", positive=True), alpha=Symbol("alpha"), d=Symbol("d")):
        if not isinstance(mu, Symbol):
            if mu <= 0:
                err_string = "mu must be positive"
                raise RVError(err_string)
        if mu in [-oo, oo]:
            if alpha in [-oo, oo]:
                if d in [-oo, oo]:
                    err_string = "all parameters must be finite"
                    raise RVError(err_string)
        X_dummy = RV(
            mu * exp(-(abs(mu * (x - d)) ** alpha)) / (2 * gamma(1 + 1 / alpha)), [-oo, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["ErrorRV"]
