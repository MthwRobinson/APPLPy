"""ChiRV distribution."""


from sympy import (
    Rational,
    Symbol,
    exp,
    gamma,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class ChiRV(RV):
    """
    Procedure Name: ChiRV
    Purpose: Creates an instance of the chi distribution
    Arguments:  1. N: a positive integer parameter
    Output:     1. A chi random variable
    """

    def __init__(self, N=Symbol("N", positive=True, integer=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(N, Symbol):
            if N <= 0 or not isinstance(N, int):
                err_string = "N must be a positive integer"
                raise RVError(err_string)
        X_dummy = RV(
            ((x ** (N - 1)) * exp(-(x**2) / 2))
            / (2 ** ((Rational(N, 2)) - 1) * gamma(Rational(N, 2))),
            [0, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["ChiRV"]
