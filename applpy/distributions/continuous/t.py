"""TRV distribution."""


from sympy import (
    Symbol,
    gamma,
    oo,
    pi,
    sqrt,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class TRV(RV):
    """
    Procedure Name: TRV
    Purpose: Creates an instance of the t distribution
    Arguments:  1. N: a positive integer parameter
    Output:     1. A log normal random variable
    """

    def __init__(self, N=Symbol("N"), positive=True, integer=True):
        if not isinstance(N, Symbol):
            if N <= 0:
                if not isinstance(N, int):
                    err_string = "N must be a positive integer"
                    raise RVError(err_string)
        X_dummy = RV(
            [
                (gamma(N / 2 + 1 / 2) * (1 + ((x**2) / N)) ** (-(N / 2) - 1 / 2))
                / (sqrt(N * pi) * gamma(N / 2))
            ],
            [-oo, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["TRV"]
