"""BinomialRV distribution."""

from sympy import Symbol, factorial, symbols

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class BinomialRV(RV):
    """
    Procedure Name: BinomialRV
    Purpose: Creates an instance of the binomial distribution
    Arguments:  1. N: a positive integer parameter
                2. p: a positive parameter between 0 and 1
    Output:     1. A binomial random variable
    """

    def __init__(self, N=Symbol("N", positive=True, integer=True), p=Symbol("p", positive=True)):
        if not isinstance(N, Symbol):
            if N <= 0:
                if not isinstance(N, int):
                    err_string = "N must be a positive integer"
                    raise RVError(err_string)
        if not isinstance(p, Symbol):
            if p <= 0 or p >= 1:
                err_string = "p must be between 0 and 1"
                raise RVError(err_string)
        X_dummy = RV(
            [(factorial(N) * p ** (x) * (1 - p) ** (N - x)) / (factorial(N - x) * factorial(x))],
            [0, N],
            ["Discrete", "pdf"],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["BinomialRV"]
