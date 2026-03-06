"""PoissonRV distribution."""

from sympy import Symbol, exp, factorial, oo, symbols

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class PoissonRV(RV):
    """
    Procedure Name: PoissonRV
    Purpose: Creates an instance of the poisson distribution
    Arguments:  1. theta: a strictly positive parameter
    Output:     1. A poisson random variable
    """

    def __init__(self, theta=Symbol("theta", positive=True)):
        if not isinstance(theta, Symbol):
            if theta <= 0:
                err_string = "theta must be positive"
                raise RVError(err_string)
        if theta in [-oo, oo]:
            err_string = "theta must be finite"
            raise RVError(err_string)
        X_dummy = RV([(theta ** (x) * exp(-theta)) / factorial(x)], [0, oo], ["Discrete", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["PoissonRV"]
