"""InverseGaussianRV distribution."""


from sympy import (
    Symbol,
    exp,
    oo,
    pi,
    sqrt,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class InverseGaussianRV(RV):
    """
    Procedure Name: InverseGaussianRV
    Purpose: Creates an instance of the inverse gaussian distribution
    Arguments:  1. theta: a strictly positive parameter
                2. mu: a strictly positive parameter
    Output:     1. An inverse gaussian random variable
    """

    def __init__(self, theta=Symbol("theta", positive=True), mu=Symbol("mu", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(theta, Symbol):
            if not isinstance(mu, Symbol):
                if theta <= 0 or mu <= 0:
                    err_string = "both parameters must be positive"
                    raise RVError(err_string)
        if theta in [-oo, oo] or mu in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            [
                (1 / 2)
                * sqrt(2)
                * sqrt(theta / (pi * x**3))
                * exp(-(1 / 2) * (theta * (x - mu) ** 2) / (mu ** (2) * x))
            ],
            [0, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["InverseGaussianRV"]
