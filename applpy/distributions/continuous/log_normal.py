"""LogNormalRV distribution."""


from sympy import (
    Symbol,
    exp,
    ln,
    oo,
    pi,
    sqrt,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class LogNormalRV(RV):
    """
    Procedure Name: LogNormalRV
    Purpose: Creates an instance of the log normal distribution
    Arguments:  1. mu: a real valued parameter
                2. sigma: a strictly positive parameter
    Output:     1. A log normal random variable
    """

    def __init__(self, mu=Symbol("mu"), sigma=Symbol("sigma", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(sigma, Symbol):
            if sigma <= 0:
                err_string = "sigma must be positive"
                raise RVError(err_string)
        if mu in [-oo, oo] or sigma in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            [
                (1 / 2)
                * (sqrt(2) * exp((-1 / 2) * ((ln(x) - mu) ** 2) / (sigma**2)))
                / (sqrt(pi) * x * sigma)
            ],
            [0, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["LogNormalRV"]
