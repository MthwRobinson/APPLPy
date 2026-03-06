"""MuthRV distribution."""


from sympy import (
    Symbol,
    exp,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class MuthRV(RV):
    """
    Procedure Name: MuthRV
    Purpose: Creates an instance of the Muth distribution
    Arguments:  1. kappa: a strictly positive parameter
    Output:     1. A log normal random variable
    """

    def __init__(self, kappa=Symbol("kappa", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(kappa, Symbol):
            if kappa <= 0:
                err_string = "kappa must be positive"
                raise RVError(err_string)
        if kappa in [-oo, oo]:
            err_string = "kappa must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            [(exp(kappa * x) - kappa) * exp((-exp(kappa * x) / kappa) + kappa * x + (1 / kappa))],
            [0, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["MuthRV"]
