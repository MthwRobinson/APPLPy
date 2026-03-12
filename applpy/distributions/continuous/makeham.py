"""MakehamRV distribution."""

from sympy import (
    Symbol,
    exp,
    log,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class MakehamRV(RV):
    """
    Procedure Name: MakehamRV
    Purpose: Creates an instance of the Makeham distribution
    Arguments:  1. theta: a strictly positive parameter
                2. delta: a strictly positive parameter
                3: kappa: a strictly positive parameter
    Output:     1. A log normal random variable
    """

    def __init__(
        self,
        theta=Symbol("theta", positive=True),
        delta=Symbol("delta", positive=True),
        kappa=Symbol("kappa"),
    ):
        # x = Symbol('x', positive = True)
        if not isinstance(theta, Symbol):
            if not isinstance(delta, Symbol):
                if theta <= 0 or delta <= 0:
                    err_string = "alpha and delta must be positive"
                    raise RVError(err_string)
        if theta in [-oo, oo] or delta in [-oo, oo] or kappa in [-oo, oo]:
            err_string = "all parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            (theta + delta * kappa**x) * exp(-theta * x - delta * (kappa**x - 1) / log(kappa)),
            [0, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["MakehamRV"]
