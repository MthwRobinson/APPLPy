"""GammaRV distribution."""


from sympy import (
    Symbol,
    exp,
    gamma,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class GammaRV(RV):
    """
    Procedure Name: GammaRV
    Purpose: Creates an instance of the gamma distribution
    Arguments:  1. theta: a strictly positive parameter
                2. kappa: a strictly positive parameter
    Output:     1. A chi squared random variable
    """

    def __init__(self, theta=Symbol("theta", positive=True), kappa=Symbol("kappa", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(theta, Symbol):
            if not isinstance(kappa, Symbol):
                if theta <= 0 or kappa <= 0:
                    err_string = "both parameters must be positive"
                    raise RVError(err_string)
        if theta in [-oo, oo] or kappa in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            (theta * (theta * x) ** (kappa - 1) * exp(-theta * x)) / (gamma(kappa)), [0, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [theta, kappa]
        self.cache = {}


__all__ = ["GammaRV"]
