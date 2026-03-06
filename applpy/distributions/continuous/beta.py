"""BetaRV distribution."""


from sympy import (
    Symbol,
    gamma,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class BetaRV(RV):
    """
    Procedure Name: BetaRV
    Purpose: Creates an instance of the beta distribution
    Arguments:  1. alpha: a strictly positive parameter
                2. beta: a strictly positive parameter
    Output:     1. A beta random variable
    """

    def __init__(self, alpha=Symbol("alpha", positive=True), beta=Symbol("beta"), positive=True):
        # x = Symbol('x', positive = True)
        if alpha in [-oo, oo]:
            if beta in [-oo, oo]:
                err_string = "Both parameters must be finite"
                raise RVError(err_string)
        if alpha <= 0 and not isinstance(alpha, Symbol):
            if beta <= 0 and beta.__class__.name__ != "Symbol":
                err_string = "Both parameters must be positive"
                raise RVError(err_string)
        X_dummy = RV(
            (gamma(alpha + beta) * (x ** (alpha - 1)) * (1 - x) ** (beta - 1))
            / (gamma(alpha) * gamma(beta)),
            [0, 1],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["BetaRV"]
