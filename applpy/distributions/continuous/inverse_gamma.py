"""InverseGammaRV distribution."""


from sympy import (
    Symbol,
    exp,
    gamma,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class InverseGammaRV(RV):
    """
    Procedure Name: InverseGammaRV
    Purpose: Creates an instance of the inverse gamma distribution
    Arguments:  1. alpha: a strictly positive parameter
                2. beta: a strictly positive parameter
    Output:     1. An inverse gamma random variable
    """

    def __init__(self, alpha=Symbol("alpha", positive=True), beta=Symbol("beta", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(alpha, Symbol):
            if not isinstance(beta, Symbol):
                if alpha <= 0 or beta <= 0:
                    err_string = "both parameters must be positive"
                    raise RVError(err_string)
        if alpha in [-oo, oo] or beta in [-oo, oo]:
            err_string = "both parameters must be finite"
        X_dummy = RV(
            [(x ** (1 - alpha) * exp(-1 / (x * beta))) / (gamma(alpha) * beta ** (alpha))], [0, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["InverseGammaRV"]
