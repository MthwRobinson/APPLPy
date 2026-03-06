"""LogGammaRV distribution."""


from sympy import (
    Symbol,
    exp,
    gamma,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class LogGammaRV(RV):
    """
    Procedure Name: LogGammaRV
    Purpose: Creates an instance of the log gamma distribution
    Arguments:  1. alpha: a strictly positive parameter
                2. beta: a strictly positive parameter
    Output:     1. A log gamma random variable
    """

    def __init__(self, alpha=Symbol("alpha", positive=True), beta=Symbol("beta", positive=True)):
        if not isinstance(alpha, Symbol):
            if not isinstance(beta, Symbol):
                if alpha <= 0 or beta <= 0:
                    err_string = "both parameters must be positive"
                    raise RVError(err_string)
        if alpha in [-oo, oo] or beta in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            [(exp(x * beta) * exp(-exp(x) / alpha)) / (alpha ** (beta) * gamma(beta))], [-oo, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["LogGammaRV"]
