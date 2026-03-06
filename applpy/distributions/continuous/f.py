"""FRV distribution."""


from sympy import (
    Symbol,
    gamma,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class FRV(RV):
    """
    Procedure Name: FRV
    Purpose: Creates an instance of the f distribution
    Arguments:  1. n1: a strictly positive parameter
                2. n2: a strictly positive parameter
    Output:     1. A chi squared random variable
    """

    def __init__(self, n1=Symbol("n1", positive=True), n2=Symbol("n2", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(n1, Symbol):
            if not isinstance(n2, Symbol):
                if n1 <= 0 or n2 <= 0:
                    err_string = "both parameters must be positive"
                    raise RVError(err_string)
        if n1 in [-oo, oo] or n2 in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            gamma((n1 + n2) / 2)
            * (n1 / n2) ** (n1 / 2)
            * x ** (n1 / 2 - 1)
            / gamma(n1 / 2)
            * gamma(n2 / 2)
            * ((n1 / n2) * x + 1) ** ((n1 + n2) / 2),
            [0, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["FRV"]
