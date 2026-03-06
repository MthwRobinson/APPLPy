"""ErrorIIRV distribution."""


from sympy import (
    Symbol,
    exp,
    gamma,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class ErrorIIRV(RV):
    """
    Procedure Name: ErrorIIRV
    Purpose: Creates an instance of the error II distribution
    Arguments:  1. a: a real valued parameter
                2. b: a real valued parameter
                3. c: a real valued parameter
    Output:     1. An error II random variable
    """

    def __init__(self, a=Symbol("a"), b=Symbol("b"), c=Symbol("c")):
        if a in [-oo, oo]:
            if b in [-oo, oo]:
                if c in [-oo, oo]:
                    err_string = "all parameters must be finite"
                    raise RVError(err_string)
        X_dummy = RV(
            exp(
                -((abs(x - a)) ** (2 / c) / (2 * b))
                / ((b ** (c / 2)) * 2 ** (c / 2 + 1) * gamma(c / 2 + 1))
            ),
            [-oo, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["ErrorIIRV"]
