"""IDBRV distribution."""


from sympy import (
    Symbol,
    exp,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class IDBRV(RV):
    """
    Procedure Name: IDBRV
    Purpose: Creates an instance of the idb distribution
    Arguments:  1. theta: a real valued parameter
                2. delta: a real valued parameter
                3. kappa: a real valued parameter
    Output:     1. An idb random variable
    """

    def __init__(self, theta=Symbol("theta"), delta=Symbol("delta"), kappa=Symbol("kappa")):
        # x = Symbol('x', positive = True)
        if theta in [-oo, oo] or delta in [-oo, oo] or kappa in [-oo, oo]:
            err_string = "all parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(1 - (1 + kappa * x) ** (-theta / kappa) * exp(-delta * x**2 / 2), [0, oo])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["IDBRV"]
