"""GeometricRV distribution."""

from sympy import Symbol, oo, symbols

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class GeometricRV(RV):
    """
    Procedure Name: GeometricRV
    Purpose: Creates an instance of the geometric distribution
    Arguments:  1. p: a positive parameter between 0 and 1
    Output:     1. A geometric random variable
    """

    def __init__(self, p=Symbol("p", positive=True)):
        if not isinstance(p, Symbol):
            if p <= 0 or p >= 1:
                err_string = "p must be between 0 and 1"
                raise RVError(err_string)
        X_dummy = RV([p * (1 - p) ** (x - 1)], [1, oo], ["Discrete", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["GeometricRV"]
