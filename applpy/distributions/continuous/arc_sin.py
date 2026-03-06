"""ArcSinRV distribution."""


from sympy import (
    pi,
    sqrt,
    symbols,
)

from ...rv import RV

x, y, z, t, v = symbols("x y z t v")


class ArcSinRV(RV):
    """
    Procedure Name: ArcSinRV
    Purpose: Creates an instance of the arc sin distribution
    Arguments:  1. None
    Output:     1. An arc sin random variable
    """

    def __init__(self):
        # x = Symbol('x', postive=True)
        X_dummy = RV(1 / (pi * sqrt(x * (1 - x))), [0, 1])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["ArcSinRV"]
