"""ArcTanRV distribution."""

from sympy import (
    Symbol,
    atan,
    oo,
    pi,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class ArcTanRV(RV):
    """
    Procedure Name: ArcTanRV
    Purpose: Creates an instance of the arc tan distribution
    Arguments:  1. alpha: a strictly positive parameter
    Output:     1. An arc tan random variable
    """

    def __init__(self, alpha=Symbol("alpha", positive=True), phi=Symbol("phi")):
        # Return an error if invalid parameters are entered
        if alpha in [-oo, oo]:
            if phi in [-oo, oo]:
                err_string = "Both parameters must be finite"
                raise RVError(err_string)
        if alpha <= 0:
            err_string = "Alpha must be positive"
            raise RVError(err_string)
        X_dummy = RV(
            alpha / ((atan(alpha * phi) + pi / 2) * (1 + alpha**2 * (x - phi) ** 2)), [0, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["ArcTanRV"]
