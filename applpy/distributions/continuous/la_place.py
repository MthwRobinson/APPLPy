"""LaPlaceRV distribution."""

from sympy import (
    Symbol,
    exp,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class LaPlaceRV(RV):
    """
    Procedure Name: LaPlaceRV
    Purpose: Creates an instance of the LaPlace distribution
    Arguments:  1. omega: a strictly positive parameter
                2. theta: a real valued parameter
    Output:     1. A LaPlace random variable
    """

    def __init__(self, omega=Symbol("omega", positive=True), theta=Symbol("theta")):
        if not isinstance(omega, Symbol):
            if omega <= 0:
                err_string = "omega must be positive"
        if omega in [-oo, oo] or theta in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(exp(-abs(x - theta) / omega) / (2 * omega), [-oo, oo], ["continuous", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["LaPlaceRV"]
