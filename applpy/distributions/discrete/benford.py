"""BenfordRV distribution."""

from sympy import ln, symbols

from ...rv import RV

x, y, z, t, v = symbols("x y z t v")


class BenfordRV(RV):
    """
    Procedure Name: BenfordRV
    Purpose: Creates an instance of the Benford distribution
    Arguments:  1. None
    Output:     1. A Benford random variable
    """

    def __init__(self):
        X_dummy = RV([(ln((1 / x) + 1)) / (ln(10))], [1, 9], ["Discrete", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["BenfordRV"]
