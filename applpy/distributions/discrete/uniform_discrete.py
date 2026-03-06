"""UniformDiscreteRV distribution."""

from sympy import Rational, Symbol, symbols

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class UniformDiscreteRV(RV):
    """
    Procedure Name: UniformDiscreteRV
    Purpose: Creates an instance of the uniform discrete distribution
    Arguments:  1. a: the beggining point of the interval
                2. b: the end point of the interval (note: b>a)
    Output:     1. A uniform discrete random variable
    """

    def __init__(self, a=Symbol("a"), b=Symbol("b"), k=1):
        if b <= a:
            err_string = "b is only valid if b > a"
            raise RVError(err_string)
        if (b - a) % k != 0:
            err_string = "(b-a) must be divisble by k"
            raise RVError(err_string)
        n = int((b - a) / k)
        X_dummy = RV(
            [Rational(1, n + 1) for i in range(1, n + 2)],
            [a + i * k for i in range(n + 1)],
            ["discrete", "pdf"],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["UniformDiscreteRV"]
