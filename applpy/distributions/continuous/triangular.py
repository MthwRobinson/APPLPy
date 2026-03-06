"""TriangularRV distribution."""


from sympy import (
    Symbol,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class TriangularRV(RV):
    """
    Procedure Name: TriangularRV
    Purpose: Creates an instance of the triangular distribution
    Arguments:  1. a: a real valued parameter
                2. b: a real valued parameter
                3. c: a real valued parameter
                ** Note: a<b<c ***
    Output:     1. A triangular variable
    """

    def __init__(self, a=Symbol("a"), b=Symbol("b"), c=Symbol("c")):
        if not isinstance(a, Symbol):
            if not isinstance(b, Symbol):
                if not isinstance(c, Symbol):
                    if a >= b or b >= c or a >= c:
                        err_string = "the parameters must be in ascending order"
                        raise RVError(err_string)
        if a in [-oo, oo] or b in [-oo, oo] or c in [-oo, oo]:
            err_string = "all parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            [(2 * (x - a)) / ((c - a) * (b - a)), (2 * (c - x)) / ((c - a) * (c - b))], [a, b, c]
        )
        self.func = X_dummy.func
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [a, b, c]
        self.cache = {}


__all__ = ["TriangularRV"]
