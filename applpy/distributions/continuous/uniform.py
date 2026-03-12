"""UniformRV distribution."""

from random import random

from sympy import (
    Symbol,
    oo,
    simplify,
    symbols,
)

from ...rv import IDF, RV, RVError
from .param_check import param_check

x, y, z, t, v = symbols("x y z t v")


class UniformRV(RV):
    """
    Procedure Name: UniformRV
    Purpose: Creates an instance of the uniform distribution
    Arguments:  1. a: a real valued parameter
                2. b: a real valued parameter
                ** Note: b>a **
    Output:     1. A uniform random variable
    """

    def __init__(self, a=Symbol("a"), b=Symbol("b")):
        if not isinstance(a, Symbol):
            if not isinstance(b, Symbol):
                if a >= b:
                    err_string = "the parameters must be in ascending order"
                    raise RVError(err_string)
        if a in [-oo, oo] or b in [-oo, oo]:
            err_string = "all parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(simplify((b - a) ** (-1)), [a, b], ["continuous", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [a, b]
        self.cache = {}

    def variate(self, n=1, s=None, method="special"):
        # If no parameter is specified, return an error
        if not param_check(self.parameter):
            raise RVError("Not all parameters specified")

        # Check to see if the user specified a valid method
        method_list = ["special", "inverse"]
        if method not in method_list:
            error_string = "an invalid method was specified"
            raise RVError(error_string)

        # If the inverse method is specified, compute variates using
        #   the IDF function
        if method == "inverse":
            Xidf = IDF(self)
            varlist = [IDF(Xidf, random()) for i in range(1, n + 1)]
            return varlist

        # Generate uniform variates
        idf_func = -t * self.parameter[0] + t * self.parameter[1] + self.parameter[0]
        varlist = []
        for i in range(n):
            if s is None:
                val = random()
            else:
                val = s
            var = idf_func.subs(t, val)
            varlist.append(var)
        varlist.sort()
        return varlist


__all__ = ["UniformRV"]
