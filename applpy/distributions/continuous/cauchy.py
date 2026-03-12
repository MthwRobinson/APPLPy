"""CauchyRV distribution."""

from random import random

from sympy import (
    Symbol,
    cot,
    oo,
    pi,
    symbols,
)

from ...rv import IDF, RV, RVError
from .param_check import param_check

x, y, z, t, v = symbols("x y z t v")


class CauchyRV(RV):
    """
    Procedure Name: CauchyRV
    Purpose: Creates an instance of the Cauchy distribution
    Arguments:  1. a: a real valued parameter
                2. alpha: a stictly positive parameter
    Output:     1. A Cauchy random variable
    """

    def __init__(self, a=Symbol("a"), alpha=Symbol("alpha"), positive=True):
        if a in [-oo, oo]:
            err_string = "Both parameters must be finite"
            if alpha in [-oo, oo]:
                raise RVError(err_string)
        if not isinstance(alpha, Symbol):
            if alpha <= 0:
                err_string = "alpha must be positive"
                raise RVError(err_string)
        X_dummy = RV(
            (1) / (alpha * pi * (1 + ((x - a) ** 2 / alpha**2))), [-oo, oo], ["continuous", "pdf"]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [a, alpha]
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

        # Generate cauchy variates
        idf_func = self.parameter[0] - cot(pi * t) * self.parameter[1]
        varlist = []
        for i in range(n):
            if s is None:
                val = random()
            else:
                val = s
            var = idf_func.subs(t, val).evalf()
            varlist.append(var)
        varlist.sort()
        return varlist


__all__ = ["CauchyRV"]
