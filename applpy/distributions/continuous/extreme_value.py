"""ExtremeValueRV distribution."""

from random import random

from sympy import (
    Symbol,
    exp,
    ln,
    oo,
    symbols,
)

from ...rv import IDF, RV, RVError
from .param_check import param_check

x, y, z, t, v = symbols("x y z t v")


class ExtremeValueRV(RV):
    """
    Procedure Name: ExtremeValueRV
    Purpose: Creates an instance of the extreme value distribution
    Arguments:  1. alpha: a real valued parameter
                2. beta: a real valued parameter
    Output:     1. An extreme value random variable
    """

    def __init__(self, alpha=Symbol("alpha"), beta=Symbol("beta")):
        if alpha in [-oo, oo]:
            if beta in [-oo, oo]:
                err_string = "both parameters must be finite"
                raise RVError(err_string)
        X_dummy = RV(
            (beta * exp((x * beta) - ((exp(x * beta)) / alpha))) / alpha,
            [-oo, oo],
            ["continuous", "pdf"],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [alpha, beta]
        self.cache = {}

    def variate(self, n=1, s=None, method="special"):
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

        idf_func = (ln(self.parameter[0]) + ln(ln(-1 / (t - 1)))) / self.parameter[1]
        varlist = []
        for i in range(n):
            if s is None:
                val = random()
            else:
                val = s
            var = idf_func.subs(t, val)
            varlist.append(var)
        return varlist


__all__ = ["ExtremeValueRV"]
