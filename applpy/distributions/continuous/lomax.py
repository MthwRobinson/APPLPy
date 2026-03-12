"""LomaxRV distribution."""

from random import random

from sympy import (
    Symbol,
    oo,
    symbols,
)

from ...rv import IDF, RV, RVError
from .param_check import param_check

x, y, z, t, v = symbols("x y z t v")


class LomaxRV(RV):
    """
    Procedure Name: LomaxRV
    Purpose: Creates an instance of the lomax distribution
    Arguments:  1. kappa: a strictly positive parameter
                2. theta: a strictly positive parameter
    Output:     1. A lomax random variable
    """

    def __init__(self, kappa=Symbol("kappa", positive=True), theta=Symbol("theta", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(kappa, Symbol):
            if not isinstance(theta, Symbol):
                if kappa <= 0 or theta <= 0:
                    err_string = "both parameters must be positive"
                    raise RVError(err_string)
        if kappa in [-oo, oo] or theta in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            [theta * kappa * (1 + theta * x) ** (-kappa - 1)], [0, oo], ["continuous", "pdf"]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameters = [kappa, theta]
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

        idf_func = ((1 - t) ** (1 / self.parameter[0]) - 1) / self.parameter[1]
        varlist = []
        for i in range(n):
            if s is None:
                val = random()
            else:
                val = s
            var = idf_func.subs(t, val)
            varlist.append(var)
        return varlist


__all__ = ["LomaxRV"]
