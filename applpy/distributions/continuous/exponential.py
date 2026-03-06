"""ExponentialRV distribution."""

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


class ExponentialRV(RV):
    """
    Procedure Name: ExponentialRV
    Purpose: Creates an instance of the exponential distribution
    Arguments:  1. theta: a strictly positive parameter
    Output:     1. An exponential random variable
    """

    def __init__(self, theta=Symbol("theta", positive=True)):
        # x = Symbol('x', positive = True)
        if not isinstance(theta, Symbol):
            if theta <= 0:
                err_string = "theta must be positive"
                raise RVError(err_string)
        if theta in [-oo, oo]:
            err_string = "theta must be finite"
            raise RVError(err_string)
        X_dummy = RV([theta * exp(-theta * x)], [0, oo])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [theta]
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

        # Generate exponential variates
        idf_func = (-ln(1 - t)) / (self.parameter[0])
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


__all__ = ["ExponentialRV"]
