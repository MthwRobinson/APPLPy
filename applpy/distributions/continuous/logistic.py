"""LogisticRV distribution."""

from random import random

from sympy import (
    Symbol,
    exp,
    ln,
    oo,
    symbols,
)

from ...conversion import idf
from ...rv import RV, RVError
from .param_check import param_check

x, y, z, t, v = symbols("x y z t v")


class LogisticRV(RV):
    """
    Procedure Name: LogisticRV
    Purpose: Creates an instance of the logistic distribution
    Arguments:  1. kappa: a strictly positive parameter
                2. theta: a strictly positive parameter
    Output:     1. A logistic random variable
    """

    def __init__(self, kappa=Symbol("kappa", positive=True), theta=Symbol("theta", positive=True)):
        if not isinstance(kappa, Symbol):
            if not isinstance(theta, Symbol):
                if kappa <= 0 or theta <= 0:
                    err_string = "both parameters must be positive"
                    raise RVError(err_string)
        if kappa in [-oo, oo] or theta in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            [(theta ** (kappa) * kappa * exp(kappa * x)) / (1 + (theta * exp(x)) ** kappa) ** 2],
            [-oo, oo],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [kappa, theta]
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
        #   the idf function
        if method == "inverse":
            Xidf = idf(self)
            varlist = [idf(Xidf, random()) for i in range(1, n + 1)]
            return varlist

        idf_func = -(
            (ln(-t / (t - 1)) + self.parameter[0] * ln(self.parameter[1])) / self.parameter[1]
        )
        varlist = []
        for i in range(n):
            if s is None:
                val = random()
            else:
                val = s
            var = idf_func.subs(t, val)
            varlist.append(var)
        return varlist


__all__ = ["LogisticRV"]
