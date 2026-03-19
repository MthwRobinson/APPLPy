"""NormalRV distribution."""

from random import random

from sympy import (
    Symbol,
    cos,
    exp,
    ln,
    oo,
    pi,
    sqrt,
    symbols,
)

from ...rv import idf, RV, RVError
from .param_check import param_check
from .uniform import UniformRV

x, y, z, t, v = symbols("x y z t v")


class NormalRV(RV):
    """
    Procedure Name: NormalRV
    Purpose: Creates an instance of the normal distribution
    Arguments:  1. mu: a real valued parameter
                2. sigma: a strictly positive parameter
    Output:     1. A normal random variable
    """

    def __init__(self, mu=Symbol("mu"), sigma=Symbol("sigma", positive=True)):
        if not isinstance(sigma, Symbol):
            if sigma <= 0:
                err_string = "sigma must be positive"
                raise RVError(err_string)
        if sigma in [-oo, oo] or mu in [-oo, oo]:
            err_string = "both parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            (exp((-((x - mu) ** 2)) / (2 * sigma**2)) * sqrt(2)) / (2 * sigma * sqrt(pi)), [-oo, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.parameter = [mu, sigma]
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
        #   the idf function
        if method == "inverse":
            Xidf = idf(self)
            varlist = [idf(Xidf, random()) for i in range(1, n + 1)]
            return varlist

        if s is not None and n == 1:
            return [idf(self, s)]

        # Otherwise, use the Box-Muller method to compute variates
        mean = self.parameter[0]
        var = self.parameter[1]
        U = UniformRV(0, 1)

        def Z1(val1_val2):
            return sqrt(-2 * ln(val1_val2[0])) * cos(2 * pi * val1_val2[1]).evalf()

        def gen_uniform(x):
            return U.variate(n=1)[0]

        val_pairs = [(gen_uniform(1), gen_uniform(1)) for i in range(1, n + 1)]
        varlist = [Z1(pair) for pair in val_pairs]
        normlist = [(mean + sqrt(var) * val).evalf() for val in varlist]
        return normlist


__all__ = ["NormalRV"]
