"""BernoulliRV distribution."""

from sympy import Symbol, symbols


from .binomial import BinomialRV


x, y, z, t, v = symbols("x y z t v")


class BernoulliRV(BinomialRV):
    """
    Procedure Name: BernoulliRV
    Purpose: Creates an instance of the bernoulli distribution
    Arguments:  1. p: a positive parameter between 0 and 1
    Output:     1. A bernoulli random variable
    """

    def __init__(self, p=Symbol("p", positive=True)):
        X_dummy = BinomialRV(1, p)
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["BernoulliRV"]
