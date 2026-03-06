"""Compatibility module for APPLPy distribution classes.

Continuous and discrete distributions now live under:
- applpy.distributions.continuous
- applpy.distributions.discrete

This module preserves the historical import path.
"""

from sympy import sqrt, symbols

from .bivariate import BivariateRV
from .distributions.continuous import *
from .distributions.continuous import __all__ as _continuous_all
from .distributions.discrete import *
from .distributions.discrete import __all__ as _discrete_all

x, y, z, t, v = symbols("x y z t v")


class ExampleRV(BivariateRV):
    def __init__(self):
        X_dummy = BivariateRV([(21 / 4) * x**2 * y], [[1 - y, y - sqrt(x)]], ["continuous", "pdf"])
        self.func = X_dummy.func
        self.constraints = X_dummy.constraints
        self.ftype = X_dummy.ftype


__all__ = _continuous_all + _discrete_all + ["ExampleRV"]
