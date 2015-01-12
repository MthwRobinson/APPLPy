"""
Bivariate Extension Module

1. Defines the Bivariate Random Variable Class
2. Computation of expected values
3. Computation of marginal distributions
4. Transformation of bivariate random variables

The algorithms implemented in this module were developed by
    Jeff Yang, John Drew and Larry Leemis and originally implemented in Maple

2012. J Yang, L Leemis and J Drew. 'Automating Bivariate Transformations'.
    INFORMS Journal on Computing Vol. 24, No. 1.
"""
from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, Add, Mul, Integer, function,
                   binomial, pprint,log,expand,zoo,latex,Piecewise)
from sympy.plotting.plot import plot
from random import random
import numpy as np
import plot as plt
import pylab as pyplt
from .rv import RV, RVError
x,y,z,t=symbols('x y z t')

"""
    A Probability Progamming Language (APPL) -- Python Edition
    Copyright (C) 2001,2002,2008,2010,2014 Andrew Glen, Larry
    Leemis, Diane Evans, Matthew Robinson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

class BivariateRV:
    """
    BivariateRV Class:
    Defines the data structure for bivariate random variables
    Defines special procedures for bivariate random variables
    """

    def __init__(self,func,constraints,ftype=['continuous','pdf']):
        """
        Creates an instance of the bivariate random variable class

        Data Structure:
            self.func: a list of the functions f(x,y) for the random variable
            self.constraints: a list of constraints for the random variable.
                The list of constraints must satisfy the following conditions:
                    1. The constraints must be entered in adjacent order;
                        clockwise or counterclockwise is acceptable
                    2. The constraints must completely enclose a region
                    3. The constraints must be entered as strictly inequalities
                        in the form 0<f(x,y). For instance, x**2<sqrt(y) would
                        be entered as sqrt(y)-x**2.
                    4. Except for constraints inthe form x<a or x>a, each
                        constraint must pass the vertical line test. i.e. There
                        should only be one y value associated with each x value.
        """



















