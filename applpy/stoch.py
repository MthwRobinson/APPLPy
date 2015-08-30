"""
Stocastic Processes Module

1. The Markov Chain Class

from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, Add, Mul, Integer, function,
                   binomial, pprint,log,expand,zoo,latex,Piecewise,Rational,
                   Sum,S,Float)
"""

from sympy.plotting.plot import plot
from random import random
import numpy as np
import plot as plt
import pylab as pyplt
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

class MarkovChain:
    """
    Markov Chain Class
    Defines the data structure for APPLPy Markov Chains
    Defines procedures relating to APPLPy Markov Chains
    """

    def __init__(self,P,init):
        """
        Procedure Name: __init__
        Purpose: Initializes and instance of the Markov Chain class
        Arguments:  1. P: the transition matrix of the markov chain
                    2. init: the initial distribution for the markov chain
        Output:     1. And instance of the Markov Chain class
        """
        # Check to make 











    
