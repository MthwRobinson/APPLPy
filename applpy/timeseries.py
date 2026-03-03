"""
Time Series Module

Defines an extension for computing exact distributions for auto-regressive,
    moving average and other stochastic processes

The algorithms implemented in this module were developed by
   Keith Webb and originally implemented in Maple

Procedures:
    1.
"""

from sympy import (
    symbols,
)

x, y, z, t, v = symbols("x y z t v")

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
