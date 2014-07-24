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


"""
Bayesian Statistics Module

Defines procedures for Bayesian parameter estimation

"""

from rv import *

"""
Bayesian Procedures:

Procedures:
    1.

"""

def BayesMenu():
    print 'ApplPy Procedures'
    print ""
    print 'Procedure Notation'
    print ""
    print 'X is a likelihood function'
    print 'Y is a prior distribution'
    print 'x is an observed data point'
    print 'Data is an observed set of data'
    print 'entered as a list --> ex. Data=[1,12.4,34,.52.45,64]'
    print 'low and high are numeric'
    print ""
    print ""

    print 'Functional Form Conversion'
    print 'Posterior(X,Y,x,param), BayesUpdate(X,Y,Data,param)'
    print 'PosteriorPreidictive(X,Y,Data,param), TwoSample(X,Y,Data1,Data2)'
    print 'CS(m,s,alpha,n,type),Jeffreys(X,low,high,param)'
    print ""


    
