from __future__ import division
import numpy as np
from rv import *

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
Discrete Random Variable Module

Defines procedures for operating on discrete random variables
using linear algebra

"""

"""
Procedures on One Random Variable

Procedures:
    1. MeanDiscrete(RVar)
    2. VarDiscrete(RVar)

"""

def MeanDiscrete(RVar):
    """
    Procedure Name: MeanDiscrete
    Purpose: Compute the mean of a discrete random variable
    Arguments:  1. RVar: A discrete random variable
    Output:     1. The mean of the random variable
    """
    # Check the random variable to make sure it is discrete
    if RVar.ftype[0]=='continuous':
        raise RVError('the random variable must be continuous')
    elif RVar.ftype[0]=='Discrete':
        try:
            RVar=Convert(RVar)
        except:
            err_string='the support of the random variable'
            err_string+=' must be finite'
            raise RVError(err_string)
    # Convert the random variable to PDF form
    X_dummy=PDF(RVar)
    # Convert the value and the support of the pdf to numpy
    #   matrices
    support=np.matrix(X_dummy.support)
    pdf=np.matrix(X_dummy.func)
    # Use the numpy element wise multiplication function to
    #   determine a vector of the values of f(x)*x
    vals=np.multiply(support,pdf)
    # Sum the values of f(x)*x to find the mean
    meanval=vals.sum()
    return meanval
    
def VarDiscrete(RVar):
    """
    Procedure Name: VarDiscrete
    Purpose: Compute the variance of a discrete random variable
    Arguments:  1. RVar: a discrete random variable
    Output:     1. The variance of the random variable
    """
    # Check the random variable to make sure it is discrete
    if RVar.ftype[0]=='continuous':
        raise RVError('the random variable must be continuous')
    elif RVar.ftype[0]=='Discrete':
        try:
            RVar=Convert(RVar)
        except:
            err_string='the support of the random variable'
            err_string+=' must be finite'
            raise RVError(err_string)
    # Convert the random variable to PDF form
    X_dummy=PDF(RVar)
    # Mind the mean of the random variable
    EX=MeanDiscrete(RVar)
    # Convert the values and support of the random variable
    #   to vector form
    support=np.matrix(RVar.support)
    pdf=np.matrix(RVar.func)
    # Find E(X^2) by creating a vector containing the values
    #   of f(x)*x**2 and summing the result
    supportsqr=np.multiply(support,support)
    EXXvals=np.multiply(supportsqr,pdf)
    EXX=EXXvals.sum()
    # Find Var(X)=E(X^2)-E(X)^2
    var=EXX-(EX**2)
    return var
    
    
    

























