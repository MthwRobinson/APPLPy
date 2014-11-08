from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, plot, Add, Mul, Integer, function,
                   binomial, pprint, nsolve)
from random import random
from .rv import (RV, RVError, CDF, PDF, BootstrapRV,
                 ExpectedValue)
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

"""
Statistics Module

Defines procedures for parameter estimation

"""



"""
Parameter Estimation Procedures

Procedures:
    1. KSTest(RVar,data)
    1. MOM(RVar,data,parameters)
    2. MLE(RVar,data,parameters,censor)
"""

def KSTest(RVar,data):
    """
    Procedure Name: KSTest
    Purpose: Calculates the Kolmogorov-Smirnoff test statistic
                for the empirical CDF of the sample data versus
                the CDF of a fitted distribution with random
                variable X
    Arguments:  1. RVar: A random variable model
                2. data: A data sample in list format
    Output:     1. The Kolmogorov-Smirnoff test statistics
    """
    # Create an empirical CDF from the data sample
    EmpCDF=CDF(BootstrapRV(data))
    m=len(EmpCDF.support)
    # Compute fitted CDF values
    FX=CDF(RVar)
    FittedCDFValue=[]
    for i in EmpCDF.support:
        FittedCDFValue.append(CDF(FX,i).evalf())
    # Compute the KS test statistic
    KS=0
    for i in range(m-1):
        Dpos=abs(EmpCDF.func[i+1]-FittedCDFValue[i]).evalf()
        Dneg=abs(FittedCDFValue[i]-EmpCDF.func[i]).evalf()
        KS=max(max(KS,Dpos),Dneg)
    KS=max(KS,abs(FittedCDFValue[m-1]).evalf())
    return KS
        
           
def MOM(RVar,data,parameters,guess=None,numeric=False):
    """
    Procedure Name: MLE
    Purpose: Estimates parameters using the method of moments
    Arguments:  1. RVar: A random variable model
                2. data: The data sample
                3. parameters: The list of parameters to estimate
                4. guess: An initial guess for the unknown parameters,
                    required if numerical methods are being used
                5. numeric: A binary variable. If True, MOM will attempt
                    to solve for unknown parameters using numerical
                    methods
    Output:     1. The estimates in dictionary form
    """

    # Convert the random variable to pdf form
    fx=PDF(RVar)
    # Creat a bootstrap random variable from the sample
    xstar=BootstrapRV(data)
    # Create a list of equations to solve
    soln_eqn=[]
    for i in range(len(parameters)):
        val=ExpectedValue(xstar,x**(i+1))
        expect=ExpectedValue(fx,x**(i+1))
        soln_eqn.append(val-expect)
    # Create a list of solutions
    if numeric==False:
        try:
            soln=solve(soln_eqn,set(parameters))
        except:
            err_string='MOM failed to solve for the parameters,'
            err_string+=' please try numerical MOM'
            raise RVError(err_string)
    elif numeric==True:
        if guess==None:
            err_string='an initial guess must be entered to'
            err_string+=' solve MLE numerically'
            raise RVError(err_string)
        soln_tup=tuple(soln_eqn)
        param_tup=tuple(parameters)
        guess_tup=tuple(guess)
        soln=nsolve(soln_tup,param_tup,guess_tup)    
            
    return soln
        

def MLE(RVar,data,parameters,guess=None,numeric=False,censor=None):
    """
    Procedure Name: MLE
    Purpose: Estimates parameters using maximum likelihood estimation
    Arguments:  1. RVar: A random variable model
                2. data: The data sample
                3. parameters: The parameters to be estimated
                4. censor: A binary list of 0's and 1's where 1
                    indicates an observed value and 0 indicates
                    a right censored value
                5. guess: An initial guess for the unknown parameters,
                    required if numerical methods are being used
                6. numeric: A binary variable. If True, MOM will attempt
                    to solve for unknown parameters using numerical
                    methods
    Output:     1. A list of parameter estimates
    """

    # Return an error message if the distribution is piece-wiwse
    if len(RVar.func)!=1:
        raise RVError('MLE does not accept piecewise models')
    # Convert the random variable to its PDF form
    fx=PDF(RVar)   
    if censor==None:
        LogLike=0
        for i in range(len(data)):
            func=ln(fx.func[0])
            LogLike+=func.subs(x,data[i])
    # Otherwise, use the given value as a censor
    elif censor!=None:
        # Check to make sure the list contains only 1's and
        #   0's
        for i in range(len(censor)):
            if censor[i] not in [0,1]:
                return RVError('Censor may contain only 1s and 0s')
        # Check to make sure the censor list is the same
        #   length as the data list
        if len(censor)!=len(data):
            return RVError('Data and censor must be the same length')
        hx=HF(RVar)
        chx=CHF(RVar)
        # Split up the sample data into two lists, censored
        #   and uncensored
        censored=[]
        uncensored=[]
        for i in range(len(data)):
            if censor[i]==1:
                uncensored.append(data[i])
            elif censor[i]==0:
                censored.append(data[i])
        # Compute and simplify the log-likelihood function
        Logh=0
        Sumch=0
        for i in range(len(uncensored)):
            func=ln(hx.func[0])
            Logh+=func.subs(x,uncensored[i])
        for i in range(len(data)):
            func=ln(chx.func[0])
            Sumch+=func.subs(x,data[i])
        LogLike=simplify(Logh-Sumch)
    # Differentiate the log likelihood function with respect to
    #   each parameter and equate to 0
    DiffLogLike=[]
    for i in range(len(parameters)):
        func=diff(LogLike,parameters[i])
        DiffLogLike.append(simplify(func))
    # Solve for each parameter
    if numeric==False:
        try:
            soln=solve(DiffLogLike,set(parameters))
        except:
            err_string='MLE failed to solve for the parameters, '
            err_string+='please try the numeric MLE method'
            raise RVError(err_string)
    elif numeric==True:
        if guess==None:
            err_string='an initial guess must be entered to'
            err_string+=' solve MLE numerically'
            raise RVError(err_string)
        diff_tup=tuple(DiffLogLike)
        param_tup=tuple(parameters)
        guess_tup=tuple(guess)
        soln=nsolve(diff_tup,param_tup,guess_tup)
    return soln
