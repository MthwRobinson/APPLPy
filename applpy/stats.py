"""
Statistics Module

Defines procedures for parameter estimation

Procedures:
    1. KSTest(RVar,data)
    2. MOM(RVar,data,parameters)
    3. MLE(RVar,data,parameters,censor)
    4. MLEExponential(data)
    5. MLENormal(data,mu,sigma)
    6. MLEPoisson(data)
    7. MLEWeibull(data,censor)
"""

from sympy import (
    symbols,
    diff,
    sqrt,
    ln,
    simplify,
    solve,
    nsolve,
    log,
)
from .conversion import cdf, chf, hf, pdf
from .rv import RVError, BootstrapRV, expected_value, mean, variance

x, y, z, t = symbols("x y z t")

"""
    A Probability Progamming Language (APPL) -- Python Edition
    Copyright (C) 2001,2002,2008,2010,2014 Andrew Glen, Larry
    Leemis, Diane Evans, Matthew Robinson

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


def KSTest(RVar, data):
    """
    Procedure Name: KSTest
    Purpose: Calculates the Kolmogorov-Smirnoff test statistic
                for the empirical cdf of the sample data versus
                the cdf of a fitted distribution with random
                variable X
    Arguments:  1. RVar: A random variable model
                2. data: A data sample in list format
    Output:     1. The Kolmogorov-Smirnoff test statistics
    """
    # Create an empirical cdf from the data sample
    EmpCDF = cdf(BootstrapRV(data))
    m = len(EmpCDF.support)
    # Compute fitted cdf values
    FX = cdf(RVar)
    FittedCDFValue = []
    for i in EmpCDF.support:
        FittedCDFValue.append(cdf(FX, i).evalf())
    # Compute the KS test statistic
    KS = 0
    for i in range(m - 1):
        Dpos = abs(EmpCDF.func[i + 1] - FittedCDFValue[i]).evalf()
        Dneg = abs(FittedCDFValue[i] - EmpCDF.func[i]).evalf()
        KS = max(max(KS, Dpos), Dneg)
    KS = max(KS, abs(FittedCDFValue[m - 1]).evalf())
    return KS


def MOM(RVar, data, parameters, guess=None, numeric=False):
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
    fx = pdf(RVar)
    # Creat a bootstrap random variable from the sample
    xstar = BootstrapRV(data)
    # Create a list of equations to solve
    soln_eqn = []
    for i in range(len(parameters)):
        val = expected_value(xstar, x ** (i + 1))
        expect = expected_value(fx, x ** (i + 1))
        soln_eqn.append(val - expect)
    # Create a list of solutions
    if not numeric:
        try:
            soln = solve(soln_eqn, set(parameters))
        except Exception:
            err_string = "MOM failed to solve for the parameters,"
            err_string += " please try numerical MOM"
            raise RVError(err_string)
    elif numeric:
        if guess is None:
            err_string = "an initial guess must be entered to"
            err_string += " solve MLE numerically"
            raise RVError(err_string)
        soln_tup = tuple(soln_eqn)
        param_tup = tuple(parameters)
        guess_tup = tuple(guess)
        soln = nsolve(soln_tup, param_tup, guess_tup)

    return soln


def MLE(RVar, data, parameters, guess=None, numeric=False, censor=None):
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
                6. numeric: A binary variable. If True, MLE will attempt
                    to solve for unknown parameters using numerical
                    methods
    Output:     1. A list of parameter estimates
    """

    # Return an error message if the distribution is piece-wiwse
    if len(RVar.func) != 1:
        raise RVError("MLE does not accept piecewise models")
    # If the random variable has a hard-coded MLE procedure, use
    #   the corresponding procedure
    if RVar.__class__.__name__ == "NormalRV":
        if censor is None:
            if len(parameters) == 2:
                return MLENormal(data)
            if len(parameters) == 1:
                string = "MLE is estimating mu and sigma parameters"
                string += " for the Normal distribution"
                print(string)
                return MLENormal(data)
    if RVar.__class__.__name__ == "ExponentialRV":
        return MLEExponential(data)
    if RVar.__class__.__name__ == "WeibullRV":
        return MLEWeibull(data, censor)
    if RVar.__class__.__name__ == "PoissonRV":
        return MLEPoisson(data)
    # Convert the random variable to its pdf form
    fx = pdf(RVar)
    if censor is None:
        LogLike = 0
        for i in range(len(data)):
            func = ln(fx.func[0])
            LogLike += func.subs(x, data[i])
    # Otherwise, use the given value as a censor
    elif censor is not None:
        # Check to make sure the list contains only 1's and
        #   0's
        for i in range(len(censor)):
            if censor[i] not in [0, 1]:
                return RVError("Censor may contain only 1s and 0s")
        # Check to make sure the censor list is the same
        #   length as the data list
        if len(censor) != len(data):
            return RVError("Data and censor must be the same length")
        hx = hf(RVar)
        chx = chf(RVar)
        # Split up the sample data into two lists, censored
        #   and uncensored
        censored = []
        uncensored = []
        for i in range(len(data)):
            if censor[i] == 1:
                uncensored.append(data[i])
            elif censor[i] == 0:
                censored.append(data[i])
        # Compute and simplify the log-likelihood function
        Logh = 0
        Sumch = 0
        for i in range(len(uncensored)):
            func = ln(hx.func[0])
            Logh += func.subs(x, uncensored[i])
        for i in range(len(data)):
            func = ln(chx.func[0])
            Sumch += func.subs(x, data[i])
        LogLike = simplify(Logh - Sumch)
    # Differentiate the log likelihood function with respect to
    #   each parameter and equate to 0
    DiffLogLike = []
    for i in range(len(parameters)):
        func = diff(LogLike, parameters[i])
        DiffLogLike.append(simplify(func))
    # Solve for each parameter
    if not numeric:
        try:
            soln = solve(DiffLogLike, set(parameters))
        except Exception:
            err_string = "MLE failed to solve for the parameters, "
            err_string += "please try the numeric MLE method"
            raise RVError(err_string)
    elif numeric:
        if guess is None:
            err_string = "an initial guess must be entered to"
            err_string += " solve MLE numerically"
            raise RVError(err_string)
        diff_tup = tuple(DiffLogLike)
        param_tup = tuple(parameters)
        guess_tup = tuple(guess)
        soln = nsolve(diff_tup, param_tup, guess_tup)
    return soln


def MLEExponential(data):
    """
    Procedure Name: MLEExponential
    Purpose: Conduct maximul likelihood estimation on an
                exponential distribution
    Input:  1. data: a data set
    Output: 1. soln: an estimation for the unknown parameter
    """
    Xstar = BootstrapRV(data)
    theta = 1 / mean(Xstar)
    soln = [theta]
    return soln


def MLENormal(data, mu=None, sigma=None):
    """
    Procedure Name: MLENormal
    Purpose: Conduct maximum likelihood estimation on a normal
                distribution with at least one unknown parameter
    Input:  1. data: a data set
            2. mu: an optional parameter that holds mu constant. If a
                value is entered, MLENormal will estimate sigma given
                a fixed mu
            3. sigma: an optional parameter that holds sigma constant
    Output: 1. soln: a list of estimates for the unknown parameters
                in the form [mu,sigma]
    """
    Xstar = BootstrapRV(data)
    if mu is None:
        mu = mean(Xstar)
    if sigma is None:
        sigma = sqrt(variance(Xstar))
    soln = [mu, sigma]
    return soln


def MLEPoisson(data):
    """
    Procedure Name: MLEPoisson
    Purpose: Conduct maximum likelihood estimation for the Poisson
                distribution
    Input:  1. data: a data set
    Output: 1. soln: a list of estimates for the unknown parameter
                in the form [theta]
    """
    Xstar = BootstrapRV(data)
    meanX = mean(Xstar)
    soln = [meanX]
    return soln


def MLEWeibull(data, censor=None):
    """
    Procedure Name: MLEWeibull
    Purpose: Conduct maximum likelihood estimation for the Weibull
                distribution with arbitrary right censor
    Input:  1. data: a data set
            2. censor: a indicator list where 1 is an observed data
                point and 0 is an unobserved data point
    Output: 1. soln: a list of estimates for the unknown parameters
                in the form [theta,kappa]
    """
    # If a list of right censored values is not provided, set
    #   the right censor list to contain all 1's, indicating
    #   that every value was observed
    n = len(data)
    if censor is not None:
        Delta = censor
    else:
        Delta = [1 for obs in data]
    # Set tolerance and initial estimate
    epsilon = 0.000000001
    c = 1
    # Compute the number of observed failures
    r = sum(Delta)

    # Calculate s1
    s1 = 0
    for i in range(n):
        if Delta[i] == 1:
            s1 += log(data[i])
    # Calculate s2 (beginning of random censoring adjustment)
    s2 = 0
    for i in range(n):
        s2 += data[i] ** c
    # Calculate s3
    s3 = 0
    for i in range(n):
        s3 += data[i] ** c * log(data[i])
    while r * s3 - s1 * s2 <= 0:
        c = c * 1.1
        s2 = 0
        for i in range(n):
            s2 += data[i] ** c
        s3 = 0
        for i in range(n):
            s3 += data[i] ** c * log(data[i])
    # Calculate s2 (beginning of first iteration of while loop)
    s2 = 0
    for i in range(n):
        s2 += data[i] ** c
    # Calculate s3
    s3 = 0
    for i in range(n):
        s3 += data[i] ** c * log(data[i])

    # Calculate q and c
    q = r * s2 / (r * s3 - s1 * s2)
    c = (c + q) / 2
    counter = 0
    while abs(c - q) > epsilon and counter < 100:
        counter += 1
        s2 = 0
        for i in range(n):
            s2 += data[i] ** c
        s3 = 0
        for i in range(n):
            s3 += data[i] ** c * log(data[i])
        q = r * s2 / (r * s3 - s1 * s2)
        c = (c + q) / 2

    # Calculate the MLEs
    chat = c
    s2 = 0
    for i in range(n):
        s2 += data[i] ** c
    bhat = (s2 / r) ** (1 / c)
    soln = [1 / bhat, chat]
    return soln
