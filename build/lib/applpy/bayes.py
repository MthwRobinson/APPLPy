"""
Bayesian Statistics Module

Defines procedures for Bayesian parameter estimation

Bayesian Procedures:

Procedures:
    1. BayesUpdate(LikeRV,PriorRV,data,param)
    2. CredibleSet(X,alpha)
    3. JeffreysPrior(LikeRV,low,high,param)
    4. Posterior(LikeRV,PriorRV,data,param)
    5. PosteriorPredictive(LikeRV,PriorRV,data,param)
    
"""

from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, Add, Mul, Integer, function,
                   binomial, pprint, log)
from .rv import (RV, RVError, CDF, PDF, BootstrapRV,
                 ExpectedValue,Mean, Variance, Truncate)
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

    print 'Bayesian Statistics Procedures'
    print 'Posterior(X,Y,x,param), BayesUpdate(X,Y,Data,param)'
    print 'PosteriorPreidictive(X,Y,Data,param), TwoSample(X,Y,Data1,Data2)'
    print 'CS(m,s,alpha,n,type),Jeffreys(X,low,high,param)'
    print ""

def Posterior(LikeRV,PriorRV,data=[],param=Symbol('theta')):
    """
    Procedure Name: BayesUpdate
    Purpose: Derive a posterior distribution for a parameter
                given a likelihood function, a prior distribution and
                an observation
    Arguments:  1. LikeRV: The likelihood function (a random variable)
                2. PriorRV: A prior distribution (a random variable)
                3. data: a data observation
                4. param: the uknown parameter in the likelihood function
                    (a sympy symbol)
    Output:     1. PostRV: A posterior distribution
    """
    # If the unknown parameter is not a symbol, return an error
    if type(param)!=Symbol:
        raise RVError('the unknown parameter must be a symbol')
    if PriorRV.ftype[0]=='continuous':
        # Extract the likelihood function from the likelhood random
        #   variable
        likelihood=LikeRV.func[0].subs(x,data[0])
        for i in range(1,len(data)):    
            likelihood*=LikeRV.func[0].subs(x,data[i])
        likelihood=simplify(likelihood)
        likelihood=likelihood.subs(param,x)
        # Create a list of proportional posterior distributions
        FunctionList=[]
        for i in range(len(PriorRV.func)):
            # extract the prior distribution
            prior=PriorRV.func[i]
            # multiply by the likelihood function
            proppost=likelihood*prior
            # substitute the data observation
            proppost=simplify(proppost)
            # add to the function list
            FunctionList.append(proppost)
        PropPost=RV(FunctionList,PriorRV.support,['continuous','pdf'])
        # Normalize the posterior distribution
        PostRV=Truncate(PropPost,[PriorRV.support[0],PriorRV.support[-1]])
        return PostRV
    # If the prior distribution is discrete and the likelihood function
    #   is continuous, compute the posterior distribution
    if PriorRV.ftype[0]=='discrete' and LikeRV.fype[0]=='continuous':
        # Compute a distribution that is proportional to the posterior
        #   distribution
        List1=[]
        for i in range(len(PriorRV.support)):
            likelihood=LikeRV.func[0]
            likelihood=likelihood.subs(x,data)
            # Substitute each point that appears in the support of
            #   the prior distribution into the likelihood distribution
            subslike=likelihood.subs(param,PriorRV.support[i])
            prior=PriorRV.func[i]
            # Multiply the prior distribution by the likelihood function
            priorXlike=simplify(priorXsubslike)
            List1.append(priorXlike)
        # Find the marginal distribution
        marginal=sum(List1)
        # Find the posterior distribution by dividing each value
        #   in PriorXLike by the marginal distribution
        List2=[]
        for i in range(len(List1)):
            List2.append(List1[i]/marginal)
        PostRV=RV(List2,PriorRV.support,PriorRV.ftype)
        return PostRV
    # If the prior distribution and the likelihood function are both
    #   discrete, compute the posterior distribution
    if PriorRV.ftype[0]=='discrete' and LikeRV.ftype[0]=='discrete':
        # If the prior distribution and the likelihood function do not
        #   have the same sizes, return and error
        if len(PriorRV.func)!=len(LikeRV.func):
            string='the number of values in the prior distribution and'
            string+='likelihood function must be the same'
            raise RVError(string)
        # Multiply the prior distribution by the likelihood function
        priorXlike=[]
        for i in range(len(PriorRV.func)):
            val=PriorRV.func[i]*LikeRV.func[i]
            priorXlike.append(val)
        # Compute the marginal distribution to normalize the posterior
        k=sum(priorXlike)
        # Compute the posterior distribution
        posteriorlist=[]
        for i in range(len(priorXlike)):
            val=priorXlike/k
            posteriorlist.append(val)
        PostRV=RV(posteriorlist,PriorRV.support,PriorRV.ftype)
        return PostRV

def CredibleSet(PostRV,alpha):
    """
    Procedure Name: CredibleSet
    Purpose: Produce a credible set given a likelihood function
                and a confidence level
    Arguments:  1. PostRV: The distribution of the parameter
                2. alpha: the confidence level
    Output:     1. CredSet: a credible set in the form of a list
    """
    # If alpha is not between 0 and 1, return an error
    if alpha<0 or alpha>1:
        raise RVError('alpha must be between 0 and 1')
    # Computer the upper bound of the credible set
    lower=PostRV.variate(n=1,s=alpha/2)[0]
    # Compute the lower bound of the credible set
    upper=PostRV.variate(n=1,s=1-(alpha/2))[0]
    CredSet=[lower,upper]
    return CredSet

def JeffreysPrior(LikeRV,low,high,param):
    """
    Procedure Name: JeffreysPrior
    Purpose: Derive a Jeffreys Prior for a likelihood function
    Arguments:  1. LikeRV: The likelihood function (a random variable)
                2. low: the lower support
                3. high: the upper support
                4. param: the unknown parameter
    Output:     1. JeffRV: the Jeffreys Prior distribution
    """
    # If the likelihood function is continuous, compute the Jeffreys
    #   Prior
    if LikeRV.ftype[0]=='continuous':
        likelihood=LikeRV.func[0]
        loglike=ln(likelihood)
        logdiff=diff(loglike,param)
        jefffunc=sqrt(integrate(likelihood*logdiff**2,
                                (x,LikeRV.support[0],
                                 LikeRV.support[1])))
        jefffunc=simplify(jefffunc)
        jefffunc=jefffunc.subs(param,x)
        JeffRV=RV([jefffunc],[low,high],LikeRV.ftype)
        return JeffRV

# Old BayesUpdate code ... the Posterior procedure now computes
# the posterior distribution with only one integration. New
# code runs much faster
def BayesUpdate(LikeRV,PriorRV,data=[],param=Symbol('theta')):
    """
    Procedure Name: Posterior
    Purpose: Derive a posterior distribution for a parameter
                given a likelihood function, a prior distribution and
                a data set
    Arguments:  1. LikeRV: The likelihood function (a random variable)
                2. PriorRV: A prior distribution (a random variable)
                3. data: a data set
                4. param: the uknown parameter in the likelihood function
                    (a sympy symbol)
    Output:     1. PostRV: A posterior distribution
    """
    # Find the posterior distribution for the first observation
    PostRV=BayesUpdate(LikeRV,PriorRV,[data[0]],param)
    # If there are multiple observations, continue bayesian updating
    #   for each observation in the data set
    if len(data)>1:
        for i in range(1,len(data)):
            # Set the previous posterior distribution as the new
            #   prior distribution
            NewPrior=PostRV
            # Compute the new posterior distribution for the next
            #   observation in the data set
            PostRV=BayesUpdate(LikeRV,NewPrior,[data[i]],param)
    return PostRV

def PosteriorPredictive(LikeRV,PriorRV,data=[],param=Symbol('theta')):
    """
    Procedure Name: PosteriorPredictive
    Purpose: Derive a posterior predictive distribution to predict the next
                observation, given a likelihood function, a prior
                distribution and a data vector
    Arguments:  1. LikeRV: The likelihood function (a random variable)
                2. PriorRV: A prior distribution (a random variable)
                3. data: a data set
                4. param: the uknown parameter in the likelihood function
                    (a sympy symbol)
    Output:     1. PostPredRV: A posterior predictive distribution
    """
    # If the prior distribution is continuous, compute the posterior
    #   predictive distribution
    if PriorRV.ftype[0]=='continuous':
        # Compute the posterior distribution
        PostRV=Posterior(LikeRV,PriorRV,data,param)
        posteriorfunc=PostRV.func[0].subs(x,param)
        likelihoodfunc=LikeRV.func[0]
        postXlike=posteriorfunc*likelihoodfunc
        postpredict=integrate(postXlike,
                              (param,PriorRV.support[0],
                               PriorRV.support[1]))
        postpredict=simplify(postpredict)
        PostPredRV=RV([postpredict],LikeRV.support,LikeRV.ftype)
        return PostPredRV
