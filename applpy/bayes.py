from __future__ import division
from sympy import *
from rv import *
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
Bayesian Statistics Module

Defines procedures for Bayesian parameter estimation

"""

"""
Bayesian Procedures:

Procedures:
    1. BayesUpdate(LikeRV,PriorRV,data,param)

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

def BayesUpdate(LikeRV,PriorRV,data=[],param=Symbol('theta')):
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
    # If -oo or oo is in the support of either random variable,
    #   return an error
    if LikeRV.ftype[0]=='Discrete':
        if max(LikeRV.support)==oo or min(LikeRV.support)==-oo:
            string='discrete RVs with infinite supports are not'
            string+='supported'
            raise RVError(string)
    if PriorRV.ftype[0]=='Discrete':
        if max(PriorRV.support)==oo or min(PriorRV.support)==-oo:
            string='discrete RVs with infinite supports are not'
            string+='supported'
            raise RVError(string)
    # If the prior distribution is continuous, compute the posterior
    #   distribution
    if PriorRV.ftype[0]=='continuous':
        # Extract the likelihood function from the likelhood random
        #   variable
        likelihood=LikeRV.func[0]
        # Create a list of proportional posterior distributions
        FunctionList=[]
        for i in range(len(PriorRV.func)):
            # extract the prior distribution
            prior=PriorRV.func[i]
            # multiply by the likelihood function
            proppost=likelihood*prior
            # substitute the data observation
            proppost=proppost.subs(x,data)
            proppost=simplify(proppost)
            # add to the function list
            FunctionList.append(proppost)
        # Find the normalizing constant
        k=0
        for i in range(len(PriorRV.func)):
            # Find the area under the curve for each segment
            #   and add to the total
            segment=integrate(PriorRV.func[i],
                              (param,PriorRV.support[i],
                               PriorRV.support[i+1]))
            k+=segment
        k=k.subs(x,data)
        # Divide each segment of the function list by the normalizing
        #   constant to find the posterior distribution
        FinalList=[]
        for i in range(len(FunctionList)):
            finalfunc=FunctionList[i]/k
            finalfunc=simplify(finalfunc)
            FinalList.append(finalfunc)
        # Convert the list of posterior distributions to RV form
        PostRV=RV(FinalList,PriorRV.support,['continuous','pdf'])
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

  
        
        
            
        
        
    





     















