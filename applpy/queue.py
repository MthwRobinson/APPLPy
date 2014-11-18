from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, plot, Add, Mul, Integer, function,
                   binomial)
from sympy.mpmath import (nsum,nprod)
from random import random
from .rv import (RV, RVError, CDF, CHF, HF, IDF, IDF, PDF, SF,
                 BootstrapRV, Convert, Mean, Convolution, Mixture)
from .dist_type import (ErlangRV, ExponentialRV)
x,y,z,t,v=symbols('x y z t v')

"""
    A Probability Progamming Language (APPL) -- Python Edition
    Copyright (C) 2001,2002,2008,2010,2014 Andrew Glen, Larry
    Leemis, Diane Evans, Matthew Robinson, William Kaczynski

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
Queuing Module

Defines an extension for computing the sojourn time distribution for the
    nth customer in an M/M/s queue

The algorithms implemented in this module were developed by
    William Kaczynski and originally implemented in Maple

2012. W Kaczynski, L Leemis and J Drew. 'Transient Queuing Analysis'.
    INFORMS Journal on Computing Vol. 24, No. 1.

"""

def QueueMenu():
    print 'ApplPy Procedures'
    print ""
    print 'Procedure Notation'
    print ""
    print 'X is the distribution of the time between arrivals.'
    print 'Y is the service time distribution.'
    print 'n is the total number of customers in the system.'
    print 'k is the number of customers in the system as time 0'
    print 's is the number of identical parallel servers'
    print 'a is the first customer of interest'
    print 'b is the second customer of interest (a<b)'
    print ""
    print ""

    print 'Queue Procedures'
    print 'Queue(X,Y,n,k,s), Cov(X,Y,a,b), kCov(X,Y,a,b,n,k)'
    print ""


"""
Procedures:
    1. Queue(X,Y,n,k,s)
"""

def Queue(X,Y,n,k,s):
    """
    Procedure Name: Queue
    Purpose: Computes the sojourn time distribution for the nth
                customer in an M/M/s queue, given k customers are
                in the system at time 0.
    Arguments:  1. X: the distribution of the time between arrivals
                        (must be an ExponentialRV)
                2. Y: the service time distribution
                        (must be an ExponentialRV)
                3. n: the total number of customers in the system
                4. k: the number of customers in the system at time 0
                5. s: the number of identical parallel servers
    Output:   
    """
    rho=Symbol('rho')
    rho_subs=(1/Mean(X))/(s*(1/Mean(Y)))
    lst=BuildDist(X,Y,n,k,s)
    probs=MMSQprob(n,k,s)
    # Substitute the value of rho into the probability list
    sub_probs=[]
    for element in probs:
        sub_element=element.subs(rho,rho_subs)
        sub_probs.append(sub_element)
    TIS=Mixture(sub_probs,lst)
    return TIS
    

'''
The following procedures are used to build the Queue, Cov and kCov
    procedures. They are not intended for end use for the user. _Q
    does not import into the APPLPy namespace to avoid conflict with
    the sympy.assumptions procedure Q.
'''

"""
Procedures:
    1. BuildDist(X,Y,n,k,s)
    1. MMSQprob(n,k,s)
    2. _Q(n,i,k,s)
"""

def BuildDist(X,Y,n,k,s):
    """
    Procedure Name: BuildDist
    Purpose: Creates the appropriate conditional sojourn time
                distribution for each case where a customer
                arrives to find i=1 to i=n+k customers present
                in an M/M/s queue with k customers initially present
    Arguments:  1. X: the distribution of the time between arrivals
                        (must be an ExponentialRV)
                2. Y: the service time distribution
                3. n: the total number of customers in the system
                4. k: the numober of customers in the system at time 0
                5. s: the number of identical parallel servers
    Output:     1. lst: the sojourn time distributions in segments
                    (a list of APPLPy random variables)
    """
    # Raise an error if both either of the distributions are not
    #   exponential
    if X.__class__.__name__!='ExponentialRV':
        err_string='both distributions in the queue must be'
        err_string+='exponential'
        raise RVError(err_string)

    if Y.__class__.__name__!='ExponentialRV':
        err_string='both distributions in the queue must be'
        err_string+='exponential'
        raise RVError(err_string)

    # Pre-compute the mean of y to avoid multiple integrations
    meany=Mean(Y)
    # Place positive assumptions on x to simplify output
    x=Symbol('x',positive=True)
    
    lst=[]
    for i in range(1,n+k+1):
        if s==1:
            lst.append(ErlangRV(1/meany,i))
        else:
            if i<=s or s>n+k:
                lst.append(Y)
            else:
                lst.append(Convolution(ErlangRV(s*(1/meany),i-s),Y))
    return lst

def MMSQprob(n,k,s):
    """
    Procedure Name: MMSQprob
    Purpose: Computes Pk(n,i) for an M/M/s queue, which is the
                probability that customer n will see i customers
                in the system counting himself at time T_n with
                k customers initially in the system at time 0.
    Arguments:  1. n: The total number of customers in the system
                2. k: The number of customers in the system at time 0
                3. s: The number of parallel servers
    Output:     1. Pk: A list of ordered probabilities
    """
    lst=[]
    for i in range(1,n+k+1):
        lst.append(_Q(n,i,k,s))
    return lst

def _Q(n,i,k,s):
    """
    Procedure Name: _Q
    Purpose: Computes the single probability Pk(n,i) for an M/M/s
                queue recursively.
    Arguments:  1. n: The total number of customers in the system
                2. i: An integer value
                3. k: The number of customers in the system at time 0
                4. s: The number of parallel servers
    Output:     1. Pk: A single probability for an M/M/s queue
    """
    rho=Symbol('rho')
    if k>=1 and i==k+n:
        if k>=s:
            p=(rho/(rho+1))**n
        elif k+n<=s:
            p=(rho**n)/(nprod(lambda j: rho+(k+j-1)/s,[1,n]))
        elif k<s and s<k+n:
            p=(rho**n)/((rho+1)**(n-s+k)*
                        (nprod(lambda j: rho+(k+j-1)/s,[1,s-k])))
    if k==0 and i==n:
        if n<=s:
            p=(rho**n)/nprod(lambda j: rho+(j-1)/s,[1,n])
        elif n>s:
            p=(rho**n)/((rho+1)**(n-s)*
                        nprod(lambda j: rho+(j-1)/s,[1,s]))
    if i==1:
        p=1-nsum(lambda j: _Q(n,j,k,s),[2,n+k])
    if k>=1 and i>=2 and i<=k and n==1:
        if k<=s:
            p=rho/(rho+(i-1)/s)*nprod(lambda j:
                                      1-rho/(rho+(k-j+1)/s),[1,k-i+1])
        elif k>s and i>s:
            p=rho/(rho+1)**(k-i+2)
        elif i<=s and s<k:
            p=rho/((rho+1)**(k-s+1)*(rho+(i-1/s))*
                   nprod(lambda j: 1-rho/(rho(s-j)/s),[1,s-i]))
    if n>=2 and i>=2 and i<=k+n-1:
        if i>s:
            p=rho/(rho+1)*nsum(lambda j:
                               (1/(rho+1)**(j-i+1)*_Q(n-1,j,k,s)),
                               [i-1,k+n-1])
        elif i<=s:
            p=rho/(rho+(i-1)/s)*(
                nsum(lambda j:
                     nprod(lambda h: 1-rho/(rho+(j-h+1)/s),[1,j-i+1])*
                           _Q(n-1,j,k,s),[i-1,s-1]) +
                nprod(lambda h: 1-rho/(rho+(s-h)/s),[1,s-i]) *
                nsum(lambda j: (1/(rho+1))**(j-s+1)*_Q(n-1,j,k,s),[s,k+n-1]))
    return simplify(p)
                
        
    
            

    














    
