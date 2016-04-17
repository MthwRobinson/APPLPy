"""
Queuing Module

Defines an extension for computing the sojourn time distribution for the
    nth customer in an M/M/s queue

The algorithms implemented in this module were developed by
    William Kaczynski and originally implemented in Maple

2012. W Kaczynski, L Leemis and J Drew. 'Transient Queuing Analysis'.
    INFORMS Journal on Computing Vol. 24, No. 1.

Procedures:
    1. Queue(X,Y,n,k,s)
"""

from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, Add, Mul, Integer, function,
                   binomial)
from mpmath import (nsum,nprod)
from random import random
import numpy as np
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


def Queue(X,Y,n,k=0,s=1):
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
    Output:     1. Probability distribution for an M/M/s queue
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
Queue Sub-Procedures:
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
                

"""
Cov/kCov Sub-Procedures:
    1. cases(n)
    2. caseprob(n,P,meanX,meanY)
    3. Cprime(n,C)
    4. ini(n)
    5. kcases(n,k)
    6. kcaseprob(n,k,P,meanX,meanY)
    7. kpath(n,k,A)
    8. kprobvec(n,k,meanX,meanY)
    9. okay(n,E)
    10. path(n,A)
    11. probvec(n,meanX,meanY)
    12. swapa(n,A)
    13. swapb(n,B)
"""

'''
Re-look code for cases,kcases,ini,swapa,swapb and okay for errors
    probvec does not sum to 1, error is most likely in one of those
    procedures
'''

def cases(n):         
    """
    Procedure Name: cases
    Purpose: Generates all possible arrival/departure sequences for
                n customers in an M/M/1 queue initially empty and
                idle.
    Arguments:  1. n: the total number of customers in the system
    Output:     1. C: a list of sequences consisting of 1s and -1s,
                        where 1s represent an arrival and -1s
                        represent a departure
    """
    # Compute the nth Catalan number
    c=factorial(2*n)/factorial(n)/factorial(n+1)
    C=np.zeros((c,2*n))
    for i in range(c):
        # Initialize the matrix C
        if i==0:
            C[i]=ini(n)
        # Produce the successor the C[i]
        else:
            C[i]=swapa(n,C[i-1])
        # Check to see if the successor is legal
        #   If not, call swapb
        if okay(n,C[i])==False:
            C[i]=swapb(n,C[i-1])
    return C

def caseprob(n,P,meanX,meanY):
    """
    Procedure Name: caseprob
    Purpose: Computes the probability associated with a given row of
                the case matrix C as represented by the path created
                by path(n,A)
    Input:  1. n: the total number of customers in the system
            2. P: the path of a given case
            3. meanX: the mean of the arrival distribution
            4. meanY: the mean of the service time distribution
    Output:  1. p: the probability of the case passed to the procedure
    """
    
    p=1
    row=n
    col=1
    for j in range(2*n-1):
        if P[row-1][col]==1 and col<n:
            row-=1
            p=p*1/meanY/(1/meanX+1/meanY)
        elif P[row-1][col]==1 and col==n:
            row-=1
        elif P[row][col+1]==1 and row+col>n+1:
            col+=1
            p=p*1/meanX/(1/meanX+1/meanY)
        else:
            col+=1
    return p

def Cprime(n,C):
    """
    Procedure Name: Cprime
    Purpose: Produces a matrix C' that is the distribution segment
                matrix where each row represents the distribution
                segments for the case represented by the corresponding
                row in the case matrix C. The elements of C' are
                limited to a 0, 1 and 2. 0 implies no sojourn time
                distribution segment due to an empyting of the system.
                1 implies a copeting risk of an arrival or completion of
                service and is distributed Exp(theta+mu) and 2 implies a
                service completion distribution leg, which is distributed
                Exp(mu).
    Input:      1. n: the total number of customers in the system
                2. C: the case matrix
    Output:     1. prime: a matrix with the same number of rows as C and
                            2n-1 columns
    """

    prime=np.zeros((np.size(C,1),2*n-1))
    for i in range(np.size(C,1)):
        row=n
        col=1
        pat=path(n,C[i])
        dist=np.zeros((1,2*n-1))
        for j in range(2*n-1):
            if pat[row-1][col]==1 and col<n:
                row-=1
                dist[0][j]=1
            elif pat[row-1][col]==1 and col==n:
                row-=1
                dist[0][j]=2
            elif pat[row-1][col]==1 and row+col>n+1:
                col+=1
                dist[0][j]=1
            else:
                col+=1
                dist[0][j]=0
        prime[i]=dist
    return prime
    

def ini(n):
    """
    Procedure Name: ini
    Purpose: Initializes a matrix C according to Ruskey and Williams
                Returns the first row of C to enable use of prefix
                shift algorithm.
    Arguments:  1. n: the total number of customers in the system
    Output:     1. L: a row vector, the first row of C
    """
    L=-np.ones(2*n)
    L[0]=1
    for i in range(2,n+1):
        L[i]=1
    for i in range(n+1,2*n):
        L[i]=-1
    return L

def kcases(n,k):
    """
    Procedure Name: kcases
    Purpose: Generates all possible arrival/departure sequences for n
                customers in an M/M/1 queue with k customers initially
                present.
    Arguments:  1. n: the total number of customers in the system
                2. k: the number of customers initially present in the
                        system.
    Output:     1. C: a list of sequences consiting of 1's and -1s where
                        1 represents an arrival and -1 represents a
                        departure.
    """
    C=cases(n+k)
    j=0
    while j<np.size(C,1):
        # if the sum of row j of C from column 1 to k != k
        if np.sum(C[j,0:k])!=k:
            # delete the jth row of C
            C=np.delete(C,j,0)
        else:
            j+=1
    # delete column 1st through kth column of C
    C=np.delete(C,np.s_[0:k],1)
    return C


def kcaseprob(n,k,P,meanX,meanY):
    """
    Procedure Name: caseprob
    Purpose: Computes the probability associated with a given row of
                the case matrix C as represented by the path created
                by kpath(n,k,A)
    Input:  1. n: the total number of customers in the system
            2. k: the total number of customers initially present
                    in the system
            3. P: the path of a given case
            4. meanX: the mean of the arrival distributon
            5. meanY: the mean of the service time distribution
    Output:  1. p: the probability of the case passed to the procedure
    """
    
    p=1
    row=n+k
    col=0
    for j in range(2*n+k):
        if P[row-1][col]==1 and col<n:
            row-=1
            p=p*1/meanY/(1/meanX+1/meanY)
        elif P[row-1][col]==1 and col==n:
            row-=1
        elif P[row][col+1]==1 and row+col>n+1:
            col+=1
            p=p*1/meanX/(1/meanX+1/meanY)
        else:
            col+=1
    return p
    

def kCprime(n,k,C):
    """
    Procedure Name: Cprime
    Purpose: Produces a matrix C' that is the distribution segment
                matrix where each row represents the distribution
                segments for the case represented by the corresponding
                row in the case matrix C. The elements of C' are
                limited to a 0, 1 and 2. 0 implies no sojourn time
                distribution segment due to an empyting of the system.
                1 implies a copeting risk of an arrival or completion of
                service and is distributed Exp(theta+mu) and 2 implies a
                service completion distribution leg, which is distributed
                Exp(mu).
    Input:      1. n: the total number of customers in the system
                2. k: the number of customers initially in the system
                2. C: the case matrix
    Output:     1. prime: a matrix with the same number of rows as C and
                            2*(n+1)+k columns
    """

    prime=np.zeros((np.size(C,1),2*n+k))
    for i in range(np.size(C,1)):
        row=n+k
        col=0
        pat=kpath(n,k,C[i])
        dist=np.zeros((1,2*n+k))
        for j in range(2*n+k):
            if pat[row-1][col]==1 and col<n:
                row-=1
                dist[0][j]=1
            elif pat[row-1][col]==1 and col==n:
                row-=1
                dist[0][j]=2
            elif pat[row-1][col+1]==1 and row+col>n+1:
                col+=1
                dist[0][j]=1
            else:
                col+=1
                dist[0][j]=0
        prime[i]=dist
    return prime
    

def kpath(n,k,A):
    """
    Procedures Name: kpath
    Purpose: Creates a path that starts at the lower left corner
                of the matrix and moves to the upper right corner.
                The first leg of the path is always the arrival of
                customer 1. A 1 to the right of the previous 1
                signifies an arrival, while a 1 above the previous
                1 signifies a service completion.
    Arugments:  1. n: the total number of customers in the system
                2. k: the number of customers initially present in
                        the system
                3. A: A row from the case matrix C.
    Output:     1. pat: A path matrix of size (n+k+1)x(n+1)
    """

    row=n+k
    col=0
    pat=np.zeros((n+k+1,n+1))
    pat[n+k][0]=1
    for j in range(2*n+k):
        if A[j]==1:
            col+=1
            pat[row][col]=1
        else:
            row-=1
            pat[row][col]=1
    return pat


def kprobvec(n,k,meanX,meanY):
    """
    Procedure Name: probvec
    Purpose: Uses the caseprob procedure to successivly build a vector
                of probabilities, one for each case of the C matrix.
    Input:  1. n: the total number of customers in the system
            2. meanX: the mean of the arrival distribution
            3. meanY: the mean of the service time distribution
    Output: 1. p: a probability vector of length 2n!/n!/(n+1)!
    """
    C=kcases(n,k)
    p=np.zeros(np.size(C,1))
    for i in range(np.size(C,1)):
        p[i]=kcaseprob(n,k,kpath(n,k,C[i]),meanX,meanY)
    return p
    

def okay(n,E):
    """
    Procedure Name: okay
    Purpose: Checks the output of swapa for an illegal prefix shift,
                meaning the result contains an impossible arrival/
                service sequence.
    Arguments:  1. n: the total number of customers in the system
                2. E: the vector resulting from swapa
    Output:     1. test: a binary indicator where True signfies the
                    successor is legal and False signifies that the
                    successor is illegal
    """
    test=True
    s=0
    for i in range(2*n-1):
        s+=E[i]
        if s<0:
            test=False
            break
    return test

def path(n,A):
    """
    Procedure Name: path
    Purpose: Creates a path that starts at the lower left corner
                of the matrix and moves to the upper right corner.
                The first leg of the path is always the arrival of
                customer 1. A 1 to the right of the previous 1 signifies
                an arrival, while a 1 above the previous 1 signifies
                a service completion.
    Arguments:  1. n: the total number of customers in the system
                2. A: a row from the case matrix C.
    Output:     1. pat: A path matrix of size (n+1)x(n+1)
    """

    row=n
    col=1
    pat=np.zeros((n+1,n+1))
    pat[n,0]=1
    pat[n,1]=1
    for j in range(1,2*n):
        if A[j]==1:
            col+=1
            pat[row,col]=1
        else:
            row-=1
            pat[row,col]=1
    return pat   


def probvec(n,meanX,meanY):
    """
    Procedure Name: probvec
    Purpose: Uses the caseprob procedure to successivly build a vector
                of probabilities, one for each case of the C matrix.
    Input:  1. n: the total number of customers in the system
            2. meanX: the mean of the arrival distribution
            3. meanY: the mean of the service time distribution
    Output: 1. p: a probability vector of length 2n!/n!/(n+1)!
    """

    c=factorial(2*n)/factorial(n)/factorial(n+1)
    p=np.zeros(c)
    for i in range(c):
        p[i]=caseprob(n,path(n,cases(n)[i]),meanX,meanY)
    return p


def swapa(n,A):
    """
    Procedure Name: swapa
    Purpose: Conducts the (k+1)st prefix shift in creating all
                instances of the case matrix, C, according
                to Ruskey and Williams
    Arguments:  1. n: the total number of customers in the system
                2. A: row i of matrix C
    Output:     1. R: the successor of C
    """
    R=A
    check=1
    for i in range(1,2*n-1):
        if R[i]==-1 and R[i+1]==1:
            temp1=R[i+2]
            R[2:(i+2)]=R[1:(i+1)]
            check=0
            R[1]=temp1
        if check==0:
            break
    return R

def swapb(n,B):
    """
    Procedure Name: swapb
    Purpose: Conducts the kth prefix shift in creating all
                instances of the case matrix, C, accoring
                to Ruskey and Williams
    Arguments:  1. n: the number of customers in the system
                2. B: row i of matrix C
    Output:     1. R: the successor of C
    """
    R=B
    check=1
    for i in range(1,2*n-2):
        if R[i]==-1 and R[i+1]==1:
            temp=R[i+1]
            R[2:(i+1)]=R[1:i]
            check=0
            R[1]=temp
        if check==0:
            break
    return R

