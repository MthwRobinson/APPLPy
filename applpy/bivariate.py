"""
Bivariate Extension Module

1. Defines the Bivariate Random Variable Class
2. Computation of expected values
3. Computation of marginal distributions
4. Transformation of bivariate random variables

The algorithms implemented in this module were developed by
    Jeff Yang, John Drew and Larry Leemis and originally implemented in Maple

2012. J Yang, L Leemis and J Drew. 'Automating Bivariate Transformations'.
    INFORMS Journal on Computing Vol. 24, No. 1.
"""
from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, Add, Mul, Integer, function,
                   binomial, pprint,log,expand,zoo,latex,Piecewise)
from sympy.plotting.plot import plot
from random import random
import numpy as np
import plot as plt
import pylab as pyplt
from .rv import RV, RVError
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

class BivariateRV:
    """
    BivariateRV Class:
    Defines the data structure for bivariate random variables
    Defines special procedures for bivariate random variables
    """

    def __init__(self,func,constraints,ftype=['continuous','pdf']):
        """
        Procedure Name: __init__
        Purpose: Creates an instance of the bivariate random variable class
        Arguments:
            self.func: a list of the functions f(x,y) for the random variable
            self.constraints: a list of constraints for the random variable.
                The list of constraints must satisfy the following conditions:
                    1. The constraints must be entered in adjacent order;
                        clockwise or counterclockwise is acceptable
                    2. The constraints must completely enclose a region
                    3. The constraints must be entered as strictly inequalities
                        in the form 0<f(x,y). For instance, x**2<sqrt(y) would
                        be entered as sqrt(y)-x**2.
                    4. Except for constraints inthe form x<a or x>a, each
                        constraint must pass the vertical line test. i.e. There
                        should only be one y value associated with each x value.
            self.ftype: a list of two strings. The first indicates whether the
                random varable is discrete or continuous. The second specifies
                the form for the represenation of the random variable (pdf,cdf,
                etc)
        Output:     1. An instance of the bivariate random variable class
        """

        # If the function argument is not given in the form to a list, change
        #   it into list format

        if isinstance(func,list)!=True:
            func1=func
            func=[func1]
        # Check to make sure that the constraints are given in the form of
        #   a list
        if isinstance(constraints,list)!=True:
            raise RVError('Constraints must be entered as a list')
        # Make sure that the random variable is either discrete or continuous
        if ftype[0] not in ['continuous','discrete','Discrete']:
            err_string='Random variables must be discrete or continuous'
            raise RVError(err_string)
        # Check to make sure the constraint list has the correct length
        # The list of constraints should have the same number of elements
        #   as the list of functions
        if len(constraints)-len(func)!=0:
            err_string='a bivariate random variable must have one set of'
            err_string+=' constraints for each function that is entered'
        
        # Initialize the random variable
        self.func=func
        self.constraints=constraints
        self.ftype=ftype
        self.cache=None

        """
        Special Class Methods:

        1. __repr__(self)
        2. __len__(self)
        """

        def __repr__(self):
            """
            Procedure Name: __repr__
            Purpose: Sets the default string display setting for the bivariate
                        random variable class
            Arguments:  1. self: the random variable
            Output:     1. A series of print statements describing each
                            segment of the random variable
            """
            return repr(self.display(opt='repr'))

    def __len__(self):
        """
        Procedure Name: __len__
        Purpose: Sets the behavior for the len() procedure when an instance
                    of the random variable class is given as input. This
                    procedure will return the number of pieces if the
                    distribution is piecewise.
        Arguments:  1. self: the random variable
        Output:     1. the number of segments in the random variable
        """
        return len(self.func)

        """
        Utility Methods

        Procedures:
            1. add_to_cache(self,object_name,object)
            2. display(self)
            3. init_cache(self)
            4. verifyPDF(self)
        """

    def add_to_cache(self,object_name,obj):
        """
        Procedure Name: add_to_cache
        Purpose: Stores properties of the random variable (i.e. mean, variance,
                    cdf, sf) in memory. The next time a function is called to
                    compute that property, APPLPy will retrieve the object
                    from memory.
        Arguments:  1. self: the random variable
                    2. object_name: the key for the object in the cache
                        dictionary
                    3. obj: the object to be stored in memory.
        Output:     1. No output. The self.cache property of the random
                        variable is modified to include the specified
                        object.
        """
        # If a cache for the random variable does not exist, initialize it
        if self.cache==None:
            self.init_cache()
        # Add an object to the cache dictionary
        self.cache[object_name]=obj

    def display(self,opt='repr'):
        """
        Procedure Name: display
        Purpose: Displays the random variable in an interactive environment
        Arugments:  1. self: the random variable
        Output:     1. A print statement for each piece of the distribution
                        indicating the function and the relevant support
        """
        if self.ftype[0] in ['continuous','Discrete']:
            print ('%s %s'%(self.ftype[0],self.ftype[1]))
            for i in range(len(self.func)):
                cons_list=['0<'+str(cons) for cons in self.constraints[i]]
                cons_string=', '.join(cons_list)
                print('for x,y enclosed in the region:')
                print(cons_string)
                print('---------------------------')
                pprint(self.func[i])
                print('---------------------------')
                if i<len(self.func)-1:
                    print(' ');print(' ')
            
        if self.ftype[0]=='discrete':
            print '%s %s where {x->f(x)}:'%(self.ftype[0],
                                            self.ftype[1])
            for i in range(len(self.support)):
                if i!=(len(self.support)-1):
                    print '{%s -> %s}, '%(self.support[i],
                                          self.func[i]),
                else:
                    print '{%s -> %s}'%(self.support[i],
                                        self.func[i])

    def init_cache(self):
        """
        Procedure Name: init_cache
        Purpose: Initializes the cache for the random variable
        Arguments:  1. self: the random variable
        Output:     1. The cache attribute for the random variable
                            is initialized
        """
        self.cache={}

    def verifyPDF(self):
        """
        Procedure Name: verifyPDF
        Purpose: Verifies where or not the random variable is valid. It first
                    checks to make sure the pdf of the random variable
                    integrates to one. It then checks to make sure the random
                    variable is strictly positive
        Arguments:  1. self: the random variable
        Output:     1.  print statement that displays that volume under the
                            random variable and a second statement that
                            shows whether or not the random variable is valid
        """

        # Check to make sure that the random variable is entered as a
        #   continuous pdf

        if ftype!=['continuous','pdf']:
            err_string='verifyPDF currently only supports continuous pdfs'
            raise RVError(err_string)

        totalPDF=0
        absPDF=0

        # i loops through the number of segments of XY
        for i in range(len(self.func)):
            x='x';y='y'
            ncons=len(self.constraints[i])

            # list of x intercepts and corresponding y intercepts
            xinters=[]
            yinters=[]

            # corresponding lines 1 and 2
            line1=[]
            line2=[]

            # j loops through the constraints for segment i
            for j in range(ncons):
                cons_j=self.constraints[i][j]
                cons_mod=self.constraints[i][(j+1)%ncons]   

                # Use solve to compute the intersect point for each of the
                #   adjacent constraints. cons_j is the jth constraint
                #   and cons_mod uses modular division to find the adjacent
                #   constraint (moves to 0 after last adjacent segment).
                #   Intercepts are created first as a list to all the
                #   algorithm to detect multiple intercepts
                
                temp=solve([cons_j,cons_mod],x,y,dict=True)
                if cons_j==oo and cons_mod==0:
                    if cons_j==x:
                        temp=[{x:oo,y:0}]
                    else:
                        temp=[{x:0,y:oo}]
                if cons_j==-oo and cons_mod==0:
                    if cons_j==x:
                        temp=[{x:-oo,y:0}]
                    else:
                        temp=[{x:0,y:-oo}]
                if cons_j==0 and cons_mod==oo:
                    if cons_j==x:
                        temp=[{y:oo,x:0}]
                    else:
                        temp=[{y:0,x:oo}]
                if cons_j==0 and cons_mod==-oo:
                    if cons_j==x:
                        temp=[{y:-oo,x:0}]
                    else:
                        temp=[{y:0,x:-oo}]

                if len(temp)>1:
                    err_string='Adjacent constraints intersect at '
                    err_string='two or more points'
                    raise RVError(err_string)
                elif len(temp)==0:
                    err_string='Adjacent constraints do not intersect'
                    raise RVError(err_string)

                if len(temp)!=0:
                    line1.append(cons_j)
                    line2.append(cons_mod)
                if len(temp)!=0:
                    xinters.append(temp[0][x])
                    yinters.append(temp[0][y])
                if len(xinters)==ncons+1:
                    print('Unbounded')

            # Bubble sort all four lists with respect to xinters
            for ib in range(len(xinters)-1):
                for jb in range(ib+1,len(xinters)):
                    if xinters[ib]>xinters[jb]:
                        # Swap relevant indices in for the intercepts and lines
                        tempb=xinters[ib]
                        xinters[ib]=xinters[jb]
                        xinters[jb]=tempb
                        
                        tempb=yinters[ib]
                        yinters[ib]=yinters[jb]
                        yinters[jb]=tempb

                        tempb=line1[ib]
                        line1[ib]=line1[jb]
                        line1[jb]=tempb

                        tempb=line2[ib]
                        line2[ib]=line2[jb]
                        line2[jb]=tempb

            # Default yupper and ylower assuming the start off is a vert line
            set1=_union(line1[0],line2[0])
            set2=_union(line1[1],line2[1])
            yupper=_intersect(set1,set2)
            ylower=_intersect(set1,set2)

            start=1

            # Start off from a point (figure out yupper and ylower)

            if xinters[0]!=yinters[1]:
                eqn=solve(line1[0],y)-solve(line2[0],y)
                area=integrate(eqn,(x,xinters[0],xinters[1]))

                # PDF evaluated over the segment
                if area>0:
                    yupper=line1[0]
                    ylower=line2[0]
                else:
                    ylower=line1[0]
                    yupper=line2[0]
                y_0=solve(ylower,y)
                y_1=solve(yupper,y)
                x_0=xinters[0]
                x_1=xinters[1]
                totalPDF+=integrate(integrate(XY.func[i],
                                              (y,y_0,y_1)),
                                    (x,x_0,x_1))
                # Not yet supported by sympy
                #absPDF+=integrate(integrate(abs(XY.func[i]),
                #                            (y,y_0,y_1)),
                #                  (x,x_0,x_1))
                start=2

            # Triangle case (left = point, right = line)
            if start==2 and len(xinters)==3 and xinters[2]==xinters[1]:
                start=4

            # Begin calculating PDFs
            ind=start-1
            while ind<len(xinters):
                # left xinters lie on a vertical line
                if xinters[ind]==xinters[ind+1] and ind!=len(xinters):
                    # y ind < ind+1 => ylower is the other line intersecting
                    #   ind and yupper is otherline intersecting ind+1
                    if yinters[ind]<yinters[ind+1]:
                        ylower=_union(line1[ind],line2[ind]).remove(ylower)
                        yupper=_union(line1[ind+1],line2[ind+1]).remove(ylower)
                    else:
                        yupper=_union(line1[ind],line2[ind]).remove(ylower)
                        ylower=_union(line1[ind+1],line2[ind+1]).remove(ylower)
                ylower=ylower[0]
                yupper=yupper[0]
                # Infinity case
                lowerB=solve(ylower,y)
                upperB=solve(yupper,y)
                if yupper==oo:
                    upperB=oo
                if ylower==-oo:
                    lowerB=-oo
                totalPDF+=integrate(integrate(XY.func[i],
                                              (y,lowerB,upperB)),
                                    (x,xinters[ind],xinters[ind+1]))
                # Not yet supported by sympy
                #absPDF+=integrate(integrate(abs(XY.func[i]),
                #                            (y,lowerB,upperB)),
                #                  (x,xinters[ind],xinters[ind+1]))

                ind+=1
                # Left x is only one point
                
            
'''
Supporting Functions:

1. _intersection(a,b)
2. _union(a,b)

'''
                    
def _intersection(a,b):
    """
    Procedure Name: _intersection
    Purpose: Returns the intersection of two lists
    Arguments:  1. a: a list
                2. b: a list
    Output:     1. A list containing the intersection of a and b
    """
    if type(a) != list:
        a=[a]
    if type(b) != list:
        b=[b]
    return list(set(a) & set(b))
                
def _union(a,b):
    """
    Procedure Name: _union
    Purpose: Returns the union of two lists
    Arguments:  1. a: a list
                2. b: a list
    Output:     1. A list containing the union of a and b
    """
    if type(a) != list:
        a=[a]
    if type(b) != list:
        b=[b]                       
    return list(set(a) | set(b))
    
















