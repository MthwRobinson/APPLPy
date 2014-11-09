from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, plot, Add, Mul, Integer, function,
                   binomial, pprint,log)
from random import random
import numpy as np
import plot as plt
import pylab as pyplt
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
Main Random Variable Module

1. The Random Variable class
2. Procedures for changing functional form
3. Operations on one random variable
4. Operations on two random variables
5. Plots

"""

class RVError(Exception):
    """
    RVError Class
    Defines a custom error message for exceptions relating
    to the random variable class
    """
    def __init__(self,value):
        self.value=value
    def __str__(self):
        return repr(self.value)

class RV:
    """
    RV Class
    Defines the data structure of ApplPy random variables
    Defines procedures relating to ApplPy random variables
    """

    def __init__(self,func,support,ftype=['continuous','pdf']):
        """
        Creates an instance of the random variable class
            The random variable default is to produce a continuous pdf
        Checks the random variable for errors
        """

        # Check for errors in the data structure of the random
        #   variable

        # Check to make sure that the given function is in the
        #   form of a list
        # If it is not in the form of a list, place it in a list
        if isinstance(func,list)!=True:
            func1=func
            func=[func1]
        # Check to make sure that the given support is in the form of
        #   a list
        if isinstance(support,list)!=True:
            raise RVError('Support must be a list')
        # Check to make sure that the random variable is either
        #   discrete or continuous
        if ftype[0] not in ['continuous','discrete','Discrete']:
            string='Random variables must either be discrete'
            string+=' or continuous'
            raise RVError(string)
        # Check to make sure that the support list has the correct
        #   length
        # The support list should be one element larger than the
        #   function list for continuous distributions, and the same
        #   size for discrete
        if ftype[0] in ['continuous','Discrete']:
            if len(support)-len(func)!=1:
                string='Support has incorrect number of elements'
                raise RVError(string)
        if ftype[0]=='discrete':
            if len(support)-len(func)!=0:
                string='Support has incorrect number of elements'
                raise RVError(string)
        # Check to make sure that the elements of the support list are
        #   in ascending order
        for i in range(len(support)-1):
            # Only compare if the supports are numbers
            if type(support[i]) in [int,float]:
                if type(support[i+1]) in [int,float]:
                    if support[i]>support[i+1]:
                        raise RVError('Support is not in ascending order')
        # Initialize the random variable
        self.func=func
        self.support=support
        self.ftype=ftype

    """
    Special Class Methods

    Procedures:
        1. display(self)
        2. __repr__(self)
        3. __len__(self)
        4. __add__(self,other)
    """

    def display(self,opt='repr'):
        """
        Creates a default print setting for the random variable class
        """
        if self.ftype[0] in ['continuous','Discrete']:
            for i in range(len(self.func)):
                print('for %s <= x <= %s'%(self.support[i],
                                           self.support[i+1]))
                print('---------------------------')
                pprint(self.func[i])
                print('---------------------------')
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

    def __repr__(self):
        """
        Sets the default string display setting for the random
        variable class
        """
        return repr(self.display(opt='repr'))

    def __len__(self):
        """
        Sets the behavior for the len() procedure when an instance
            of the random variable class is given as input
        """
        return len(self.func)

    # Set the behavior for the operators '+,-,*,/'

    def __add__(self,other):
        """
        Sets the behavior of the '+' operator
        """
        return Convolution(self,other)

    def __sub__(self,other):
        """
        Sets the behavior of the '-' operator
        """
        gX=[[-x],[-oo,oo]]
        RVar=Transform(other,gX)
        return Convolution(self,RVar)

    def __mul__(self,other):
        """
        Sets the behavior of the '*' operator
        """
        return Product(self,other)

    def __truediv__(self,other):
        """
        Sets the behavior of the '/' operator
        """
        gX=[[1/x,1/x],[-oo,0,oo]]
        RVar=Transform(other,gX)
        return Product(self,RVar)

    """
    Utility Methods

    Procedures:
        1. verifyPDF(self)
        2. variate(self,n)
    """
    def verifyPDF(self):
        """
        Checks whether of not the pdf of a random variable is valid
        """
        # If the random variable is continuous, verify the PDF
        if self.ftype[0]=='continuous':
            # Check to ensure that the distribution is fully
            #   specified
            for piece in self.func:
                    func_symbols=piece.atoms(Symbol)
                    if len(func_symbols)>1:
                        err_string='distribution must be fully'
                        err_string+=' specified'
                        raise RVError(err_string)
            # Convert the random variable to PDF form
            X_dummy=PDF(self)
            # Check to ensure that the area under the PDF is 1
            print 'Now checking for area...'
            area=0
            for i in range(len(X_dummy.func)):
                val=integrate(X_dummy.func[i],(x,X_dummy.support[i],
                                               X_dummy.support[i+1]))
                area+=val
            print 'The area under f(x) is: %s'%(area)
            # Check absolute value
            print 'Now checking for absolute value...'
            #
            # The following code should work in future versions of SymPy
            # Currently, Sympy is having difficulty consistently integrating
            # the absolute value of a function symbolically
            #
            #abs_area=0
            #for i in range(len(X_dummy.func)):
                #val=integrate(Abs(X_dummy.func[i],(x,X_dummy.support[i],
                #                                   X_dummy.support[i+1]))
                #abs_area+=val
            abs_flag=True
            val_list=[]
            quant_list=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
            for i in range(len(quant_list)):
                val=self.variate(s=quant_list[i])[0]
                val_list.append(val)
            for i in range(len(val_list)):
                if val_list[i]<0:
                    abs_flag=False
            print 'The pdf of the random variable:'
            print '%s'%(X_dummy.func)
            print 'continuous pdf with support %s'%(X_dummy.support)
            if area>.9999 and area<1.00001 and abs_flag==True:
                print 'is valid'
            else:
                print 'is not valid'
        # If the random variable is in a discrete functional form,
        #   verify the PDF
        if self.ftype[0]=='Discrete':
            # Convert the random variable to PDF form
            X_dummy=PDF(self)
            # Check to ensure that the area under the PDF is 1
            print 'Now checking for area...'
            area=0
            for i in range(len(X_dummy.func)):
                val=summation(X_dummy.func[i],(x,X_dummy.support[i],
                                               X_dummy.support[i+1]))
                area+=val
            print 'The area under f(x) is: %s'%(area)
            # Check absolute value
            print 'Now checking for absolute value...'
            abs_flag=True
            val_list=[]
            quant_list=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
            for i in range(len(quant_list)):
                val=self.variate(s=quant_list[i])[0]
                val_list.append(val)
            for i in range(len(val_list)):
                if val_list[i]<0:
                    abs_flag=False
            print 'The pdf of the random variable:'
            print '%s'%(X_dummy.func)
            print 'discrete pdf with support %s'%(X_dummy.support)
            if area>.9999 and area<1.00001 and abs_flag==True:
                print 'is valid'
            else:
                print 'is not valid'
        # If the random variable is discrete, verify the PDF
        if self.ftype[0]=='discrete':
            # Convert the random variable to PDF form
            X_dummy=PDF(self)
            # Check to ensure that the area under the PDF is 1
            print 'Now checking for area...'
            area=sum(X_dummy.func)
            #for i in range(len(self.support)):
            #    area+=self.func[i]
            print 'The area under f(x) is: %s'%(area)
            # Check for absolute value
            print 'Now checking for absolute value...'
            abs_flag=True
            for i in range(len(self.func)):
                if self.func[i]<0:
                    abs_flag=False
            print 'The pdf of the random variable'
            if area>.9999 and area<1.0001 and abs_flag==True:
                print 'is valid'
            else:
                print 'is not valid'

    def variate(self,n=1,s='sim',sensitivity=.00001):
        """
        Generates a list of n random variates from the random variable
            using the Newton-Raphson Method
        """   
        # Find the cdf and pdf functions (to avoid integrating for
            # each variate
        cdf=CDF(self)
        pdf=PDF(self)
        mean=Mean(self)
        # Create a list of variates
        varlist=[]
        for i in range(n):
            guess=mean
            if s=='sim':
                val=random()
            else:
                val=s
            for i in range(10):
                try:
                    if len(self.func)==1:
                        guess=(guess-
                               ((cdf.func[0].subs(x,guess)-val)/
                                     pdf.func[0].subs(x,guess)))
                        guess=guess.evalf()
                    else:
                        guess=(guess-((CDF(cdf,guess)-val)/
                                      PDF(pdf,guess))).evalf()
                except:
                    if guess>self.support[len(self.support)-1]:
                        cfunc=cdf.func[len(self.func)-1].subs(x,guess)
                        pfunc=pdf.func[len(self.func)-1].subs(x,guess)
                        guess=(guess-((cfunc-val)/pfunc)).evalf()
                    if guess<self.support[0]:
                        cfunc=cdf.func[0].subs(x,guess)
                        pfunc=pdf.func[0].subs(x,guess)
                        guess=(guess-((cfunc-val)/pfunc)).evalf()
            varlist.append(guess)            
        varlist.sort()
        return varlist

"""
Procedures:
    1. CDF(RVar,value)
    2. CHF(RVar,value)
    3. HF(RVar,value)
    4. IDF(RVar,value)
    5. PDF(RVar,value)
    6. SF(RVar,value)
    7. BootstrapRV(varlist)
    8. Convert(RVar,inc)
"""

def check_value(value,sup):
    # Not intended for use by end user
    """
    Procedure Name: check_value
    Purpose: Check to see if a value passed to CDF,CHF,HF,PDF or
                SF is in the support of the random variable
    Arguments:  1. value: The value passed to RV procedure
                2. sup: The support of the RV in the procedure
    Output:     1. True if the value given is within the support
                2. False otherwise
    """
    if value==x:
        return True
    else:
        max_idx=len(sup)-1
        if float(value)<float(sup[0]) or float(value)>float(sup[max_idx]):
            return False
        else:
            return True

def CDF(RVar,value=x):
    """
    Procedure Name: CDF
    Purpose: Compute the cdf of a random variable
    Arguments:  1. RVar: A random variable
                2. value: An integer or floating point number
    Output:     1. CDF of a random variable (if value not specified)
                2. Value of the CDF at a given point
                    (if value is specified)
    """

    # Check to make sure the value given is within the random
    #   variable's support
    if value.__class__.__name__!='Symbol':
        if value>RVar.support[-1] or value<RVar.support[0]:
            string='Value is not within the support of the random variable'        
            raise RVError(string)

    # If the distribution is continous, find and return the distribution
    #   of the random variable
    if RVar.ftype[0]=='continuous':
        # If the random variable is already a cdf, nothing needs to
        #   be done
        if RVar.ftype[1]=='cdf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                        cdfvalue=RVar.func[i].subs(x,value)
                        return simplify(cdfvalue)
        # If the random variable is a sf, find and return the cdf of the
        #   random variable
        if RVar.ftype[0]=='sf':
            X_dummy=SF(RVar)
            # Compute the sf for each segment
            cdflist=[]
            for i in range(len(X_dummy.func)):
                cdflist.append(1-X_dummy.func[i])
            # If no value is specified, return the sf function
            if value==x:
                return RV(cdflist,X_dummy.support,['continuous','cdf'])
            # If not, return the value of the cdf at the specified value
            else:
                for i in range(len(X_dummy.support)):
                    if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                        cdfvalue=cdflist[i].subs(x,value)
                        return simplify(cdfvalue)
        # If the random variable is not a cdf or sf, compute the pdf of
        #   the random variable, and then compute the cdf by integrating
        #   over each segment of the random variable
        else:
            X_dummy=PDF(RVar)
            # Substitue the dummy variable 't' into the dummy rv
            funclist=[]
            for i in range(len(X_dummy.func)):
                newfunc=X_dummy.func[i].subs(x,t)
                funclist.append(newfunc)
            # Integrate to find the cdf
            cdflist=[]
            for i in range(len(funclist)):
                cdffunc=integrate(funclist[i],(t,X_dummy.support[i],x))
                # Adjust the constant of integration
                if i!=0:
                    const=(cdflist[i-1].subs(x,X_dummy.support[i])-
                    cdffunc.subs(x,X_dummy.support[i]))
                    cdffunc=cdffunc+const
                if i==0:
                    const=0-cdffunc.subs(x,X_dummy.support[i])
                    cdffunc=cdffunc+const
                cdflist.append(simplify(cdffunc))
            # If no value is specified, return the cdf
            if value==x:
                return RV(cdflist,X_dummy.support,['continuous','cdf'])
            # If a value is specified, return the value of the cdf
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                        cdfvalue=cdflist[i].subs(x,value)
                        return simplify(cdfvalue)

    # If the distribution is in discrete functional, find and return the
    #   distribution of the random variable
    if RVar.ftype[0]=='Discrete':
        # If the random variable is already a cdf, nothing needs to
        #   be done
        if RVar.ftype[1]=='cdf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                        cdfvalue=RVar.func[i].subs(x,value)
                        return simplify(cdfvalue)
        # If the random variable is a sf, find and return the cdf of the
        #   random variable
        if RVar.ftype[0]=='sf':
            X_dummy=SF(RVar)
            # Compute the sf for each segment
            cdflist=[]
            for i in range(len(X_dummy.func)):
                cdflist.append(1-X_dummy.func[i])
            # If no value is specified, return the sf function
            if value==x:
                return RV(cdflist,X_dummy.support,['continuous','cdf'])
            # If not, return the value of the cdf at the specified value
            else:
                for i in range(len(X_dummy.support)):
                    if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                        cdfvalue=cdflist[i].subs(x,value)
                        return simplify(cdfvalue)
        # If the random variable is not a cdf or sf, compute the pdf of
        #   the random variable, and then compute the cdf by summing
        #   over each segment of the random variable
        else:
            X_dummy=PDF(RVar)
            # Substitue the dummy variable 't' into the dummy rv
            funclist=[]
            for i in range(len(X_dummy.func)):
                newfunc=X_dummy.func[i].subs(x,t)
                funclist.append(newfunc)
            # Integrate to find the cdf
            cdflist=[]
            for i in range(len(funclist)):
                cdffunc=summation(funclist[i],(t,X_dummy.support[i],x))
                # Adjust the constant of integration
                if i!=0:
                    const=(cdflist[i-1].subs(x,X_dummy.support[i])-
                    cdffunc.subs(x,X_dummy.support[i]))
                    cdffunc=cdffunc+const
                if i==0:
                    const=0-cdffunc.subs(x,X_dummy.support[i])
                    cdffunc=cdffunc+const
                cdflist.append(simplify(cdffunc))
            # If no value is specified, return the cdf
            if value==x:
                return RV(cdflist,X_dummy.support,['Discrete','cdf'])
            # If a value is specified, return the value of the cdf
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                        cdfvalue=cdflist[i].subs(x,value)
                        return simplify(cdfvalue)
                    
    # If the distribution is discrete, find and return the cdf of
    #   the random variable
    if RVar.ftype[0]=='discrete':
        # If the distribution is already a cdf, nothing needs to
        #   be done
        if RVar.ftype[1]=='cdf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar)):
                    if RVar.support[i]==value:
                        return RVar.func[i]
                    if RVar.support[i]<value:
                        if RVar.support[i+1]>value:
                            return RVar.func[i]
        # If the distribution is a sf, find the cdf by reversing the
        #   function list
        if RVar.ftype[1] in ['sf','chf','hf']:
            X_dummy=SF(RVar)
            newfunc=[]
            for i in reversed(range(len(X_dummy.func))):
                newfunc.append(X_dummy.func[i])
            Xsf=RV(newfunc,X_dummy.support,['discrete','cdf'])
            if value==x:
                return Xsf
            if value!=x:
                X_dummy=CDF(X_dummy)
                for i in range(len(X_dummy.support)):
                    if X_dummy.support[i]==value:
                        return X_dummy.func[i]
                    if X_dummy.support[i]<value:
                        if X_dummy.support[i+1]>value:
                            return X_dummy.func[i]
        # If the distribution is not a cdf or sf, find the pdf and
        #   then compute the cdf by summation
        else:
            X_dummy=PDF(RVar)
            cdffunc=[]
            area=0
            for i in range(len(X_dummy.support)):
                area+=X_dummy.func[i]
                cdffunc.append(area)
            if value==x:
                return RV(cdffunc,X_dummy.support,['discrete','cdf'])
            if value!=x:
                X_dummy=CDF(X_dummy)
                for i in range(len(X_dummy.support)):
                    if X_dummy.support[i]==value:
                        return X_dummy.func[i]
                    if X_dummy.support[i]<value:
                        if X_dummy.support[i+1]>value:
                            return X_dummy.func[i]
                

def CHF(RVar,value=x):
    """
    Procedure Name: CHF
    Purpose: Compute the chf of a random variable
    Arguments:  1. RVar: A random variable
                2. value: An integer or floating point number
                    (optional)
    Output:     1. CHF of a random variable (if value not specified)
                2. Value of the CHF at a given point
                    (if value is specified)
    """
    
    # Check to make sure the value given is within the random
    #   variable's support
    if value.__class__.__name__!='Symbol':
        if value>RVar.support[-1] or value<RVar.support[0]:
            string='Value is not within the support of the random variable'        
            raise RVError(string)

    # If the distribution is continuous, find and return the chf of
    #   the random variable
    if RVar.ftype[0]=='continuous':
        # If the distribution is already a chf, nothing needs to
        #   be done
        if RVar.ftype[1]=='chf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i]:
                        if value<=RVar.support[i+1]:
                            chfvalue=RVar.func[i].subs(x,value)
                            return simplify(chfvalue)
        # Otherwise, find and return the chf
        else:
            X_dummy=SF(RVar)
            # Generate a list of sf functions
            sflist=[]
            for i in range(len(X_dummy.func)):
                sflist.append(X_dummy.func[i])
            # Generate chf functions
            chffunc=[]
            for i in range(len(sflist)):
                newfunc=-ln(sflist[i])
                chffunc.append(simplify(newfunc))
            # If a value is not specified, return the chf of the
            #   random variable
            if value==x:
                return RV(chffunc,X_dummy.support,['continuous','chf'])
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.func[i]:
                        if value<=RVar.support[i+1]:
                            chfvalue=chffunc[i].subs(x,value)
                            return simplify(chfvalue)

    # If the distribution is a discrete function, find and return the chf of
    #   the random variable
    if RVar.ftype[0]=='Discrete':
        # If the distribution is already a chf, nothing needs to
        #   be done
        if RVar.ftype[1]=='chf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i]:
                        if value<=RVar.support[i+1]:
                            chfvalue=RVar.func[i].subs(x,value)
                            return simplify(chfvalue)
        # Otherwise, find and return the chf
        else:
            X_dummy=SF(RVar)
            # Generate a list of sf functions
            sflist=[]
            for i in range(len(X_dummy.func)):
                sflist.append(X_dummy.func[i])
            # Generate chf functions
            chffunc=[]
            for i in range(len(sflist)):
                newfunc=-ln(sflist[i])
                chffunc.append(simplify(newfunc))
            # If a value is not specified, return the chf of the
            #   random variable
            if value==x:
                return RV(chffunc,X_dummy.support,['Discrete','chf'])
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.func[i]:
                        if value<=RVar.support[i+1]:
                            chfvalue=chffunc[i].subs(x,value)
                            return simplify(chfvalue)
                    
    # If the random variable is discrete, find and return the chf
    if RVar.ftype[0]=='discrete':
        # If the distribution is already a chf, nothing needs to
        #   be done
        if RVar.ftype[1]=='chf':
            if value==x:
                return RVar
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return RVar.func[RVar.support.index(value)]
        # Otherwise, use the survivor function to find the chf
        else:
            X_sf=SF(RVar)
            chffunc=[]
            for i in range(len(X_sf.func)):
                chffunc.append(-log(X_sf.func[i]))
            if value==x:
                return RV(chffunc,X_sf.support,['discrete','chf'])
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return chffunc[RVar.support.index(value)]
                    

def HF(RVar,value=x):
    """
    Procedure Name: HF
    Purpose: Compute the hf of a random variable
    Arguments:  1. RVar: A random variable
                2. value: An integer or floating point number
                    (optional)
    Output:     1. HF of a random variable (if value not specified)
                2. Value of the HF at a given point
                    (if value is specified)
    """
    
    # Check to make sure the value given is within the random
    #   variable's support
    if value.__class__.__name__!='Symbol':
        if value>RVar.support[-1] or value<RVar.support[0]:
            string='Value is not within the support of the random variable'        
            raise RVError(string)

    # If the distribution is continuous, find and return the hf of
    #   the random variable
    if RVar.ftype[0]=='continuous':
        # If the distribution is already a hf, nothing needs to be
        #   done
        if RVar.ftype[1]=='hf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i]:
                        if value<=RVar.support[i+1]:
                            hfvalue=RVar.func[i].subs(x,value)
                            return simplify(hfvalue)
        # If the distribution is in chf form, use differentiation
        #   to find the hf
        if RVar.ftype[1]=='chf':
            X_dummy=CHF(RVar)
            # Generate a list of hf functions
            hflist=[]
            for i in range(len(X_dummy.func)):
                newfunc=diff(X_dummy.func[i],x)
                hflist.append(newfunc)
            if value==x:
                return RV(hflist,X_dummy.support,['continuous','hf'])
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i]:
                        if value<=RVar.support[i+1]:
                            hfvalue=hflist[i].subs(x,value)
                            return simplify(hfvalue)
        # In all other cases, use the pdf and the sf to find the hf
        else:
            X_pdf=PDF(RVar).func
            X_sf=SF(RVar).func
            # Create a list of hf functions
            hflist=[]
            for i in range(len(RVar.func)):
                hfunc=(X_pdf[i])/(X_sf[i])
                hflist.append(simplify(hfunc))
            if value==x:
                return RV(hflist,RVar.support,['continuous','hf'])
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i]:
                        if value<=RVar.support[i+1]:
                            hfvalue=hflist[i].subs(x,value)
                            return simplify(hfvalue)

    # If the distribution is a discrete function, find and return the hf of
    #   the random variable
    if RVar.ftype[0]=='Discrete':
        # If the distribution is already a hf, nothing needs to be
        #   done
        if RVar.ftype[1]=='hf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i]:
                        if value<=RVar.support[i+1]:
                            hfvalue=RVar.func[i].subs(x,value)
                            return simplify(hfvalue)
        # In all other cases, use the pdf and the sf to find the hf
        else:
            X_pdf=PDF(RVar).func
            X_sf=SF(RVar).func
            # Create a list of hf functions
            hflist=[]
            for i in range(len(RVar.func)):
                hfunc=(X_pdf[i])/(X_sf[i])
                hflist.append(simplify(hfunc))
            if value==x:
                return RV(hflist,RVar.support,['Discrete','hf'])
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i]:
                        if value<=RVar.support[i+1]:
                            hfvalue=hflist[i].subs(x,value)
                            return simplify(hfvalue)

    # If the random variable is discrete, find and return the hf
    if RVar.ftype[0]=='discrete':
        # If the distribution is already a hf, nothing needs
        #   to be done
        if RVar.ftype[1]=='hf':
            if value==x:
                return RVar
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return RVar.func[RVar.support.index(value)]
        # Otherwise, use the pdf and sf to find the hf
        else:
            X_pdf=PDF(RVar)
            X_sf=SF(RVar)
            hffunc=[]
            for i in range(len(X_pdf.func)):
                hffunc.append(X_pdf.func[i]/X_sf.func[i])
            if value==x:
                return RV(hffunc,X_pdf.support,['discrete','hf'])
            if value!=x:
                if value not in X_pdf.support:
                    return 0
                else:
                    return hffunc[X_pdf.support.index(value)]


def IDF(RVar,value=x):
    """
    Procedure Name: IDF
    Purpose: Compute the idf of a random variable
    Arguments:  1. RVar: A random variable
                2. value: An integer or floating point number
                    (optional)
    Output:     1. IDF of a random variable (if value not specified)
                2. Value of the IDF at a given point
                    (if value is specified)
    """
    
    # Check to make sure the value given is within the random
    #   variable's support
    if value.__class__.__name__!='Symbol':
        if value>RVar.support[-1] or value<RVar.support[0]:
            string='Value is not within the support of the random variable'        
            raise RVError(string)
    # If the distribution is continuous, find and return the idf
    #   of the random variable
    if RVar.ftype[0]=='continuous':
        if value==x:
            if RVar.ftype[1]=='idf':
                return self
            # Convert the random variable to its CDF form
            X_dummy=CDF(RVar)
            # Create values used to check for correct inverse
            check=[]
            for i in range(len(X_dummy.support)-1):
                if X_dummy.support[i]==-oo and X_dummy.support[i+1]==oo:
                    check.append(0)
                elif X_dummy.support[i]==-oo and X_dummy.support[i+1]!=oo:
                    check.append(X_dummy.support[i+1]-1)
                elif X_dummy.support[i]!=-oo and X_dummy.support[i+1]==oo:
                    check.append(X_dummy.support[i]+1)
                else:
                    check.append((X_dummy.support[i]+
                                  X_dummy.support[i+1])/2)
            # Use solve to create a list of candidate inverse functions
            # Check to see which of the candidate inverse functions is correct
            idffunc=[]
            for i in range(len(X_dummy.func)):
                invlist=solve(X_dummy.func[i]-t,x)
                if len(invlist)==1:
                    idffunc.append(invlist[0])
                else:
                    # The flag is used to determine if two separate inverses
                    #   could represent the inverse of the CDF. If this is the
                    #   case, an exception is raised
                    flag=False
                    for j in range(len(invlist)):
                        val=invlist[j].subs(t,X_dummy.func[i].subs(x,check[i])).evalf()
                        if abs(val-check[i])<.00001:
                            if flag==True:
                                raise RVError('Could not find the correct inverse')
                            idffunc.append(invlist[j])
                            flag=True
            # Create a list of supports for the IDF
            idfsup=[]
            for i in range(len(X_dummy.support)):
                idfsup.append(CDF(X_dummy,X_dummy.support[i]))
            # Replace t with x
            idffunc2=[]
            for i in range(len(idffunc)):
                func=idffunc[i].subs(t,x)
                idffunc2.append(simplify(func))
            # Return the IDF
            return RV(idffunc2,idfsup,['continuous','idf'])
                    
            
        # If a value is specified, use the newton-raphson method to generate a random variate
        if value!=x:
            X_dummy=IDF(RVar)
            for i in range(len(X_dummy.support)):
                if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                    idfvalue=X_dummy.func[i].subs(t,value)
                    return simplify(idfvalue)
            #varlist=RVar.variate(s=value)
            #return varlist[0]
                
    # If the distribution is a discrete function, find and return the idf
    #   of the random variable
    if RVar.ftype[0]=='Discrete':
        if value==x:
            if RVar.ftype[1]=='idf':
                return self
            # Convert the random variable to its CDF form
            X_dummy=CDF(RVar)
            # Create values used to check for correct inverse
            check=[]
            for i in range(len(X_dummy.support)-1):
                if X_dummy.support[i]==-oo and X_dummy.support[i+1]==oo:
                    check.append(0)
                elif X_dummy.support[i]==-oo and X_dummy.support[i+1]!=oo:
                    check.append(X_dummy.support[i+1]-1)
                elif X_dummy.support[i]!=-oo and X_dummy.support[i+1]==oo:
                    check.append(X_dummy.support[i]+1)
                else:
                    check.append((X_dummy.support[i]+
                                  X_dummy.support[i+1])/2)
            # Use solve to create a list of candidate inverse functions
            # Check to see which of the candidate inverse functions is correct
            idffunc=[]
            for i in range(len(X_dummy.func)):
                invlist=solve(X_dummy.func[i]-t,x)
                if len(invlist)==1:
                    idffunc.append(invlist[0])
                else:
                    # The flag is used to determine if two separate inverses
                    #   could represent the inverse of the CDF. If this is the
                    #   case, an exception is raised
                    flag=False
                    for j in range(len(invlist)):
                        val=invlist[j].subs(t,X_dummy.func[i].subs(x,check[i])).evalf()
                        if abs(val-check[i])<.00001:
                            if flag==True:
                                raise RVError('Could not find the correct inverse')
                            idffunc.append(invlist[j])
                            flag=True
            # Create a list of supports for the IDF
            idfsup=[]
            for i in range(len(X_dummy.support)):
                idfsup.append(CDF(X_dummy,X_dummy.support[i]))
            # Replace t with x
            idffunc2=[]
            for i in range(len(idffunc)):
                func=idffunc[i].subs(t,x)
                idffunc2.append(simplify(func))
            # Return the IDF
            return RV(idffunc2,idfsup,['Discrete','idf'])
                    
            
        # If a value is specified, use the newton-raphson method to generate a random variate
        if value!=x:
            X_dummy=IDF(RVar)
            for i in range(len(X_dummy.support)):
                if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                    idfvalue=X_dummy.func[i].subs(t,value)
                    return simplify(idfvalue)
            #varlist=RVar.variate(s=value)
            #return varlist[0]

    # If the distribution is discrete, find and return the idf of the random variable
    if RVar.ftype[0]=='discrete':
        # If the distribution is already an idf, nothing needs to be done
        if RVar.ftype[1]=='idf':
            if value==x:
                return RVar
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return RVar.func[RVar.support.index(value)]
        # Otherwise, find the cdf, and then invert it
        else:
            # If the distribution is a chf or hf, convert to an sf first
            if RVar.ftype[1]=='chf' or RVar.ftype[1]=='hf':
                X_dummy0=SF(RVar)
                X_dummy=CDF(X_dummy0)
            else:
               X_dummy=CDF(RVar)
            if value==x:
                return RV(X_dummy.support,X_dummy.func,['discrete','idf'])
            if value!=x:
                if value not in RVar.func:
                    return 0
                else:
                    return RVar.support[RVar.func.index(value)]
            


def PDF(RVar,value=x):
    """
    Procedure Name: PDF
    Purpose: Compute the pdf of a random variable
    Arguments:  1. RVar: A random variable
                2. value: An integer or floating point number (optional)
    Output:     1. PDF of a random variable (if value not specified)
                2. Value of the PDF at a given point (if value is specified)
    """
    
    # Check to make sure the value given is within the random
    #   variable's support
    if value.__class__.__name__!='Symbol':
        if value>RVar.support[-1] or value<RVar.support[0]:
            string='Value is not within the support of the random variable'        
            raise RVError(string)
    
    # If the distribution is continuous, find and return the pdf of the random variable
    if RVar.ftype[0]=='continuous':
        # If the distribution is already a pdf, nothing needs to be done
        if RVar.ftype[1]=='pdf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                        pdfvalue=RVar.func[i].subs(x,value)
                        return simplify(pdfvalue)
        # If the distribution is a hf or chf, use integration to find the pdf
        if RVar.ftype[1]=='hf' or RVar.ftype[1]=='chf':
            X_dummy=HF(RVar)
            # Substitute the dummy variable 't' into the hazard function
            hfsubslist=[]
            for i in range(len(X_dummy.func)):
                newfunc=X_dummy.func[i].subs(x,t)
                hfsubslist.append(newfunc)
            # Integrate the hazard function
            intlist=[]
            for i in range(len(hfsubslist)):
                newfunc=integrate(hfsubslist[i],(t,X_dummy.support[i],x))
                # Correct the constant of integration
                if i!=0:
                    const=intlist[i-1].subs(x,X_dummy.support[i])-newfunc.subs(x,X_dummy.support[i])
                    newfunc=newfunc+const
                if i==0:
                    const=0-newfunc.subs(x,X_dummy.support[i])
                    newfunc=newfunc+const
                intlist.append(simplify(newfunc))
            # Multiply to find the pdf
            pdffunc=[]
            for i in range(len(intlist)):
                newfunc=X_dummy.func[i]*exp(-intlist[i])
                pdffunc.append(simplify(newfunc))
            if value==x:
                return RV(pdffunc,RVar.support,['continuous','pdf'])
            if value!=x:
                for i in range(len(X_dummy.support)):
                    if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                        pdfvalue=pdffunc[i].subs(x,value)
                        return simplify(pdfvalue)
        # In all other cases, find the pdf by differentiating the cdf
        else:
            X_dummy=CDF(RVar)
            if value==x:
                pdflist=[]
                for i in range(len(X_dummy.func)):
                    pdflist.append(diff(X_dummy.func[i],x))
                return RV(pdflist,RVar.support,['continuous','pdf'])
            if value!=x:
                for i in range(len(X_dummy.support)):
                    for i in range(len(X_dummy.support)):
                        if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                            pdffunc=diff(X_dummy.func[i],x)
                            pdfvalue=pdffunc.subs(x,value)
                            return simplify(pdfvalue)

    # If the distribution is a discrete function, find and return the pdf
    if RVar.ftype[0]=='Discrete':
        # If the distribution is already a pdf, nothing needs to be done
        if RVar.ftype[1]=='pdf':
            if value==x:
                return RVar
            if value!=x:
                for i in range(len(RVar.support)):
                    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                        pdfvalue=RVar.func[i].subs(x,value)
                        return simplify(pdfvalue)
        # If the distribution is a hf or chf, use summation to find the pdf
        if RVar.ftype[1]=='hf' or RVar.ftype[1]=='chf':
            X_dummy=HF(RVar)
            # Substitute the dummy variable 't' into the hazard function
            hfsubslist=[]
            for i in range(len(X_dummy.func)):
                newfunc=X_dummy.func[i].subs(x,t)
                hfsubslist.append(newfunc)
            # Integrate the hazard function
            sumlist=[]
            for i in range(len(hfsubslist)):
                newfunc=summation(hfsubslist[i],(t,X_dummy.support[i],x))
                # Correct the constant of integration
                if i!=0:
                    const=sumlist[i-1].subs(x,X_dummy.support[i])-newfunc.subs(x,X_dummy.support[i])
                    newfunc=newfunc+const
                if i==0:
                    const=0-newfunc.subs(x,X_dummy.support[i])
                    newfunc=newfunc+const
                intlist.append(simplify(newfunc))
            # Multiply to find the pdf
            pdffunc=[]
            for i in range(len(intlist)):
                newfunc=X_dummy.func[i]*exp(-sumlist[i])
                pdffunc.append(simplify(newfunc))
            if value==x:
                return RV(pdffunc,RVar.support,['Discrete','pdf'])
            if value!=x:
                for i in range(len(X_dummy.support)):
                    if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                        pdfvalue=pdffunc[i].subs(x,value)
                        return simplify(pdfvalue)
        # In all other cases, find the pdf by differentiating the cdf
        else:
            X_dummy=CDF(RVar)
            if value==x:
                pdflist=[]
                # Find the pmf by subtracting CDF(X,x)-CDF(X,x-1)
                for i in range(len(X_dummy.func)):
                    funcX1=X_dummy.func[i]
                    funcX0=X_dummy.func[i].subs(x,x-1)
                    pmf=simplify(funcX1-funcX0)
                    pdflist.append(pmf)
                return RV(pdflist,RVar.support,['Discrete','pdf'])
            if value!=x:
                for i in range(len(X_dummy.support)):
                    for i in range(len(X_dummy.support)):
                        if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                            funcX1=X_dummy.func[i]
                            funcX0=X_dummy.func[i].subs(x,x-1)
                            pmf=simplify(funcX1-funcX0)
                            pdfvalue=pmf.subs(x,value)
                            return simplify(pdfvalue)
                        
    # If the distribution is discrete, find and return the pdf of the random variable
    if RVar.ftype[0]=='discrete':
        # If the distribution is already a pdf, nothing needs to be done
        if RVar.ftype[1]=='pdf':
            if value==x:
                return RVar
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return RVar.func[RVar.support.index(value)]  
        # Otherwise, find the cdf of the random variable, and compute the pdf
        #   by finding differences
        else:
            X_dummy=CDF(RVar)
            pdffunc=[]
            for i in range(len(X_dummy.func)):
                if i==0:
                    pdffunc.append(X_dummy.func[i])
                else:
                    pdffunc.append(X_dummy.func[i]-X_dummy.func[i-1])
            if value==x:
                return RV(pdffunc,X_dummy.support,['discrete','pdf'])
            if value!=x:
                if value not in X_dummy.support:
                    return 0
                else:
                    return pdffunc.func[X_dummy.support.index(value)]

def SF(RVar,value=x):
    """
    Procedure Name: SF
    Purpose: Compute the SF of a random variable
    Arguments:  1. RVar: A random variable
                2. value: An integer or floating point number (optional)
    Output:     1. SF of a random variable (if value not specified)
                2. Value of the SF at a given point (if value is specified)
    """
    
    # Check to make sure the value given is within the random
    #   variable's support
    if value.__class__.__name__!='Symbol':
        if value>RVar.support[-1] or value<RVar.support[0]:
            string='Value is not within the support of the random variable'        
            raise RVError(string)
        
    # If the distribution is continuous, find and return the sf of the random variable
    if RVar.ftype[0]=='continuous':
        # If the distribution is already a sf, nothing needs to be done
        if RVar.ftype[1]=='sf':
            if value==x:
                return RVar
            else:
                return 1-CDF(RVar,value)
                #for i in range(len(RVar.support)):
                #    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                #        sfvalue=RVar.func[i].subs(x,value)
                #        return simplify(sfvalue)
        # If not, then use subtraction to find the sf
        else:
            X_dummy=CDF(RVar)
            # Compute the sf for each segment
            sflist=[]
            for i in range(len(X_dummy.func)):
                sflist.append(1-X_dummy.func[i])
            if value==x:
                return RV(sflist,RVar.support,['continuous','sf'])
            if value!=x:
                return 1-CDF(RVar,value)
                #for i in range(len(X_dummy.support)):
                #    if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                #        sfvalue=sflist[i].subs(x,value)
                #        return simplify(sfvalue)

    # If the distribution is continuous, find and return the sf of the random variable
    if RVar.ftype[0]=='continuous':
        # If the distribution is already a sf, nothing needs to be done
        if RVar.ftype[1]=='sf':
            if value==x:
                return RVar
            else:
                return 1-CDF(RVar,value)
                #for i in range(len(RVar.support)):
                #    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                #        sfvalue=RVar.func[i].subs(x,value)
                #        return simplify(sfvalue)
        # If not, then use subtraction to find the sf
        else:
            X_dummy=CDF(RVar)
            # Compute the sf for each segment
            sflist=[]
            for i in range(len(X_dummy.func)):
                sflist.append(1-X_dummy.func[i])
            if value==x:
                return RV(sflist,RVar.support,['continuous','sf'])
            if value!=x:
                return 1-CDF(RVar,value)
                #for i in range(len(X_dummy.support)):
                #    if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                #        sfvalue=sflist[i].subs(x,value)
                #        return simplify(sfvalue)

    # If the distribution is a discrete function, find and return the sf of the random variable
    if RVar.ftype[0]=='Discrete':
        # If the distribution is already a sf, nothing needs to be done
        if RVar.ftype[1]=='sf':
            if value==x:
                return RVar
            else:
                return 1-CDF(RVar,value)
                #for i in range(len(RVar.support)):
                #    if value>=RVar.support[i] and value<=RVar.support[i+1]:
                #        sfvalue=RVar.func[i].subs(x,value)
                #        return simplify(sfvalue)
        # If not, then use subtraction to find the sf
        else:
            X_dummy=CDF(RVar)
            # Compute the sf for each segment
            sflist=[]
            for i in range(len(X_dummy.func)):
                sflist.append(1-X_dummy.func[i])
            if value==x:
                return RV(sflist,RVar.support,['Discrete','sf'])
            if value!=x:
                return 1-CDF(RVar,value)
                #for i in range(len(X_dummy.support)):
                #    if value>=X_dummy.support[i] and value<=X_dummy.support[i+1]:
                #        sfvalue=sflist[i].subs(x,value)
                #        return simplify(sfvalue)

    # If the distribution is a discrete function, find and return the sf of the random variable
    if RVar.ftype[0]=='discrete':
        # If the distribution is already an sf, nothing needs to be done
        if RVar.ftype[1]=='sf':
            if value==x:
                return RVar
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return RVar.func[RVar.support.index(value)]
        # If the distribution is a chf use exp(-chf) to find sf
        if RVar.ftype[1]=='chf':
            X_dummy=CHF(RVar)
            sffunc=[]
            for i in range(len(X_dummy.func)):
                sffunc.append(exp(-(X_dummy.func[i])))
            if value==x:
                return RV(sffunc,X_dummy.support,['discrete','sf'])
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return sffunc[RVar.support.index(value)]
        # If the distribution is a hf, use bootstrap rv to find sf:
        if RVar.ftype[1]=='hf':
            X_pdf=BootstrapRV(RVar.support)
            X_hf=RVar
            sffunc=[]
            for i in range(len(RVar.func)):
                sffunc.append(X_pdf.func[i]/X_hf.func[i])
            if value==x:
                return RV(sffunc,RVar.support,['discrete','sf'])
            if value!=x:
                if value not in RVar.support:
                    return 0
                else:
                    return sffunc[RVar.support.index(value)]
        # Otherwise, find the cdf of the random variable, and reverse the function
        #   argument
        else:
            X_dummy=CDF(RVar)
            newfunc=[]
            for i in range(len(X_dummy.func)):
                if i==0:
                    newfunc.append(0)
                else:
                    newfunc.append(1-X_dummy.func[i-1])
            Xsf=RV(newfunc,X_dummy.support,['discrete','sf'])
            if value==x:
                return Xsf
            if value!=x:
                if value not in Xsf.support:
                    return 0
                else:
                    return Xsf.func[Xsf.support.index(value)]
        

def BootstrapRV(varlist,symbolic=False):
    """
    Procedure Name: Bootstrap RV
    Purpose: Generate a discrete random variable from a list of variates
    Arguments: 1. varlist: A list of variates
    Output:    1. A discrete random variable, where each element in the given variate
                    list is equally probable
    """
    # Sort the list of variables
    varlist.sort()
    # Find the number of elements in the list of variates
    numel=len(varlist)
    # Use varlist to generate the function and support for the random variable
    #   Count number of times element appears in varlist, divide by number
    #   of elements
    funclist=[]
    supplist=[]
    for i in range(len(varlist)):
        if varlist[i] not in funclist:
            supplist.append(varlist[i])
            funclist.append(varlist.count(varlist[i])/numel)
    # Return the result as a discrete random variable
    return RV(funclist,supplist,['discrete','pdf'])

def Convert(RVar,inc=1):
    """
    Procedure Name: Convert
    Purpose: Convert a discrete random variable from functional to explicit form
    Arguments:  1. RVar: A functional discrete random variable
                2. inc: An increment value
    Output:     1. A discrete random variable in explicit form
    """
    # If the random variable is not in functional form, return
    #   an error
    if RVar.ftype[0]!='Discrete':
        raise RVError('The random variable must be Discrete')
    # If the rv has infinite support, return an error
    if (oo or -oo) in RVar.support:
        raise RVError('Convert does not work for infinite support')
    # Create the support of explicit discrete rv
    i=RVar.support[0]
    discrete_supp=[]
    while i<=RVar.support[1]:
        discrete_supp.append(i)
        i+=inc
    # Create the function values for the explicit rv
    discrete_func=[]
    for i in range(len(discrete_supp)):
        val=RVar.func[0].subs(x,discrete_supp[i])
        discrete_func.append(val)
    # Return the random variable in discrete form
    return RV(discrete_func,discrete_supp,
              ['discrete',RVar.ftype[1]])

"""
Procedures on One Random Variable

Procedures:
    1. ConvolutionIID(RVar,n)
    2. CoefOfVar(RVar)
    3. ExpectedValue(RVar,gX)
    4. Entropy(RVar)
    5. Kurtosis(RVar)
    6. MaximumIID(RVar,n)
    7. Mean(RVar)
    8. MeanDiscrete(RVar)
    9. MGF(RVar)
    10. MinimumIID(RVar,n)
    11. OrderStat(RVar,n,r)
    12. ProductIID(RVar,n)
    13. Skewness(RVar)
    14. Transform(RVar,gX)
    15. Truncate(RVar,[lw,up])
    16. Variance(RVar)
    17. VarDiscrete(RVar)
"""

def ConvolutionIID(RVar,n):
    """
    Procedure Name: ConvolutionIID
    Purpose: Compute the convolution of n iid random variables
    Arguments:  1. RVar: A random variable
                2. n: an integer
    Output:     1. The convolution of n iid random variables
    """
    # Check to make sure n is an integer
    if type(n)!=int:
        raise RVError('The second argument must be an integer')

    # Compute the iid convolution
    X_dummy=PDF(RVar)
    X_final=X_dummy
    for i in range(n-1):
        X_final+=X_dummy
    return X_final

def CoefOfVar(RVar):
    """
    Procedure Name: CoefOfVar
    Purpose: Compute the coefficient of variation of a random variable
    Arguments:  1. RVar: A random variable
    Output:     1. The coefficient of variation
    """
    # Compute the coefficient of varation
    expect=Mean(RVar)
    sig=Variance(RVar)
    cov=(sqrt(sig))/expect
    return simplify(cov)

def ExpectedValue(RVar,gX=x):
    """
    Procedure Name: ExpectedValue
    Purpose: Computes the expected value of X
    Arguments:  1. RVar: A random variable
                2. gX: A transformation of x
    Output:     1. E(gX)
    """
    # Conver the random variable to its PDF form
    fx=PDF(RVar)
    # If the distribution is continuous, compute the expected
    #   value
    if fx.ftype[0]=='continuous':
        Expect=0
        for i in range(len(fx.func)):
            Expect+=integrate(gX*fx.func[i],
                              (x,fx.support[i],fx.support[i+1]))
        return simplify(Expect)

    # If the distribution is a discrete function, compute the expected
    #   value
    if fx.ftype[0]=='Discrete':
        Expect=0
        for i in range(len(fx.func)):
            Expect+=summation(gX*fx.func[i],
                              (x,fx.support[i],fx.support[i+1]))
        return simplify(Expect)

    # If the distribution is discrete, compute the expected
    #   value
    if fx.ftype[0]=='discrete':
        # Transform the random variable, and then use the
        #   mean procedure to find the expected value
        fx_trans=Transform(fx,[[gX],[-oo,oo]])
        Expect=MeanDiscrete(fx_trans)
        return simplify(Expect)

def Entropy(RVar):
    """
    Procedure Name: Entropy
    Purpose: Compute the entory of a random variable
    Arguments:  1. RVar: A random variable
    Output:     1. The entropy of a random variable
    """
    entropy=ExpectedValue(RVar,log(x,2))
    return simplify(entropy)

def Kurtosis(RVar):
    """
    Procedure Name: Kurtosis
    Purpose: Compute the Kurtosis of a random variable
    Arguments:  1. RVar: A random variable
    Output:     1. The kurtosis of a random variable
    """
    # Compute the kurtosis
    expect=Mean(RVar)
    sig=sqrt(Variance(RVar))
    Term1=ExpectedValue(RVar,x**4)
    Term2=4*expect*ExpectedValue(RVar,x**3)
    Term3=6*(expect**2)*ExpectedValue(RVar,x**2)
    Term4=3*expect**4
    kurt=(Term1-Term2+Term3-Term4)/(sig**4)
    return simplify(kurt)

def MaximumIID(RVar,n):
    """
    Procedure Name: MaximumIID
    Purpose: Comput the maximum of n iid random variables
    Arguments:  1. RVar: A random variable
                2. n: an integer
    Output:     1. The maximum of n iid random variables
    """
    # Check to make sure n is an integer
    if type(n)!=int:
        raise RVError('The second argument must be an integer')

    # Compute the iid maximum
    X_dummy=RVar
    X_final=X_dummy
    for i in range(n-1):
        X_final=Maximum(X_final,X_dummy)
    return X_final

def Mean(RVar):
    """
    Procedure Name: Mean
    Purpose: Compute the mean of a random variable
    Arguments: 1. RVar: A random variable
    Output:    1. The mean of a random variable
    """
    # Find the PDF of the random variable
    X_dummy=PDF(RVar)
    # If the random variable is continuous, find and return the mean
    if RVar.ftype[0]=='continuous':
        # Create list of x*f(x)
        meanfunc=[]
        for i in range(len(X_dummy.func)):
            meanfunc.append(x*X_dummy.func[i])
        # Integrate to find the mean
        meanval=0
        for i in range(len(X_dummy.func)):
            val=integrate(meanfunc[i],(x,X_dummy.support[i],X_dummy.support[i+1]))
            meanval+=val
        return simplify(meanval)

    # If the random variable is a discrete function, find and return the mean
    if RVar.ftype[0]=='Discrete':
        # Create list of x*f(x)
        meanfunc=[]
        for i in range(len(X_dummy.func)):
            meanfunc.append(x*X_dummy.func[i])
        # Sum to find the mean
        meanval=0
        for i in range(len(X_dummy.func)):
            val=summation(meanfunc[i],(x,X_dummy.support[i],X_dummy.support[i+1]))
            meanval+=val
        return simplify(meanval)

    # If the random variable is discrete, find and return the variance
    if RVar.ftype[0]=='discrete':
        return MeanDiscrete(RVar)
        #
        # Legacy mean code ... update uses faster numpy implementation
        #
        # Create a list of x*f(x)
        #meanlist=[]
        #for i in range(len(X_dummy.func)):
        #    meanlist.append(X_dummy.func[i]*X_dummy.support[i])
        # Sum to find the mean
        #meanval=0
        #for i in range(len(meanlist)):
        #    meanval+=meanlist[i]
        #return simplify(meanval)


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
def MGF(RVar):
    """
    Procedure Name: MGF
    Purpose: Compute the moment generating function of a random variable
    Arguments:  1. RVar: A random variable
    Output:     1. The moment generating function
    """
    mgf=ExpectedValue(RVar,exp(t*x))
    return simplify(mgf)

def MinimumIID(RVar,n):
    """
    Procedure Name: MinimumIID
    Purpose: Compute the minimum of n iid random variables
    Arguments:  1. RVar: A random variable
                2. n: an integer
    Output:     1. The minimum of n iid random variables
    """
    # Check to make sure n is an integer
    if type(n)!=int:
        raise RVError('The second argument must be an integer')

    # Compute the iid minimum
    X_dummy=RVar
    X_final=X_dummy
    for i in range(n-1):
        X_final=Minimum(X_final,X_dummy)
    return X_final

def NextCombination(Previous,N):
    """
    Procedure Name: NextCombination
    Purpose: Generates the next lexicographical combination of
                size n. Designed for use in the OrderStat
                procedure.
    Arguments:  1. Previous: A list
                2. N: A positive integer
    Output:     1. The next combination
    """
    # Initialize the Next list
    Next=[]
    for i in range(len(Previous)):
        Next.append(Previous[i])
    n=len(Next)
    # If the value in the final position of the combination is not the
    #   maximum value it can attain, N, then increment it by 1
    if Next[n-1]!=N:
        Next[n-1]+=1
    # If the final position in the combination is already at its maximum
    #   value, then move left trhough the combination and find the next
    #   possible value that can be incremented
    else:
        MoveLeft=True
        for i in reversed(range(1,n)):
            indx=i-1
            if Next[indx]<N+i-n:
                Next[indx]+=1
                for j in range(1,(n-i+1)):
                    Next[indx+j]=Next[(indx+j)-1]+1
                MoveLeft=False
            if MoveLeft==False:
                break            
    return(Next)

def NextPermutation(Previous):
    """
    Procedure Name: NextPermutation
    Purpose: Generate the next lexicographical permutation of
                the given list. Designed for use in the OrderStat
                procedure.
    Arguments:  1. Previous: A list
    Output:     1. The next permutation
    """
    # Initialize the Next list
    Next=[]
    Temp2=[]
    for i in range(len(Previous)):
        Next.append(Previous[i])
        Temp2.append(None)
    n=len(Previous)
    flag=False

    # Find the largest index value i for which Next[i]<Next[i+1]
    for i in reversed(range(1,n)):
        while flag==False:
            indx=i-1
            if Next[indx]<Next[indx+1]:
                flag=True
                OrigVal=Next[indx]
                SwapIndex=indx+1
            # Find the smallest value Next[j] for which Next[i]<Next[j]
            #   and i<j
            for j in reversed(range(SwapIndex,n)):
                if Next[j]<Next[SwapIndex]:
                    if Next[j]>OrigVal:
                        SwapIndex=j
            Temp1=Next[SwapIndex]
            Swap=Next[indx]
            Next[SwapIndex]=Swap
            Next[indx]=Temp1
            # Reverse the order of the values to the right of the leftmost
            #   swapped value
            for k in range(indx+1,n):
                Temp2[k]=Next[k]
            for m in range(indx+1,n):
                Next[m]=Temp2[n+indx-m]
    return(Next)
            
def OrderStat(RVar,n,r,replace='w'):
    """
    Procedure Name: OrderStat
    Purpose: Compute the distribution of the rth order statistic
                from a sample puplation of n
    Arguments:  1. RVar: A random variable
                2. n: The number of items randomly drawn from the rv
                3. r: The index of the order statistic
    Output:     1. The desired r out of n OrderStatistic
    """
    if r>n:
        raise RVError('The index cannot be greater than the sample size')

    # If the distribution is continuous, find and return the value of the
    #   order statistic
    if RVar.ftype[0]=='continuous':
        # Compute the PDF, CDF and SF of the random variable
        pdf_dummy=PDF(RVar)
        cdf_dummy=CDF(RVar)
        sf_dummy=SF(RVar)
        # Compute the factorial constant
        const=(factorial(n))/(factorial(r-1)*factorial(n-r))
        # Compute the distribution of the order statistic for each
        #   segment
        ordstat_func=[]
        for i in range(len(RVar.func)):
            fx=pdf_dummy.func[i]
            Fx=cdf_dummy.func[i]
            Sx=sf_dummy.func[i]
            ordfunc=const*(Fx**(r-1))*(Sx**(n-r))*fx
            ordstat_func.append(simplify(ordfunc))
        # Return the distribution of the order statistic
        return RV(ordstat_func,RVar.support,['continuous','pdf'])

    # If the distribution is continuous, find and return the value of
    #   the order statistic
    if RVar.ftype[0]=='discrete':
        #
        # For discrete distributions:
        #   1. Need to add support for symbolic discrete distributions
        #   2. Need to add procedure that converts from dot form
        #       to no-dot form
        #   -- This will allow for the use of discrete distributions
        #       such as the Poisson and Binomial distributions
        #
        if replace not in ['w','wo']:
            raise RVError('Replace must be w or wo')
        fx=PDF(RVar)
        Fx=CDF(RVar)
        Sx=SF(RVar)
        N=len(fx.support)
        # With replacement
        if replace=='w':
            # Numeric PDF
            if type(RVar.func[0])!=Symbol:
                # If N is one, return the order stat
                if N==1:
                    return RV(1,RVar.support,['discrete','pdf'])
                # Add the first term
                else:
                    OSproblist=[]
                    os_sum=0
                    for w in range(n-r+1):
                        val=(binomial(n,w)*
                             (fx.func[0]**(n-w))*
                             (Sx.func[1]**(w)))
                        os_sum+=val
                    OSproblist.append(os_sum)
                # Add term 2 through N-1
                for k in range(2,N):
                    os_sum=0
                    for w in range(n-r+1):
                        for u in range(r):
                            val=(factorial(n)/
                                 (factorial(u)*factorial(n-u-w)
                                  *factorial(w))*
                                 (Fx.func[k-2]**u)*
                                 (fx.func[k-1]**(n-u-w))*
                                 (Sx.func[k]**(w)))
                            os_sum+=val
                    OSproblist.append(os_sum)
                # Add term N
                os_sum=0
                for u in range(r):
                    val=(binomial(n,u)*
                         (Fx.func[N-2]**u)*
                         (fx.func[N-1]**(n-u)))
                    os_sum+=val
                OSproblist.append(os_sum)
                return RV(OSproblist,RVar.support,['discrete','pdf'])

        if replace=='wo':
            if n>4:
                print 'When sampling without replacement, n must be'
                print 'less than 4'
                raise RVError('n greater than 4')
            # Determine if the PDF has equally likely probabilities
            EqLike=True
            for i in range(len(fx.func)):
                if fx.func[0]!=fx.func[i]:
                    EqLike=False
                if EqLike==False:
                    break
            # Create blank order stat function list
            fxOS=[]
            for i in range(len(fx.func)):
                fxOS.append(0)
            # If the probabilities are equally likely
            if EqLike==True:
                # Need to add algorithm for symbolic 'r'
                for i in range(r,(N-n+r+1)):
                    indx=i-1
                    val=((binomial(i-1,r-1)*
                         binomial(1,1)*
                         binomial(N-i,n-r))/
                         (binomial(N,n)))
                    fxOS[indx]=val
                return RV(fxOS,fx.support,['discrete','pdf'])
            # If the probabilities are not equally likely
            elif EqLike==False:
                # If the sample size is 1
                if n==1:
                    fxOS=[]
                    for i in range(len(fx.func)):
                        fxOS.append(fx.func[i])
                    return(fxOS,fx.support,['discrete','pdf'])
                elif n==N:
                    fxOS[n-1]=1
                    return RV(fxOS,fx.support,['discrete','pdf'])
                else:
                    # Create null ProbStorage array of size nXN
                    # Initialize to contain all zeroes
                    print n,N
                    ProbStorage=[]
                    for i in range(n):
                        row_list=[]
                        for j in range(N):
                            row_list.append(0)
                        ProbStorage.append(row_list)
                    # Create the first lexicographical combo of
                    #   n items
                    combo=range(1,n+1)
                    for i in range(1,(binomial(N,n)+1)):
                        # Assign perm as the current combo
                        perm=[]
                        for j in range(len(combo)):
                            perm.append(combo[j])
                        # Compute the probability of obtaining the
                        #   given permutation
                        for j in range(1,factorial(n)+1):
                            PermProb=fx.func[perm[0]]
                            cumsum=fx.func[perm[0]]
                            for m in range(1,n):
                                PermProb*=fx.func[perm[m]]/(1-cumsum)
                                cumsum+=fx.func[perm[m]]
                            print perm,PermProb,cumsum
                            # Order each permutation and determine
                            #   which value sits in the rth
                            #   ordered position
                            orderedperm=[]
                            for m in range(len(perm)):
                                orderedperm.append(perm[m])
                            orderedperm.sort()
                            for m in range(n):
                                for k in range(N):
                                    if orderedperm[m]==k+1:
                                        ProbStorage[m][k]=(PermProb+
                                                           ProbStorage[m][k])
                            # Find the next lexicographical permutation
                            perm=NextPermutation(perm)
                        # Find the next lexicographical combination
                        combo=NextCombination(combo,N)

def ProductIID(RVar,n):
    """
    Procedure Name: ProductIID
    Purpose: Compute the product of n iid random variables
    Arguments:  1. RVar: A random variable
                2. n: an integer
    Output:     1. The product of n iid random variables
    """
    # Check to make sure n is an integer
    if type(n)!=int:
        raise RVError('The second argument must be an integer')

    # Compute the iid convolution
    X_dummy=PDF(RVar)
    X_final=X_dummy
    for i in range(n-1):
        X_final*=X_dummy
    return X_final

def Skewness(RVar):
    """
    Procedure Name: Skewness
    Purpose: Compute the skewness of a random variable
    Arguments:  1. RVar: A random variable
    Output:     1. The skewness of the random variable
    """
    # Compute the skewness
    expect=Mean(RVar)
    sig=sqrt(Variance(RVar))
    Term1=ExpectedValue(RVar,x**3)
    Term2=3*expect*ExpectedValue(RVar,x**2)
    Term3=2*expect**3
    skew=(Term1-Term2+Term3)/(sig**3)
    return simplify(skew)
                            

def Transform(RVar,gXt):
    """
    Procedure Name: Transform
    Purpose: Compute the transformation of a random variable
                by a a function g(x)
    Arguments:  1. RVar: A random variable
                2. gX: A transformation in list of two lists format
    Output:     1. The transformation of RVar       
    """
    
    # Check to make sure support of transform is in ascending order
    for i in range(len(gXt[1])-1):
        if gXt[1][i]>gXt[1][i+1]:
            raise RVError('Transform support is not in ascending order')

    # Convert the RV to its PDF form
    X_dummy=PDF(RVar)
            
    # If the distribution is continuous, find and return the transformation
    if RVar.ftype[0]=='continuous':
        # Adjust the transformation to include the support of the random
        #   variable
        gXold=[]
        for i in range(len(gXt)):
            gXold.append(gXt[i])
        gXsupp=[]
        for i in range(len(gXold[1])):
            gXsupp.append(gXold[1][i])
        # Add the support of the random variable into the support
        #   of the transformation
        for i in range(len(X_dummy.support)):
            if X_dummy.support[i] not in gXsupp:
                gXsupp.append(X_dummy.support[i])
        gXsupp.sort()
        # Find which segment of the transformation applies, and add it
        #   to the transformation list
        gXfunc=[]
        for i in range(len(gXsupp)-1):
            for j in range(len(gXold[0])):
                if gXsupp[i]>=gXold[1][j]:
                    if gXsupp[i]<=gXold[1][j+1]:
                        gXfunc.append(gXold[0][j])
                        break
        # Set the adjusted transformation as gX
        gX=[]
        gX.append(gXfunc)
        gX.append(gXsupp)
        # If the support of the transformation does not match up with the
        #   support of the RV, adjust the support of the transformation
        
        # Traverse list to find elements that are not within the support
        #   of the rv
        for i in range(len(gX[1])):
            if gX[1][i]<X_dummy.support[0]:
                gX[1][i]=X_dummy.support[0]
            if gX[1][i]>X_dummy.support[len(X_dummy.support)-1]:
                gX[1][i]=X_dummy.support[len(X_dummy.support)-1]
        # Delete segments of the transformation that will not be used
        for i in range(len(gX[0])-1):
            if gX[1][i]==gX[1][i+1]:
                gX[0].remove(gX[0][i])
                gX[1].remove(gX[1][i+1])
        # Create a list of mappings x->g(x)
        mapping=[]
        for i in range(len(gX[0])):
            mapping.append([gX[0][i].subs(x,gX[1][i]),
                            gX[0][i].subs(x,gX[1][i+1])])
        # Create the support for the transformed random variable
        trans_supp=[]
        for i in range(len(mapping)):
            for j in range(2):
                if mapping[i][j] not in trans_supp:
                    trans_supp.append(mapping[i][j])
        trans_supp.sort()
        # Find which segment of the transformation each transformation
        #   function applies to
        applist=[]
        for i in range(len(mapping)):
            temp=[]
            for j in range(len(trans_supp)-1):
                if min(mapping[i])<=trans_supp[j]:
                    if max(mapping[i])>=trans_supp[j+1]:
                        temp.append(j)
            applist.append(temp)
        # Find the appropriate inverse for each g(x)
        ginv=[]
        for i in range(len(gX[0])):
            # Find the 'test point' for the inverse
            if [gX[1][i],gX[1][i+1]]==[-oo,oo]:
                c=0
            elif gX[1][i]==-oo and gX[1][i+1]!=oo:
                c=gX[1][i+1]-1
            elif gX[1][i]!=-oo and gX[1][i+1]==oo:
                c=gX[1][i]+1
            else:
                c=(gX[1][i]+gX[1][i+1])/2
            # Create a list of possible inverses
            invlist=solve(gX[0][i]-t,x)
            # Use the test point to determine the correct inverse
            for j in range(len(invlist)):
                # If g-1(g(c))=c, then the inverse is correct
                if invlist[j].subs(t,gX[0][i].subs(x,c))==c:
                    ginv.append(invlist[j])
        # Find the transformation function for each segment
        seg_func=[]
        for i in range(len(X_dummy.func)):
            # Only find transformation for applicable segments
            for j in range(len(gX[0])):
                if gX[1][j]>=X_dummy.support[i]:
                    if gX[1][j+1]<=X_dummy.support[i+1]:
                        if type(X_dummy.func[i]) not in [float,int]:
                            tran=X_dummy.func[i].subs(x,ginv[j])*diff(ginv[j],t)
                        else:
                            tran=X_dummy.func[i]*diff(ginv[j],t)
                        seg_func.append(tran)
        # Sum the transformations for each piece of the transformed
        #   random variable
        trans_func=[]
        for i in range(len(trans_supp)-1):
            h=0
            for j in range(len(seg_func)):
                if i in applist[j]:
                    if mapping[j][0]<mapping[j][1]:
                        h=h+seg_func[j]
                    else:
                        h=h-seg_func[j]
            trans_func.append(h)
        # Substitute x into the transformed random variable
        trans_func2=[]
        for i in range(len(trans_func)):
            trans_func2.append(simplify(trans_func[i].subs(t,x)))
        # Create and return the random variable
        return RV(trans_func2,trans_supp,['continuous','pdf'])

    # If the distribution is discrete, find and return the transformation
    if RVar.ftype[0]=='discrete':
        gX=gXt
        trans_sup=[]
        # Find the portion of the transformation each element
        #   in the random variable applies to, and then transform it
        for i in range(len(X_dummy.support)):
            for j in range(len(gX[1])-1):
                if X_dummy.support[i]>=gX[1][j]:
                    if X_dummy.support[i]<=gX[1][j+1]:
                        trans_sup.append(gX[0][j].subs(x,X_dummy.support[i]))
        # Sort the function and support lists for the convolution
        sortlist=zip(trans_sup,X_dummy.func)
        sortlist.sort()
        translist=[]
        funclist=[]
        for i in range(len(sortlist)):
            translist.append(sortlist[i][0])
            funclist.append(sortlist[i][1])
        # Combine redundant elements in the list
        translist2=[]
        funclist2=[]
        for i in range(len(translist)):
            if translist[i] not in translist2:
                translist2.append(translist[i])
                funclist2.append(funclist[i])
            elif translist[i] in translist2:
                idx=translist2.index(translist[i])
                funclist2[idx]+=funclist[i]
        # Return the transformed random variable
        return RV(funclist2,translist2,['discrete','pdf'])

def Truncate(RVar,supp):
    """
    Procedure Name: Truncate
    Purpose: Truncate a random variable
    Arguments: 1. RVar: A random variable
               2. supp: The support of the truncated random variable
    Output:    1. A truncated random variable
    """
    # Check to make sure the support of the truncated random
    #   variable is given in ascending order
    if supp[0]>supp[1]:
        raise RVError('The support must be given in ascending order')
    
    # Conver the random variable to its pdf form
    X_dummy=PDF(RVar)
    cdf_dummy=CDF(RVar)

    # If the random variable is continuous, find and return
    #   the truncated random variable
    if RVar.ftype[0]=='continuous':
        # Find the area of the truncated random variable
        area=CDF(cdf_dummy,supp[1])-CDF(cdf_dummy,supp[0])
        # Cut out parts of the distribution that don't fall
        #   within the new limits
        for i in range(len(X_dummy.func)):
            if supp[0]>=X_dummy.support[i]:
                if supp[0]<=X_dummy.support[i+1]:
                    lwindx=i
            if supp[1]>=X_dummy.support[i]:
                if supp[1]<=X_dummy.support[i+1]:
                    upindx=i
        truncfunc=[]
        for i in range(len(X_dummy.func)):
            if i>=lwindx and i<=upindx:
                truncfunc.append(X_dummy.func[i]/area)
        truncsupp=[supp[0]]
        upindx+=1
        for i in range(len(X_dummy.support)):
            if i>lwindx and i<upindx:
                truncsupp.append(X_dummy.support[i])
        truncsupp.append(supp[1])
        # Return the truncated random variable
        return RV(truncfunc,truncsupp,['continuous','pdf'])

    # If the random variable is a discrete function, find and return
    #   the truncated random variable
    if RVar.ftype[0]=='Discrete':
        # Find the area of the truncated random variable
        area=CDF(cdf_dummy,supp[1])-CDF(cdf_dummy,supp[0])
        # Cut out parts of the distribution that don't fall
        #   within the new limits
        for i in range(len(X_dummy.func)):
            if supp[0]>=X_dummy.support[i]:
                if supp[0]<=X_dummy.support[i+1]:
                    lwindx=i
            if supp[1]>=X_dummy.support[i]:
                if supp[1]<=X_dummy.support[i+1]:
                    upindx=i
        truncfunc=[]
        for i in range(len(X_dummy.func)):
            if i>=lwindx and i<=upindx:
                truncfunc.append(X_dummy.func[i]/area)
        truncsupp=[supp[0]]
        upindx+=1
        for i in range(len(X_dummy.support)):
            if i>lwindx and i<upindx:
                truncsupp.append(X_dummy.support[i])
        truncsupp.append(supp[1])
        # Return the truncated random variable
        return RV(truncfunc,truncsupp,['Discrete','pdf'])

    # If the distribution is discrete, find and return the
    #   truncated random variable
    if RVar.ftype[0]=='discrete':
        # Find the area of the truncated random variable
        area=0
        for i in range(len(X_dummy.support)):
            if X_dummy.support[i]>=supp[0]:
                if X_dummy.support[i]<=supp[1]:
                    area+=X_dummy.func[i]
        # Truncate the random variable and find the probability
        #   at each point
        truncfunc=[]
        truncsupp=[]
        for i in range(len(X_dummy.support)):
            if X_dummy.support[i]>=supp[0]:
                if X_dummy.support[i]<=supp[1]:
                    truncfunc.append(X_dummy.func[i]/area)
                    truncsupp.append(X_dummy.support[i])
        # Return the truncated random variable
        return RV(truncfunc,truncsupp,['discrete','pdf'])     


def Variance(RVar):
    """
    Procedure Name: Variance
    Purpose: Compute the variance of a random variable
    Arguments: 1. RVar: A random variable
    Output:    1. The variance of a random variable
    """
    # Find the PDF of the random variable
    X_dummy=PDF(RVar)
    # If the random variable is continuous, find and return the variance
    if RVar.ftype[0]=='continuous':
        # Find the mean of the random variable
        EX=Mean(X_dummy)
        # Find E(X^2)
        # Create list of (x**2)*f(x)
        varfunc=[]
        for i in range(len(X_dummy.func)):
            varfunc.append((x**2)*X_dummy.func[i])
        # Integrate to find E(X^2)
        exxval=0
        for i in range(len(X_dummy.func)):
            val=integrate(varfunc[i],(x,X_dummy.support[i],X_dummy.support[i+1]))
            exxval+=val
        # Find Var(X)=E(X^2)-E(X)^2
        var=exxval-(EX**2)
        return simplify(var)

    # If the random variable is a discrete function, find and return the variance
    if RVar.ftype[0]=='Discrete':
        # Find the mean of the random variable
        EX=Mean(X_dummy)
        # Find E(X^2)
        # Create list of (x**2)*f(x)
        varfunc=[]
        for i in range(len(X_dummy.func)):
            varfunc.append((x**2)*X_dummy.func[i])
        # Sum to find E(X^2)
        exxval=0
        for i in range(len(X_dummy.func)):
            val=summation(varfunc[i],(x,X_dummy.support[i],X_dummy.support[i+1]))
            exxval+=val
        # Find Var(X)=E(X^2)-E(X)^2
        var=exxval-(EX**2)
        return simplify(var)

    # If the random variable is discrete, find and return the variance
    if RVar.ftype[0]=='discrete':
        return VarDiscrete(RVar)
        #
        # Legacy variance code ... update uses faster numpy implementation
        #
        # Find the mean of the random variable
        #EX=Mean(X_dummy)
        # Find E(X^2)
        # Create a list of (x**2)*f(x)
        #exxlist=[]
        #for i in range(len(X_dummy.func)):
        #    exxlist.append(X_dummy.func[i]*(X_dummy.support[i])**2)
        # Sum to find E(X^2)
        #exxval=0
        #for i in range(len(exxlist)):
        #    exxval+=exxlist[i]
        # Find Var(X)=E(X^2)-E(X)^2
        #var=exxval-(EX**2)
        #return simplify(var)

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

"""
Procedures on Two Random Variables

Procedures:
    1. Convolution(RVar1,RVar2)
    2. Maximum(RVar1,RVar2)
    3. Minimum(RVar1,RVar2)
    4. Mixture(MixParameters,MixRVs)
    5. Product(RVar1,RVar2)
"""
                    
def Convolution(RVar1,RVar2):
    """
    Procedure Name: Convolution
    Purpose: Compute the convolution of two independent
                random variables
    Arguments:  1. RVar1: A random variable
                2. RVar2: A random variable
    Output:     1. The convolution of RVar1 and RVar2        
    """
    # If the two random variables are not both continuous or
    #   both discrete, return an error
    if RVar1.ftype[0]!=RVar2.ftype[0]:
        raise RVError('Both random variables must have the same type')

    # Convert both random variables to their PDF form
    X1_dummy=PDF(RVar1)
    X2_dummy=PDF(RVar2)

    # If the distributions are continuous, find and return the convolution
    #   of the two random variables
    if RVar1.ftype[0]=='continuous':
        # If the two distributions are both lifetime distributions, treat
        #   as a special case
        if RVar1.support==[0,oo] and RVar2.support==[0,oo]:
            func1=X1_dummy.func[0]
            func2=X2_dummy.func[0].subs(x,z-x)
            conv=integrate(func1*func2,(x,0,z))
            return RV([conv.subs(z,x)],[0,oo],['continuous','pdf'])
        # Otherwise, compute the convolution using the product method
        else:
            gln=[[ln(x)],[0,oo]]
            ge=[[exp(x),exp(x)],[-oo,0,oo]]
            temp1=Transform(X1_dummy,ge)
            temp2=Transform(X2_dummy,ge)
            temp3=Product(temp1,temp2)
            fz=Transform(temp3,gln)
            convfunc=[]
            for i in range(len(fz.func)):
                convfunc.append(simplify(fz.func[i]))
            return RV(convfunc,fz.support,['continuous','pdf'])
            

    # If the distributions are discrete, find and return the convolution
    #   of the two random variables.
    if RVar1.ftype[0]=='discrete':
        # Convert each random variable to its pdf form
        X1_dummy=PDF(RVar1)
        X2_dummy=PDF(RVar2)
        # Create function and support lists for the convolution of the
        #   two random variables
        convlist=[]
        funclist=[]
        for i in range(len(X1_dummy.support)):
            for j in range(len(X2_dummy.support)):
                convlist.append(X1_dummy.support[i]+X2_dummy.support[j])
                funclist.append(X1_dummy.func[i]*X2_dummy.func[j])
        # Sort the function and support lists for the convolution
        sortlist=zip(convlist,funclist)
        sortlist.sort()
        convlist2=[]
        funclist2=[]
        for i in range(len(sortlist)):
            convlist2.append(sortlist[i][0])
            funclist2.append(sortlist[i][1])
        # Remove redundant elements in the support list
        convlist3=[]
        funclist3=[]
        for i in range(len(convlist2)):
            if convlist2[i] not in convlist3:
                convlist3.append(convlist2[i])
                funclist3.append(funclist2[i])
            else:
                funclist3[convlist3.index(convlist2[i])]+=funclist2[i]
        # Create and return the new random variable
        return RV(funclist3,convlist3,['discrete','pdf'])

def Maximum(RVar1,RVar2):
    """
    Procedure Name: Maximum
    Purpose: Compute cdf of the maximum of RVar1 and RVar2
    Arguments:  1. RVar1: A random variable
                2. RVar2: A random variable
    Output:     1. The cdf of the maximum distribution
    """

    # If the two random variables are not of the same type
    #   raise an error
    if RVar1.ftype[0]!=RVar2.ftype[0]:
        raise RVError('The RVs must both be discrete or continuous')

    # If the distributions are continuous, find and return the max
    if RVar1.ftype[0]=='continuous':
        # Special case for lifetime distributions
        if RVar1.support==[0,oo] and RVar2.support==[0,oo]:
            cdf_dummy1=CDF(RVar1)
            cdf_dummy2=CDF(RVar2)
            cdf1=cdf_dummy1.func[0]
            cdf2=cdf_dummy2.func[0]
            maxfunc=cdf1*cdf2
            return RV(simplify(maxfunc),[0,oo],['continuous','cdf'])
        # Otherwise, compute the min using the full algorithm
        Fx=CDF(RVar1)
        Fy=CDF(RVar2)
        # Create a support list for the 
        max_supp=[]
        for i in range(len(Fx.support)):
            if Fx.support[i] not in max_supp:
                max_supp.append(Fx.support[i])
        for i in range(len(Fy.support)):
            if Fy.support[i] not in max_supp:
                max_supp.append(Fy.support[i])
        max_supp.sort()
        # Remove any elements that are above the lower support max
        lowval=max(min(Fx.support),min(Fy.support))
        max_supp2=[]
        for i in range(len(max_supp)):
            if max_supp[i]>=lowval:
                max_supp2.append(max_supp[i])
        # Compute the maximum function for each segment
        xindx=0
        yindx=0
        max_func=[]
        for i in range(len(max_supp2)-1):
            if max_supp2[i]>Fx.support[0]:
                currFx=0
            elif max_supp2[i]==Fx.support[xindx]:
                currFx=Fx.func[xindx]
                xindx+=1
            if max_supp2[i]>Fy.support[yindx]:
                currFy=0
            elif max_supp2[i]==Fy.support[yindx]:
                currFy=Fy.func[yindx]
                yindx+=1
            Fmax=-(1-currFx)*(1-currFy)
            max_func.append(simplify(Fmax))
        # Return the random variable
        return RV(max_func,max_supp2,['continuous','cdf'])
    
    # If the distributions are discrete, find and return
    #   the maximum of the two rv's
    if RVar1.ftype[0]=='discrete':
        # Convert X and Y to their PDF representations
        fx=PDF(RVar1)
        fy=PDF(RVar2)
        # Make a list of possible combinations of X and Y
        combo_list=[]
        prob_list=[]
        for i in range(len(fx.support)):
            for j in range(len(fy.support)):
                combo_list.append([fx.support[i],fy.support[j]])
                prob_list.append(fx.func[i]*fy.func[j])

        # Old code for computing probability for each pair, had
        # floating point issues, PDF wouldn't recognize a number
        # as being in the support
        #prob_list=[]
        #for i in range(len(combo_list)):
        #    val=PDF(fx,combo_list[i][0])*PDF(fy,combo_list[j][1])
        #    prob_list.append(val)
        
        # Find the min value for each combo
        max_list=[]
        for i in range(len(combo_list)):
            max_list.append(max(combo_list[i][0],combo_list[i][1]))
        # Compute the probability for each possible min
        max_supp=[]
        max_func=[]
        for i in range(len(max_list)):
            if max_list[i] not in max_supp:
                max_supp.append(max_list[i])
                max_func.append(prob_list[i])
            else:
                indx=max_supp.index(max_list[i])
                max_func[indx]+=prob_list[i]
        # Sort the elements of the rv
        zip_list=zip(max_supp,max_func)
        zip_list.sort()
        max_supp=[]
        max_func=[]
        for i in range(len(zip_list)):
            max_supp.append(zip_list[i][0])
            max_func.append(zip_list[i][1])
        # Return the minimum random variable
        return RV(max_func,max_supp,['discrete','pdf'])

def Minimum(RVar1,RVar2):
    """
    Procedure Name: Minimum
    Purpose: Compute the distribution of the minimum of RVar1 and RVar2
    Arguments:  1. RVar1: A random variable
                2. RVar2: A random variable
    Output:     1. The minimum of the two random variables
    """

    # If the two random variables are not of the same type
    #   raise an error
    if RVar1.ftype[0]!=RVar2.ftype[0]:
        raise RVError('The RVs must both be discrete or continuous')

    # If the distributions are continuous, find and return the min
    if RVar1.ftype[0]=='continuous':
        # Special case for lifetime distributions
        if RVar1.support==[0,oo] and RVar2.support==[0,oo]:
            sf_dummy1=SF(RVar1)
            sf_dummy2=SF(RVar2)
            sf1=sf_dummy1.func[0]
            sf2=sf_dummy2.func[0]
            minfunc=1-(sf1*sf2)
            return RV(simplify(minfunc),[0,oo],['continuous','cdf'])
        # Otherwise, compute the min using the full algorithm
        Fx=CDF(RVar1)
        Fy=CDF(RVar2)
        # Create a support list for the 
        min_supp=[]
        for i in range(len(Fx.support)):
            if Fx.support[i] not in min_supp:
                min_supp.append(Fx.support[i])
        for i in range(len(Fy.support)):
            if Fy.support[i] not in min_supp:
                min_supp.append(Fy.support[i])
        min_supp.sort()
        # Remove any elements that are above the lower support max
        highval=min(max(Fx.support),max(Fy.support))
        min_supp2=[]
        for i in range(len(min_supp)):
            if min_supp[i]<=highval:
                min_supp2.append(min_supp[i])
        # Compute the minimum function for each segment
        xindx=0
        yindx=0
        min_func=[]
        for i in range(len(min_supp2)-1):
            if min_supp2[i]<Fx.support[0]:
                currFx=0
            elif min_supp2[i]==Fx.support[xindx]:
                currFx=Fx.func[xindx]
                xindx+=1
            if min_supp2[i]<Fy.support[yindx]:
                currFy=0
            elif min_supp2[i]==Fy.support[yindx]:
                currFy=Fy.func[yindx]
                yindx+=1
            Fmin=1-(1-currFx)*(1-currFy)
            min_func.append(simplify(Fmin))
        # Return the random variable
        return RV(min_func,min_supp2,['continuous','cdf'])

    # If the distributions are discrete, find and return
    #   the minimum of the two rv's
    if RVar1.ftype[0]=='discrete':
        # Convert X and Y to their PDF representations
        fx=PDF(RVar1)
        fy=PDF(RVar2)
        # Make a list of possible combinations of X and Y
        combo_list=[]
        prob_list=[]
        for i in range(len(fx.support)):
            for j in range(len(fy.support)):
                combo_list.append([fx.support[i],fy.support[j]])
                prob_list.append(fx.func[i]*fy.func[j])
                
        # Old code for computing probability for each pair, had
        # floating point issues, PDF wouldn't recognize a number
        # as being in the support
        #prob_list=[]
        #for i in range(len(combo_list)):
        #    val=PDF(fx,combo_list[i][0])*PDF(fy,combo_list[j][1])
        #    prob_list.append(val)
        # Find the min value for each combo
        
        min_list=[]
        for i in range(len(combo_list)):
            min_list.append(min(combo_list[i][0],combo_list[i][1]))
        # Compute the probability for each possible min
        min_supp=[]
        min_func=[]
        for i in range(len(min_list)):
            if min_list[i] not in min_supp:
                min_supp.append(min_list[i])
                min_func.append(prob_list[i])
            else:
                indx=min_supp.index(min_list[i])
                min_func[indx]+=prob_list[i]
        # Sort the elements of the rv
        zip_list=zip(min_supp,min_func)
        zip_list.sort()
        min_supp=[]
        min_func=[]
        for i in range(len(zip_list)):
            min_supp.append(zip_list[i][0])
            min_func.append(zip_list[i][1])
        # Return the minimum random variable
        return RV(min_func,min_supp,['discrete','pdf'])

def Mixture(MixParameters,MixRVs):
    """
    Procedure Name: Mixture
    Purpose: Mixes random variables X1,X2,...,Xn
    Arguments:   1. MixParameters: A mix of probability weights
                 2. MixRVs: RV's X1,X2,...,Xn
    Output:      1. The mixture RV
    """

    # Check to make sure that the arguments are lists
    if type(MixParameters)!=list or type(MixRVs)!=list:
        raise RVError('Both arguments must be in list format')
    # Check to make sure the lists are of equal length
    if len(MixParameters)!=len(MixRVs):
        raise RVError('Mix parameter and RV lists must be the same length')
    # Check to make sure that the mix parameters are numeric
    # and sum to 1
    total=0
    for i in range(len(MixParameters)):
        if type(MixParameters[i])==Symbol:
            raise RVError('ApplPy does not support symbolic mixtures')
        total+=MixParameters[i]
    if total<.9999 or total>1.0001:
        raise RVError('Mix parameters must sum to one')
    # Check to ensure that the mix rv's are all of the same type
    #   (discrete or continuous)
    for i in range(len(MixRVs)):
        if MixRVs[0].ftype[0]!=MixRVs[i].ftype[0]:
            raise RVError('Mix RVs must be all continuous or discrete')
    # Convert the Mix RVs to their PDF form
    Mixfx=[]
    for i in range(len(MixRVs)):
        Mixfx.append(PDF(MixRVs[i]))

    # If the distributions are continuous, find and return the
    #   mixture pdf
    if Mixfx[0].ftype[0]=='continuous':
        # Compute the support of the mixture as the union of the supports
        #   of the mix rvs
        MixSupp=[]
        for i in range(len(Mixfx)):
            for j in range(len(Mixfx[i].support)):
                if Mixfx[i].support[j] not in MixSupp:
                    MixSupp.append(Mixfx[i].support[j])
        MixSupp.sort()
        # Compute and return the mixed PDF
        fxnew=[]
        for i in range(len(MixSupp)-1):
            newMixfx=0
            for j in range(len(MixParameters)):
                m=len(Mixfx[j].support)-1
                for k in range(m):
                    if Mixfx[j].support[k]<=MixSupp[i]:
                        if MixSupp[i+1]<=Mixfx[j].support[k+1]:
                            buildfx=Mixfx[j].func[k]*MixParameters[j]
                            newMixfx+=buildfx
            simplify(newMixfx)
            fxnew.append(newMixfx)
        # Return the mixture rv
        return RV(fxnew,MixSupp,['continuous','pdf'])

    # If the distributions are discrete, find and return the
    #   mixture pdf
    if Mixfx[0].ftype[0]=='discrete':
        # Compute the mixture rv by summing over the weights
        MixSupp=[]
        fxnew=[]
        for i in range(len(Mixfx)):
            for j in range(len(Mixfx[i].support)):
                if Mixfx[i].support[j] not in MixSupp:
                    MixSupp.append(Mixfx[i].support[j])
                    fxnew.append(Mixfx[i].func[j]*MixParameters[i])
                else:
                    indx=MixSupp.index(Mixfx[i].support[j])
                    val=Mixfx[i].func[j]*MixParameters[i]
                    fxnew[indx]+=val
        # Sort the values
        zip_list=zip(MixSupp,fxnew)
        zip_list.sort()
        fxnew=[]
        MixSupp=[]
        for i in range(len(zip_list)):
            fxnew.append(zip_list[i][1])
            MixSupp.append(zip_list[i][0])
        return RV(fxnew,MixSupp,['discrete','pdf'])
        


def Product(RVar1,RVar2):
    """
    Procedure Name: Product
    Purpose: Compute the product of two independent
                random variables
    Arguments:  1. RVar1: A random variable
                2. RVar2: A random variable
    Output:     1. The product of RVar1 and RVar2        
    """
    # If the random variable is continuous, find and return the
    #   product of the two random variables
    if RVar1.ftype[0]=='continuous':
        v=Symbol('v',positive=True)
        # Place zero in the support of X if it is not there already
        X1=PDF(RVar1)
        xfunc=[]
        xsupp=[]
        for i in range(len(X1.func)):
            xfunc.append(X1.func[i])
            xsupp.append(X1.support[i])
            if X1.support[i]<0:
                if X1.support[i+1]>0:
                    xfunc.append(X1.func[i])
                    xsupp.append(0)
        xsupp.append(X1.support[len(X1.support)-1])
        X_dummy=RV(xfunc,xsupp,['continuous','pdf'])
        # Place zero in the support of Y if it is not already there
        Y1=PDF(RVar2)
        yfunc=[]
        ysupp=[]
        for i in range(len(Y1.func)):
            yfunc.append(Y1.func[i])
            ysupp.append(Y1.support[i])
            if Y1.support[i]<0:
                if Y1.support[i+1]>0:
                    yfunc.append(Y1.func[i])
                    ysupp.append(0)
        ysupp.append(Y1.support[len(Y1.support)-1])
        Y_dummy=RV(yfunc,ysupp,['continuous','pdf'])
        # Initialize the support list for the product V=X*Y
        vsupp=[]
        for i in range(len(X_dummy.support)):
            for j in range(len(Y_dummy.support)):
                val=X_dummy.support[i]*Y_dummy.support[j]
                if val==nan:
                    val=0
                if val not in vsupp:
                    vsupp.append(val)
        vsupp.sort()
        # Initialize the pdf segments of v
        vfunc=[]
        for i in range(len(vsupp)-1):
            vfunc.append(0)
        # Loop through each piecewise segment of X
        for i in range(len(X_dummy.func)):
            # Loop through each piecewise segment of Y
            for j in range(len(Y_dummy.func)):
                # Define the corner of the rectangular region
                a=X_dummy.support[i]
                b=X_dummy.support[i+1]
                c=Y_dummy.support[j]
                d=Y_dummy.support[j+1]
                # If the region is in the first quadrant, compute the
                #   required integrals sequentially
                if a>=0 and c>=0:
                    if type(Y_dummy.func[j]) not in [float,int]:
                        gj=Y_dummy.func[j].subs(x,v/x)
                    else:
                        gj=Y_dummy.func[j]
                    fi=X_dummy.func[i]
                    pv=simplify(integrate(fi*gj*(1/x),(x,a,b)))
                    if d<oo:
                        qv=simplify(integrate(fi*gj*(1/x),(x,v/d,b)))
                    if c>0:
                        rv=simplify(integrate(fi*gj*(1/x),(x,a,v/c)))
                    if c>0 and d<oo and a*d<b*c:
                        sv=simplify(integrate(fi*gj*(1/x),(x,v/d,v/c)))
                    # 1st Qd, Scenario 1
                    if c==0 and d==oo:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=0:
                                vfunc[k]+=pv
                    # 1st Qd, Scenario 2
                    if c==0 and d<oo:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=0 and vsupp[k+1]<=a*d:
                                vfunc[k]+=pv
                            if vsupp[k]>=a*d and vsupp[k+1]<=b*d:
                                vfunc[k]+=qv
                    # 1st Qd, Scenario 3
                    if c>0 and d==oo:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=b*c:
                                vfunc[k]+=pv
                            if vsupp[k]>=a*c and vsupp[k+1]<=b*c:
                                vfunc[k]+=rv
                    # 1st Qd, Scenario 4
                    if c>0 and d<oo:
                        # Case 1
                        if a*d<b*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=a*c and vsupp[k+1]<=a*d:
                                    vfunc[k]+=rv
                                if vsupp[k]>=a*d and vsupp[k+1]<=b*c:
                                    vfunc[k]+=sv
                                if vsupp[k]>=b*c and vsupp[k+1]<=b*d:
                                    vfunc[k]+=qv
                        # Case 2
                        if a*d==b*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=a*c and vsupp[k+1]<=a*d:
                                    vfunc[k]+=rv
                                if vsupp[k]>=b*c and vsupp[k+1]<=b*d:
                                    vfunc[k]+=qv
                        # Case 3
                        if a*d>b*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=a*c and vsupp[k+1]<=b*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=b*c and vsupp[k+1]<=a*d:
                                    vfunc[k]+=pv
                                if vsupp[k]>=a*d and vsupp[k+1]<=b*d:
                                    vfunc[k]+=qv
                # If the region is in the second quadrant, compute
                #   the required integrals sequentially
                if a<0 and c<0:
                    if type(Y_dummy.func[j]) not in [float,int]:
                        gj=Y_dummy.func[j].subs(x,v/x)
                    else:
                        gj=Y_dummy.func[j]
                    fi=X_dummy.func[i]
                    pv=-simplify(integrate(fi*gj*(1/x),(x,a,b)))
                    if d<0:
                        qv=-simplify(integrate(fi*gj*(1/x),(x,(v/d),b)))
                    if c>-oo:
                        rv=-simplify(integrate(fi*gj*(1/x),(x,a,(v/c))))
                    if c>-oo and d<0:
                        sv=-simplify(integrate(fi*gj*(1/x),(x,(v/d),(v/c))))
                    # 2nd Qd, Scenario 1
                    if c==-oo and d==0:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=0:
                                vfunc[k]+=pv
                    # 2nd Qd, Scenario 2
                    if c==-oo and d<0:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=a*d and vsupp[k+1]<=oo:
                                vfunc[k]+=pv
                            if vsupp[k]>=b*d and vsupp[k+1]<=a*d:
                                vfunc[k]+=qv
                    # 2nd Qd, Scenario 3
                    if c>-oo and d==0:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=0 and vsupp[k+1]<=b*c:
                                vfunc[k]+=pv
                            if vsupp[k]>=b*c and vsupp[k+1]<=a*c:
                                vfunc[k]+=rv
                    # 2nd Qd, Scenario 4
                    if c>-oo and d<0:
                        # Case 1
                        if a*d>b*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=a*d and vsupp[k+1]<=a*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=b*c and vsupp[k+1]<=a*d:
                                    vfunc[k]+=sv
                                if vsupp[k]>=b*d and vsupp[k+1]<=b*c:
                                    vfunc[k]+=qv
                        # Case 2
                        if a*d==b*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=a*d and vsupp[k+1]<=a*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=b*d and vsupp[k+1]<=b*c:
                                    vfunc[k]+=qv
                        # Case 3
                        if a*d<b*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=b*c and vsupp[k+1]<=a*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=a*d and vsupp[k+1]<=b*c:
                                    vfunc[k]+=pv
                                if vsupp[k]>=b*d and vsupp[k+1]<=a*d:
                                    vfunc[k]+=qv
                # If the region is in the third quadrant, compute
                #   the required integrals sequentially
                if a<0 and c>=0:
                    if type(Y_dummy.func[j]) not in [float,int]:
                        gj=Y_dummy.func[j].subs(x,v/x)
                    else:
                        gj=Y_dummy.func[j]
                    fi=X_dummy.func[i]
                    pv=-simplify(integrate(fi*gj*(1/x),(x,a,b)))
                    if d<oo:
                        qv=-simplify(integrate(fi*gj*(1/x),(x,a,(v/d))))
                    if c>0:
                        rv=-simplify(integrate(fi*gj*(1/x),(x,(v/c),b)))
                    if c>0 and d<oo:
                        sv=-simplify(integrate(fi*gj*(1/x),(x,(v/c),(v/d))))
                    # 3rd Qd, Scenario 1
                    if c==0 and d==oo:
                        for k in range(len(vfunc)):
                            if vsupp[k+1]<=0:
                                vfunc[k]+=pv
                    # 3rd Qd, Scenario 2
                    if c==0 and d<oo:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=b*d and vsupp[k+1]<=0:
                                vfunc[k]+=pv
                            if vsupp[k]>=a*d and vsupp[k+1]<=b*d:
                                vfunc[k]+=qv
                    # 3rd Qd, Scenario 3
                    if c>0 and d<oo:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=-oo and vsupp[k+1]<=a*c:
                                vfunc[k]+=pv
                            if vsupp[k]>=a*c and vsupp[k+1]<=b*c:
                                vfunc[k]+=rv
                    # 3rd Qd, Scenario 4
                    if c>0 and d<oo:
                        # Case 1
                        if b*d>a*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=b*d and vsupp[k+1]<=b*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=a*c and vsupp[k+1]<=b*d:
                                    vfunc[k]+=sv
                                if vsupp[k]>=a*d and vsupp[k+1]<=a*c:
                                    vfunc[k]+=qv
                        # Case 2
                        if a*c==b*d:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=a*d and vsupp[k+1]<=a*c:
                                    vfunc[k]+=qv
                                if vsupp[k]>=b*d and vsupp[k+1]<=b*c:
                                    vfunc[k]+=rv
                        # Case 3
                        if a*c>b*d:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=a*c and vsupp[k+1]<=b*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=b*d and vsupp[k+1]<=a*c:
                                    vfunc[k]+=pv
                                if vsupp[k]>=a*d and vsupp[k+1]<=b*d:
                                    vfunc[k]+=qv
                # If the region is in the fourth quadrant, compute
                #   the required integrals sequentially
                if a>=0 and c<0:
                    if type(Y_dummy.func[j]) not in [float,int]:
                        gj=Y_dummy.func[j].subs(x,v/x)
                    else:
                        gj=Y_dummy.func[j]
                    fi=X_dummy.func[i]
                    pv=simplify(integrate(fi*gj*(1/x),(x,a,b)))
                    if d<0:
                        qv=simplify(integrate(fi*gj*(1/x),(x,a,(v/d))))
                    if c>-oo:
                        rv=simplify(integrate(fi*gj*(1/x),(x,(v/c),b)))
                    if c>-oo and d<0:
                        sv=simplify(integrate(fi*gj*(1/x),(x,(v/c),(v/d))))
                    # 4th Qd, Scenario 1
                    if c==oo and d==0:
                        for k in range(len(vfunc)):
                            if vsupp[k+1]<=0:
                                vfunc[k]+=pv
                    # 4th Qd, Scenario 2
                    if c==oo and d<0:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=-oo and vsupp[k+1]<=b*d:
                                vfunc[k]+=pv
                            if vsupp[k]>=b*d and vsupp[k+1]<=a*d:
                                vfunc[k]+=qv
                    # 4th Qd, Scenario 3
                    if c>-oo and d==0:
                        for k in range(len(vfunc)):
                            if vsupp[k]>=a*c and vsupp[k+1]<=0:
                                vfunc[k]+=pv
                            if vsupp[k]>=b*c and vsupp[k+1]<=a*c:
                                vfunc[k]+=rv
                    # 4th Qd, Scenario 4
                    if c>-oo and d<0:
                        # Case 1
                        if a*c>b*d:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=b*c and vsupp[k+1]<=b*d:
                                    vfunc[k]+=rv
                                if vsupp[k]>=b*d and vsupp[k+1]<=a*c:
                                    vfunc[k]+=sv
                                if vsupp[k]>=a*c and vsupp[k+1]<=a*d:
                                    vfunc[k]+=qv
                        # Case 2
                        if a*d==b*c:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=b*c and vsupp[k+1]<=a*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=a*c and vsupp[k+1]<=a*d:
                                    vfunc[k]+=qv
                        # Case 3
                        if a*c<b*d:
                            for k in range(len(vfunc)):
                                if vsupp[k]>=b*c and vsupp[k+1]<=a*c:
                                    vfunc[k]+=rv
                                if vsupp[k]>=a*c and vsupp[k+1]<=b*d:
                                    vfunc[k]+=pv
                                if vsupp[k]>=b*d and vsupp[k+1]<=a*d:
                                    vfunc[k]+=qv                   
        vfunc_final=[]
        for i in range(len(vfunc)):
            if type(vfunc[i]) not in [int,float]:
                vfunc_final.append(simplify(vfunc[i]).subs(v,x))
            else:
                vfunc_final.append(vfunc[i])
        return RV(vfunc_final,vsupp,['continuous','pdf'])
    # If the distributions are discrete, find and return the product
    #   of the two random variables.
    if RVar1.ftype[0]=='discrete':
        # Convert each random variable to its pdf form
        X1_dummy=PDF(RVar1)
        X2_dummy=PDF(RVar2)
        # Create function and support lists for the product of the
        #   two random variables
        prodlist=[]
        funclist=[]
        for i in range(len(X1_dummy.support)):
            for j in range(len(X2_dummy.support)):
                prodlist.append(X1_dummy.support[i]*X2_dummy.support[j])
                funclist.append(X1_dummy.func[i]*X2_dummy.func[j])
        # Sort the function and support lists for the convolution
        sortlist=zip(prodlist,funclist)
        sortlist.sort()
        prodlist2=[]
        funclist2=[]
        for i in range(len(sortlist)):
            prodlist2.append(sortlist[i][0])
            funclist2.append(sortlist[i][1])
        # Remove redundant elements in the support list
        prodlist3=[]
        funclist3=[]
        for i in range(len(prodlist2)):
            if prodlist2[i] not in prodlist3:
                prodlist3.append(prodlist2[i])
                funclist3.append(funclist2[i])
            else:
                funclist3[prodlist3.index(prodlist2[i])]+=funclist2[i]
        # Create and return the new random variable
        return RV(funclist3,prodlist3,['discrete','pdf'])

def ProductDiscrete(RVar1,RVar2):
    """
    Procedure Name: ProductDiscrete
    Purpose: Compute the product of two independent
                discrete random variables
    Arguments:  1. RVar1: A random variable
                2. RVar2: A random variable
    Output:     1. The product of RVar1 and RVar2        
    """
    # Ensure that both random variables are discrete
    if RVar1.ftype[0]!='discrete' or RVar2.ftype[0]!='discrete':
        raise RVError('both random variables must be discrete')
    # Convert both random variables to pdf form
    X_dummy1=PDF(RVar1)
    X_dummy2=PDF(RVar2)
    # Convert the support and the value of each random variable
    #   into a numpy matrix
    support1=np.matrix(X_dummy1.support)
    support2=np.matrix(X_dummy2.support)
    pdf1=np.matrix(X_dummy1.func)
    pdf2=np.matrix(X_dummy2.func)
    # Find all possible values of support1*support2 and val1*val2
    #   by computing (X1)'*X2, flatten into a row vector
    prodsupport=support1.T*support2
    prodsupport=prodsupport.flatten()
    prodpdf=pdf1.T*pdf2
    prodpdf=prodpdf.flatten()
    #
    # Stack the support vector and the value vector into a matrix
    #prodmatrix=np.vstack([prodsupport,prodpdf]).T
    #
    #
    # Convert the resulting vectors into lists
    supportlist=prodsupport.tolist()[0]
    pdflist=prodpdf.tolist()[0]
    # Sort the function and support lists for the product
    sortlist=zip(supportlist,pdflist)
    sortlist.sort()
    prodlist2=[]
    funclist2=[]
    for i in range(len(sortlist)):
        prodlist2.append(sortlist[i][0])
        funclist2.append(sortlist[i][1])
    # Remove redundant elements in the support list
    prodlist3=[]
    funclist3=[]
    for i in range(len(prodlist2)):
        if prodlist2[i] not in prodlist3:
            prodlist3.append(prodlist2[i])
            funclist3.append(funclist2[i])
        else:
            funclist3[prodlist3.index(prodlist2[i])]+=funclist2[i]
    # Create and return the new random variable
    return RV(funclist3,prodlist3,['discrete','pdf'])
  
"""
Utilities

Procedures:
    1. Histogram(Sample,bins)
    2. PlotDist(RVar,suplist)
    3. PlotDisplay(plot_list,suplist)
    4. PlotEmpCDF(data)
    5. PPPlot(RVar,Sample)
    6. QQPlot(RVar,Sample)
"""

def Histogram(Sample,Bins=None):
    """
    Procedure: Histogram
    Purpose: Construct a histogram from a sample of data
    Arguments: 1. Sample: The data sample from which to construct
                    the histogram
               2. bins: The number of bins in the histogram
    Output:    1. A histogram plot   
    """
    # Check to ensure that the sample is given as a list
    if type(Sample)!=list:
        raise RVError('The data sample must be entered as a list')

    Sample.sort()
    if Bins==None:
        Bins=1
        for i in range(1,len(Sample)):
            if Sample[i]!=Sample[i-1]:
                Bins+=1

    plt.ion()
    plt.hist(Sample,bins=Bins,normed=True)
    plt.ylabel('Relative Frequency')
    plt.xlabel('Observation Value')
    plt.title('Histogram')
    plt.grid(True)

def PlotDist(RVar,suplist=None,opt=None,color='red',
             display=True):
    """
    Procedure: Plot Dist
    Purpose: Plot a random variable
    Arguments:  1. RVar: A random variable
                2. suplist: A list of supports for the plot
    Output:     1. A plot of the random variable
    """
    # Create the labels for the plot
    if RVar.ftype[1]=='cdf':
        lab1='F(x)'
        lab2='Cumulative Distribution Function'
    elif RVar.ftype[1]=='chf':
        lab1='H(x)'
        lab2='Cumulative Hazard Function'
    elif RVar.ftype[1]=='hf':
        lab1='h(x)'
        lab2='Hazard Function'
    elif RVar.ftype[1]=='idf':
        lab1='F-1(s)'
        lab2='Inverse Density Function'
    elif RVar.ftype[1]=='pdf':
        lab1='f(x)'
        lab2='Probability Density Function'
    elif RVar.ftype[1]=='sf':
        lab1='S(X)'
        lab2='Survivor Function'

    if opt=='EMPCDF':
        lab2='Empirical CDF'

    # If the distribution is continuous, plot the function
    if RVar.ftype[0]=='continuous':
        # Return an error if the plot supports are not
        #   within the support of the random variable
        if suplist!=None:
            if suplist[0]>suplist[1]:
                raise RVError('Support list must be in ascending order')
            if suplist[0]<RVar.support[0]:
                raise RVError('Plot supports must fall within RV support')
            if suplist[1]>RVar.support[1]:
                raise RVError('Plot support must fall within RV support')
        # Cut out parts of the distribution that don't fall
        #   within the limits of the plot
        if suplist==None:
            # Since plotting is numeric, the lower support cannot be -oo
            if RVar.support[0]==-oo:
                support1=float(RVar.variate(s=.01)[0])
            else:
                support1=float(RVar.support[0])
            # Since plotting is numeric, the upper support cannot be oo
            if RVar.support[len(RVar.support)-1]==oo:
                support2=float(RVar.variate(s=.99)[0])
            else:
                support2=float(RVar.support[len(RVar.support)-1])
            suplist=[support1,support2]                
        for i in range(len(RVar.func)):
            if suplist[0]>=RVar.support[i]:
                if suplist[0]<=RVar.support[i+1]:
                    lwindx=i
            if suplist[1]>=RVar.support[i]:
                if suplist[1]<=RVar.support[i+1]:
                    upindx=i
        # Create a list of functions for the plot
        plotfunc=[]
        for i in range(len(RVar.func)):
            if i>=lwindx and i<=upindx:
                plotfunc.append(RVar.func[i])
        # Create a list of supports for the plot
        plotsupp=[suplist[0]]
        upindx+=1
        for i in range(len(RVar.support)):
            if i>lwindx and i<upindx:
                plotsupp.append(RVar.support[i])
        plotsupp.append(suplist[1])
        plt.ioff()
        initial_plot=plot(plotfunc[0],(x,plotsupp[0],plotsupp[1]),
                          title=lab2,show=False,line_color=color)
        for i in range(1,len(plotfunc)):
            plot_extension=plot(plotfunc[i],
                                (x,plotsupp[i],plotsupp[i+1]),
                                show=False,line_color=color)
            initial_plot.append(plot_extension[0])
        if display==True:
            plt.ion()
            initial_plot.show()
        else:
            return initial_plot

        # Old PlotDist code before sympy created the
        #   plotting front-end
        #print plotsupp
        # Parse the functions for matplotlib
        #plot_func=[]
        #for i in range(len(plotfunc)):
        #    strfunc=str(plotfunc[i])
        #    plot_func.append(strfunc)
        #print plot_func
        # Display the plot
        #if opt!='display':
        #    plt.ion()
        #plt.mat_plot(plot_func,plotsupp,lab1,lab2,'continuous')
        
    if RVar.ftype[0]=='discrete' or RVar.ftype[0]=='Discrete':
        if RVar.ftype[0]=='Discrete':
            if RVar.support[-1]!=oo:
                RVar=Convert(RVar)
            else:
                p=1;i=RVar.support[0]
                while p>.00001:
                    p=PDF(RVar,i).evalf()
                    i+=1
                newsupport=RVar.support
                newsupport[-1]=i
                RVar=RV(RVar.func,newsupport,RVar.ftype)
                RVar=Convert(RVar)
        if display==True:
            pyplt.ion()
        #plt.mat_plot(RVar.func,RVar.support,lab1,lab2,'discrete')
        pyplt.plot(RVar.support,RVar.func,'ro')
        pyplt.xlabel('x')
        if lab1!=None:
            pyplt.ylabel(lab1)
        if lab2!=None:
            pyplt.title(lab2)

def PlotDisplay(plot_list):
    if len(plot_list)<2:
        raise RVError('PlotDisplay requires a list with multiple plots')
    plt.ion()
    totalplot=plot_list[0]
    for graph in plot_list[1:]:
        totalplot.append(graph[0])
    totalplot.show()        

def PlotEmpCDF(data):
    """
    Procedure Name: PlotEmpCDF
    Purpose: Plots an empirical CDF, given a data set
    Arguments:  1. data: A data sample
    Output:     1. An empirical cdf of the data 
    """

    # Create a bootstrap random variable from the data
    Xstar=BootstrapRV(data)
    PlotDist(CDF(Xstar),opt='EMPCDF')

def PPPlot(RVar,Sample):
    """
    Procedure Name: PPPlot
    Purpose: Plots the model probability versus the sample
                probability
    Arguments:  1. RVar: A random variable
                2. Sample: An experimental sample
    Output:     1. A PPPlot comparing the sample to a theoretical
                    model
    """
    # Return an error message if the sample is not given as
    #   a list
    if type(Sample)!=list:
        raise RVError('The data sample must be given as a list')

    # Create a list of quantiles
    n=len(Sample)
    Sample.sort()
    plist=[]
    for i in range(1,n+1):
        p=(i-(1/2))/n
        plist.append(p)

    # Create a list of CDF values for the sample and the
    # theoretical model
    FX=CDF(RVar)
    fxstar=BootstrapRV(Sample)
    FXstar=CDF(fxstar)

    FittedCDF=[]
    ObservedCDF=[]
    for i in range(len(plist)):
        FittedCDF.append(CDF(FX,Sample[i]))
        ObservedCDF.append(CDF(FXstar,Sample[i]))

    # Plot the results  
    plt.ion()
    plt.prob_plot(ObservedCDF,FittedCDF,'PP Plot')


def QQPlot(RVar,Sample):
    """
    Procedure: QQPlot
    Purpose: Plots the q_i quantile of a fitted distribution
                versus the q_i quantile of the sample dist
    Arguments:  1. RVar: A random variable
                2. Sample: Sample data
    Output:     1. QQ Plot
    """
    # Return an error message is the sample is not given as
    #   a list
    if type(Sample)!=list:
        raise RVError('The data sample must be given as a list')

    # Create a list of quantiles
    n=len(Sample)
    Sample.sort()
    qlist=[]
    for i in range(1,n+1):
        q=(i-(1/2))/n
        qlist.append(q)
    # Create 'fitted' list
    Fitted=[]
    for i in range(len(qlist)):
        Fitted.append(RVar.variate(s=qlist[i])[0])

    # Plot the results
    plt.ion()
    plt.prob_plot(Sample,Fitted,'QQ Plot')
