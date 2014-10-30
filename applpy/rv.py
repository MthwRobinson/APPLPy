from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, plot, Add, Mul, Integer, function,
                   binomial)
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

Defines the random variable class
Defines procedures for changing functional form

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
            # Generate the pieces of the piecewise function
            piece_list=[]
            for i in range(len(self.func)):
                f=self.func[i]
                sup='x>=%s'%(self.support[i])
                try:
                    tup=(f,eval(sup))
                except:
                    tup=(f,sup)
                piece_list.append(tup)
            piece_list.append((0,True))
            piece_input='Piecewise('+str(piece_list)+')'
            piece2=piece_input.replace(piece_input[10],'')
            n=len(piece2)-2
            piece3=piece2.replace(piece2[n],'')
            # Create symbols for use in the piecewise
            #   function display
            theta=Symbol('theta');kappa=Symbol('kappa');
            a=Symbol('a');b=Symbol('b');c=Symbol('c');
            p=Symbol('p');N=Symbol('N');alpha=Symbol('alpha')
            beta=Symbol('beta');mu=Symbol('mu');sigma=Symbol('sigma')
            try:
                p=eval(piece3)
                print '%s %s with support %s:'%(self.ftype[0],
                                                self.ftype[1],
                                                self.support)
                if opt=='repr':
                    return self.func
                elif opt=='piecewise':
                    return p
            except:
                print '%s %s with support %s:'%(self.ftype[0],
                                                self.ftype[1],
                                                self.support)
                return self.func
            
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
        

def BootstrapRV(varlist):
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
        if opt!='display':
            plt.ion()
        #plt.mat_plot(RVar.func,RVar.support,lab1,lab2,'discrete')
        pyplt.plot(RVar.support,RVar.func,'ro')
        pyplt.xlabel('x')
        if lab1!=None:
            pyplt.ylabel(lab1)
        if lab2!=None:
            pyplt.title(lab2)

def PlotDisplay(plot_list,suplist=None):
    plt.ion()
    # Create a plot of each random variable in the plot list
    for i in range(len(plot_list)):
        PlotDist(plot_list[i],suplist,'display')

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
