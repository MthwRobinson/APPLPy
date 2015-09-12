"""
Stocastic Processes Module

1. The Markov Chain Class
"""

from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, Add, Mul, Integer, function,
                   binomial, pprint,log,expand,zoo,latex,Piecewise,Rational,
                   Sum,S,Float)
from .rv import (RV, RVError, CDF, CHF, HF, IDF, IDF, PDF, SF,
                 BootstrapRV, Convert)
from sympy.plotting.plot import plot
from random import random
import numpy as np
import pandas as pd
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

class StochError(Exception):
    """
    Stoch Error Class
    Defines a custom error messages for exceptions relating
    to stochastic processes
    """
    def __init__(self,value):
        self.value=value
    def __str__(self):
        return repr(self.value)

class MarkovChain:
    """
    Markov Chain Class
    Defines the data structure for APPLPy Markov Chains
    Defines procedures relating to APPLPy Markov Chains
    """

    def __init__(self,P,init=None,states=None):
        """
        Procedure Name: __init__
        Purpose: Initializes and instance of the Markov Chain class
        Arguments:  1. P: the transition matrix of the markov chain
                    2. init: the initial distribution for the markov chain
                            if the initial distribution is entered as a row
                            vector, it will be transposed into a column vector
                            (for more convenient and flexible input)
        Output:     1. And instance of the Markov Chain class
        """
        # Check to ensure that the transition probability matrix is entered
        #   as an array or as a list. If it is not, raise an error
        if type(P) != np.ndarray:
            if type(P) != list:
                err_string = 'The transition probability matrix must '
                err_string += 'be entered as a list of lists or as a '
                err_string += 'numpy array'
                raise StochError(err_string)
            else:
                P = np.array(P)
        # Optionally set up the markov process to recognizes names for the
        #   states in the state space
        self.state_space = None
        if states != None:
            # If the number of states does not equal the dimension of the
            #   transition matrix, return an error
            if len(states) != P.shape[0]:
                err_string = 'The number of states in the state space '
                err_string += 'must be equal to the dimensions of the '
                err_string += 'transition probability matrix'
                raise StochError(err_string)
            # Convert the state labels to strings, set the state space
            #   for the markov chain
            state_space = [str(state_label) for state_label in states]
            self.state_space = state_space
        # Check to make sure that the transition probability matrix is a square
        #   matrix
        if P.shape[0] != P.shape[1]:
            err_string = 'The transition probability matrix must be a'
            err_string += ' square matrix'
            raise StochError(err_string)
        # Check to make sure each row in the transition probability matrix
        #   sums to 1
        num_error=.000001
        for i in range(P.shape[0]):
            if sum(P[i])>1+num_error or sum(P[i])<1-num_error:
                err_string = 'Each row in the transition probability matrix'
                err_string += ' sum to one. '
                row_id = 'Row %s does not sum to one.' % (str(i+1))
                err_string += row_id
                raise StochError(err_string)
        self.P=P
        self.P_print=self.matrix_convert(P)

        # If an initial distribution is specified, check to make sure that it
        #   is entered as an array or list
        if init != None:
            if type(init) != np.ndarray:
                if type(init) != list:
                    err_string = 'The initial distribution must '
                    err_string += 'be entered as a list or as a '
                    err_string += 'numpy array'
                    raise StochError(err_string)
                else:
                    init=np.array(init)
            # Check to make sure each the initial distribution sums to 1
            num_error=.000001
            if sum(init)>1+num_error or sum(init)<1-num_error:
                err_string = 'The initial distribution must sum to one'
                raise StochError(err_string)
            self.init=init
            self.init_print=self.vector_convert(init)
        else:
            self.init=None
            self.init_print=None
        # Initialize the state of the system to the initial distribution
        self.state=init
        self.state_print=self.init_print
        self.steps=0

    """
    Special Class Methods

    Procedures:
        1. display(self)
        2. matrix_convert(self,matrix)
        3. reset(self)
        4. step(self,n)
        5. trans_mat(self,n)
        6. vector_convert(self,vector)
    """
    def display(self):
        """
        Procedure Name: display
        Purpose: Displays the markov process in an interactive environment
        Arugments:  1. self: the markov process
        Output:     1. The transition probability matrix
                    2. The initial state of the system
                    3. The current state of the system
        """
        # Store display versions of the transition probability matrix and
        #   initial state for future use
        print 'The transition probability matrix:'
        print self.P_print
        print '----------------------------------------'
        print 'The initial system state:'
        print self.init_print
        print'-----------------------------------------'
        print 'The current system state after %s steps:'%(self.steps)
        print self.state_print      
        

    def matrix_convert(self,matrix):
        """
        Procedure Name: matrix_convert
        Purpose: Converts matrices to pandas data frame so that they can
                    be displayed with state space labels
        Arugments:  1. self: the markov process
                    2. matrix: the matrix to be converted for display
        Output:     1. The matrix in display format
        """
        display_mat = pd.DataFrame(matrix, index=self.state_space,
                                    columns = self.state_space)
        return display_mat

    def reset(self):
        """
        Procedure Name: resets
        Purpose: Resets the current state of the system to n=0
        Arugments:  1. self: the markov process
        Output:     1. The system reset to time n=0
        """
        self.state=self.init
        self.state_print=self.init_print
        return self.state_print
    
    def step(self,n):
        """
        Procedure Name: step
        Purpose: Moves the system forward n steps and saves the current
                    state of the system
        Arguments:  1. n: the number of steps the system takes forward
        Output:     1. The state of the system after n additional steps
        """
        # Check to make sure that the number of steps is an integer value
        if type(n) != int:
            err_string = 'The number of steps in a discrete time markov chain'
            err_string = ' must be an integer value'
            raise StochError(err_string)
        # Compute the probability transition matrix for n steps
        Pk = self.trans_mat(n)
        # Compute the state of the system after n additional steps
        new_state = np.dot(self.state,Pk)
        # Set the system state to the new state
        self.state=new_state
        self.state_print=self.vector_convert(new_state)
        self.steps+=n
        return self.state
            
    def trans_mat(self,n):
        """
        Procedure Name: trans_mat
        Purpose: Computes the state of the system after n steps
        Arguments:  1. n: the number of steps the system takes forward
        Output:     1. The transition probability matix for n steps
        """
        # Check to make sure that the number of steps is an integer value
        if type(n) != int:
            err_string = 'The number of steps in a discrete time markov chain'
            err_string = ' must be an integer value'
            raise StochError(err_string)
        # Compute the transition probability transition matrix for n steps
        eigen = np.linalg.eig(self.P)
        Dk = np.diag(eigen[0]**n)
        T = eigen[1]
        Tinv = np.linalg.inv(T)
        Pk = np.dot(np.dot(T,Dk),Tinv)
        return Pk

    def vector_convert(self,vector):
        """
        Procedure Name: vector_convert
        Purpose: Converts vectors to pandas data frame so that they can
                    be displayed with state space labels
        Arugments:  1. self: the markov process
                    2. vector: the vector to be converted for display
        Output:     1. The vector in display format
        """
        display_vec = pd.DataFrame(vector, index=self.state_space)
        return display_vec


        
        
        
        











    
