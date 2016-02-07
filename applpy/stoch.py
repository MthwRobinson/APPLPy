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
        else:
            state_space = range(P.shape[0])
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
        1. __repr__(self)
    """
    
    def __repr__(self):
        """
        Procedure Name: __repr__
        Purpose: Sets the default string display setting for the MC class
        Arguments:  1. self: the markov chain
        Output:     1. Print statements showing the transition probability
                        matrix and the initial state of the system
        """
        return repr(self.display())

    
    """
    Utility Class Methods

    Procedures:
        1. display(self)
        2. matrix_convert(self,matrix)
        3. vector_convert(self,vector)
    """
    def display(self,option='trans mat',n=1,method='float'):
        """
        Procedure Name: display
        Purpose: Displays the markov process in an interactive environment
        Arugments:  1. self: the markov process
        Output:     1. The transition probability matrix
                    2. The initial state of the system
                    3. The current state of the system
        """
        option_list = ['trans mat','steady state']
        if option not in option_list:
            options = ''
            for option_type in option_list:
                options += option_type+', '
            err_string = 'Invalid option. Valid options are: '
            err_string += options
            raise StochError(err_string)
        if option == 'trans mat':
            # Check to make sure that the number of steps is an integer value
            if (type(n) != int) and (n.__class__.__name__!='Symbol'):
                err_string = 'The number of steps in a discrete'
                err_string = ' time markov chain must be an integer value'
                raise StochError(err_string)
            if n == 1:
                print 'The transition probability matrix:'
                print self.P_print
            else:
                print 'The transition probability matrix after %s steps:'%(n)
                print self.matrix_convert(self.trans_mat(n,method=method))
            print '----------------------------------------'
            print 'The initial system state:'
            print self.init_print
        if option == 'steady state':
            print 'The steady state probabilities are:'
            print self.vector_convert(self.steady_state(method=method))
        if option == 'init':
            print 'The initial conditions are:'
            print self.vector_convert(self.init)
        

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

    """
    Functional Class Methods

    Procedures:
        1. probability(self,state,given)
        2. steady_state(self)
        3. trans_mat(self,n)
    """
    def probability(self,states,given=None,method='float'):
        """
        Procedure Name: probability
        Purpose: Computes the probability of reaching a state, given that
                    another state has been realized
        Arguments:  1. states: a list of tuples. The first entry in each
                        tuple is the time period when the state is realized
                        and the second entry is the state
                    2. given: an optional list of conditions, expressed as
                        tuples. When entered, the procedure conditions
                        the probability on these states
        Output:     1. A probability
        """
        # Check to make sure the states and conditional statements are
        #   in the proper form
        for state in states:
            if type(state) != tuple:
                err_string = 'Each state must be entered as a tuple'
                raise StochError(err_string)
            if len(state) != 2:
                err_string = 'Each state must be a tuple with two elements, '
                err_string += 'the first is the time period and the second '
                err_string += 'is the name of the state'
                raise StochError(err_string)
            if state[1] not in self.state_space:
                err_string = 'A state was entered that does not appear '
                err_string += 'in the state space of the Markov Chain'
                raise StochError(err_string)
        # If no conditions are given, check to make sure that initial
        #   conditions are specified
        if given == None:
            if type(self.init) != np.ndarray:
                if self.init == None:
                    err_string = 'Unconditional probabilities can only be '
                    err_string += 'computed if initial conditions are '
                    err_string += 'specified.'
                    raise StochError(err_string)
        # Make sure that the state for a time period is not specified
        #   more than once
        states.sort()
        states_specified = []
        for i in range(len(states)-1):
            states_specified.append(states[i])
            if states[i][0] == states[i+1][0]:
                err_string = 'Two different states were specified for '
                err_string += 'the same time period'
                raise StochError(err_string)
        states_specified.append(states[-1])
        if given != None:
            given.sort()
            for i in range(len(given)-1):
                err_string = 'Two different states were specified '
                err_string += 'the same time period'
                if given[i][0] == given[i+1][0]:
                    raise StochError(err_string)
                if given[i] in states_specified:
                    raise StochError(err_string)
            if given[-1] in states_specified:
                raise StochError(err_string)

        # If no conditions are specified, compute the probability
        if given == None:
            prev_time = 0
            prev_state = None
            step_mat = {1:self.P_print}
            init_states = self.init_print[0]
            total_prob = 1
            while len(states) > 0:
                current_time = states[0][0]
                current_state = states[0][1]
                time_diff = current_time - prev_time
                # If we haven't compute it yet, compute transition
                #   matrix using C-K equations. Store it in a dict
                #   so that it does not need to be computed again
                if time_diff not in step_mat and time_diff != 0:
                    trans = self.trans_mat(n=time_diff,method=method)
                    step_mat[time_diff] = self.matrix_convert(trans)
                # If this is the first iteration, condition on the
                #   distribution of the initial states
                if prev_state == None:
                    if time_diff == 0:
                        total_prob *= init_states[current_state]
                    else:
                        init_prob = 0
                        for state in self.state_space:
                            prob_to_state = init_states[state]
                            p_n = step_mat[time_diff][current_state][state]
                            prob_to_state *= p_n
                            init_prob += prob_to_state
                        total_prob *= init_prob
                # If this is not the first iteration, compute the
                #   transition probability
                else:
                    total_prob *= step_mat[time_diff][current_state][prev_state]
                prev_state = current_state
                prev_time = current_time
                del states[0]
            # If conditions are specified, compute the probability
        if type(given) == list:
            if given[0][0] < states[0][0]:
                shift = given[0][0]
                total_states = given + states
                for i,element in enumerate(total_states):
                    total_states[i] = (element[0]-shift,element[1])
                init_prob = self.init_print[0][given[0][0]]
                total_prob = self.probability(states=total_states,
                                              method=method)/init_prob
            else:
                total_states = given + states
                total_prob = self.probability(states=total_states,
                                              method=method)
        return total_prob
                        

    def steady_state(self, method = 'float'):
        """
        Procedure Name: steady_state
        Purpose: Computes the long run fraction of time spent in state i
        Arguments:  1. None
        Output:     1. A vector containing the long run fraction of time
                        spent in state i
        """
        # Need to add code to check to make sure that the markov chain
        #   is irreducible, aperiodic and positive recurrent

        if method not in ['float', 'rational']:
            raise StochError('Method must be either float or rational')

        trans_mat = self.P
        size = np.size(trans_mat,axis=0)
        # The steady state probabilities are found by solving the following
        #   system: Pj = sum( Pij*Pj ) for all j, 1 = sum(Pj)
        if method == 'float':
            trans_mat_T = trans_mat.transpose()
            A = trans_mat_T - np.identity(size)
            B = np.vstack((A,np.array([1 for i in range(size)])))
            B = B[1:,]
            a = [0 for i in range(size)]
            a[-1]=1
            b = np.array(a).reshape(-1,1)
            soln = np.dot(np.linalg.inv(B),b)
            return soln
        # The stead state probabilities are computing by explicity solving
        #   the system of equations using computer algebra
        if method == 'rational':
            a = symbols('a0:%d'%(size),positive=True)
            eqns = []
            norm_eqn = -1
            for i in range(1,size):
                current_eqn = 0
                for j in range(size):
                    current_eqn += trans_mat[j][i]*a[j]
                current_eqn -= a[i]
                norm_eqn += a[i]
                eqns.append(current_eqn)
            eqns.append(norm_eqn+a[0])
            solns = solve(eqns)
            soln = [solns[a[i]] for i in range(size)]
            return np.array(soln)
            
        
    def trans_mat(self,n=Symbol('n',positive=True),method='float'):
        """
        Procedure Name: trans_mat
        Purpose: Computes the state of the system after n steps
        Arguments:  1. n: the number of steps the system takes forward
        Output:     1. The transition probability matix for n steps
        """
        # Check to make sure that the number of steps is an integer value
        if (type(n) != int) and (n.__class__.__name__!='Symbol'):
            err_string = 'The number of steps in a discrete time markov chain'
            err_string = ' must be an integer value'
            raise StochError(err_string)
        # Compute the transition probability transition matrix for n steps
        # To efficiently compute powers of the matrix, this algorithm
        #   finds the eigen decomposition of the matrix, and then computes
        #   the power of the elements in the diagonal matrix
        if method == 'float':
            eigen = np.linalg.eig(self.P)
            Dk = np.diag(eigen[0]**n)
            T = eigen[1]
            Tinv = np.linalg.inv(T)
            Pk = np.dot(np.dot(T,Dk),Tinv)
            return Pk
        if method == 'rational':
            Pk = self.P
            for i in range(n-1):
                Pk = np.dot(self.P,Pk)
            return Pk



        
        
        











    
