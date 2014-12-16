from __future__ import division

from pylab import (plot, xlabel, ylabel, title, grid, arange,
                   ion, ioff)
from sympy import (symbols)
x,y,z,t,v=symbols('x y z t v')

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
Plotting Module

Defines procedures for plotting random variables

"""
    
def mat_plot(funclist,suplist,lab1=None,lab2=None,ftype='continuous'):
    """
    Procedure Name: mat_plot
    Purpose: Create a matplotlib plot of a random variable
    Arguments:  1. RVar: A random variable
                2. suplist: The support of the plot
    Output:     1. A plot of the random variable
    """
    # if the random variable is continuous, plot the function
    if ftype=='continuous':
        for i in range(len(funclist)):
            if funclist[i]=='0':
                continue
            if 'x' in funclist[i]:
                x=arange(suplist[i],suplist[i+1],0.01)
                s=eval(funclist[i])
                plot(x,s,linewidth=1.0,color='green')
            else:
                plot([suplist[i],suplist[i+1]],
                     [funclist[i],funclist[i]],
                      linewidth=1.0,color='green')
        if lab1=='idf':
            xlabel('s')
        else:
            xlabel('x')
        if lab1!=None:
            ylabel(lab1)
        if lab2!=None:
            title(lab2)
        grid(True)
    # If the random variable is discrete, plot the function
    if ftype=='discrete':
        plot(suplist,funclist,'ro')
        if lab1=='F-1(s)':
            xlabel('s')
        else:
            xlabel('x')
        if lab1!=None:
            ylabel(lab1)
        if lab2!=None:
            title(lab2)
        grid(True)

def prob_plot(Sample,Fitted,plot_type):
    """
    Procedure Name: prob_plot
    Purpose: Create a mat plot lib plot to compare sample distributions
                with theoretical models
    Arguments:  1. Sample: Data sample quantiles
                2. Model: Model quantiles
    Output:     1. A probability plot that compares data with a model
    """
    plot(Fitted,Sample,'ro')
    x=arange(min(min(Sample),min(Fitted)),
             max(max(Sample),max(Fitted)),0.01)
    s=x
    plot(x,s,linewidth=1.0,color='red')
    if plot_type=='QQ Plot':
        xlabel('Model Quantiles')
        ylabel('Sample Quantiles')
    elif plot_type=='PP Plot':
        xlabel('Model CDF')
        ylabel('Sample CDF')
    title(plot_type)
    grid(True)
    

