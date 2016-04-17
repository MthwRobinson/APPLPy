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

from __future__ import division

from sympy import *

from .rv import *
from .stoch import *
from .plot import *
from .dist_type import *
from .stats import *
from .bayes import *
from .queue_dist import *
from .bivariate import *
from .timeseries import *

x,y,z,t=symbols('x y z t')
k,m,n=symbols('k m n',integers=True)
f,g,h=symbols('f g h',cls=Function)
import sys
sys.display_hook=pprint

def Menu():
    print '-----------------'
    print 'Welcome to ApplPy'
    print '-----------------'
    print ''
    print 'ApplPy Procedures'
    print ""
    print 'Procedure Notation'
    print ""
    print 'Capital letters are random variables'
    print 'Lower case letters are number'
    print 'Greek letters are parameters'
    print 'gX indicates a function'
    print 'n and r are positive integers where n>=r'
    print 'Square brackets [] denote a list'
    print 'Curly bracks {} denote an optional variable'
    print ""
    print ""

    print 'RV Class Procedures'
    print 'X.variate(n,x),X.verifyPDF()'
    print ""

    print 'Functional Form Conversion'
    print 'CDF(X,{x}),CHF(X,{x}),HF(X,{x}),IDF(X,{x})'
    print 'PDF(X,{x}),SF(X,{x}),BootstrapRV([data])'
    print 'Convert(X,{x})'
    print ""    

    print 'Procedures on One Random Variable'
    print 'ConvolutionIID(X,n),CoefOfVar(X),ExpectedValue(X,gx)'
    print 'Kurtosis(X),MaximumIID(X,n),Mean(X),MGF(X)'
    print 'MinimumIID(X,n),OrderStat(X,n,r),ProductIID(X,n)'
    print 'Skewness(X),Transform(X,gX),Truncate(X,[x1,x2])'
    print 'Variance(X)'
    print ""

    print 'Procedures on Two Random Variables'
    print 'Convolution(X,Y),Maximum(X,Y),Minimum(X,Y)'
    print 'Mixture([p1,p2],[X,Y]),Product(X,Y)'
    print ""

    print 'Statistics Procedures'
    print 'KSTest(X,[sample]), MOM(X,[sample],[parameters])'
    print 'MLE(X,[sample],[parameters],censor)'
    print ""

    print 'Utilities'
    print 'PlotDist(X,{[x1,x2]}),PlotDisplay([plotlist],{[x1,x2]})'
    print 'PPPlot(X,[sample]),QQPlot(X,[sample])'
    print ""

    print 'Continuous Distributions'
    print 'ArcSinRV(),ArcTanRV(alpha,phi),BetaRV(alpha,beta)'
    print 'CauchyRV(a,alpha),ChiRV(N),ChiSquareRV(N),ErlangRV(theta,N)'
    print 'ErrorRV(mu,alpha,d),ErrorIIRV(a,b,c),ExponentialRV(theta)'
    print 'ExponentialPowerRV(theta,kappa),ExtremeValueRV(alpha,beta)'
    print 'FRV(n1,n2),GammaRV(theta,kappa),GompertzRV(theta,kappa)'
    print 'GeneralizedParetoRV(theta,delta,kappa),IDBRV(theta,delta,kappa)'
    print 'InverseGaussianRV(theta,mu),InverseGammaRV(alpha,beta)'
    print 'KSRV(n),LaPlaceRV(omega,theta), LogGammaRV(alpha,beta)'
    print 'LogisticRV(kappa,theta),LogLogisticRV(theta,kappa)'
    print 'LogNormalRV(mu,sigma),LomaxRV(kappa,theta)'
    print 'MakehamRV(theta,delta,kappa),MuthRV(kappa),NormalRV(mu,sigma)'
    print 'ParetoRV(theta,kappa),RayleighRV(theta),TriangularRV(a,b,c)'
    print 'TRV(N),UniformRV(a,b),WeibullRV(theta,kappa)'
    print ""

    print 'Discrete Distributions'
    print 'BenfordRV(),BinomialRV(n,p),GeometricRV(p),PoissonRV(theta)'
    print ''
