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
Distribution Subclass Module

Defines commonly used distributions as subclasses of the
    RV class

"""

from rv import *

def param_check(param):
    for i in range(len(param)):
        if type(param[i])!=int and type(param[i])!=float:
            return False
        else:
            return True

"""
Continuous Distributions

"""

class BetaRV(RV):
    def __init__(self,alpha=Symbol('alpha',positive=True),
                 beta=Symbol('beta'),positive=True):
        X_dummy=RV((gamma(alpha+beta)*(x**(alpha-1))*(1-x)**(beta-1))/
               (gamma(alpha)*gamma(beta)),[0,1])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class CauchyRV(RV):
    def __init__(self,a=Symbol('a'),
                 alpha=Symbol('alpha'),positive=True):
        X_dummy=RV((1)/(alpha*pi*(1+((x-a)**2/alpha**2))),[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[a,alpha]

    def variate(self,n=1,s='sim'):
        # If no parameter is specified, return an error
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        # Generate exponential variates
        idf_func=self.parameter[0]-cot(pi*t)*self.parameter[1]
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val).evalf()
            varlist.append(var)
        varlist.sort()
        return varlist

class ChiRV(RV):
    def __init__(self,N=Symbol('N',positive=True)):
        X_dummy=RV(((x**(N-1))*exp(-x**2/2))/
                   (2**((N/2)-1)*gamma(N/2)),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ChiSquareRV(RV):
    def __init__(self,N=Symbol('N',positive=True)):
        X_dummy=RV((x**((N/2)-1)*exp(-x/2))/
                   (2**(N/2)*gamma(N/2)),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ErlangRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 N=Symbol('N',positive=True)):
        X_dummy=RV((theta*(theta*x)**(N-1)*exp(-theta*x))/
                   (factorial(N-1)),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ExponentialRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True)):
        X_dummy=RV([theta*exp(-theta*x)],[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[theta]

    def variate(self,n=1,s='sim'):
        # If no parameter is specified, return an error
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        # Generate exponential variates
        idf_func=(-ln(1-t))/(self.parameter[0])
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        varlist.sort()
        return varlist

class ExponentialPowerRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa',positive=True)):
        X_dummy=RV(exp(1-exp(theta*x**(kappa)))*exp(theta*x**(kappa))*
                   theta*kappa*x**(kappa-1),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[theta,kappa]

    def variate(self,n=1,s='sim'):
        # If no parameter is specified, return an error
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        # Generate exponential power variates
        idf_func=exp((-ln(self.parameter[0])+ln(ln(1-ln(1-s))))/
                     self.parameter[1])
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
            varlist.sort()
        return varlist

class ExtremeValueRV(RV):
    def __init__(self,alpha=Symbol('alpha'),beta=Symbol('beta')):
        X_dummy=RV((beta*exp((x*beta)-((exp(x*beta))/alpha)))/
                   alpha,[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[alpha,beta]

    def variate(self,n=1,s='sim'):
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        idf_func=(ln(self.parameter[0])+ln(ln(-1/(t-1))))/self.parameter[1]
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        return varlist

class GammaRV(RV):
    def __init__(self,theta=Symbol('theta'),kappa=Symbol('kappa')):
        X_dummy=RV((theta*(theta*x)**(kappa-1)*exp(-theta*x))/(gamma(kappa)),
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[theta,kappa]

class GompertzRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa')):
        X_dummy=RV([theta*kappa**(x)*exp(-(theta*(kappa**(x)-1))/ln(kappa))],
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[theta,kappa]

    def variate(self,n=1,s='sim'):
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        idf_func=-((ln(self.parameter[0])-
                   ln(self.parameter[0]-ln(1-t)*ln(self.parameter[1])))
                   /ln(self.parameter[1]))
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        return varlist

class InverseGaussianRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 mu=Symbol('mu',positive=True)):
        X_dummy=RV([(1/2)*sqrt(2)*sqrt(theta/(pi*x**3))*
                    exp(-(1/2)*(theta*(x-mu)**2)/(mu**(2)*x))],
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype


class InverseGammaRV(RV):
    def __init__(self,alpha=Symbol('alpha',positive=True),
                 beta=Symbol('beta',positive=True)):
        X_dummy=RV([(x**(1-alpha)*exp(-1/(x*beta)))/
                    (gamma(alpha)*beta**(alpha))],
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class LogGammaRV(RV):
    def __init__(self,alpha=Symbol('alpha',positive=True),
                 beta=Symbol('beta',positive=True)):
        X_dummy=RV([(exp(x*beta)*exp(-exp(x)/alpha))/
                    (alpha**(beta)*gamma(beta))],[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class LogisticRV(RV):
    def __init__(self,kappa=Symbol('kappa',positive=True),
                 theta=Symbol('theta',positive=True)):
        X_dummy=RV([(theta**(kappa)*kappa*exp(kappa*x))/
                    (1+(theta*exp(x))**kappa)**2],[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[kappa,theta]

    def variate(self,n=1,s='sim'):
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        idf_func=-((ln(-t/(t-1))+self.parameter[0]*ln(self.parameter[1]))/
                   self.parameter[1])
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        return varlist

class LogLogisticRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa',positive=True)):
        X_dummy=RV([(theta*kappa*(theta*x)**(kappa-1))/
                    (1+(theta*x)**(kappa))**2],[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[theta,kappa]

    def variate(self,n=1,s='sim'):
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        idf_func=exp((ln(-t/(t-1))-self.parameter[1]*ln(self.parameter[0]))/
                     self.parameter[1])
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        return varlist
    
class LogNormalRV(RV):
    def __init__(self,mu=Symbol('mu'),
                 sigma=Symbol('sigma',positive=True)):
        X_dummy=([(1/2)*(sqrt(2)*exp((-1/2)*((ln(x)-mu)**2)/(sigma**2)))/
                  (sqrt(pi)*x*sigma)],[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class LomaxRV(RV):
    def __init__(self,kappa=Symbol('kappa',positive=True),
                 theta=Symbol('theta',positive=True)):
        X_dummy=([theta*kappa*(1+theta*x)**(-kappa-1)],[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameters=[kappa,theta]

    def variate(self,n=1,s='sim'):
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        idf_func=((1-t)**(1/self.parameter[0])-1)/self.parameter[1]
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        return varlist

class MuthRV(RV):
    def __init__(self,kappa=Symbol('kappa',positive=True)):
        X_dummy=RV([(exp(kappa*x)-kappa)*exp((-exp(kappa*x)/kappa)+
                                             kappa*x+(1/kappa))]
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class NormalRV(RV):
    def __init__(self,mu=Symbol('mu'),
                 sigma=Symbol('sigma',positive=True)):
        X_dummy=RV((exp((-(x-mu)**2)/(2*sigma**2))*sqrt(2))/(2*sigma*sqrt(pi))
                   ,[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[mu,sigma]

class ParetoRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa',positive=True)):
        X_dummy=RV([(kappa*theta**(kappa))/(x**(kappa+1))],[theta,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class Rayleigh(RV):
    def __init__(self,theta=Symbol('theta',positive=True)):
        X_dummy=RV([2*theta**(2)*x*exp(-theta**(2)*x**2)],[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class TriangularRV(RV):
    def __init__(self,a=Symbol('a'),b=Symbol('b'),c=Symbol('c')):
        X_dummy=RV([(2*(x-a))/((c-a)*(b-a)),(2*(c-x))/((c-a)*(c-b))],[a,b,c])
        self.func=X_dummy.func
        a=Symbol('a');b=Symbol('b');c=Symbol('c')
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[a,b,c]

class TRV(RV):
    def __init__(self,N=Symbol('N'),positive=True):
        X_dummy=RV([(gamma(N/2+1/2)*(1+((x**2)/N))**(-(N/2)-1/2))/
                    (sqrt(N*pi)*gamma(N/2))],[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class UniformRV(RV):
    def __init__(self,a=Symbol('a'),b=Symbol('b')):
        X_dummy=RV((b-a)**(-1),[a,b])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[a,b]

    def variate(self,n=1,sim='sim'):
        # If no parameter is specified, return an error
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        # Generate uniform variates
        idf_func=-t*self.parameter[0]+t*self.parameter[1]+self.parameter[0]
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        varlist.sort()
        return varlist

class WeibullRV(RV):   
    def __init__(self,theta=Symbol('theta'),kappa=Symbol('kappa')):
        X_dummy=RV(kappa*theta**(kappa)*x**(kappa-1)*exp(-(theta*x)**kappa),
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[theta,kappa]


    def variate(self,n=1,s='sim'):
        # If no parameter is specified, return an error
        if param_check(self.parameter)==False:
            raise RVError('Not all parameters specified')

        # Generate weibull variates
        idf_func=exp(-(-ln(-ln(1-t))+self.parameter[1]*ln(self.parameter[0]))/
                     self.parameter[1])
        varlist=[]
        for i in range(n):
            if s=='sim':
                val=random()
            else:
                val=s
            var=idf_func.subs(t,val)
            varlist.append(var)
        varlist.sort()
        return varlist

"""
Discrete Distributions

"""

class BenfordRV(RV):
    def __init__(self):
        X_dummy=RV([(ln((1/x)+1))/(ln(10))],[1,9],['Discrete','pdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class BinomialRV(RV):
    def __init__(self,N=Symbol('N',positive=True),
                 p=Symbol('p',positive=True)):
        X_dummy=RV([(factorial(N)*p**(x)*(1-p)**(N-x))/
                    (factorial(N-x)*factorial(x))],[0,N],
                   ['Discrete','pdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class GeometricRV(RV):
    def __init__(self,p=Symbol('p',positive=True)):
        X_dummy=RV([p*(1-p)**(x-1)],[1,oo],['Discrete','pdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class PoissonRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True)):
        X_dummy=RV([(theta**(x)*exp(-theta))/factorial(x)],
                   [0,oo],['Discrete','pdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

