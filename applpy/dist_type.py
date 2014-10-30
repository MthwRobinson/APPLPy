from __future__ import division
from sympy import (Symbol, symbols, oo, integrate, summation, diff,
                   exp, pi, sqrt, factorial, ln, floor, simplify,
                   solve, nan, plot, Add, Mul, Integer, function,
                   binomial)
from random import random
from .rv import (RV, RVError, CDF, CHF, HF, IDF, IDF, PDF, SF,
                 BootstrapRV, Convert)
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
Distribution Subclass Module

Defines commonly used distributions as subclasses of the
    RV class

"""



def param_check(param):
    flag=True
    count=0
    for element in param:
        try:
            if element.__class__.name=='Symbol':
                flag=False
        except:
            if type(element)=='Symbol':
                flag=False
    return flag

"""
Continuous Distributions

"""

class ArcSinRV(RV):
    def __init__(self):
        X_dummy=RV(1/(pi*sqrt(x*(1-x))),[0,1])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ArcTanRV(RV):
    def __init__(self,alpha=Symbol('alpha',positive=True),
                 phi=Symbol('phi')):
        # Return an error if invalid parameters are entered
        if alpha in [-oo,oo]:
            if phi in [-oo,oo]:
                err_string='Both parameters must be finite'
                raise RVError(err_string)
        if alpha <= 0:
            err_string='Alpha must be positive'
            raise RVError(err_string)
        X_dummy=RV(alpha/(atan(alpha*phi)+pi/2)*(1+alpha**2*(x-phi)**2),
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class BetaRV(RV):
    def __init__(self,alpha=Symbol('alpha',positive=True),
                 beta=Symbol('beta'),positive=True):
        if alpha in [-oo,oo]:
            if beta in [-oo,oo]:
                err_string='Both parameters must be finite'
                raise RVError(err_string)
        if alpha<=0 and alpha.__class__.__name__!='Symbol' :
                if beta<=0 and beta.__class__.name__!='Symbol':
                    err_string='Both parameters must be positive'
                    raise RVError(err_string)
        X_dummy=RV((gamma(alpha+beta)*(x**(alpha-1))*(1-x)**(beta-1))/
               (gamma(alpha)*gamma(beta)),[0,1])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class CauchyRV(RV):
    def __init__(self,a=Symbol('a'),
                 alpha=Symbol('alpha'),positive=True):
        if a in [-oo,oo]:
            err_string='Both parameters must be finite'
            if alpha in [-oo,oo]:
                raise RVError(err_string)
        if alpha.__class__.__name__!='Symbol':
            if alpha<=0:
                err_string='alpha must be positive'
                raise RVError(err_string)
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
    def __init__(self,N=Symbol('N',positive=True,
                               integer=True)):
        if N.__class__.__name__!='Symbol':
            if N<=0 or type(N)!=int:
                err_string='N must be a positive integer'
                raise RVError(err_string)
        X_dummy=RV(((x**(N-1))*exp(-x**2/2))/
                   (2**((N/2)-1)*gamma(N/2)),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ChiSquareRV(RV):
    def __init__(self,N=Symbol('N',positive=True,
                               integer=True)):
        if N.__class__.__name__!='Symbol':
            if N<=0 or type(N)!=int:
                err_string='N must be a positive integer'
                raise RVError(err_string)
        X_dummy=RV((x**((N/2)-1)*exp(-x/2))/
                   (2**(N/2)*gamma(N/2)),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ErlangRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 N=Symbol('N',positive=True,integer=True)):
        if N.__class__.__name__!='Symbol':
            if N<=0 or type(N)!=int:
                err_string='N must be a positive integer'
                raise RVError(err_string)
        if theta.__class__.__name__!='Symbol':
            if theta<=0:
                err_string='theta must be positive'
                raise RVError(err_string)
        if theta in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV((theta*(theta*x)**(N-1)*exp(-theta*x))/
                   (factorial(N-1)),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ErrorRV(RV):
    def __init__(self,mu=Symbol('mu',positive=True),
                 alpha=Symbol('alpha'),d=Symbol('d')):
        if mu.__class__.__name__!='Symbol':
            if mu<=0:
                err_string='mu must be positive'
                raise RVError(err_string)
        if mu in [-oo,oo]:
            if alpha in [-oo,oo]:
                if d in [-oo,oo]:
                    err_string='all parameters must be finite'
                    raise RVError(err_string)
        X_dummy=RV(mu*exp(-abs(mu*(x-d))**alpha)/
                   (2*gamma(1+1/alpha)),[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ErrorIIRV(RV):
    def __init__(self,a=Symbol('a'),b=Symbol('b'),
                 c=Symbol('c')):
        if a in [-oo,oo]:
            if b in [-oo,oo]:
                if c in [-oo,oo]:
                    err_string='all parameters must be finite'
                    raise RVError(err_string)
        X_dummy=RV(exp(-((abs(x-a))**(2/c)/(2*b))/
                       ((b**(c/2))*2**(c/2+1)*gamma(c/2+1))),
                   [-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class ExponentialRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True)):
        if theta.__class__.__name__!='Symbol':
            if theta<=0:
                err_string='theta must be positive'
                raise RVError(err_string)
        if theta in [-oo,oo]:
            err_string='theta must be finite'
            raise RVError(err_string)
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
        if theta.__class__.__name__!='Symbol':
            if kappa.__class__.__name__!='Symbol':
                if theta<=0 or kappa<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
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
        if alpha in [-oo,oo]:
            if beta in [-oo,oo]:
                err_string='both parameters must be finite'
                raise RVError(err_string)
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

class FRV(RV):
    def __init__(self,n1=Symbol('n1',positive=True),
                 n2=Symbol('n2',positive=True)):
        if n1.__class__.__name__!='Symbol':
            if n2.__class__.__name__!='Symbol':
                if n1<=0 or n2<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if n1 in [-oo,oo] or n2 in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV(gamma((n1+n2)/2)*(n1/n2)**(n1/2)*x**(n1/2-1)/
                   gamma(n1/2)*gamma(n2/2)*((n1/n2)*x+1)**((n1+n2)/2),
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class GammaRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa',positive=True)):
        if theta.__class__.__name__!='Symbol':
            if kappa.__class__.__name__!='Symbol':
                if theta<=0 or kappa<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if theta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV((theta*(theta*x)**(kappa-1)*exp(-theta*x))/(gamma(kappa)),
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[theta,kappa]

class GeneralizedParetoRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 delta=Symbol('delta'),kappa=Symbol('kappa')):
        if theta.__class__.__name__!='Symbol':
            if theta<=0:
                err_string='theta must be positive'
                raise RVError(err_string)
        if theta in [-oo,oo] or delta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='all parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV((theta+kappa/(x+delta))*(1+x/delta)**(-kappa)*
                   exp(-theta*x),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class GompertzRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa')):
        if theta.__class__.__name__!='Symbol':
            if theta<=0:
                err_string='theta must be positive'
                raise RVError(err_string)
        if theta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
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

class IDBRV(RV):
    def __init__(self,theta=Symbol('theta'),delta=Symbol('delta'),
                 kappa=Symbol('kappa')):
        if theta in [-oo,oo] or delta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='all parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV(1-(1+kappa*x)**(-theta/kappa)*
                   exp(-delta*x**2/2),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class InverseGaussianRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 mu=Symbol('mu',positive=True)):
        if theta.__class__.__name__!='Symbol':
            if mu.__class__.__name__!='Symbol':
                if theta<=0 or mu<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if theta in [-oo,oo] or mu in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV([(1/2)*sqrt(2)*sqrt(theta/(pi*x**3))*
                    exp(-(1/2)*(theta*(x-mu)**2)/(mu**(2)*x))],
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype


class InverseGammaRV(RV):
    def __init__(self,alpha=Symbol('alpha',positive=True),
                 beta=Symbol('beta',positive=True)):
        if alpha.__class__.__name__!='Symbol':
            if beta.__class__.__name__!='Symbol':
                if alpha<=0 or beta<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if alpha in [-oo,oo] or beta in [-oo,oo]:
            err_string='both parameters must be finite'
        X_dummy=RV([(x**(1-alpha)*exp(-1/(x*beta)))/
                    (gamma(alpha)*beta**(alpha))],
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class KSRV(RV):
    def __init__(self,n=Symbol('n',positive=True,
                               integer=True)):
        if n.__class__.__name__!='Symbol':
            if n<=0:
                if type(n)!=int:
                    err_string='n must be a positive integer'
                    raise RVError(err_string)
        #Phase 1
        N=n
        m=floor(3*N/2)+(N%2)-1
        vv=range(m+1)
        vvalue=[]
        for i in range(len(vv)):
            vvalue.append(0)
        vv=dict(zip(vv,vvalue))
        vv[0]=0
        g=1/(2*N)
        mm=0
        for i in range(1,N):
            mm+=1
            vv[mm]=i*g
        for j in range(2*floor(N/2)+1,2*N,2):
            mm+=1
            vv[mm]=j*g
        #Phase 2
        # Generate the c array
        cidx=[];cval=[]
        for k in range(1,m+1):
            cidx.append(k)
            cval.append((vv[k-1]+vv[k])/2)
        c=dict(zip(cidx,cval))
        # Generate the x array
        xidx=[];xval=[]
        for k in range(1,N+1):
            xidx.append(k)
            xval.append((2*k-1)*g)
        x=dict(zip(xidx,xval))
        # Generate an NxN A array
        aidx=range(1,N+1);aval=[]
        for i in aidx:
            aval.append(0)
        arow=dict(zip(aidx,aval));A=dict(zip(aidx,aval))
        for i in aidx:
            A[i]=arow
        # Insert values into the A array
        for i in range(2,N+1):
            for j in range(1,i):
                A[i][j]=0
        for k in range(1,m+1):
            for i in range(1,N+1):
                for j in range(i,N+1):
                    A[i][j]=0
            z=max(floor(N*c[k]-1/2),0)
            l=min(floor(2*N*c[k])+1,N)
            for i in range(1,N+1):
                for j in range(max(i,z+1),min(N,i+l-1)+1):
                    A[i][j]=1
        # Create a 1xm P array
        Pidx=[];Pval=[]
        for i in range(1,m+1):
            Pidx.append(i)
            Pval.append(0)
        P=dict(zip(Pidx,Pval))
        # Create an NxN F array
        fidx=range(1,N+1);fval=[]
        for i in fidx:
            fval.append(0)
        frow=dict(zip(fidx,fval));F=dict(zip(fidx,fval))
        for i in fidx:
            F[i]=frow
        # Create an NxN V array
        vidx=range(1,N+1);vval=[]
        for i in vidx:
            vval.append(0)
        vrow=dict(zip(vidx,vval));V=dict(zip(vidx,vval))
        for i in vidx:
            V[i]=vrow
        # Create a list of indexed u variables
        varstring='u:'+str(N+1)
        u=symbols(varstring)
        for k in range(2,m+1):
            z=int(max(floor(N*c[k]-1/2),0))
            l=int(min(floor(2*N*c[k])+1,N))
            F[N][N]=integrate(1,(u[N],x[N]-v,1))
            V[N][N]=integrate(1,(u[N],u[N-1],1))
            for i in reversed(range(2,N,-1)):
                if i+l>N:
                    S=0
                else:
                    S=F[i+1][i+l]
                if i+l>N+1:
                    F[i][N]=integrate(V[i+1][N],
                                      (u[i],x[N]-v,floor(x[i]+c[k])))
                    V[i][N]=integrate(V[i+1][N],
                                      (u[i],u[i-1],floor(x[i]+c[k])))
                if i+l==N+1:
                    F[i][N]=integrate(V[i+1][N],
                                      (u[i],x[N]-v,x[i]+v))
                if i+l<N+1:
                    F[i][i+l-1]=integrate(V[i+1][i+l-1]+S,
                                          (u[i],x[N]-v,x[i]+v))
                S+=F[i+1][min(i+l-1,N)]
                for j in reversed(range(max(i+1,z+2),min(N-1,i+l-2)+1,-1)):
                    F[i][j]=integrate(V[i+1][j]+S,
                                      (u[i],x[j]-v,x[j+1]-v))
                    V[i][j]=integrate(V[i+1][j]+S,
                                      (u[i],u[i-1],x[j+1]-v))
                    S+=F[i+1][j]
                if z+1<=i:
                    V[i][i]=integrate(S,(u[i],u[i-1],x[i+1]-v))
                if z+1>i:
                    V[i][z+1]=integrate(V[i+1][z+1]+S,
                                        (u[i],u[i-1],x[z+2]-v))
                if z+1<i:
                    F[i][i]=integrate(S,(u[i],x[i]-v,x[i+1]-v))
            if l==N:
                S=0
            else:
                S=F[2][l+1]
            if l<N:
                F[1][l]=integrate(V[2][l]+S,(u[1],x[l]-v,x[1]+v))
            S+=F[2][j]
            for j in reversed(range(max(2,z+1),min(N-1,l-1)+1,-1)):
                F[1][j]=integrate(V[2][j]+S,
                          (u[1],(x[j]-v)*(floor(x[j]-c[k])+1),
                           x[j+1]-v))
                S+=F[2][j]
            if z==0:
                F[1][1]=integrate(S,(u[1],0,x[2]-v))
            P[k]=0
            for j in range(z+1,l+1):
                P[k]+=F[1][j]
            P[k]=factorial(N)*P[k]
        # Create the support and function list for the KSRV
        KSspt=[];KSCDF=[]
        x=Symbol('x')
        for i in range(0,m+1):
            KSspt.append(vv[i]+1/(2*N))
        for i in range(1,m+1):
            func=(x-1)/(2*N)
            KSCDF.append(simplify(func.subs(v,P[i])))
        # Remove redundant elements from the list
        KSCDF2=[];KSspt2=[]
        KSspt2.append(KSspt[0])
        KSCDF2.append(KSCDF[0])
        for i in range(1,len(KSCDF)):
            if KSCDF[i]!=KSCDF[i-1]:
                KSCDF2.append(KSCDF[i])
                KSspt2.append(KSspt[i])
        KSspt2.append(KSspt[-1])
        X_dummy=RV(KSCDF2,KSspt2,['continuous','cdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class LaPlaceRV(RV):
    def __init__(self,omega=Symbol('omega',positive=True),
                 theta=Symbol('theta')):
        if omega.__class__.__name__!='Symbol':
            if omega<=0:
                err_string='omega must be positive'
        if omega in [-oo,oo] or theta in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV(exp(-abs(x-theta)/omega)/(2*omega),[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        
class LogGammaRV(RV):
    def __init__(self,alpha=Symbol('alpha',positive=True),
                 beta=Symbol('beta',positive=True)):
        if alpha.__class__.__name__!='Symbol':
            if beta.__class__.__name__!='Symbol':
                if alpha<=0 or beta<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if alpha in [-oo,oo] or beta in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV([(exp(x*beta)*exp(-exp(x)/alpha))/
                    (alpha**(beta)*gamma(beta))],[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class LogisticRV(RV):
    def __init__(self,kappa=Symbol('kappa',positive=True),
                 theta=Symbol('theta',positive=True)):
        if kappa.__class__.__name__!='Symbol':
            if theta.__class__.__name__!='Symbol':
                if kappa<=0 or theta<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if kappa in [-oo,oo] or theta in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
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
        if kappa.__class__.__name__:
            if theta.__class__.__name__:
                if theta<=0 or kappa<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if theta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
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
        if sigma.__class__.__name__!='Symbol':
            if sigma<=0:
                err_string='sigma must be positive'
                raise RVError(err_string)
        if mu in [-oo,oo] or sigma in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV([(1/2)*(sqrt(2)*exp((-1/2)*((ln(x)-mu)**2)/(sigma**2)))/
                  (sqrt(pi)*x*sigma)],[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class LomaxRV(RV):
    def __init__(self,kappa=Symbol('kappa',positive=True),
                 theta=Symbol('theta',positive=True)):
        if kappa.__class__.__name__!='Symbol':
            if theta.__class__.__name__!='Symbol':
                if kappa<=0 or theta<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if kappa in [-oo,oo] or theta in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV([theta*kappa*(1+theta*x)**(-kappa-1)],[0,oo])
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

class MakehamRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 delta=Symbol('delta',positive=True),
                 kappa=Symbol('kappa')):
        if theta.__class__.__name__!='Symbol':
            if delta.__class__.__name__!='Symbol':
                if theta<=0 or delta<=0:
                    err_string='alpha and delta must be positive'
                    raise RVError(err_string)
        if theta in [-oo,oo] or delta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='all parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV((theta+delta*kappa**x)*
                   exp(-theta*x-delta*(kappa**x-1)/log(kappa)),[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class MuthRV(RV):
    def __init__(self,kappa=Symbol('kappa',positive=True)):
        if kappa.__class__.__name__!='Symbol':
            if kappa<=0:
                err_string='kappa must be positive'
                raise RVError(err_string)
        if kappa in [-oo,oo]:
            err_string='kappa must be finite'
            raise RVError(err_string)
        X_dummy=RV([(exp(kappa*x)-kappa)*exp((-exp(kappa*x)/kappa)+
                                             kappa*x+(1/kappa))],
                   [0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class NormalRV(RV):
    def __init__(self,mu=Symbol('mu'),
                 sigma=Symbol('sigma',positive=True)):
        if sigma.__class__.__name__!='Symbol':
            if sigma<=0:
                err_string='sigma must be positive'
                raise RVError(err_string)
        if sigma in [-oo,oo] or mu in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV((exp((-(x-mu)**2)/(2*sigma**2))*sqrt(2))/(2*sigma*sqrt(pi))
                   ,[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[mu,sigma]

class ParetoRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa',positive=True)):
        if theta.__class__.__name__!='Symbol':
            if kappa.__class__.__name__!='Symbol':
                if theta<=0 or kappa<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if theta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV([(kappa*theta**(kappa))/(x**(kappa+1))],[theta,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class RayleighRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True)):
        if theta.__class__.__name__!='Symbol':
            if theta<=0:
                err_string='both parameters must be positive'
                raise RVError(err_string)
        if theta in [-oo,oo]:
            err_string='both parameters must be finite'
        X_dummy=RV([2*theta**(2)*x*exp(-theta**(2)*x**2)],[0,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class TriangularRV(RV):
    def __init__(self,a=Symbol('a'),b=Symbol('b'),c=Symbol('c')):
        if a.__class__.__name__!='Symbol':
            if b.__class__.__name__!='Symbol':
                if c.__class__.__name__!='Symbol':
                    if a>=b or b>=c or a>=c:
                        err_string='the parameters must be in ascending order'
                        raise RVError(err_string)
        if a in [-oo,oo] or b in [-oo,oo] or c in [-oo,oo]:
            err_string='all parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV([(2*(x-a))/((c-a)*(b-a)),(2*(c-x))/((c-a)*(c-b))],[a,b,c])
        self.func=X_dummy.func
        a=Symbol('a');b=Symbol('b');c=Symbol('c')
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[a,b,c]

class TRV(RV):
    def __init__(self,N=Symbol('N'),positive=True,integer=True):
        if N.__class__.__name__!='Symbol':
            if N<=0:
                if type(N)!=int:
                    err_string='N must be a positive integer'
                    raise RVError(err_string)
        X_dummy=RV([(gamma(N/2+1/2)*(1+((x**2)/N))**(-(N/2)-1/2))/
                    (sqrt(N*pi)*gamma(N/2))],[-oo,oo])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class UniformRV(RV):
    def __init__(self,a=Symbol('a'),b=Symbol('b')):
        if a.__class__.__name__!='Symbol':
            if b.__class__.__name__!='Symbol':
                if a>=b:
                    err_string='the parameters must be in ascending order'
                    raise RVError(err_string)
        if a in [-oo,oo] or b in [-oo,oo]:
            err_string='all parameters must be finite'
            raise RVError(err_string)
        X_dummy=RV(simplify((b-a)**(-1)),[a,b])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype
        self.parameter=[a,b]

    def variate(self,n=1,s='sim'):
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
    def __init__(self,theta=Symbol('theta',positive=True),
                 kappa=Symbol('kappa',positive=True)):
        if theta.__class__.__name__!='Symbol':
            if kappa.__class__.__name__!='Symbol':
                if theta<=0 or kappa<=0:
                    err_string='both parameters must be positive'
                    raise RVError(err_string)
        if theta in [-oo,oo] or kappa in [-oo,oo]:
            err_string='both parameters must be finite'
            raise RVError(err_string)
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
    def __init__(self,N=Symbol('N',positive=True,integer=True),
                 p=Symbol('p',positive=True)):
        if N.__class__.__name__!='Symbol':
            if N<=0:
                if type(N)!=int:
                    err_string='N must be a positive integer'
                    raise RVError(err_string)
        if p.__class__.__name__!='Symbol':
            if p<=0 or p>=1:
                err_string='p must be between 0 and 1'
                raise RVError(err_string)
        X_dummy=RV([(factorial(N)*p**(x)*(1-p)**(N-x))/
                    (factorial(N-x)*factorial(x))],[0,N],
                   ['Discrete','pdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class GeometricRV(RV):
    def __init__(self,p=Symbol('p',positive=True)):
        if p.__class__.__name__!='Symbol':
            if p<=0 or p>=1:
                err_string='p must be between 0 and 1'
                raise RVError(err_string)
        X_dummy=RV([p*(1-p)**(x-1)],[1,oo],['Discrete','pdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

class PoissonRV(RV):
    def __init__(self,theta=Symbol('theta',positive=True)):
        if theta.__class__.__name__!='Symbol':
            if theta<=0:
                err_string='theta must be positive'
                raise RVError(err_string)
        if theta in [-oo,oo]:
            err_string='theta must be finite'
            raise RVError(err_string)
        X_dummy=RV([(theta**(x)*exp(-theta))/factorial(x)],
                   [0,oo],['Discrete','pdf'])
        self.func=X_dummy.func
        self.support=X_dummy.support
        self.ftype=X_dummy.ftype

