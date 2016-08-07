from sympy import *
init_session()
from applpy import *

# Bayes
theta = Symbol('theta')
X = BinomialRV(12,theta)
Y = TriangularRV(0, Rational(1,2), 1)
data = [5]
P = Posterior(X,Y,data,theta)
CredibleSet(P,.05)


theta=Symbol('theta')
X=BinomialRV(12,theta)
Y=TriangularRV(Rational(0),Rational(1,2),Rational(1))
data=[5]
P=Posterior(X,Y,data,theta)
P.display()

# Bootstrapping

X=NormalRV(2,2)
data=X.variate(n=10)
Xstar=BootstrapRV(data)
Xstar.display()