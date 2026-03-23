from sympy import Rational, Symbol, init_session, oo
import matplotlib.pyplot as plt

init_session()
from applpy import (
    MarkovChain,
    Maximum,
    MaximumIID,
    Minimum,
    OrderStat,
    RV,
    cdf,
    convolution,
    convolution_iid,
    plot_dist,
    product,
    product_iid,
    x,
)
from applpy.bayes import CredibleSet, Posterior
from applpy.distributions.continuous import (
    BetaRV,
    ExponentialRV,
    NormalRV,
    TriangularRV,
    UniformRV,
    WeibullRV,
)
from applpy.distributions.discrete import BinomialRV
from applpy.moments import (
    coef_of_var,
    expected_value,
    kurtosis,
    mean,
    mgf,
    skewness,
    variance,
)
from applpy.queue_dist import Queue
from applpy.rv import bootstrap_rv
from applpy.stats import KSTest, MLE, MOM
from applpy.transform import convert as Convert
from applpy.transform import mixture as Mixture
from applpy.transform import transform as Transform
from applpy.transform import truncate as Truncate

# Bayes
theta = Symbol("theta")
X = BinomialRV(12, theta)
Y = TriangularRV(0, Rational(1, 2), 1)
data = [5]
P = Posterior(X, Y, data, theta)
CredibleSet(P, 0.05)

theta = Symbol("theta")
X = BinomialRV(12, theta)
Y = TriangularRV(Rational(0), Rational(1, 2), Rational(1))
data = [5]
P = Posterior(X, Y, data, theta)
P.display()

# Bootstrapping

X = NormalRV(2, 2)
data = X.variate(n=10)
Xstar = bootstrap_rv(data)
Xstar.display()

# Converting RV Type

X = NormalRV(2, 2)
cdf(X, cache=True)
T = TriangularRV(2, 4, 6)
hf(T).display()
cdf(T, Rational(5, 2))

# Expected Values
X = WeibullRV()
coef_of_var(X)
X = NormalRV(2, 2)
expected_value(X, x**2)
X = BetaRV(2, 2)
kurtosis(X)
X = ExponentialRV()
mean(X)
X = UniformRV()
mgf(X)
X = BetaRV(2, 3)
skewness(X)
X = UniformRV()
X = WeibullRV()
variance(X)

# Markov Chains
Y = MarkovChain(P=[[0.97, 0.03], [0.04, 0.96]], init=[0.7, 0.3], states=["red", "blue"])
Y.display()

# Minimum and Maximum
X = TriangularRV(Rational(2), Rational(4), Rational(6))
Y = TriangularRV(Rational(3), Rational(5), Rational(7))
Z = TriangularRV(Rational(4), Rational(5), Rational(9))
A = Maximum(X, Y)
A.display()
B = Minimum(X, Y, Z)
B.display()
C = MaximumIID(X, 3)
C.display()

# Mixtures
X1 = TriangularRV(Rational(2), Rational(4), Rational(6))
X2 = TriangularRV(Rational(3), Rational(5), Rational(7))
X3 = TriangularRV(Rational(1), Rational(5), Rational(9))
mixtures = [Rational(1, 4), Rational(1, 4), Rational(1, 2)]
Y = Mixture(mixtures, [X1, X2, X3])
Y.display()

# Order Statistics
X = TriangularRV(Rational(1), Rational(5), Rational(9))
Y = OrderStat(X, 5, 2)
Y.display()

# Plot Dist
X = TriangularRV(2, 4, 6)
Y = TriangularRV(3, 5, 7)
Z = Mixture([0.4, 0.6], [X, Y])
plot_dist(Z)
plot_dist(Z, color="g")

X = ExponentialRV(1 / 3)
Y = ExponentialRV(1 / 2)
plot_dist(X, color="red")
plot_dist(Y, color="blue")

Xstar = bootstrap_rv(X.variate(n=100))
plot_dist(cdf(X))
plot_dist(cdf(Xstar))
plt.title("Comparison of Exponential CDF and Bootstrapped EDF")

# Queue
X = ExponentialRV(Rational(1, 3))
Y = ExponentialRV(Rational(1, 2))
Q = Queue(X, Y, n=3, k=2, s=3)
Q.display()

# RV Algebra
X = ExponentialRV(Rational(1, 2))
Y = ExponentialRV(Rational(1, 3))

Z = X + Y
Z.display()

Z = X - 4
Z.display()

Z = -Y
Z.display()

U = UniformRV(Rational(1), Rational(3))
U2 = UniformRV(Rational(4), Rational(5))
Z = U * U2
Z.display()

Z = U / 4
Z.display()

Z = U**3
Z.display()

U = UniformRV(Rational(1), Rational(2))
U2 = UniformRV(Rational(3), Rational(4))
Z = convolution(U, U2)
Z.display()

Z = convolution_iid(U, 3)
Z.display()

X = UniformRV(Rational(2), Rational(4))
Y = UniformRV(Rational(3), Rational(5))
Z = product(X, Y)
Z.display()

Z = product_iid(X, 3)
Z.display()

# Random Variable Methods

X = TriangularRV(2, 4, 7)
X.verifyPDF()
Y = RV([x**2], [0, 10], ["continuous", "pdf"])
Y.verifyPDF()

X = BinomialRV(5, 0.25)
X.display()
Convert(X)

T = TriangularRV(1, 4, 6)
T.display()
T.latex()

U = UniformRV(Rational(1), Rational(2))
Z = convolution_iid(U, 3)
Z.display()
Z.simplify()

# Saving Random Variables
X = ExponentialRV(Rational(1, 3))
Y = ExponentialRV(Rational(1, 4))
Z = X + Y
Z.save("conv.rv")

# Loading serialized RVs is not demonstrated because a top-level loader API
# is not currently exposed in this package version.

# Statistics

X = NormalRV(2, 2)
data = X.variate(n=30)
KSTest(X, data)

mu = Symbol("mu")
sigma = Symbol("sigma", positive=True)
X = NormalRV(mu, sigma)
data = NormalRV(0, 2).variate(n=40)
MLE(X, data, [mu, sigma])

theta = Symbol("theta", positive=True)
kappa = Symbol("kappa", positive=True)
W = WeibullRV(theta, kappa)
data = WeibullRV(2, 4).variate(n=40)
MOM(W, data, [theta, kappa], guess=[2.2, 3.8], numeric=True)

# Transform

X = TriangularRV(Rational(2), Rational(4), Rational(5))
gX = [[x**2, x**2], [-oo, 0, oo]]
Y = Transform(X, gX)
Y.display()

# Truncation

X = BetaRV(2, 2)
Y = Truncate(X, [Rational(1, 4), Rational(3, 4)])
Y.display()

# Variate Generation

X = TriangularRV(2, 4, 10)
X.variate(n=5)
X.variate(n=5, method="inverse")
X.variate(s=0.5, method="inverse")
