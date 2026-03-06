"""Discrete random variable distributions."""

from sympy import Rational, Symbol, exp, factorial, ln, oo, symbols

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class BenfordRV(RV):
    """
    Procedure Name: BenfordRV
    Purpose: Creates an instance of the Benford distribution
    Arguments:  1. None
    Output:     1. A Benford random variable
    """

    def __init__(self):
        X_dummy = RV([(ln((1 / x) + 1)) / (ln(10))], [1, 9], ["Discrete", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


class BinomialRV(RV):
    """
    Procedure Name: BinomialRV
    Purpose: Creates an instance of the binomial distribution
    Arguments:  1. N: a positive integer parameter
                2. p: a positive parameter between 0 and 1
    Output:     1. A binomial random variable
    """

    def __init__(self, N=Symbol("N", positive=True, integer=True), p=Symbol("p", positive=True)):
        if not isinstance(N, Symbol):
            if N <= 0:
                if not isinstance(N, int):
                    err_string = "N must be a positive integer"
                    raise RVError(err_string)
        if not isinstance(p, Symbol):
            if p <= 0 or p >= 1:
                err_string = "p must be between 0 and 1"
                raise RVError(err_string)
        X_dummy = RV(
            [(factorial(N) * p ** (x) * (1 - p) ** (N - x)) / (factorial(N - x) * factorial(x))],
            [0, N],
            ["Discrete", "pdf"],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


class BernoulliRV(BinomialRV):
    """
    Procedure Name: BernoulliRV
    Purpose: Creates an instance of the bernoulli distribution
    Arguments:  1. p: a positive parameter between 0 and 1
    Output:     1. A bernoulli random variable
    """

    def __init__(self, p=Symbol("p", positive=True)):
        X_dummy = BinomialRV(1, p)
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


class GeometricRV(RV):
    """
    Procedure Name: GeometricRV
    Purpose: Creates an instance of the geometric distribution
    Arguments:  1. p: a positive parameter between 0 and 1
    Output:     1. A geometric random variable
    """

    def __init__(self, p=Symbol("p", positive=True)):
        if not isinstance(p, Symbol):
            if p <= 0 or p >= 1:
                err_string = "p must be between 0 and 1"
                raise RVError(err_string)
        X_dummy = RV([p * (1 - p) ** (x - 1)], [1, oo], ["Discrete", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


class PoissonRV(RV):
    """
    Procedure Name: PoissonRV
    Purpose: Creates an instance of the poisson distribution
    Arguments:  1. theta: a strictly positive parameter
    Output:     1. A poisson random variable
    """

    def __init__(self, theta=Symbol("theta", positive=True)):
        if not isinstance(theta, Symbol):
            if theta <= 0:
                err_string = "theta must be positive"
                raise RVError(err_string)
        if theta in [-oo, oo]:
            err_string = "theta must be finite"
            raise RVError(err_string)
        X_dummy = RV([(theta ** (x) * exp(-theta)) / factorial(x)], [0, oo], ["Discrete", "pdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


class UniformDiscreteRV(RV):
    """
    Procedure Name: UniformDiscreteRV
    Purpose: Creates an instance of the uniform discrete distribution
    Arguments:  1. a: the beggining point of the interval
                2. b: the end point of the interval (note: b>a)
    Output:     1. A uniform discrete random variable
    """

    def __init__(self, a=Symbol("a"), b=Symbol("b"), k=1):
        if b <= a:
            err_string = "b is only valid if b > a"
            raise RVError(err_string)
        if (b - a) % k != 0:
            err_string = "(b-a) must be divisble by k"
            raise RVError(err_string)
        n = int((b - a) / k)
        X_dummy = RV(
            [Rational(1, n + 1) for i in range(1, n + 2)],
            [a + i * k for i in range(n + 1)],
            ["discrete", "pdf"],
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}




__all__ = [
    "BenfordRV",
    "BinomialRV",
    "BernoulliRV",
    "GeometricRV",
    "PoissonRV",
    "UniformDiscreteRV",
]
