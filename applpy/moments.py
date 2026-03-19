"""
Moment and summary statistics utilities for APPLPy random variables.
"""

from applpy_rust import FastRV
from sympy import Sum, exp, integrate, log, simplify, sqrt, summation, symbols

from applpy.rv import BootstrapRV, PDF

x, y, z, t = symbols("x y z t")


def CoefOfVar(random_variable, cache=False):
    """
    Procedure Name: CoefOfVar
    Purpose: Compute the coefficient of variation of a random variable
    Arguments:  1. random_variable: A random variable
    Output:     1. The coefficient of variation
    """
    if isinstance(random_variable, list):
        Xstar = BootstrapRV(random_variable)
        return CoefOfVar(Xstar)

    if random_variable.cache is not None and "cov" in random_variable.cache:
        return random_variable.cache["cov"]

    # Compute the coefficient of varation
    expect = Mean(random_variable)
    sig = Variance(random_variable)
    cov = (sqrt(sig)) / expect
    cov = simplify(cov)
    if cache:
        random_variable.add_to_cache("cov", cov)
    return cov


def ExpectedValue(random_variable, gX=x):
    """
    Procedure Name: ExpectedValue
    Purpose: Computes the expected value of X
    Arguments:  1. random_variable: A random variable
                2. gX: A transformation of x
    Output:     1. E(gX)
    """
    if isinstance(random_variable, list):
        Xstar = BootstrapRV(random_variable)
        return ExpectedValue(Xstar, gX)

    fx = PDF(random_variable)

    if fx.is_continuous():
        Expect = 0
        for i in range(len(fx.func)):
            Expect += integrate(gX * fx.func[i], (x, fx.support[i], fx.support[i + 1]))
        return simplify(Expect)

    if fx.is_discrete_functional():
        Expect = 0
        for i in range(len(fx.func)):
            Expect += summation(gX * fx.func[i], (x, fx.support[i], fx.support[i + 1]))
        return simplify(Expect)

    if fx.is_discrete():
        fx_support = [gX.subs(x, value) for value in fx.support]
        fx_trans = FastRV(
            function=fx.func,
            support=fx_support,
            functional_form=fx.functional_form,
            domain_type=fx.domain_type,
        )
        return fx_trans.mean()


def Entropy(random_variable, cache=False):
    """
    Procedure Name: Entropy
    Purpose: Compute the entory of a random variable
    Arguments:  1. random_variable: A random variable
    Output:     1. The entropy of a random variable
    """
    if isinstance(random_variable, list):
        Xstar = BootstrapRV(random_variable)
        return Entropy(Xstar)

    if random_variable.cache is not None and "entropy" in random_variable.cache:
        return random_variable.cache["entropy"]

    entropy = ExpectedValue(random_variable, log(x, 2))
    entropy = simplify(entropy)
    if cache:
        random_variable.add_to_cache("entropy", entropy)
    return simplify(entropy)


def Kurtosis(random_variable, cache=False):
    """
    Procedure Name: Kurtosis
    Purpose: Compute the Kurtosis of a random variable
    Arguments:  1. random_variable: A random variable
    Output:     1. The kurtosis of a random variable
    """
    if isinstance(random_variable, list):
        Xstar = BootstrapRV(random_variable)
        return Kurtosis(Xstar)

    if random_variable.cache is not None and "kurtosis" in random_variable.cache:
        return random_variable.cache["kurtosis"]

    # Compute the kurtosis
    expect = Mean(random_variable)
    sig = sqrt(Variance(random_variable))
    Term1 = ExpectedValue(random_variable, x**4)
    Term2 = 4 * expect * ExpectedValue(random_variable, x**3)
    Term3 = 6 * (expect**2) * ExpectedValue(random_variable, x**2)
    Term4 = 3 * expect**4
    kurt = (Term1 - Term2 + Term3 - Term4) / (sig**4)
    kurt = simplify(kurt)

    if cache:
        random_variable.add_to_cache("kurtosis", kurt)
    return simplify(kurt)


def Mean(random_variable, cache=False):
    """
    Procedure Name: Mean
    Purpose: Compute the mean of a random variable
    Arguments: 1. random_variable: A random variable
    Output:    1. The mean of a random variable
    """
    if isinstance(random_variable, list):
        Xstar = BootstrapRV(random_variable)
        return Mean(Xstar)

    if random_variable.cache is not None and "mean" in random_variable.cache:
        return random_variable.cache["mean"]

    X_dummy = PDF(random_variable)

    if X_dummy.is_continuous():
        # Create list of x*f(x)
        meanfunc = []
        for i in range(len(X_dummy.func)):
            meanfunc.append(x * X_dummy.func[i])
        # Integrate to find the mean
        meanval = 0
        for i in range(len(X_dummy.func)):
            val = integrate(meanfunc[i], (x, X_dummy.support[i], X_dummy.support[i + 1]))
            meanval += val
        meanval = simplify(meanval)
        if cache:
            random_variable.add_to_cache("mean", meanval)
        return simplify(meanval)

    if X_dummy.is_discrete_functional():
        # Create list of x*f(x)
        meanfunc = []
        for i in range(len(X_dummy.func)):
            meanfunc.append(x * X_dummy.func[i])
        # Sum to find the mean
        meanval = 0
        for i in range(len(X_dummy.func)):
            val = Sum(meanfunc[i], (x, X_dummy.support[i], X_dummy.support[i + 1])).doit()
            meanval += val
        meanval = simplify(meanval)
        if cache:
            random_variable.add_to_cache("mean", meanval)
        return simplify(meanval)

    if X_dummy.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type="discrete",
        )
        return fast_rv.mean()


def MGF(random_variable, cache=False):
    """
    Procedure Name: MGF
    Purpose: Compute the moment generating function of a random variable
    Arguments:  1. random_variable: A random variable
    Output:     1. The moment generating function
    """
    if random_variable.cache is not None and "mgf" in random_variable.cache:
        return random_variable.cache["mgf"]

    mgf = ExpectedValue(random_variable, exp(t * x))
    mgf = simplify(mgf)
    if cache:
        random_variable.add_to_cache("mgf", mgf)
    return mgf


def Skewness(random_variable, cache=False):
    """
    Procedure Name: Skewness
    Purpose: Compute the skewness of a random variable
    Arguments:  1. random_variable: A random variable
    Output:     1. The skewness of the random variable
    """
    from .rv import BootstrapRV

    if isinstance(random_variable, list):
        Xstar = BootstrapRV(random_variable)
        return Skewness(Xstar)

    if random_variable.cache is not None and "skewness" in random_variable.cache:
        return random_variable.cache["skewness"]

    # Compute the skewness
    expect = Mean(random_variable)
    sig = sqrt(Variance(random_variable))
    Term1 = ExpectedValue(random_variable, x**3)
    Term2 = 3 * expect * ExpectedValue(random_variable, x**2)
    Term3 = 2 * expect**3
    skew = (Term1 - Term2 + Term3) / (sig**3)
    skew = simplify(skew)
    if cache:
        random_variable.add_to_cache("skewness", skew)
    return simplify(skew)


def Variance(random_variable, cache=False):
    """
    Procedure Name: Variance
    Purpose: Compute the variance of a random variable
    Arguments: 1. random_variable: A random variable
    Output:    1. The variance of a random variable
    """
    from .rv import BootstrapRV, PDF

    if isinstance(random_variable, list):
        Xstar = BootstrapRV(random_variable)
        return Variance(Xstar)

    if random_variable.cache is not None and "variance" in random_variable.cache:
        return random_variable.cache["variance"]

    X_dummy = PDF(random_variable)

    if X_dummy.is_continuous():
        # Find the mean of the random variable
        EX = Mean(X_dummy)
        # Find E(X^2)
        # Create list of (x**2)*f(x)
        varfunc = []
        for i in range(len(X_dummy.func)):
            varfunc.append((x**2) * X_dummy.func[i])
        # Integrate to find E(X^2)
        exxval = 0
        for i in range(len(X_dummy.func)):
            val = integrate(varfunc[i], (x, X_dummy.support[i], X_dummy.support[i + 1]))
            exxval += val
        # Find Var(X)=E(X^2)-E(X)^2
        var = exxval - (EX**2)
        var = simplify(var)
        if cache:
            random_variable.add_to_cache("variance", var)
        return simplify(var)

    if X_dummy.is_discrete_functional():
        EX = Mean(X_dummy)
        # Find E(X^2)
        # Create list of (x**2)*f(x)
        varfunc = []
        for i in range(len(X_dummy.func)):
            varfunc.append((x**2) * X_dummy.func[i])
        # Sum to find E(X^2)
        exxval = 0
        for i in range(len(X_dummy.func)):
            val = summation(varfunc[i], (x, X_dummy.support[i], X_dummy.support[i + 1]))
            exxval += val
        # Find Var(X)=E(X^2)-E(X)^2
        var = exxval - (EX**2)
        var = simplify(var)
        if cache:
            random_variable.add_to_cache("variance", var)
        return simplify(var)

    if X_dummy.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type=random_variable.domain_type,
        )
        return fast_rv.variance()
