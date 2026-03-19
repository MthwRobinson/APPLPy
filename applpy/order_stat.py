"""Order statistics and extrema operations for random variables."""

from sympy import Symbol, binomial, factorial, integrate, oo, simplify

from . import rust_bindings
from .rv import Convert, RV, RVError, MaximumRV, MinimumRV, cdf, pdf, sf, x


def MaximumIID(random_variable, n=Symbol("n")):
    """
    Procedure Name: MaximumIID
    Purpose: Compute the maximum of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The maximum of n iid random variables
    """
    if not isinstance(n, int):
        if not isinstance(n, Symbol):
            raise RVError("The second argument must be an integer")

    if isinstance(n, Symbol):
        return OrderStat(random_variable, n, n)

    x_dummy = random_variable
    x_final = x_dummy
    for _ in range(n - 1):
        x_final = Maximum(x_final, x_dummy)
    return pdf(x_final)


def MinimumIID(random_variable, n):
    """
    Procedure Name: MinimumIID
    Purpose: Compute the minimum of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The minimum of n iid random variables
    """
    if not isinstance(n, int):
        if not isinstance(n, Symbol):
            raise RVError("The second argument must be an integer")

    if isinstance(n, Symbol):
        return OrderStat(random_variable, 1, n)

    x_dummy = random_variable
    x_final = x_dummy
    for _ in range(n - 1):
        x_final = Minimum(x_final, x_dummy)
    return pdf(x_final)


def OrderStat(random_variable, n, r, replace="w"):
    """
    Procedure Name: OrderStat
    Purpose: Compute the distribution of the rth order statistic
                from a sample puplation of n
    Arguments:  1. random_variable: A random variable
                2. n: The number of items randomly drawn from the rv
                3. r: The index of the order statistic
    Output:     1. The desired r out of n OrderStatistic
    """
    if not isinstance(r, Symbol) and not isinstance(n, Symbol):
        if r > n:
            raise RVError("The index cannot be greater than the sample size")
    if replace not in ["w", "wo"]:
        raise RVError("Replace must be w or wo")

    if random_variable.is_continuous():
        if replace == "wo":
            err_string = "OrderStat without replacement not implemented "
            err_string += "for continuous random variables"
            raise RVError(err_string)
        pdf_dummy = pdf(random_variable)
        cdf_dummy = cdf(random_variable)
        sf_dummy = sf(random_variable)
        const = (factorial(n)) / (factorial(r - 1) * factorial(n - r))
        ordstat_func = []
        for i in range(len(random_variable.func)):
            fx = pdf_dummy.func[i]
            Fx = cdf_dummy.func[i]
            Sx = sf_dummy.func[i]
            ordfunc = const * (Fx ** (r - 1)) * (Sx ** (n - r)) * fx
            ordstat_func.append(simplify(ordfunc))
        return RV(ordstat_func, random_variable.support, ["continuous", "pdf"])

    if random_variable.is_discrete_functional():
        if (-oo not in random_variable.support) and (oo not in random_variable.support):
            x_dummy = Convert(random_variable)
            return OrderStat(x_dummy, n, r, replace)
        err_string = "OrderStat is not currently implemented for "
        err_string += "discrete RVs with infinite support"
        raise RVError(err_string)

    if random_variable.is_discrete():
        fast_rv = rust_bindings.discrete_order_stat(random_variable, n, r, replace)
        return RV(
            func=fast_rv.function,
            support=fast_rv.support,
            functional_form=fast_rv.functional_form,
            domain_type=fast_rv.domain_type,
        )


def RangeStat(random_variable, n, replace="w"):
    """
    Procedure Name: RangeStat
    Purpose: Compute the distribution of the range of n iid rvs
    Arguments:  1. random_variable: A random variable
                2. n: an integer
                3. replace: indicates with or without replacment
    Output:     1. The dist of the range of n iid random variables
    """
    if n < 2:
        err_string = "Only one item sampled from the population"
        raise RVError(err_string)
    if replace not in ["w", "wo"]:
        raise RVError("Replace must be w or wo")
    fX = pdf(random_variable)
    z = Symbol("z")
    if fX.is_continuous():
        if replace == "wo":
            err_string = "OrderStat without replacement not implemented "
            err_string += "for continuous random variables"
            raise RVError(err_string)
        FX = cdf(random_variable)
        nsegs = len(FX.func)
        fXRange = []
        for i in range(nsegs):
            ffX = integrate(
                n
                * (n - 1)
                * (FX.func[i].subs(x, z) - FX.func[i].subs(x, z - x)) ** (n - 2)
                * fX.func[i].subs(x, z - x)
                * fX.func[i].subs(x, z),
                (z, x, fX.support[i + 1]),
            )
            fXRange.append(ffX)
        range_rv = RV(
            fXRange,
            fX.support,
            functional_form=fX.functional_form,
            domain_type=fX.domain_type,
        )
        return range_rv
    if fX.is_discrete_functional():
        if (-oo not in fX.support) and (oo not in fX.support):
            x_dummy = Convert(random_variable)
            return RangeStat(x_dummy, n, replace)
    if fX.is_discrete():
        fX = pdf(random_variable)
        FX = cdf(random_variable)
        N = len(fX.support)
        if N < 2:
            err_string = "The population only consists of 1 element"
            raise RVError(err_string)
        if replace == "w":
            s = fX.support
            p = fX.func
            k = 0
            sum(range(1, N + 1))
            rs = [0 for i in range(N**2)]
            rp = [0 for i in range(N**2)]
            for i in range(N):
                for j in range(N):
                    rs[k] = s[j] - s[i]
                    rp[k] = (
                        sum(p[i : j + 1]) ** n
                        - sum(p[i + 1 : j + 1]) ** n
                        - sum(p[i:j]) ** n
                        + sum(p[i + 1 : j]) ** n
                    )
                    k += 1
            sortedr = list(zip(*sorted(zip(rs, rp))))
            sortrs = list(sortedr[0])
            sortrp = list(sortedr[1])
            sortrs2 = []
            sortrp2 = []
            for i in range(len(sortrs)):
                if sortrs[i] not in sortrs2:
                    if sortrp[i] > 0:
                        sortrs2.append(sortrs[i])
                        sortrp2.append(sortrp[i])
                elif sortrs[i] in sortrs2:
                    idx = sortrs2.index(sortrs[i])
                    sortrp2[idx] += sortrp[i]
            return RV(sortrp2, sortrs2, ["discrete", "pdf"])
        if replace == "wo":
            err_string = "RangeStat current not implemented without "
            err_string += "replacement"
            raise RVError(err_string)
            if n == N:
                fXRange = [1]
                fXSupport = [N - 1]
            else:
                fXRange = [0 for i in range(N)]
                fXSupport = [value for value in fX.support]
                combo = [value for value in range(1, n + 1)]
                for _ in range(binomial(N, n)):
                    perm = [elem for elem in combo]
                    for _ in range(factorial(n)):
                        PermProb = fX.func[perm[0]]
                        cumsum = fX.func[perm[0]]
                        for m in range(1, n):
                            PermProb *= fX.func[perm[m]] / (1 - cumsum)
                            cumsum += fX.func[perm[m]]
                        hi_val = max(perm)
                        lo_val = min(perm)
                        range_value = hi_val - lo_val
                        for k in range(N - 1):
                            if range_value == k + 1:
                                fXRange[k] += PermProb
                        perm = rust_bindings.next_permutation(perm)
                    combo = rust_bindings.next_combination(combo, N)
                print(len(fXRange), len(fXSupport))
                return RV(
                    fXRange,
                    fXSupport,
                    functional_form=fX.functional_form,
                    domain_type=fX.domain_type,
                )


def Maximum(*argv):
    """
    Procedure Name: Maximum
    Purpose: Compute the maximum of a list of random variables
    Arugments:  1. *argv: a series of random variables
    Output:     1. The maximum distribution
    """
    i = 0
    for rv in argv:
        if i == 0:
            temp = rv
        else:
            temp = MaximumRV(temp, rv)
        i += 1
    return temp


def Minimum(*argv):
    """
    Procedure Name: Minimum
    Purpose: Compute the minimum of a list of random variables
    Arugments:  1. *argv: a series of random variables
    Output:     1. The minimum distribution
    """
    i = 0
    for rv in argv:
        if i == 0:
            temp = rv
        else:
            temp = MinimumRV(temp, rv)
        i += 1
    return temp
