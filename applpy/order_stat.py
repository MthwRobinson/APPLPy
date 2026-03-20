"""Order statistics and extrema operations for random variables."""

from sympy import Symbol, factorial, integrate, oo, simplify

from . import rust_bindings
from .rv import Convert, RV, RVError, cdf, pdf, sf, x


def maximum_iid(random_variable, n=Symbol("n")):
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
        return order_stat(random_variable, n, n)

    base_rv = random_variable
    current_max_rv = base_rv
    for _ in range(n - 1):
        current_max_rv = maximum(current_max_rv, base_rv)
    return pdf(current_max_rv)


def minimum_iid(random_variable, n):
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
        return order_stat(random_variable, 1, n)

    base_rv = random_variable
    current_min_rv = base_rv
    for _ in range(n - 1):
        current_min_rv = minimum(current_min_rv, base_rv)
    return pdf(current_min_rv)


def order_stat(random_variable, n, r, replace="w"):
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
        pdf_rv = pdf(random_variable)
        cdf_rv = cdf(random_variable)
        sf_rv = sf(random_variable)
        normalization_const = (factorial(n)) / (factorial(r - 1) * factorial(n - r))
        order_stat_func = []
        for i in range(len(random_variable.func)):
            pdf_segment = pdf_rv.func[i]
            cdf_segment = cdf_rv.func[i]
            sf_segment = sf_rv.func[i]
            order_stat_segment = (
                normalization_const
                * (cdf_segment ** (r - 1))
                * (sf_segment ** (n - r))
                * pdf_segment
            )
            order_stat_func.append(simplify(order_stat_segment))
        return RV(order_stat_func, random_variable.support, ["continuous", "pdf"])

    if random_variable.is_discrete_functional():
        if (-oo not in random_variable.support) and (oo not in random_variable.support):
            converted_rv = Convert(random_variable)
            return order_stat(converted_rv, n, r, replace)
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


def range_stat(random_variable, n, replace="w"):
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
    pdf_rv = pdf(random_variable)
    integration_variable = Symbol("z")
    if pdf_rv.is_continuous():
        if replace == "wo":
            err_string = "OrderStat without replacement not implemented "
            err_string += "for continuous random variables"
            raise RVError(err_string)
        cdf_rv = cdf(random_variable)
        num_segments = len(cdf_rv.func)
        range_pdf_segments = []
        for i in range(num_segments):
            range_pdf_segment = integrate(
                n
                * (n - 1)
                * (
                    cdf_rv.func[i].subs(x, integration_variable)
                    - cdf_rv.func[i].subs(x, integration_variable - x)
                )
                ** (n - 2)
                * pdf_rv.func[i].subs(x, integration_variable - x)
                * pdf_rv.func[i].subs(x, integration_variable),
                (integration_variable, x, pdf_rv.support[i + 1]),
            )
            range_pdf_segments.append(range_pdf_segment)
        range_rv = RV(
            range_pdf_segments,
            pdf_rv.support,
            functional_form=pdf_rv.functional_form,
            domain_type=pdf_rv.domain_type,
        )
        return range_rv
    if pdf_rv.is_discrete_functional():
        if (-oo not in pdf_rv.support) and (oo not in pdf_rv.support):
            converted_rv = Convert(random_variable)
            return range_stat(converted_rv, n, replace)
    if pdf_rv.is_discrete():
        fast_rv = rust_bindings.discrete_range_stat(random_variable, n, replace)
        return RV(
            func=fast_rv.function,
            support=fast_rv.support,
            functional_form=fast_rv.functional_form,
            domain_type=fast_rv.domain_type,
        )


def maximum_rv(random_variable_1, random_variable_2):
    """
    Procedure Name: MaximumRV
    Purpose: Compute cdf of the maximum of random_variable_1 and random_variable_2
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The cdf of the maximum distribution
    """

    if random_variable_1.domain_type != random_variable_2.domain_type:
        raise RVError("The RVs must both be discrete or continuous")

    if random_variable_1.is_continuous():
        if random_variable_1.support == [0, oo] and random_variable_2.support == [0, oo]:
            cdf_dummy1 = cdf(random_variable_1)
            cdf_dummy2 = cdf(random_variable_2)
            cdf1 = cdf_dummy1.func[0]
            cdf2 = cdf_dummy2.func[0]
            maxfunc = cdf1 * cdf2
            return pdf(RV(simplify(maxfunc), [0, oo], ["continuous", "cdf"]))

        Fx = cdf(random_variable_1)
        Fy = cdf(random_variable_2)
        max_supp = []
        for i in range(len(Fx.support)):
            if Fx.support[i] not in max_supp:
                max_supp.append(Fx.support[i])
        for i in range(len(Fy.support)):
            if Fy.support[i] not in max_supp:
                max_supp.append(Fy.support[i])
        max_supp.sort()

        lowval = max(min(Fx.support), min(Fy.support))
        max_supp2 = []
        for i in range(len(max_supp)):
            if max_supp[i] >= lowval:
                max_supp2.append(max_supp[i])

        max_func = []
        for i in range(len(max_supp2) - 1):
            value = max_supp2[i]
            currFx = 1
            for j in range(len(Fx.func)):
                if value >= Fx.support[j] and value < Fx.support[j + 1]:
                    currFx = Fx.func[j]
                    break
            currFy = 1
            for j in range(len(Fy.func)):
                if value >= Fy.support[j] and value < Fy.support[j + 1]:
                    currFy = Fy.func[j]
            Fmax = currFx * currFy
            max_func.append(simplify(Fmax))
        return pdf(RV(max_func, max_supp2, ["continuous", "cdf"]))

    if random_variable_1.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Maximum does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_1 = Convert(random_variable_1)
    if random_variable_2.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Maximum does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_2 = Convert(random_variable_2)

    if random_variable_1.is_discrete():
        fast_rv = rust_bindings.discrete_maximum(random_variable_1, random_variable_2)
        return RV(
            func=fast_rv.function,
            support=fast_rv.support,
            functional_form=fast_rv.functional_form,
            domain_type=fast_rv.domain_type,
        )


def minimum_rv(random_variable_1, random_variable_2):
    """
    Procedure Name: MinimumRV
    Purpose: Compute the distribution of the minimum of random_variable_1 and random_variable_2
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The minimum of the two random variables
    """

    if random_variable_1.domain_type != random_variable_2.domain_type:
        raise RVError("The RVs must both be discrete or continuous")

    if random_variable_1.is_continuous():
        if random_variable_1.support == [0, oo] and random_variable_2.support == [0, oo]:
            sf_dummy1 = sf(random_variable_1)
            sf_dummy2 = sf(random_variable_2)
            sf1 = sf_dummy1.func[0]
            sf2 = sf_dummy2.func[0]
            minfunc = 1 - (sf1 * sf2)
            return pdf(RV(simplify(minfunc), [0, oo], ["continuous", "cdf"]))

        Fx = cdf(random_variable_1)
        Fy = cdf(random_variable_2)
        min_supp = []
        for i in range(len(Fx.support)):
            if Fx.support[i] not in min_supp:
                min_supp.append(Fx.support[i])
        for i in range(len(Fy.support)):
            if Fy.support[i] not in min_supp:
                min_supp.append(Fy.support[i])
        min_supp.sort()

        highval = min(max(Fx.support), max(Fy.support))
        min_supp2 = []
        for i in range(len(min_supp)):
            if min_supp[i] <= highval:
                min_supp2.append(min_supp[i])

        min_func = []
        for i in range(len(min_supp2) - 1):
            value = min_supp2[i]
            currFx = 0
            for j in range(len(Fx.func)):
                if value >= Fx.support[j] and value <= Fx.support[j + 1]:
                    currFx = Fx.func[j]
                    break
            currFy = 0
            for j in range(len(Fy.func)):
                if value >= Fy.support[j] and value <= Fy.support[j + 1]:
                    currFy = Fy.func[j]
            Fmin = 1 - ((1 - currFx) * (1 - currFy))
            min_func.append(simplify(Fmin))

        return pdf(RV(min_func, min_supp2, ["continuous", "cdf"]))

    if random_variable_1.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Minimum does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_1 = Convert(random_variable_1)
    if random_variable_2.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Minimum does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_2 = Convert(random_variable_2)

    if random_variable_1.is_discrete():
        fast_rv = rust_bindings.discrete_minimum(random_variable_1, random_variable_2)
        return RV(
            func=fast_rv.function,
            support=fast_rv.support,
            functional_form=fast_rv.functional_form,
            domain_type=fast_rv.domain_type,
        )


def maximum(*argv):
    """
    Procedure Name: Maximum
    Purpose: Compute the maximum of a list of random variables
    Arugments:  1. *argv: a series of random variables
    Output:     1. The maximum distribution
    """
    argument_index = 0
    for rv in argv:
        if argument_index == 0:
            running_max_rv = rv
        else:
            running_max_rv = maximum_rv(running_max_rv, rv)
        argument_index += 1
    return running_max_rv


def minimum(*argv):
    """
    Procedure Name: Minimum
    Purpose: Compute the minimum of a list of random variables
    Arugments:  1. *argv: a series of random variables
    Output:     1. The minimum distribution
    """
    argument_index = 0
    for rv in argv:
        if argument_index == 0:
            running_min_rv = rv
        else:
            running_min_rv = minimum_rv(running_min_rv, rv)
        argument_index += 1
    return running_min_rv


# Backward-compatible aliases
MaximumIID = maximum_iid
MinimumIID = minimum_iid
OrderStat = order_stat
RangeStat = range_stat
MaximumRV = maximum_rv
MinimumRV = minimum_rv
Maximum = maximum
Minimum = minimum
