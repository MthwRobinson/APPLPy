"""Order statistics and extrema operations for random variables."""

from sympy import Symbol, binomial, factorial, integrate, oo, simplify

from . import rust_bindings
from .rv import Convert, RV, RVError, MaximumRV, MinimumRV, cdf, pdf, sf, x


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
                normalization_const * (cdf_segment ** (r - 1)) * (sf_segment ** (n - r)) * pdf_segment
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
        pdf_rv = pdf(random_variable)
        cdf_rv = cdf(random_variable)
        support_size = len(pdf_rv.support)
        if support_size < 2:
            err_string = "The population only consists of 1 element"
            raise RVError(err_string)
        if replace == "w":
            support_values = pdf_rv.support
            probability_values = pdf_rv.func
            range_index = 0
            sum(range(1, support_size + 1))
            range_support_candidates = [0 for i in range(support_size**2)]
            range_probability_candidates = [0 for i in range(support_size**2)]
            for i in range(support_size):
                for j in range(support_size):
                    range_support_candidates[range_index] = support_values[j] - support_values[i]
                    range_probability_candidates[range_index] = (
                        sum(probability_values[i : j + 1]) ** n
                        - sum(probability_values[i + 1 : j + 1]) ** n
                        - sum(probability_values[i:j]) ** n
                        + sum(probability_values[i + 1 : j]) ** n
                    )
                    range_index += 1
            sorted_candidates = list(
                zip(*sorted(zip(range_support_candidates, range_probability_candidates)))
            )
            sorted_range_support = list(sorted_candidates[0])
            sorted_range_probabilities = list(sorted_candidates[1])
            merged_range_support = []
            merged_range_probabilities = []
            for i in range(len(sorted_range_support)):
                if sorted_range_support[i] not in merged_range_support:
                    if sorted_range_probabilities[i] > 0:
                        merged_range_support.append(sorted_range_support[i])
                        merged_range_probabilities.append(sorted_range_probabilities[i])
                elif sorted_range_support[i] in merged_range_support:
                    support_index = merged_range_support.index(sorted_range_support[i])
                    merged_range_probabilities[support_index] += sorted_range_probabilities[i]
            return RV(merged_range_probabilities, merged_range_support, ["discrete", "pdf"])
        if replace == "wo":
            err_string = "RangeStat current not implemented without "
            err_string += "replacement"
            raise RVError(err_string)
            if n == support_size:
                range_pdf_values = [1]
                range_support_values = [support_size - 1]
            else:
                range_pdf_values = [0 for i in range(support_size)]
                range_support_values = [value for value in pdf_rv.support]
                combination = [value for value in range(1, n + 1)]
                for _ in range(binomial(support_size, n)):
                    permutation = [elem for elem in combination]
                    for _ in range(factorial(n)):
                        permutation_probability = pdf_rv.func[permutation[0]]
                        cumulative_probability = pdf_rv.func[permutation[0]]
                        for m in range(1, n):
                            permutation_probability *= (
                                pdf_rv.func[permutation[m]] / (1 - cumulative_probability)
                            )
                            cumulative_probability += pdf_rv.func[permutation[m]]
                        hi_val = max(permutation)
                        lo_val = min(permutation)
                        range_value = hi_val - lo_val
                        for k in range(support_size - 1):
                            if range_value == k + 1:
                                range_pdf_values[k] += permutation_probability
                        permutation = rust_bindings.next_permutation(permutation)
                    combination = rust_bindings.next_combination(combination, support_size)
                print(len(range_pdf_values), len(range_support_values))
                return RV(
                    range_pdf_values,
                    range_support_values,
                    functional_form=pdf_rv.functional_form,
                    domain_type=pdf_rv.domain_type,
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
            running_max_rv = MaximumRV(running_max_rv, rv)
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
            running_min_rv = MinimumRV(running_min_rv, rv)
        argument_index += 1
    return running_min_rv


# Backward-compatible aliases
MaximumIID = maximum_iid
MinimumIID = minimum_iid
OrderStat = order_stat
RangeStat = range_stat
Maximum = maximum
Minimum = minimum
