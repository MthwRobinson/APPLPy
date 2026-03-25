"""
Transformation procedures extracted from `applpy.rv`.
"""

from sympy import Float, Symbol, diff, limit, oo, simplify, solve, sqrt as sympy_sqrt, zoo

from .conversion import cdf, pdf
from .rv import RV, RVError, t, x


try:
    import applpy_rust
except ImportError:
    raise ImportError(
        "applpy_rust extension is not built. "
        "Run `uv sync --extra rust` then "
        "`uv run --no-sync maturin develop -m rust/Cargo.toml`."
    )


def transform(random_variable, transform_spec):
    """
    Procedure Name: Transform
    Purpose: Compute the transformation of a random variable
                by a a function g(x)
    Arguments:  1. random_variable: A random variable
                2. gX: A transformation in list of two lists format
    Output:     1. The transformation of random_variable
    """

    # Check to make sure support of transform is in ascending order
    for i in range(len(transform_spec[1]) - 1):
        if transform_spec[1][i] > transform_spec[1][i + 1]:
            raise RVError("Transform support is not in ascending order")

    pdf_random_variable = pdf(random_variable)
    if random_variable.is_continuous():
        return _transform_continuous(pdf_random_variable, transform_spec)
    if random_variable.is_discrete_functional():
        return _transform_discrete_functional(random_variable, transform_spec)
    if random_variable.is_discrete():
        return _transform_discrete(pdf_random_variable, transform_spec)


def convert(random_variable, inc=1):
    """
    Procedure Name: Convert
    Purpose: Convert a discrete random variable from functional to
                explicit form
    Arguments:  1. random_variable: A functional discrete random variable
                2. inc: An increment value
    Output:     1. A discrete random variable in explicit form
    """
    # If the random variable is not in functional form, return
    #   an error
    if not random_variable.is_discrete_functional():
        raise RVError("The random variable must be discrete_functional")
    # If the rv has infinite support, return an error
    if (oo or -oo) in random_variable.support:
        raise RVError("Convert does not work for infinite support")
    # Create the support of explicit discrete rv
    i = random_variable.support[0]
    discrete_supp = []
    while i <= random_variable.support[1]:
        discrete_supp.append(i)
        i += inc
    # Create the function values for the explicit rv
    discrete_func = []
    for i in range(len(discrete_supp)):
        val = random_variable.func[0].subs(x, discrete_supp[i])
        discrete_func.append(val)
    # Return the random variable in discrete form
    return RV(discrete_func, discrete_supp, ["discrete", random_variable.functional_form])


def power(random_variable, n):
    """
    Procedure Name: Pow
    Purpose: Compute the transformation of a random variable by an exponent
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The transformation of the RV by x**n
    """
    if not isinstance(n, int):
        err_str = "n must be an integer"
        raise RVError(err_str)
    # If n is even, then g is a two-to-one transformation
    if n % 2 == 0:
        g = [[x**n, x**n], [-oo, 0, oo]]
    # If n is odd, the g is a one-to-one transformation
    elif n % 2 == 1:
        g = [[x**n], [-oo, oo]]
    return transform(random_variable, g)


def sqrt(random_variable):
    """
    Procedure Name: Sqrt
    Purpose: Computes the transformation of a random variable by sqrt(x)
    Arguments:  1. random_variable: A random variable
    Output:     1. The random variable transformed by sqrt(x)
    """
    for element in random_variable.support:
        if element < 0:
            err_string = "A negative value appears in the support of the"
            err_string += " random variable."
            raise RVError(err_string)

    u = [[sympy_sqrt(x)], [0, oo]]
    new_random_variable = transform(random_variable, u)
    return new_random_variable


def _transform_continuous(pdf_random_variable, transform_spec):
    # Adjust the transformation to include the support of the random
    #   variable
    transform_spec_copy = []
    for i in range(len(transform_spec)):
        transform_spec_copy.append(transform_spec[i])
    transform_support = []
    for i in range(len(transform_spec_copy[1])):
        transform_support.append(transform_spec_copy[1][i])
    # Add the support of the random variable into the support
    #   of the transformation
    for i in range(len(pdf_random_variable.support)):
        if pdf_random_variable.support[i] not in transform_support:
            transform_support.append(pdf_random_variable.support[i])
    transform_support.sort()
    # Find which segment of the transformation applies, and add it
    #   to the transformation list
    transform_functions = []
    for i in range(1, len(transform_support)):
        for j in range(len(transform_spec_copy[0])):
            if transform_support[i] >= transform_spec_copy[1][j]:
                if transform_support[i] <= transform_spec_copy[1][j + 1]:
                    transform_functions.append(transform_spec_copy[0][j])
                    break
    # Set the adjusted transformation as the active transform spec.
    active_transform = []
    active_transform.append(transform_functions)
    active_transform.append(transform_support)
    # If the support of the transformation does not match up with the
    #   support of the RV, adjust the support of the transformation

    # Traverse list to find elements that are not within the support
    #   of the rv
    for i in range(len(active_transform[1])):
        if active_transform[1][i] < pdf_random_variable.support[0]:
            active_transform[1][i] = pdf_random_variable.support[0]
        if (
            active_transform[1][i]
            > pdf_random_variable.support[len(pdf_random_variable.support) - 1]
        ):
            active_transform[1][i] = pdf_random_variable.support[
                len(pdf_random_variable.support) - 1
            ]
    # Delete segments of the transformation that will not be used
    function_indexes_to_remove = []
    support_indexes_to_remove = []
    for i in range(len(active_transform[0]) - 1):
        if active_transform[1][i] == active_transform[1][i + 1]:
            function_indexes_to_remove.append(i)
            support_indexes_to_remove.append(i + 1)
    for i in range(len(function_indexes_to_remove)):
        index = function_indexes_to_remove[i]
        del active_transform[0][index - i]
    for i in range(len(support_indexes_to_remove)):
        index = support_indexes_to_remove[i]
        del active_transform[1][index - i]
    # Create a list of mappings x->g(x)
    mapped_intervals = []
    for i in range(len(active_transform[0])):
        left_mapped_value = active_transform[0][i].subs(x, active_transform[1][i])
        if left_mapped_value == zoo:
            left_mapped_value = limit(active_transform[0][i], x, active_transform[1][i])
        right_mapped_value = active_transform[0][i].subs(x, active_transform[1][i + 1])
        if right_mapped_value == zoo:
            right_mapped_value = limit(active_transform[0][i + 1], x, active_transform[1][i + 1])
        mapped_intervals.append([left_mapped_value, right_mapped_value])
    # Create the support for the transformed random variable
    transformed_support = []
    for i in range(len(mapped_intervals)):
        for j in range(2):
            if mapped_intervals[i][j] not in transformed_support:
                transformed_support.append(mapped_intervals[i][j])
    if zoo in transformed_support:
        error_string = "complex infinity appears in the support, "
        error_string += "please check for an undefined transformation "
        error_string += "such as 1/0"
        raise RVError(error_string)
    transformed_support.sort()
    # Find which segment of the transformation each transformation
    #   function applies to
    applicable_interval_indexes = []
    for i in range(len(mapped_intervals)):
        matching_indexes = []
        for j in range(len(transformed_support) - 1):
            if min(mapped_intervals[i]) <= transformed_support[j]:
                if max(mapped_intervals[i]) >= transformed_support[j + 1]:
                    matching_indexes.append(j)
        applicable_interval_indexes.append(matching_indexes)
    # Find the appropriate inverse for each g(x)
    inverse_functions = []
    for i in range(len(active_transform[0])):
        # Find the test point for selecting the correct inverse.
        if [active_transform[1][i], active_transform[1][i + 1]] == [-oo, oo]:
            test_point = 0
        elif active_transform[1][i] == -oo and active_transform[1][i + 1] != oo:
            test_point = active_transform[1][i + 1] - 1
        elif active_transform[1][i] != -oo and active_transform[1][i + 1] == oo:
            test_point = active_transform[1][i] + 1
        else:
            test_point = (active_transform[1][i] + active_transform[1][i + 1]) / 2
        # Create a list of possible inverses
        candidate_inverses = solve(active_transform[0][i] - t, x)
        # Use the test point to determine the correct inverse.
        selected_inverse = False
        for j in range(len(candidate_inverses)):
            # If g-1(g(test_point))=test_point, then the inverse is correct
            inverse_test_value = candidate_inverses[j].subs(
                t, active_transform[0][i].subs(x, test_point)
            )
            if simplify(inverse_test_value - test_point) == 0:
                inverse_functions.append(candidate_inverses[j])
                selected_inverse = True
                break
            try:
                if inverse_test_value <= Float(float(test_point), 10) + 0.0000001:
                    if inverse_test_value >= Float(float(test_point), 10) - 0.0000001:
                        inverse_functions.append(candidate_inverses[j])
                        selected_inverse = True
                        break
            except Exception:
                if j == len(candidate_inverses) - 1 and len(inverse_functions) < i + 1:
                    inverse_functions.append(None)
                    selected_inverse = True
        # Some symbolic comparisons do not trigger either branch above.
        # Fall back to the only available inverse when the mapping is
        # unambiguous.
        if not selected_inverse and len(candidate_inverses) == 1:
            inverse_functions.append(candidate_inverses[0])
    # Find the transformation function for each segment'
    segment_transform_functions = []
    for i in range(len(pdf_random_variable.func)):
        # Only find transformation for applicable segments
        for j in range(len(active_transform[0])):
            if active_transform[1][j] >= pdf_random_variable.support[i]:
                if active_transform[1][j + 1] <= pdf_random_variable.support[i + 1]:
                    if j >= len(inverse_functions) or inverse_functions[j] is None:
                        continue
                    if not isinstance(pdf_random_variable.func[i], (float, int)):
                        transformed_piece = pdf_random_variable.func[i].subs(
                            x, inverse_functions[j]
                        )
                        transformed_piece = transformed_piece * diff(inverse_functions[j], t)
                    else:
                        transformed_piece = pdf_random_variable.func[i] * diff(
                            inverse_functions[j], t
                        )
                    segment_transform_functions.append(transformed_piece)
    # Sum the transformations for each piece of the transformed
    #   random variable
    transformed_functions_by_interval = []
    for i in range(len(transformed_support) - 1):
        combined_interval_function = 0
        for j in range(len(segment_transform_functions)):
            if i in applicable_interval_indexes[j]:
                if mapped_intervals[j][0] < mapped_intervals[j][1]:
                    combined_interval_function = (
                        combined_interval_function + segment_transform_functions[j]
                    )
                else:
                    combined_interval_function = (
                        combined_interval_function - segment_transform_functions[j]
                    )
        transformed_functions_by_interval.append(combined_interval_function)
    # Substitute x into the transformed random variable
    transformed_pdf_functions = []
    for i in range(len(transformed_functions_by_interval)):
        if not isinstance(transformed_functions_by_interval[i], (int, float)):
            transformed_pdf_functions.append(
                simplify(transformed_functions_by_interval[i].subs(t, x))
            )
        else:
            transformed_pdf_functions.append(transformed_functions_by_interval[i])
    # Create and return the random variable
    return RV(transformed_pdf_functions, transformed_support, ["continuous", "pdf"])


def _transform_discrete_functional(random_variable, transform_spec):
    for element in random_variable.support:
        if (element in [-oo, oo]) or (isinstance(element, Symbol)):
            err_string = "Transform is not implemented for discrete "
            err_string += "random variables with symbolic or inifinite "
            err_string += "support"
            raise RVError(err_string)
    converted_random_variable = convert(random_variable)
    return transform(converted_random_variable, transform_spec)


def _transform_discrete(pdf_random_variable, transform_spec):
    active_transform = transform_spec
    transformed_support_points = []
    # Find the portion of the transformation each element
    #   in the random variable applies to, and then transform it
    for i in range(len(pdf_random_variable.support)):
        support_point = pdf_random_variable.support[i]
        if support_point < min(active_transform[1]) or support_point > max(active_transform[1]):
            transformed_support_points.append(support_point)
        for j in range(len(active_transform[1]) - 1):
            if (
                support_point >= active_transform[1][j]
                and support_point <= active_transform[1][j + 1]
            ):
                transformed_support_points.append(
                    active_transform[0][j].subs(x, pdf_random_variable.support[i])
                )
                break
                # Break is required, otherwise points on the boundaries
                #   between two segments of the transformation will
                #   be entered twice
    # Sort the function and support lists
    sorted_pairs = list(zip(transformed_support_points, pdf_random_variable.func))
    sorted_pairs.sort()
    sorted_support_points = []
    sorted_probabilities = []
    for i in range(len(sorted_pairs)):
        sorted_support_points.append(sorted_pairs[i][0])
        sorted_probabilities.append(sorted_pairs[i][1])
    # Combine redundant elements in the list
    unique_support_points = []
    aggregated_probabilities = []
    for i in range(len(sorted_support_points)):
        if sorted_support_points[i] not in unique_support_points:
            unique_support_points.append(sorted_support_points[i])
            aggregated_probabilities.append(sorted_probabilities[i])
        elif sorted_support_points[i] in unique_support_points:
            support_index = unique_support_points.index(sorted_support_points[i])
            aggregated_probabilities[support_index] += sorted_probabilities[i]
    # Return the transformed random variable
    return RV(aggregated_probabilities, unique_support_points, ["discrete", "pdf"])


def truncate(random_variable, support_interval):
    """
    Procedure Name: Truncate
    Purpose: Truncate a random variable
    Arguments: 1. random_variable: A random variable
               2. support_interval: The support of the truncated random variable
    Output:    1. A truncated random variable
    """
    # Check to make sure the support of the truncated random
    #   variable is given in ascending order
    if support_interval[0] > support_interval[1]:
        raise RVError("The support must be given in ascending order")

    # Conver the random variable to its pdf form
    pdf_random_variable = pdf(random_variable)
    cdf_random_variable = cdf(random_variable)

    # If the random variable is continuous, find and return
    #   the truncated random variable
    if random_variable.is_continuous():
        return _truncate_continuous(pdf_random_variable, cdf_random_variable, support_interval)

    # If the random variable is a discrete function, find and return
    #   the truncated random variable
    if random_variable.is_discrete_functional():
        return _truncate_discrete_functional(
            pdf_random_variable, cdf_random_variable, support_interval
        )

    # If the distribution is discrete, find and return the
    #   truncated random variable
    if random_variable.is_discrete():
        return _truncate_discrete(pdf_random_variable, support_interval)


def _truncate_continuous(pdf_random_variable, cdf_random_variable, support_interval):
    # Find the area of the truncated random variable
    truncation_area = cdf(cdf_random_variable, support_interval[1]) - cdf(
        cdf_random_variable, support_interval[0]
    )
    # Cut out parts of the distribution that don't fall
    #   within the new limits
    for i in range(len(pdf_random_variable.func)):
        if support_interval[0] >= pdf_random_variable.support[i]:
            if support_interval[0] <= pdf_random_variable.support[i + 1]:
                lower_piece_index = i
        if support_interval[1] >= pdf_random_variable.support[i]:
            if support_interval[1] <= pdf_random_variable.support[i + 1]:
                upper_piece_index = i
    truncated_functions = []
    for i in range(len(pdf_random_variable.func)):
        if i >= lower_piece_index and i <= upper_piece_index:
            truncated_functions.append(simplify(pdf_random_variable.func[i] / truncation_area))
    truncated_support = [support_interval[0]]
    upper_piece_index += 1
    for i in range(len(pdf_random_variable.support)):
        if i > lower_piece_index and i < upper_piece_index:
            truncated_support.append(pdf_random_variable.support[i])
    truncated_support.append(support_interval[1])
    # Return the truncated random variable
    return RV(truncated_functions, truncated_support, ["continuous", "pdf"])


def _truncate_discrete_functional(pdf_random_variable, cdf_random_variable, support_interval):
    # Find the area of the truncated random variable
    truncation_area = cdf(cdf_random_variable, support_interval[1]) - cdf(
        cdf_random_variable, support_interval[0]
    )
    # Cut out parts of the distribution that don't fall
    #   within the new limits
    for i in range(len(pdf_random_variable.func)):
        if support_interval[0] >= pdf_random_variable.support[i]:
            if support_interval[0] <= pdf_random_variable.support[i + 1]:
                lower_piece_index = i
        if support_interval[1] >= pdf_random_variable.support[i]:
            if support_interval[1] <= pdf_random_variable.support[i + 1]:
                upper_piece_index = i
    truncated_functions = []
    for i in range(len(pdf_random_variable.func)):
        if i >= lower_piece_index and i <= upper_piece_index:
            truncated_functions.append(pdf_random_variable.func[i] / truncation_area)
    truncated_support = [support_interval[0]]
    upper_piece_index += 1
    for i in range(len(pdf_random_variable.support)):
        if i > lower_piece_index and i < upper_piece_index:
            truncated_support.append(pdf_random_variable.support[i])
    truncated_support.append(support_interval[1])
    # Return the truncated random variable
    return RV(truncated_functions, truncated_support, ["discrete_functional", "pdf"])


def _truncate_discrete(pdf_random_variable, support_interval):
    min_support, max_support = tuple(support_interval)
    fast_rv = applpy_rust.truncate_discrete(pdf_random_variable, min_support, max_support)
    return RV(
        func=fast_rv.function,
        support=fast_rv.support,
        functional_form=fast_rv.functional_form,
        domain_type=fast_rv.domain_type,
    )


def mixture(mix_parameters, mix_random_variables):
    """
    Procedure Name: Mixture
    Purpose: Mixes random variables X1,X2,...,Xn
    Arguments:   1. mix_parameters: A mix of probability weights
                 2. mix_random_variables: RV's X1,X2,...,Xn
    Output:      1. The mixture RV
    """

    # Check to make sure that the arguments are lists
    if not isinstance(mix_parameters, list) or not isinstance(mix_random_variables, list):
        raise RVError("Both arguments must be in list format")
    # Check to make sure the lists are of equal length
    if len(mix_parameters) != len(mix_random_variables):
        raise RVError("Mix parameter and RV lists must be the same length")
    # Check to ensure that the mix rv's are all of the same type
    #   (discrete or continuous)
    for i in range(len(mix_random_variables)):
        if mix_random_variables[0].domain_type != mix_random_variables[i].domain_type:
            raise RVError("Mix RVs must be all continuous or discrete")
    # Convert the Mix RVs to their PDF form
    mixture_pdf_random_variables = []
    for i in range(len(mix_random_variables)):
        mixture_pdf_random_variables.append(pdf(mix_random_variables[i]))

    # If the distributions are continuous, find and return the
    #   mixture pdf
    if mixture_pdf_random_variables[0].is_continuous():
        return _mixture_continuous(mix_parameters, mixture_pdf_random_variables)

    if mixture_pdf_random_variables[0].is_discrete_functional():
        mixture_pdf_random_variables = _mixture_discrete_functional(mixture_pdf_random_variables)

    # If the distributions are discrete, find and return the
    #   mixture pdf
    if mixture_pdf_random_variables[0].is_discrete():
        return _mixture_discrete(mix_parameters, mixture_pdf_random_variables)


def _mixture_continuous(mix_parameters, mixture_pdf_random_variables):
    # Compute the support of the mixture as the union of the supports
    #   of the mix rvs
    mixture_support = []
    for i in range(len(mixture_pdf_random_variables)):
        for j in range(len(mixture_pdf_random_variables[i].support)):
            if mixture_pdf_random_variables[i].support[j] not in mixture_support:
                mixture_support.append(mixture_pdf_random_variables[i].support[j])
    mixture_support.sort()
    # Compute and return the mixed PDF
    mixture_functions = []
    for i in range(len(mixture_support) - 1):
        interval_mixture_function = 0
        for j in range(len(mix_parameters)):
            segment_count = len(mixture_pdf_random_variables[j].support) - 1
            for k in range(segment_count):
                if mixture_pdf_random_variables[j].support[k] <= mixture_support[i]:
                    if mixture_support[i + 1] <= mixture_pdf_random_variables[j].support[k + 1]:
                        weighted_piece = mixture_pdf_random_variables[j].func[k] * mix_parameters[j]
                        interval_mixture_function += weighted_piece
        simplify(interval_mixture_function)
        mixture_functions.append(interval_mixture_function)
    # Return the mixture rv
    return RV(mixture_functions, mixture_support, ["continuous", "pdf"])


def _mixture_discrete_functional(mixture_pdf_random_variables):
    # If the random variables are discrete in functional form,
    #   convert each one to explicit discrete form.
    for i in range(len(mixture_pdf_random_variables)):
        for support_value in mixture_pdf_random_variables[i].support:
            if not isinstance(support_value, (int, float)):
                err_string = "Mixture does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        mixture_pdf_random_variables[i] = convert(mixture_pdf_random_variables[i])
    return mixture_pdf_random_variables


def _mixture_discrete(mix_parameters, mixture_pdf_random_variables):
    fast_rv = applpy_rust.mixture_discrete(
        random_variables=mixture_pdf_random_variables,
        mix_weights=mix_parameters,
    )
    return RV(
        func=fast_rv.function,
        support=fast_rv.support,
        functional_form=fast_rv.functional_form,
        domain_type=fast_rv.domain_type,
    )


# Backward-compatible aliases for legacy APPLPy function names.
Convert = convert
Pow = power
Sqrt = sqrt
Transform = transform
Mixture = mixture
Truncate = truncate
