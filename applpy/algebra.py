"""
Algebraic operations on one or two random variables.
"""

from sympy import Symbol, exp, expand, integrate, ln, nan, oo, simplify

from .conversion import pdf
from .rv import RV, RVError, x
from .transform import Convert, transform


def convolution_iid(random_variable, n):
    """
    Procedure Name: convolution_iid
    Purpose: Compute the convolution of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The convolution of n iid random variables
    """
    # Check to make sure n is an integer
    if not isinstance(n, int):
        raise RVError("The second argument must be an integer")

    # Compute the iid convolution
    pdf_random_variable = pdf(random_variable)
    sum_rv = pdf_random_variable
    for _ in range(n - 1):
        sum_rv += pdf_random_variable
    return pdf(sum_rv)


def product_iid(random_variable, n):
    """
    Procedure Name: product_iid
    Purpose: Compute the product of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The product of n iid random variables
    """
    # Check to make sure n is an integer
    if not isinstance(n, int):
        raise RVError("The second argument must be an integer")

    # Compute the iid convolution
    pdf_random_variable = pdf(random_variable)
    product_rv = pdf_random_variable
    for _ in range(n - 1):
        product_rv *= pdf_random_variable
    return pdf(product_rv)


def convolution(random_variable_1, random_variable_2):
    """
    Procedure Name: convolution
    Purpose: Compute the convolution of two independent
                random variables
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The convolution of random_variable_1 and random_variable_2
    """
    if random_variable_1.domain_type != random_variable_2.domain_type:
        discrete_domain_types = ["discrete", "discrete_functional"]
        if (random_variable_1.domain_type not in discrete_domain_types) and (
            random_variable_2.domain_type not in discrete_domain_types
        ):
            raise RVError("Both random variables must have the same type")

    # Convert both random variables to their PDF form
    left_pdf_rv = pdf(random_variable_1)
    right_pdf_rv = pdf(random_variable_2)

    # If the distributions are continuous, find and return the convolution
    #   of the two random variables
    if random_variable_1.is_continuous():
        # left_pdf_rv.drop_assumptions()
        # right_pdf_rv.drop_assumptions()
        # If the two distributions are both lifetime distributions, treat
        #   as a special case
        if random_variable_1.support == [0, oo] and random_variable_2.support == [0, oo]:
            # x=Symbol('x',positive=True)
            sum_symbol = Symbol("z", positive=True)
            left_function = left_pdf_rv.func[0]
            right_function = right_pdf_rv.func[0].subs(x, sum_symbol - x)
            convolution_integrand = expand(left_function * right_function)
            convolution_expression = integrate(
                convolution_integrand, (x, 0, sum_symbol), conds="none"
            )
            convolution_expression = convolution_expression.subs(sum_symbol, x)
            convolution_expression = simplify(convolution_expression)
            return RV([convolution_expression], [0, oo], ["continuous", "pdf"])
        # Otherwise, compute the convolution using the product method
        elif random_variable_1.support == [0, 1] and random_variable_2.support == [0, 1]:
            sum_symbol = Symbol("z", positive=True)
            integration_symbol = Symbol("xx", positive=True)
            left_function = left_pdf_rv.func[0].subs(x, integration_symbol)
            right_function = right_pdf_rv.func[0].subs(x, sum_symbol - integration_symbol)
            lower_interval_function = integrate(
                left_function * right_function,
                (integration_symbol, 0, sum_symbol),
            )
            lower_interval_function = lower_interval_function.subs(sum_symbol, x)
            upper_interval_function = integrate(
                left_function * right_function,
                (integration_symbol, sum_symbol - 1, 1),
            )
            upper_interval_function = upper_interval_function.subs(sum_symbol, x)
            return RV(
                [lower_interval_function, upper_interval_function],
                [0, 1, 2],
                ["continuous", "pdf"],
            )
        else:
            log_transform = [[ln(x)], [0, oo]]
            exp_transform = [[exp(x), exp(x)], [-oo, 0, oo]]
            transformed_left_rv = transform(left_pdf_rv, exp_transform)
            transformed_right_rv = transform(right_pdf_rv, exp_transform)
            transformed_product_rv = product(transformed_left_rv, transformed_right_rv)
            convolution_rv = transform(transformed_product_rv, log_transform)
            simplified_functions = []
            for index in range(len(convolution_rv.func)):
                simplified_functions.append(simplify(convolution_rv.func[index]))
            return RV(simplified_functions, convolution_rv.support, ["continuous", "pdf"])

    # If the two random variables are discrete in functinonal form,
    #   find and return the convolution of the two random variables
    if random_variable_1.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "convolution does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_1 = Convert(random_variable_1)
    if random_variable_2.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "convolution does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_2 = Convert(random_variable_2)

    if random_variable_1.is_discrete():
        fast_rv_1 = random_variable_1.to_fast_rv()
        fast_rv_2 = random_variable_2.to_fast_rv()
        return RV.from_fast_rv(fast_rv_1 + fast_rv_2)


def product(random_variable_1, random_variable_2):
    """
    Procedure Name: product
    Purpose: Compute the product of two independent
                random variables
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The product of random_variable_1 and random_variable_2
    """
    if random_variable_1.domain_type != random_variable_2.domain_type:
        raise RVError("both random variables must have the same functional form")

    if random_variable_1.is_continuous():
        # left_pdf_rv.drop_assumptions()
        # right_pdf_rv.drop_assumptions()
        v = Symbol("v", positive=True)
        # Place zero in the support of X if it is not there already
        left_pdf_rv = pdf(random_variable_1)
        left_functions = []
        left_support = []
        for i in range(len(left_pdf_rv.func)):
            left_functions.append(left_pdf_rv.func[i])
            left_support.append(left_pdf_rv.support[i])
            if left_pdf_rv.support[i] < 0:
                if left_pdf_rv.support[i + 1] > 0:
                    left_functions.append(left_pdf_rv.func[i])
                    left_support.append(0)
        left_support.append(left_pdf_rv.support[len(left_pdf_rv.support) - 1])
        left_pdf_with_zero = RV(left_functions, left_support, ["continuous", "pdf"])
        # Place zero in the support of Y if it is not already there
        right_pdf_rv = pdf(random_variable_2)
        right_functions = []
        right_support = []
        for i in range(len(right_pdf_rv.func)):
            right_functions.append(right_pdf_rv.func[i])
            right_support.append(right_pdf_rv.support[i])
            if right_pdf_rv.support[i] < 0:
                if right_pdf_rv.support[i + 1] > 0:
                    right_functions.append(right_pdf_rv.func[i])
                    right_support.append(0)
        right_support.append(right_pdf_rv.support[len(right_pdf_rv.support) - 1])
        right_pdf_with_zero = RV(right_functions, right_support, ["continuous", "pdf"])
        # Initialize the support list for the product V=X*Y
        product_support = []
        for i in range(len(left_pdf_with_zero.support)):
            for j in range(len(right_pdf_with_zero.support)):
                val = left_pdf_with_zero.support[i] * right_pdf_with_zero.support[j]
                if val == nan:
                    val = 0
                if val not in product_support:
                    product_support.append(val)
        product_support.sort()
        # Initialize the pdf segments of v
        product_functions = []
        for i in range(len(product_support) - 1):
            product_functions.append(0)
        # Loop through each piecewise segment of X
        for i in range(len(left_pdf_with_zero.func)):
            # Loop through each piecewise segment of Y
            for j in range(len(right_pdf_with_zero.func)):
                # Define the corner of the rectangular region
                a = left_pdf_with_zero.support[i]
                b = left_pdf_with_zero.support[i + 1]
                c = right_pdf_with_zero.support[j]
                d = right_pdf_with_zero.support[j + 1]
                # If the region is in the first quadrant, compute the
                #   required integrals sequentially
                if a >= 0 and c >= 0:
                    v = Symbol("v", positive=True)
                    if not isinstance(right_pdf_with_zero.func[j], (float, int)):
                        right_function = right_pdf_with_zero.func[j].subs(x, v / x)
                    else:
                        right_function = right_pdf_with_zero.func[j]
                    left_function = left_pdf_with_zero.func[i]
                    base_integral = integrate(left_function * right_function * (1 / x), (x, a, b))
                    if d < oo:
                        upper_integral = integrate(
                            left_function * right_function * (1 / x), (x, v / d, b)
                        )
                    if c > 0:
                        lower_integral = integrate(
                            left_function * right_function * (1 / x), (x, a, v / c)
                        )
                    if c > 0 and d < oo and a * d < b * c:
                        middle_integral = integrate(
                            left_function * right_function * (1 / x), (x, v / d, v / c)
                        )
                    # 1st Qd, Scenario 1
                    if c == 0 and d == oo:
                        for k in range(len(product_functions)):
                            if product_support[k] >= 0:
                                product_functions[k] += base_integral
                    # 1st Qd, Scenario 2
                    if c == 0 and d < oo:
                        for k in range(len(product_functions)):
                            if product_support[k] >= 0 and product_support[k + 1] <= a * d:
                                product_functions[k] += base_integral
                            if product_support[k] >= a * d and product_support[k + 1] <= b * d:
                                product_functions[k] += upper_integral
                    # 1st Qd, Scenario 3
                    if c > 0 and d == oo:
                        for k in range(len(product_functions)):
                            if product_support[k] >= b * c:
                                product_functions[k] += base_integral
                            if product_support[k] >= a * c and product_support[k + 1] <= b * c:
                                product_functions[k] += lower_integral
                    # 1st Qd, Scenario 4
                    if c > 0 and d < oo:
                        # Case 1
                        if a * d < b * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= a * c and product_support[k + 1] <= a * d:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= a * d and product_support[k + 1] <= b * c:
                                    product_functions[k] += middle_integral
                                if product_support[k] >= b * c and product_support[k + 1] <= b * d:
                                    product_functions[k] += upper_integral
                        # Case 2
                        if a * d == b * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= a * c and product_support[k + 1] <= a * d:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= b * c and product_support[k + 1] <= b * d:
                                    product_functions[k] += upper_integral
                        # Case 3
                        if a * d > b * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= a * c and product_support[k + 1] <= b * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= b * c and product_support[k + 1] <= a * d:
                                    product_functions[k] += base_integral
                                if product_support[k] >= a * d and product_support[k + 1] <= b * d:
                                    product_functions[k] += upper_integral
                # If the region is in the second quadrant, compute
                #   the required integrals sequentially
                if a < 0 and c < 0:
                    v = Symbol("v", positive=True)
                    if not isinstance(right_pdf_with_zero.func[j], (float, int)):
                        right_function = right_pdf_with_zero.func[j].subs(x, v / x)
                    else:
                        right_function = right_pdf_with_zero.func[j]
                    left_function = left_pdf_with_zero.func[i]
                    base_integral = -integrate(left_function * right_function * (1 / x), (x, a, b))
                    if d < 0:
                        upper_integral = -integrate(
                            left_function * right_function * (1 / x), (x, (v / d), b)
                        )
                    if c > -oo:
                        lower_integral = -integrate(
                            left_function * right_function * (1 / x), (x, a, (v / c))
                        )
                    if c > -oo and d < 0:
                        middle_integral = -integrate(
                            left_function * right_function * (1 / x), (x, (v / d), (v / c))
                        )
                    # 2nd Qd, Scenario 1
                    if c == -oo and d == 0:
                        for k in range(len(product_functions)):
                            if product_support[k] >= 0:
                                product_functions[k] += base_integral
                    # 2nd Qd, Scenario 2
                    if c == -oo and d < 0:
                        for k in range(len(product_functions)):
                            if product_support[k] >= a * d and product_support[k + 1] <= oo:
                                product_functions[k] += base_integral
                            if product_support[k] >= b * d and product_support[k + 1] <= a * d:
                                product_functions[k] += upper_integral
                    # 2nd Qd, Scenario 3
                    if c > -oo and d == 0:
                        for k in range(len(product_functions)):
                            if product_support[k] >= 0 and product_support[k + 1] <= b * c:
                                product_functions[k] += base_integral
                            if product_support[k] >= b * c and product_support[k + 1] <= a * c:
                                product_functions[k] += lower_integral
                    # 2nd Qd, Scenario 4
                    if c > -oo and d < 0:
                        # Case 1
                        if a * d > b * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= a * d and product_support[k + 1] <= a * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= b * c and product_support[k + 1] <= a * d:
                                    product_functions[k] += middle_integral
                                if product_support[k] >= b * d and product_support[k + 1] <= b * c:
                                    product_functions[k] += upper_integral
                        # Case 2
                        if a * d == b * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= a * d and product_support[k + 1] <= a * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= b * d and product_support[k + 1] <= b * c:
                                    product_functions[k] += upper_integral
                        # Case 3
                        if a * d < b * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= b * c and product_support[k + 1] <= a * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= a * d and product_support[k + 1] <= b * c:
                                    product_functions[k] += base_integral
                                if product_support[k] >= b * d and product_support[k + 1] <= a * d:
                                    product_functions[k] += upper_integral
                # If the region is in the third quadrant, compute
                #   the required integrals sequentially
                if a < 0 and c >= 0:
                    v = Symbol("v", negative=True)
                    if not isinstance(right_pdf_with_zero.func[j], (float, int)):
                        right_function = right_pdf_with_zero.func[j].subs(x, v / x)
                    else:
                        right_function = right_pdf_with_zero.func[j]
                    left_function = left_pdf_with_zero.func[i]
                    base_integral = -integrate(left_function * right_function * (1 / x), (x, a, b))
                    if d < oo:
                        upper_integral = -integrate(
                            left_function * right_function * (1 / x), (x, a, (v / d))
                        )
                    if c > 0:
                        lower_integral = -integrate(
                            left_function * right_function * (1 / x), (x, (v / b), c)
                        )
                    if c > 0 and d < oo:
                        middle_integral = -integrate(
                            left_function * right_function * (1 / x), (x, (v / c), (v / d))
                        )
                    # 3rd Qd, Scenario 1
                    if c == 0 and d == oo:
                        for k in range(len(product_functions)):
                            if product_support[k + 1] <= 0:
                                product_functions[k] += base_integral
                    # 3rd Qd, Scenario 2
                    if c == 0 and d < oo:
                        for k in range(len(product_functions)):
                            if product_support[k] >= b * d and product_support[k + 1] <= 0:
                                product_functions[k] += base_integral
                            if product_support[k] >= a * d and product_support[k + 1] <= b * d:
                                product_functions[k] += upper_integral
                    # 3rd Qd, Scenario 3
                    if c > 0 and d == oo:
                        for k in range(len(product_functions)):
                            if product_support[k] >= -oo and product_support[k + 1] <= a * c:
                                product_functions[k] += base_integral
                            if product_support[k] >= a * c and product_support[k + 1] <= b * c:
                                product_functions[k] += lower_integral
                    # 3rd Qd, Scenario 4
                    if c > 0 and d < oo:
                        # Case 1
                        if b * d > a * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= b * d and product_support[k + 1] <= b * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= a * c and product_support[k + 1] <= b * d:
                                    product_functions[k] += middle_integral
                                if product_support[k] >= a * d and product_support[k + 1] <= a * c:
                                    product_functions[k] += upper_integral
                        # Case 2
                        if a * c == b * d:
                            for k in range(len(product_functions)):
                                if product_support[k] >= a * d and product_support[k + 1] <= a * c:
                                    product_functions[k] += upper_integral
                                if product_support[k] >= b * d and product_support[k + 1] <= b * c:
                                    product_functions[k] += lower_integral
                        # Case 3
                        if a * c > b * d:
                            for k in range(len(product_functions)):
                                if product_support[k] >= a * c and product_support[k + 1] <= b * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= b * d and product_support[k + 1] <= a * c:
                                    product_functions[k] += base_integral
                                if product_support[k] >= a * d and product_support[k + 1] <= b * d:
                                    product_functions[k] += upper_integral
                # If the region is in the fourth quadrant, compute
                #   the required integrals sequentially
                if a >= 0 and c < 0:
                    v = Symbol("v", negative=True)
                    if not isinstance(right_pdf_with_zero.func[j], (float, int)):
                        right_function = right_pdf_with_zero.func[j].subs(x, v / x)
                    else:
                        right_function = right_pdf_with_zero.func[j]
                    left_function = left_pdf_with_zero.func[i]
                    base_integral = integrate(left_function * right_function * (1 / x), (x, a, b))
                    if d < 0:
                        upper_integral = integrate(
                            left_function * right_function * (1 / x), (x, a, (v / d))
                        )
                    if c > -oo:
                        lower_integral = integrate(
                            left_function * right_function * (1 / x), (x, (v / c), b)
                        )
                    if c > -oo and d < 0:
                        middle_integral = integrate(
                            left_function * right_function * (1 / x), (x, (v / c), (v / d))
                        )
                    # 4th Qd, Scenario 1
                    if c == oo and d == 0:
                        for k in range(len(product_functions)):
                            if product_support[k + 1] <= 0:
                                product_functions[k] += base_integral
                    # 4th Qd, Scenario 2
                    if c == oo and d < 0:
                        for k in range(len(product_functions)):
                            if product_support[k] >= -oo and product_support[k + 1] <= b * d:
                                product_functions[k] += base_integral
                            if product_support[k] >= b * d and product_support[k + 1] <= a * d:
                                product_functions[k] += upper_integral
                    # 4th Qd, Scenario 3
                    if c > -oo and d == 0:
                        for k in range(len(product_functions)):
                            if product_support[k] >= a * c and product_support[k + 1] <= 0:
                                product_functions[k] += base_integral
                            if product_support[k] >= b * c and product_support[k + 1] <= a * c:
                                product_functions[k] += lower_integral
                    # 4th Qd, Scenario 4
                    if c > -oo and d < 0:
                        # Case 1
                        if a * c > b * d:
                            for k in range(len(product_functions)):
                                if product_support[k] >= b * c and product_support[k + 1] <= b * d:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= b * d and product_support[k + 1] <= a * c:
                                    product_functions[k] += middle_integral
                                if product_support[k] >= a * c and product_support[k + 1] <= a * d:
                                    product_functions[k] += upper_integral
                        # Case 2
                        if a * d == b * c:
                            for k in range(len(product_functions)):
                                if product_support[k] >= b * c and product_support[k + 1] <= a * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= a * c and product_support[k + 1] <= a * d:
                                    product_functions[k] += upper_integral
                        # Case 3
                        if a * c < b * d:
                            for k in range(len(product_functions)):
                                if product_support[k] >= b * c and product_support[k + 1] <= a * c:
                                    product_functions[k] += lower_integral
                                if product_support[k] >= a * c and product_support[k + 1] <= b * d:
                                    product_functions[k] += base_integral
                                if product_support[k] >= b * d and product_support[k + 1] <= a * d:
                                    product_functions[k] += upper_integral
        product_functions_final = []
        for i in range(len(product_functions)):
            if not isinstance(product_functions[i], (int, float)):
                product_functions_final.append(simplify(product_functions[i]).subs(v, x))
            else:
                product_functions_final.append(product_functions[i])
        return RV(product_functions_final, product_support, ["continuous", "pdf"])

    if random_variable_1.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "product does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_1 = Convert(random_variable_1)
    if random_variable_2.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "product does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_2 = Convert(random_variable_2)

    if random_variable_1.is_discrete():
        fast_rv_1 = random_variable_1.to_fast_rv()
        fast_rv_2 = random_variable_2.to_fast_rv()
        return RV.from_fast_rv(fast_rv_1 * fast_rv_2)


# Backward-compatible aliases for legacy APPLPy function names.
ConvolutionIID = convolution_iid
ProductIID = product_iid
Convolution = convolution
Product = product
