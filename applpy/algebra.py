"""
Algebraic operations on one or two random variables.
"""

from sympy import Symbol, exp, expand, integrate, ln, nan, oo, simplify

from .rv import RV, RVError, Convert, Transform, pdf, x


def ConvolutionIID(random_variable, n):
    """
    Procedure Name: ConvolutionIID
    Purpose: Compute the convolution of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The convolution of n iid random variables
    """
    # Check to make sure n is an integer
    if not isinstance(n, int):
        raise RVError("The second argument must be an integer")

    # Compute the iid convolution
    X_dummy = pdf(random_variable)
    X_final = X_dummy
    for _ in range(n - 1):
        X_final += X_dummy
    return pdf(X_final)


def ProductIID(random_variable, n):
    """
    Procedure Name: ProductIID
    Purpose: Compute the product of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The product of n iid random variables
    """
    # Check to make sure n is an integer
    if not isinstance(n, int):
        raise RVError("The second argument must be an integer")

    # Compute the iid convolution
    X_dummy = pdf(random_variable)
    X_final = X_dummy
    for _ in range(n - 1):
        X_final *= X_dummy
    return pdf(X_final)


def Convolution(random_variable_1, random_variable_2):
    """
    Procedure Name: Convolution
    Purpose: Compute the convolution of two independent
                random variables
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The convolution of random_variable_1 and random_variable_2
    """
    # If the two random variables are not both continuous or
    #   both discrete, return an error
    if random_variable_1.domain_type != random_variable_2.domain_type:
        discr = ["discrete", "discrete_functional"]
        if (random_variable_1.domain_type not in discr) and (
            random_variable_2.domain_type not in discr
        ):
            raise RVError("Both random variables must have the same type")

    # Convert both random variables to their PDF form
    X1_dummy = pdf(random_variable_1)
    X2_dummy = pdf(random_variable_2)

    # If the distributions are continuous, find and return the convolution
    #   of the two random variables
    if random_variable_1.is_continuous():
        # X1_dummy.drop_assumptions()
        # X2_dummy.drop_assumptions()
        # If the two distributions are both lifetime distributions, treat
        #   as a special case
        if random_variable_1.support == [0, oo] and random_variable_2.support == [0, oo]:
            # x=Symbol('x',positive=True)
            z = Symbol("z", positive=True)
            func1 = X1_dummy.func[0]
            func2 = X2_dummy.func[0].subs(x, z - x)
            int_func = expand(func1 * func2)
            conv = integrate(int_func, (x, 0, z), conds="none")
            conv_final = conv.subs(z, x)
            conv = expand(conv_final)
            conv = simplify(conv_final)
            return RV([conv_final], [0, oo], ["continuous", "pdf"])
        # Otherwise, compute the convolution using the product method
        elif random_variable_1.support == [0, 1] and random_variable_2.support == [0, 1]:
            z = Symbol("z", positive=True)
            xx = Symbol("xx", positive=True)
            func1 = X1_dummy.func[0].subs(x, xx)
            func2 = X2_dummy.func[0].subs(x, z - xx)
            fz1 = integrate(func1 * func2, (xx, 0, z))
            fz1 = fz1.subs(z, x)
            fz2 = integrate(func1 * func2, (xx, z - 1, 1))
            fz2 = fz2.subs(z, x)
            return RV([fz1, fz2], [0, 1, 2], ["continuous", "pdf"])
        else:
            gln = [[ln(x)], [0, oo]]
            ge = [[exp(x), exp(x)], [-oo, 0, oo]]
            temp1 = Transform(X1_dummy, ge)
            temp2 = Transform(X2_dummy, ge)
            temp3 = Product(temp1, temp2)
            fz = Transform(temp3, gln)
            convfunc = []
            for i in range(len(fz.func)):
                convfunc.append(simplify(fz.func[i]))
            return RV(convfunc, fz.support, ["continuous", "pdf"])

    # If the two random variables are discrete in functinonal form,
    #   find and return the convolution of the two random variables
    if random_variable_1.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Convolution does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_1 = Convert(random_variable_1)
    if random_variable_2.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Convolution does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_2 = Convert(random_variable_2)

    # If the distributions are discrete, find and return the convolution
    #   of the two random variables.
    if random_variable_1.is_discrete():
        # Convert each random variable to its pdf form
        X1_dummy = pdf(random_variable_1)
        X2_dummy = pdf(random_variable_2)
        # Create function and support lists for the convolution of the
        #   two random variables
        convlist = []
        funclist = []
        for i in range(len(X1_dummy.support)):
            for j in range(len(X2_dummy.support)):
                convlist.append(X1_dummy.support[i] + X2_dummy.support[j])
                funclist.append(X1_dummy.func[i] * X2_dummy.func[j])
        # Sort the function and support lists for the convolution
        sortlist = list(zip(convlist, funclist))
        sortlist.sort()
        convlist2 = []
        funclist2 = []
        for i in range(len(sortlist)):
            convlist2.append(sortlist[i][0])
            funclist2.append(sortlist[i][1])
        # Remove redundant elements in the support list
        convlist3 = []
        funclist3 = []
        for i in range(len(convlist2)):
            if convlist2[i] not in convlist3:
                convlist3.append(convlist2[i])
                funclist3.append(funclist2[i])
            else:
                funclist3[convlist3.index(convlist2[i])] += funclist2[i]
        # Create and return the new random variable
        return RV(funclist3, convlist3, ["discrete", "pdf"])


def Product(random_variable_1, random_variable_2):
    """
    Procedure Name: Product
    Purpose: Compute the product of two independent
                random variables
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The product of random_variable_1 and random_variable_2
    """
    # If the random variable is continuous, find and return the
    #   product of the two random variables
    if random_variable_1.is_continuous():
        # X1_dummy.drop_assumptions()
        # X2_dummy.drop_assumptions()
        v = Symbol("v", positive=True)
        # Place zero in the support of X if it is not there already
        X1 = pdf(random_variable_1)
        xfunc = []
        xsupp = []
        for i in range(len(X1.func)):
            xfunc.append(X1.func[i])
            xsupp.append(X1.support[i])
            if X1.support[i] < 0:
                if X1.support[i + 1] > 0:
                    xfunc.append(X1.func[i])
                    xsupp.append(0)
        xsupp.append(X1.support[len(X1.support) - 1])
        X_dummy = RV(xfunc, xsupp, ["continuous", "pdf"])
        # Place zero in the support of Y if it is not already there
        Y1 = pdf(random_variable_2)
        yfunc = []
        ysupp = []
        for i in range(len(Y1.func)):
            yfunc.append(Y1.func[i])
            ysupp.append(Y1.support[i])
            if Y1.support[i] < 0:
                if Y1.support[i + 1] > 0:
                    yfunc.append(Y1.func[i])
                    ysupp.append(0)
        ysupp.append(Y1.support[len(Y1.support) - 1])
        Y_dummy = RV(yfunc, ysupp, ["continuous", "pdf"])
        # Initialize the support list for the product V=X*Y
        vsupp = []
        for i in range(len(X_dummy.support)):
            for j in range(len(Y_dummy.support)):
                val = X_dummy.support[i] * Y_dummy.support[j]
                if val == nan:
                    val = 0
                if val not in vsupp:
                    vsupp.append(val)
        vsupp.sort()
        # Initialize the pdf segments of v
        vfunc = []
        for i in range(len(vsupp) - 1):
            vfunc.append(0)
        # Loop through each piecewise segment of X
        for i in range(len(X_dummy.func)):
            # Loop through each piecewise segment of Y
            for j in range(len(Y_dummy.func)):
                # Define the corner of the rectangular region
                a = X_dummy.support[i]
                b = X_dummy.support[i + 1]
                c = Y_dummy.support[j]
                d = Y_dummy.support[j + 1]
                # If the region is in the first quadrant, compute the
                #   required integrals sequentially
                if a >= 0 and c >= 0:
                    v = Symbol("v", positive=True)
                    if not isinstance(Y_dummy.func[j], (float, int)):
                        gj = Y_dummy.func[j].subs(x, v / x)
                    else:
                        gj = Y_dummy.func[j]
                    fi = X_dummy.func[i]
                    pv = integrate(fi * gj * (1 / x), (x, a, b))
                    if d < oo:
                        qv = integrate(fi * gj * (1 / x), (x, v / d, b))
                    if c > 0:
                        rv = integrate(fi * gj * (1 / x), (x, a, v / c))
                    if c > 0 and d < oo and a * d < b * c:
                        sv = integrate(fi * gj * (1 / x), (x, v / d, v / c))
                    # 1st Qd, Scenario 1
                    if c == 0 and d == oo:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= 0:
                                vfunc[k] += pv
                    # 1st Qd, Scenario 2
                    if c == 0 and d < oo:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= 0 and vsupp[k + 1] <= a * d:
                                vfunc[k] += pv
                            if vsupp[k] >= a * d and vsupp[k + 1] <= b * d:
                                vfunc[k] += qv
                    # 1st Qd, Scenario 3
                    if c > 0 and d == oo:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= b * c:
                                vfunc[k] += pv
                            if vsupp[k] >= a * c and vsupp[k + 1] <= b * c:
                                vfunc[k] += rv
                    # 1st Qd, Scenario 4
                    if c > 0 and d < oo:
                        # Case 1
                        if a * d < b * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= a * c and vsupp[k + 1] <= a * d:
                                    vfunc[k] += rv
                                if vsupp[k] >= a * d and vsupp[k + 1] <= b * c:
                                    vfunc[k] += sv
                                if vsupp[k] >= b * c and vsupp[k + 1] <= b * d:
                                    vfunc[k] += qv
                        # Case 2
                        if a * d == b * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= a * c and vsupp[k + 1] <= a * d:
                                    vfunc[k] += rv
                                if vsupp[k] >= b * c and vsupp[k + 1] <= b * d:
                                    vfunc[k] += qv
                        # Case 3
                        if a * d > b * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= a * c and vsupp[k + 1] <= b * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= b * c and vsupp[k + 1] <= a * d:
                                    vfunc[k] += pv
                                if vsupp[k] >= a * d and vsupp[k + 1] <= b * d:
                                    vfunc[k] += qv
                # If the region is in the second quadrant, compute
                #   the required integrals sequentially
                if a < 0 and c < 0:
                    v = Symbol("v", positive=True)
                    if not isinstance(Y_dummy.func[j], (float, int)):
                        gj = Y_dummy.func[j].subs(x, v / x)
                    else:
                        gj = Y_dummy.func[j]
                    fi = X_dummy.func[i]
                    pv = -integrate(fi * gj * (1 / x), (x, a, b))
                    if d < 0:
                        qv = -integrate(fi * gj * (1 / x), (x, (v / d), b))
                    if c > -oo:
                        rv = -integrate(fi * gj * (1 / x), (x, a, (v / c)))
                    if c > -oo and d < 0:
                        sv = -integrate(fi * gj * (1 / x), (x, (v / d), (v / c)))
                    # 2nd Qd, Scenario 1
                    if c == -oo and d == 0:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= 0:
                                vfunc[k] += pv
                    # 2nd Qd, Scenario 2
                    if c == -oo and d < 0:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= a * d and vsupp[k + 1] <= oo:
                                vfunc[k] += pv
                            if vsupp[k] >= b * d and vsupp[k + 1] <= a * d:
                                vfunc[k] += qv
                    # 2nd Qd, Scenario 3
                    if c > -oo and d == 0:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= 0 and vsupp[k + 1] <= b * c:
                                vfunc[k] += pv
                            if vsupp[k] >= b * c and vsupp[k + 1] <= a * c:
                                vfunc[k] += rv
                    # 2nd Qd, Scenario 4
                    if c > -oo and d < 0:
                        # Case 1
                        if a * d > b * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= a * d and vsupp[k + 1] <= a * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= b * c and vsupp[k + 1] <= a * d:
                                    vfunc[k] += sv
                                if vsupp[k] >= b * d and vsupp[k + 1] <= b * c:
                                    vfunc[k] += qv
                        # Case 2
                        if a * d == b * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= a * d and vsupp[k + 1] <= a * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= b * d and vsupp[k + 1] <= b * c:
                                    vfunc[k] += qv
                        # Case 3
                        if a * d < b * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= b * c and vsupp[k + 1] <= a * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= a * d and vsupp[k + 1] <= b * c:
                                    vfunc[k] += pv
                                if vsupp[k] >= b * d and vsupp[k + 1] <= a * d:
                                    vfunc[k] += qv
                # If the region is in the third quadrant, compute
                #   the required integrals sequentially
                if a < 0 and c >= 0:
                    v = Symbol("v", negative=True)
                    if not isinstance(Y_dummy.func[j], (float, int)):
                        gj = Y_dummy.func[j].subs(x, v / x)
                    else:
                        gj = Y_dummy.func[j]
                    fi = X_dummy.func[i]
                    pv = -integrate(fi * gj * (1 / x), (x, a, b))
                    if d < oo:
                        qv = -integrate(fi * gj * (1 / x), (x, a, (v / d)))
                    if c > 0:
                        rv = -integrate(fi * gj * (1 / x), (x, (v / b), c))
                    if c > 0 and d < oo:
                        sv = -integrate(fi * gj * (1 / x), (x, (v / c), (v / d)))
                    # 3rd Qd, Scenario 1
                    if c == 0 and d == oo:
                        for k in range(len(vfunc)):
                            if vsupp[k + 1] <= 0:
                                vfunc[k] += pv
                    # 3rd Qd, Scenario 2
                    if c == 0 and d < oo:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= b * d and vsupp[k + 1] <= 0:
                                vfunc[k] += pv
                            if vsupp[k] >= a * d and vsupp[k + 1] <= b * d:
                                vfunc[k] += qv
                    # 3rd Qd, Scenario 3
                    if c > 0 and d == oo:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= -oo and vsupp[k + 1] <= a * c:
                                vfunc[k] += pv
                            if vsupp[k] >= a * c and vsupp[k + 1] <= b * c:
                                vfunc[k] += rv
                    # 3rd Qd, Scenario 4
                    if c > 0 and d < oo:
                        # Case 1
                        if b * d > a * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= b * d and vsupp[k + 1] <= b * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= a * c and vsupp[k + 1] <= b * d:
                                    vfunc[k] += sv
                                if vsupp[k] >= a * d and vsupp[k + 1] <= a * c:
                                    vfunc[k] += qv
                        # Case 2
                        if a * c == b * d:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= a * d and vsupp[k + 1] <= a * c:
                                    vfunc[k] += qv
                                if vsupp[k] >= b * d and vsupp[k + 1] <= b * c:
                                    vfunc[k] += rv
                        # Case 3
                        if a * c > b * d:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= a * c and vsupp[k + 1] <= b * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= b * d and vsupp[k + 1] <= a * c:
                                    vfunc[k] += pv
                                if vsupp[k] >= a * d and vsupp[k + 1] <= b * d:
                                    vfunc[k] += qv
                # If the region is in the fourth quadrant, compute
                #   the required integrals sequentially
                if a >= 0 and c < 0:
                    v = Symbol("v", negative=True)
                    if not isinstance(Y_dummy.func[j], (float, int)):
                        gj = Y_dummy.func[j].subs(x, v / x)
                    else:
                        gj = Y_dummy.func[j]
                    fi = X_dummy.func[i]
                    pv = integrate(fi * gj * (1 / x), (x, a, b))
                    if d < 0:
                        qv = integrate(fi * gj * (1 / x), (x, a, (v / d)))
                    if c > -oo:
                        rv = integrate(fi * gj * (1 / x), (x, (v / c), b))
                    if c > -oo and d < 0:
                        sv = integrate(fi * gj * (1 / x), (x, (v / c), (v / d)))
                    # 4th Qd, Scenario 1
                    if c == oo and d == 0:
                        for k in range(len(vfunc)):
                            if vsupp[k + 1] <= 0:
                                vfunc[k] += pv
                    # 4th Qd, Scenario 2
                    if c == oo and d < 0:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= -oo and vsupp[k + 1] <= b * d:
                                vfunc[k] += pv
                            if vsupp[k] >= b * d and vsupp[k + 1] <= a * d:
                                vfunc[k] += qv
                    # 4th Qd, Scenario 3
                    if c > -oo and d == 0:
                        for k in range(len(vfunc)):
                            if vsupp[k] >= a * c and vsupp[k + 1] <= 0:
                                vfunc[k] += pv
                            if vsupp[k] >= b * c and vsupp[k + 1] <= a * c:
                                vfunc[k] += rv
                    # 4th Qd, Scenario 4
                    if c > -oo and d < 0:
                        # Case 1
                        if a * c > b * d:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= b * c and vsupp[k + 1] <= b * d:
                                    vfunc[k] += rv
                                if vsupp[k] >= b * d and vsupp[k + 1] <= a * c:
                                    vfunc[k] += sv
                                if vsupp[k] >= a * c and vsupp[k + 1] <= a * d:
                                    vfunc[k] += qv
                        # Case 2
                        if a * d == b * c:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= b * c and vsupp[k + 1] <= a * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= a * c and vsupp[k + 1] <= a * d:
                                    vfunc[k] += qv
                        # Case 3
                        if a * c < b * d:
                            for k in range(len(vfunc)):
                                if vsupp[k] >= b * c and vsupp[k + 1] <= a * c:
                                    vfunc[k] += rv
                                if vsupp[k] >= a * c and vsupp[k + 1] <= b * d:
                                    vfunc[k] += pv
                                if vsupp[k] >= b * d and vsupp[k + 1] <= a * d:
                                    vfunc[k] += qv
        vfunc_final = []
        for i in range(len(vfunc)):
            if not isinstance(vfunc[i], (int, float)):
                vfunc_final.append(simplify(vfunc[i]).subs(v, x))
            else:
                vfunc_final.append(vfunc[i])
        return RV(vfunc_final, vsupp, ["continuous", "pdf"])

    # If the two random variables are discrete in functinonal form,
    #   find and return the product of the two random variables
    if random_variable_1.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Product does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_1 = Convert(random_variable_1)
    if random_variable_2.is_discrete_functional():
        for num in random_variable_1.support:
            if not isinstance(num, (int, float)):
                err_string = "Product does not currently work with"
                err_string = " RVs that have symbolic or infinite support"
                raise RVError(err_string)
        random_variable_2 = Convert(random_variable_2)

    # If the distributions are discrete, find and return the product
    #   of the two random variables.
    if random_variable_1.is_discrete():
        # Convert each random variable to its pdf form
        X1_dummy = pdf(random_variable_1)
        X2_dummy = pdf(random_variable_2)
        # Create function and support lists for the product of the
        #   two random variables
        prodlist = []
        funclist = []
        for i in range(len(X1_dummy.support)):
            for j in range(len(X2_dummy.support)):
                prodlist.append(X1_dummy.support[i] * X2_dummy.support[j])
                funclist.append(X1_dummy.func[i] * X2_dummy.func[j])
        # Sort the function and support lists for the convolution
        sortlist = list(zip(prodlist, funclist))
        sortlist.sort()
        prodlist2 = []
        funclist2 = []
        for i in range(len(sortlist)):
            prodlist2.append(sortlist[i][0])
            funclist2.append(sortlist[i][1])
        # Remove redundant elements in the support list
        prodlist3 = []
        funclist3 = []
        for i in range(len(prodlist2)):
            if prodlist2[i] not in prodlist3:
                prodlist3.append(prodlist2[i])
                funclist3.append(funclist2[i])
            else:
                funclist3[prodlist3.index(prodlist2[i])] += funclist2[i]
        # Create and return the new random variable
        return RV(funclist3, prodlist3, ["discrete", "pdf"])
