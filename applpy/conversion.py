"""
Functional form conversion utilities for APPLPy random variables.
"""

from applpy_rust import FastRV

from sympy import (
    Symbol,
    symbols,
    oo,
    integrate,
    summation,
    diff,
    exp,
    ln,
    simplify,
    solve,
    Rational,
    Sum,
)

from .rv import RV, RVError

x, y, z, t = symbols("x y z t")


def _convert(random_variable, inc=1):
    # Local import avoids circular import with applpy.transform.
    from .transform import convert

    return convert(random_variable, inc)


# Procedures for changing functional form
#
# Procedures:
#     cdf(random_variable,value)
#     chf(random_variable,value)
#     hf(random_variable,value)
#     idf(random_variable,value)
#     pdf(random_variable,value)
#     sf(random_variable,value)


def cdf(random_variable, value=x, cache=False):
    """
    Procedure Name: cdf
    Purpose: Compute the cdf of a random variable
    Arguments:  1. random_variable: A random variable
                2. value: An integer or floating point number
                3. cache: A binary variable. If True, the result will
                    be stored in memory for later use. (default is False)
    Output:     1. cdf of a random variable (if value not specified)
                2. Value of the cdf at a given point
                    (if value is specified)
    """

    # Check to make sure the value given is within the random
    #   variable's support
    if not isinstance(value, Symbol):
        if value > random_variable.support[-1]:
            return 1
        if value < random_variable.support[0]:
            return 0

    # If the cdf of the random variable is already cached in memory,
    #   retriew the value of the cdf and return in.
    if random_variable.cache is not None and "cdf" in random_variable.cache:
        if value == x:
            return random_variable.cache["cdf"]
        else:
            return cdf(random_variable.cache["cdf"], value)

    # If the distribution is continous, find and return the distribution
    #   of the random variable
    if random_variable.is_continuous():
        # Short-cut for Weibull, jump straight to the closed form cdf
        if "Weibull" in random_variable.__class__.__name__:
            if value == x:
                Fx = RV(random_variable.cdf, [0, oo], ["continuous", "cdf"])
                return Fx
            else:
                return simplify(random_variable.cdf.subs(x, value))

        # If the random variable is already a cdf, nothing needs to
        #   be done
        if random_variable.is_cdf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if (
                        value >= random_variable.support[i]
                        and value <= random_variable.support[i + 1]
                    ):
                        cdfvalue = random_variable.func[i].subs(x, value)
                        return simplify(cdfvalue)
        # If the random variable is a sf, find and return the cdf of the
        #   random variable
        if random_variable.domain_type == "sf":
            X_dummy = sf(random_variable)
            # Compute the sf for each segment
            cdflist = []
            for i in range(len(X_dummy.func)):
                cdflist.append(1 - X_dummy.func[i])
            # If no value is specified, return the sf function
            if value == x:
                cdffunc = RV(cdflist, X_dummy.support, ["continuous", "cdf"])
                if cache:
                    random_variable.add_to_cache("cdf", cdffunc)
                return cdffunc
            # If not, return the value of the cdf at the specified value
            else:
                for i in range(len(X_dummy.support)):
                    if value >= X_dummy.support[i] and value <= X_dummy.support[i + 1]:
                        cdfvalue = cdflist[i].subs(x, value)
                        return simplify(cdfvalue)
        # If the random variable is not a cdf or sf, compute the pdf of
        #   the random variable, and then compute the cdf by integrating
        #   over each segment of the random variable
        else:
            X_dummy = pdf(random_variable)
            # Substitue the dummy variable 't' into the dummy rv
            funclist = []
            for i in range(len(X_dummy.func)):
                number_types = (int, float, Rational)
                if not isinstance(X_dummy.func[i], number_types):
                    newfunc = X_dummy.func[i].subs(x, t)
                else:
                    newfunc = X_dummy.func[i]
                funclist.append(newfunc)
            # Integrate to find the cdf
            cdflist = []
            for i in range(len(funclist)):
                cdffunc = integrate(funclist[i], (t, X_dummy.support[i], x))
                # Adjust the constant of integration
                if i != 0:
                    const = cdflist[i - 1].subs(x, X_dummy.support[i]) - cdffunc.subs(
                        x, X_dummy.support[i]
                    )
                    cdffunc = cdffunc + const
                if i == 0:
                    const = 0 - cdffunc.subs(x, X_dummy.support[i])
                    cdffunc = cdffunc + const
                cdflist.append(simplify(cdffunc))
            # If no value is specified, return the cdf
            if value == x:
                cdffunc = RV(cdflist, X_dummy.support, ["continuous", "cdf"])
                if cache:
                    random_variable.add_to_cache("cdf", cdffunc)
                return cdffunc
            # If a value is specified, return the value of the cdf
            if value != x:
                for i in range(len(random_variable.support)):
                    if (
                        value >= random_variable.support[i]
                        and value <= random_variable.support[i + 1]
                    ):
                        cdfvalue = cdflist[i].subs(x, value)
                        return simplify(cdfvalue)

    # If the distribution is in discrete functional, find and return the
    #   distribution of the random variable
    if random_variable.is_discrete_functional():
        # If the support is finite, then convert to expanded form and compute
        #   the cdf
        if oo not in random_variable.support:
            if -oo not in random_variable.support:
                random_variable_2 = _convert(random_variable)
                return cdf(random_variable_2, value)
        # If the random variable is already a cdf, nothing needs to
        #   be done
        if random_variable.is_cdf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if (
                        value >= random_variable.support[i]
                        and value <= random_variable.support[i + 1]
                    ):
                        cdfvalue = random_variable.func[i].subs(x, value)
                        return simplify(cdfvalue)
        # If the random variable is a sf, find and return the cdf of the
        #   random variable
        if random_variable.domain_type == "sf":
            X_dummy = sf(random_variable)
            # Compute the sf for each segment
            cdflist = []
            for i in range(len(X_dummy.func)):
                cdflist.append(1 - X_dummy.func[i])
            # If no value is specified, return the sf function
            if value == x:
                cdffunc = RV(cdflist, X_dummy.support, ["discrete_functional", "cdf"])
                if cache:
                    random_variable.add_to_cache("cdf", cdffunc)
                return cdffunc
            # If not, return the value of the cdf at the specified value
            else:
                for i in range(len(X_dummy.support)):
                    if value >= X_dummy.support[i] and value <= X_dummy.support[i + 1]:
                        cdfvalue = cdflist[i].subs(x, value)
                        return simplify(cdfvalue)
        # If the random variable is not a cdf or sf, compute the pdf of
        #   the random variable, and then compute the cdf by summing
        #   over each segment of the random variable
        else:
            X_dummy = pdf(random_variable)
            # Substitue the dummy variable 't' into the dummy rv
            funclist = []
            for i in range(len(X_dummy.func)):
                newfunc = X_dummy.func[i].subs(x, t)
                funclist.append(newfunc)
            # Sum to find the cdf
            cdflist = []
            for i in range(len(funclist)):
                cdffunc = Sum(funclist[i], (t, X_dummy.support[i], x)).doit()
                # Adjust the constant of integration
                if i != 0:
                    const = cdflist[i - 1].subs(x, X_dummy.support[i]) - cdffunc.subs(
                        x, X_dummy.support[i]
                    )
                    # cdffunc=cdffunc+const
                if i == 0:
                    const = 0 - cdffunc.subs(x, X_dummy.support[i])
                    # cdffunc=cdffunc+const
                cdflist.append(simplify(cdffunc))
            # If no value is specified, return the cdf
            if value == x:
                cdffunc = RV(cdflist, X_dummy.support, ["discrete_functional", "cdf"])
                if cache:
                    random_variable.add_to_cache("cdf", cdffunc)
                return cdffunc
            # If a value is specified, return the value of the cdf
            if value != x:
                for i in range(len(random_variable.support)):
                    if (
                        value >= random_variable.support[i]
                        and value <= random_variable.support[i + 1]
                    ):
                        cdfvalue = cdflist[i].subs(x, value)
                        return simplify(cdfvalue)

    if random_variable.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type="discrete",
        )
        cdf_random_variable = fast_rv.to_cdf()

        if value != x:
            return cdf_random_variable.evaluate(value)

        return RV(
            func=cdf_random_variable.function,
            support=cdf_random_variable.support,
            functional_form=cdf_random_variable.functional_form,
            domain_type="discrete",
        )


def chf(random_variable, value=x, cache=False):
    """
    Procedure Name: chf
    Purpose: Compute the chf of a random variable
    Arguments:  1. random_variable: A random variable
                2. value: An integer or floating point number
                    (optional)
    Output:     1. chf of a random variable (if value not specified)
                2. Value of the chf at a given point
                    (if value is specified)
    """

    # Check to make sure the value given is within the random
    #   variable's support
    if not isinstance(value, Symbol):
        if value > random_variable.support[-1] or value < random_variable.support[0]:
            if not random_variable.is_discrete():
                string = "Value is not within the support of the random variable"
                raise RVError(string)

    # If the chf of the random variable is already cached in memory,
    #   retriew the value of the chf and return in.
    if random_variable.cache is not None and "chf" in random_variable.cache:
        if value == x:
            return random_variable.cache["chf"]
        else:
            return chf(random_variable.cache["chf"], value)

    # If the distribution is continuous, find and return the chf of
    #   the random variable
    if random_variable.is_continuous():
        # If the distribution is already a chf, nothing needs to
        #   be done
        if random_variable.is_chf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.support[i]:
                        if value <= random_variable.support[i + 1]:
                            chfvalue = random_variable.func[i].subs(x, value)
                            return simplify(chfvalue)
        # Otherwise, find and return the chf
        else:
            X_dummy = sf(random_variable)
            # Generate a list of sf functions
            sflist = []
            for i in range(len(X_dummy.func)):
                sflist.append(X_dummy.func[i])
            # Generate chf functions
            chffunc = []
            for i in range(len(sflist)):
                newfunc = -ln(sflist[i])
                chffunc.append(simplify(newfunc))
            # If a value is not specified, return the chf of the
            #   random variable
            if value == x:
                chffunc = RV(chffunc, X_dummy.support, ["continuous", "chf"])
                if cache:
                    random_variable.add_to_cache("chf", chffunc)
                return chffunc
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.func[i]:
                        if value <= random_variable.support[i + 1]:
                            chfvalue = chffunc[i].subs(x, value)
                            return simplify(chfvalue)

    # If the distribution is a discrete function, find and return the chf of
    #   the random variable
    if random_variable.is_discrete_functional():
        # If the distribution is already a chf, nothing needs to
        #   be done
        if random_variable.is_chf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.support[i]:
                        if value <= random_variable.support[i + 1]:
                            chfvalue = random_variable.func[i].subs(x, value)
                            return simplify(chfvalue)
        # If the support is finite, then convert to expanded form and compute
        #   the chf
        if oo not in random_variable.support:
            if -oo not in random_variable.support:
                random_variable_2 = _convert(random_variable)
                return chf(random_variable_2, value)
        # Otherwise, find and return the chf
        else:
            X_dummy = sf(random_variable)
            # Generate a list of sf functions
            sflist = []
            for i in range(len(X_dummy.func)):
                sflist.append(X_dummy.func[i])
            # Generate chf functions
            chffunc = []
            for i in range(len(sflist)):
                newfunc = -ln(sflist[i])
                chffunc.append(simplify(newfunc))
            # If a value is not specified, return the chf of the
            #   random variable
            if value == x:
                chfrv = RV(chffunc, X_dummy.support, ["discrete_functional", "chf"])
                if cache:
                    random_variable.add_to_cache("chf", chfrv)
                return chfrv
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.func[i]:
                        if value <= random_variable.support[i + 1]:
                            chfvalue = chffunc[i].subs(x, value)
                            return simplify(chfvalue)

    if random_variable.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type="discrete",
        )
        chf_random_variable = fast_rv.to_chf()

        if value != x:
            return chf_random_variable.evaluate(value)

        return RV(
            func=chf_random_variable.function,
            support=chf_random_variable.support,
            functional_form=chf_random_variable.functional_form,
            domain_type="discrete",
        )


def hf(random_variable, value=x, cache=False):
    """
    Procedure Name: hf
    Purpose: Compute the hf of a random variable
    Arguments:  1. random_variable: A random variable
                2. value: An integer or floating point number
                    (optional)
    Output:     1. hf of a random variable (if value not specified)
                2. Value of the hf at a given point
                    (if value is specified)
    """

    # Check to make sure the value given is within the random
    #   variable's support
    if not isinstance(value, Symbol):
        if value > random_variable.support[-1] or value < random_variable.support[0]:
            if not random_variable.is_discrete():
                string = "Value is not within the support of the random variable"
                raise RVError(string)

    # If the hf of the random variable is already cached in memory,
    #   retriew the value of the hf and return in.
    if random_variable.cache is not None and "hf" in random_variable.cache:
        if value == x:
            return random_variable.cache["hf"]
        else:
            return hf(random_variable.cache["hf"], value)

    # If the distribution is continuous, find and return the hf of
    #   the random variable
    if random_variable.is_continuous():
        # If the distribution is already a hf, nothing needs to be
        #   done
        if random_variable.is_hf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.support[i]:
                        if value <= random_variable.support[i + 1]:
                            hfvalue = random_variable.func[i].subs(x, value)
                            return simplify(hfvalue)
        # If the distribution is in chf form, use differentiation
        #   to find the hf
        if random_variable.is_chf():
            X_dummy = chf(random_variable)
            # Generate a list of hf functions
            hflist = []
            for i in range(len(X_dummy.func)):
                newfunc = diff(X_dummy.func[i], x)
                hflist.append(newfunc)
            if value == x:
                hfrv = RV(hflist, X_dummy.support, ["continuous", "hf"])
                if cache:
                    random_variable.add_to_cache("hf", hfrv)
                return hfrv
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.support[i]:
                        if value <= random_variable.support[i + 1]:
                            hfvalue = hflist[i].subs(x, value)
                            return simplify(hfvalue)
        # In all other cases, use the pdf and the sf to find the hf
        else:
            X_pdf = pdf(random_variable).func
            X_sf = sf(random_variable).func
            # Create a list of hf functions
            hflist = []
            for i in range(len(random_variable.func)):
                hfunc = (X_pdf[i]) / (X_sf[i])
                hflist.append(simplify(hfunc))
            if value == x:
                hfrv = RV(hflist, random_variable.support, ["continuous", "hf"])
                if cache:
                    random_variable.add_to_cache("hf", hfrv)
                return hfrv
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.support[i]:
                        if value <= random_variable.support[i + 1]:
                            hfvalue = hflist[i].subs(x, value)
                            return simplify(hfvalue)

    # If the distribution is a discrete function, find and return the hf of
    #   the random variable
    if random_variable.is_discrete_functional():
        # If the distribution is already a hf, nothing needs to be
        #   done
        if random_variable.is_hf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.support[i]:
                        if value <= random_variable.support[i + 1]:
                            hfvalue = random_variable.func[i].subs(x, value)
                            return simplify(hfvalue)
        # If the support is finite, then convert to expanded form and compute
        #   the hf
        if oo not in random_variable.support:
            if -oo not in random_variable.support:
                random_variable_2 = _convert(random_variable)
                return hf(random_variable_2, value)
        # In all other cases, use the pdf and the sf to find the hf
        else:
            X_pdf = pdf(random_variable).func
            X_sf = sf(random_variable).func
            # Create a list of hf functions
            hflist = []
            for i in range(len(random_variable.func)):
                hfunc = (X_pdf[i]) / (X_sf[i])
                hflist.append(simplify(hfunc))
            if value == x:
                hfrv = RV(hflist, random_variable.support, ["discrete_functional", "hf"])
                if cache:
                    random_variable.add_to_cache("hf", hfrv)
                return hfrv
            if value != x:
                for i in range(len(random_variable.support)):
                    if value >= random_variable.support[i]:
                        if value <= random_variable.support[i + 1]:
                            hfvalue = hflist[i].subs(x, value)
                            return simplify(hfvalue)

    if random_variable.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type="discrete",
        )
        hf_random_variable = fast_rv.to_hf()

        if value != x:
            return hf_random_variable.evaluate(value)

        return RV(
            func=hf_random_variable.function,
            support=hf_random_variable.support,
            functional_form=hf_random_variable.functional_form,
            domain_type="discrete",
        )


def idf(random_variable, value=x, cache=False):
    """
    Procedure Name: idf
    Purpose: Compute the idf of a random variable
    Arguments:  1. random_variable: A random variable
                2. value: An integer or floating point number
                    (optional)
    Output:     1. idf of a random variable (if value not specified)
                2. Value of the idf at a given point
                    (if value is specified)
    """

    # Check to make sure the value given is within the random
    #   variable's support
    if not isinstance(value, Symbol):
        if value > 1 or value < 0:
            if not random_variable.is_discrete():
                string = "Value is not within the support of the random variable"
                raise RVError(string)

    # If the idf of the random variable is already cached in memory,
    #   retriew the value of the idf and return in.
    if random_variable.cache is not None and "idf" in random_variable.cache:
        if value == x:
            return random_variable.cache["idf"]
        else:
            return idf(random_variable.cache["idf"], value)

    # If the distribution is continuous, find and return the idf
    #   of the random variable
    if random_variable.is_continuous():
        if value == x:
            if random_variable.is_idf():
                return random_variable
            # Convert the random variable to its cdf form
            X_dummy = cdf(random_variable)
            # Create values used to check for correct inverse
            check = []
            for i in range(len(X_dummy.support) - 1):
                if X_dummy.support[i] == -oo and X_dummy.support[i + 1] == oo:
                    check.append(0)
                elif X_dummy.support[i] == -oo and X_dummy.support[i + 1] != oo:
                    check.append(X_dummy.support[i + 1] - 1)
                elif X_dummy.support[i] != -oo and X_dummy.support[i + 1] == oo:
                    check.append(X_dummy.support[i] + 1)
                else:
                    check.append((X_dummy.support[i] + X_dummy.support[i + 1]) / 2)
            # Use solve to create a list of candidate inverse functions
            # Check to see which of the candidate inverse functions is correct
            idffunc = []
            for i in range(len(X_dummy.func)):
                invlist = solve(X_dummy.func[i] - t, x)
                if len(invlist) == 1:
                    idffunc.append(invlist[0])
                else:
                    # The flag is used to determine if two separate inverses
                    #   could represent the inverse of the cdf. If this is the
                    #   case, an exception is raised
                    flag = False
                    for j in range(len(invlist)):
                        subsfunc = X_dummy.func[i]
                        val = invlist[j].subs(t, subsfunc.subs(x, check[i])).evalf()
                        if abs(val - check[i]) < 0.00001:
                            if flag:
                                error_string = "Could not find the"
                                error_string += " correct inverse"
                                raise RVError(error_string)
                            idffunc.append(invlist[j])
                            flag = True
            # Create a list of supports for the idf
            idfsup = []
            for i in range(len(X_dummy.support)):
                idfsup.append(cdf(X_dummy, X_dummy.support[i]))
            # Replace t with x
            idffunc2 = []
            for i in range(len(idffunc)):
                func = idffunc[i].subs(t, x)
                idffunc2.append(simplify(func))
            # Return the idf
            idfrv = RV(idffunc2, idfsup, ["continuous", "idf"])
            if cache:
                random_variable.add_to_cache("idf", idfrv)
            return idfrv

        # If a value is specified, return the value of the idf at x=value
        if value != x:
            X_dummy = idf(random_variable)
            for i in range(len(X_dummy.support)):
                if value >= X_dummy.support[i] and value <= X_dummy.support[i + 1]:
                    idfvalue = X_dummy.func[i].subs(x, value)
                    return simplify(idfvalue)

    # If the distribution is a discrete function, find and return the idf
    #   of the random variable
    if random_variable.is_discrete_functional():
        # If the support is finite, then convert to expanded form and compute
        #   the idf
        if oo not in random_variable.support:
            if -oo not in random_variable.support:
                random_variable_2 = _convert(random_variable)
                return idf(random_variable_2, value)
        if value == x:
            if random_variable.is_idf():
                return random_variable
            # Convert the random variable to its cdf form
            X_dummy = cdf(random_variable)
            # Create values used to check for correct inverse
            check = []
            for i in range(len(X_dummy.support) - 1):
                if X_dummy.support[i] == -oo and X_dummy.support[i + 1] == oo:
                    check.append(0)
                elif X_dummy.support[i] == -oo and X_dummy.support[i + 1] != oo:
                    check.append(X_dummy.support[i + 1] - 1)
                elif X_dummy.support[i] != -oo and X_dummy.support[i + 1] == oo:
                    check.append(X_dummy.support[i] + 1)
                else:
                    check.append((X_dummy.support[i] + X_dummy.support[i + 1]) / 2)
            # Use solve to create a list of candidate inverse functions
            # Check to see which of the candidate inverse functions is correct
            idffunc = []
            for i in range(len(X_dummy.func)):
                invlist = solve(X_dummy.func[i] - t, x)
                if len(invlist) == 1:
                    idffunc.append(invlist[0])
                else:
                    # The flag is used to determine if two separate inverses
                    #   could represent the inverse of the cdf. If this is the
                    #   case, an exception is raised
                    flag = False
                    for j in range(len(invlist)):
                        subsfunc = X_dummy.func[i]
                        val = invlist[j].subs(t, subsfunc.subs(x, check[i])).evalf()
                        if abs(val - check[i]) < 0.00001:
                            if flag:
                                error_string = "Could not find the correct"
                                error_string += " inverse"
                                raise RVError(error_string)
                            idffunc.append(invlist[j])
                            flag = True
            # Create a list of supports for the idf
            idfsup = []
            for i in range(len(X_dummy.support)):
                idfsup.append(cdf(X_dummy, X_dummy.support[i]))
            # Replace t with x
            idffunc2 = []
            for i in range(len(idffunc)):
                func = idffunc[i].subs(t, x)
                idffunc2.append(simplify(func))
            # Return the idf
            idfsup[0] = 0
            idfrv = RV(idffunc2, idfsup, ["discrete_functional", "idf"])
            if cache:
                random_variable.add_to_cache("idf", idfrv)
            return idfrv

        # If a value is specified, find thevalue of the idf
        if value != x:
            X_dummy = idf(random_variable)
            for i in range(len(X_dummy.support)):
                if value >= X_dummy.support[i] and value <= X_dummy.support[i + 1]:
                    idfvalue = X_dummy.func[i].subs(x, value)
                    return simplify(idfvalue)

    if random_variable.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type="discrete",
        )
        idf_random_variable = fast_rv.to_idf()

        if value != x:
            return idf_random_variable.evaluate(value)

        return RV(
            func=idf_random_variable.function,
            support=idf_random_variable.support,
            functional_form=idf_random_variable.functional_form,
            domain_type="discrete",
        )


def pdf(random_variable, value=x, cache=False):
    """
    Procedure Name: pdf
    Purpose: Compute the pdf of a random variable
    Arguments:  1. random_variable: A random variable
                2. value: An integer or floating point number (optional)
    Output:     1. pdf of a random variable (if value not specified)
                2. Value of the pdf at a given point (if value is specified)
    """

    # Check to make sure the value given is within the random
    #   variable's support
    if not isinstance(value, Symbol):
        if value > random_variable.support[-1] or value < random_variable.support[0]:
            if not random_variable.is_discrete():
                string = "Value is not within the support of the random variable"
                raise RVError(string)

    # If the pdf of the random variable is already cached in memory,
    #   retriew the value of the pdf and return in.
    if random_variable.cache is not None and "pdf" in random_variable.cache:
        if value == x:
            return random_variable.cache["pdf"]
        else:
            return pdf(random_variable.cache["pdf"], value)

    # If the distribution is continuous, find and return the pdf of the
    # random variable
    if random_variable.is_continuous():
        # If the distribution is already a pdf, nothing needs to be done
        if random_variable.is_pdf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if (
                        value >= random_variable.support[i]
                        and value <= random_variable.support[i + 1]
                    ):
                        pdfvalue = random_variable.func[i].subs(x, value)
                        return simplify(pdfvalue)
        # If the distribution is a hf or chf, use integration to find the pdf
        if random_variable.is_hf() or random_variable.is_chf():
            X_dummy = hf(random_variable)
            # Substitute the dummy variable 't' into the hazard function
            hfsubslist = []
            for i in range(len(X_dummy.func)):
                newfunc = X_dummy.func[i].subs(x, t)
                hfsubslist.append(newfunc)
            # Integrate the hazard function
            intlist = []
            for i in range(len(hfsubslist)):
                newfunc = integrate(hfsubslist[i], (t, X_dummy.support[i], x))
                # Correct the constant of integration
                if i != 0:
                    const = intlist[i - 1].subs(x, X_dummy.support[i])
                    const = const - newfunc.subs(x, X_dummy.support[i])
                    newfunc = newfunc + const
                if i == 0:
                    const = 0 - newfunc.subs(x, X_dummy.support[i])
                    newfunc = newfunc + const
                intlist.append(simplify(newfunc))
            # Multiply to find the pdf
            pdffunc = []
            for i in range(len(intlist)):
                newfunc = X_dummy.func[i] * exp(-intlist[i])
                pdffunc.append(simplify(newfunc))
            if value == x:
                pdfrv = RV(pdffunc, random_variable.support, ["continuous", "pdf"])
                if cache:
                    random_variable.add_to_cache("pdf", pdfrv)
                return pdfrv
            if value != x:
                for i in range(len(X_dummy.support)):
                    if value >= X_dummy.support[i]:
                        if value <= X_dummy.support[i + 1]:
                            pdfvalue = pdffunc[i].subs(x, value)
                            return simplify(pdfvalue)
        # In all other cases, find the pdf by differentiating the cdf
        else:
            X_dummy = cdf(random_variable)
            if value == x:
                pdflist = []
                for i in range(len(X_dummy.func)):
                    pdflist.append(diff(X_dummy.func[i], x))
                pdfrv = RV(pdflist, random_variable.support, ["continuous", "pdf"])
                if cache:
                    random_variable.add_to_cache("pdf", pdfrv)
                return pdfrv
            if value != x:
                for i in range(len(X_dummy.support)):
                    for i in range(len(X_dummy.support)):
                        if value >= X_dummy.support[i]:
                            if value <= X_dummy.support[i + 1]:
                                pdffunc = diff(X_dummy.func[i], x)
                                pdfvalue = pdffunc.subs(x, value)
                                return simplify(pdfvalue)

    # If the distribution is a discrete function, find and return the pdf
    if random_variable.is_discrete_functional():
        # If the distribution is already a pdf, nothing needs to be done
        if random_variable.is_pdf():
            if value == x:
                return random_variable
            if value != x:
                for i in range(len(random_variable.support)):
                    if (
                        value >= random_variable.support[i]
                        and value <= random_variable.support[i + 1]
                    ):
                        pdfvalue = random_variable.func[i].subs(x, value)
                        return simplify(pdfvalue)
        # If the support is finite, then convert to expanded form and compute
        #   the pdf
        if oo not in random_variable.support:
            if -oo not in random_variable.support:
                random_variable_2 = _convert(random_variable)
                return pdf(random_variable_2, value)
        # If the distribution is a hf or chf, use summation to find the pdf
        if random_variable.is_hf() or random_variable.is_chf():
            X_dummy = hf(random_variable)
            # Substitute the dummy variable 't' into the hazard function
            hfsubslist = []
            for i in range(len(X_dummy.func)):
                newfunc = X_dummy.func[i].subs(x, t)
                hfsubslist.append(newfunc)
            # Integrate the hazard function
            sumlist = []
            for i in range(len(hfsubslist)):
                newfunc = summation(hfsubslist[i], (t, X_dummy.support[i], x))
                # Correct the constant of integration
                if i != 0:
                    const = sumlist[i - 1].subs(x, X_dummy.support[i])
                    const = const - newfunc.subs(x, X_dummy.support[i])
                    newfunc = newfunc + const
                if i == 0:
                    const = 0 - newfunc.subs(x, X_dummy.support[i])
                    newfunc = newfunc + const
                intlist.append(simplify(newfunc))
            # Multiply to find the pdf
            pdffunc = []
            for i in range(len(intlist)):
                newfunc = X_dummy.func[i] * exp(-sumlist[i])
                pdffunc.append(simplify(newfunc))
            if value == x:
                pdfrv = RV(pdffunc, random_variable.support, ["discrete_functional", "pdf"])
                if cache:
                    random_variable.add_to_cache("pdf", pdfrv)
                return pdfrv
            if value != x:
                for i in range(len(X_dummy.support)):
                    if value >= X_dummy.support[i] and value <= X_dummy.support[i + 1]:
                        pdfvalue = pdffunc[i].subs(x, value)
                        return simplify(pdfvalue)
        # In all other cases, find the pdf by differentiating the cdf
        else:
            X_dummy = cdf(random_variable)
            if value == x:
                pdflist = []
                # Find the pmf by subtracting cdf(X,x)-cdf(X,x-1)
                for i in range(len(X_dummy.func)):
                    funcX1 = X_dummy.func[i]
                    funcX0 = X_dummy.func[i].subs(x, x - 1)
                    pmf = simplify(funcX1 - funcX0)
                    pdflist.append(pmf)
                pdfrv = RV(pdflist, random_variable.support, ["discrete_functional", "pdf"])
                if cache:
                    random_variable.add_to_cache("pdf", pdfrv)
                return pdfrv
            if value != x:
                for i in range(len(X_dummy.support)):
                    for i in range(len(X_dummy.support)):
                        if value >= X_dummy.support[i]:
                            if value <= X_dummy.support[i + 1]:
                                funcX1 = X_dummy.func[i]
                                funcX0 = X_dummy.func[i].subs(x, x - 1)
                                pmf = simplify(funcX1 - funcX0)
                                pdfvalue = pmf.subs(x, value)
                                return simplify(pdfvalue)

    if random_variable.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type="discrete",
        )
        pdf_random_variable = fast_rv.to_pdf()

        if value != x:
            return pdf_random_variable.evaluate(value)

        return RV(
            func=pdf_random_variable.function,
            support=pdf_random_variable.support,
            functional_form=pdf_random_variable.functional_form,
            domain_type="discrete",
        )


def sf(random_variable, value=x, cache=False):
    """
    Procedure Name: sf
    Purpose: Compute the sf of a random variable
    Arguments:  1. random_variable: A random variable
                2. value: An integer or floating point number (optional)
    Output:     1. sf of a random variable (if value not specified)
                2. Value of the sf at a given point (if value is specified)
    """

    # Check to make sure the value given is within the random
    #   variable's support
    if not isinstance(value, Symbol):
        if value > random_variable.support[-1]:
            return 0
        if value < random_variable.support[0]:
            return 1

    # If the sf of the random variable is already cached in memory,
    #   retriew the value of the sf and return in.
    if random_variable.cache is not None and "sf" in random_variable.cache:
        if value == x:
            return random_variable.cache["sf"]
        else:
            return sf(random_variable.cache["sf"], value)

    # If the distribution is continuous, find and return the sf of the
    # random variable
    if random_variable.is_continuous():
        # If the distribution is already a sf, nothing needs to be done
        if random_variable.is_sf():
            if value == x:
                return random_variable
            else:
                return 1 - cdf(random_variable, value)
        # If not, then use subtraction to find the sf
        else:
            X_dummy = cdf(random_variable)
            # Compute the sf for each segment
            sflist = []
            for i in range(len(X_dummy.func)):
                sflist.append(1 - X_dummy.func[i])
            if value == x:
                sfrv = RV(sflist, random_variable.support, ["continuous", "sf"])
                if cache:
                    random_variable.add_to_cache("sf", sfrv)
                return sfrv
            if value != x:
                return 1 - cdf(random_variable, value)

    # If the distribution is discrete, find and return the sf of the
    # random variable
    if random_variable.is_discrete_functional():
        if oo not in random_variable.support:
            if -oo not in random_variable.support:
                random_variable_2 = _convert(random_variable)
                return sf(random_variable_2, value)
        # If the distribution is already a sf, nothing needs to be done
        if random_variable.is_sf():
            if value == x:
                return random_variable
            else:
                return 1 - cdf(random_variable, value)
        # If not, then use subtraction to find the sf
        else:
            X_dummy = cdf(random_variable)
            # Compute the sf for each segment
            sflist = []
            for i in range(len(X_dummy.func)):
                sflist.append(1 - X_dummy.func[i])
            if value == x:
                sfrv = RV(sflist, random_variable.support, ["continuous", "sf"])
                if cache:
                    random_variable.add_to_cache("sf", sfrv)
                return sfrv
            if value != x:
                return 1 - cdf(random_variable, value)

    if random_variable.is_discrete_functional():
        # If the distribution is already a sf, nothing needs to be done
        if random_variable.is_sf():
            if value == x:
                return random_variable
            else:
                return 1 - cdf(random_variable, value)
        # If not, then use subtraction to find the sf
        else:
            X_dummy = cdf(random_variable)
            # Compute the sf for each segment
            sflist = []
            for i in range(len(X_dummy.func)):
                sflist.append(1 - X_dummy.func[i])
            if value == x:
                sfrv = RV(sflist, random_variable.support, ["discrete_functional", "sf"])
                if cache:
                    random_variable.add_to_cache("sf", sfrv)
                return sfrv
            if value != x:
                return 1 - cdf(random_variable, value)

    if random_variable.is_discrete():
        fast_rv = FastRV(
            function=random_variable.func,
            support=random_variable.support,
            functional_form=random_variable.functional_form,
            domain_type="discrete",
        )
        sf_random_variable = fast_rv.to_sf()

        if value != x:
            return sf_random_variable.evaluate(value)

        return RV(
            func=sf_random_variable.function,
            support=sf_random_variable.support,
            functional_form=sf_random_variable.functional_form,
            domain_type="discrete",
        )


# Backward-compatible aliases
CDF = cdf
CHF = chf
HF = hf
IDF = idf
PDF = pdf
SF = sf
