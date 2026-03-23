"""
Transformation procedures extracted from `applpy.rv`.
"""

from sympy import Float, Symbol, diff, limit, oo, simplify, solve, zoo

from .rv import Convert, RV, RVError, cdf, pdf, t, x


def transform(random_variable, gXt):
    """
    Procedure Name: Transform
    Purpose: Compute the transformation of a random variable
                by a a function g(x)
    Arguments:  1. random_variable: A random variable
                2. gX: A transformation in list of two lists format
    Output:     1. The transformation of random_variable
    """

    # Check to make sure support of transform is in ascending order
    for i in range(len(gXt[1]) - 1):
        if gXt[1][i] > gXt[1][i + 1]:
            raise RVError("Transform support is not in ascending order")

    # Convert the RV to its PDF form
    X_dummy = pdf(random_variable)

    # If the distribution is continuous, find and return the transformation
    if random_variable.is_continuous():
        # Adjust the transformation to include the support of the random
        #   variable
        gXold = []
        for i in range(len(gXt)):
            gXold.append(gXt[i])
        gXsupp = []
        for i in range(len(gXold[1])):
            gXsupp.append(gXold[1][i])
        # Add the support of the random variable into the support
        #   of the transformation
        for i in range(len(X_dummy.support)):
            if X_dummy.support[i] not in gXsupp:
                gXsupp.append(X_dummy.support[i])
        gXsupp.sort()
        # Find which segment of the transformation applies, and add it
        #   to the transformation list
        gXfunc = []
        for i in range(1, len(gXsupp)):
            for j in range(len(gXold[0])):
                if gXsupp[i] >= gXold[1][j]:
                    if gXsupp[i] <= gXold[1][j + 1]:
                        gXfunc.append(gXold[0][j])
                        break
        # Set the adjusted transformation as gX
        gX = []
        gX.append(gXfunc)
        gX.append(gXsupp)
        # If the support of the transformation does not match up with the
        #   support of the RV, adjust the support of the transformation

        # Traverse list to find elements that are not within the support
        #   of the rv
        for i in range(len(gX[1])):
            if gX[1][i] < X_dummy.support[0]:
                gX[1][i] = X_dummy.support[0]
            if gX[1][i] > X_dummy.support[len(X_dummy.support) - 1]:
                gX[1][i] = X_dummy.support[len(X_dummy.support) - 1]
        # Delete segments of the transformation that will not be used
        gX0_removal = []
        gX1_removal = []
        for i in range(len(gX[0]) - 1):
            if gX[1][i] == gX[1][i + 1]:
                gX0_removal.append(i)
                gX1_removal.append(i + 1)
        for i in range(len(gX0_removal)):
            index = gX0_removal[i]
            del gX[0][index - i]
        for i in range(len(gX1_removal)):
            index = gX1_removal[i]
            del gX[1][index - i]
        # Create a list of mappings x->g(x)
        mapping = []
        for i in range(len(gX[0])):
            gXsubs1 = gX[0][i].subs(x, gX[1][i])
            if gXsubs1 == zoo:
                gXsubs1 = limit(gX[0][i], x, gX[1][i])
            gXsubs2 = gX[0][i].subs(x, gX[1][i + 1])
            if gXsubs2 == zoo:
                gXsubs2 = limit(gX[0][i + 1], x, gX[1][i + 1])
            mapping.append([gXsubs1, gXsubs2])
        # Create the support for the transformed random variable
        trans_supp = []
        for i in range(len(mapping)):
            for j in range(2):
                if mapping[i][j] not in trans_supp:
                    trans_supp.append(mapping[i][j])
        if zoo in trans_supp:
            error_string = "complex infinity appears in the support, "
            error_string += "please check for an undefined transformation "
            error_string += "such as 1/0"
            raise RVError(error_string)
        trans_supp.sort()
        # Find which segment of the transformation each transformation
        #   function applies to
        applist = []
        for i in range(len(mapping)):
            temp = []
            for j in range(len(trans_supp) - 1):
                if min(mapping[i]) <= trans_supp[j]:
                    if max(mapping[i]) >= trans_supp[j + 1]:
                        temp.append(j)
            applist.append(temp)
        # Find the appropriate inverse for each g(x)
        ginv = []
        for i in range(len(gX[0])):
            # Find the 'test point' for the inverse
            if [gX[1][i], gX[1][i + 1]] == [-oo, oo]:
                c = 0
            elif gX[1][i] == -oo and gX[1][i + 1] != oo:
                c = gX[1][i + 1] - 1
            elif gX[1][i] != -oo and gX[1][i + 1] == oo:
                c = gX[1][i] + 1
            else:
                c = (gX[1][i] + gX[1][i + 1]) / 2
            # Create a list of possible inverses
            invlist = solve(gX[0][i] - t, x)
            # Use the test point to determine the correct inverse
            selected_inverse = False
            for j in range(len(invlist)):
                # If g-1(g(c))=c, then the inverse is correct
                test = invlist[j].subs(t, gX[0][i].subs(x, c))
                if simplify(test - c) == 0:
                    ginv.append(invlist[j])
                    selected_inverse = True
                    break
                try:
                    if test <= Float(float(c), 10) + 0.0000001:
                        if test >= Float(float(c), 10) - 0.0000001:
                            ginv.append(invlist[j])
                            selected_inverse = True
                            break
                except Exception:
                    if j == len(invlist) - 1 and len(ginv) < i + 1:
                        ginv.append(None)
                        selected_inverse = True
            # Some symbolic comparisons do not trigger either branch above.
            # Fall back to the only available inverse when the mapping is
            # unambiguous.
            if not selected_inverse and len(invlist) == 1:
                ginv.append(invlist[0])
        # Find the transformation function for each segment'
        seg_func = []
        for i in range(len(X_dummy.func)):
            # Only find transformation for applicable segments
            for j in range(len(gX[0])):
                if gX[1][j] >= X_dummy.support[i]:
                    if gX[1][j + 1] <= X_dummy.support[i + 1]:
                        if j >= len(ginv) or ginv[j] is None:
                            continue
                        if not isinstance(X_dummy.func[i], (float, int)):
                            tran = X_dummy.func[i].subs(x, ginv[j])
                            tran = tran * diff(ginv[j], t)
                        else:
                            tran = X_dummy.func[i] * diff(ginv[j], t)
                        seg_func.append(tran)
        # Sum the transformations for each piece of the transformed
        #   random variable
        trans_func = []
        for i in range(len(trans_supp) - 1):
            h = 0
            for j in range(len(seg_func)):
                if i in applist[j]:
                    if mapping[j][0] < mapping[j][1]:
                        h = h + seg_func[j]
                    else:
                        h = h - seg_func[j]
            trans_func.append(h)
        # Substitute x into the transformed random variable
        trans_func2 = []
        for i in range(len(trans_func)):
            if not isinstance(trans_func[i], (int, float)):
                trans_func2.append(simplify(trans_func[i].subs(t, x)))
            else:
                trans_func2.append(trans_func[i])
        # Create and return the random variable
        return RV(trans_func2, trans_supp, ["continuous", "pdf"])

    # If the distribution in symbolic discrete, convert it and then compute
    #   the transformation
    if random_variable.is_discrete_functional():
        for element in random_variable.support:
            if (element in [-oo, oo]) or (isinstance(element, Symbol)):
                err_string = "Transform is not implemented for discrete "
                err_string += "random variables with symbolic or inifinite "
                err_string += "support"
                raise RVError(err_string)
        X_dummy = Convert(random_variable)
        return transform(X_dummy, gXt)

    # If the distribution is discrete, find and return the transformation
    if random_variable.is_discrete():
        gX = gXt
        trans_sup = []
        # Find the portion of the transformation each element
        #   in the random variable applies to, and then transform it
        for i in range(len(X_dummy.support)):
            X_support = X_dummy.support[i]
            if X_support < min(gX[1]) or X_support > max(gX[1]):
                trans_sup.append(X_support)
            for j in range(len(gX[1]) - 1):
                if X_support >= gX[1][j] and X_support <= gX[1][j + 1]:
                    trans_sup.append(gX[0][j].subs(x, X_dummy.support[i]))
                    break
                    # Break is required, otherwise points on the boundaries
                    #   between two segments of the transformation will
                    #   be entered twice
        # Sort the function and support lists
        sortlist = list(zip(trans_sup, X_dummy.func))
        sortlist.sort()
        translist = []
        funclist = []
        for i in range(len(sortlist)):
            translist.append(sortlist[i][0])
            funclist.append(sortlist[i][1])
        # Combine redundant elements in the list
        translist2 = []
        funclist2 = []
        for i in range(len(translist)):
            if translist[i] not in translist2:
                translist2.append(translist[i])
                funclist2.append(funclist[i])
            elif translist[i] in translist2:
                idx = translist2.index(translist[i])
                funclist2[idx] += funclist[i]
        # Return the transformed random variable
        return RV(funclist2, translist2, ["discrete", "pdf"])


def truncate(random_variable, supp):
    """
    Procedure Name: Truncate
    Purpose: Truncate a random variable
    Arguments: 1. random_variable: A random variable
               2. supp: The support of the truncated random variable
    Output:    1. A truncated random variable
    """
    # Check to make sure the support of the truncated random
    #   variable is given in ascending order
    if supp[0] > supp[1]:
        raise RVError("The support must be given in ascending order")

    # Conver the random variable to its pdf form
    X_dummy = pdf(random_variable)
    cdf_dummy = cdf(random_variable)

    # If the random variable is continuous, find and return
    #   the truncated random variable
    if random_variable.is_continuous():
        # Find the area of the truncated random variable
        area = cdf(cdf_dummy, supp[1]) - cdf(cdf_dummy, supp[0])
        # Cut out parts of the distribution that don't fall
        #   within the new limits
        for i in range(len(X_dummy.func)):
            if supp[0] >= X_dummy.support[i]:
                if supp[0] <= X_dummy.support[i + 1]:
                    lwindx = i
            if supp[1] >= X_dummy.support[i]:
                if supp[1] <= X_dummy.support[i + 1]:
                    upindx = i
        truncfunc = []
        for i in range(len(X_dummy.func)):
            if i >= lwindx and i <= upindx:
                truncfunc.append(simplify(X_dummy.func[i] / area))
        truncsupp = [supp[0]]
        upindx += 1
        for i in range(len(X_dummy.support)):
            if i > lwindx and i < upindx:
                truncsupp.append(X_dummy.support[i])
        truncsupp.append(supp[1])
        # Return the truncated random variable
        return RV(truncfunc, truncsupp, ["continuous", "pdf"])

    # If the random variable is a discrete function, find and return
    #   the truncated random variable
    if random_variable.is_discrete_functional():
        # Find the area of the truncated random variable
        area = cdf(cdf_dummy, supp[1]) - cdf(cdf_dummy, supp[0])
        # Cut out parts of the distribution that don't fall
        #   within the new limits
        for i in range(len(X_dummy.func)):
            if supp[0] >= X_dummy.support[i]:
                if supp[0] <= X_dummy.support[i + 1]:
                    lwindx = i
            if supp[1] >= X_dummy.support[i]:
                if supp[1] <= X_dummy.support[i + 1]:
                    upindx = i
        truncfunc = []
        for i in range(len(X_dummy.func)):
            if i >= lwindx and i <= upindx:
                truncfunc.append(X_dummy.func[i] / area)
        truncsupp = [supp[0]]
        upindx += 1
        for i in range(len(X_dummy.support)):
            if i > lwindx and i < upindx:
                truncsupp.append(X_dummy.support[i])
        truncsupp.append(supp[1])
        # Return the truncated random variable
        return RV(truncfunc, truncsupp, ["discrete_functional", "pdf"])

    # If the distribution is discrete, find and return the
    #   truncated random variable
    if random_variable.is_discrete():
        # Find the area of the truncated random variable
        area = 0
        for i in range(len(X_dummy.support)):
            if X_dummy.support[i] >= supp[0]:
                if X_dummy.support[i] <= supp[1]:
                    area += X_dummy.func[i]
        # Truncate the random variable and find the probability
        #   at each point
        truncfunc = []
        truncsupp = []
        for i in range(len(X_dummy.support)):
            if X_dummy.support[i] >= supp[0]:
                if X_dummy.support[i] <= supp[1]:
                    truncfunc.append(X_dummy.func[i] / area)
                    truncsupp.append(X_dummy.support[i])
        # Return the truncated random variable
        return RV(truncfunc, truncsupp, ["discrete", "pdf"])


def mixture(MixParameters, MixRVs):
    """
    Procedure Name: Mixture
    Purpose: Mixes random variables X1,X2,...,Xn
    Arguments:   1. MixParameters: A mix of probability weights
                 2. MixRVs: RV's X1,X2,...,Xn
    Output:      1. The mixture RV
    """

    # Check to make sure that the arguments are lists
    if not isinstance(MixParameters, list) or not isinstance(MixRVs, list):
        raise RVError("Both arguments must be in list format")
    # Check to make sure the lists are of equal length
    if len(MixParameters) != len(MixRVs):
        raise RVError("Mix parameter and RV lists must be the same length")
    # Check to ensure that the mix rv's are all of the same type
    #   (discrete or continuous)
    for i in range(len(MixRVs)):
        if MixRVs[0].domain_type != MixRVs[i].domain_type:
            raise RVError("Mix RVs must be all continuous or discrete")
    # Convert the Mix RVs to their PDF form
    Mixfx = []
    for i in range(len(MixRVs)):
        Mixfx.append(pdf(MixRVs[i]))

    # If the distributions are continuous, find and return the
    #   mixture pdf
    if Mixfx[0].is_continuous():
        # Compute the support of the mixture as the union of the supports
        #   of the mix rvs
        MixSupp = []
        for i in range(len(Mixfx)):
            for j in range(len(Mixfx[i].support)):
                if Mixfx[i].support[j] not in MixSupp:
                    MixSupp.append(Mixfx[i].support[j])
        MixSupp.sort()
        # Compute and return the mixed PDF
        fxnew = []
        for i in range(len(MixSupp) - 1):
            newMixfx = 0
            for j in range(len(MixParameters)):
                m = len(Mixfx[j].support) - 1
                for k in range(m):
                    if Mixfx[j].support[k] <= MixSupp[i]:
                        if MixSupp[i + 1] <= Mixfx[j].support[k + 1]:
                            buildfx = Mixfx[j].func[k] * MixParameters[j]
                            newMixfx += buildfx
            simplify(newMixfx)
            fxnew.append(newMixfx)
        # Return the mixture rv
        return RV(fxnew, MixSupp, ["continuous", "pdf"])

    # If the two random variables are discrete in functinonal form,
    #   find and return the mixture of the two random variables
    for i in range(len(Mixfx)):
        if Mixfx[i].is_discrete_functional():
            for num in Mixfx[i].support:
                if not isinstance(num, (int, float)):
                    err_string = "Mixture does not currently work with"
                    err_string = " RVs that have symbolic or infinite support"
                    raise RVError(err_string)
            Mixfx[i] = Convert(Mixfx[i])

    # If the distributions are discrete, find and return the
    #   mixture pdf
    if Mixfx[0].is_discrete():
        # Compute the mixture rv by summing over the weights
        MixSupp = []
        fxnew = []
        for i in range(len(Mixfx)):
            for j in range(len(Mixfx[i].support)):
                if Mixfx[i].support[j] not in MixSupp:
                    MixSupp.append(Mixfx[i].support[j])
                    fxnew.append(Mixfx[i].func[j] * MixParameters[i])
                else:
                    indx = MixSupp.index(Mixfx[i].support[j])
                    val = Mixfx[i].func[j] * MixParameters[i]
                    fxnew[indx] += val
        # Sort the values
        zip_list = list(zip(MixSupp, fxnew))
        zip_list.sort()
        fxnew = []
        MixSupp = []
        for i in range(len(zip_list)):
            fxnew.append(zip_list[i][1])
            MixSupp.append(zip_list[i][0])
        return RV(fxnew, MixSupp, ["discrete", "pdf"])


# Backward-compatible aliases for legacy APPLPy function names.
Transform = transform
Mixture = mixture
Truncate = truncate
