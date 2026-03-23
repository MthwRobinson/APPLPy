import numpy as np
from matplotlib.pylab import arange, grid, ion, plot, step, title, xlabel, ylabel
from sympy import oo, symbols

x, y, z, t, v = symbols("x y z t v")


def mat_plot(funclist, suplist, lab1=None, lab2=None, ftype="continuous"):
    """
    Procedure Name: mat_plot
    Purpose: Create a matplotlib plot of a random variable
    Arguments:  1. RVar: A random variable
                2. suplist: The support of the plot
    Output:     1. A plot of the random variable
    """
    # if the random variable is continuous, plot the function
    if ftype == "continuous":
        for i in range(len(funclist)):
            if funclist[i] == "0":
                continue
            if "x" in funclist[i]:
                x = arange(suplist[i], suplist[i + 1], 0.01)
                s = eval(funclist[i])
                plot(x, s, linewidth=1.0, color="green")
            else:
                plot(
                    [suplist[i], suplist[i + 1]],
                    [funclist[i], funclist[i]],
                    linewidth=1.0,
                    color="green",
                )
        if lab1 == "idf":
            xlabel("s")
        else:
            xlabel("x")
        if lab1 is not None:
            ylabel(lab1)
        if lab2 is not None:
            title(lab2)
        grid(True)
    # If the random variable is discrete, plot the function
    if ftype == "discrete":
        plot(suplist, funclist, "ro")
        if lab1 == "F-1(s)":
            xlabel("s")
        else:
            xlabel("x")
        if lab1 is not None:
            ylabel(lab1)
        if lab2 is not None:
            title(lab2)
        grid(True)


def prob_plot(Sample, Fitted, plot_type):
    """
    Procedure Name: prob_plot
    Purpose: Create a mat plot lib plot to compare sample distributions
                with theoretical models
    Arguments:  1. Sample: Data sample quantiles
                2. Model: Model quantiles
    Output:     1. A probability plot that compares data with a model
    """
    plot(Fitted, Sample, "ro")
    x = arange(min(min(Sample), min(Fitted)), max(max(Sample), max(Fitted)), 0.01)
    s = x
    plot(x, s, linewidth=1.0, color="red")
    if plot_type == "QQ Plot":
        xlabel("Model Quantiles")
        ylabel("Sample Quantiles")
    elif plot_type == "PP Plot":
        xlabel("Model CDF")
        ylabel("Sample CDF")
    title(plot_type)
    grid(True)


def plot_dist(random_variable, suplist=None, opt=None, color="r", display=True):
    """
    Procedure: plot_dist
    Purpose: Plot a random variable
    Arguments:  1. random_variable: A random variable
                2. suplist: A list of supports for the plot
    Output:     1. A plot of the random variable
    """
    # Local import to avoid circular import with applpy.rv.
    from applpy.conversion import pdf
    from applpy.rv import RV, RVError
    from applpy.transform import Convert

    # Keep argument for backward compatibility.
    _ = display

    if random_variable.is_cdf():
        lab2 = "Cumulative Distribution Function"
    elif random_variable.is_chf():
        lab2 = "Cumulative Hazard Function"
    elif random_variable.is_hf():
        lab2 = "Hazard Function"
    elif random_variable.is_idf():
        lab2 = "Inverse Density Function"
    elif random_variable.is_pdf():
        lab2 = "Probability Density Function"
    elif random_variable.is_sf():
        lab2 = "Survivor Function"
    else:
        lab2 = None

    if opt == "EMPCDF":
        lab2 = "Empirical CDF"

    if random_variable.is_continuous():
        if suplist is not None:
            if suplist[0] > suplist[1]:
                raise RVError("Support list must be in ascending order")
            if suplist[0] < random_variable.support[0]:
                raise RVError("Plot supports must fall within RV support")
            if suplist[1] > random_variable.support[1]:
                raise RVError("Plot support must fall within RV support")

        if suplist is None:
            if random_variable.support[0] == -oo:
                support1 = float(random_variable.variate(s=0.01)[0])
            else:
                support1 = float(random_variable.support[0])
            if random_variable.support[-1] == oo:
                support2 = float(random_variable.variate(s=0.99)[0])
            else:
                support2 = float(random_variable.support[-1])
            suplist = [support1, support2]

        for i in range(len(random_variable.func)):
            if suplist[0] >= float(random_variable.support[i]):
                if suplist[0] <= float(random_variable.support[i + 1]):
                    lwindx = i
            if suplist[1] >= float(random_variable.support[i]):
                if suplist[1] <= float(random_variable.support[i + 1]):
                    upindx = i

        plotfunc = []
        for i in range(len(random_variable.func)):
            if i >= lwindx and i <= upindx:
                plotfunc.append(random_variable.func[i])

        plotsupp = [suplist[0]]
        upindx += 1
        for i in range(len(random_variable.support)):
            if i > lwindx and i < upindx:
                plotsupp.append(random_variable.support[i])
        plotsupp.append(suplist[1])

        for i, function in enumerate(plotfunc):

            def f(y):
                return function.subs(x, y).evalf()

            x_range = np.arange(
                plotsupp[i], plotsupp[i + 1], abs(plotsupp[i + 1] - plotsupp[i]) / 1000
            )
            y_range = np.array([f(num) for num in x_range])
            plot(x_range, y_range, color)
        title(lab2)

    if random_variable.is_discrete() or random_variable.is_discrete_functional():
        if random_variable.is_discrete_functional():
            if random_variable.support[-1] != oo:
                random_variable = Convert(random_variable)
            else:
                p = 1
                i = random_variable.support[0]
                while p > 0.00001:
                    p = pdf(random_variable, i).evalf()
                    i += 1
                newsupport = random_variable.support
                newsupport[-1] = i
                random_variable = RV(
                    random_variable.func,
                    newsupport,
                    functional_form=random_variable.functional_form,
                    domain_type=random_variable.domain_type,
                )
                random_variable = Convert(random_variable)
        step(random_variable.support, random_variable.func)
        if lab2 is not None:
            title(lab2)


def plot_emp_cdf(data):
    """
    Procedure Name: plot_emp_cdf
    Purpose: Plots an empirical CDF, given a data set
    Arguments:  1. data: A data sample
    Output:     1. An empirical cdf of the data
    """
    from applpy.conversion import cdf
    from applpy.rv import bootstrap_rv

    xstar = bootstrap_rv(data)
    plot_dist(cdf(xstar), opt="EMPCDF")


def pp_plot(random_variable, sample):
    """
    Procedure Name: pp_plot
    Purpose: Plots the model probability versus the sample
                probability
    Arguments:  1. random_variable: A random variable
                2. sample: An experimental sample
    Output:     1. A pp_plot comparing the sample to a theoretical
                    model
    """
    from applpy.conversion import cdf
    from applpy.rv import bootstrap_rv, RVError

    if not isinstance(sample, list):
        raise RVError("The data sample must be given as a list")

    n = len(sample)
    sample = sorted(sample)
    plist = []
    for i in range(1, n + 1):
        p = (i - (1 / 2)) / n
        plist.append(p)

    fx = cdf(random_variable)
    fxstar = bootstrap_rv(sample)
    fxstar_cdf = cdf(fxstar)

    fitted_cdf = []
    observed_cdf = []
    for i in range(len(plist)):
        fitted_cdf.append(cdf(fx, sample[i]))
        observed_cdf.append(cdf(fxstar_cdf, sample[i]))

    ion()
    prob_plot(observed_cdf, fitted_cdf, "PP Plot")


def qq_plot(random_variable, sample):
    """
    Procedure: qq_plot
    Purpose: Plots the q_i quantile of a fitted distribution
                versus the q_i quantile of the sample dist
    Arguments:  1. random_variable: A random variable
                2. sample: sample data
    Output:     1. QQ Plot
    """
    from applpy.rv import RVError

    if not isinstance(sample, list):
        raise RVError("The data sample must be given as a list")

    n = len(sample)
    sample = sorted(sample)
    qlist = []
    for i in range(1, n + 1):
        q = (i - (1 / 2)) / n
        qlist.append(q)

    fitted = []
    for i in range(len(qlist)):
        fitted.append(random_variable.variate(s=qlist[i])[0])

    ion()
    prob_plot(sample, fitted, "QQ Plot")


# Backward compatibility aliases for legacy APPLPy naming.
PlotDist = plot_dist
PlotEmpCDF = plot_emp_cdf
PPPlot = pp_plot
QQPlot = qq_plot
