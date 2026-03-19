"""
Main Random Variable Module

1. The Random Variable class
2. Procedures for changing functional form
3. Operations on one random variable
4. Operations on two random variables
5. Plots

Procedures On One Random Variable:
    ConvolutionIID(random_variable,n)
    MaximumIID(random_variable,n)
    MinimumIID(random_variable,n)
    OrderStat(random_variable,n,r)
    ProductIID(random_variable,n)
    RangeStat(random_variable,n)
    Transform(random_variable,gX)
    Truncate(random_variable,[lw,up])
    variance(random_variable)
    VerifyPDF(random_variable)

Procedures On Two Random Variables:
    Convolution(random_variable_1,random_variable_2)
    Maximum(random_variable_1,random_variable_2)
    Minimum(random_variable_1,random_variable_2)
    Mixture(MixParameters,MixRVs)
    Product(random_variable_1,random_variable_2)

Plotting Procedures:
    Histogram(sample,bins)
    PlotDist(random_variable,suplist)
    PlotDisplay(plot_list,suplist)
    PlotEmpCDF(data)
    PPPlot(random_variable,sample)
    QQPlot(random_variable,sample)
"""

from applpy import rust_bindings

from sympy import (
    Symbol,
    symbols,
    oo,
    integrate,
    summation,
    diff,
    exp,
    sqrt,
    factorial,
    ln,
    simplify,
    solve,
    nan,
    binomial,
    pprint,
    expand,
    zoo,
    latex,
    Rational,
    Float,
    limit,
)
from random import random
import numpy as np
import pickle
from enum import Enum

import matplotlib.pylab as plt

x, y, z, t = symbols("x y z t")

# A Probability Progamming Language (APPL) -- Python Edition
# Copyright (C) 2001,2002,2008,2010,2014 Andrew Glen, Larry
# Leemis, Diane Evans, Matthew Robinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class RVError(Exception):
    """
    RVError Class
    Defines a custom error message for exceptions relating
    to the random variable class
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FunctionalForm(str, Enum):
    CDF = "cdf"
    CHF = "chf"
    HF = "hf"
    IDF = "idf"
    PDF = "pdf"
    SF = "sf"


class DomainType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    DISCRETE_FUNCTIONAL = "discrete_functional"


class RV:
    """
    RV Class
    Defines the data structure of ApplPy random variables
    Defines procedures relating to ApplPy random variables
    """

    @staticmethod
    def _normalize_functional_form(functional_form):
        if isinstance(functional_form, FunctionalForm):
            return functional_form.value
        return functional_form

    @staticmethod
    def _normalize_domain_type(domain_type):
        if isinstance(domain_type, DomainType):
            return domain_type.value
        # Legacy spelling from APPLPy Python 2 code.
        if domain_type in ["Discrete", "discrete_functional"]:
            return DomainType.DISCRETE_FUNCTIONAL.value
        return domain_type

    @classmethod
    def _parse_ftype(cls, ftype):
        if not isinstance(ftype, list) or len(ftype) != 2:
            raise RVError("ftype must be a list with [domain_type, functional_form]")

        domain_type = cls._normalize_domain_type(ftype[0])
        functional_form = cls._normalize_functional_form(ftype[1])
        return domain_type, functional_form

    def __init__(self, func, support, ftype=None, functional_form=None, domain_type=None):
        """
        Creates an instance of the random variable class
            Pass either `ftype` or both `functional_form` and `domain_type`
        Checks the random variable for errors
        """

        if ftype is None and (functional_form, domain_type) == (None, None):
            functional_form = "pdf"
            domain_type = "continuous"

        using_ftype = ftype is not None
        using_new_args = (functional_form is not None) or (domain_type is not None)

        if using_ftype == using_new_args:
            raise ValueError(
                "Pass either ftype or both functional_form and domain_type, but not both."
            )

        if using_ftype:
            domain_type, functional_form = self._parse_ftype(ftype)
        else:
            if functional_form is None or domain_type is None:
                raise ValueError(
                    "Pass either ftype or both functional_form and domain_type, but not both."
                )
            domain_type = self._normalize_domain_type(domain_type)
            functional_form = self._normalize_functional_form(functional_form)

        # Check for errors in the data structure of the random variable

        # Check to make sure that the given function is in the
        #   form of a list
        # If it is not in the form of a list, place it in a list
        if not isinstance(func, list):
            func1 = func
            func = [func1]
        # Check to make sure that the given support is in the form of
        #   a list
        if not isinstance(support, list):
            raise RVError("Support must be a list")
        # Check to make sure that the random variable is either
        #   discrete or continuous
        if domain_type not in [item.value for item in DomainType]:
            string = "Random variables must either be discrete"
            string += " or continuous"
            raise RVError(string)
        # Check to make sure that the support list has the correct
        #   length
        # The support list should be one element larger than the
        #   function list for continuous distributions, and the same
        #   size for discrete
        if domain_type in [DomainType.CONTINUOUS.value, DomainType.DISCRETE_FUNCTIONAL.value]:
            if len(support) - len(func) != 1:
                string = "Support has incorrect number of elements"
                raise RVError(string)
        if domain_type == DomainType.DISCRETE.value:
            if len(support) - len(func) != 0:
                string = "Support has incorrect number of elements"
                raise RVError(string)
        # Check to make sure that the elements of the support list are
        #   in ascending order
        for i in range(len(support) - 1):
            # Only compare if the supports are numbers
            if isinstance(support[i], (int, float)):
                if isinstance(support[i + 1], (int, float)):
                    if support[i] > support[i + 1]:
                        raise RVError("Support is not in ascending order")
        # Initialize the random variable
        self.func = func
        self.support = support
        self.domain_type = domain_type
        self.functional_form = functional_form
        self.cache = None
        self.filename = None

    @property
    def ftype(self):
        # Backward compatibility for legacy callers.
        return [self.domain_type, self.functional_form]

    @ftype.setter
    def ftype(self, ftype):
        domain_type, functional_form = self._parse_ftype(ftype)
        self.domain_type = domain_type
        self.functional_form = functional_form

    def _has_functional_form(self, functional_form):
        return self.functional_form == functional_form.value

    def _has_domain_type(self, domain_type):
        return self.domain_type == domain_type.value

    def is_cdf(self):
        return self._has_functional_form(FunctionalForm.CDF)

    def is_chf(self):
        return self._has_functional_form(FunctionalForm.CHF)

    def is_hf(self):
        return self._has_functional_form(FunctionalForm.HF)

    def is_idf(self):
        return self._has_functional_form(FunctionalForm.IDF)

    def is_pdf(self):
        return self._has_functional_form(FunctionalForm.PDF)

    def is_sf(self):
        return self._has_functional_form(FunctionalForm.SF)

    def is_continuous(self):
        return self._has_domain_type(DomainType.CONTINUOUS)

    def is_discrete(self):
        return self._has_domain_type(DomainType.DISCRETE)

    def is_discrete_functional(self):
        return self._has_domain_type(DomainType.DISCRETE_FUNCTIONAL)

    def __repr__(self):
        """
        Procedure Name: __repr__
        Purpose: Sets the default string display setting for the random
                    variable class
        Arguments:  1. self: the random variable
        Output:     1. A series of print statements describing
                        each segment of the random variable
        """
        return repr(self.display(opt="repr"))

    def __len__(self):
        """
        Procedure Name: __len__
        Purpose: Sets the behavior for the len() procedure when an instance
                    of the random variable class is given as input. This
                    procedure will return the number of pieces if the
                    distribution is piecewise.
        Arguments:  1. self: the random variable
        Output:     1. the number of segments in the random variable
        """
        return len(self.func)

    # The following procedures set the behavior for the +,-,* and / operators,
    #  as well as the behavior for negation and absolute value. If the
    #   operators are used with two random variables are used, APPLPy calls
    #   the product or convolution commands. If the operators are used
    #   with a random variable and a constant, the random variable can be
    #   shifted or scaled.

    def __pos__(self):
        """
        Procedure Name: __pos__
        Purpose: Implements the behavior for the positive operator
        Arguments:  1. self: the random variable
        Output:     1. The same random variable
        """
        return self

    def __neg__(self):
        """
        Procedure Name: __neg__
        Purpose: Implements the behavior for negation
        Arguments:  1. self: the random variable
        Output:     1. The negative transformation of the random variable
        """
        gX = [[-x], [-oo, oo]]
        neg = Transform(self, gX)
        return neg

    def __abs__(self):
        """
        Procedure Name: __abs__
        Purpose: Implements the behavior of random variables passed to the
                    abs() function
        Arguments:  1. self: the random variable
        Output:     1. The absolute value of the random variable
        """
        gX = [[abs(x)], [-oo, oo]]
        abs_rv = Transform(self, gX)
        return abs_rv

    def __add__(self, other):
        """
        Procedure Name: __add__
        Purpose: If two random variables are passed to the + operator,
                    the convolution of those random variables is returned.
                    If a constant is added to the random variable, the
                    random variable is shifted by that constant
        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        # If the random variable is added to another random variable,
        #   return the convolution of the two random variables
        if "RV" in other.__class__.__name__:
            try:
                return Convolution(self, other)
            except Exception:
                return Convolution(other, self)
            else:
                raise RVError("Could not compute the convolution")
        # If the random variable is added to a constant, shift
        # the random variable
        if isinstance(other, (float, int)):
            gX = [[x + other], [-oo, oo]]
            return Transform(self, gX)

    def __radd__(self, other):
        """
        Procedure Name: __radd__
        Purpose: If two random variables are passed to the + operator,
                    the convolution of those random variables is returned.
                    If a constant is added to the random variable, the
                    random variable is shifted by that constant.

                    __radd__ implements the reflection of __add__

        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Procedure Name: __sub__
        Purpose: If two random variables are passed to the - operator,
                    the difference of those random variables is returned.
                    If a constant is subracted from the random variable, the
                    random variable is shifted by that constant
        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        # If the random variable is subtracted by another random variable,
        #   return the difference of the two random variables
        if "RV" in other.__class__.__name__:
            gX = [[-x], [-oo, oo]]
            random_variable = Transform(other, gX)
            return Convolution(self, random_variable)
        # If the random variable is subtracted by a constant, shift
        # the random variable
        if isinstance(other, (float, int)):
            gX = [[x - other], [-oo, oo]]
            return Transform(self, gX)

    def __rsub__(self, other):
        """
        Procedure Name: __rsub__
        Purpose: If two random variables are passed to the - operator,
                    the difference of those random variables is returned.
                    If a constant is subracted from the random variable, the
                    random variable is shifted by that constant
        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        # Perform an negative transformation of the random variable
        neg_self = -self
        # Add the two components
        return neg_self.__add__(other)

    def __mul__(self, other):
        """
        Procedure Name: __mul__
        Purpose: If two random variables are passed to the * operator,
                    the product of those random variables is returned.
                    If a constant is multiplied by the random variable, the
                    random variable is scaled by that constant
        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        # If the random variable is multiplied by another random variable,
        #   return the product of the two random variables
        if "RV" in other.__class__.__name__:
            try:
                return Product(self, other)
            except Exception:
                return Product(other, self)
            else:
                raise RVError("Could not compute the product")
        # If the random variable is multiplied by a constant, scale
        # the random variable
        if isinstance(other, (float, int)):
            gX = [[x * other], [-oo, oo]]
            return Transform(self, gX)

    def __rmul__(self, other):
        """
        Procedure Name: __rmul__
        Purpose: If two random variables are passed to the * operator,
                    the product of those random variables is returned.
                    If a constant is multiplied by the random variable, the
                    random variable is scaled by that constant
        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Procedure Name: __truediv__
        Purpose: If two random variables are passed to the / operator,
                    the quotient of those random variables is returned.
                    If a constant is multiplied by the random variable, the
                    random variable is scaled by the inverse of that constant
        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        # If the random variable is divided by another random variable,
        #   return the quotient of the two random variables
        if "RV" in other.__class__.__name__:
            gX = [[1 / x, 1 / x], [-oo, 0, oo]]
            random_variable = Transform(other, gX)
            return Product(self, random_variable)
        # If the random variable is divided by a constant, scale
        # the random variable by theinverse of the constant
        if isinstance(other, (float, int)):
            gX = [[x / other], [-oo, oo]]
            return Transform(self, gX)

    def __rtruediv__(self, other):
        """
        Procedure Name: __rtruediv__
        Purpose: If two random variables are passed to the / operator,
                    the quotient of those random variables is returned.
                    If a constant is multiplied by the random variable, the
                    random variable is scaled by the inverse of that constant
        Arguments:  1. self: the random variable
                    2. other: a constant or random variable
        Output:     1. A new random variable
        """
        ## Invert the random variable
        gX = [[1 / x, 1 / x], [-oo, 0, oo]]
        invert = Transform(self, gX)
        ## Call the multiplication function
        div_rv = invert.__mul__(other)
        return div_rv

    def __pow__(self, n):
        """
        Procedure Name: __pow__
        Purpose: If the '**' operator is used on a random variable, the
            IID product of the random variable is returned
        Arguments:  1. self: the random variable
                    2. n: the number of iid random variables
        Output:     1. The distribution of n iid random variables
        """
        # Raise an error if a non-integer value is passed to n
        if not isinstance(n, int):
            error_string = "a random variable can only be raised to an"
            error_string += " integer value"
            raise RVError(error_string)

        pow_rv = Pow(self, n)
        return pow_rv

    def __eq__(self, other):
        """
        Procedure Name: __eq__
        Purpose: Checks for equality of the two random variables by using
                    the following algorithm:
                        1. Test if the support of both random variables
                            are equal
                        2. Test to see if each section of the random variable
                            simplifies to zero when subtracted from the
                            corresponding segment of the second random
                            variables
        Arguments:  1. self: the random variable
                    2. other: a second random variable
        Output:     1. True if the the random variables are equal, False
                        otherwise
        """
        # If the other is not a random variable, return an error
        if "RV" not in other.__class__.__name__:
            error_string = "a random variable can only be checked for"
            error_string += " equality with another random variable"
            raise RVError(error_string)
        # Check to see if the supports of the random variables are
        #   equal
        if not self.support == other.support:
            return False
        # Subtract each each segment from self from the corresponding
        #   segment from other, check to see if the difference
        #   simplifies to zero
        for i in range(len(self.func)):
            difference = self.func[i] - other.func[i]
            difference = simplify(difference)
            difference = expand(difference)
            if not difference == 0:
                return False
        # If all of the segments simplify to zero, return True
        return True

    def add_assumptions(self, option):
        """
        Procedure Name: drop_assumptions
        Purpose: Adds assumptions on the random variable to support operations
            on multiple random variables
        Arugments:  1. self: the random variable
                    2. option: the type of assumption to add
        Output:     1. Modified function with assumptions added
        """
        if option not in ["positive", "negative", "nonpositive", "nonnegative"]:
            err_str = "The only available options are positive, negative,"
            err_str += " nonpositive and nonnegative"
            raise RVError(err_str)
        if option == "positive":
            x = Symbol("x", positive=True)
        elif option == "negative":
            x = Symbol("x", negative=True)
        elif option == "nonpositive":
            x = Symbol("x", nonpositive=True)
        elif option == "nonnegative":
            x = Symbol("x", nonnegative=True)
        for i, function in enumerate(self.func):
            function = function.subs(Symbol("x"), x)
            self.func[i] = function

    def add_to_cache(self, object_name, obj):
        """
        Procedure Name: add_to_cache
        Purpose: Stores properties of the random variable (i.e. mean, variance,
                    cdf, sf) in memory. The next time a function is called to
                    compute that property, APPLPy will retrieve the object
                    from memory.
        Arguments:  1. self: the random variable
                    2. object_name: the key for the object in the cache
                        dictionary
                    3. obj: the object to be stored in memory.
        Output:     1. No output. The self.cache property of the random
                        variable is modified to include the specified
                        object.
        """
        # If a cache for the random variable does not exist, initialize it
        if self.cache is None:
            self.init_cache()
        # Add an object to the cache dictionary
        self.cache[object_name] = obj

    def display(self, opt="repr"):
        """
        Procedure Name: display
        Purpose: Displays the random variable in an interactive environment
        Arugments:  1. self: the random variable
        Output:     1. A print statement for each piece of the distribution
                        indicating the function and the relevant support
        """
        if self.is_continuous() or self.is_discrete_functional():
            print(("%s %s" % (self.domain_type, self.functional_form)))
            for i in range(len(self.func)):
                print(("for %s <= x <= %s" % (self.support[i], self.support[i + 1])))
                print("---------------------------")
                pprint(self.func[i])
                print("---------------------------")
                if i < len(self.func) - 1:
                    print(" ")
                    print(" ")

        if self.is_discrete():
            print("%s %s where {x->f(x)}:" % (self.domain_type, self.functional_form))
            for i in range(len(self.support)):
                if i != (len(self.support) - 1):
                    print("{%s -> %s}, " % (self.support[i], self.func[i]), end=" ")
                else:
                    print("{%s -> %s}" % (self.support[i], self.func[i]))

    def drop_assumptions(self):
        """
        Procedure Name: drop_assumptions
        Purpose: Drops assumptions on the random variable to support operations
            on multiple random variables
        Arugments:  1. self: the random variable
        Output:     1. Modified function with assumptions dropped
        """
        x = Symbol("x")
        for i, function in enumerate(self.func):
            function = function.subs(Symbol("x", negative=True), x)
            function = function.subs(Symbol("x", nonnegative=True), x)
            function = function.subs(Symbol("x", nonpositive=True), x)
            function = function.subs(Symbol("x", positive=True), x)
            self.func[i] = function

    def init_cache(self):
        """
        Procedure Name: init_cache
        Purpose: Initializes the cache for the random variable
        Arguments:  1. self: the random variable
        Output:     1. The cache attribute for the random variable
                            is initialized
        """
        self.cache = {}

    def latex(self):
        """
        Procedure Name: latex
        Purpose: Outputs the latex code for the random variable
        Arugments:  1.self: the random variable
        Output:     1. The latex code for the random variable
        """
        if not (self.is_continuous() or self.is_discrete_functional()):
            error_string = "latex is only designed to work for continuous"
            error_string += " distributions and discrete distributions that "
            error_string += "are represented in functional form"
            raise RVError(error_string)
        # Generate the pieces of the piecewise function
        piece_list = []
        for i in range(len(self.func)):
            f = self.func[i]
            sup = "x>=%s" % (self.support[i])
            tup = (f, eval(sup))
            piece_list.append(tup)
        piece_list.append((0, True))
        piece_input = "Piecewise(" + str(piece_list) + ")"
        piece2 = piece_input.replace(piece_input[10], "")
        n = len(piece2) - 2
        piece3 = piece2.replace(piece2[n], "")
        # Create symbols for use in the piecewise
        #   function display
        Symbol("theta")
        Symbol("kappa")
        Symbol("a")
        Symbol("b")
        Symbol("c")
        p = Symbol("p")
        Symbol("N")
        Symbol("alpha")
        Symbol("beta")
        Symbol("mu")
        Symbol("sigma")

        p = eval(piece3)
        return latex(p)

    def save(self, filename=None):
        """
        Procedure Name: save
        Purpose: Saves a random variable to disk in binary format
        Arguments:  1. self: the random variable
                    2. filename: the name of the file that will
                        store the random variable. If none is
                        specified, the most recently used file
                        name is used
        Output:     1. The random variable is stored to disk
        """
        if filename is None:
            if self.filename is None:
                err_string = "Please specify a file name, this random "
                err_string += "has never been saved before "
                raise RVError(err_string)
            else:
                filename = self.filename
        else:
            self.filename = filename
        fileObject = open(filename, "wb")
        pickle.dump(self, fileObject)

    def simplify(self, assumption=None):
        """
        Procedure Name: simplify
        Purpose: Uses assumptions to help simplify the random variable
        Arguments:  1. self: the random variable.
        Output:     1. A list of assumptions for each segment in the random
                        variable
        """
        for i, segment in enumerate(self.func):
            if self.support[i] < 0 and self.support[i + 1] <= 0:
                x2 = Symbol("x2", negative=True)
            elif self.support[i + 1] > 0 and self.support[i] >= 0:
                x2 = Symbol("x2", positive=True)
            else:
                x2 = Symbol("x2")
            new_func = segment.subs(x, x2)
            new_func = simplify(new_func)
            self.func[i] = new_func.subs(x2, x)
        new_func = []
        new_support = []
        for i in range(len(self.func)):
            if i == 0:
                new_func.append(self.func[i])
                new_support.append(self.support[i])
            else:
                if self.func[i] != self.func[i - 1]:
                    new_func.append(self.func[i])
                    new_support.append(self.support[i])
        new_support.append(self.support[-1])
        self.func = new_func
        self.support = new_support
        self.display()

    def verify_pdf(self):
        """
        Procedure Name: verify_pdf
        Purpose: Verifies whether or not the random variable is valid. It first
                    checks to make sure the pdf of the random variable
                    integrates to one. It then checks to make sure the random
                    variable is strictly positive.
        Arguments:  1. self: the random variable.
        Output:     1. A print statement indicating the area under the pdf
                        and a print statement indicating whether or not the
                        random variable is valid.
        """
        # If the random variable is continuous, verify the PDF
        if self.is_continuous():
            # Check to ensure that the distribution is fully
            #   specified
            for piece in self.func:
                func_symbols = piece.atoms(Symbol)
                if len(func_symbols) > 1:
                    err_string = "distribution must be fully"
                    err_string += " specified"
                    raise RVError(err_string)
            # Convert the random variable to PDF form
            X_dummy = pdf(self)
            # Check to ensure that the area under the PDF is 1
            print("Now checking for area...")
            area = 0
            for i in range(len(X_dummy.func)):
                val = integrate(X_dummy.func[i], (x, X_dummy.support[i], X_dummy.support[i + 1]))
                area += val
            print("The area under f(x) is: %s" % (area))
            # Check absolute value
            print("Now checking for absolute value...")
            #
            # The following code should work in future versions of SymPy
            # Currently, Sympy is having difficulty consistently integrating
            # the absolute value of a function symbolically
            #
            # abs_area=0
            # for i in range(len(X_dummy.func)):
            # val=integrate(Abs(X_dummy.func[i],(x,X_dummy.support[i],
            #                                   X_dummy.support[i+1]))
            # abs_area+=val
            abs_flag = True
            val_list = []
            quant_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for i in range(len(quant_list)):
                val = self.variate(s=quant_list[i])[0]
                val_list.append(val)
            for i in range(len(val_list)):
                if val_list[i] < 0:
                    abs_flag = False
            print("The pdf of the random variable:")
            print("%s" % (X_dummy.func))
            print("continuous pdf with support %s" % (X_dummy.support))
            if area > 0.9999 and area < 1.00001 and abs_flag:
                print("is valid")
                return True
            else:
                print("is not valid")
                return False
        # If the random variable is in a discrete functional form,
        #   verify the PDF
        if self.is_discrete_functional():
            # Convert the random variable to PDF form
            X_dummy = pdf(self)
            # Check to ensure that the area under the PDF is 1
            print("Now checking for area...")
            area = 0
            for i in range(len(X_dummy.func)):
                val = summation(X_dummy.func[i], (x, X_dummy.support[i], X_dummy.support[i + 1]))
                area += val
            print("The area under f(x) is: %s" % (area))
            # Check absolute value
            print("Now checking for absolute value...")
            abs_flag = True
            val_list = []
            quant_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for i in range(len(quant_list)):
                val = self.variate(s=quant_list[i])[0]
                val_list.append(val)
            for i in range(len(val_list)):
                if val_list[i] < 0:
                    abs_flag = False
            print("The pdf of the random variable:")
            print("%s" % (X_dummy.func))
            print("discrete pdf with support %s" % (X_dummy.support))
            if area > 0.9999 and area < 1.00001 and abs_flag:
                print("is valid")
                return True
            else:
                print("is not valid")
                return False
        # If the random variable is discrete, verify the PDF
        if self.is_discrete():
            X_dummy = pdf(self)
            is_valid = rust_bindings.verify_discrete_pdf(X_dummy.func)
            if is_valid:
                print("is valid")
            else:
                print("is not valid")

    def variate(self, n=1, s=None, sensitivity=None, method="newton-raphson"):
        """
        Procedure Name: variate
        Purpose: Generates a list of n random variates from the random variable
                    using the Newton-Raphson Method
        Arguments:  1. self: the random variable
                    2. n: the number of variates (default is n=1)
                    3. s: the percentile of the variate (default is random)
                    4. method: specifies the method for variate generaton
                                valid methods are:
                                1. 'newton-raphson'
                                2. 'inverse'
                    5. sensitivity: value indicating how close two interations
                            must be for the variate generator to reach
                            convergence. (default is .1% of the mean)
        """

        # Check to see if the user specified a valid method
        method_list = ["newton-raphson", "inverse"]
        if method not in method_list:
            error_string = "an invalid method was specified"
            raise RVError(error_string)

        # If the inverse method is specified, compute variates using
        #   the IDF function
        if method == "inverse":
            Xidf = idf(self)
            varlist = [idf(Xidf, random()) for i in range(1, n + 1)]
            return varlist

        # Find the cdf and pdf functions (to avoid integrating for
        # each variate
        cdf_rv = cdf(self)
        pdf_rv = pdf(self)
        mean_value = mean(self)
        if sensitivity is None:
            # If sensitivity is not specified, set the sensitivity to be
            #   .1% of the mean value for random variates
            if s is None:
                sensitivity = abs(0000.1 * mean_value)
            # If a percentile is specified, set sensitivity to be .01%
            #   of the mean value
            else:
                sensitivity = abs(0000.1 * mean_value)
        # Create a list of variates
        varlist = []
        for i in range(n):
            guess_last = oo
            guess = mean_value
            if s is None:
                val = random()
            else:
                val = s
            while abs(guess_last - guess) > sensitivity:
                guess_last = guess
                try:
                    if len(self.func) == 1:
                        guess = guess - (
                            (cdf_rv.func[0].subs(x, guess) - val) / pdf_rv.func[0].subs(x, guess)
                        )
                        guess = guess.evalf()
                    else:
                        guess = (guess - ((cdf(cdf_rv, guess) - val) / pdf(pdf_rv, guess))).evalf()
                except Exception:
                    if guess > self.support[len(self.support) - 1]:
                        cfunc = cdf_rv.func[len(self.func) - 1].subs(x, guess)
                        pfunc = pdf_rv.func[len(self.func) - 1].subs(x, guess)
                        guess = (guess - ((cfunc - val) / pfunc)).evalf()
                    if guess < self.support[0]:
                        cfunc = cdf_rv.func[0].subs(x, guess)
                        pfunc = pdf_rv.func[0].subs(x, guess)
                        guess = (guess - ((cfunc - val) / pfunc)).evalf()
                # print guess
            varlist.append(guess)
        varlist.sort()
        return varlist


# Conversion Procedures:
#     cdf(random_variable,value)
#     chf(random_variable,value)
#     hf(random_variable,value)
#     idf(random_variable,value)
#     pdf(random_variable,value)
#     sf(random_variable,value)
#     BootstrapRV(varlist)
#     Convert(random_variable,inc)


def check_value(value, sup):
    # Not intended for use by end user
    """
    Procedure Name: check_value
    Purpose: Check to see if a value passed to CDF,CHF,HF,PDF or
                SF is in the support of the random variable
    Arguments:  1. value: The value passed to RV procedure
                2. sup: The support of the RV in the procedure
    Output:     1. True if the value given is within the support
                2. False otherwise
    """
    if value == x:
        return True
    else:
        max_idx = len(sup) - 1
        if float(value) < float(sup[0]) or float(value) > float(sup[max_idx]):
            return False
        else:
            return True


def cdf(random_variable, value=x, cache=False):
    from .conversion import cdf as _cdf

    return _cdf(random_variable, value=value, cache=cache)


def chf(random_variable, value=x, cache=False):
    from .conversion import chf as _chf

    return _chf(random_variable, value=value, cache=cache)


def hf(random_variable, value=x, cache=False):
    from .conversion import hf as _hf

    return _hf(random_variable, value=value, cache=cache)


def idf(random_variable, value=x, cache=False):
    from .conversion import idf as _idf

    return _idf(random_variable, value=value, cache=cache)


def pdf(random_variable, value=x, cache=False):
    from .conversion import pdf as _pdf

    return _pdf(random_variable, value=value, cache=cache)


def sf(random_variable, value=x, cache=False):
    from .conversion import sf as _sf

    return _sf(random_variable, value=value, cache=cache)


# Backward-compatible aliases
CDF = cdf
CHF = chf
HF = hf
IDF = idf
PDF = pdf
SF = sf


def BootstrapRV(varlist, symbolic=False):
    """
    Procedure Name: Bootstrap RV
    Purpose: Generate a discrete random variable from a list of variates
    Arguments: 1. varlist: A list of variates
    Output:    1. A discrete random variable, where each element in the
                    given variate list is equally probable
    """
    # Sort the list of variables
    varlist.sort()
    # Find the number of elements in the list of variates
    numel = len(varlist)
    # Use varlist to generate the function and support for the random variable
    #   Count number of times element appears in varlist, divide by number
    #   of elements
    funclist = []
    supplist = []
    for i in range(len(varlist)):
        if varlist[i] not in supplist:
            supplist.append(varlist[i])
            funclist.append(Rational(varlist.count(varlist[i]), numel))
    # Return the result as a discrete random variable
    return RV(funclist, supplist, ["discrete", "pdf"])


def Convert(random_variable, inc=1):
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


# Procedures on One Random Variable
#
# Procedures:
#     ConvolutionIID(random_variable,n)
#     coef_of_var(random_variable)
#     expected_value(random_variable,gX)
#     entropy(random_variable)
#     kurtosis(random_variable)
#     MaximumIID(random_variable,n)
#     mean(random_variable)
#     mgf(random_variable)
#     MinimumIID(random_variable,n)
#     OrderStat(random_variable,n,r)
#     Power(Rvar,n)
#     ProductIID(random_variable,n)
#     skewness(random_variable)
#     SqRt(random_variable)
#     Transform(random_variable,gX)
#     Truncate(random_variable,[lw,up])
#     variance(random_variable)


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
    for i in range(n - 1):
        X_final += X_dummy
    return pdf(X_final)


def coef_of_var(random_variable, cache=False):
    from .moments import coef_of_var as _CoefOfVar

    return _CoefOfVar(random_variable, cache=cache)


def expected_value(random_variable, gX=x):
    from .moments import expected_value as _ExpectedValue

    return _ExpectedValue(random_variable, gX=gX)


def entropy(random_variable, cache=False):
    from .moments import entropy as _Entropy

    return _Entropy(random_variable, cache=cache)


def kurtosis(random_variable, cache=False):
    from .moments import kurtosis as _Kurtosis

    return _Kurtosis(random_variable, cache=cache)


def mean(random_variable, cache=False):
    from .moments import mean as _Mean

    return _Mean(random_variable, cache=cache)


def mgf(random_variable, cache=False):
    from .moments import mgf as _MGF

    return _MGF(random_variable, cache=cache)


def skewness(random_variable, cache=False):
    from .moments import skewness as _Skewness

    return _Skewness(random_variable, cache=cache)


def variance(random_variable, cache=False):
    from .moments import variance as _Variance

    return _Variance(random_variable, cache=cache)


# Backward-compatible aliases
CoefOfVar = coef_of_var
ExpectedValue = expected_value
Entropy = entropy
Kurtosis = kurtosis
Mean = mean
MGF = mgf
Skewness = skewness
Variance = variance


def MaximumIID(random_variable, n=Symbol("n")):
    """
    Procedure Name: MaximumIID
    Purpose: Compute the maximum of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The maximum of n iid random variables
    """
    # Check to make sure n is an integer
    if not isinstance(n, int):
        if not isinstance(n, Symbol):
            raise RVError("The second argument must be an integer")

    # If n is symbolic, find and return the maximum using
    #   OrderStat (may need to test and see if this is more
    #   efficient than using the for loop for non symbolic parameters)
    if isinstance(n, Symbol):
        return OrderStat(random_variable, n, n)
    # Compute the iid maximum
    else:
        X_dummy = random_variable
        X_final = X_dummy
        for i in range(n - 1):
            X_final = Maximum(X_final, X_dummy)
        return pdf(X_final)


def MinimumIID(random_variable, n):
    """
    Procedure Name: MinimumIID
    Purpose: Compute the minimum of n iid random variables
    Arguments:  1. random_variable: A random variable
                2. n: an integer
    Output:     1. The minimum of n iid random variables
    """
    # Check to make sure n is an integer
    if not isinstance(n, int):
        if not isinstance(n, Symbol):
            raise RVError("The second argument must be an integer")

    # If n is symbolic, find and return the maximum using
    #   OrderStat (may need to test and see if this is more
    #   efficient than using the for loop for non symbolic parameters)
    if isinstance(n, Symbol):
        return OrderStat(random_variable, 1, n)
    # Compute the iid minimum
    else:
        X_dummy = random_variable
        X_final = X_dummy
        for i in range(n - 1):
            X_final = Minimum(X_final, X_dummy)
        return pdf(X_final)


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

    # If the distribution is continuous, find and return the value of the
    #   order statistic
    if random_variable.is_continuous():
        if replace == "wo":
            err_string = "OrderStat without replacement not implemented "
            err_string += "for continuous random variables"
            raise RVError(err_string)
        # Compute the PDF, CDF and SF of the random variable
        pdf_dummy = pdf(random_variable)
        cdf_dummy = cdf(random_variable)
        sf_dummy = sf(random_variable)
        # Compute the factorial constant
        const = (factorial(n)) / (factorial(r - 1) * factorial(n - r))
        # Compute the distribution of the order statistic for each
        #   segment
        ordstat_func = []
        for i in range(len(random_variable.func)):
            fx = pdf_dummy.func[i]
            Fx = cdf_dummy.func[i]
            Sx = sf_dummy.func[i]
            ordfunc = const * (Fx ** (r - 1)) * (Sx ** (n - r)) * fx
            ordstat_func.append(simplify(ordfunc))
        # Return the distribution of the order statistic
        return RV(ordstat_func, random_variable.support, ["continuous", "pdf"])

    # If the distribution is in discrete symbolic form, convert it to
    #   discrete explicit form and find the order statistic
    if random_variable.is_discrete_functional():
        if (-oo not in random_variable.support) and (oo not in random_variable.support):
            X_dummy = Convert(random_variable)
            return OrderStat(X_dummy, n, r, replace)
        else:
            err_string = "OrderStat is not currently implemented for "
            err_string += "discrete RVs with infinite support"
            raise RVError(err_string)

    # If the distribution is continuous, find and return the value of
    #   the order statistic
    if random_variable.is_discrete():
        fx = pdf(random_variable)
        Fx = cdf(random_variable)
        Sx = sf(random_variable)
        N = len(fx.support)
        # With replacement
        if replace == "w":
            # Numeric PDF
            if not isinstance(random_variable.func[0], Symbol):
                # If N is one, return the order stat
                if N == 1:
                    return RV(1, random_variable.support, ["discrete", "pdf"])
                # Add the first term
                else:
                    OSproblist = []
                    os_sum = 0
                    for w in range(n - r + 1):
                        val = binomial(n, w) * (fx.func[0] ** (n - w)) * (Sx.func[1] ** (w))
                        os_sum += val
                    OSproblist.append(os_sum)
                # Add term 2 through N-1
                for k in range(2, N):
                    os_sum = 0
                    for w in range(n - r + 1):
                        for u in range(r):
                            val = (
                                factorial(n)
                                / (factorial(u) * factorial(n - u - w) * factorial(w))
                                * (Fx.func[k - 2] ** u)
                                * (fx.func[k - 1] ** (n - u - w))
                                * (Sx.func[k] ** (w))
                            )
                            os_sum += val
                    OSproblist.append(os_sum)
                # Add term N
                os_sum = 0
                for u in range(r):
                    val = binomial(n, u) * (Fx.func[N - 2] ** u) * (fx.func[N - 1] ** (n - u))
                    os_sum += val
                OSproblist.append(os_sum)
                return RV(OSproblist, random_variable.support, ["discrete", "pdf"])

        if replace == "wo":
            """
            if n>4:
                err_string = 'When sampling without replacement, n must be '
                err_string += 'less than 4'
                raise RVError(err_string)
            """
            # Determine if the PDF has equally likely probabilities
            EqLike = True
            for i in range(len(fx.func)):
                if fx.func[0] != fx.func[i]:
                    EqLike = False
                if not EqLike:
                    break
            # Create blank order stat function list
            fxOS = []
            for i in range(len(fx.func)):
                fxOS.append(0)
            # If the probabilities are equally likely
            if EqLike:
                # Need to add algorithm for symbolic 'r'
                for i in range(r, (N - n + r + 1)):
                    indx = i - 1
                    val = (binomial(i - 1, r - 1) * binomial(1, 1) * binomial(N - i, n - r)) / (
                        binomial(N, n)
                    )
                    fxOS[indx] = val
                return RV(fxOS, fx.support, ["discrete", "pdf"])
            # If the probabilities are not equally likely
            elif not EqLike:
                # If the sample size is 1
                if n == 1:
                    fxOS = []
                    for i in range(len(fx.func)):
                        fxOS.append(fx.func[i])
                    return (fxOS, fx.support, ["discrete", "pdf"])
                elif n == N:
                    fxOS[n - 1] = 1
                    return RV(fxOS, fx.support, ["discrete", "pdf"])
                else:
                    # Create null ProbStorage array of size nXN
                    # Initialize to contain all zeroes
                    print(n, N)
                    ProbStorage = []
                    for i in range(n):
                        row_list = []
                        for j in range(N):
                            row_list.append(0)
                        ProbStorage.append(row_list)
                    # Create the first lexicographical combo of
                    #   n items
                    combo = list(range(1, n + 1))
                    for i in range(1, (binomial(N, n) + 1)):
                        # Assign perm as the current combo
                        perm = []
                        for j in range(len(combo)):
                            perm.append(combo[j])
                        # Compute the probability of obtaining the
                        #   given permutation
                        for j in range(1, factorial(n) + 1):
                            PermProb = fx.func[perm[0]]
                            cumsum = fx.func[perm[0]]
                            for m in range(1, n):
                                PermProb *= fx.func[perm[m]] / (1 - cumsum)
                                cumsum += fx.func[perm[m]]
                            print(perm, PermProb, cumsum)
                            # Order each permutation and determine
                            #   which value sits in the rth
                            #   ordered position
                            orderedperm = []
                            for m in range(len(perm)):
                                orderedperm.append(perm[m])
                            orderedperm.sort()
                            for m in range(n):
                                for k in range(N):
                                    if orderedperm[m] == k + 1:
                                        ProbStorage[m][k] = PermProb + ProbStorage[m][k]
                            # Find the next lexicographical permutation
                            perm = rust_bindings.next_permutation(perm)
                        # Find the next lexicographical combination
                        combo = rust_bindings.next_combination(combo, N)


def Pow(random_variable, n):
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
    return Transform(random_variable, g)


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
    for i in range(n - 1):
        X_final *= X_dummy
    return pdf(X_final)


def RangeStat(random_variable, n, replace="w"):
    """
    Procedure Name: RangeStat
    Purpose: Compute the distribution of the range of n iid rvs
    Arguments:  1. random_variable: A random variable
                2. n: an integer
                3. replace: indicates with or without replacment
    Output:     1. The dist of the range of n iid random variables
    """
    # Check to make sure that n >= 2, otherwise there is no range
    if n < 2:
        err_string = "Only one item sampled from the population"
        raise RVError(err_string)
    if replace not in ["w", "wo"]:
        raise RVError("Replace must be w or wo")
    # Convert the random variable to its PDF form
    fX = pdf(random_variable)
    # If the random variable is continuous and its CDF is tractable,
    #   find the PDF of the range statistic
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
        RangeRV = RV(
            fXRange,
            fX.support,
            functional_form=fX.functional_form,
            domain_type=fX.domain_type,
        )
        return RangeRV
    # If the random variable is discrete symbolic, convert it to discrete
    #   explicit and compute the range statistic
    if fX.is_discrete_functional():
        if (-oo not in fX.support) and (oo not in fX.support):
            X_dummy = Convert(random_variable)
            return RangeStat(X_dummy, n, replace)
    # If the reandom variable is discrete explicit, find and return the
    #   range stat
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
            # rs is an array that holds the range support values
            # rp is an array that holds the range probability mass values
            # There are 1 + 2 + 3 + ... + N possible range support values
            #   if the support is of size N. 'uppers' is this limit
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
            # Sort rs and rp together by rs
            sortedr = list(zip(*sorted(zip(rs, rp))))
            sortrs = list(sortedr[0])
            sortrp = list(sortedr[1])
            # Combine redundant elements in the list
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
                # Create the first lexicographical combo of n items
                combo = [value for value in range(1, n + 1)]
                for i in range(binomial(N, n)):
                    # Assign perm as the current combo
                    perm = [elem for elem in combo]
                    # Compute the probability of obtaining the permutation
                    for j in range(factorial(n)):
                        PermProb = fX.func[perm[0]]
                        cumsum = fX.func[perm[0]]
                        for m in range(1, n):
                            PermProb *= fX.func[perm[m]] / (1 - cumsum)
                            cumsum += fX.func[perm[m]]
                        # Find the maximum and minimum elements of the
                        #   permutation and then determine their difference
                        HiVal = max(perm)
                        LoVal = min(perm)
                        Range = HiVal - LoVal
                        for k in range(N - 1):
                            if Range == k + 1:
                                fXRange[k] += PermProb
                        # Find the next lexicographical permutation
                        perm = rust_bindings.next_permutation(perm)
                    combo = rust_bindings.next_combination(combo, N)
                print(len(fXRange), len(fXSupport))
                return RV(
                    fXRange,
                    fXSupport,
                    functional_form=fX.functional_form,
                    domain_type=fX.domain_type,
                )


def Sqrt(random_variable):
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
    u = [[sqrt(x)], [0, oo]]
    NewRvar = Transform(random_variable, u)
    return NewRvar


def Transform(random_variable, gXt):
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
                        # print X_dummy.func[i], ginv[j]
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
        return Transform(X_dummy, gXt)

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


def Truncate(random_variable, supp):
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
        # area=0
        # for i in range(len(X_dummy.func)):
        #    val=integrate(X_dummy.func[i],(x,X_dummy.support[i],
        #                    X_dummy.support[i+1]))
        #    area+=val
        # print area
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


def VerifyPDF(random_variable):
    """
    Procedure Name: VerifyPDF
    Purpose: Calls self.verify_pdf(). For compatibility with
                original APPL syntax
    Arguments:  1. random_variable: a discrete random variable
    Output:     1. A function call to self.verify_pdf()
    """
    return random_variable.verify_pdf()


# Procedures on Two Random Variables
#
# Procedures:
#     Convolution(random_variable_1,random_variable_2)
#     Maximum(random_variable_1,random_variable_2)
#     Minimum(random_variable_1,random_variable_2)
#     Mixture(MixParameters,MixRVs)
#     Product(random_variable_1,random_variable_2)


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


def Maximum(*argv):
    """
    Procedure Name: Maximum
    Purpose: Compute the maximum of a list of random variables
    Arugments:  1. *argv: a series of random variables
    Output:     1. The maximum distribution
    """
    # Loop over the arguments and compute the distribution of the maximum
    #   of each argument
    i = 0
    for rv in argv:
        # For the first argument, create a temporary variable containing
        #   that rv
        if i == 0:
            temp = rv
        # For all others, find the maximum of the temporary variable and
        #   the rv
        else:
            temp = MaximumRV(temp, rv)
        i += 1
    return temp


def MaximumRV(random_variable_1, random_variable_2):
    """
    Procedure Name: MaximumRV
    Purpose: Compute cdf of the maximum of random_variable_1 and random_variable_2
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The cdf of the maximum distribution
    """

    # If the two random variables are not of the same type
    #   raise an error
    if random_variable_1.domain_type != random_variable_2.domain_type:
        raise RVError("The RVs must both be discrete or continuous")

    # If the distributions are continuous, find and return the max
    if random_variable_1.is_continuous():
        # X1_dummy.drop_assumptions()
        # X2_dummy.drop_assumptions()
        # Special case for lifetime distributions
        if random_variable_1.support == [0, oo] and random_variable_2.support == [0, oo]:
            cdf_dummy1 = cdf(random_variable_1)
            cdf_dummy2 = cdf(random_variable_2)
            cdf1 = cdf_dummy1.func[0]
            cdf2 = cdf_dummy2.func[0]
            maxfunc = cdf1 * cdf2
            return pdf(RV(simplify(maxfunc), [0, oo], ["continuous", "cdf"]))
        # Otherwise, compute the max using the full algorithm
        # Set up the support for X
        Fx = cdf(random_variable_1)
        Fy = cdf(random_variable_2)
        # Create a support list for the
        max_supp = []
        for i in range(len(Fx.support)):
            if Fx.support[i] not in max_supp:
                max_supp.append(Fx.support[i])
        for i in range(len(Fy.support)):
            if Fy.support[i] not in max_supp:
                max_supp.append(Fy.support[i])
        max_supp.sort()
        # Remove any elements that are above the lower support max
        lowval = max(min(Fx.support), min(Fy.support))
        max_supp2 = []
        for i in range(len(max_supp)):
            if max_supp[i] >= lowval:
                max_supp2.append(max_supp[i])
        # Compute the maximum function for each segment
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

    # If the two random variables are discrete in functinonal form,
    #   find and return the maximum of the two random variables
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

    # If the distributions are discrete, find and return
    #   the maximum of the two rv's
    if random_variable_1.is_discrete():
        # Convert X and Y to their PDF representations
        fx = pdf(random_variable_1)
        fy = pdf(random_variable_2)
        # Make a list of possible combinations of X and Y
        combo_list = []
        prob_list = []
        for i in range(len(fx.support)):
            for j in range(len(fy.support)):
                combo_list.append([fx.support[i], fy.support[j]])
                prob_list.append(fx.func[i] * fy.func[j])

        # Old code for computing probability for each pair, had
        # floating point issues, PDF wouldn't recognize a number
        # as being in the support
        # prob_list=[]
        # for i in range(len(combo_list)):
        #    val=pdf(fx,combo_list[i][0])*pdf(fy,combo_list[j][1])
        #    prob_list.append(val)

        # Find the max value for each combo
        max_list = []
        for i in range(len(combo_list)):
            max_list.append(max(combo_list[i][0], combo_list[i][1]))
        # Compute the probability for each possible max
        max_supp = []
        max_func = []
        for i in range(len(max_list)):
            if max_list[i] not in max_supp:
                max_supp.append(max_list[i])
                max_func.append(prob_list[i])
            else:
                indx = max_supp.index(max_list[i])
                max_func[indx] += prob_list[i]
        # Sort the elements of the rv
        zip_list = list(zip(max_supp, max_func))
        zip_list.sort()
        max_supp = []
        max_func = []
        for i in range(len(zip_list)):
            max_supp.append(zip_list[i][0])
            max_func.append(zip_list[i][1])
        # Return the minimum random variable
        return pdf(RV(max_func, max_supp, ["discrete", "pdf"]))


def Minimum(*argv):
    """
    Procedure Name: Minimum
    Purpose: Compute the minimum of a list of random variables
    Arugments:  1. *argv: a series of random variables
    Output:     1. The minimum distribution
    """
    # Loop over the arguments and compute the distribution of the maximum
    #   of each argument
    i = 0
    for rv in argv:
        # For the first argument, create a temporary variable containing
        #   that rv
        if i == 0:
            temp = rv
        # For all others, find the minimum of the temporary variable and
        #   the rv
        else:
            temp = MinimumRV(temp, rv)
        i += 1
    return temp


def MinimumRV(random_variable_1, random_variable_2):
    """
    Procedure Name: MinimumRV
    Purpose: Compute the distribution of the minimum of random_variable_1 and random_variable_2
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The minimum of the two random variables
    """

    # If the two random variables are not of the same type
    #   raise an error
    if random_variable_1.domain_type != random_variable_2.domain_type:
        raise RVError("The RVs must both be discrete or continuous")

    # If the distributions are continuous, find and return the min
    if random_variable_1.is_continuous():
        # X1_dummy.drop_assumptions()
        # X2_dummy.drop_assumptions()
        # Special case for lifetime distributions
        if random_variable_1.support == [0, oo] and random_variable_2.support == [0, oo]:
            sf_dummy1 = sf(random_variable_1)
            sf_dummy2 = sf(random_variable_2)
            sf1 = sf_dummy1.func[0]
            sf2 = sf_dummy2.func[0]
            minfunc = 1 - (sf1 * sf2)
            return pdf(RV(simplify(minfunc), [0, oo], ["continuous", "cdf"]))
        # Otherwise, compute the min using the full algorithm
        Fx = cdf(random_variable_1)
        Fy = cdf(random_variable_2)
        # Create a support list for the
        min_supp = []
        for i in range(len(Fx.support)):
            if Fx.support[i] not in min_supp:
                min_supp.append(Fx.support[i])
        for i in range(len(Fy.support)):
            if Fy.support[i] not in min_supp:
                min_supp.append(Fy.support[i])
        min_supp.sort()

        # Remove any elements that are above the lower support max
        highval = min(max(Fx.support), max(Fy.support))
        min_supp2 = []
        for i in range(len(min_supp)):
            if min_supp[i] <= highval:
                min_supp2.append(min_supp[i])

        # Compute the minimum function for each segment
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

        # Return the random variable
        return pdf(RV(min_func, min_supp2, ["continuous", "cdf"]))

    # If the two random variables are discrete in functinonal form,
    #   find and return the minimum of the two random variables
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

    # If the distributions are discrete, find and return
    #   the minimum of the two rv's
    if random_variable_1.is_discrete():
        # Convert X and Y to their PDF representations
        fx = pdf(random_variable_1)
        fy = pdf(random_variable_2)
        # Make a list of possible combinations of X and Y
        combo_list = []
        prob_list = []
        for i in range(len(fx.support)):
            for j in range(len(fy.support)):
                combo_list.append([fx.support[i], fy.support[j]])
                prob_list.append(fx.func[i] * fy.func[j])

        # Old code for computing probability for each pair, had
        # floating point issues, PDF wouldn't recognize a number
        # as being in the support
        # prob_list=[]
        # for i in range(len(combo_list)):
        #    val=pdf(fx,combo_list[i][0])*pdf(fy,combo_list[j][1])
        #    prob_list.append(val)
        # Find the min value for each combo

        min_list = []
        for i in range(len(combo_list)):
            min_list.append(min(combo_list[i][0], combo_list[i][1]))
        # Compute the probability for each possible min
        min_supp = []
        min_func = []
        for i in range(len(min_list)):
            if min_list[i] not in min_supp:
                min_supp.append(min_list[i])
                min_func.append(prob_list[i])
            else:
                indx = min_supp.index(min_list[i])
                min_func[indx] += prob_list[i]
        # Sort the elements of the rv
        zip_list = list(zip(min_supp, min_func))
        zip_list.sort()
        min_supp = []
        min_func = []
        for i in range(len(zip_list)):
            min_supp.append(zip_list[i][0])
            min_func.append(zip_list[i][1])
        # Return the minimum random variable
        return pdf(RV(min_func, min_supp, ["discrete", "pdf"]))


def Mixture(MixParameters, MixRVs):
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
    # Check to make sure that the mix parameters are numeric
    # and sum to 1
    """
    total=0
    for i in range(len(MixParameters)):
        if isinstance(MixParameters[i], Symbol):
            raise RVError('ApplPy does not support symbolic mixtures')
        total+=MixParameters[i]
    if total<.9999 or total>1.0001:
        raise RVError('Mix parameters must sum to one')
    """
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
        # X1_dummy.drop_assumptions()
        # X2_dummy.drop_assumptions()
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


def ProductDiscrete(random_variable_1, random_variable_2):
    """
    Procedure Name: ProductDiscrete
    Purpose: Compute the product of two independent
                discrete random variables
    Arguments:  1. random_variable_1: A random variable
                2. random_variable_2: A random variable
    Output:     1. The product of random_variable_1 and random_variable_2
    """
    # Ensure that both random variables are discrete
    if not random_variable_1.is_discrete() or not random_variable_2.is_discrete():
        raise RVError("both random variables must be discrete")
    # Convert both random variables to pdf form
    X_dummy1 = pdf(random_variable_1)
    X_dummy2 = pdf(random_variable_2)
    # Convert the support and the value of each random variable
    #   into numpy arrays
    support1 = np.asarray(X_dummy1.support, dtype=object)
    support2 = np.asarray(X_dummy2.support, dtype=object)
    pdf1 = np.asarray(X_dummy1.func, dtype=object)
    pdf2 = np.asarray(X_dummy2.func, dtype=object)
    # Find all possible values of support1*support2 and val1*val2
    #   via the pairwise outer product, flatten into vectors
    prodsupport = np.outer(support1, support2).flatten()
    prodpdf = np.outer(pdf1, pdf2).flatten()
    #
    # Stack the support vector and the value vector into a matrix
    # prodmatrix=np.vstack([prodsupport,prodpdf]).T
    #
    #
    # Convert the resulting vectors into lists
    supportlist = prodsupport.tolist()
    pdflist = prodpdf.tolist()
    # Sort the function and support lists for the product
    sortlist = list(zip(supportlist, pdflist))
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


# Utilities
#
# Procedures:
#     Histogram(sample,bins)
#     LoadRV(filename)
#     PlotClear()
#     PlotDist(random_variable,suplist)
#     PlotDisplay(plot_list,suplist)
#     PlotEmpCDF(data)
#     PlotLimits(limits, axis)
#     PPPlot(random_variable,sample)
#     QQPlot(random_variable,sample)


def Histogram(sample, Bins=None):
    """
    Procedure: Histogram
    Purpose: Construct a histogram from a sample of data
    Arguments: 1. sample: The data sample from which to construct
                    the histogram
               2. bins: The number of bins in the histogram
    Output:    1. A histogram plot
    """
    # Check to ensure that the sample is given as a list
    if not isinstance(sample, list):
        raise RVError("The data sample must be entered as a list")

    sample.sort()
    if Bins is None:
        Bins = 1
        for i in range(1, len(sample)):
            if sample[i] != sample[i - 1]:
                Bins += 1

    plt.ion()
    plt.hist(sample, bins=Bins, normed=True)
    plt.ylabel("Relative Frequency")
    plt.xlabel("Observation Value")
    plt.title("Histogram")
    plt.grid(True)


def LoadRV(filename):
    """
    Procedure: LoadRV
    Purpose: Load a random variable from a binary file
    Aruments:   1. filename: the name of the file
                    where the random variable is stored
    Output:     1. The stored random variable
    """
    fileObject = open(filename, "r")
    random_variable = pickle.load(fileObject)
    if "RV" not in random_variable.__class__.__name__:
        print("WARNING: Object loaded is not a random variable")
    return random_variable


def PlotClear():
    """
    Procedure: PlotClear
    Purpose: Clears the plot display
    Arguments:  None
    Output:     1. Clear plot display
    """
    plt.clf()


def PlotLimits(limits, axis):
    """
    Procedure: PlotLimits
    Purpose: Sets the limits of a plot
    Arguments:  1. limits: A list of plot limits
    Output:     1. Plot with limits reset
    """
    axes = plt.gca()
    if axis == "x":
        axes.set_xlim(limits)
    elif axis == "y":
        axes.set_ylim(limits)
    else:
        err_str = 'The axis parameter in PlotLimits must be "x" or "y"'
        raise RVError(err_str)


def PlotDist(random_variable, suplist=None, opt=None, color="r", display=True):
    """
    Procedure: PlotDist
    Purpose: Plot a random variable
    Arguments:  1. random_variable: A random variable
                2. suplist: A list of supports for the plot
    Output:     1. A plot of the random variable
    """
    # Create the labels for the plot
    if random_variable.is_cdf():
        # lab1='F(x)'
        lab2 = "Cumulative Distribution Function"
    elif random_variable.is_chf():
        # lab1='H(x)'
        lab2 = "Cumulative Hazard Function"
    elif random_variable.is_hf():
        # lab1='h(x)'
        lab2 = "Hazard Function"
    elif random_variable.is_idf():
        # lab1='F-1(s)'
        lab2 = "Inverse Density Function"
    elif random_variable.is_pdf():
        # lab1='f(x)'
        lab2 = "Probability Density Function"
    elif random_variable.is_sf():
        # lab1='S(X)'
        lab2 = "Survivor Function"

    if opt == "EMPCDF":
        lab2 = "Empirical CDF"

    # If the distribution is continuous, plot the function
    if random_variable.is_continuous():
        # Return an error if the plot supports are not
        #   within the support of the random variable
        if suplist is not None:
            if suplist[0] > suplist[1]:
                raise RVError("Support list must be in ascending order")
            if suplist[0] < random_variable.support[0]:
                raise RVError("Plot supports must fall within RV support")
            if suplist[1] > random_variable.support[1]:
                raise RVError("Plot support must fall within RV support")
        # Cut out parts of the distribution that don't fall
        #   within the limits of the plot
        if suplist is None:
            # Since plotting is numeric, the lower support cannot be -oo
            if random_variable.support[0] == -oo:
                support1 = float(random_variable.variate(s=0.01)[0])
            else:
                support1 = float(random_variable.support[0])
            # Since plotting is numeric, the upper support cannot be oo
            if random_variable.support[len(random_variable.support) - 1] == oo:
                support2 = float(random_variable.variate(s=0.99)[0])
            else:
                support2 = float(random_variable.support[len(random_variable.support) - 1])
            suplist = [support1, support2]
        for i in range(len(random_variable.func)):
            if suplist[0] >= float(random_variable.support[i]):
                if suplist[0] <= float(random_variable.support[i + 1]):
                    lwindx = i
            if suplist[1] >= float(random_variable.support[i]):
                if suplist[1] <= float(random_variable.support[i + 1]):
                    upindx = i
        # Create a list of functions for the plot
        plotfunc = []
        for i in range(len(random_variable.func)):
            if i >= lwindx and i <= upindx:
                plotfunc.append(random_variable.func[i])
        # Create a list of supports for the plot
        plotsupp = [suplist[0]]
        upindx += 1
        for i in range(len(random_variable.support)):
            if i > lwindx and i < upindx:
                plotsupp.append(random_variable.support[i])
        plotsupp.append(suplist[1])

        # print plotfunc, plotsupp
        for i, function in enumerate(plotfunc):

            def f(y):
                return function.subs(x, y).evalf()

            x_range = np.arange(
                plotsupp[i], plotsupp[i + 1], abs(plotsupp[i + 1] - plotsupp[i]) / 1000
            )
            y_range = np.array([f(num) for num in x_range])
            plt.plot(x_range, y_range, color)
        plt.title(lab2)

        """
        Old plot method using the sympy plot
        plt.ioff()
        print plotfunc, plotsupp
        initial_plot=plot(plotfunc[0],(x,plotsupp[0],plotsupp[1]),
                          title=lab2,show=False,line_color=color)
        for i in range(1,len(plotfunc)):
            plot_extension=plot(plotfunc[i],
                               (x,plotsupp[i],plotsupp[i+1]),
                                show=False,line_color=color)
            initial_plot.append(plot_extension[0])
        if display is True:
            plt.ion()
            initial_plot.show()
            return initial_plot
        else:
            return initial_plot
        """

        # Old PlotDist code before sympy created the
        #   plotting front-end
        # print plotsupp
        # Parse the functions for matplotlib
        # plot_func=[]
        # for i in range(len(plotfunc)):
        #    strfunc=str(plotfunc[i])
        #    plot_func.append(strfunc)
        # print plot_func
        # Display the plot
        # if opt!='display':
        #    plt.ion()
        # plt.mat_plot(plot_func,plotsupp,lab1,lab2,'continuous')

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
        # if display==True:
        #    plt.ion()
        # plt.mat_plot(random_variable.func,random_variable.support,lab1,lab2,'discrete')
        plt.step(random_variable.support, random_variable.func)
        # plt.xlabel('x')
        # if lab1!=None:
        #    plt.ylabel(lab1)
        if lab2 is not None:
            plt.title(lab2)


def PlotDisplay(plot_list):
    if len(plot_list) < 2:
        raise RVError("PlotDisplay requires a list with multiple plots")
    plt.ion()
    totalplot = plot_list[0]
    for graph in plot_list[1:]:
        totalplot.append(graph[0])
    totalplot.show()


def PlotEmpCDF(data):
    """
    Procedure Name: PlotEmpCDF
    Purpose: Plots an empirical CDF, given a data set
    Arguments:  1. data: A data sample
    Output:     1. An empirical cdf of the data
    """

    # Create a bootstrap random variable from the data
    Xstar = BootstrapRV(data)
    PlotDist(cdf(Xstar), opt="EMPCDF")


def PPPlot(random_variable, sample):
    """
    Procedure Name: PPPlot
    Purpose: Plots the model probability versus the sample
                probability
    Arguments:  1. random_variable: A random variable
                2. sample: An experimental sample
    Output:     1. A PPPlot comparing the sample to a theoretical
                    model
    """
    # Return an error message if the sample is not given as
    #   a list
    if not isinstance(sample, list):
        raise RVError("The data sample must be given as a list")

    # Create a list of quantiles
    n = len(sample)
    sample.sort()
    plist = []
    for i in range(1, n + 1):
        p = (i - (1 / 2)) / n
        plist.append(p)

    # Create a list of CDF values for the sample and the
    # theoretical model
    FX = cdf(random_variable)
    fxstar = BootstrapRV(sample)
    FXstar = cdf(fxstar)

    FittedCDF = []
    ObservedCDF = []
    for i in range(len(plist)):
        FittedCDF.append(cdf(FX, sample[i]))
        ObservedCDF.append(cdf(FXstar, sample[i]))

    # Plot the results
    plt.ion()
    plt.prob_plot(ObservedCDF, FittedCDF, "PP Plot")


def QQPlot(random_variable, sample):
    """
    Procedure: QQPlot
    Purpose: Plots the q_i quantile of a fitted distribution
                versus the q_i quantile of the sample dist
    Arguments:  1. random_variable: A random variable
                2. sample: sample data
    Output:     1. QQ Plot
    """
    # Return an error message is the sample is not given as
    #   a list
    if not isinstance(sample, list):
        raise RVError("The data sample must be given as a list")

    # Create a list of quantiles
    n = len(sample)
    sample.sort()
    qlist = []
    for i in range(1, n + 1):
        q = (i - (1 / 2)) / n
        qlist.append(q)
    # Create 'fitted' list
    Fitted = []
    for i in range(len(qlist)):
        Fitted.append(random_variable.variate(s=qlist[i])[0])

    # Plot the results
    plt.ion()
    plt.prob_plot(sample, Fitted, "QQ Plot")


# Backward-compatible re-export for direct imports from applpy.rv.
