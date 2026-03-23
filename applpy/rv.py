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

from applpy import rust_bindings

from sympy import (
    Symbol,
    symbols,
    oo,
    integrate,
    summation,
    simplify,
    pprint,
    expand,
    latex,
    Rational,
)
from random import random
from enum import Enum

x, y, z, t = symbols("x y z t")


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
        from .transform import transform

        gX = [[-x], [-oo, oo]]
        neg = transform(self, gX)
        return neg

    def __abs__(self):
        """
        Procedure Name: __abs__
        Purpose: Implements the behavior of random variables passed to the
                    abs() function
        Arguments:  1. self: the random variable
        Output:     1. The absolute value of the random variable
        """
        from .transform import transform

        gX = [[abs(x)], [-oo, oo]]
        abs_rv = transform(self, gX)
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
        from .algebra import convolution
        from .transform import transform

        if "RV" in other.__class__.__name__:
            try:
                return convolution(self, other)
            except Exception:
                return convolution(other, self)
            else:
                raise RVError("Could not compute the convolution")
        # If the random variable is added to a constant, shift
        # the random variable
        if isinstance(other, (float, int)):
            gX = [[x + other], [-oo, oo]]
            return transform(self, gX)

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
        from .algebra import convolution
        from .transform import transform

        if "RV" in other.__class__.__name__:
            gX = [[-x], [-oo, oo]]
            random_variable = transform(other, gX)
            return convolution(self, random_variable)
        # If the random variable is subtracted by a constant, shift
        # the random variable
        if isinstance(other, (float, int)):
            gX = [[x - other], [-oo, oo]]
            return transform(self, gX)

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
        from .algebra import product
        from .transform import transform

        if "RV" in other.__class__.__name__:
            try:
                return product(self, other)
            except Exception:
                return product(other, self)
            else:
                raise RVError("Could not compute the product")
        # If the random variable is multiplied by a constant, scale
        # the random variable
        if isinstance(other, (float, int)):
            gX = [[x * other], [-oo, oo]]
            return transform(self, gX)

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
        from .algebra import product
        from .transform import transform

        if "RV" in other.__class__.__name__:
            gX = [[1 / x, 1 / x], [-oo, 0, oo]]
            random_variable = transform(other, gX)
            return product(self, random_variable)
        # If the random variable is divided by a constant, scale
        # the random variable by theinverse of the constant
        if isinstance(other, (float, int)):
            gX = [[x / other], [-oo, oo]]
            return transform(self, gX)

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
        from .transform import transform

        gX = [[1 / x, 1 / x], [-oo, 0, oo]]
        invert = transform(self, gX)
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

        from .transform import power

        pow_rv = power(self, n)
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
        from .conversion import pdf

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

        from .conversion import cdf, idf, pdf
        from .moments import mean

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


def bootstrap_rv(varlist, symbolic=False):
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


def verify_pdf(random_variable):
    """
    Procedure Name: verify_pdf
    Purpose: Calls self.verify_pdf(). For compatibility with
                original APPL syntax
    Arguments:  1. random_variable: a discrete random variable
    Output:     1. A function call to self.verify_pdf()
    """
    return random_variable.verify_pdf()


# Backward-compatible aliases for legacy APPLPy function names.
BootstrapRV = bootstrap_rv
VerifyPDF = verify_pdf
