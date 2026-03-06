"""GeneralizedParetoRV distribution."""


from sympy import (
    Symbol,
    exp,
    oo,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class GeneralizedParetoRV(RV):
    """
    Procedure Name: GeneralizedParetoRV
    Purpose: Creates an instance of the generalized pareto distribution
    Arguments:  1. theta: a strictly positive parameter
                2. delta: a real valued parameter
                3. kappa: a real valued parameter
    Output:     1. A generalized pareto random variable
    """

    def __init__(
        self, theta=Symbol("theta", positive=True), delta=Symbol("delta"), kappa=Symbol("kappa")
    ):
        x = Symbol("x", positive=True)
        if not isinstance(theta, Symbol):
            if theta <= 0:
                err_string = "theta must be positive"
                raise RVError(err_string)
        if theta in [-oo, oo] or delta in [-oo, oo] or kappa in [-oo, oo]:
            err_string = "all parameters must be finite"
            raise RVError(err_string)
        X_dummy = RV(
            (theta + kappa / (x + delta)) * (1 + x / delta) ** (-kappa) * exp(-theta * x), [0, oo]
        )
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["GeneralizedParetoRV"]
