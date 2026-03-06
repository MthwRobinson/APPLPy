"""BivariateNormalRV distribution."""

from sympy import Symbol, exp, oo, pi, sqrt

from ...bivariate import BivariateRV
from ...rv import RVError


class BivariateNormalRV(BivariateRV):
    """
    Procedure Name: BivariateNormalRV
    Purpose: Creates an instance of the bivariate normal distribution
    Arugments:  1. mu: a real valued parameter
                2. sigma1: a strictly positive parameter
                3. sigma2: a strictly positive parameter
                4. rho: a parameter >=0 and <=1
    Output:     1. A bivariate normal random variable
    """

    def __init__(
        self,
        mu=Symbol("mu"),
        sigma1=Symbol("sigma1", positive=True),
        sigma2=Symbol("sigma2", positive=True),
        rho=Symbol("rho"),
    ):
        if not isinstance(rho, Symbol):
            if rho < 0 or rho > 1:
                err_string = "rho must be >=0 and <=1"
                raise RVError(err_string)
        if not isinstance(sigma1, Symbol):
            if sigma1 <= 0:
                err_string = "sigma1 must be positive"
                raise RVError(err_string)
        if not isinstance(sigma2, Symbol):
            if sigma2 <= 0:
                err_string = "sigma2 must be positive"
                raise RVError(err_string)

        pdf_func = (1 / (2 * pi * sigma1 * sigma2 * sqrt(1 - rho**2))) * exp(
            -mu / (2 * (1 - rho**2))
        )
        X_dummy = BivariateRV([pdf_func], [[oo], [oo]], ["continuous", "pdf"])

        self.func = X_dummy.func
        self.constraints = X_dummy.constraints
        self.ftype = X_dummy.ftype


__all__ = ["BivariateNormalRV"]
