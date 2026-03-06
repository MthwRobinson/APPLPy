# Continuous Distributions

This guide summarizes each continuous distribution currently provided in `applpy.distributions.continuous`.

## ArcSinRV
Description: Arc-sine distribution on `(0, 1)` with density concentrated near the boundaries.
Parameters: none.
Typical uses: proportion models with U-shaped behavior and boundary-heavy outcomes.

```python
from applpy import CDF
from applpy.distributions.continuous import ArcSinRV

x = ArcSinRV()
print(CDF(x, 0.5))
```

## ArcTanRV
Description: Arctangent-based continuous model with tunable location and scale behavior.
Parameters: `alpha` (scale-like, positive), `phi` (location/shift).
Typical uses: heavy-tail style modeling when you want a smooth CDF shape.

```python
from applpy import CDF
from applpy.distributions.continuous import ArcTanRV

x = ArcTanRV(alpha=2, phi=0)
print(CDF(x, 1))
```

## BetaRV
Description: Beta distribution on `(0, 1)` for bounded probabilities and rates.
Parameters: `alpha` (shape, positive), `beta` (shape, positive).
Typical uses: Bayesian priors for Bernoulli/binomial rates and bounded random effects.

```python
from applpy import Mean
from applpy.distributions.continuous import BetaRV

x = BetaRV(alpha=2, beta=5)
print(Mean(x))
```

## BivariateNormalRV
Description: Joint normal distribution for two correlated continuous variables.
Parameters: `mu` (shared location), `sigma1` (std. dev. of variable 1), `sigma2` (std. dev. of variable 2), `rho` (correlation in `[-1, 1]`).
Typical uses: correlated measurement errors and two-dimensional Gaussian modeling.

```python
from applpy.distributions.continuous import BivariateNormalRV

x = BivariateNormalRV(mu=0, sigma1=1, sigma2=2, rho=0.5)
print(x)
```

## CauchyRV
Description: Cauchy distribution with very heavy tails and undefined mean/variance.
Parameters: `a` (location), `alpha` (scale, positive).
Typical uses: robust modeling and processes with occasional extreme values.

```python
from applpy import CDF
from applpy.distributions.continuous import CauchyRV

x = CauchyRV(a=0, alpha=1)
print(CDF(x, 0))
```

## ChiRV
Description: Chi distribution for the square root of chi-square variables.
Parameters: `N` (degrees of freedom, positive integer).
Typical uses: norms of Gaussian vectors and magnitude-type measurements.

```python
from applpy import Mean
from applpy.distributions.continuous import ChiRV

x = ChiRV(N=4)
print(Mean(x))
```

## ChiSquareRV
Description: Chi-square distribution for sums of squared standard normals.
Parameters: `N` (degrees of freedom, positive integer).
Typical uses: variance inference, goodness-of-fit testing, and likelihood ratio methods.

```python
from applpy import CDF
from applpy.distributions.continuous import ChiSquareRV

x = ChiSquareRV(N=6)
print(CDF(x, 5))
```

## ErlangRV
Description: Erlang distribution (Gamma with integer shape) for waiting times.
Parameters: `theta` (scale, positive), `N` (shape, positive integer).
Typical uses: queueing and service-time models with staged exponential phases.

```python
from applpy import Mean
from applpy.distributions.continuous import ErlangRV

x = ErlangRV(theta=2, N=3)
print(Mean(x))
```

## ErrorRV
Description: Flexible continuous error-family model used in reliability/statistical fitting.
Parameters: `mu` (location-like), `alpha` (shape), `d` (shape).
Typical uses: non-normal error modeling with adjustable tail and shape behavior.

```python
from applpy import CDF
from applpy.distributions.continuous import ErrorRV

x = ErrorRV(mu=1, alpha=2, d=1)
print(CDF(x, 1))
```

## ErrorIIRV
Description: Type-II error-family distribution with tunable shape and scale.
Parameters: `a` (location/shape), `b` (scale/shape), `c` (shape).
Typical uses: reliability and skewed lifetime data modeling.

```python
from applpy import CDF
from applpy.distributions.continuous import ErrorIIRV

x = ErrorIIRV(a=0, b=1, c=2)
print(CDF(x, 1))
```

## ExponentialRV
Description: Exponential distribution with constant hazard rate.
Parameters: `theta` (scale, positive).
Typical uses: inter-arrival times, memoryless waiting-time systems, and reliability baselines.

```python
from applpy import CDF, Mean
from applpy.distributions.continuous import ExponentialRV

x = ExponentialRV(theta=10)
print(Mean(x))
print(CDF(x, 5))
```

## ExponentialPowerRV
Description: Exponential-power (generalized normal/Laplace family) distribution.
Parameters: `theta` (scale, positive), `kappa` (shape, positive).
Typical uses: peaked or heavy/light-tailed error models.

```python
from applpy import CDF
from applpy.distributions.continuous import ExponentialPowerRV

x = ExponentialPowerRV(theta=1, kappa=2)
print(CDF(x, 0))
```

## ExtremeValueRV
Description: Extreme-value (Gumbel-type) distribution for maxima/minima analysis.
Parameters: `alpha` (location), `beta` (scale).
Typical uses: block maxima, environmental extremes, and risk analysis.

```python
from applpy import CDF
from applpy.distributions.continuous import ExtremeValueRV

x = ExtremeValueRV(alpha=0, beta=1)
print(CDF(x, 1))
```

## FRV
Description: F distribution from a ratio of scaled chi-square random variables.
Parameters: `n1` (numerator degrees of freedom), `n2` (denominator degrees of freedom).
Typical uses: ANOVA, nested-model comparisons, and variance-ratio testing.

```python
from applpy import CDF
from applpy.distributions.continuous import FRV

x = FRV(n1=5, n2=10)
print(CDF(x, 1))
```

## GammaRV
Description: Gamma distribution for positive continuous quantities.
Parameters: `theta` (scale, positive), `kappa` (shape, positive).
Typical uses: waiting times, rainfall/claim amounts, and Bayesian priors for rates.

```python
from applpy import Mean
from applpy.distributions.continuous import GammaRV

x = GammaRV(theta=2, kappa=3)
print(Mean(x))
```

## GeneralizedParetoRV
Description: Generalized Pareto distribution with flexible tail behavior.
Parameters: `theta` (scale, positive), `delta` (location/threshold), `kappa` (shape/tail).
Typical uses: peaks-over-threshold extreme-value analysis.

```python
from applpy import CDF
from applpy.distributions.continuous import GeneralizedParetoRV

x = GeneralizedParetoRV(theta=1, delta=0, kappa=0.2)
print(CDF(x, 2))
```

## GompertzRV
Description: Gompertz distribution with exponentially changing hazard.
Parameters: `theta` (scale, positive), `kappa` (shape).
Typical uses: mortality and survival models.

```python
from applpy import Mean
from applpy.distributions.continuous import GompertzRV

x = GompertzRV(theta=1, kappa=0.5)
print(Mean(x))
```

## IDBRV
Description: Inverse distribution family used for skewed positive lifetimes.
Parameters: `theta` (scale/location), `delta` (shape/location), `kappa` (shape).
Typical uses: reliability and lifetime modeling with asymmetric tails.

```python
from applpy import CDF
from applpy.distributions.continuous import IDBRV

x = IDBRV(theta=1, delta=0, kappa=2)
print(CDF(x, 1))
```

## InverseGammaRV
Description: Inverse-gamma distribution for positive scales and variances.
Parameters: `alpha` (shape, positive), `beta` (scale, positive).
Typical uses: Bayesian priors for variance and precision-like quantities.

```python
from applpy import Mean
from applpy.distributions.continuous import InverseGammaRV

x = InverseGammaRV(alpha=3, beta=2)
print(Mean(x))
```

## InverseGaussianRV
Description: Inverse Gaussian (Wald) distribution for first-passage times.
Parameters: `theta` (scale, positive), `mu` (mean/location, positive).
Typical uses: diffusion hitting times and positive skewed duration data.

```python
from applpy import CDF
from applpy.distributions.continuous import InverseGaussianRV

x = InverseGaussianRV(theta=2, mu=1)
print(CDF(x, 1))
```

## KSRV
Description: Kolmogorov-Smirnov related distribution for KS statistics.
Parameters: `n` (sample size, positive integer).
Typical uses: modeling the finite-sample KS test statistic distribution.

```python
from applpy import CDF
from applpy.distributions.continuous import KSRV

x = KSRV(n=20)
print(CDF(x, 0.2))
```

## LaPlaceRV
Description: Laplace (double exponential) distribution with sharp center and heavier tails than normal.
Parameters: `omega` (scale, positive), `theta` (location).
Typical uses: robust error modeling and L1-style noise assumptions.

```python
from applpy import CDF
from applpy.distributions.continuous import LaPlaceRV

x = LaPlaceRV(omega=1, theta=0)
print(CDF(x, 0))
```

## LogGammaRV
Description: Log-gamma distribution on positive support with flexible skew.
Parameters: `alpha` (shape, positive), `beta` (scale/rate, positive).
Typical uses: skewed positive data and transformed gamma-type models.

```python
from applpy import Mean
from applpy.distributions.continuous import LogGammaRV

x = LogGammaRV(alpha=2, beta=1)
print(Mean(x))
```

## LogisticRV
Description: Logistic distribution with sigmoid CDF and moderate tails.
Parameters: `kappa` (scale, positive), `theta` (location).
Typical uses: latent-variable models and growth/response curves.

```python
from applpy import CDF
from applpy.distributions.continuous import LogisticRV

x = LogisticRV(kappa=1, theta=0)
print(CDF(x, 0))
```

## LogLogisticRV
Description: Log-logistic distribution with positive support and heavy tails.
Parameters: `theta` (scale, positive), `kappa` (shape, positive).
Typical uses: survival/reliability modeling and hazard functions with non-monotonic behavior.

```python
from applpy import CDF
from applpy.distributions.continuous import LogLogisticRV

x = LogLogisticRV(theta=1, kappa=2)
print(CDF(x, 1))
```

## LogNormalRV
Description: Log-normal distribution for multiplicative positive processes.
Parameters: `mu` (log-location), `sigma` (log-scale, positive).
Typical uses: finance, biological growth, and right-skewed duration/size data.

```python
from applpy import Mean
from applpy.distributions.continuous import LogNormalRV

x = LogNormalRV(mu=0, sigma=1)
print(Mean(x))
```

## LomaxRV
Description: Lomax (Pareto Type II) heavy-tailed positive distribution.
Parameters: `kappa` (shape, positive), `theta` (scale, positive).
Typical uses: claim severity, internet traffic, and heavy-tail reliability data.

```python
from applpy import CDF
from applpy.distributions.continuous import LomaxRV

x = LomaxRV(kappa=2, theta=1)
print(CDF(x, 2))
```

## MakehamRV
Description: Makeham survival distribution with age-dependent and constant hazard terms.
Parameters: `theta` (scale, positive), `delta` (positive shape), `kappa` (shape).
Typical uses: actuarial mortality and lifetime risk decomposition.

```python
from applpy import CDF
from applpy.distributions.continuous import MakehamRV

x = MakehamRV(theta=1, delta=1, kappa=0.2)
print(CDF(x, 1))
```

## MuthRV
Description: Muth distribution for right-skewed positive outcomes.
Parameters: `kappa` (shape, positive).
Typical uses: reliability and response-time modeling.

```python
from applpy import CDF
from applpy.distributions.continuous import MuthRV

x = MuthRV(kappa=1)
print(CDF(x, 1))
```

## NormalRV
Description: Normal (Gaussian) distribution.
Parameters: `mu` (mean/location), `sigma` (standard deviation, positive).
Typical uses: measurement noise, CLT approximations, and baseline parametric modeling.

```python
from applpy import CDF, Mean
from applpy.distributions.continuous import NormalRV

x = NormalRV(mu=0, sigma=1)
print(Mean(x))
print(CDF(x, 0))
```

## ParetoRV
Description: Pareto heavy-tail distribution on positive support.
Parameters: `theta` (scale/minimum, positive), `kappa` (shape/tail index, positive).
Typical uses: wealth modeling, file-size distributions, and tail-risk studies.

```python
from applpy import CDF
from applpy.distributions.continuous import ParetoRV

x = ParetoRV(theta=1, kappa=2)
print(CDF(x, 2))
```

## RayleighRV
Description: Rayleigh distribution for magnitudes of 2D normal vectors.
Parameters: `theta` (scale, positive).
Typical uses: signal processing, wind-speed approximations, and random vector magnitudes.

```python
from applpy import Mean
from applpy.distributions.continuous import RayleighRV

x = RayleighRV(theta=2)
print(Mean(x))
```

## TRV
Description: Student's t distribution with heavier tails than normal.
Parameters: `N` (degrees of freedom).
Typical uses: small-sample inference and robust mean modeling.

```python
from applpy import CDF
from applpy.distributions.continuous import TRV

x = TRV(N=10)
print(CDF(x, 0))
```

## TriangularRV
Description: Triangular distribution over a bounded interval with a mode.
Parameters: `a` (lower bound), `b` (mode), `c` (upper bound).
Typical uses: project planning and bounded expert-estimate inputs.

```python
from applpy import Mean
from applpy.distributions.continuous import TriangularRV

x = TriangularRV(a=0, b=2, c=5)
print(Mean(x))
```

## UniformRV
Description: Continuous uniform distribution over `[a, b]`.
Parameters: `a` (lower bound), `b` (upper bound).
Typical uses: non-informative bounded models and simulation primitives.

```python
from applpy import CDF
from applpy.distributions.continuous import UniformRV

x = UniformRV(a=0, b=1)
print(CDF(x, 0.25))
```

## WeibullRV
Description: Weibull distribution with flexible monotone hazard behavior.
Parameters: `theta` (scale, positive), `kappa` (shape, positive).
Typical uses: reliability engineering, failure-time analysis, and survival modeling.

```python
from applpy import Mean
from applpy.distributions.continuous import WeibullRV

x = WeibullRV(theta=2, kappa=1.5)
print(Mean(x))
```
