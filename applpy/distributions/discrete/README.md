# Discrete Distributions

This guide summarizes each discrete distribution currently provided in `applpy.distributions.discrete`.

## BenfordRV
Description: Benford's law distribution for leading digits.
Parameters: none.
Typical uses: fraud/anomaly detection and first-digit frequency checks.

```python
from applpy.conversion import pdf
from applpy.distributions.discrete import BenfordRV

x = BenfordRV()
print(pdf(x, 1))
```

## BernoulliRV
Description: Bernoulli distribution for single yes/no trials.
Parameters: `p` (success probability in `(0, 1)`).
Typical uses: binary outcomes, click/no-click events, and pass/fail indicators.

```python
from applpy.moments import mean
from applpy.distributions.discrete import BernoulliRV

x = BernoulliRV(p=0.3)
print(mean(x))
```

## BinomialRV
Description: Binomial distribution for number of successes across fixed independent trials.
Parameters: `N` (number of trials, positive integer), `p` (success probability in `(0, 1)`).
Typical uses: conversion counts, defect counts, and repeated binary experiments.

```python
from applpy.conversion import cdf
from applpy.distributions.discrete import BinomialRV

x = BinomialRV(N=10, p=0.4)
print(cdf(x, 5))
```

## GeometricRV
Description: Geometric distribution for number of trials until first success.
Parameters: `p` (success probability in `(0, 1)`).
Typical uses: waiting-time counts for repeated Bernoulli processes.

```python
from applpy.moments import mean
from applpy.distributions.discrete import GeometricRV

x = GeometricRV(p=0.25)
print(mean(x))
```

## PoissonRV
Description: Poisson distribution for event counts in a fixed interval.
Parameters: `theta` (rate/intensity, positive).
Typical uses: arrivals, incident counts, and rare-event modeling.

```python
from applpy.conversion import cdf
from applpy.distributions.discrete import PoissonRV

x = PoissonRV(theta=3)
print(cdf(x, 2))
```

## UniformDiscreteRV
Description: Discrete uniform distribution over equally likely integer values in a range.
Parameters: `a` (lower bound), `b` (upper bound), `k` (step size, default `1`).
Typical uses: fair integer sampling and simple finite-state simulations.

```python
from applpy.conversion import pdf
from applpy.distributions.discrete import UniformDiscreteRV

x = UniformDiscreteRV(a=1, b=6)
print(pdf(x, 3))
```
