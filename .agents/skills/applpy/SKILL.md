---
name: applpy
description: Build APPLPy probability models from natural-language problem statements. Use when a request asks to translate prose into APPLPy random variables, distribution choices, composition operators (convolution, product, min/max, order statistics, mixtures, transforms, truncation), or Markov-chain/reliability constructions and return runnable Python modeling code plus key probability results.
---

# APPLPy Natural-Language Modeling

## Overview

Translate natural-language probability modeling requests into runnable APPLPy code.
Prefer APPLPy APIs that are already exercised in repository docs and functional tests.

## Workflow

1. Extract the model target from the prompt.
2. Identify variable type for each quantity: continuous RV, discrete RV, empirical/bootstrap RV, or Markov chain.
3. Select distributions and parameters that match the described support and behavior.
4. Compose derived random variables with APPLPy operators.
5. Compute requested outputs with `pdf`, `cdf`, `hf`, `mean`, `variance`, `expected_value`, quantiles via `.variate(s=...)`, or Markov-chain methods.
6. Return a complete Python snippet with imports and a short result summary.

## Parse The Request

Convert prose into explicit modeling assumptions before writing code.

- Determine support and domain constraints: bounded interval, nonnegative lifetime, event counts, binary outcomes, or finite states.
- Determine if variables are independent, identically distributed, correlated, or empirical from samples.
- Determine composition intent: sums, products, minima/maxima, order statistics, mixtures, or transformed variables.
- Determine output intent: distribution form, probability query, moments, percentile, system reliability, or long-run chain behavior.
- State assumptions when input is under-specified, then choose the simplest defensible model.

## Choose A Base Distribution

Use the distribution guides first.

- Discrete catalog and examples: `applpy/distributions/discrete/README.md`
- Continuous catalog and examples: `applpy/distributions/continuous/README.md`

Default mapping heuristics:

- Binary success/failure: `BernoulliRV(p)`
- Number of successes in `N` trials: `BinomialRV(N, p)`
- Count in interval: `PoissonRV(theta)`
- Waiting time/lifetime with memoryless hazard: `ExponentialRV(theta)`
- Positive skewed waiting times: `GammaRV(theta, kappa)` or `WeibullRV(theta, kappa)`
- Bounded proportion/rate on `(0,1)`: `BetaRV(alpha, beta)`
- Symmetric measurement noise: `NormalRV(mu, sigma)`
- Bounded expert estimate with mode: `TriangularRV(a, b, c)`
- Finite integer range with equal mass: `UniformDiscreteRV(a, b, k=1)`
- Bounded continuous range with equal density: `UniformRV(a, b)`

## Compose Models

Use these APPLPy construction patterns from `docs/examples.py` and `test_applpy/functional`.

- Sum of independent RVs: `x_rv + y_rv` or `convolution(x_rv, y_rv)`
- Sum of IID RVs: `convolution_iid(x_rv, n)`
- Product of RVs: `x_rv * y_rv` or `product(x_rv, y_rv)`
- Product of IID RVs: `product_iid(x_rv, n)`
- Max/min systems: `Maximum(...)`, `Minimum(...)`, `MaximumIID(x_rv, n)`, `MinimumIID(x_rv, n)`
- Order statistic: `OrderStat(x_rv, n, r)`
- Mixture model: `mixture([w1, ...], [rv1, ...])`
- Transform variable: `transform(x_rv, [[exprs...], [breakpoints...]])`
- Truncate support: `truncate(x_rv, [lower, upper])`
- Empirical/bootstrapped RV from samples: `bootstrap_rv(sample)`

For reliability block diagrams, model series with `Minimum*` and parallel with `Maximum*`, then compose subsystems.

## Markov-Chain Modeling

Use `MarkovChain` for discrete-time state-transition requests.

- Build with `MarkovChain(P=..., init=..., states=...)` or positional `MarkovChain(P, states=...)`.
- Path and conditional probabilities: `.probability(events, given=...)`
- Long-run behavior: `.long_run_probs(method="rational")`
- State structure: `.classify_states()`, `.reachability()`, `.reducible`
- Absorbing-chain questions: `.absorption_prob(state)`, `.absorption_steps()`

Use notebook-backed functional tests as behavior references:
`test_applpy/functional/test_discrete_time_markov_chain.py`.

## Output Template

Return runnable code using this shape.

```python
from sympy import Rational
from applpy import ...
from applpy.distributions.continuous import ...
from applpy.distributions.discrete import ...

# 1) define base RVs / chain
# 2) compose model
# 3) compute requested metrics
```

Then include a short text summary listing key assumptions and computed quantities.

## Validation Checklist

Before finalizing, verify:

- Parameters satisfy distribution constraints described in distribution READMEs.
- Mixture weights are nonnegative and sum to `1`.
- Support breakpoints and transform/truncation intervals are valid and ordered.
- Requested API names exist in `applpy/__init__.py` exports or module paths.
- The returned code includes all imports required to run as-is.

## Source Anchors

Use these files as primary references for skill behavior.

- `docs/examples.py`
- `test_applpy/functional/test_notebook_examples.py`
- `test_applpy/functional/test_paper_examples.py`
- `test_applpy/functional/test_reliability_block_diagrams.py`
- `test_applpy/functional/test_clt_examples.py`
- `test_applpy/functional/test_discrete_time_markov_chain.py`
- `applpy/distributions/discrete/README.md`
- `applpy/distributions/continuous/README.md`
