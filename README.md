# APPLPy

### A Probability Programming Language -- Python Edition

## What is APPLPy?

APPLPy stands for A Probability Programming Language -- Python Edition. The primary goal of APPLPy is to provide an open-source conceptual probability package capable of manipulating random variables symbolically. Although the Python implementation is a recent development, a version based on the Maple computer algebra system has been used for over a decade. The Maple implementation, called APPL, has been successfully integrated into mathematical statistics and computing courses at the College of William and Mary, the United States Military Academy and Colorado College, while also facilitating research in areas ranging from order statistics to queuing theory. The hope of APPLPy is to make the computational capabilities of APPL available to researchers and educators on an open-source platform.

The current capabilities of APPLPy include:

1. Conversion between PDF,CDF,SF,HF,CHF and IDF representations of random variables
2. Computation of expected values, with both numeric and symbolic output
3. Plotting distributions, including piece-wise distributions
4. One-to-one and many-to-one transformations of piecewise distributions
5. Random Variable Algebra (Sums/Differences/Products/Division)
6. Random sampling from distributions
7. Bootstrapping data sets
8. Bayesian inference with ad-hoc prior distributions
9. Computation of distributions for M/M/s queues
10. Analysis of Discrete Time Markov Chains

## How is APPLPy Used?

Although APPLPy can be used for a variety of purposes, it is best suited to fill three special roles. First, it enables students to gain an intuitive understanding of mathematical statistics by automating tedious, calculus-intensive algorithms. As such, students can experiment with different models without having to perform difficult derivations or produce ad-hoc code. Second, it allows students to check hand derived results. This aids the learning process by providing a quick and reliable answer key. Finally, it allows researchers to explore systems whose properties would be intractable to derive by hand. As mentioned above, the Maple-based APPL software has already spawned a variety of insightful research. APPLPy has the potential to continue along this pathway. The simplicity of APPLPy's syntax allow users to explore stochastic models with ease.

# Installing APPLPy

APPLPy now targets Python 3 only.

Runtime dependencies are declared in the package metadata and will be installed automatically:

1. `sympy`
2. `numpy`
3. `scipy`
4. `mpmath`
5. `matplotlib`
6. `seaborn`
7. `pandas`

## Install with uv

Clone the repository and install with `uv`:

```shell
git clone https://github.com/MthwRobinson/APPLPy.git
cd APPLPy
uv sync
```

Or use the project `Makefile`:

```shell
make install
```

If you prefer `pip`-style uv commands, use:

```shell
uv pip install .
```

To install development dependencies (`ipython`, `jupyter`, and `ruff`) as well:

```shell
uv sync --extra dev
```

Or with `make`:

```shell
make install-dev
```

## Rust Extension Development (`pyo3` + `maturin`)

Install Rust extension tooling:

```shell
make install-rust
```

Build and install the Rust extension into the project environment:

```shell
make rust-develop
```

Use this for local development so Python imports the locally built Rust module from your active environment.

Validate import and call a dummy Rust function:

```shell
make rust-check-import
```

`--no-sync` is used for these commands so `uv run` does not prune the `maturin develop` install.

## Build Distribution Artifacts (Including Rust Bindings)

Create all distribution artifacts in `dist/`:

```shell
make build-dist-rust
```

This produces:

1. APPLPy Python source/wheel artifacts from `uv build`
2. A platform-specific Rust extension wheel from `maturin build`

## Development Commands

Use the `Makefile` targets for linting and formatting:

```shell
make check   # Run Ruff lint checks in applpy/
make tidy    # Apply Ruff fixes and format code in applpy/
```

## Running with Docker

Build the Docker image:

```shell
make docker-build
```

Run an interactive Python session in the container:

```shell
make docker-run
```

Run Jupyter Lab in Docker and expose it on `http://127.0.0.1:8888`:

```shell
make docker-run-jupter
```

## Running Tests Locally

Install test dependencies and run all tests with coverage:

```shell
make install-test
make test
```

Run just functional tests:

```shell
make test-functional
```

Run just unit tests:

```shell
make test-unit
```

Run a specific test file (or test path) instead of the full suite:

```shell
make test TEST=test_applpy/unit/test_rv.py
make test-functional TEST=test_applpy/functional/test_notebook_examples.py
make test-unit TEST=test_applpy/unit/test_rv.py
```

Run lint checks locally:

```shell
make check
```

## Example Workflows

The following are a few examples of how to use APPLPy.

### Exponential Distribution

```python
from sympy import Rational
from applpy import CDF, HF, Mean, ExponentialRV

x_rv = ExponentialRV(Rational(1, 100))
print(Mean(x_rv))   # 100
print(CDF(x_rv, 0)) # 0
print(HF(x_rv, 0))  # 1/100
```

### Triangular Portfolio

```python
from applpy import CDF, Mean, Variance, TriangularRV

inv_1 = TriangularRV(-2, 1, 3)
inv_2 = TriangularRV(-10, 3, 20)
portfolio = inv_1 + inv_2

print(CDF(inv_1, 0))   # 4/15
print(Variance(inv_1)) # 19/18
print(Mean(portfolio)) # 5
```

### Mixture and IID convolution

```python
from sympy import Rational
from applpy import CDF, convolution_iid, Mean, Mixture, TriangularRV, UniformRV

mix = Mixture(
    [Rational(1, 4), Rational(1, 4), Rational(1, 2)],
    [TriangularRV(2, 4, 6), TriangularRV(3, 5, 7), TriangularRV(1, 5, 9)],
)

print(Mean(mix))                  # 19/4
print(float(CDF(mix, 4).evalf())) # 0.296875
print(float(Mean(convolution_iid(UniformRV(1, 2), 3)).evalf())) # 4.5
```

## Distribution References

See the following documentation for a list of distributions that are available in the package.

- [Continuous Distributions](applpy/distributions/continuous/README.md)
- [Discrete Distributions](applpy/distributions/discrete/README.md)

## Using The APPLPy Agent Skill

This repository includes a local agent skill at `.agents/skills/applpy` for translating natural-language probability questions into APPLPy models and runnable code.

Invoke it directly in your prompt with:

```text
$applpy Model a 3-component reliability system where each component lifetime is Exponential(1/7), compute the system mean lifetime.
```

You can also call it with an explicit path:

```text
Use $applpy at /home/matt/repos/applpy/.agents/skills/applpy to build a Markov chain model from this transition matrix and compute long-run probabilities.
```
