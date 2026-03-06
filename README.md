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

Run a specific module directly with `pytest`:

```shell
uv run pytest -q test_applpy/functional/test_notebook_examples.py
```

After installation, prefer explicit module-level imports:

```python
from applpy.dist_type import ExponentialRV
from applpy.rv import CDF
```
