import pytest
from sympy import Rational, exp

from applpy import (
    BootstrapRV,
    CDF,
    ConvolutionIID,
    ExponentialRV,
    HF,
    Maximum,
    MaximumIID,
    Mean,
    Minimum,
    PDF,
    TriangularRV,
    UniformRV,
    Variance,
)


BALL_BEARING_SAMPLE = [
    17.88,
    28.92,
    33.00,
    41.52,
    42.12,
    45.60,
    48.48,
    51.84,
    51.96,
    54.12,
    55.56,
    67.80,
    68.64,
    68.64,
    68.88,
    84.12,
    93.12,
    98.64,
    105.12,
    105.84,
    127.92,
    128.04,
    173.40,
]


def test_exponential_examples_from_notebook():
    x_rv = ExponentialRV(Rational(1, 100))

    assert Mean(x_rv) == 100
    assert CDF(x_rv, 0) == 0
    assert HF(x_rv, 0) == Rational(1, 100)
    assert CDF(x_rv, 10) == 1 - exp(Rational(-1, 10))


def test_maximum_and_minimum_compositions_from_notebook():
    x_rv = ExponentialRV(Rational(1, 100))

    assert Mean(Maximum(x_rv, x_rv)) == 150
    assert Mean(MaximumIID(x_rv, 2)) == 150

    system_2 = Minimum(x_rv, Minimum(Maximum(x_rv, x_rv), Maximum(x_rv, x_rv)))
    assert Mean(system_2) == Rational(160, 3)


def test_bootstrap_examples_from_notebook():
    bootstrap_rv = BootstrapRV(BALL_BEARING_SAMPLE)
    max_2 = MaximumIID(bootstrap_rv, 2)
    conv_3 = ConvolutionIID(bootstrap_rv, 3)

    assert float(Mean(bootstrap_rv).evalf()) == pytest.approx(72.2243478260870)
    assert PDF(bootstrap_rv, 68.64) == Rational(2, 23)
    assert float(Mean(max_2).evalf()) == pytest.approx(92.1683931947070)
    assert float(Mean(conv_3).evalf()) == pytest.approx(216.673043478261)


def test_uniform_arithmetic_examples_from_notebook():
    uniform_1_to_2 = UniformRV(1, 2)
    uniform_neg_2_to_neg_1 = UniformRV(-2, -1)

    assert float(Mean(uniform_1_to_2 + uniform_1_to_2).evalf()) == pytest.approx(3.0)
    assert float(Mean(uniform_neg_2_to_neg_1 + uniform_neg_2_to_neg_1).evalf()) == pytest.approx(
        -3.0
    )
    assert Mean(uniform_1_to_2 + uniform_neg_2_to_neg_1) == 0

    assert float(Mean(uniform_1_to_2 * uniform_1_to_2).evalf()) == pytest.approx(2.25)
    assert float(Mean(uniform_neg_2_to_neg_1 * uniform_neg_2_to_neg_1).evalf()) == pytest.approx(
        2.25
    )
    assert Mean(uniform_1_to_2 * uniform_neg_2_to_neg_1) == 0


def test_triangular_distribution_examples_from_notebook():
    inv_1 = TriangularRV(-2, 1, 3)
    inv_2 = TriangularRV(-10, 3, 20)
    portfolio = inv_1 + inv_2

    assert CDF(inv_1, 0) == Rational(4, 15)
    assert Variance(inv_1) == Rational(19, 18)
    assert Mean(portfolio) == 5
    assert float(CDF(portfolio, 0).evalf()) == pytest.approx(0.226068376068376)
