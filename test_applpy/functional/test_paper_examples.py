import pytest
from sympy import Rational, Symbol, sqrt

from applpy import (
    BootstrapRV,
    ChiRV,
    ConvolutionIID,
    ExponentialRV,
    Kurtosis,
    MarkovChain,
    MaximumIID,
    Mean,
    Minimum,
    RV,
    Skewness,
    Variance,
)


def test_exponential_sum_example_from_paper():
    x_rv = ExponentialRV(Rational(1, 2))
    y_rv = ExponentialRV(Rational(1, 3))
    z_rv = x_rv + y_rv

    assert Mean(z_rv) == 5


def test_discrete_convolution_moments_from_paper():
    x = Symbol("x")
    x_rv = RV([x / 21], [1, 6], ["Discrete", "pdf"])
    y_rv = ConvolutionIID(x_rv, 5)

    assert Mean(y_rv) == Rational(65, 3)
    assert Variance(y_rv) == Rational(100, 9)
    assert Skewness(y_rv) == Rational(-13, 50)
    assert Kurtosis(y_rv) == Rational(1431, 500)


def test_markov_chain_probability_from_paper():
    chain = MarkovChain(
        P=[
            [Rational(45, 100), Rational(48, 100), Rational(7, 100)],
            [Rational(5, 100), Rational(70, 100), Rational(25, 100)],
            [Rational(1, 100), Rational(50, 100), Rational(49, 100)],
        ],
        init=[Rational(5, 100), Rational(80, 100), Rational(15, 100)],
        states=["stormy", "cloudy", "sunny"],
    )

    assert chain.probability([(5, "cloudy")], method="rational") == Rational(974185141, 1562500000)


def test_chi_distribution_ratios_from_paper():
    assert float((Mean(ChiRV(3)) / sqrt(3)).evalf()) == pytest.approx(0.921317731923561)
    assert float((Mean(ChiRV(99)) / sqrt(99)).evalf()) == pytest.approx(0.997477976071264)


def test_bootstrap_system_means_from_paper():
    alist = [0.2135, 0.3153, 0.3841, 0.3946, 0.4707, 0.5107, 0.5783, 0.5960, 0.6404, 0.7601]
    blist = [
        0.0482,
        0.0945,
        0.1149,
        0.2441,
        0.287,
        0.311,
        0.3362,
        0.3924,
        0.4166,
        0.4194,
        0.6217,
        0.698,
    ]
    clist = [
        0.0474,
        0.2732,
        0.2828,
        0.2952,
        0.3370,
        0.3627,
        0.3698,
        0.4348,
        0.4968,
        0.5061,
        0.5093,
        0.5211,
        0.5266,
        0.6654,
        0.6951,
        0.6975,
        0.7694,
    ]

    a_rv = BootstrapRV(alist)
    b_rv = BootstrapRV(blist)
    c_rv = BootstrapRV(clist)

    s_star = Minimum(MaximumIID(a_rv, 2), Minimum(MaximumIID(b_rv, 2), MaximumIID(c_rv, 2)))
    s_star_1 = Minimum(MaximumIID(a_rv, 3), MaximumIID(b_rv, 3), MaximumIID(c_rv, 3))
    s = Minimum(MaximumIID(a_rv, 2), MaximumIID(b_rv, 2), MaximumIID(c_rv, 2))

    assert float(Mean(s_star).evalf()) == pytest.approx(0.37544247308727413)
    assert float(Mean(s_star_1).evalf()) == pytest.approx(0.438389764206852)
    assert float(Mean(s).evalf()) == pytest.approx(0.37544247308727413)
