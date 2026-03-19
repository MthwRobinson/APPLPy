import pytest
from sympy import Integer, Rational, Symbol

from applpy import Mean as TopLevelMean
from applpy.moments import (
    CoefOfVar,
    Entropy,
    ExpectedValue,
    Kurtosis,
    Mean,
    MGF,
    Skewness,
    Variance,
)
from applpy.rv import RV, x


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def _discrete_pdf():
    return RV([Rational(1, 4), Rational(3, 4)], [1, 2], ["discrete", "pdf"])


def _functional_discrete_pdf():
    return RV([x], [1, 3], ["discrete_functional", "pdf"])


def test_moments_and_summary_statistics_for_multiple_ftypes():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()
    functional_discrete = _functional_discrete_pdf()

    assert Mean(continuous) == Rational(1, 2)
    assert Variance(continuous) == Rational(1, 12)
    assert ExpectedValue(continuous, x**2) == Rational(1, 3)
    assert Entropy(continuous) < 0
    assert MGF(continuous).subs(Symbol("t"), 0) == 1
    assert CoefOfVar(continuous) > 0
    assert Skewness(continuous) == 0
    assert Kurtosis(continuous) == Rational(9, 5)

    assert Mean(discrete) == Rational(7, 4)
    assert Variance(discrete) == Rational(3, 16)
    assert ExpectedValue(discrete, x**2) == Rational(13, 4)
    assert Entropy(discrete) > 0
    assert CoefOfVar(discrete) > 0
    assert Skewness(discrete) < 0
    assert Kurtosis(discrete) > 0

    assert Mean(functional_discrete) == 14
    assert Variance(functional_discrete) == -160


def test_expected_value_error_path():
    with pytest.raises(AttributeError, match="cache"):
        ExpectedValue("not-an-rv")


def test_variance_list_input():
    assert Variance([1, 2, 3]) == Rational(2, 3)


def test_top_level_mean_export():
    assert TopLevelMean(_uniform_continuous_pdf()) == Rational(1, 2)
