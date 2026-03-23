import pytest
from sympy import Integer, Rational, Symbol

import applpy.moments as moments_module
from applpy import mean as TopLevelMean
from applpy.moments import (
    coef_of_var,
    entropy,
    expected_value,
    kurtosis,
    mean,
    mgf,
    skewness,
    variance,
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

    assert mean(continuous) == Rational(1, 2)
    assert variance(continuous) == Rational(1, 12)
    assert expected_value(continuous, x**2) == Rational(1, 3)
    assert entropy(continuous) < 0
    assert mgf(continuous).subs(Symbol("t"), 0) == 1
    assert coef_of_var(continuous) > 0
    assert skewness(continuous) == 0
    assert kurtosis(continuous) == Rational(9, 5)

    assert mean(discrete) == Rational(7, 4)
    assert variance(discrete) == Rational(3, 16)
    assert expected_value(discrete, x**2) == Rational(13, 4)
    assert entropy(discrete) > 0
    assert coef_of_var(discrete) > 0
    assert skewness(discrete) < 0
    assert kurtosis(discrete) > 0

    assert mean(functional_discrete) == 14
    assert variance(functional_discrete) == -160


def test_expected_value_error_path():
    with pytest.raises(AttributeError, match="cache"):
        expected_value("not-an-rv")


def test_variance_list_input():
    assert variance([1, 2, 3]) == Rational(2, 3)


def test_top_level_mean_export():
    assert TopLevelMean(_uniform_continuous_pdf()) == Rational(1, 2)


def test_random_variable_is_discrete_branches_use_fast_rv(monkeypatch):
    discrete = _discrete_pdf()
    continuous = _uniform_continuous_pdf()
    fast_rv_inits = []

    class FakeFastRV:
        def __init__(self, *, function, support, functional_form, domain_type):
            fast_rv_inits.append(
                {
                    "function": function,
                    "support": support,
                    "functional_form": functional_form,
                    "domain_type": domain_type,
                }
            )

        def coefficient_of_variation(self):
            return Rational(11, 10)

        def entropy(self):
            return Rational(7, 10)

        def kurtosis(self):
            return Rational(13, 10)

        def skewness(self):
            return Rational(-3, 10)

    monkeypatch.setattr(moments_module, "FastRV", FakeFastRV)

    assert coef_of_var(discrete) == Rational(11, 10)
    assert entropy(discrete) == Rational(7, 10)
    assert kurtosis(discrete) == Rational(13, 10)
    assert skewness(discrete) == Rational(-3, 10)
    assert len(fast_rv_inits) == 4
    assert all(init["domain_type"] == "discrete" for init in fast_rv_inits)

    # Continuous random variables should not enter the random_variable.is_discrete() branches.
    assert coef_of_var(continuous) > 0
    assert entropy(continuous) < 0
    assert kurtosis(continuous) == Rational(9, 5)
    assert skewness(continuous) == 0
    assert len(fast_rv_inits) == 4
