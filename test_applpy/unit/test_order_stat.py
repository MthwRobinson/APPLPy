import importlib

import pytest
from sympy import Integer, Rational

from applpy import Maximum, MaximumIID, Minimum, MinimumIID, OrderStat, RangeStat
from applpy.order_stat import maximum_iid, minimum_iid, order_stat, range_stat
from applpy.rv import RV, RVError

order_stat_module = importlib.import_module("applpy.order_stat")


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def _discrete_pdf():
    return RV([Rational(1, 4), Rational(3, 4)], [1, 2], ["discrete", "pdf"])


def _discrete_pdf_bernoulli():
    return RV([Rational(1, 2), Rational(1, 2)], [0, 1], ["discrete", "pdf"])


def test_order_stat_operations_for_continuous_and_discrete():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()
    bernoulli = _discrete_pdf_bernoulli()

    assert isinstance(MaximumIID(continuous, 2), RV)
    assert isinstance(MaximumIID(discrete, 2), RV)
    assert isinstance(MinimumIID(continuous, 2), RV)
    assert isinstance(MinimumIID(discrete, 2), RV)
    assert isinstance(OrderStat(continuous, 3, 2), RV)
    assert isinstance(OrderStat(discrete, 3, 2, "w"), RV)
    assert isinstance(OrderStat(bernoulli, 2, 1, "wo"), RV)
    assert isinstance(RangeStat(continuous, 3), RV)
    assert isinstance(RangeStat(discrete, 2), RV)
    assert isinstance(Maximum(continuous, continuous), RV)
    assert isinstance(Maximum(discrete, bernoulli), RV)
    assert isinstance(Minimum(continuous, continuous), RV)
    assert isinstance(Minimum(discrete, bernoulli), RV)
    assert isinstance(Maximum(continuous, continuous, continuous), RV)
    assert isinstance(Minimum(discrete, discrete, discrete), RV)


def test_order_stat_error_paths():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()

    with pytest.raises(RVError, match="greater than the sample size"):
        OrderStat(continuous, 2, 3)
    with pytest.raises(RVError, match="Replace must be w or wo"):
        OrderStat(continuous, 3, 1, "invalid")
    with pytest.raises(RVError, match="Only one item sampled"):
        RangeStat(discrete, 1, "w")
    with pytest.raises(RVError, match="must both be discrete or continuous"):
        Maximum(continuous, discrete)
    with pytest.raises(RVError, match="must both be discrete or continuous"):
        Minimum(continuous, discrete)


def test_order_stat_discrete_singleton_path():
    singleton = RV([1], [5], ["discrete", "pdf"])
    assert OrderStat(singleton, 1, 1, "w").ftype == ["discrete", "pdf"]


def test_range_stat_discrete_with_replacement_distribution():
    discrete_uniform = RV(
        [Rational(1, 4), Rational(1, 4), Rational(1, 4), Rational(1, 4)],
        [1, 2, 3, 4],
        ["discrete", "pdf"],
    )

    result = RangeStat(discrete_uniform, 2, "w")

    assert result.support == [0, 1, 2, 3]
    assert result.func == [Rational(1, 4), Rational(3, 8), Rational(1, 4), Rational(1, 8)]
    assert result.ftype == ["discrete", "pdf"]


def test_range_stat_discrete_without_replacement_distribution():
    discrete_uniform = RV(
        [Rational(1, 4), Rational(1, 4), Rational(1, 4), Rational(1, 4)],
        [1, 2, 3, 4],
        ["discrete", "pdf"],
    )

    result = RangeStat(discrete_uniform, 2, "wo")

    assert result.support == [1, 2, 3]
    assert result.func == [Rational(1, 2), Rational(1, 3), Rational(1, 6)]
    assert result.ftype == ["discrete", "pdf"]


def test_snake_case_names_match_compat_aliases():
    assert order_stat_module.maximum is order_stat_module.Maximum
    assert maximum_iid is order_stat_module.MaximumIID
    assert order_stat_module.minimum is order_stat_module.Minimum
    assert minimum_iid is order_stat_module.MinimumIID
    assert order_stat is order_stat_module.OrderStat
    assert range_stat is order_stat_module.RangeStat
    assert order_stat_module.maximum_rv is order_stat_module.MaximumRV
    assert order_stat_module.minimum_rv is order_stat_module.MinimumRV
