import pytest
from sympy import Integer, Rational

from applpy import Maximum, MaximumIID, Minimum, MinimumIID, OrderStat, RangeStat
import applpy.order_stat as order_stat_module
from applpy.order_stat import maximum, maximum_iid, minimum, minimum_iid, order_stat, range_stat
from applpy.rv import RV, RVError


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
    with pytest.raises(RVError, match="without replacement not implemented"):
        OrderStat(continuous, 3, 1, "wo")
    with pytest.raises(RVError, match="Only one item sampled"):
        RangeStat(discrete, 1, "w")
    with pytest.raises(RVError, match="current not implemented without"):
        RangeStat(discrete, 2, "wo")
    with pytest.raises(RVError, match="must both be discrete or continuous"):
        Maximum(continuous, discrete)
    with pytest.raises(RVError, match="must both be discrete or continuous"):
        Minimum(continuous, discrete)


def test_order_stat_discrete_singleton_path():
    singleton = RV([1], [5], ["discrete", "pdf"])
    assert OrderStat(singleton, 1, 1, "w").ftype == ["discrete", "pdf"]


def test_snake_case_names_match_compat_aliases():
    assert maximum is order_stat_module.Maximum
    assert maximum_iid is order_stat_module.MaximumIID
    assert minimum is order_stat_module.Minimum
    assert minimum_iid is order_stat_module.MinimumIID
    assert order_stat is order_stat_module.OrderStat
    assert range_stat is order_stat_module.RangeStat
