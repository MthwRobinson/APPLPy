import pytest
from sympy import Rational, oo

from applpy.rv import (
    RV,
    RVError,
    BootstrapRV,
    CDF,
    Convert,
    NextCombination,
    NextPermutation,
    check_value,
    x,
)


def test_rv_init_wraps_scalar_function_and_sets_defaults():
    rv = RV(1, [0, 1], ["continuous", "pdf"])

    assert rv.func == [1]
    assert rv.support == [0, 1]
    assert rv.ftype == ["continuous", "pdf"]
    assert rv.cache is None


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda: RV(1, (0, 1), ["continuous", "pdf"]), "Support must be a list"),
        (
            lambda: RV(1, [0, 1], ["invalid", "pdf"]),
            "Random variables must either be discrete or continuous",
        ),
        (
            lambda: RV([1, 2], [0, 1], ["continuous", "pdf"]),
            "Support has incorrect number of elements",
        ),
        (
            lambda: RV([1], [2, 1], ["continuous", "pdf"]),
            "Support is not in ascending order",
        ),
    ],
)
def test_rv_init_validation_errors(builder, message):
    with pytest.raises(RVError, match=message):
        builder()


def test_len_and_eq_behaviors():
    rv_one = RV([x + x], [0, 1], ["continuous", "pdf"])
    rv_two = RV([2 * x], [0, 1], ["continuous", "pdf"])

    assert len(rv_one) == 1
    assert rv_one == rv_two

    with pytest.raises(RVError, match="only be checked for equality"):
        rv_one == 123


def test_add_to_cache_initializes_and_updates_cache():
    rv = RV(1, [0, 1], ["continuous", "pdf"])

    rv.add_to_cache("mean", Rational(1, 2))
    rv.add_to_cache("variance", Rational(1, 12))

    assert rv.cache == {"mean": Rational(1, 2), "variance": Rational(1, 12)}


def test_check_value_support_bounds():
    support = [0, 1]

    assert check_value(x, support) is True
    assert check_value(Rational(1, 2), support) is True
    assert check_value(-1, support) is False
    assert check_value(2, support) is False


def test_cdf_for_simple_continuous_pdf_and_cache():
    rv = RV(1, [0, 1], ["continuous", "pdf"])

    cdf_rv = CDF(rv, cache=True)

    assert cdf_rv.func == [x]
    assert cdf_rv.support == [0, 1]
    assert CDF(rv, Rational(1, 4)) == Rational(1, 4)
    assert CDF(rv, -1) == 0
    assert CDF(rv, 2) == 1
    assert rv.cache["cdf"] is cdf_rv
    assert CDF(rv) is cdf_rv


def test_bootstrap_rv_creates_discrete_pdf_with_frequencies():
    rv = BootstrapRV([3, 1, 3, 2])

    assert rv.ftype == ["discrete", "pdf"]
    assert rv.support == [1, 2, 3]
    assert rv.func == [Rational(1, 4), Rational(1, 4), Rational(1, 2)]


def test_convert_discrete_functional_to_explicit_form():
    functional_rv = RV([x], [1, 3], ["Discrete", "pdf"])

    explicit_rv = Convert(functional_rv)

    assert explicit_rv.ftype == ["discrete", "pdf"]
    assert explicit_rv.support == [1, 2, 3]
    assert explicit_rv.func == [1, 2, 3]


def test_convert_validation_errors():
    with pytest.raises(RVError, match="must be Discrete"):
        Convert(RV(1, [0, 1], ["continuous", "pdf"]))

    with pytest.raises(RVError, match="infinite support"):
        Convert(RV([x], [0, oo], ["Discrete", "pdf"]))


def test_next_combination_advances_lexicographically():
    assert NextCombination([1, 2, 4], 5) == [1, 2, 5]
    assert NextCombination([1, 4, 5], 5) == [2, 3, 4]


def test_next_permutation_advances_lexicographically_for_increasing_input():
    assert NextPermutation([1, 2, 3]) == [1, 3, 2]
