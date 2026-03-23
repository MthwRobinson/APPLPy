import pytest
from sympy import Integer, Rational, oo

from applpy import mixture as top_level_mixture
from applpy import transform as top_level_transform
from applpy import truncate as top_level_truncate
from applpy.rv import RV, RVError, x
from applpy.transform import (
    Convert,
    Pow,
    Sqrt,
    convert,
    mixture,
    power,
    sqrt,
    transform,
    truncate,
)


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def _piecewise_continuous_pdf():
    return RV([x, 2 - x], [0, 1, 2], ["continuous", "pdf"])


def _discrete_pdf():
    return RV([Rational(1, 4), Rational(3, 4)], [1, 2], ["discrete", "pdf"])


def _discrete_pdf_bernoulli():
    return RV([Rational(1, 2), Rational(1, 2)], [0, 1], ["discrete", "pdf"])


def _functional_discrete_pdf():
    return RV([x], [1, 3], ["discrete_functional", "pdf"])


def test_top_level_imports_still_work():
    assert top_level_transform is not None
    assert top_level_truncate is not None
    assert top_level_mixture is not None


def test_transform_and_truncate_happy_paths():
    continuous = _uniform_continuous_pdf()
    piecewise = _piecewise_continuous_pdf()
    discrete = _discrete_pdf()

    assert isinstance(transform(discrete, [[x + 1, x + 2], [0, 1, 2]]), RV)
    assert isinstance(transform(piecewise, [[x, x**2], [0, 1, 2]]), RV)
    assert isinstance(truncate(continuous, [Rational(1, 4), Rational(3, 4)]), RV)
    assert isinstance(truncate(discrete, [1, 1]), RV)


def test_mixture_happy_paths():
    continuous = _uniform_continuous_pdf()
    piecewise = _piecewise_continuous_pdf()
    discrete = _discrete_pdf()
    bernoulli = _discrete_pdf_bernoulli()

    assert isinstance(mixture([Rational(1, 3), Rational(2, 3)], [continuous, piecewise]), RV)
    assert isinstance(mixture([Rational(1, 2), Rational(1, 2)], [discrete, bernoulli]), RV)


def test_transform_and_mixture_error_paths():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()

    with pytest.raises(RVError, match="not in ascending order"):
        transform(_uniform_continuous_pdf(), [[x], [1, 0]])
    with pytest.raises(RVError, match="same length"):
        mixture([Rational(1, 2)], [continuous, continuous])
    with pytest.raises(RVError, match="all continuous or discrete"):
        mixture([Rational(1, 2), Rational(1, 2)], [continuous, discrete])


def test_convert_discrete_functional_to_explicit_form():
    functional_rv = _functional_discrete_pdf()

    explicit_rv = convert(functional_rv)

    assert explicit_rv.ftype == ["discrete", "pdf"]
    assert explicit_rv.support == [1, 2, 3]
    assert explicit_rv.func == [1, 2, 3]


def test_convert_validation_errors():
    with pytest.raises(RVError, match="must be discrete_functional"):
        convert(RV(1, [0, 1], ["continuous", "pdf"]))

    with pytest.raises(RVError, match="infinite support"):
        convert(RV([x], [0, oo], ["discrete_functional", "pdf"]))


def test_pow_and_sqrt_happy_paths():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()

    assert isinstance(power(continuous, 2), RV)
    assert isinstance(power(discrete, 2), RV)
    assert isinstance(sqrt(continuous), RV)
    assert isinstance(sqrt(discrete), RV)


def test_pow_requires_integer():
    with pytest.raises(RVError, match="must be an integer"):
        power(_uniform_continuous_pdf(), Rational(3, 2))


def test_sqrt_negative_support_error():
    negative_support = RV(Integer(1), [-1, 0], ["continuous", "pdf"])
    with pytest.raises(RVError, match="negative value appears in the support"):
        sqrt(negative_support)


def test_legacy_aliases_for_convert_pow_sqrt():
    continuous = _uniform_continuous_pdf()
    functional_rv = _functional_discrete_pdf()

    assert Convert(functional_rv) == convert(functional_rv)
    assert Pow(continuous, 2) == power(continuous, 2)
    assert Sqrt(continuous) == sqrt(continuous)
