import pytest
from sympy import Integer, Rational

from applpy import mixture as top_level_mixture
from applpy import transform as top_level_transform
from applpy import truncate as top_level_truncate
from applpy.rv import RV, RVError, x
from applpy.transform import mixture, transform, truncate


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def _piecewise_continuous_pdf():
    return RV([x, 2 - x], [0, 1, 2], ["continuous", "pdf"])


def _discrete_pdf():
    return RV([Rational(1, 4), Rational(3, 4)], [1, 2], ["discrete", "pdf"])


def _discrete_pdf_bernoulli():
    return RV([Rational(1, 2), Rational(1, 2)], [0, 1], ["discrete", "pdf"])


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
