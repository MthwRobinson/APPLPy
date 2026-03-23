import pytest
from sympy import Integer, Rational

from applpy import Mixture as TopLevelMixture
from applpy import Transform as TopLevelTransform
from applpy import Truncate as TopLevelTruncate
from applpy.rv import RV, RVError, x
from applpy.transform import Mixture, Transform, Truncate


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def _piecewise_continuous_pdf():
    return RV([x, 2 - x], [0, 1, 2], ["continuous", "pdf"])


def _discrete_pdf():
    return RV([Rational(1, 4), Rational(3, 4)], [1, 2], ["discrete", "pdf"])


def _discrete_pdf_bernoulli():
    return RV([Rational(1, 2), Rational(1, 2)], [0, 1], ["discrete", "pdf"])


def test_top_level_imports_still_work():
    assert TopLevelTransform is not None
    assert TopLevelTruncate is not None
    assert TopLevelMixture is not None


def test_transform_and_truncate_happy_paths():
    continuous = _uniform_continuous_pdf()
    piecewise = _piecewise_continuous_pdf()
    discrete = _discrete_pdf()

    assert isinstance(Transform(discrete, [[x + 1, x + 2], [0, 1, 2]]), RV)
    assert isinstance(Transform(piecewise, [[x, x**2], [0, 1, 2]]), RV)
    assert isinstance(Truncate(continuous, [Rational(1, 4), Rational(3, 4)]), RV)
    assert isinstance(Truncate(discrete, [1, 1]), RV)


def test_mixture_happy_paths():
    continuous = _uniform_continuous_pdf()
    piecewise = _piecewise_continuous_pdf()
    discrete = _discrete_pdf()
    bernoulli = _discrete_pdf_bernoulli()

    assert isinstance(Mixture([Rational(1, 3), Rational(2, 3)], [continuous, piecewise]), RV)
    assert isinstance(Mixture([Rational(1, 2), Rational(1, 2)], [discrete, bernoulli]), RV)


def test_transform_and_mixture_error_paths():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()

    with pytest.raises(RVError, match="not in ascending order"):
        Transform(_uniform_continuous_pdf(), [[x], [1, 0]])
    with pytest.raises(RVError, match="same length"):
        Mixture([Rational(1, 2)], [continuous, continuous])
    with pytest.raises(RVError, match="all continuous or discrete"):
        Mixture([Rational(1, 2), Rational(1, 2)], [continuous, discrete])

