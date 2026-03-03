import pytest
from sympy import Symbol, sqrt

from applpy import ConvolutionIID, ExponentialRV, Mean, NormalRV, Transform, Variance


def _sample_mean_exact_rv(n):
    x = Symbol("x")
    x_rv = ExponentialRV(1)
    y_rv = ConvolutionIID(x_rv, n)
    return Transform(y_rv, [[x / n], [0, float("inf")]])


def test_exact_sample_mean_moments_from_clt_notebook():
    for n in (1, 3, 10):
        exact_rv = _sample_mean_exact_rv(n)
        assert Mean(exact_rv) == 1
        assert float(Variance(exact_rv).evalf()) == pytest.approx(1 / n)


def test_clt_90th_percentile_for_n3_from_notebook_workflow():
    n = 3
    exact_90 = _sample_mean_exact_rv(n).variate(s=0.90)[0]
    clt_90 = NormalRV(1, 1 / sqrt(n)).variate(s=0.90)[0]

    assert float(exact_90.evalf()) == pytest.approx(1.77056422475209)
    assert float(clt_90.evalf()) == pytest.approx(1.73990414134756)
    assert float((exact_90 - clt_90).evalf()) == pytest.approx(0.0306600834045304)


def test_clt_95th_percentile_difference_shrinks_with_n():
    differences = []
    for n in (1, 2, 3, 5, 10):
        exact_95 = _sample_mean_exact_rv(n).variate(s=0.95)[0]
        clt_95 = NormalRV(1, 1 / sqrt(n)).variate(s=0.95)[0]
        differences.append(float((exact_95 - clt_95).evalf()))

    assert all(diff > 0 for diff in differences)
    assert differences == sorted(differences, reverse=True)
