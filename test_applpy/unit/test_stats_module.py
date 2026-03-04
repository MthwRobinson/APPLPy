import pytest
from sympy import Rational, exp, oo, symbols

from applpy.dist_type import ExponentialRV, NormalRV, PoissonRV
from applpy.rv import RV, RVError, x
from applpy.stats import (
    KSTest,
    MLE,
    MOM,
)


def test_kstest_and_mom_symbolic_and_numeric_paths():
    theta = symbols("theta", positive=True)

    ks = KSTest(ExponentialRV(1), [0.1, 0.2, 0.3, 0.4])
    assert float(ks) > 0

    assert MOM(ExponentialRV(theta), [1, 2, 3], [theta]) == {theta: Rational(1, 2)}
    mom_numeric = MOM(ExponentialRV(theta), [1, 2, 3], [theta], guess=[2], numeric=True)
    assert float(mom_numeric[0]) == pytest.approx(0.5)

    with pytest.raises(RVError, match="initial guess"):
        MOM(ExponentialRV(theta), [1, 2, 3], [theta], numeric=True)


def test_mle_shortcuts_and_piecewise_rejection(capsys):
    mu, sigma = symbols("mu sigma")

    assert MLE(ExponentialRV(1), [1, 2, 3], []) == [Rational(1, 2)]
    assert MLE(PoissonRV(2), [1, 2, 3], [symbols("theta")]) == [2]

    normal_two = MLE(NormalRV(0, 1), [1, 2, 3], [mu, sigma])
    assert normal_two == [2, Rational(2, 3) ** Rational(1, 2)]

    normal_one = MLE(NormalRV(0, 1), [1, 2, 3], [mu])
    assert normal_one == normal_two
    assert "estimating mu and sigma" in capsys.readouterr().out

    piecewise = RV([x, 2 - x], [0, 1, 2], ["continuous", "pdf"])
    with pytest.raises(RVError, match="piecewise"):
        MLE(piecewise, [1, 2], [symbols("theta")])


def test_mle_generic_paths_and_censor_validation():
    theta = symbols("theta", positive=True)
    generic_exp = RV(theta * exp(-theta * x), [0, oo], ["continuous", "pdf"])

    assert MLE(generic_exp, [1, 2, 3], [theta]) == {theta: Rational(1, 2)}
    mle_numeric = MLE(generic_exp, [1, 2, 3], [theta], numeric=True, guess=[2])
    assert float(mle_numeric[0]) == pytest.approx(0.5)
    assert MLE(generic_exp, [1, 2, 3], [theta], censor=[1, 0, 1]) == []

    bad_value = MLE(generic_exp, [0.2, 0.4], [theta], censor=[1, 2])
    assert isinstance(bad_value, RVError)
    assert "only 1s and 0s" in str(bad_value)

    bad_length = MLE(generic_exp, [0.2, 0.4], [theta], censor=[1])
    assert isinstance(bad_length, RVError)
    assert "same length" in str(bad_length)


# NOTE: this is a long running test that I'll fix later
# def test_direct_mle_helper_functions():
#     assert MLEExponential([1, 2, 3]) == [Rational(1, 2)]
#     assert MLENormal([1, 2, 3]) == [2, Rational(2, 3) ** Rational(1, 2)]
#     assert MLENormal([1, 2, 3], mu=1) == [1, Rational(2, 3) ** Rational(1, 2)]
#     assert MLENormal([1, 2, 3], sigma=7) == [2, 7]
#     assert MLEPoisson([1, 2, 3]) == [2]
#
#     theta_hat, kappa_hat = MLEWeibull([1, 2, 3])
#     assert float(theta_hat) > 0
#     assert float(kappa_hat) > 0
#
#     theta_hat_c, kappa_hat_c = MLEWeibull([1, 2, 3], censor=[1, 0, 1])
#     assert float(theta_hat_c) > 0
#     assert float(kappa_hat_c) > 0
