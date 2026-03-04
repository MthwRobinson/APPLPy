import pytest
from sympy import Rational, symbols

from applpy.bayes import (
    BayesMenu,
    BayesUpdate,
    CredibleSet,
    JeffreysPrior,
    Posterior,
    PosteriorPredictive,
)
from applpy.rv import RV, RVError, x


def test_bayes_menu_prints_procedure_list(capsys):
    BayesMenu()
    out = capsys.readouterr().out
    assert "Bayesian Statistics Procedures" in out
    assert "Posterior" in out


def test_posterior_continuous_prior_and_posterior_predictive():
    theta = symbols("theta")
    like = RV(theta * x + 1, [0, 1], ["continuous", "pdf"])
    prior = RV(1, [0, 1], ["continuous", "pdf"])

    post = Posterior(like, prior, [Rational(1, 2)], theta)
    assert post.ftype == ["continuous", "pdf"]
    assert post.support == [0, 1]
    assert post.func[0] == 2 * x / 5 + Rational(4, 5)

    post_pred = PosteriorPredictive(like, prior, [Rational(1, 2)], theta)
    assert post_pred.ftype == ["continuous", "pdf"]
    assert post_pred.support == [0, 1]
    assert post_pred.func[0] == 8 * x / 15 + 1


def test_posterior_rejects_non_symbol_parameter():
    theta = symbols("theta")
    like = RV(theta * x + 1, [0, 1], ["continuous", "pdf"])
    prior = RV(1, [0, 1], ["continuous", "pdf"])

    with pytest.raises(RVError, match="must be a symbol"):
        Posterior(like, prior, [Rational(1, 2)], 3)


def test_credible_set_and_alpha_validation():
    post = RV(1, [0, 1], ["continuous", "pdf"])
    lower, upper = CredibleSet(post, 0.2)
    assert lower <= upper

    with pytest.raises(RVError, match="between 0 and 1"):
        CredibleSet(post, 1.2)


def test_jeffreys_prior_for_continuous_likelihood():
    theta = symbols("theta")
    like = RV(theta * x + 1, [0, 1], ["continuous", "pdf"])

    jeff = JeffreysPrior(like, 0, 1, theta)
    assert jeff.ftype == ["continuous", "pdf"]
    assert jeff.support == [0, 1]
    assert jeff.func[0].has(x)


def test_discrete_prior_path_currently_raises_attribute_error():
    theta = symbols("theta")
    like = RV(theta * x + 1, [0, 1], ["continuous", "pdf"])
    prior_discrete = RV([Rational(1, 2), Rational(1, 2)], [0, 1], ["discrete", "pdf"])

    with pytest.raises(AttributeError, match="fype"):
        Posterior(like, prior_discrete, [Rational(1, 2)], theta)


def test_bayes_update_currently_recurses_until_error():
    theta = symbols("theta")
    like = RV(theta * x + 1, [0, 1], ["continuous", "pdf"])
    prior = RV(1, [0, 1], ["continuous", "pdf"])

    with pytest.raises(RecursionError):
        BayesUpdate(like, prior, [Rational(1, 2)], theta)
