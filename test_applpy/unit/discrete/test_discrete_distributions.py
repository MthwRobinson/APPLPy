import pytest
from sympy import Rational, oo

from applpy.distributions.discrete import (
    BenfordRV,
    BernoulliRV,
    BinomialRV,
    GeometricRV,
    PoissonRV,
    UniformDiscreteRV,
)
from applpy.rv import mean, RVError


def test_discrete_distribution_constructors_have_expected_structure():
    benford = BenfordRV()
    binomial = BinomialRV(4, Rational(1, 2))
    bernoulli = BernoulliRV(Rational(1, 3))
    geometric = GeometricRV(Rational(1, 4))
    poisson = PoissonRV(3)
    uniform_discrete = UniformDiscreteRV(1, 5, 2)

    assert benford.ftype == ["discrete_functional", "pdf"]
    assert benford.support == [1, 9]

    assert binomial.ftype == ["discrete_functional", "pdf"]
    assert binomial.support == [0, 4]

    assert bernoulli.ftype == ["discrete_functional", "pdf"]
    assert bernoulli.support == [0, 1]

    assert geometric.ftype == ["discrete_functional", "pdf"]
    assert geometric.support == [1, oo]

    assert poisson.ftype == ["discrete_functional", "pdf"]
    assert poisson.support == [0, oo]

    assert uniform_discrete.ftype == ["discrete", "pdf"]
    assert uniform_discrete.support == [1, 3, 5]
    assert uniform_discrete.func == [Rational(1, 3), Rational(1, 3), Rational(1, 3)]


def test_parameter_validation_errors_for_common_distributions():
    with pytest.raises(RVError, match="p must be between 0 and 1"):
        BinomialRV(2, 1)

    with pytest.raises(RVError, match="p must be between 0 and 1"):
        GeometricRV(0)

    with pytest.raises(RVError, match="theta must be positive"):
        PoissonRV(0)

    with pytest.raises(RVError, match="b is only valid if b > a"):
        UniformDiscreteRV(5, 4)


def test_uniform_discrete_validation_for_step_divisibility():
    with pytest.raises(RVError, match="divisble by k"):
        UniformDiscreteRV(0, 5, 2)


def test_binomial_mean_is_consistent():
    assert mean(BinomialRV(4, Rational(1, 2))) == 2
