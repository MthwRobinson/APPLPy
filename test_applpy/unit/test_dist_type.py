import pytest
from sympy import Rational, exp, ln, oo

from applpy.dist_type import (
    ArcSinRV,
    BernoulliRV,
    BinomialRV,
    BivariateNormalRV,
    ExponentialRV,
    GeometricRV,
    PoissonRV,
    UniformDiscreteRV,
    UniformRV,
    WeibullRV,
    param_check,
)
from applpy.rv import CDF, Mean, RVError, x


def test_param_check_detects_unspecified_symbol_parameters():
    assert param_check([1, Rational(1, 2)]) is True
    assert param_check(ExponentialRV().parameter) is False


def test_continuous_distribution_constructors_have_expected_structure():
    arcsin = ArcSinRV()
    exponential = ExponentialRV(2)
    uniform = UniformRV(0, 4)
    weibull = WeibullRV(2, 3)

    assert arcsin.ftype == ["continuous", "pdf"]
    assert arcsin.support == [0, 1]

    assert exponential.ftype == ["continuous", "pdf"]
    assert exponential.support == [0, oo]

    assert uniform.ftype == ["continuous", "pdf"]
    assert uniform.func[0] == Rational(1, 4).evalf()
    assert uniform.support == [0, 4]

    assert weibull.ftype == ["continuous", "pdf"]
    assert weibull.support == [0, oo]
    assert weibull.cdf.subs(x, Rational(1, 2)) == 1 - exp(-1)


def test_discrete_distribution_constructors_have_expected_structure():
    binomial = BinomialRV(4, Rational(1, 2))
    bernoulli = BernoulliRV(Rational(1, 3))
    geometric = GeometricRV(Rational(1, 4))
    poisson = PoissonRV(3)
    uniform_discrete = UniformDiscreteRV(1, 5, 2)

    assert binomial.ftype == ["Discrete", "pdf"]
    assert binomial.support == [0, 4]

    assert bernoulli.ftype == ["Discrete", "pdf"]
    assert bernoulli.support == [0, 1]

    assert geometric.ftype == ["Discrete", "pdf"]
    assert geometric.support == [1, oo]

    assert poisson.ftype == ["Discrete", "pdf"]
    assert poisson.support == [0, oo]

    assert uniform_discrete.ftype == ["discrete", "pdf"]
    assert uniform_discrete.support == [1, 3, 5]
    assert uniform_discrete.func == [Rational(1, 3), Rational(1, 3), Rational(1, 3)]


def test_exponential_variate_supports_special_and_inverse_methods():
    rv = ExponentialRV(2)

    assert rv.variate(n=2, s=Rational(1, 2), method="special") == [ln(2) / 2, ln(2) / 2]

    inverse_samples = rv.variate(n=3, method="inverse")
    assert len(inverse_samples) == 3
    assert all(sample >= 0 for sample in inverse_samples)


def test_uniform_variate_supports_special_and_inverse_methods():
    rv = UniformRV(0, 4)

    assert rv.variate(n=2, s=Rational(1, 4), method="special") == [1, 1]

    inverse_samples = rv.variate(n=3, method="inverse")
    assert len(inverse_samples) == 3
    assert all(0 <= sample <= 4 for sample in inverse_samples)


def test_weibull_variate_supports_special_and_inverse_methods():
    rv = WeibullRV(2, 3)

    special_samples = rv.variate(n=2, s=Rational(1, 2), method="special")
    assert len(special_samples) == 2
    assert all(sample >= 0 for sample in special_samples)

    inverse_samples = rv.variate(n=3, method="inverse")
    assert len(inverse_samples) == 3
    assert all(sample >= 0 for sample in inverse_samples)


def test_variate_requires_specified_parameters_for_symbolic_defaults():
    with pytest.raises(RVError, match="Not all parameters specified"):
        ExponentialRV().variate()

    with pytest.raises(RVError, match="Not all parameters specified"):
        UniformRV().variate()

    with pytest.raises(RVError, match="Not all parameters specified"):
        WeibullRV().variate()


def test_variate_rejects_unknown_method():
    with pytest.raises(RVError, match="invalid method"):
        ExponentialRV(1).variate(method="bad")

    with pytest.raises(RVError, match="invalid method"):
        UniformRV(0, 1).variate(method="bad")


def test_parameter_validation_errors_for_common_distributions():
    with pytest.raises(RVError, match="theta must be positive"):
        ExponentialRV(-1)

    with pytest.raises(RVError, match="parameters must be in ascending order"):
        UniformRV(2, 1)

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


def test_bivariate_normal_parameter_validation_and_shape():
    rv = BivariateNormalRV(0, 1, 1, Rational(1, 2))
    assert rv.ftype == ["continuous", "pdf"]
    assert len(rv.func) == 1
    assert len(rv.constraints) == 2

    with pytest.raises(RVError, match="rho must be >=0 and <=1"):
        BivariateNormalRV(0, 1, 1, 2)

    with pytest.raises(RVError, match="sigma1 must be positive"):
        BivariateNormalRV(0, 0, 1, 0)

    with pytest.raises(RVError, match="sigma2 must be positive"):
        BivariateNormalRV(0, 1, 0, 0)


def test_weibull_cdf_shortcut_and_binomial_mean_are_consistent():
    rv = WeibullRV(2, 3)
    assert CDF(rv, Rational(1, 2)) == 1 - exp(-1)

    assert Mean(BinomialRV(4, Rational(1, 2))) == 2
