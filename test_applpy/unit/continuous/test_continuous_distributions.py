import pytest
from sympy import Rational, exp, ln, oo

from applpy.distributions.continuous import (
    ArcSinRV,
    ArcTanRV,
    BivariateNormalRV,
    BetaRV,
    CauchyRV,
    ChiRV,
    ChiSquareRV,
    ErlangRV,
    ErrorIIRV,
    ErrorRV,
    ExponentialPowerRV,
    ExponentialRV,
    ExtremeValueRV,
    FRV,
    GammaRV,
    GeneralizedParetoRV,
    GompertzRV,
    IDBRV,
    InverseGammaRV,
    InverseGaussianRV,
    KSRV,
    LaPlaceRV,
    LogGammaRV,
    LogLogisticRV,
    LogisticRV,
    LogNormalRV,
    LomaxRV,
    MakehamRV,
    MuthRV,
    NormalRV,
    ParetoRV,
    RayleighRV,
    TRV,
    TriangularRV,
    UniformRV,
    WeibullRV,
    param_check,
)
from applpy.rv import CDF, RVError, x


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


def test_weibull_cdf_shortcut_is_consistent():
    rv = WeibullRV(2, 3)
    assert CDF(rv, Rational(1, 2)) == 1 - exp(-1)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: ArcTanRV(2, 0),
        lambda: BetaRV(2, 3),
        lambda: CauchyRV(0, 2),
        lambda: ChiRV(3),
        lambda: ChiSquareRV(3),
        lambda: ErlangRV(2, 3),
        lambda: ErrorRV(2, 2, 0),
        lambda: ErrorIIRV(1, 2, 3),
        lambda: ExponentialPowerRV(2, 3),
        lambda: ExtremeValueRV(2, 3),
        lambda: FRV(2, 3),
        lambda: GammaRV(2, 3),
        lambda: GeneralizedParetoRV(2, 3, 4),
        lambda: GompertzRV(2, 3),
        lambda: IDBRV(1, 2, 3),
        lambda: InverseGaussianRV(2, 3),
        lambda: InverseGammaRV(2, 3),
        lambda: KSRV(1),
        lambda: LaPlaceRV(2, 1),
        lambda: LogGammaRV(2, 3),
        lambda: LogisticRV(2, 3),
        lambda: LogLogisticRV(2, 3),
        lambda: LogNormalRV(0, 2),
        lambda: LomaxRV(2, 3),
        lambda: MakehamRV(2, 3, 4),
        lambda: MuthRV(2),
        lambda: NormalRV(0, 1),
        lambda: ParetoRV(2, 3),
        lambda: RayleighRV(2),
        lambda: TriangularRV(0, 1, 2),
        lambda: TRV(3),
    ],
)
def test_additional_distribution_constructors_return_rv_shapes(builder):
    rv = builder()
    assert rv.ftype[1] in {"pdf", "cdf"}
    assert len(rv.support) >= 2


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda: ArcTanRV(0, 0), "Alpha must be positive"),
        (lambda: ChiRV(0), "N must be a positive integer"),
        (lambda: ErlangRV(-1, 2), "theta must be positive"),
        (lambda: ErrorRV(0, 2, 0), "mu must be positive"),
        (lambda: ExponentialPowerRV(-1, 3), "both parameters must be positive"),
        (lambda: FRV(-1, 3), "both parameters must be positive"),
        (lambda: GammaRV(-1, 3), "both parameters must be positive"),
        (lambda: TriangularRV(2, 1, 3), "ascending order"),
        (lambda: NormalRV(0, 0), "sigma must be positive"),
    ],
)
def test_additional_distribution_parameter_validation_errors(builder, message):
    with pytest.raises(RVError, match=message):
        builder()


def test_additional_variate_paths_for_supported_distributions():
    assert len(CauchyRV(0, 1).variate(n=2, s=Rational(1, 2), method="special")) == 2
    assert len(ExponentialPowerRV(2, 3).variate(n=2, s=Rational(1, 5), method="special")) == 2
    assert len(ExtremeValueRV(2, 3).variate(n=2, s=Rational(1, 5), method="special")) == 2
    assert len(GompertzRV(2, 3).variate(n=2, s=Rational(1, 5), method="special")) == 2
    assert len(LogisticRV(2, 3).variate(n=2, s=Rational(1, 5), method="special")) == 2
    assert len(LogLogisticRV(2, 3).variate(n=2, s=Rational(1, 5), method="special")) == 2
    assert len(NormalRV(0, 1).variate(n=2, s=Rational(1, 5), method="special")) == 2


def test_lomax_variate_currently_exposes_parameter_attribute_bug():
    with pytest.raises(AttributeError, match="parameter"):
        LomaxRV(2, 3).variate(n=1, s=Rational(1, 2), method="special")
