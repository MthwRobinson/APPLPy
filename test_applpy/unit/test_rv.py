import pytest
from sympy import Integer, Rational, exp, oo

from applpy import rust_bindings
from applpy.rv import (
    RV,
    RVError,
    BootstrapRV,
    Convolution,
    ConvolutionIID,
    Histogram,
    LoadRV,
    Maximum,
    MaximumIID,
    MaximumRV,
    Minimum,
    MinimumIID,
    MinimumRV,
    Mixture,
    OrderStat,
    PPPlot,
    PlotClear,
    PlotDist,
    PlotEmpCDF,
    PlotLimits,
    Pow,
    Product,
    ProductDiscrete,
    ProductIID,
    QQPlot,
    RangeStat,
    Sqrt,
    Transform,
    Truncate,
    VerifyPDF,
    x,
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


def test_rv_init_wraps_scalar_function_and_sets_defaults():
    rv = RV(1, [0, 1], ["continuous", "pdf"])

    assert rv.func == [1]
    assert rv.support == [0, 1]
    assert rv.domain_type == "continuous"
    assert rv.functional_form == "pdf"
    assert rv.cache is None


def test_rv_init_requires_exactly_one_parameter_set():
    with pytest.raises(
        ValueError, match="Pass either ftype or both functional_form and domain_type"
    ):
        RV(1, [0, 1], ["continuous", "pdf"], functional_form="pdf", domain_type="continuous")

    with pytest.raises(
        ValueError, match="Pass either ftype or both functional_form and domain_type"
    ):
        RV(1, [0, 1], functional_form="pdf")


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


def test_bootstrap_rv_creates_discrete_pdf_with_frequencies():
    rv = BootstrapRV([3, 1, 3, 2])

    assert rv.ftype == ["discrete", "pdf"]
    assert rv.support == [1, 2, 3]
    assert rv.func == [Rational(1, 4), Rational(1, 4), Rational(1, 2)]


def test_next_combination_advances_lexicographically():
    assert rust_bindings.next_combination([1, 2, 4], 5) == [1, 2, 5]
    assert rust_bindings.next_combination([1, 4, 5], 5) == [2, 3, 4]


def test_next_permutation_advances_lexicographically_for_increasing_input():
    assert rust_bindings.next_permutation([1, 2, 3]) == [1, 3, 2]


def test_operator_overloads_for_scalars_and_rvs():
    rv = _uniform_continuous_pdf()
    discrete = _discrete_pdf_bernoulli()

    assert (+rv) is rv
    assert isinstance(-rv, RV)
    assert isinstance(rv + 1, RV)
    assert isinstance(1 + rv, RV)
    assert isinstance(rv - 1, RV)
    assert isinstance(1 - rv, RV)
    assert isinstance(rv * 2, RV)
    assert isinstance(2 * rv, RV)
    assert isinstance(rv / 2, RV)
    assert isinstance(2 / rv, RV)
    assert isinstance(rv + rv, RV)
    assert isinstance(rv - rv, RV)
    assert isinstance(discrete * discrete, RV)
    assert isinstance(discrete**2, RV)

    with pytest.raises(NotImplementedError):
        abs(rv)

    with pytest.raises(TypeError):
        discrete / discrete

    with pytest.raises(RVError, match="integer value"):
        discrete ** Rational(3, 2)


def test_assumptions_cache_and_simplify_helpers():
    rv = _uniform_continuous_pdf()
    for assumption in ["positive", "negative", "nonpositive", "nonnegative"]:
        current = _uniform_continuous_pdf()
        current.add_assumptions(assumption)
        current.drop_assumptions()
        assert isinstance(current, RV)

    with pytest.raises(RVError, match="only available options"):
        rv.add_assumptions("bad-option")

    simp = RV([Integer(1), Integer(1)], [-1, 0, 1], ["continuous", "pdf"])
    simp.simplify()
    assert simp.func == [1]
    assert simp.support == [-1, 1]


def test_latex_and_save_edge_cases(tmp_path):
    rv = _uniform_continuous_pdf()
    with pytest.raises(NameError):
        rv.latex()

    with pytest.raises(RVError, match="only designed to work"):
        _discrete_pdf().latex()

    with pytest.raises(RVError, match="specify a file name"):
        _uniform_continuous_pdf().save()

    out_file = tmp_path / "sample.rv"
    rv.save(str(out_file))
    assert out_file.exists()

    with pytest.raises(UnicodeDecodeError):
        LoadRV(str(out_file))


def test_single_rv_transformative_operations():
    continuous = _uniform_continuous_pdf()
    piecewise = _piecewise_continuous_pdf()
    discrete = _discrete_pdf()
    bernoulli = _discrete_pdf_bernoulli()

    assert isinstance(ConvolutionIID(continuous, 2), RV)
    assert isinstance(ConvolutionIID(discrete, 2), RV)
    assert isinstance(MaximumIID(continuous, 2), RV)
    assert isinstance(MaximumIID(discrete, 2), RV)
    assert isinstance(MinimumIID(continuous, 2), RV)
    assert isinstance(MinimumIID(discrete, 2), RV)
    assert isinstance(OrderStat(continuous, 3, 2), RV)
    assert isinstance(OrderStat(discrete, 3, 2, "w"), RV)
    assert isinstance(OrderStat(bernoulli, 2, 1, "wo"), RV)
    assert isinstance(Pow(continuous, 2), RV)
    assert isinstance(Pow(discrete, 2), RV)
    assert isinstance(ProductIID(continuous, 2), RV)
    assert isinstance(ProductIID(discrete, 2), RV)
    assert isinstance(RangeStat(continuous, 3), RV)
    assert isinstance(RangeStat(discrete, 2), RV)
    assert isinstance(Sqrt(continuous), RV)
    assert isinstance(Sqrt(discrete), RV)
    assert isinstance(Transform(discrete, [[x + 1, x + 2], [0, 1, 2]]), RV)
    assert isinstance(Transform(piecewise, [[x, x**2], [0, 1, 2]]), RV)
    assert isinstance(Truncate(continuous, [Rational(1, 4), Rational(3, 4)]), RV)
    assert isinstance(Truncate(discrete, [1, 1]), RV)
    assert VerifyPDF(continuous) is True
    assert VerifyPDF(discrete) is None


def test_single_rv_error_paths():
    with pytest.raises(RVError, match="must be an integer"):
        ConvolutionIID(_uniform_continuous_pdf(), Rational(3, 2))
    with pytest.raises(RVError, match="must be an integer"):
        Pow(_uniform_continuous_pdf(), Rational(3, 2))
    with pytest.raises(RVError, match="greater than the sample size"):
        OrderStat(_uniform_continuous_pdf(), 2, 3)
    with pytest.raises(RVError, match="Replace must be w or wo"):
        OrderStat(_uniform_continuous_pdf(), 3, 1, "invalid")
    with pytest.raises(RVError, match="without replacement not implemented"):
        OrderStat(_uniform_continuous_pdf(), 3, 1, "wo")


def test_two_rv_operations_for_continuous_and_discrete():
    continuous = _uniform_continuous_pdf()
    piecewise = _piecewise_continuous_pdf()
    discrete = _discrete_pdf()
    bernoulli = _discrete_pdf_bernoulli()

    assert isinstance(Convolution(continuous, continuous), RV)
    assert isinstance(Convolution(discrete, bernoulli), RV)
    assert isinstance(MaximumRV(continuous, continuous), RV)
    assert isinstance(MaximumRV(discrete, bernoulli), RV)
    assert isinstance(Maximum(continuous, continuous), RV)
    assert isinstance(Maximum(discrete, bernoulli), RV)
    assert isinstance(MinimumRV(continuous, continuous), RV)
    assert isinstance(MinimumRV(discrete, bernoulli), RV)
    assert isinstance(Minimum(continuous, continuous), RV)
    assert isinstance(Minimum(discrete, bernoulli), RV)
    assert isinstance(Mixture([Rational(1, 3), Rational(2, 3)], [continuous, piecewise]), RV)
    assert isinstance(Mixture([Rational(1, 2), Rational(1, 2)], [discrete, bernoulli]), RV)
    assert isinstance(Product(continuous, continuous), RV)
    assert isinstance(Product(discrete, bernoulli), RV)
    assert isinstance(ProductDiscrete(discrete, bernoulli), RV)


def test_two_rv_operations_error_paths():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()

    with pytest.raises(RVError, match="must both be discrete or continuous"):
        Maximum(continuous, discrete)
    with pytest.raises(RVError, match="must both be discrete or continuous"):
        Minimum(continuous, discrete)
    with pytest.raises(RVError, match="same length"):
        Mixture([Rational(1, 2)], [continuous, continuous])
    with pytest.raises(RVError, match="all continuous or discrete"):
        Mixture([Rational(1, 2), Rational(1, 2)], [continuous, discrete])
    with pytest.raises(RVError, match="both random variables must be discrete"):
        ProductDiscrete(continuous, discrete)


def test_plotting_and_misc_utility_paths():
    continuous = _uniform_continuous_pdf()
    functional_discrete = _functional_discrete_pdf()

    PlotClear()
    PlotLimits([0, 1], "x")
    PlotLimits([0, 2], "y")
    with pytest.raises(RVError, match='must be "x" or "y"'):
        PlotLimits([0, 1], "z")

    PlotDist(continuous, display=False)
    PlotDist(functional_discrete)
    PlotEmpCDF([1, 2, 3, 4])

    with pytest.raises(RVError, match="ascending order"):
        PlotDist(continuous, suplist=[1, 0], display=False)
    with pytest.raises(RVError, match="within RV support"):
        PlotDist(continuous, suplist=[-1, 0], display=False)

    with pytest.raises(RVError, match="entered as a list"):
        Histogram("invalid")
    with pytest.raises(AttributeError, match="normed"):
        Histogram([1, 2, 3, 4], Bins=2)

    with pytest.raises(RVError, match="given as a list"):
        PPPlot(continuous, "invalid")
    with pytest.raises(RVError, match="given as a list"):
        QQPlot(continuous, "invalid")
    with pytest.raises(AttributeError, match="prob_plot"):
        PPPlot(continuous, [0.1, 0.2, 0.3])
    with pytest.raises(AttributeError, match="prob_plot"):
        QQPlot(continuous, [0.1, 0.2, 0.3])


def test_variate_method_paths():
    continuous = _uniform_continuous_pdf()

    deterministic_samples = continuous.variate(n=2, s=Rational(2, 5))
    assert len(deterministic_samples) == 2
    assert all(abs(float(value) - 0.4) < 1e-9 for value in deterministic_samples)
    inverse_samples = continuous.variate(n=3, method="inverse")
    assert len(inverse_samples) == 3
    assert all(0 <= value <= 1 for value in inverse_samples)

    with pytest.raises(RVError, match="invalid method"):
        continuous.variate(method="bad-method")


def test_verifypdf_wrapper_function():
    assert VerifyPDF(_uniform_continuous_pdf()) is True


def test_save_reuses_filename_when_already_known(tmp_path):
    rv = _uniform_continuous_pdf()
    out_file = tmp_path / "roundtrip.rv"
    rv.save(str(out_file))
    rv.save()
    assert out_file.exists()


def test_operations_on_symmetric_support_cover_additional_branches():
    symmetric = RV([Rational(1, 2), Rational(1, 2)], [-1, 0, 1], ["continuous", "pdf"])
    positive = _uniform_continuous_pdf()
    discrete = _discrete_pdf()

    assert isinstance(Convolution(symmetric, symmetric), RV)
    assert isinstance(Product(symmetric, symmetric), RV)
    assert isinstance(MaximumRV(symmetric, symmetric), RV)
    assert isinstance(MinimumRV(symmetric, symmetric), RV)

    # Variable-arity paths recurse through previous results.
    assert isinstance(Maximum(positive, positive, positive), RV)
    assert isinstance(Minimum(discrete, discrete, discrete), RV)


def test_lifetime_continuous_special_case_paths():
    lifetime = RV([exp(-x)], [0, oo], ["continuous", "pdf"])

    assert isinstance(Convolution(lifetime, lifetime), RV)
    assert isinstance(MaximumRV(lifetime, lifetime), RV)
    assert isinstance(MinimumRV(lifetime, lifetime), RV)


def test_product_continuous_quadrant_case_coverage():
    interval_pairs = [
        ((1, 2), (3, 4)),
        ((1, 2), (2, 4)),
        ((2, 3), (1, 2)),
        ((-2, -1), (-3, -2)),
        ((-2, -1), (-2, -1)),
        ((-3, -2), (-2, -1)),
        ((-2, -1), (2, 3)),
        ((-3, -2), (1, 2)),
        ((-3, -2), (2, 3)),
        ((2, 3), (-2, -1)),
        ((1, 2), (-2, -1)),
        ((1, 2), (-3, -2)),
    ]
    for (a, b), (c, d) in interval_pairs:
        left = RV(Integer(1), [a, b], ["continuous", "pdf"])
        right = RV(Integer(1), [c, d], ["continuous", "pdf"])
        product = Product(left, right)
        assert isinstance(product, RV)
        assert product.ftype == ["continuous", "pdf"]


def test_product_discrete_symbolic_support_error_path():
    symbolic_support = RV([x], [x, 3], ["discrete_functional", "pdf"])
    regular = _functional_discrete_pdf()
    with pytest.raises(RVError, match="symbolic or infinite support"):
        Product(symbolic_support, regular)


def test_sqrt_and_transform_additional_error_paths():
    negative_support = RV(Integer(1), [-1, 0], ["continuous", "pdf"])
    with pytest.raises(RVError, match="negative value appears in the support"):
        Sqrt(negative_support)

    with pytest.raises(RVError, match="not in ascending order"):
        Transform(_uniform_continuous_pdf(), [[x], [1, 0]])


def test_plot_display_requires_multiple_plots():
    with pytest.raises(RVError, match="requires a list with multiple plots"):
        from applpy.rv import PlotDisplay

        PlotDisplay([object()])


def test_discrete_stat_and_convolution_edge_paths():
    with pytest.raises(RVError, match="Only one item sampled"):
        RangeStat(_discrete_pdf(), 1, "w")
    with pytest.raises(RVError, match="current not implemented without"):
        RangeStat(_discrete_pdf(), 2, "wo")

    singleton = RV([1], [5], ["discrete", "pdf"])
    assert OrderStat(singleton, 1, 1, "w").ftype == ["discrete", "pdf"]

    with pytest.raises(RVError, match="symbolic or infinite support"):
        Convolution(RV([x], [x, 3], ["discrete_functional", "pdf"]), _discrete_pdf())
