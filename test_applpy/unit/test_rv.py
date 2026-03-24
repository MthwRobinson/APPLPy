import pytest
from sympy import Integer, Rational

from applpy.algebra import _product_discrete as ProductDiscrete
from applpy.appl_plot import plot_dist, plot_emp_cdf, pp_plot, qq_plot
from applpy import rust_bindings
from applpy.rv import (
    RV,
    RVError,
    bootstrap_rv,
    verify_pdf,
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
    rv = bootstrap_rv([3, 1, 3, 2])

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


def test_single_rv_transformative_operations():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()
    assert verify_pdf(continuous) is True
    assert verify_pdf(discrete) is None


def test_two_rv_operations_for_continuous_and_discrete():
    discrete = _discrete_pdf()
    bernoulli = _discrete_pdf_bernoulli()

    assert isinstance(ProductDiscrete(discrete, bernoulli), RV)


def test_two_rv_operations_error_paths():
    discrete = _discrete_pdf()

    with pytest.raises(RVError, match="both random variables must be discrete"):
        ProductDiscrete(_uniform_continuous_pdf(), discrete)


def test_plotting_and_misc_utility_paths():
    continuous = _uniform_continuous_pdf()
    functional_discrete = _functional_discrete_pdf()

    plot_dist(continuous, display=False)
    plot_dist(functional_discrete)
    plot_emp_cdf([1, 2, 3, 4])

    with pytest.raises(RVError, match="ascending order"):
        plot_dist(continuous, suplist=[1, 0], display=False)
    with pytest.raises(RVError, match="within RV support"):
        plot_dist(continuous, suplist=[-1, 0], display=False)

    with pytest.raises(RVError, match="given as a list"):
        pp_plot(continuous, "invalid")
    with pytest.raises(RVError, match="given as a list"):
        qq_plot(continuous, "invalid")

    pp_plot(continuous, [0.1, 0.2, 0.3])
    qq_plot(continuous, [0.1, 0.2, 0.3])


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
    assert verify_pdf(_uniform_continuous_pdf()) is True
