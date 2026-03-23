import pytest
from sympy import Integer, Rational, exp, oo

from applpy import Convolution, ConvolutionIID, Product, ProductIID
from applpy.rv import RV, RVError, x


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def _discrete_pdf():
    return RV([Rational(1, 4), Rational(3, 4)], [1, 2], ["discrete", "pdf"])


def _discrete_pdf_bernoulli():
    return RV([Rational(1, 2), Rational(1, 2)], [0, 1], ["discrete", "pdf"])


def _functional_discrete_pdf():
    return RV([x], [1, 3], ["discrete_functional", "pdf"])


def test_functions_still_importable_from_base_package():
    assert callable(ConvolutionIID)
    assert callable(ProductIID)
    assert callable(Convolution)
    assert callable(Product)


def test_single_rv_convolution_and_product_iid_operations():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()

    assert isinstance(ConvolutionIID(continuous, 2), RV)
    assert isinstance(ConvolutionIID(discrete, 2), RV)
    assert isinstance(ProductIID(continuous, 2), RV)
    assert isinstance(ProductIID(discrete, 2), RV)


def test_convolution_iid_error_paths():
    with pytest.raises(RVError, match="must be an integer"):
        ConvolutionIID(_uniform_continuous_pdf(), Rational(3, 2))


def test_two_rv_convolution_and_product_for_continuous_and_discrete():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()
    bernoulli = _discrete_pdf_bernoulli()

    assert isinstance(Convolution(continuous, continuous), RV)
    assert isinstance(Convolution(discrete, bernoulli), RV)
    assert isinstance(Product(continuous, continuous), RV)
    assert isinstance(Product(discrete, bernoulli), RV)


def test_operations_on_symmetric_support_cover_additional_branches():
    symmetric = RV([Rational(1, 2), Rational(1, 2)], [-1, 0, 1], ["continuous", "pdf"])

    assert isinstance(Convolution(symmetric, symmetric), RV)
    assert isinstance(Product(symmetric, symmetric), RV)


def test_lifetime_continuous_special_case_paths():
    lifetime = RV([exp(-x)], [0, oo], ["continuous", "pdf"])

    assert isinstance(Convolution(lifetime, lifetime), RV)


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


def test_discrete_convolution_edge_paths():
    with pytest.raises(RVError, match="symbolic or infinite support"):
        Convolution(RV([x], [x, 3], ["discrete_functional", "pdf"]), _discrete_pdf())
