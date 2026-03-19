import pytest
from sympy import Integer, Rational, oo

from applpy.conversion import cdf, chf, hf, idf, pdf, sf
from applpy.rv import RV, RVError, Convert, check_value, x


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def _discrete_pdf():
    return RV([Rational(1, 4), Rational(3, 4)], [1, 2], ["discrete", "pdf"])


def _functional_discrete_pdf():
    return RV([x], [1, 3], ["discrete_functional", "pdf"])


def test_check_value_support_bounds():
    support = [0, 1]

    assert check_value(x, support) is True
    assert check_value(Rational(1, 2), support) is True
    assert check_value(-1, support) is False
    assert check_value(2, support) is False


def test_cdf_for_simple_continuous_pdf_and_cache():
    rv = RV(1, [0, 1], ["continuous", "pdf"])

    cdf_rv = cdf(rv, cache=True)

    assert cdf_rv.func == [x]
    assert cdf_rv.support == [0, 1]
    assert cdf(rv, Rational(1, 4)) == Rational(1, 4)
    assert cdf(rv, -1) == 0
    assert cdf(rv, 2) == 1
    assert rv.cache["cdf"] is cdf_rv
    assert cdf(rv) is cdf_rv


def test_convert_discrete_functional_to_explicit_form():
    functional_rv = RV([x], [1, 3], ["discrete_functional", "pdf"])

    explicit_rv = Convert(functional_rv)

    assert explicit_rv.ftype == ["discrete", "pdf"]
    assert explicit_rv.support == [1, 2, 3]
    assert explicit_rv.func == [1, 2, 3]


def test_convert_validation_errors():
    with pytest.raises(RVError, match="must be discrete_functional"):
        Convert(RV(1, [0, 1], ["continuous", "pdf"]))

    with pytest.raises(RVError, match="infinite support"):
        Convert(RV([x], [0, oo], ["discrete_functional", "pdf"]))


def test_conversion_family_for_continuous_and_discrete_distributions():
    continuous = _uniform_continuous_pdf()
    discrete = _discrete_pdf()
    functional_discrete = _functional_discrete_pdf()

    assert cdf(continuous, Rational(1, 4)) == Rational(1, 4)
    assert sf(continuous, Rational(1, 4)) == Rational(3, 4)
    assert hf(continuous, Rational(1, 4)) == Rational(4, 3)
    assert chf(continuous).ftype == ["continuous", "chf"]
    assert idf(continuous, Rational(1, 2)) == Rational(1, 2)
    assert pdf(cdf(continuous), Rational(1, 4)) == 1

    assert cdf(discrete, 1) == Rational(1, 4)
    assert cdf(discrete, 0) == 0
    assert cdf(discrete, 3) == 1
    assert sf(discrete, 1) == Rational(3, 4)
    assert sf(discrete, 0) == 1
    assert sf(discrete, 3) == 0
    assert hf(discrete, 1) == Rational(1, 3)
    assert hf(discrete, 0) == 0
    assert hf(discrete, 3) == oo
    assert chf(discrete, 2) > 0
    assert chf(discrete, 0) == 0
    assert chf(discrete, 3) == oo
    assert idf(discrete, Rational(1, 2)) == 2
    assert idf(discrete, -1) is None
    assert idf(discrete, 2) is None
    assert pdf(discrete, 0) == 0
    assert pdf(discrete, 3) == 0

    converted = Convert(functional_discrete)
    assert cdf(functional_discrete) == cdf(converted)
    assert isinstance(pdf(functional_discrete), RV)
    assert pdf(converted).ftype == ["discrete", "pdf"]
    assert sf(functional_discrete) == sf(converted)


def test_conversion_out_of_support_errors():
    rv = _uniform_continuous_pdf()
    for fn in [pdf, hf, chf, idf]:
        with pytest.raises(RVError, match="within the support"):
            fn(rv, 100)
    assert sf(rv, 100) == 0


def test_conversion_roundtrips_across_precomputed_continuous_forms():
    continuous = _uniform_continuous_pdf()
    cdf_rv = cdf(continuous)
    sf_rv = sf(continuous)
    hf_rv = hf(continuous)
    chf_rv = chf(continuous)
    idf_rv = idf(continuous)

    for source in [cdf_rv, hf_rv, chf_rv]:
        assert cdf(source).ftype == ["continuous", "cdf"]
        assert sf(source).ftype == ["continuous", "sf"]
        assert hf(source).ftype == ["continuous", "hf"]
        assert chf(source).ftype == ["continuous", "chf"]
        assert idf(source).ftype == ["continuous", "idf"]
        assert pdf(source).ftype == ["continuous", "pdf"]

    for fn in [cdf, hf, idf, pdf]:
        with pytest.raises(RecursionError):
            fn(sf_rv)
    assert sf(sf_rv).ftype == ["continuous", "sf"]
    assert chf(sf_rv).ftype == ["continuous", "chf"]

    for fn in [cdf, sf, hf, chf, pdf]:
        with pytest.raises(RecursionError):
            fn(idf_rv)
    assert idf(idf_rv).ftype == ["continuous", "idf"]


def test_conversion_roundtrips_across_precomputed_discrete_forms():
    discrete = _discrete_pdf()
    cdf_rv = cdf(discrete)
    sf_rv = sf(discrete)
    hf_rv = hf(discrete)
    chf_rv = chf(discrete)
    idf_rv = idf(discrete)

    for source in [cdf_rv, sf_rv, hf_rv, chf_rv]:
        assert cdf(source).ftype == ["discrete", "cdf"]
        assert sf(source).ftype == ["discrete", "sf"]
        assert hf(source).ftype == ["discrete", "hf"]
        assert chf(source).ftype == ["discrete", "chf"]
        assert idf(source).ftype == ["discrete", "idf"]
        assert pdf(source).ftype == ["discrete", "pdf"]

    # hf/chf conversions should be idempotent when the source is already in that form.
    assert hf(hf_rv).func == hf_rv.func
    assert hf(hf_rv).support == hf_rv.support
    assert chf(chf_rv).func == chf_rv.func
    assert chf(chf_rv).support == chf_rv.support

    assert idf(idf_rv).ftype == ["discrete", "idf"]


def test_functional_discrete_conversion_branches():
    functional_pdf = RV([x], [1, 3], ["discrete_functional", "pdf"])
    functional_cdf = RV([x / 3], [1, 3], ["discrete_functional", "cdf"])
    functional_sf = RV([1 - x / 3], [1, 3], ["discrete_functional", "sf"])
    functional_hf = RV([x / (4 - x)], [1, 3], ["discrete_functional", "hf"])
    functional_chf = RV([-x], [1, 3], ["discrete_functional", "chf"])

    assert cdf(functional_cdf, 2) == Rational(2, 3)
    assert cdf(functional_pdf).ftype == ["discrete", "cdf"]
    assert sf(functional_pdf).ftype == ["discrete", "sf"]
    assert hf(functional_pdf).ftype == ["discrete", "hf"]
    assert chf(functional_pdf).ftype == ["discrete", "chf"]
    assert pdf(functional_cdf).ftype == ["discrete", "pdf"]
    assert chf(functional_chf, 2) == -2
    assert hf(functional_hf, 2) == 1
    assert sf(functional_sf).ftype == ["discrete", "sf"]


def test_discrete_idf_sf_hf_cross_conversions_and_known_idf_bug():
    discrete = _discrete_pdf()
    assert idf(hf(discrete)).ftype == ["discrete", "idf"]
    assert idf(chf(discrete)).ftype == ["discrete", "idf"]
    assert sf(hf(discrete)).ftype == ["discrete", "sf"]
    assert sf(chf(discrete)).ftype == ["discrete", "sf"]

    functional_idf = RV([2 * x], [0, 1], ["discrete_functional", "idf"])
    with pytest.raises(UnboundLocalError):
        idf(functional_idf, Rational(1, 2))
