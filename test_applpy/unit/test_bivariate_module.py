import pytest

from applpy.bivariate import BivariateRV, _intersection, _union
from applpy.rv import RVError, x, y


def test_bivariate_init_len_and_cache_helpers():
    rv = BivariateRV(2, [[x, y, 1 - x - y]])

    assert len(rv) == 1
    assert rv.func == [2]
    assert rv.ftype == ["continuous", "pdf"]

    rv.add_to_cache("mean", 1)
    assert rv.cache == {"mean": 1}


def test_bivariate_init_validation_errors():
    with pytest.raises(RVError, match="Constraints must be entered as a list"):
        BivariateRV(1, (x, y))

    with pytest.raises(RVError, match="must be discrete or continuous"):
        BivariateRV(1, [[x, y, 1 - x - y]], ["bad", "pdf"])


def test_display_paths_for_continuous_and_discrete(capsys):
    rv = BivariateRV(2, [[x, y, 1 - x - y]])
    rv.display()
    out = capsys.readouterr().out
    assert "continuous pdf" in out
    assert "enclosed in the region" in out

    discrete_rv = BivariateRV(1, [[x, y, 1 - x - y]], ["discrete", "pdf"])
    with pytest.raises(AttributeError, match="support"):
        discrete_rv.display()


def test_verify_pdf_rejects_non_continuous_pdf():
    rv = BivariateRV(2, [[x, y, 1 - x - y]], ["continuous", "cdf"])
    with pytest.raises(RVError, match="only supports continuous pdfs"):
        rv.verifyPDF()


def test_verify_pdf_current_symbol_lookup_error_path():
    rv = BivariateRV(2, [[x, y, 1 - x - y]])
    with pytest.raises(KeyError, match="x"):
        rv.verifyPDF()


def test_set_operation_helpers_accept_scalars_and_lists():
    assert _intersection([1, 2], [2, 3]) == [2]
    assert _intersection(2, [1, 2]) == [2]

    union = _union([1], 2)
    assert set(union) == {1, 2}
