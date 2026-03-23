import pytest
from sympy import Integer

from applpy import plot_dist as top_level_plot_dist
from applpy import appl_plot
from applpy.rv import RV, RVError


class PlotRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_mat_plot_continuous_and_discrete_paths(monkeypatch):
    plot_recorder = PlotRecorder()
    labels = {"x": None, "y": None, "title": None, "grid": None}

    monkeypatch.setattr(appl_plot, "plot", plot_recorder)
    monkeypatch.setattr(appl_plot, "arange", lambda start, stop, step: [start, stop])
    monkeypatch.setattr(appl_plot, "xlabel", lambda val: labels.__setitem__("x", val))
    monkeypatch.setattr(appl_plot, "ylabel", lambda val: labels.__setitem__("y", val))
    monkeypatch.setattr(appl_plot, "title", lambda val: labels.__setitem__("title", val))
    monkeypatch.setattr(appl_plot, "grid", lambda val: labels.__setitem__("grid", val))

    appl_plot.mat_plot(["x", "2", "0"], [0, 1, 2, 3], lab1="idf", lab2="Test", ftype="continuous")

    assert len(plot_recorder.calls) == 2
    assert labels == {"x": "s", "y": "idf", "title": "Test", "grid": True}

    plot_recorder.calls.clear()
    appl_plot.mat_plot([0.2, 0.8], [1, 2], lab1="F-1(s)", lab2="Disc", ftype="discrete")

    assert len(plot_recorder.calls) == 1
    assert labels == {"x": "s", "y": "F-1(s)", "title": "Disc", "grid": True}


def test_prob_plot_sets_identity_line_and_axis_labels(monkeypatch):
    plot_recorder = PlotRecorder()
    labels = {"x": None, "y": None, "title": None, "grid": None}

    monkeypatch.setattr(appl_plot, "plot", plot_recorder)
    monkeypatch.setattr(appl_plot, "arange", lambda start, stop, step: [start, stop])
    monkeypatch.setattr(appl_plot, "xlabel", lambda val: labels.__setitem__("x", val))
    monkeypatch.setattr(appl_plot, "ylabel", lambda val: labels.__setitem__("y", val))
    monkeypatch.setattr(appl_plot, "title", lambda val: labels.__setitem__("title", val))
    monkeypatch.setattr(appl_plot, "grid", lambda val: labels.__setitem__("grid", val))

    appl_plot.prob_plot([1, 2, 3], [1, 2, 3], "QQ Plot")
    assert len(plot_recorder.calls) == 2
    assert labels == {
        "x": "Model Quantiles",
        "y": "Sample Quantiles",
        "title": "QQ Plot",
        "grid": True,
    }

    appl_plot.prob_plot([0.1, 0.2], [0.3, 0.4], "PP Plot")
    assert labels == {
        "x": "Model CDF",
        "y": "Sample CDF",
        "title": "PP Plot",
        "grid": True,
    }


def _uniform_continuous_pdf():
    return RV(Integer(1), [0, 1], ["continuous", "pdf"])


def test_top_level_plotdist_import_points_to_appl_plot():
    assert top_level_plot_dist is appl_plot.plot_dist


def test_plot_dist_validation_and_plot_calls(monkeypatch):
    calls = {"plot": 0, "title": []}
    rv = _uniform_continuous_pdf()

    monkeypatch.setattr(
        appl_plot, "plot", lambda *args, **kwargs: calls.__setitem__("plot", calls["plot"] + 1)
    )
    monkeypatch.setattr(appl_plot, "title", lambda value: calls["title"].append(value))

    appl_plot.plot_dist(rv, suplist=[0, 1], display=False)
    assert calls["plot"] >= 1
    assert "Probability Density Function" in calls["title"]

    with pytest.raises(RVError, match="ascending order"):
        appl_plot.plot_dist(rv, suplist=[1, 0], display=False)
    with pytest.raises(RVError, match="within RV support"):
        appl_plot.plot_dist(rv, suplist=[-1, 0], display=False)


def test_plot_emp_cdf_delegates_to_plot_dist(monkeypatch):
    observed = {}

    def fake_plot_dist(random_variable, suplist=None, opt=None, color="r", display=True):
        observed["opt"] = opt
        observed["display"] = display

    monkeypatch.setattr(appl_plot, "plot_dist", fake_plot_dist)
    appl_plot.plot_emp_cdf([1, 2, 3, 4])

    assert observed["opt"] == "EMPCDF"
    assert observed["display"] is True


def test_pp_and_qq_plot_call_prob_plot(monkeypatch):
    rv = _uniform_continuous_pdf()
    seen = []

    monkeypatch.setattr(appl_plot, "ion", lambda: None)
    monkeypatch.setattr(appl_plot, "prob_plot", lambda sample, fitted, kind: seen.append(kind))

    appl_plot.pp_plot(rv, [0.1, 0.2, 0.3])
    appl_plot.qq_plot(rv, [0.1, 0.2, 0.3])

    assert seen == ["PP Plot", "QQ Plot"]

    with pytest.raises(RVError, match="given as a list"):
        appl_plot.pp_plot(rv, "invalid")
    with pytest.raises(RVError, match="given as a list"):
        appl_plot.qq_plot(rv, "invalid")
