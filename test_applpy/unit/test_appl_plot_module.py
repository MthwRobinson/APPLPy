from applpy import appl_plot


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
