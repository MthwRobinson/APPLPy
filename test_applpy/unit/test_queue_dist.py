import numpy as np
import pytest
from sympy import Symbol, oo

from applpy.distributions.continuous import ExponentialRV, UniformRV
from applpy.queue_dist import (
    BuildDist,
    Cprime,
    MMSQprob,
    Queue,
    QueueMenu,
    _Q,
    cases,
    ini,
    kCprime,
    kcases,
    kpath,
    kprobvec,
    okay,
    path,
    probvec,
    swapa,
    swapb,
)
from applpy.rv import RVError


def test_queue_menu_prints_expected_labels(capsys):
    QueueMenu()
    out = capsys.readouterr().out
    assert "Queue Procedures" in out
    assert "Queue(X,Y,n,k,s)" in out


def test_build_dist_validates_exponential_inputs():
    with pytest.raises(RVError, match="queue must beexponential"):
        BuildDist(UniformRV(0, 1), ExponentialRV(1), 1, 0, 1)

    with pytest.raises(RVError, match="queue must beexponential"):
        BuildDist(ExponentialRV(1), UniformRV(0, 1), 1, 0, 1)


def test_build_dist_branching_for_servers_and_queue():
    x_rv = ExponentialRV(2)
    y_rv = ExponentialRV(3)

    single_server = BuildDist(x_rv, y_rv, n=2, k=0, s=1)
    assert len(single_server) == 2
    assert all(rv.__class__.__name__ == "ErlangRV" for rv in single_server)

    multi_server_no_wait = BuildDist(x_rv, y_rv, n=2, k=0, s=3)
    assert len(multi_server_no_wait) == 2
    assert all(rv.__class__.__name__ == "ExponentialRV" for rv in multi_server_no_wait)

    multi_server_with_wait = BuildDist(x_rv, y_rv, n=3, k=1, s=2)
    assert len(multi_server_with_wait) == 4
    assert any(rv.__class__.__name__ == "RV" for rv in multi_server_with_wait)


def test_mmsqprob_and_q_return_expected_symbolic_forms():
    rho = Symbol("rho")
    probs = MMSQprob(1, 1, 1)
    assert probs == [1 / (rho + 1), rho / (rho + 1)]
    assert _Q(1, 1, 1, 1) == 1 / (rho + 1)
    assert _Q(1, 2, 1, 1) == rho / (rho + 1)


def test_queue_constructs_mixture_distribution():
    rv = Queue(ExponentialRV(2), ExponentialRV(3), n=1, k=0, s=1)
    assert rv.ftype == ["continuous", "pdf"]
    assert rv.support == [0, oo]


def test_case_helpers_for_small_n():
    assert np.array_equal(ini(2), np.array([1.0, -1.0, 1.0, -1.0]))
    assert cases(1).shape == (1, 2)
    assert probvec(1, meanX=1, meanY=1).tolist() == [1.0]
    assert okay(2, [-1, 1, -1, 1]) is False
    assert okay(2, [1, -1, 1, -1]) is True

    first_case = cases(1)[0]
    case_path = path(1, first_case)
    assert case_path.shape == (2, 2)
    assert case_path[1, 0] == 1

    assert np.array_equal(swapa(2, ini(2).copy()), np.array([1.0, -1.0, -1.0, -1.0]))
    assert np.array_equal(swapb(2, ini(2).copy()), np.array([1.0, 1.0, 1.0, -1.0]))


def test_kpath_and_known_error_paths():
    row = np.array([1, -1, 1])
    with pytest.raises(IndexError):
        kpath(1, 1, row)

    with pytest.raises(IndexError):
        kcases(2, 1)

    with pytest.raises(IndexError):
        Cprime(1, cases(1))

    with pytest.raises(IndexError):
        kCprime(1, 1, np.array([row]))

    with pytest.raises(IndexError):
        kprobvec(2, 1, meanX=1, meanY=1)
