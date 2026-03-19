import pytest
from sympy import Rational

from applpy import ExponentialRV, MaximumIID, mean, Minimum, MinimumIID


def _reliability_block_diagram_system_mean(scale_parameter_n):
    component_lifetime = ExponentialRV(Rational(1, 7))
    series_subsystem = MinimumIID(component_lifetime, 3)
    parallel_subsystem = MaximumIID(component_lifetime, scale_parameter_n)
    reliability_block_diagram_system = Minimum(series_subsystem, parallel_subsystem)
    return mean(reliability_block_diagram_system)


def test_reliability_block_diagram_minimum_iid_mean_from_notebook():
    component_lifetime = ExponentialRV(Rational(1, 7))
    reliability_block_diagram_system = MinimumIID(component_lifetime, 4)

    assert mean(reliability_block_diagram_system) == Rational(7, 4)


def test_reliability_block_diagram_mean_by_parallel_size_from_notebook():
    expected_means = {
        1: 1.75,
        2: 2.1,
        3: 2.216666666666667,
        4: 2.2666666666666666,
        5: 2.2916666666666665,
        6: 2.3055555555555554,
        7: 2.313888888888889,
        8: 2.319191919191919,
        9: 2.3227272727272728,
        10: 2.325174825174825,
        11: 2.326923076923077,
        12: 2.3282051282051284,
        13: 2.3291666666666666,
        14: 2.329901960784314,
    }

    actual_means = {
        n: float(_reliability_block_diagram_system_mean(scale_parameter_n=n)) for n in range(1, 15)
    }

    assert actual_means == pytest.approx(expected_means)
    assert list(actual_means.values()) == sorted(actual_means.values())


def test_reliability_block_diagram_n8_mean_from_notebook():
    assert _reliability_block_diagram_system_mean(scale_parameter_n=8) == Rational(1148, 495)
