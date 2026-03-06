from applpy.dist_type import BivariateNormalRV, ExampleRV, ExponentialRV, PoissonRV


def test_example_bivariate_distribution_shape():
    rv = ExampleRV()
    assert rv.ftype == ["continuous", "pdf"]
    assert len(rv.func) == 1
    assert len(rv.constraints) == 1


def test_distribution_import_paths_are_supported():
    from applpy import ExponentialRV as top_level_exponential
    from applpy.distributions.continuous import BivariateNormalRV as continuous_bivariate_normal
    from applpy.distributions.continuous import ExponentialRV as continuous_exponential
    from applpy.distributions.discrete import PoissonRV as discrete_poisson

    assert top_level_exponential is ExponentialRV
    assert continuous_exponential is ExponentialRV
    assert discrete_poisson is PoissonRV
    assert continuous_bivariate_normal is BivariateNormalRV
