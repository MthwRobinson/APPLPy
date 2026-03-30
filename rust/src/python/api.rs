#![allow(clippy::useless_conversion)]
#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::algorithms::algebra;
use crate::algorithms::number::Number;
use crate::algorithms::order_stat;
use crate::algorithms::rv;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
use crate::algorithms::transform;

#[pyfunction(name = "discrete_order_stat", signature = (random_variable, n, r, replace="w"))]
pub fn discrete_order_stat_py(
    random_variable: &Bound<'_, PyAny>,
    n: u64,
    r: u64,
    replace: &str,
) -> PyResult<FastRV> {
    let random_variable: FastRV = random_variable.extract()?;

    let variant = match replace {
        "w" => order_stat::OrderStatVariant::WithReplacement,
        "wo" => order_stat::OrderStatVariant::WithoutReplacement,
        _ => {
            return Err(PyValueError::new_err(
                "Invalid OrderStatVariant: expected 'w' or 'wo'",
            ));
        }
    };

    let order_stat_rv = order_stat::discrete_order_stat(&random_variable.inner, n, r, variant)
        .map_err(PyValueError::new_err)?;

    let fast_rv = FastRV::new(
        order_stat_rv.function,
        order_stat_rv.support,
        order_stat_rv.functional_form,
        order_stat_rv.domain_type,
    );

    Ok(fast_rv)
}

#[pyfunction(name = "discrete_range_stat", signature = (random_variable, n, replace="w"))]
pub fn discrete_range_stat_py(
    random_variable: &Bound<'_, PyAny>,
    n: u64,
    replace: &str,
) -> PyResult<FastRV> {
    let random_variable: FastRV = random_variable.extract()?;

    let variant = match replace {
        "w" => order_stat::OrderStatVariant::WithReplacement,
        "wo" => order_stat::OrderStatVariant::WithoutReplacement,
        _ => {
            return Err(PyValueError::new_err(
                "Invalid OrderStatVariant: expected 'w' or 'wo'",
            ));
        }
    };

    let range_stat_rv = order_stat::discrete_range_stat(&random_variable.inner, n, variant)
        .map_err(PyValueError::new_err)?;

    let fast_rv = FastRV::new(
        range_stat_rv.function,
        range_stat_rv.support,
        range_stat_rv.functional_form,
        range_stat_rv.domain_type,
    );

    Ok(fast_rv)
}

#[pyfunction(name = "discrete_maximum", signature = (random_variable_1, random_variable_2))]
pub fn discrete_maximum_py(
    random_variable_1: &Bound<'_, PyAny>,
    random_variable_2: &Bound<'_, PyAny>,
) -> PyResult<FastRV> {
    let random_variable_1: FastRV = random_variable_1.extract()?;
    let random_variable_2: FastRV = random_variable_2.extract()?;

    let max_rv = order_stat::discrete_maximum(&random_variable_1.inner, &random_variable_2.inner)
        .map_err(PyValueError::new_err)?;

    Ok(FastRV::new(
        max_rv.function,
        max_rv.support,
        max_rv.functional_form,
        max_rv.domain_type,
    ))
}

#[pyfunction(name = "discrete_minimum", signature = (random_variable_1, random_variable_2))]
pub fn discrete_minimum_py(
    random_variable_1: &Bound<'_, PyAny>,
    random_variable_2: &Bound<'_, PyAny>,
) -> PyResult<FastRV> {
    let random_variable_1: FastRV = random_variable_1.extract()?;
    let random_variable_2: FastRV = random_variable_2.extract()?;

    let min_rv = order_stat::discrete_minimum(&random_variable_1.inner, &random_variable_2.inner)
        .map_err(PyValueError::new_err)?;

    Ok(FastRV::new(
        min_rv.function,
        min_rv.support,
        min_rv.functional_form,
        min_rv.domain_type,
    ))
}

#[pyfunction(name = "product_discrete", signature = (random_variable_1, random_variable_2))]
pub fn product_discrete_py(
    random_variable_1: &Bound<'_, PyAny>,
    random_variable_2: &Bound<'_, PyAny>,
) -> PyResult<FastRV> {
    let random_variable_1: FastRV = random_variable_1.extract()?;
    let random_variable_2: FastRV = random_variable_2.extract()?;

    let product_rv = algebra::product_discrete(&random_variable_1.inner, &random_variable_2.inner)
        .map_err(PyValueError::new_err)?;

    Ok(FastRV::new(
        product_rv.function,
        product_rv.support,
        product_rv.functional_form,
        product_rv.domain_type,
    ))
}

#[pyfunction(name = "convolution_discrete", signature = (random_variable_1, random_variable_2))]
pub fn convolution_discrete_py(
    random_variable_1: &Bound<'_, PyAny>,
    random_variable_2: &Bound<'_, PyAny>,
) -> PyResult<FastRV> {
    let random_variable_1: FastRV = random_variable_1.extract()?;
    let random_variable_2: FastRV = random_variable_2.extract()?;

    let sum_rv = algebra::convolution_discrete(&random_variable_1.inner, &random_variable_2.inner)
        .map_err(PyValueError::new_err)?;

    Ok(FastRV::new(
        sum_rv.function,
        sum_rv.support,
        sum_rv.functional_form,
        sum_rv.domain_type,
    ))
}

#[pyfunction(name = "next_combination", signature = (previous, n))]
pub fn next_combination_py(previous: Vec<usize>, n: usize) -> PyResult<Option<Vec<usize>>> {
    if previous.is_empty() {
        return Err(PyValueError::new_err("Previous must not be empty"));
    }
    Ok(order_stat::next_combination(&previous, n))
}

#[pyfunction(name = "next_permutation", signature = (previous))]
pub fn next_permutation_py(previous: Vec<usize>) -> PyResult<Option<Vec<usize>>> {
    if previous.is_empty() {
        return Err(PyValueError::new_err("Previous must not be empty"));
    }
    Ok(order_stat::next_permutation(&previous))
}

#[pyfunction(name = "verify_discrete_pdf", signature = (function, tolerance=1e-6))]
pub fn verify_discrete_pdf_py(function: Vec<Number>, tolerance: Option<f64>) -> PyResult<bool> {
    if let Ok(result) = rv::verify_pdf(&function, tolerance) {
        return Ok(result);
    }

    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "pdf validation failed",
    ))
}

#[pyfunction(name = "bootstrap_rv", signature = (variates))]
pub fn bootstrap_rv_py(variates: Vec<Number>) -> PyResult<FastRV> {
    let random_variable = rv::bootstrap_rv(&variates).map_err(PyValueError::new_err)?;
    let fast_rv = FastRV::new(
        random_variable.function,
        random_variable.support,
        random_variable.functional_form,
        random_variable.domain_type,
    );
    Ok(fast_rv)
}

#[pyfunction(name = "truncate_discrete", signature = (random_variable, min_support, max_support))]
pub fn truncate_discrete_py(
    random_variable: &Bound<'_, PyAny>,
    min_support: Number,
    max_support: Number,
) -> PyResult<FastRV> {
    let random_variable: FastRV = random_variable.extract()?;
    let truncated_rv =
        transform::truncate_discrete(&random_variable.inner, min_support, max_support)
            .map_err(PyValueError::new_err)?;
    Ok(FastRV::new(
        truncated_rv.function,
        truncated_rv.support,
        truncated_rv.functional_form,
        truncated_rv.domain_type,
    ))
}

#[pyfunction(name = "mixture_discrete", signature = (random_variables, mix_weights))]
pub fn mixture_discrete_py(
    random_variables: Vec<Bound<'_, PyAny>>,
    mix_weights: Vec<Number>,
) -> PyResult<FastRV> {
    let extracted: PyResult<Vec<FastRV>> = random_variables
        .into_iter()
        .map(|rv| rv.extract::<FastRV>())
        .collect();
    let extracted: Vec<FastRV> = extracted?;

    let extracted_rvs: Vec<&RandomVariable> =
        extracted.iter().map(|fast_rv| &fast_rv.inner).collect();

    let mixed_rv =
        transform::mixture_discrete(&extracted_rvs, &mix_weights).map_err(PyValueError::new_err)?;

    Ok(FastRV::new(
        mixed_rv.function,
        mixed_rv.support,
        mixed_rv.functional_form,
        mixed_rv.domain_type,
    ))
}

#[pyclass]
pub struct FastRV {
    inner: RandomVariable,
}

fn format_number_list(values: &[Number]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[pymethods]
impl FastRV {
    #[new]
    pub fn new(
        function: Vec<Number>,
        support: Vec<Number>,
        functional_form: FunctionalForm,
        domain_type: DomainType,
    ) -> Self {
        Self {
            inner: rv::RandomVariable {
                function,
                support,
                functional_form,
                domain_type,
            },
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "FastRV(function=[{}], support=[{}], functional_form='{}', domain_type='{}')",
            format_number_list(&self.inner.function),
            format_number_list(&self.inner.support),
            self.inner.functional_form,
            self.inner.domain_type
        )
    }

    pub fn __add__(&self, rhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let rhs_rv = rhs.inner.clone();

        let sum_rv = self_rv + rhs_rv;

        match sum_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    pub fn __radd__(&self, lhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let lhs_rv = lhs.inner.clone();

        let sum_rv = lhs_rv + self_rv;

        match sum_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    pub fn __sub__(&self, rhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let rhs_rv = rhs.inner.clone();

        let difference_rv = self_rv - rhs_rv;

        match difference_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    pub fn __rsub__(&self, lhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let lhs_rv = lhs.inner.clone();

        let difference_rv = lhs_rv - self_rv;

        match difference_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    pub fn __mul__(&self, rhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let rhs_rv = rhs.inner.clone();

        let product_rv = self_rv * rhs_rv;

        match product_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    pub fn __rmul__(&self, lhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let lhs_rv = lhs.inner.clone();

        let product_rv = lhs_rv * self_rv;

        match product_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    pub fn __truediv__(&self, rhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let rhs_rv = rhs.inner.clone();

        let quotient_rv = self_rv / rhs_rv;

        match quotient_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    pub fn __rtruediv__(&self, lhs: FastRV) -> PyResult<FastRV> {
        let self_rv = self.inner.clone();
        let lhs_rv = lhs.inner.clone();

        let quotient_rv = lhs_rv / self_rv;

        match quotient_rv {
            Ok(rv) => {
                let fast_rv =
                    FastRV::new(rv.function, rv.support, rv.functional_form, rv.domain_type);
                Ok(fast_rv)
            }
            Err(s) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(s)),
        }
    }

    #[getter]
    pub fn function(&self) -> Vec<Number> {
        self.inner.function.clone()
    }

    #[getter]
    pub fn func(&self) -> Vec<Number> {
        self.inner.function.clone()
    }

    #[getter]
    pub fn support(&self) -> Vec<Number> {
        self.inner.support.clone()
    }

    #[getter]
    pub fn functional_form(&self) -> FunctionalForm {
        self.inner.functional_form.clone()
    }

    #[getter]
    pub fn domain_type(&self) -> DomainType {
        self.inner.domain_type.clone()
    }

    #[pyo3(signature = (tolerance=None))]
    pub fn verify_pdf(&self, tolerance: Option<f64>) -> bool {
        self.inner
            .verify_pdf(tolerance)
            .expect("veriy_pdf method failed")
    }

    pub fn evaluate(&self, value: Number) -> Option<Number> {
        self.inner.evaluate(value)
    }

    pub fn to_pdf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_pdf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "conversion to pdf failed",
        ))
    }

    pub fn to_cdf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_cdf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "conversion to cdf failed",
        ))
    }

    pub fn to_sf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_sf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "conversion to sf failed",
        ))
    }

    pub fn to_idf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_idf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "conversion to idf failed",
        ))
    }

    pub fn to_chf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_chf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "conversion to chf failed",
        ))
    }

    pub fn to_hf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_hf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "conversion to hf failed",
        ))
    }

    pub fn mean(&self) -> PyResult<Number> {
        if let Ok(mean) = self.inner.mean() {
            return Ok(mean);
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "failed to compute the mean of the random variable",
        ))
    }

    pub fn variance(&self) -> PyResult<Number> {
        if let Ok(variance) = self.inner.variance() {
            return Ok(variance);
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "failed to compute the variance of the random variable",
        ))
    }

    pub fn skewness(&self) -> PyResult<Number> {
        if let Ok(skewness) = self.inner.skewness() {
            return Ok(skewness);
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "failed to compute the skewness of the random variable",
        ))
    }

    pub fn kurtosis(&self) -> PyResult<Number> {
        if let Ok(kurtosis) = self.inner.kurtosis() {
            return Ok(kurtosis);
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "failed to compute the kurtosis of the random variable",
        ))
    }

    pub fn coefficient_of_variation(&self) -> PyResult<Number> {
        if let Ok(coefficient_of_variation) = self.inner.coefficient_of_variation() {
            return Ok(coefficient_of_variation);
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "failed to compute the coefficient_of_variation of the random variable",
        ))
    }

    pub fn entropy(&self) -> PyResult<Number> {
        if let Ok(entropy) = self.inner.entropy() {
            return Ok(entropy);
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "failed to compute the entropy of the random variable",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    #[test]
    fn repr_is_python_style_and_stable() {
        let rv = FastRV {
            inner: rv::RandomVariable {
                function: vec![Number::Rational(Rational64::new(1, 2)), Number::Integer(1)],
                support: vec![Number::Integer(0), Number::Integer(1)],
                functional_form: FunctionalForm::Pdf,
                domain_type: DomainType::Discrete,
            },
        };

        assert_eq!(
            rv.__repr__(),
            "FastRV(function=[1/2, 1], support=[0, 1], functional_form='pdf', domain_type='discrete')"
        );
    }
}
