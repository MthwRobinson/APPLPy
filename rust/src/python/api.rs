#![allow(clippy::useless_conversion)]
#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::algorithms::number::Number;
use crate::algorithms::order_stat;
use crate::algorithms::rv;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

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
            "converstion to pdf failed",
        ))
    }

    pub fn to_cdf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_cdf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "converstion to cdf failed",
        ))
    }

    pub fn to_sf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_sf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "converstion to sf failed",
        ))
    }

    pub fn to_idf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_idf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "converstion to idf failed",
        ))
    }

    pub fn to_chf(&self) -> PyResult<FastRV> {
        if let Ok(random_variable) = self.inner.to_chf() {
            return Ok(FastRV {
                inner: random_variable,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "converstion to chf failed",
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
