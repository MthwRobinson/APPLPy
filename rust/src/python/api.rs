#![allow(clippy::useless_conversion)]
#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::algorithms::number::Number;
use crate::algorithms::order_stat;
use crate::algorithms::rv;
use crate::algorithms::rv::{DomainType, FunctionalForm};

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
    inner: rv::RandomVariable,
}

#[pymethods]
impl FastRV {
    #[new]
    fn new(
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

    #[getter]
    pub fn function(&self) -> Vec<Number> {
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
}
