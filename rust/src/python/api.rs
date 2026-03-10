#![allow(clippy::useless_conversion)]
#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::algorithms::order_stat;
use crate::algorithms::rv;
use crate::algorithms::rv::Number;

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
