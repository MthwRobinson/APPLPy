#![allow(clippy::useless_conversion)]
#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::algorithms::order_stat;

#[pyfunction(name = "next_combination", signature = (previous, n))]
pub fn next_combination_py(previous: Vec<usize>, n: usize) -> PyResult<Vec<usize>> {
    if previous.is_empty() {
        return Err(PyValueError::new_err("Previous must not be empty"));
    }
    Ok(order_stat::next_combination(&previous, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_python_binding() {
        Python::with_gil(|_| {
            let result = next_combination_py(vec![1, 2, 3], 5).unwrap();
            assert_eq!(result, vec![1, 2, 4]);
        });
    }
}
