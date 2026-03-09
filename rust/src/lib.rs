use pyo3::prelude::*;

pub mod algorithms;
pub mod python;

#[pyfunction]
fn dummy_ping() -> &'static str {
    "applpy_rust_ok"
}

#[pymodule]
fn applpy_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dummy_ping, module)?)?;
    module.add_function(wrap_pyfunction!(python::api::next_combination_py, module)?)?;
    Ok(())
}
