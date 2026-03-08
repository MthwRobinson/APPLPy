use pyo3::prelude::*;

#[pyfunction]
fn dummy_ping() -> &'static str {
    "applpy_rust_ok"
}

#[pymodule]
fn applpy_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dummy_ping, module)?)?;
    Ok(())
}
