use pyo3::prelude::*;

mod order_stat;

#[pyfunction]
fn dummy_ping() -> &'static str {
    "applpy_rust_ok"
}

#[pymodule]
fn applpy_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dummy_ping, module)?)?;
    module.add_function(wrap_pyfunction!(order_stat::next_combination_py, module)?)?;
    Ok(())
}
