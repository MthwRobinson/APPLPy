use pyo3::prelude::*;

pub mod algorithms;
pub mod python;

pub use algorithms::number::Number;
pub use python::api::FastRV;

#[pyfunction]
fn dummy_ping() -> &'static str {
    "applpy_rust_ok"
}

#[pymodule]
fn applpy_rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    // order stat functions
    module.add_function(wrap_pyfunction!(python::api::next_combination_py, module)?)?;
    module.add_function(wrap_pyfunction!(python::api::next_permutation_py, module)?)?;

    // random variable class and related functions
    module.add_class::<FastRV>()?;
    module.add_function(wrap_pyfunction!(
        python::api::verify_discrete_pdf_py,
        module
    )?)?;

    // dummy function to validate imports
    module.add_function(wrap_pyfunction!(dummy_ping, module)?)?;

    Ok(())
}
