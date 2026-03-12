use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::algorithms::rv::{DomainType, FunctionalForm, Number};
use num_rational::Rational64;

impl<'py> FromPyObject<'py> for Number {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(int) = obj.extract::<i64>() {
            return Ok(Number::Integer(int));
        }

        if let Ok(float) = obj.extract::<f64>() {
            return Ok(Number::Float(float));
        }

        if obj.hasattr("p")? & obj.hasattr("q")? {
            let numerator: i64 = obj.getattr("p")?.extract()?;
            let denominator: i64 = obj.getattr("q")?.extract()?;
            let rational = Rational64::new(numerator, denominator);
            return Ok(Number::Rational(rational));
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected int, float, or sympy.Rational",
        ))
    }
}

impl IntoPy<PyObject> for Number {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let s = match self {
            Number::Float(f) => f.into_py(py),
            Number::Integer(i) => i.into_py(py),
            Number::Rational(r) => {
                let sympy = PyModule::import_bound(py, "sympy").expect("unable to import sympy");

                let rational = sympy
                    .getattr("Rational")
                    .expect("unable to import the Rational class from sympy");

                let numerator: i64 = *r.numer();
                let denominator: i64 = *r.denom();

                rational
                    .call1((numerator, denominator))
                    .expect("unable to initialize sympy Rational number")
                    .into_py(py)
            }
        };

        s
    }
}

impl<'py> FromPyObject<'py> for FunctionalForm {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s: String = obj.extract()?;
        match s.as_str() {
            "cdf" => Ok(FunctionalForm::Cdf),
            "chf" => Ok(FunctionalForm::Chf),
            "hf" => Ok(FunctionalForm::Hf),
            "idf" => Ok(FunctionalForm::Idf),
            "pdf" => Ok(FunctionalForm::Pdf),
            "sf" => Ok(FunctionalForm::Sf),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid FunctionalForm",
            )),
        }
    }
}

impl IntoPy<PyObject> for FunctionalForm {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let s = match self {
            FunctionalForm::Cdf => "cdf",
            FunctionalForm::Chf => "chf",
            FunctionalForm::Hf => "hf",
            FunctionalForm::Idf => "idf",
            FunctionalForm::Pdf => "pdf",
            FunctionalForm::Sf => "sf",
        };

        s.into_py(py)
    }
}

impl<'py> FromPyObject<'py> for DomainType {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s: String = obj.extract()?;
        match s.as_str() {
            "continuous" => Ok(DomainType::Continuous),
            "discrete" => Ok(DomainType::Discrete),
            "discrete_functional" => Ok(DomainType::DiscreteFunctional),
            // Backward compatibility with existing Python code.
            "Discrete" => Ok(DomainType::DiscreteFunctional),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid DomainType",
            )),
        }
    }
}

impl IntoPy<PyObject> for DomainType {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let s = match self {
            DomainType::Continuous => "continuous",
            DomainType::Discrete => "discrete",
            DomainType::DiscreteFunctional => "discrete_functional",
        };

        s.into_py(py)
    }
}
