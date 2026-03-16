use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm};
use num_rational::Rational64;

impl<'py> FromPyObject<'py> for Number {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        // the p and q attributes are associated with the sympy Rtaional class
        if obj.hasattr("p")? & obj.hasattr("q")? {
            let numerator: i64 = obj.getattr("p")?.extract()?;
            let denominator: i64 = obj.getattr("q")?.extract()?;
            let rational = Rational64::new(numerator, denominator);
            return Ok(Number::Rational(rational));
        }

        if let Ok(int) = obj.extract::<i64>() {
            return Ok(Number::Integer(int));
        }

        if let Ok(float) = obj.extract::<f64>() {
            return Ok(Number::Float(float));
        }

        // SymPy's singleton infinities (oo, -oo, zoo, nan) do not always coerce
        // through `extract::<f64>()`, but they appear in precomputed discrete forms.
        let text: String = obj.str()?.extract()?;
        let special = match text.as_str() {
            "oo" | "zoo" => Some(f64::INFINITY),
            "-oo" => Some(f64::NEG_INFINITY),
            "nan" => Some(f64::NAN),
            _ => None,
        };
        if let Some(value) = special {
            return Ok(Number::Float(value));
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected int, float, sympy.Rational, or sympy infinity singleton",
        ))
    }
}

impl IntoPy<PyObject> for Number {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let s = match self {
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
            Number::Float(f) => {
                let sympy = PyModule::import_bound(py, "sympy").expect("unable to import sympy");
                let float = sympy
                    .getattr("Float")
                    .expect("unable to import the Float class from sympy");

                float
                    .call1((f,))
                    .expect("unable to initialize sympy Float number")
                    .into_py(py)
            }
            Number::Integer(i) => i.into_py(py),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::conversion::{swap_discrete_cdf_and_idf, swap_discrete_cdf_and_sf};
    use crate::algorithms::rv::RandomVariable;
    use num_rational::Rational64;

    #[test]
    fn swap_discrete_cdf_and_sf_complements_function_and_toggles_form() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 1)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let swapped = swap_discrete_cdf_and_sf(&rv).unwrap();

        assert_eq!(
            swapped.function,
            vec![
                Number::Rational(Rational64::new(9, 10)),
                Number::Rational(Rational64::new(3, 5)),
                Number::Rational(Rational64::new(0, 1)),
            ]
        );
        assert_eq!(swapped.support, rv.support);
        assert!(matches!(swapped.functional_form, FunctionalForm::Sf));
        assert!(matches!(swapped.domain_type, DomainType::Discrete));
    }

    #[test]
    fn swap_discrete_cdf_and_sf_returns_error_for_non_cdf_or_sf_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.1), Number::Float(0.9)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let error = swap_discrete_cdf_and_sf(&rv).unwrap_err();
        assert!(error.contains("requires an input with the cdf or sf functional form"));
    }

    #[test]
    fn swap_discrete_cdf_and_idf_converts_cdf_to_inverse_cdf() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(3, 4)),
                Number::Rational(Rational64::new(1, 1)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let swapped = swap_discrete_cdf_and_idf(&rv).unwrap();

        assert_eq!(swapped.function, rv.support);
        assert_eq!(
            swapped.support,
            vec![
                Number::Integer(0),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(3, 4)),
            ]
        );
        assert!(matches!(swapped.functional_form, FunctionalForm::Idf));
        assert!(matches!(swapped.domain_type, DomainType::Discrete));
    }

    #[test]
    fn swap_discrete_cdf_and_idf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = swap_discrete_cdf_and_idf(&rv);

        assert!(matches!(result, Err(msg) if msg == "cannot swap cdf and idf. function is empty"));
    }

    #[test]
    fn extracts_sympy_infinity_singletons_as_float() {
        Python::with_gil(|py| {
            let sympy = PyModule::import_bound(py, "sympy").unwrap();

            let oo = sympy.getattr("oo").unwrap();
            let neg_oo = oo.call_method0("__neg__").unwrap();
            let zoo = sympy.getattr("zoo").unwrap();
            let nan = sympy.getattr("nan").unwrap();

            assert_eq!(
                oo.extract::<Number>().unwrap(),
                Number::Float(f64::INFINITY)
            );
            assert_eq!(
                neg_oo.extract::<Number>().unwrap(),
                Number::Float(f64::NEG_INFINITY)
            );
            assert_eq!(
                zoo.extract::<Number>().unwrap(),
                Number::Float(f64::INFINITY)
            );

            match nan.extract::<Number>().unwrap() {
                Number::Float(value) => assert!(value.is_nan()),
                _ => panic!("expected float variant for sympy.nan"),
            }
        });
    }

    #[test]
    fn converts_float_variant_to_sympy_float() {
        Python::with_gil(|py| {
            let sympy = PyModule::import_bound(py, "sympy").unwrap();
            let value = Number::Float(0.75).into_py(py).bind(py).clone();
            let float_cls = sympy.getattr("Float").unwrap();

            assert!(value.is_instance(&float_cls).unwrap());
            assert_eq!(value.str().unwrap().to_string(), "0.750000000000000");
        });
    }
}
