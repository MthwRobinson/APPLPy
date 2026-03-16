#![allow(dead_code)]

use std::fmt;

use num_traits::cast::ToPrimitive;

use crate::algorithms::conversion;
use crate::algorithms::number::Number;

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionalForm {
    Cdf,
    Chf,
    Hf,
    Idf,
    Pdf,
    Sf,
}

impl fmt::Display for FunctionalForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            FunctionalForm::Cdf => "cdf",
            FunctionalForm::Chf => "chf",
            FunctionalForm::Hf => "hf",
            FunctionalForm::Idf => "idf",
            FunctionalForm::Pdf => "pdf",
            FunctionalForm::Sf => "sf",
        };

        write!(f, "{}", value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DomainType {
    Continuous,
    Discrete,
    DiscreteFunctional,
}

impl fmt::Display for DomainType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            DomainType::Continuous => "continuous",
            DomainType::Discrete => "discrete",
            DomainType::DiscreteFunctional => "discrete_functional",
        };

        write!(f, "{}", value)
    }
}

#[derive(Debug, Clone)]
pub struct RandomVariable {
    pub function: Vec<Number>,
    pub support: Vec<Number>,
    pub functional_form: FunctionalForm,
    pub domain_type: DomainType,
}

impl RandomVariable {
    pub fn verify_pdf(&self, tolerance: Option<f64>) -> Result<bool, String> {
        if self.functional_form != FunctionalForm::Pdf {
            return Err("verify_pdf only works for PDFs".to_string());
        }

        verify_pdf(&self.function, tolerance)
    }

    pub fn to_pdf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => conversion::discrete_cdf_to_pdf(self),
            FunctionalForm::Chf => {
                let sf_random_variable = self.to_sf()?;
                sf_random_variable.to_pdf()
            }
            FunctionalForm::Hf => {
                let chf_random_variable = self.to_chf()?;
                chf_random_variable.to_pdf()
            }
            FunctionalForm::Pdf => Ok(self.clone()),
            FunctionalForm::Sf | FunctionalForm::Idf => {
                let cdf_random_variable = self.to_cdf()?;
                conversion::discrete_cdf_to_pdf(&cdf_random_variable)
            }
        }
    }

    pub fn to_cdf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => Ok(self.clone()),
            FunctionalForm::Chf => {
                let sf_random_variable = self.to_sf()?;
                sf_random_variable.to_cdf()
            }
            FunctionalForm::Hf => {
                let hf_random_variable = self.to_chf()?;
                hf_random_variable.to_cdf()
            }
            FunctionalForm::Idf => conversion::swap_discrete_cdf_and_idf(self),
            FunctionalForm::Pdf => conversion::discrete_pdf_to_cdf(self),
            FunctionalForm::Sf => conversion::swap_discrete_cdf_and_sf(self),
        }
    }

    pub fn to_sf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => conversion::swap_discrete_cdf_and_sf(self),
            FunctionalForm::Chf => conversion::discrete_chf_to_sf(self),
            FunctionalForm::Hf => {
                let chf_random_variable = conversion::discrete_hf_to_chf(self)?;
                chf_random_variable.to_sf()
            }
            FunctionalForm::Pdf | FunctionalForm::Idf => {
                let cdf_random_variable = self.to_cdf()?;
                conversion::swap_discrete_cdf_and_sf(&cdf_random_variable)
            }
            FunctionalForm::Sf => Ok(self.clone()),
        }
    }

    pub fn to_chf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Hf => conversion::discrete_hf_to_chf(self),
            _ => {
                let hf_function = conversion::discrete_rv_to_hf(self)?;
                hf_function.to_chf()
            }
        }
    }

    pub fn to_hf(&self) -> Result<RandomVariable, String> {
        conversion::discrete_rv_to_hf(self)
    }

    pub fn to_idf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => conversion::swap_discrete_cdf_and_idf(self),
            FunctionalForm::Chf | FunctionalForm::Hf => {
                let cdf_random_variable = self.to_cdf()?;
                cdf_random_variable.to_chf()
            }
            FunctionalForm::Idf => Ok(self.clone()),
            FunctionalForm::Pdf | FunctionalForm::Sf => {
                let cdf_random_variable = self.to_cdf()?;
                conversion::swap_discrete_cdf_and_idf(&cdf_random_variable)
            }
        }
    }
}

/// Verifies that the area under the PDF of random variable sums to 1
///
/// # Arguments
/// * `function` - the probability mass functon of the RV
/// * `support` - the support of the RV
/// * `tolerance` - sets the tolerance for how far the result
///   can deviate from 1
///
/// # Returns
/// * `valid` - a boolean indicatin if the PDF is valid
pub fn verify_pdf(function: &[Number], tolerance: Option<f64>) -> Result<bool, String> {
    let default_tolerance: f64 = 0.000001;
    let tolerance = tolerance.unwrap_or(default_tolerance);

    println!("Now checking for the area ...");
    let mut area: f64 = 0.0;
    let mut all_positive: bool = true;

    for function_value in function {
        let probability: f64 = match &function_value {
            Number::Float(x) => *x,
            Number::Integer(x) => *x as f64,
            Number::Rational(x) => x.to_f64().unwrap(),
        };

        if probability < 0.0 {
            all_positive = false;
        }

        area += probability;
    }
    println!("The area under f(x) is: {}", area);

    println!("Now checking for absolute value ...");
    if !all_positive {
        return Ok(false);
    }

    Ok((area > 1.0 - tolerance) && (area < 1.0 + tolerance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    #[test]
    fn verify_pdf_returns_err_for_non_pdf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0)],
            support: vec![Number::Float(1.0)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Continuous,
        };

        let result = rv.verify_pdf(None);
        assert!(result.is_err());
    }

    #[test]
    fn verify_pdf_accepts_exact_unit_area() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_rejects_area_outside_default_tolerance() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.49)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(!rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_uses_custom_tolerance() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.49)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(rv.verify_pdf(Some(0.02)).unwrap());
    }

    #[test]
    fn verify_pdf_supports_rational_values() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_returns_false_with_negative_function_values() {
        let rv = RandomVariable {
            function: vec![Number::Float(-0.5), Number::Float(1.5)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };
        assert!(!rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn to_pdf_returns_clone_when_already_pdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.8)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_pdf().unwrap();

        assert_eq!(result.function.len(), rv.function.len());
        assert!(matches!(result.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(result.function[1], Number::Float(x) if x == 0.8));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_pdf_converts_cdf_to_pdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_pdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.2).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.5).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.support[2], Number::Integer(3)));
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_pdf_converts_sf_to_pdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.8), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_pdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.2).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.5).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_pdf_propagates_conversion_error_for_empty_cdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_pdf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the pdf. function is empty"));
    }

    #[test]
    fn to_cdf_returns_clone_when_already_cdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_cdf().unwrap();

        assert_eq!(result.function.len(), rv.function.len());
        assert!(matches!(result.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(result.function[1], Number::Float(x) if x == 0.7));
        assert!(matches!(result.function[2], Number::Float(x) if x == 1.0));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.support[2], Number::Integer(3)));
        assert!(matches!(result.functional_form, FunctionalForm::Cdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_cdf_converts_pdf_to_cdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_cdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(result.function[1], Number::Float(x) if x == 0.7));
        assert!(matches!(result.function[2], Number::Float(x) if x == 1.0));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.support[2], Number::Integer(3)));
        assert!(matches!(result.functional_form, FunctionalForm::Cdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_cdf_converts_sf_to_cdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.7), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_cdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.7).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Cdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_cdf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_cdf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn to_sf_returns_clone_when_already_sf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.7), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_sf().unwrap();

        assert_eq!(result.function, rv.function);
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Sf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_sf_converts_cdf_to_sf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_sf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.8).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Sf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_sf_converts_pdf_to_sf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_sf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.8).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Sf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_sf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_sf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn to_idf_returns_clone_when_already_idf() {
        let rv = RandomVariable {
            function: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            support: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            functional_form: FunctionalForm::Idf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.function);
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_converts_cdf_to_idf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.support);
        assert_eq!(result.support, rv.function);
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_converts_pdf_to_idf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.support);
        assert_eq!(
            result.support,
            vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_converts_sf_to_idf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.7), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.support);
        assert_eq!(result.support.len(), 3);
        assert!(matches!(result.support[0], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert!(matches!(result.support[1], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.support[2], Number::Float(x) if (x - 0.7).abs() < 1e-12));
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_chf_converts_sf_to_chf() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(9, 10)),
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(1, 10)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_chf().unwrap();

        assert_eq!(result.support, rv.support);
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 9)),
                Number::Rational(Rational64::new(19, 9)),
                Number::Rational(Rational64::new(37, 9)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Chf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_chf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_chf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn to_hf_converts_sf_to_hf() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(9, 10)),
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(1, 10)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_hf().unwrap();

        assert_eq!(result.support, rv.support);
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 9)),
                Number::Rational(Rational64::new(2, 1)),
                Number::Rational(Rational64::new(2, 1)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Hf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_idf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }
}
