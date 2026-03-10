#![allow(dead_code)]

use num_rational::Rational64;
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone)]
pub enum Number {
    Float(f64),
    Integer(i64),
    Rational(Rational64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionalForm {
    Pdf,
    Cdf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DomainType {
    Continuous,
    Discrete,
    DiscreteFunctional,
}

#[derive(Debug, Clone)]
pub struct RandomVariable {
    pub function: Vec<Number>,
    pub support: Vec<Number>,
    pub ftype: (FunctionalForm, DomainType),
}

impl RandomVariable {
    fn verify_pdf(&self, tolerance: Option<f64>) -> Result<bool, String> {
        if self.ftype.0 != FunctionalForm::Pdf {
            return Err("verify_pdf only works for PDFs".to_string());
        }

        verify_pdf(&self.function, tolerance)
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

    #[test]
    fn verify_pdf_returns_err_for_non_pdf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0)],
            support: vec![Number::Float(1.0)],
            ftype: (FunctionalForm::Cdf, DomainType::Continuous),
        };

        let result = rv.verify_pdf(None);
        assert!(result.is_err());
    }

    #[test]
    fn verify_pdf_accepts_exact_unit_area() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            ftype: (FunctionalForm::Pdf, DomainType::Continuous),
        };

        assert!(rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_rejects_area_outside_default_tolerance() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.49)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            ftype: (FunctionalForm::Pdf, DomainType::Continuous),
        };

        assert!(!rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_uses_custom_tolerance() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.49)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            ftype: (FunctionalForm::Pdf, DomainType::Continuous),
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
            ftype: (FunctionalForm::Pdf, DomainType::Continuous),
        };

        assert!(rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_returns_false_with_negative_function_values() {
        let rv = RandomVariable {
            function: vec![Number::Float(-0.5), Number::Float(1.5)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            ftype: (FunctionalForm::Pdf, DomainType::Continuous),
        };
        assert!(!rv.verify_pdf(None).unwrap());
    }
}
