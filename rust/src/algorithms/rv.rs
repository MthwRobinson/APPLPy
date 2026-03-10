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
        let tolerance = tolerance.unwrap_or(0.00001);

        if self.ftype.0 != FunctionalForm::Pdf {
            return Err("verify_pdf only works for PDFs".to_string());
        }

        let mut area: f64 = 0.0;
        let support_len: usize = self.function.len();

        for i in 0..support_len {
            let function: f64 = match &self.function[i] {
                Number::Float(x) => *x,
                Number::Integer(x) => *x as f64,
                Number::Rational(x) => x.to_f64().unwrap(),
            };

            let support: f64 = match &self.support[i] {
                Number::Float(x) => *x,
                Number::Integer(x) => *x as f64,
                Number::Rational(x) => x.to_f64().unwrap(),
            };

            let probability = function * support;
            area += probability
        }

        Ok((area > 1.0 - tolerance) && (area < 1.0 + tolerance))
    }
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
}
