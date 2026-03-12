#![allow(dead_code)]

use std::ops::{Add, AddAssign, Div, Mul, Sub};

use num_rational::Rational64;
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone, Copy)]
pub enum Number {
    Float(f64),
    Integer(i64),
    Rational(Rational64),
}

impl Number {
    fn to_rational(self) -> Rational64 {
        match self {
            Number::Float(x) => Rational64::approximate_float(x)
                .expect("cannot convert non-finite float to Rational64"),
            Number::Integer(x) => Rational64::from_integer(x),
            Number::Rational(x) => x,
        }
    }

    fn to_f64(self) -> f64 {
        match self {
            Number::Float(x) => x,
            Number::Integer(x) => x as f64,
            Number::Rational(x) => x.to_f64().expect("cannot convert Rational64 to f64"),
        }
    }

    fn promote(self, other: Self) -> (Self, Self) {
        match (&self, &other) {
            (Number::Float(_), _) | (_, Number::Float(_)) => {
                (Number::Float(self.to_f64()), Number::Float(other.to_f64()))
            }

            (Number::Rational(_), _) | (_, Number::Rational(_)) => (
                Number::Rational(self.to_rational()),
                Number::Rational(other.to_rational()),
            ),

            _ => (self, other),
        }
    }
}

impl Default for Number {
    fn default() -> Self {
        Number::Integer(0)
    }
}

impl Add for Number {
    type Output = Number;

    fn add(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a + b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a + b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a + b),
            _ => unreachable!(),
        }
    }
}

impl AddAssign for Number {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Number {
    type Output = Number;

    fn sub(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a - b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a - b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a - b),
            _ => unreachable!(),
        }
    }
}

impl Mul for Number {
    type Output = Number;

    fn mul(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a * b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a * b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a * b),
            _ => unreachable!(),
        }
    }
}

impl Div for Number {
    type Output = Number;

    fn div(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a / b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a / b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a / b),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionalForm {
    Cdf,
    Chf,
    Hf,
    Idf,
    Pdf,
    Sf,
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
}
