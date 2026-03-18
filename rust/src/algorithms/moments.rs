#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::RandomVariable;

/// Computes the mean of a discrete random variable.
///
/// # Arguments
/// * `random_variable`: a discrete random variable
///
/// # Returns
/// * `mean`: the mean of the random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_mean;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(1), Number::Integer(3)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let mean = discrete_mean(&rv).unwrap();
/// assert_eq!(mean, Number::Rational(Rational64::new(2, 1)));
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_mean;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // CDF for support [1, 2, 4]: [0.2, 0.7, 1.0]
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Cdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let mean = discrete_mean(&rv).unwrap();
/// assert!((mean.to_f64() - 2.4).abs() < 1e-12);
/// ```
pub fn discrete_mean(random_variable: &RandomVariable) -> Result<Number, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = &pdf_random_variable.function;
    let support = &pdf_random_variable.support;

    let mean = function
        .iter()
        .zip(support.iter())
        .fold(Number::default(), |mean, (&f, &s)| mean + (f * s));
    Ok(mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
    use num_rational::Rational64;

    #[test]
    fn discrete_mean_computes_expected_value_for_discrete_pdf() {
        // Fair die: E[X] = (1+2+3+4+5+6)/6 = 7/2
        let rv = RandomVariable {
            function: vec![Number::Rational(Rational64::new(1, 6)); 6],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let mean = discrete_mean(&rv).unwrap();
        assert_eq!(mean, Number::Rational(Rational64::new(7, 2)));
    }

    #[test]
    fn discrete_mean_works_for_cdf_input_by_converting_to_pdf() {
        // PDF from this CDF is [0.2, 0.5, 0.3], so E[X] = 0.2*1 + 0.5*2 + 0.3*4 = 2.4
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let mean = discrete_mean(&rv).unwrap();
        assert!((mean.to_f64() - 2.4).abs() < 1e-12);
    }

    #[test]
    fn discrete_mean_returns_error_when_pdf_conversion_fails() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_mean(&rv);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "cannot compute the pdf. function is empty"
        );
    }
}
