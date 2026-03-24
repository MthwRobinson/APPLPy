#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

/// Truncates a discrete random variable by cutting off a portion of the support
/// and normalizing total probability of the distribution to 1.
///
/// # Arguments
/// * `random_variable` - the random variable to truncate
/// * `min_support` - the minimum support of the new random variable.
///   Must be greater than or equal to the current minimum support.
/// * `max_support` - the maximum support of the new random variable.
///   Must be less than or equal to the current maximum support.
///
/// # Returns
/// * `truncated_rv` - the truncated random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use applpy_rust::algorithms::transform::truncate_discrete;
/// use num_rational::Rational64;
///
/// let rv = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 10)),
///         Number::Rational(Rational64::new(2, 10)),
///         Number::Rational(Rational64::new(3, 10)),
///         Number::Rational(Rational64::new(4, 10)),
///     ],
///     support: vec![
///         Number::Integer(1),
///         Number::Integer(2),
///         Number::Integer(3),
///         Number::Integer(4),
///     ],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let truncated = truncate_discrete(&rv, Number::Integer(2), Number::Integer(3)).unwrap();
///
/// assert_eq!(truncated.support, vec![Number::Integer(2), Number::Integer(3)]);
/// assert_eq!(
///     truncated.function,
///     vec![
///         Number::Rational(Rational64::new(2, 5)),
///         Number::Rational(Rational64::new(3, 5)),
///     ]
/// );
/// assert!(matches!(truncated.functional_form, FunctionalForm::Pdf));
/// assert!(matches!(truncated.domain_type, DomainType::Discrete));
/// ```
pub fn truncate_discrete(
    random_variable: &RandomVariable,
    min_support: Number,
    max_support: Number,
) -> Result<RandomVariable, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = pdf_random_variable.function;
    let support = pdf_random_variable.support;

    if min_support >= max_support {
        return Err("max_support must be greater than the min_support".to_string());
    }

    let first_support = *support.first().ok_or("support is empty")?;
    if min_support < first_support {
        return Err(
            "min support must be greater than or equal to the lowest support value".to_string(),
        );
    }

    let last_support = *support.last().ok_or("support is empty")?;
    if max_support > last_support {
        return Err(
            "max support must be less than or equal to the highest support value".to_string(),
        );
    }

    let mut truncation_area = Number::Integer(0);
    for (&support_value, &function_value) in support.iter().zip(function.iter()) {
        if support_value >= min_support && support_value <= max_support {
            truncation_area += function_value;
        }
    }

    let zero = Number::Integer(0);
    if truncation_area == zero {
        return Err("there is no probability mass within the specified support range".to_string());
    }

    let mut truncated_function = Vec::new();
    let mut truncated_support = Vec::new();

    for (&support_value, &function_value) in support.iter().zip(function.iter()) {
        if support_value >= min_support && support_value <= max_support {
            let probability = function_value / truncation_area;
            truncated_function.push(probability);
            truncated_support.push(support_value);
        }
    }

    let truncated_rv = RandomVariable {
        function: truncated_function,
        support: truncated_support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };
    Ok(truncated_rv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    fn sample_discrete_rv() -> RandomVariable {
        RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 10)),
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(4, 10)),
            ],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        }
    }

    #[test]
    fn truncate_discrete_renormalizes_probabilities_within_range() {
        let rv = sample_discrete_rv();
        let truncated = truncate_discrete(&rv, Number::Integer(2), Number::Integer(3)).unwrap();

        assert_eq!(
            truncated.support,
            vec![Number::Integer(2), Number::Integer(3)]
        );
        assert_eq!(
            truncated.function,
            vec![
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(3, 5))
            ]
        );
        assert!(matches!(truncated.functional_form, FunctionalForm::Pdf));
        assert!(matches!(truncated.domain_type, DomainType::Discrete));
    }

    #[test]
    fn truncate_discrete_returns_error_when_min_support_exceeds_bounds() {
        let rv = sample_discrete_rv();
        let result = truncate_discrete(&rv, Number::Integer(0), Number::Integer(3));

        assert!(matches!(
            result,
            Err(msg) if msg == "min support must be greater than or equal to the lowest support value"
        ));
    }
}
