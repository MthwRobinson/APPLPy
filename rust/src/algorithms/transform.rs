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

/// Computes the mixture of two random variables
///
/// # Arguments
/// * `random_variables` - a list of random variables to mix
/// * `mix_weights` - the weight for each random variable. Must sum to 1
///
/// # Returns
/// * `mixed_rv` - the weighted mixture of the random variables
pub fn mixture_discrete(
    random_variables: &[&RandomVariable],
    mix_weights: &[Number],
) -> Result<RandomVariable, String> {
    if random_variables.len() != mix_weights.len() {
        return Err("the number of random variables and mix weights must be equal".to_string());
    }

    let weight_sum: Number = mix_weights.iter().copied().sum();
    let one = Number::Integer(1);
    let tolerance = Number::Float(1e-6);
    if weight_sum < one - tolerance || weight_sum > one + tolerance {
        return Err("the mix weights must sum to one".to_string());
    }

    let mut raw_mixture_function = Vec::new();
    let mut raw_mixture_support = Vec::new();

    for (&random_variable, &mix_weight) in random_variables.iter().zip(mix_weights.iter()) {
        let function = &random_variable.function;
        let support = &random_variable.support;

        for (&function_value, &support_value) in function.iter().zip(support.iter()) {
            let partial_probability = function_value * mix_weight;
            if let Some(support_index) =
                raw_mixture_support.iter().position(|&x| x == support_value)
            {
                raw_mixture_function[support_index] += partial_probability;
            } else {
                raw_mixture_support.push(support_value);
                raw_mixture_function.push(partial_probability);
            }
        }
    }

    let mut raw_mixture_pair: Vec<_> = raw_mixture_support
        .into_iter()
        .zip(raw_mixture_function)
        .collect();

    raw_mixture_pair.sort_by(|a, b| {
        let first_value = a.0.to_f64();
        let second_value = b.0.to_f64();
        first_value.total_cmp(&second_value)
    });

    let (mixture_support, mixture_function) = raw_mixture_pair.into_iter().unzip();

    let mix_rv = RandomVariable {
        function: mixture_function,
        support: mixture_support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };
    Ok(mix_rv)
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
