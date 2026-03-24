#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};


/// Truncates a discrete random variable by cutting off a portion of the support
/// and normalizng total probability of the distribution to 1
///
/// # Arguments
/// * `random_variable` - the random variable to truncate
/// * `min_support` - the minimum support of the new random variable.
///   Must be greater than or equal to the current minimum support.
/// * `max_support` - the maximum support of the new random variable.
///   Must be less than or equal to the current minimum support.
///
/// # Returns
/// * `truncated_rv` - the truncated random variable
pub fn truncate_discrete(
    random_variable: &RandomVariable,
    min_support: Number,
    max_support: Number,
) -> Result<RandomVariable, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = pdf_random_variable.function;
    let support = pdf_random_variable.support;

    let first_support = *support.first().expect("could not extract the first item");
    if min_support < first_support {
        return Err(
            "min support must be greater than or equal to the lowest support value"
            .to_string()
        );
    }

    let last_support = *support.last().expect("could not extract the first item");
    if max_support > last_support {
        return Err(
            "min support must be less than or equal to the highest support value"
            .to_string()
        );
    }

    let mut truncation_area = Number::Integer(0);
    for (&support_value, &function_value) in support.iter().zip(function.iter()) {
        if support_value >= min_support && support_value <= max_support {
            truncation_area += function_value;
        }
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
