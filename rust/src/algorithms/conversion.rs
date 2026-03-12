#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

/// Converts a discrete PDF to a discrete PDF. Modifiers the random variable in place.
pub fn discrete_pdf_to_cdf(random_variable: &mut RandomVariable) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    let function_length = function.len();

    if function_length == 0 {
        return Err("cannot compute the cdf. function is empty".to_string());
    }

    let mut cdf_function = Vec::with_capacity(function.len());

    let mut cdf_area = Number::default();
    for function_value in function {
        cdf_area += *function_value;
        cdf_function.push(cdf_area);
    }

    let cdf_random_variable = RandomVariable {
        function: cdf_function,
        support: random_variable.support.clone(),
        functional_form: FunctionalForm::Cdf,
        domain_type: DomainType::Discrete,
    };

    Ok(cdf_random_variable)
}
