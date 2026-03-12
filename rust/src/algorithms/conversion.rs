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

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    #[test]
    fn discrete_pdf_to_cdf_returns_error_for_empty_function() {
        let mut rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_pdf_to_cdf(&mut rv);
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn discrete_pdf_to_cdf_builds_running_total_and_sets_metadata() {
        let mut rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.3), Number::Float(0.5)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let cdf = discrete_pdf_to_cdf(&mut rv).unwrap();

        assert!(matches!(cdf.functional_form, FunctionalForm::Cdf));
        assert!(matches!(cdf.domain_type, DomainType::Discrete));
        assert_eq!(cdf.function.len(), 3);
        assert!(matches!(cdf.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(cdf.function[1], Number::Float(x) if x == 0.5));
        assert!(matches!(cdf.function[2], Number::Float(x) if x == 1.0));

        assert_eq!(cdf.support.len(), 3);
        assert!(matches!(cdf.support[0], Number::Integer(1)));
        assert!(matches!(cdf.support[1], Number::Integer(2)));
        assert!(matches!(cdf.support[2], Number::Integer(3)));
    }

    #[test]
    fn discrete_pdf_to_cdf_supports_rational_values() {
        let mut rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let cdf = discrete_pdf_to_cdf(&mut rv).unwrap();

        assert!(matches!(cdf.function[0], Number::Rational(x) if x == Rational64::new(1, 4)));
        assert!(matches!(cdf.function[1], Number::Rational(x) if x == Rational64::new(1, 2)));
        assert!(matches!(cdf.function[2], Number::Rational(x) if x == Rational64::new(1, 1)));
    }
}
