#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

/// Converts a discrete PDF to a discrete CDF
pub fn discrete_pdf_to_cdf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    let function_length = function.len();

    if function_length == 0 {
        return Err("cannot compute the cdf. function is empty".to_string());
    }

    let mut cdf_function = Vec::with_capacity(function_length);

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

/// Converts a discrete CDF to a discrete PDF
pub fn discrete_cdf_to_pdf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    let function_length = function.len();

    if function_length == 0 {
        return Err("cannot compute the pdf. function is empty".to_string());
    }

    let mut pdf_function = Vec::with_capacity(function_length);

    for (i, function_value) in function.iter().enumerate() {
        if i == 0 {
            pdf_function.push(*function_value)
        } else {
            let previous_value = function[i - 1];
            let pdf_value = *function_value - previous_value;
            pdf_function.push(pdf_value);
        }
    }

    let pdf_random_variable = RandomVariable {
        function: pdf_function,
        support: random_variable.support.clone(),
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };

    Ok(pdf_random_variable)
}

/// Converts between CDF and SF using the CDF = 1 - SF relatonship
pub fn swap_cdf_and_sf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let original_function = &random_variable.function;
    let function_length = original_function.len();

    if function_length == 0 {
        return Err("cannot swap cdf and sf. function is empty".to_string());
    }

    let functional_form = match &random_variable.functional_form {
        FunctionalForm::Cdf => Ok(FunctionalForm::Sf),
        FunctionalForm::Sf => Ok(FunctionalForm::Cdf),
        _ => Err("swap_cdf_and_sf only works on cdf and sf functional forms".to_string()),
    };

    let mut swapped_function = Vec::with_capacity(function_length);

    for value in original_function.iter().rev() {
        swapped_function.push(*value);
    }

    let swapped_random_variable = RandomVariable {
        function: swapped_function,
        support: random_variable.support.clone(),
        functional_form: functional_form?,
        domain_type: DomainType::Discrete,
    };

    Ok(swapped_random_variable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    #[test]
    fn discrete_pdf_to_cdf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_pdf_to_cdf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn discrete_pdf_to_cdf_builds_running_total_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.3), Number::Float(0.5)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let cdf = discrete_pdf_to_cdf(&rv).unwrap();

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
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let cdf = discrete_pdf_to_cdf(&rv).unwrap();

        assert_eq!(cdf.function[0], Number::Rational(Rational64::new(1, 4)));
        assert_eq!(cdf.function[1], Number::Rational(Rational64::new(1, 2)));
        assert_eq!(cdf.function[2], Number::Rational(Rational64::new(1, 1)));
    }

    #[test]
    fn discrete_cdf_to_pdf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_cdf_to_pdf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot compute the pdf. function is empty"));
    }

    #[test]
    fn discrete_cdf_to_pdf_differences_running_total_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 1)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let pdf = discrete_cdf_to_pdf(&rv).unwrap();

        assert!(matches!(pdf.functional_form, FunctionalForm::Pdf));
        assert!(matches!(pdf.domain_type, DomainType::Discrete));
        assert_eq!(pdf.function.len(), 3);
        assert_eq!(pdf.function[0], Number::Rational(Rational64::new(1, 10)));
        assert_eq!(pdf.function[1], Number::Rational(Rational64::new(3, 10)));
        assert_eq!(pdf.function[2], Number::Rational(Rational64::new(3, 5)));

        assert_eq!(pdf.support.len(), 3);
        assert!(matches!(pdf.support[0], Number::Integer(1)));
        assert!(matches!(pdf.support[1], Number::Integer(2)));
        assert!(matches!(pdf.support[2], Number::Integer(3)));
    }

    #[test]
    fn discrete_pdf_to_cdf_to_pdf_returns_original_pdf() {
        let original_pdf_function = vec![
            Number::Rational(Rational64::new(1, 10)),
            Number::Rational(Rational64::new(3, 10)),
            Number::Rational(Rational64::new(3, 5)),
        ];

        let rv = RandomVariable {
            function: original_pdf_function.clone(),
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let cdf = discrete_pdf_to_cdf(&rv).unwrap();
        let pdf = discrete_cdf_to_pdf(&cdf).unwrap();

        assert_eq!(pdf.function, original_pdf_function);
    }

    #[test]
    fn discrete_cdf_to_pdf_to_cdf_returns_original_cdf() {
        let original_cdf_function = vec![
            Number::Rational(Rational64::new(1, 10)),
            Number::Rational(Rational64::new(2, 5)),
            Number::Rational(Rational64::new(1, 1)),
        ];

        let rv = RandomVariable {
            function: original_cdf_function.clone(),
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let pdf = discrete_cdf_to_pdf(&rv).unwrap();
        let cdf = discrete_pdf_to_cdf(&pdf).unwrap();

        assert_eq!(cdf.function, original_cdf_function);
    }

    #[test]
    fn swap_cdf_and_sf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = swap_cdf_and_sf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot swap cdf and sf. function is empty"));
    }

    #[test]
    fn swap_cdf_and_sf_returns_error_for_non_cdf_sf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.1), Number::Float(0.3), Number::Float(0.6)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = swap_cdf_and_sf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "swap_cdf_and_sf only works on cdf and sf functional forms")
        );
    }

    #[test]
    fn swap_cdf_to_sf_reverses_function_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 1)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let swapped = swap_cdf_and_sf(&rv).unwrap();

        assert!(matches!(swapped.functional_form, FunctionalForm::Sf));
        assert!(matches!(swapped.domain_type, DomainType::Discrete));
        assert_eq!(
            swapped.function,
            vec![
                Number::Rational(Rational64::new(1, 1)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 10)),
            ]
        );
        assert_eq!(swapped.support, rv.support);
    }

    #[test]
    fn swap_sf_to_cdf_reverses_function_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 1)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 10)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let swapped = swap_cdf_and_sf(&rv).unwrap();

        assert!(matches!(swapped.functional_form, FunctionalForm::Cdf));
        assert!(matches!(swapped.domain_type, DomainType::Discrete));
        assert_eq!(
            swapped.function,
            vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 1)),
            ]
        );
        assert_eq!(swapped.support, rv.support);
    }

    #[test]
    fn swap_cdf_and_sf_twice_returns_original_random_variable() {
        let original = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 1)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let swapped_once = swap_cdf_and_sf(&original).unwrap();
        let swapped_twice = swap_cdf_and_sf(&swapped_once).unwrap();

        assert_eq!(swapped_twice.function, original.function);
        assert_eq!(swapped_twice.support, original.support);
        assert!(matches!(swapped_twice.functional_form, FunctionalForm::Cdf));
        assert!(matches!(swapped_twice.domain_type, DomainType::Discrete));
    }
}
