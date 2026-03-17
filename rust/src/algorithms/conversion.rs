#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

/// Computes the cumulative sum of a function. Used to compute both the CDF from the PDF
/// and the cumulative hazard function from the hazard function.
fn cumulative_sum(function: &[Number]) -> Vec<Number> {
    function
        .iter()
        .scan(Number::default(), |accumulator, &x| {
            *accumulator += x;
            Some(*accumulator)
        })
        .collect()
}

/// Converts a discrete PDF to a discrete CDF
pub fn discrete_pdf_to_cdf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    if function.is_empty() {
        return Err("cannot compute the cdf. function is empty".to_string());
    }

    if random_variable.functional_form != FunctionalForm::Pdf {
        return Err(
            "discrete_pdf_to_cdf requires an input with the pdf functional form".to_string(),
        );
    }

    let cdf_function = cumulative_sum(function);

    let cdf_random_variable = RandomVariable {
        function: cdf_function,
        support: random_variable.support.clone(),
        functional_form: FunctionalForm::Cdf,
        domain_type: DomainType::Discrete,
    };

    Ok(cdf_random_variable)
}

/// Converts a discrete HF to a discrete CHF
pub fn discrete_hf_to_chf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    if function.is_empty() {
        return Err("cannot compute the chf. function is empty".to_string());
    }

    if random_variable.functional_form != FunctionalForm::Hf {
        return Err("discrete_hf_to_chf requires an input with the hf functional form".to_string());
    }

    let chf_function = cumulative_sum(function);

    let chf_random_variable = RandomVariable {
        function: chf_function,
        support: random_variable.support.clone(),
        functional_form: FunctionalForm::Chf,
        domain_type: DomainType::Discrete,
    };

    Ok(chf_random_variable)
}

/// Differentiates a cumulative function to convert CDF -> PDF or CHF -> HF
fn differentiate_cumulative_function(cumulative_function: &[Number]) -> Vec<Number> {
    let differentiated_function: Vec<Number> =
        // The previous cumulative function, [0, F1, F2, F3, ...]
        std::iter::once(Number::default()).chain(cumulative_function.iter().copied())
        // The current cumulative function, [F1, F2, F3, ...]
        .zip(cumulative_function.iter().copied())
        // Computes [F1-0, F2-F1, F3-F2, ...]
        .map(|(previous, current)| current - previous)
        .collect();

    differentiated_function
}

/// Converts a discrete CDF to a discrete PDF
pub fn discrete_cdf_to_pdf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    if function.is_empty() {
        return Err("cannot compute the pdf. function is empty".to_string());
    }

    if random_variable.functional_form != FunctionalForm::Cdf {
        return Err(
            "discrete_cdf_to_pdf requires an input with the cdf functional form".to_string(),
        );
    }

    let pdf_function = differentiate_cumulative_function(function);
    let pdf_random_variable = RandomVariable {
        function: pdf_function,
        support: random_variable.support.clone(),
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };

    Ok(pdf_random_variable)
}

/// Converts a discrete CHF to a discrete HF
pub fn discrete_chf_to_hf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    if function.is_empty() {
        return Err("cannot compute the hf. function is empty".to_string());
    }

    if random_variable.functional_form != FunctionalForm::Chf {
        return Err(
            "discrete_chf_to_hf requires an input with the chf functional form".to_string(),
        );
    }

    let hf_function = differentiate_cumulative_function(function);
    let hf_random_variable = RandomVariable {
        function: hf_function,
        support: random_variable.support.clone(),
        functional_form: FunctionalForm::Hf,
        domain_type: DomainType::Discrete,
    };

    Ok(hf_random_variable)
}

/// Converts between CDF and SF using the CDF = 1 - SF relatonship
pub fn swap_discrete_cdf_and_sf(
    random_variable: &RandomVariable,
) -> Result<RandomVariable, String> {
    let original_function = &random_variable.function;
    if original_function.is_empty() {
        return Err("cannot swap cdf and sf. function is empty".to_string());
    }

    let functional_form = match &random_variable.functional_form {
        FunctionalForm::Cdf => Ok(FunctionalForm::Sf),
        FunctionalForm::Sf => Ok(FunctionalForm::Cdf),
        _ => Err(
            "swap_discrete_cdf_and_sf requires an input with the cdf or sf functional form"
                .to_string(),
        ),
    };

    let swapped_function: Vec<Number> = original_function
        .iter()
        .map(|value| Number::one() - *value)
        .collect();

    let swapped_random_variable = RandomVariable {
        function: swapped_function,
        support: random_variable.support.clone(),
        functional_form: functional_form?,
        domain_type: DomainType::Discrete,
    };

    Ok(swapped_random_variable)
}

/// Converts between discrete CDF and IDF representations.
///
/// IDF is represented as quantiles in `function` and lower probability bounds
/// in `support`.
pub fn swap_discrete_cdf_and_idf(
    random_variable: &RandomVariable,
) -> Result<RandomVariable, String> {
    let original_function = &random_variable.function;
    let original_support = &random_variable.support;
    let original_functional_form = &random_variable.functional_form;

    if original_function.is_empty() {
        return Err("cannot swap cdf and idf. function is empty".to_string());
    }

    if original_function.len() != original_support.len() {
        return Err(
            "cannot swap cdf and idf. function and support must have the same length".to_string(),
        );
    }

    let swapped_random_variable =
        match original_functional_form {
            // For discrete RVs, represent IDF as:
            // support[p_i lower bounds] -> function[quantiles]
            // where p_0 = 0 and p_i = F(x_{i-1}) for i > 0.
            FunctionalForm::Cdf => {
                let idf_support: Vec<Number> = std::iter::once(Number::default())
                    .chain(
                        original_function
                            .iter()
                            .take(original_function.len() - 1)
                            .copied(),
                    )
                    .collect();

                RandomVariable {
                    function: original_support.clone(),
                    support: idf_support,
                    functional_form: FunctionalForm::Idf,
                    domain_type: DomainType::Discrete,
                }
            }
            // Convert IDF back into CDF using the stored lower probability bounds.
            // This preserves the discrete step alignment used by SF conversions,
            // so SF starts at 1 and remains non-increasing for valid IDFs.
            FunctionalForm::Idf => {
                let cdf_function = original_support.clone();

                RandomVariable {
                    function: cdf_function,
                    support: original_function.clone(),
                    functional_form: FunctionalForm::Cdf,
                    domain_type: DomainType::Discrete,
                }
            }
            _ => return Err(
                "swap_discrete_cdf_and_idf requires an input with the cdf or idf functional form"
                    .to_string(),
            ),
        };

    Ok(swapped_random_variable)
}

/// Converts a random variable to hazard function form, using the relationship
/// HF(x) = PDF(x) / SF(x)
pub fn discrete_rv_to_hf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    let pdf = random_variable.to_pdf()?;
    let sf = random_variable.to_sf()?;

    if pdf.function.is_empty() {
        return Err("cannot compute hf. function is empty".to_string());
    }

    let hf_function: Vec<Number> = pdf
        .function
        .iter()
        .zip(sf.function.iter())
        .map(|(pdf_value, sf_value)| {
            if sf_value.to_f64() <= 0.0 {
                return Number::Float(f64::INFINITY);
            }
            *pdf_value / *sf_value
        })
        .collect();

    let hf_random_variable = RandomVariable {
        function: hf_function,
        support: pdf.support.clone(),
        functional_form: FunctionalForm::Hf,
        domain_type: DomainType::Discrete,
    };

    Ok(hf_random_variable)
}

/// Converts a discrete cumulative hazard function to a survivor function
/// using the relationship SF(x) = exp(-CHF(x))
pub fn discrete_chf_to_sf(random_variable: &RandomVariable) -> Result<RandomVariable, String> {
    if random_variable.functional_form != FunctionalForm::Chf {
        return Err(
            "discrete_chf_to_sf requires an input with the chf functional form".to_string(),
        );
    }

    if random_variable.function.is_empty() {
        return Err("cannot compute sf. function is empty".to_string());
    }

    let sf_function: Vec<Number> = random_variable
        .function
        .iter()
        .map(|value| {
            let sf_value = (-value.to_f64()).exp();
            if sf_value == 0.0 {
                Number::Integer(0)
            } else if sf_value == 1.0 {
                Number::Integer(1)
            } else {
                Number::Float(sf_value)
            }
        })
        .collect();

    let sf_random_variable = RandomVariable {
        function: sf_function,
        support: random_variable.support.clone(),
        functional_form: FunctionalForm::Sf,
        domain_type: DomainType::Discrete,
    };

    Ok(sf_random_variable)
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
    fn discrete_pdf_to_cdf_returns_error_for_non_pdf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.8)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_pdf_to_cdf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "discrete_pdf_to_cdf requires an input with the pdf functional form")
        );
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
    fn discrete_hf_to_chf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Hf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_hf_to_chf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot compute the chf. function is empty"));
    }

    #[test]
    fn discrete_hf_to_chf_returns_error_for_non_hf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.1), Number::Float(0.9)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_hf_to_chf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "discrete_hf_to_chf requires an input with the hf functional form")
        );
    }

    #[test]
    fn discrete_hf_to_chf_builds_running_total_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.3), Number::Float(0.5)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Hf,
            domain_type: DomainType::Discrete,
        };

        let chf = discrete_hf_to_chf(&rv).unwrap();

        assert!(matches!(chf.functional_form, FunctionalForm::Chf));
        assert!(matches!(chf.domain_type, DomainType::Discrete));
        assert_eq!(chf.function.len(), 3);
        assert!(matches!(chf.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(chf.function[1], Number::Float(x) if x == 0.5));
        assert!(matches!(chf.function[2], Number::Float(x) if x == 1.0));

        assert_eq!(chf.support.len(), 3);
        assert!(matches!(chf.support[0], Number::Integer(1)));
        assert!(matches!(chf.support[1], Number::Integer(2)));
        assert!(matches!(chf.support[2], Number::Integer(3)));
    }

    #[test]
    fn discrete_hf_to_chf_supports_rational_values() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Hf,
            domain_type: DomainType::Discrete,
        };

        let chf = discrete_hf_to_chf(&rv).unwrap();

        assert_eq!(chf.function[0], Number::Rational(Rational64::new(1, 4)));
        assert_eq!(chf.function[1], Number::Rational(Rational64::new(1, 2)));
        assert_eq!(chf.function[2], Number::Rational(Rational64::new(1, 1)));
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
    fn discrete_cdf_to_pdf_returns_error_for_non_cdf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.8)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_cdf_to_pdf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "discrete_cdf_to_pdf requires an input with the cdf functional form")
        );
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
    fn discrete_chf_to_hf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Chf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_chf_to_hf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot compute the hf. function is empty"));
    }

    #[test]
    fn discrete_chf_to_hf_returns_error_for_non_chf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.8)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_chf_to_hf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "discrete_chf_to_hf requires an input with the chf functional form")
        );
    }

    #[test]
    fn discrete_chf_to_hf_differences_running_total_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 1)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Chf,
            domain_type: DomainType::Discrete,
        };

        let hf = discrete_chf_to_hf(&rv).unwrap();

        assert!(matches!(hf.functional_form, FunctionalForm::Hf));
        assert!(matches!(hf.domain_type, DomainType::Discrete));
        assert_eq!(hf.function.len(), 3);
        assert_eq!(hf.function[0], Number::Rational(Rational64::new(1, 10)));
        assert_eq!(hf.function[1], Number::Rational(Rational64::new(3, 10)));
        assert_eq!(hf.function[2], Number::Rational(Rational64::new(3, 5)));

        assert_eq!(hf.support.len(), 3);
        assert!(matches!(hf.support[0], Number::Integer(1)));
        assert!(matches!(hf.support[1], Number::Integer(2)));
        assert!(matches!(hf.support[2], Number::Integer(3)));
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
    fn swap_discrete_cdf_and_sf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = swap_discrete_cdf_and_sf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot swap cdf and sf. function is empty"));
    }

    #[test]
    fn swap_discrete_cdf_and_sf_returns_error_for_non_cdf_sf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.1), Number::Float(0.3), Number::Float(0.6)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = swap_discrete_cdf_and_sf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "swap_discrete_cdf_and_sf requires an input with the cdf or sf functional form")
        );
    }

    #[test]
    fn swap_discrete_cdf_and_idf_returns_error_for_non_cdf_idf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.8)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = swap_discrete_cdf_and_idf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "swap_discrete_cdf_and_idf requires an input with the cdf or idf functional form")
        );
    }

    #[test]
    fn swap_discrete_cdf_and_idf_converts_cdf_to_inverse_cdf() {
        let cdf = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let idf = swap_discrete_cdf_and_idf(&cdf).unwrap();

        assert_eq!(idf.function, cdf.support);
        assert_eq!(
            idf.support,
            vec![Number::Integer(0), Number::Float(0.2), Number::Float(0.7)]
        );
        assert!(matches!(idf.functional_form, FunctionalForm::Idf));
        assert!(matches!(idf.domain_type, DomainType::Discrete));
    }

    #[test]
    fn swap_discrete_cdf_and_idf_converts_inverse_cdf_to_cdf() {
        let idf = RandomVariable {
            function: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            support: vec![Number::Integer(0), Number::Float(0.2), Number::Float(0.7)],
            functional_form: FunctionalForm::Idf,
            domain_type: DomainType::Discrete,
        };

        let cdf = swap_discrete_cdf_and_idf(&idf).unwrap();

        assert_eq!(
            cdf.function,
            vec![Number::Integer(0), Number::Float(0.2), Number::Float(0.7)]
        );
        assert_eq!(cdf.support, idf.function);
        assert!(matches!(cdf.functional_form, FunctionalForm::Cdf));
        assert!(matches!(cdf.domain_type, DomainType::Discrete));
    }

    #[test]
    fn swap_discrete_cdf_and_idf_returns_error_for_mismatched_lengths() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(1.0)],
            support: vec![Number::Integer(1)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = swap_discrete_cdf_and_idf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "cannot swap cdf and idf. function and support must have the same length")
        );
    }

    #[test]
    fn swap_cdf_to_sf_complements_function_and_sets_metadata() {
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

        let swapped = swap_discrete_cdf_and_sf(&rv).unwrap();

        assert!(matches!(swapped.functional_form, FunctionalForm::Sf));
        assert!(matches!(swapped.domain_type, DomainType::Discrete));
        assert_eq!(
            swapped.function,
            vec![
                Number::Rational(Rational64::new(9, 10)),
                Number::Rational(Rational64::new(3, 5)),
                Number::Rational(Rational64::new(0, 1)),
            ]
        );
        assert_eq!(swapped.support, rv.support);
    }

    #[test]
    fn swap_sf_to_cdf_complements_function_and_sets_metadata() {
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

        let swapped = swap_discrete_cdf_and_sf(&rv).unwrap();

        assert!(matches!(swapped.functional_form, FunctionalForm::Cdf));
        assert!(matches!(swapped.domain_type, DomainType::Discrete));
        assert_eq!(
            swapped.function,
            vec![
                Number::Rational(Rational64::new(0, 1)),
                Number::Rational(Rational64::new(3, 5)),
                Number::Rational(Rational64::new(9, 10)),
            ]
        );
        assert_eq!(swapped.support, rv.support);
    }

    #[test]
    fn swap_discrete_cdf_and_sf_twice_returns_original_random_variable() {
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

        let swapped_once = swap_discrete_cdf_and_sf(&original).unwrap();
        let swapped_twice = swap_discrete_cdf_and_sf(&swapped_once).unwrap();

        assert_eq!(swapped_twice.function, original.function);
        assert_eq!(swapped_twice.support, original.support);
        assert!(matches!(swapped_twice.functional_form, FunctionalForm::Cdf));
        assert!(matches!(swapped_twice.domain_type, DomainType::Discrete));
    }

    #[test]
    fn discrete_rv_to_hf_returns_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_rv_to_hf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn discrete_rv_to_hf_computes_hf_from_pdf_and_sf_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(9, 10)),
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(1, 10)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let hf = discrete_rv_to_hf(&rv).unwrap();

        assert!(matches!(hf.functional_form, FunctionalForm::Hf));
        assert!(matches!(hf.domain_type, DomainType::Discrete));
        assert_eq!(hf.support, rv.support);
        assert_eq!(
            hf.function,
            vec![
                Number::Rational(Rational64::new(1, 9)),
                Number::Rational(Rational64::new(2, 1)),
                Number::Rational(Rational64::new(2, 1)),
            ]
        );
    }

    #[test]
    fn discrete_rv_to_hf_maps_zero_sf_entries_to_infinity_without_dividing() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let hf = discrete_rv_to_hf(&rv).unwrap();

        assert_eq!(hf.function.len(), 2);
        assert_eq!(hf.function[0], Number::Rational(Rational64::new(1, 1)));
        assert!(
            matches!(hf.function[1], Number::Float(x) if x.is_infinite() && x.is_sign_positive())
        );
    }

    #[test]
    fn discrete_chf_to_sf_returns_error_for_non_chf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.1), Number::Float(0.2)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_chf_to_sf(&rv);
        assert!(
            matches!(result, Err(msg) if msg == "discrete_chf_to_sf requires an input with the chf functional form")
        );
    }

    #[test]
    fn discrete_chf_to_sf_returns_error_for_empty_function() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Chf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_chf_to_sf(&rv);
        assert!(matches!(result, Err(msg) if msg == "cannot compute sf. function is empty"));
    }

    #[test]
    fn discrete_chf_to_sf_computes_survivor_function_and_sets_metadata() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.0), Number::Float(1.0), Number::Float(2.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Chf,
            domain_type: DomainType::Discrete,
        };

        let sf = discrete_chf_to_sf(&rv).unwrap();

        assert!(matches!(sf.functional_form, FunctionalForm::Sf));
        assert!(matches!(sf.domain_type, DomainType::Discrete));
        assert_eq!(sf.support, rv.support);
        assert_eq!(sf.function.len(), 3);
        assert!(matches!(sf.function[0], Number::Integer(1)));
        assert!(matches!(sf.function[1], Number::Float(x) if (x - (-1.0f64).exp()).abs() < 1e-12));
        assert!(matches!(sf.function[2], Number::Float(x) if (x - (-2.0f64).exp()).abs() < 1e-12));
    }

    #[test]
    fn discrete_chf_to_sf_maps_positive_infinity_to_exact_zero() {
        let rv = RandomVariable {
            function: vec![Number::Float(f64::INFINITY)],
            support: vec![Number::Integer(1)],
            functional_form: FunctionalForm::Chf,
            domain_type: DomainType::Discrete,
        };

        let sf = discrete_chf_to_sf(&rv).unwrap();
        assert_eq!(sf.function, vec![Number::Integer(0)]);
    }
}
