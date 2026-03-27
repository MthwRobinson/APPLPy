#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

/// Computes the product of two independent discrete random variables
///
/// # Arguments
/// * `random_variable_1` - the first random variable
/// * `random_variable_2` - the second random variable
///
/// # Returns
/// * `product_rv` - the product of the two random variables
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::algebra::product_discrete;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv1 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(1), Number::Integer(2)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let rv2 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(2), Number::Integer(3)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let product = product_discrete(&rv1, &rv2).unwrap();
///
/// assert_eq!(
///     product.support,
///     vec![Number::Integer(2), Number::Integer(3), Number::Integer(4), Number::Integer(6)]
/// );
/// assert_eq!(
///     product.function,
///     vec![
///         Number::Rational(Rational64::new(1, 4)),
///         Number::Rational(Rational64::new(1, 4)),
///         Number::Rational(Rational64::new(1, 4)),
///         Number::Rational(Rational64::new(1, 4)),
///     ]
/// );
/// assert!(product.verify_pdf(None).unwrap());
/// ```
pub fn product_discrete(
    random_variable_1: &RandomVariable,
    random_variable_2: &RandomVariable,
) -> Result<RandomVariable, String> {
    let pdf_random_variable_1 = random_variable_1.to_pdf()?;
    let function_1 = pdf_random_variable_1.function;
    let support_1 = pdf_random_variable_1.support;

    let pdf_random_variable_2 = random_variable_2.to_pdf()?;
    let function_2 = pdf_random_variable_2.function;
    let support_2 = pdf_random_variable_2.support;

    // Compute support1 x support2 and the associated probability
    // for all combinations of support values
    let mut raw_product_support = Vec::new();
    for &s1 in support_1.iter() {
        for &s2 in support_2.iter() {
            let support_value = s1 * s2;
            raw_product_support.push(support_value);
        }
    }

    let mut raw_product_function = Vec::new();
    for &f1 in function_1.iter() {
        for &f2 in function_2.iter() {
            let probability = f1 * f2;
            raw_product_function.push(probability);
        }
    }

    // Sort the multiplied support and function values
    let mut raw_product_pairs: Vec<_> = raw_product_support
        .into_iter()
        .zip(raw_product_function)
        .collect();
    raw_product_pairs.sort_by(|a, b| {
        let first_value = a.0.to_f64();
        let second_value = b.0.to_f64();
        first_value.total_cmp(&second_value)
    });

    let (sorted_support, sorted_function): (Vec<Number>, Vec<Number>) =
        raw_product_pairs.into_iter().unzip();

    // De-duplicate the support. If a value appears multiple times in the
    // support, combine the probabilities
    let mut product_function = Vec::new();
    let mut product_support = Vec::new();
    for (&s, &probability) in sorted_support.iter().zip(sorted_function.iter()) {
        let support_index = product_support
            .iter()
            .position(|&x: &Number| x.to_f64() == s.to_f64());

        match support_index {
            Some(index) => {
                product_function[index] += probability;
            }
            None => {
                product_function.push(probability);
                product_support.push(s);
            }
        }
    }

    let product_rv = RandomVariable {
        function: product_function,
        support: product_support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };
    Ok(product_rv)
}

/// Computes the sum of two independent discrete random variables
///
/// # Arguments
/// * `random_variable_1` - the first random variable
/// * `random_variable_2` - the second random variable
///
/// # Returns
/// * `sum_rv` - the product of the two random variables
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::algebra::convolution_discrete;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv1 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(1), Number::Integer(2)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let rv2 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(2), Number::Integer(3)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let sum = convolution_discrete(&rv1, &rv2).unwrap();
///
/// assert_eq!(
///     sum.support,
///     vec![Number::Integer(3), Number::Integer(4), Number::Integer(5)]
/// );
/// assert_eq!(
///     sum.function,
///     vec![
///         Number::Rational(Rational64::new(1, 4)),
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 4)),
///     ]
/// );
/// assert!(sum.verify_pdf(None).unwrap());
/// ```
pub fn convolution_discrete(
    random_variable_1: &RandomVariable,
    random_variable_2: &RandomVariable,
) -> Result<RandomVariable, String> {
    let pdf_random_variable_1 = random_variable_1.to_pdf()?;
    let function_1 = pdf_random_variable_1.function;
    let support_1 = pdf_random_variable_1.support;

    let pdf_random_variable_2 = random_variable_2.to_pdf()?;
    let function_2 = pdf_random_variable_2.function;
    let support_2 = pdf_random_variable_2.support;

    // Find the values and probabilities for support_1 + support_2
    // for all combinations of support values
    let mut raw_conv_support = Vec::new();
    for &s1 in support_1.iter() {
        for &s2 in support_2.iter() {
            let support_sum = s1 + s2;
            raw_conv_support.push(support_sum);
        }
    }

    let mut raw_conv_function = Vec::new();
    for &f1 in function_1.iter() {
        for &f2 in function_2.iter() {
            let probability = f1 * f2;
            raw_conv_function.push(probability);
        }
    }

    // Sorts the results by the support values
    let mut raw_conv_pairs: Vec<_> = raw_conv_support
        .into_iter()
        .zip(raw_conv_function)
        .collect();

    raw_conv_pairs.sort_by(|a, b| {
        let first_value = a.0.to_f64();
        let second_value = b.0.to_f64();
        first_value.total_cmp(&second_value)
    });

    let (sorted_support, sorted_function): (Vec<Number>, Vec<Number>) =
        raw_conv_pairs.into_iter().unzip();

    // Remove redundant elements from the support
    let mut conv_support = Vec::new();
    let mut conv_function = Vec::new();

    for (&s, &f) in sorted_support.iter().zip(sorted_function.iter()) {
        let support_index = conv_support.iter().position(|&x| x == s);

        match support_index {
            Some(index) => {
                conv_function[index] += f;
            }
            None => {
                conv_support.push(s);
                conv_function.push(f);
            }
        }
    }

    let sum_rv = RandomVariable {
        function: conv_function,
        support: conv_support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };

    Ok(sum_rv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    fn two_point_pdf(
        support: [i64; 2],
        probabilities: [Rational64; 2],
        form: FunctionalForm,
    ) -> RandomVariable {
        RandomVariable {
            function: vec![
                Number::Rational(probabilities[0]),
                Number::Rational(probabilities[1]),
            ],
            support: vec![Number::Integer(support[0]), Number::Integer(support[1])],
            functional_form: form,
            domain_type: DomainType::Discrete,
        }
    }

    #[test]
    fn product_discrete_combines_duplicate_support_values() {
        let rv1 = two_point_pdf(
            [0, 1],
            [Rational64::new(1, 2), Rational64::new(1, 2)],
            FunctionalForm::Pdf,
        );
        let rv2 = two_point_pdf(
            [2, 3],
            [Rational64::new(1, 5), Rational64::new(4, 5)],
            FunctionalForm::Pdf,
        );

        let product = product_discrete(&rv1, &rv2).unwrap();

        assert_eq!(
            product.support,
            vec![Number::Integer(0), Number::Integer(2), Number::Integer(3)]
        );
        assert_eq!(
            product.function,
            vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(2, 5))
            ]
        );
        assert!(product.verify_pdf(None).unwrap());
    }

    #[test]
    fn product_discrete_accepts_cdf_inputs_by_converting_to_pdf() {
        let rv1 = two_point_pdf(
            [1, 2],
            [Rational64::new(1, 4), Rational64::new(1, 1)],
            FunctionalForm::Cdf,
        );
        let rv2 = two_point_pdf(
            [3, 5],
            [Rational64::new(1, 2), Rational64::new(1, 1)],
            FunctionalForm::Cdf,
        );

        let product = product_discrete(&rv1, &rv2).unwrap();

        assert_eq!(
            product.support,
            vec![
                Number::Integer(3),
                Number::Integer(5),
                Number::Integer(6),
                Number::Integer(10)
            ]
        );
        assert_eq!(
            product.function,
            vec![
                Number::Rational(Rational64::new(1, 8)),
                Number::Rational(Rational64::new(1, 8)),
                Number::Rational(Rational64::new(3, 8)),
                Number::Rational(Rational64::new(3, 8))
            ]
        );
        assert!(product.verify_pdf(None).unwrap());
    }

    #[test]
    fn product_discrete_preserves_pdf_and_discrete_domain() {
        let rv1 = two_point_pdf(
            [2, 4],
            [Rational64::new(1, 3), Rational64::new(2, 3)],
            FunctionalForm::Pdf,
        );
        let rv2 = two_point_pdf(
            [1, 3],
            [Rational64::new(3, 10), Rational64::new(7, 10)],
            FunctionalForm::Pdf,
        );

        let product = product_discrete(&rv1, &rv2).unwrap();

        assert_eq!(product.functional_form, FunctionalForm::Pdf);
        assert_eq!(product.domain_type, DomainType::Discrete);
        assert!(product.verify_pdf(None).unwrap());
    }

    #[test]
    fn convolution_discrete_combines_duplicate_support_values() {
        let rv1 = two_point_pdf(
            [0, 1],
            [Rational64::new(1, 2), Rational64::new(1, 2)],
            FunctionalForm::Pdf,
        );
        let rv2 = two_point_pdf(
            [2, 3],
            [Rational64::new(1, 5), Rational64::new(4, 5)],
            FunctionalForm::Pdf,
        );

        let sum = convolution_discrete(&rv1, &rv2).unwrap();

        assert_eq!(
            sum.support,
            vec![Number::Integer(2), Number::Integer(3), Number::Integer(4)]
        );
        assert_eq!(
            sum.function,
            vec![
                Number::Rational(Rational64::new(1, 10)),
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(2, 5))
            ]
        );
        assert!(sum.verify_pdf(None).unwrap());
    }

    #[test]
    fn convolution_discrete_accepts_cdf_inputs_by_converting_to_pdf() {
        let rv1 = two_point_pdf(
            [1, 2],
            [Rational64::new(1, 4), Rational64::new(1, 1)],
            FunctionalForm::Cdf,
        );
        let rv2 = two_point_pdf(
            [3, 5],
            [Rational64::new(1, 2), Rational64::new(1, 1)],
            FunctionalForm::Cdf,
        );

        let sum = convolution_discrete(&rv1, &rv2).unwrap();

        assert_eq!(
            sum.support,
            vec![
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
                Number::Integer(7)
            ]
        );
        assert_eq!(
            sum.function,
            vec![
                Number::Rational(Rational64::new(1, 8)),
                Number::Rational(Rational64::new(3, 8)),
                Number::Rational(Rational64::new(1, 8)),
                Number::Rational(Rational64::new(3, 8))
            ]
        );
        assert!(sum.verify_pdf(None).unwrap());
    }

    #[test]
    fn convolution_discrete_preserves_pdf_and_discrete_domain() {
        let rv1 = two_point_pdf(
            [2, 4],
            [Rational64::new(1, 3), Rational64::new(2, 3)],
            FunctionalForm::Pdf,
        );
        let rv2 = two_point_pdf(
            [1, 3],
            [Rational64::new(3, 10), Rational64::new(7, 10)],
            FunctionalForm::Pdf,
        );

        let sum = convolution_discrete(&rv1, &rv2).unwrap();

        assert_eq!(sum.functional_form, FunctionalForm::Pdf);
        assert_eq!(sum.domain_type, DomainType::Discrete);
        assert!(sum.verify_pdf(None).unwrap());
    }
}
