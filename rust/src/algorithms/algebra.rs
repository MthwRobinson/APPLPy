#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

/// Computes the product of two discrete random variables
///
/// # Arguments
/// * `random_variable_1` - the first random variable
/// * `random_variable_2` - the second random variable
///
/// # Returns
/// * `product_rv` - the product of the two random variables
///
/// # Examples
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
        let support_index = product_support.iter().position(|&x| x == s);

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
