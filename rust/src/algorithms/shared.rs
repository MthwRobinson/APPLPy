#![allow(dead_code)]

use crate::algorithms::number::Number;

/// Sorts the support of a random variable, while keeping the function aligned
/// with the support
pub fn sort_by_support(
    support: Vec<Number>,
    function: Vec<Number>,
) -> Result<(Vec<Number>, Vec<Number>), String> {
    if support.len() != function.len() {
        return Err("support and function must be the same length".to_string());
    }

    let mut zipped_pairs: Vec<_> = support.into_iter().zip(function).collect();

    zipped_pairs.sort_by(|a, b| {
        let first_value = a.0.to_f64();
        let second_value = b.0.to_f64();
        first_value.total_cmp(&second_value)
    });

    let (sorted_support, sorted_function) = zipped_pairs.into_iter().unzip();
    Ok((sorted_support, sorted_function))
}
