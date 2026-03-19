#![allow(dead_code)]

use statrs::function::factorial::binomial;

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

pub fn factorial_number(n: i64) -> Number {
    if n < 0 {
        panic!("factorial undefined for negative numbers");
    }

    let result: i64 = (1..=n).product();
    Number::Integer(result)
}

/// Computes the discrete `index`-th order statistic of i.i.d. samples drawn
/// with replacement from a random variable.
///
/// # Arguments
/// * `random_variable`- the random variable to compute the order state for
/// * `num_items` - the number of items randomly drawn from the random variable
/// * `index` - the 1-based index of the order statistic
///
/// # Returns
/// * `random_variable` - the random variable for the desired order statistic
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::order_stat::discrete_order_stat_with_replacement;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.5), Number::Float(0.5)],
///     support: vec![Number::Integer(0), Number::Integer(1)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// // Minimum of two draws from {0,1} with equal mass.
/// let min_of_two = discrete_order_stat_with_replacement(&rv, 2, 1).unwrap();
/// assert!((min_of_two.function[0].to_f64() - 0.75).abs() < 1e-12);
/// assert!((min_of_two.function[1].to_f64() - 0.25).abs() < 1e-12);
/// ```
pub fn discrete_order_stat_with_replacement(
    random_variable: &RandomVariable,
    num_items: u64,
    index: u64,
) -> Result<RandomVariable, String> {
    let function = &random_variable.function;
    let support = &random_variable.support;

    if function.is_empty() {
        return Err("cannot compute the order. function is empty".to_string());
    }
    if num_items == 0 {
        return Err("cannot compute the order with zero sampled items".to_string());
    }
    if index == 0 || index > num_items {
        return Err("index must be between 1 and num_items (inclusive)".to_string());
    }

    let len_support = support.len();
    if len_support == 1 {
        return Ok(RandomVariable {
            function: vec![Number::Integer(1)],
            support: support.clone(),
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        });
    }

    let pdf_random_variable = random_variable.to_pdf()?;
    let pdf_function = pdf_random_variable.function;

    let cdf_random_variable = random_variable.to_cdf()?;
    let cdf_function = cdf_random_variable.function;

    if cdf_function.len() != len_support || pdf_function.len() != len_support {
        return Err("invalid random variable: function/support length mismatch".to_string());
    }

    let mut order_stat_cdf = Vec::with_capacity(len_support);
    for &cdf_value_at_k in &cdf_function {
        let one_minus_cdf = Number::one() - cdf_value_at_k;
        let mut cdf_sum = Number::default();
        for j in index..=num_items {
            let choose = Number::Float(binomial(num_items, j));
            let j_exp = Number::Integer(
                i64::try_from(j)
                    .map_err(|_| "num_items is too large for integer exponent".to_string())?,
            );
            let remaining = Number::Integer(
                i64::try_from(num_items - j)
                    .map_err(|_| "num_items is too large for integer exponent".to_string())?,
            );
            let cdf_term = cdf_value_at_k.pow(j_exp)?;
            let sf_term = one_minus_cdf.pow(remaining)?;
            cdf_sum += choose * cdf_term * sf_term;
        }
        order_stat_cdf.push(cdf_sum);
    }

    let mut order_stat_probabilities = Vec::with_capacity(len_support);
    let mut previous_cdf = Number::default();
    for cdf_value in order_stat_cdf {
        order_stat_probabilities.push(cdf_value - previous_cdf);
        previous_cdf = cdf_value;
    }

    let random_variable = RandomVariable {
        function: order_stat_probabilities,
        support: support.clone(),
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };

    Ok(random_variable)
}

/// Computes the discrete `index`-th order statistic of i.i.d. samples drawn
/// without replacement from a random variable.
///
/// # Arguments
/// * `random_variable`- the random variable to compute the order state for
/// * `num_items` - the number of items randomly drawn from the random variable
/// * `index` - the 1-based index of the order statistic
///
/// # Returns
/// * `random_variable` - the random variable for the desired order statistic
///
/// # Examples
pub fn discrete_order_stat_without_replacement(
    random_variable: &RandomVariable,
    num_items: u64,
    index: u64,
) -> Result<RandomVariable, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = pdf_random_variable.function;
    let support = random_variable.support.clone();
    let len_function = function.len();
    let len_function_u64 = u64::try_from(len_function)
        .map_err(|_| "support length is too large to process".to_string())?;

    if function.is_empty() {
        return Err("cannot compute the order. function is empty".to_string());
    }
    if num_items == 0 {
        return Err("cannot compute the order with zero sampled items".to_string());
    }
    if num_items > len_function_u64 {
        return Err("num_items cannot exceed support length without replacement".to_string());
    }
    if index == 0 || index > num_items {
        return Err("index must be between 1 and num_items (inclusive)".to_string());
    }

    // Initialize all of the order stat probabilities as prob(x) = 0
    let mut order_stat_probabilities: Vec<Number> =
        function.iter().map(|_| Number::default()).collect();

    let all_equal = match function.first() {
        Some(first) => function.iter().all(|value| value == first),
        None => true,
    };

    // If everything is equally likely, then it's just n choose m
    if all_equal {
        let max_term = len_function_u64 - num_items + index;
        let binomial_denom = Number::Float(binomial(len_function_u64, num_items));

        for i in index..=max_term {
            let order_stat_index = usize::try_from(i - 1)
                .map_err(|_| "order statistic index is too large to process".to_string())?;

            let binomial_numer = Number::Float(binomial(i - 1, index - 1))
                * Number::Float(binomial(1, 1))
                * Number::Float(binomial(len_function_u64 - i, num_items - index));
            let value = binomial_numer / binomial_denom;
            order_stat_probabilities[order_stat_index] = value;
        }
        let order_stat_rv = RandomVariable {
            function: order_stat_probabilities,
            support: support.clone(),
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        return Ok(order_stat_rv);
    }

    // If you are only choosing one item, the order stat simplifies to the PDF
    if num_items == 1 {
        return random_variable.to_pdf();
    }

    // If you choose all of the items without replacement, then everything gets chosen
    if num_items == len_function_u64 {
        let last_index = len_function - 1;
        order_stat_probabilities[last_index] = Number::Integer(1);

        let order_stat_rv = RandomVariable {
            function: order_stat_probabilities,
            support: support.clone(),
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        return Ok(order_stat_rv);
    }

    Err("order stat not computed".to_string())
}

/// Given the previous combination, finds the next lexicographical combination.
///
/// # Arguments
/// * `previous` - the previous combination
/// * `upper_bound` - the maximum allowed value in the combination
///
/// # Returns
/// * `next` - the next combination
///
/// # Examples
///
/// ```
/// use applpy_rust::algorithms::order_stat::next_combination;
///
/// let c = vec![0, 1, 2];
/// assert_eq!(next_combination(&c, 4), Some(vec![0, 1, 3]));
///
/// let c = vec![0, 1, 4];
/// assert_eq!(next_combination(&c, 4), Some(vec![0, 2, 3]));
///
/// let c = vec![2, 3, 4];
/// assert_eq!(next_combination(&c, 4), None);
/// ```
pub fn next_combination(previous: &[usize], upper_bound: usize) -> Option<Vec<usize>> {
    let vector_length = previous.len();

    if vector_length == 0 || vector_length > upper_bound + 1 {
        return None;
    }

    if previous.iter().any(|&v| v > upper_bound) || previous.windows(2).any(|w| w[0] >= w[1]) {
        return None;
    }

    let mut next = previous.to_vec();

    for i in (0..vector_length).rev() {
        if next[i] < upper_bound + i + 1 - vector_length {
            next[i] += 1;

            let mut val = next[i];
            for x in &mut next[i + 1..] {
                val += 1;
                *x = val;
            }

            return Some(next);
        }
    }

    None
}

/// Given the previous permutation, finds the next lexicographical permutation.
///
/// # Arguments
/// * `previous` - the previous permutation
///
/// # Returns
/// * `next` - the next combination
pub fn next_permutation(previous: &[usize]) -> Option<Vec<usize>> {
    let vector_length = previous.len();

    if vector_length == 0 {
        return None;
    }

    let mut next = previous.to_vec();

    for i in (1..vector_length).rev() {
        let index = i - 1;
        if next[index] < next[index + 1] {
            let original_value = next[index];
            let mut swap_index = index + 1;

            for j in (swap_index..vector_length).rev() {
                if next[j] > original_value {
                    swap_index = j;
                    break;
                }
            }

            next.swap(index, swap_index);
            next[index + 1..].reverse();

            return Some(next);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{discrete_order_stat_with_replacement, next_combination, next_permutation};
    use crate::algorithms::number::Number;
    use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

    fn assert_close(actual: Number, expected: f64, tolerance: f64) {
        assert!((actual.to_f64() - expected).abs() < tolerance);
    }

    #[test]
    fn order_stat_with_replacement_returns_expected_min_distribution() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_order_stat_with_replacement(&rv, 2, 1).unwrap();

        assert_close(result.function[0], 0.75, 1e-12);
        assert_close(result.function[1], 0.25, 1e-12);
        assert_eq!(result.support, rv.support);
        assert_eq!(result.functional_form, FunctionalForm::Pdf);
        assert_eq!(result.domain_type, DomainType::Discrete);
    }

    #[test]
    fn order_stat_with_replacement_returns_expected_max_distribution() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_order_stat_with_replacement(&rv, 2, 2).unwrap();

        assert_close(result.function[0], 0.25, 1e-12);
        assert_close(result.function[1], 0.75, 1e-12);
    }

    #[test]
    fn order_stat_with_replacement_returns_degenerate_distribution_for_single_support_value() {
        let rv = RandomVariable {
            function: vec![Number::Integer(1)],
            support: vec![Number::Integer(7)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_order_stat_with_replacement(&rv, 3, 2).unwrap();

        assert_eq!(result.function, vec![Number::Integer(1)]);
        assert_eq!(result.support, vec![Number::Integer(7)]);
    }

    #[test]
    fn order_stat_with_replacement_rejects_invalid_index_and_sample_count() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        assert_eq!(
            discrete_order_stat_with_replacement(&rv, 0, 1).unwrap_err(),
            "cannot compute the order with zero sampled items"
        );
        assert_eq!(
            discrete_order_stat_with_replacement(&rv, 2, 0).unwrap_err(),
            "index must be between 1 and num_items (inclusive)"
        );
        assert_eq!(
            discrete_order_stat_with_replacement(&rv, 2, 3).unwrap_err(),
            "index must be between 1 and num_items (inclusive)"
        );
    }

    #[test]
    fn increments_last_value_when_below_upper_bound() {
        let previous = [1, 2, 4];

        let next = next_combination(&previous, 5);

        assert_eq!(next, vec![1, 2, 5].into());
    }

    #[test]
    fn updates_suffix_when_rightmost_value_reaches_upper_bound() {
        let previous = [1, 4, 5];

        let next = next_combination(&previous, 5);

        assert_eq!(next, vec![2, 3, 4].into());
    }

    #[test]
    fn keeps_vector_when_no_increment_is_possible() {
        let previous = [3, 4, 5];

        let next = next_combination(&previous, 5);

        assert_eq!(next, None);
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(next_combination(&[], 5), None);
    }

    #[test]
    fn test_k_greater_than_domain() {
        // k > upper_bound + 1
        assert_eq!(next_combination(&[0, 1, 2, 3], 2), None);
    }

    #[test]
    fn test_value_exceeds_upper_bound() {
        assert_eq!(next_combination(&[0, 1, 5], 4), None);
    }

    #[test]
    fn test_not_strictly_increasing() {
        assert_eq!(next_combination(&[0, 2, 2], 5), None);
    }

    #[test]
    fn test_decreasing_input() {
        assert_eq!(next_combination(&[3, 2, 1], 5), None);
    }

    #[test]
    fn test_already_last_combination() {
        assert_eq!(next_combination(&[2, 3, 4], 4), None);
    }

    #[test]
    fn test_increment_last_element() {
        assert_eq!(next_combination(&[0, 1, 2], 4), Some(vec![0, 1, 3]));
    }

    #[test]
    fn test_carry_propagation() {
        assert_eq!(next_combination(&[0, 1, 4], 4), Some(vec![0, 2, 3]));
    }

    #[test]
    fn test_full_sequence_progression() {
        let mut c = vec![0, 1, 2];
        let mut results = Vec::new();

        while let Some(next) = next_combination(&c, 4) {
            results.push(next.clone());
            c = next;
        }

        assert_eq!(
            results,
            vec![
                vec![0, 1, 3],
                vec![0, 1, 4],
                vec![0, 2, 3],
                vec![0, 2, 4],
                vec![0, 3, 4],
                vec![1, 2, 3],
                vec![1, 2, 4],
                vec![1, 3, 4],
                vec![2, 3, 4],
            ]
        );
    }

    #[test]
    fn test_single_element_combination() {
        assert_eq!(next_combination(&[2], 4), Some(vec![3]));
        assert_eq!(next_combination(&[4], 4), None);
    }

    #[test]
    fn test_next_permutation_increments_last_two_values() {
        assert_eq!(next_permutation(&[1, 2, 3]), Some(vec![1, 3, 2]));
    }

    #[test]
    fn test_next_permutation_updates_suffix_after_swap() {
        assert_eq!(next_permutation(&[1, 3, 2]), Some(vec![2, 1, 3]));
    }

    #[test]
    fn test_next_permutation_returns_none_for_last_ordering() {
        assert_eq!(next_permutation(&[3, 2, 1]), None);
    }

    #[test]
    fn test_next_permutation_empty_input() {
        assert_eq!(next_permutation(&[]), None);
    }

    #[test]
    fn test_next_permutation_single_element() {
        assert_eq!(next_permutation(&[7]), None);
    }

    #[test]
    fn test_next_permutation_full_sequence_progression() {
        let mut p = vec![0, 1, 2];
        let mut results = Vec::new();

        while let Some(next) = next_permutation(&p) {
            results.push(next.clone());
            p = next;
        }

        assert_eq!(
            results,
            vec![
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0],
            ]
        );
    }
}
