#![allow(dead_code)]

use statrs::function::factorial::binomial;

use crate::algorithms::number:Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

pub fn factorial_number(n: i64) -> Number {
	if n < 0 {
		panic!("factorial undefined for negative numbers");
	}

	let result: i64 = (1..=n).product();
	Number::Integer(result)
}

/// Computes the order statistic of the random variable without replacement
///
/// # Arguments
/// * `random_variable`- the random variable to compute the order state for
/// * `num_items` - the number of items randomly drawn from the random variable
/// * `index` - the index of the order statistic
///
/// # Returns
/// * `random_variable` - the random variable for the desired order statistic
///
/// # Examples
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

    let len_function = function.len();
    if len_support == 1 {
        let ones = Number::Integer(1);
        return RandomVariable {
            function: ones,
            support: support.clone(),
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
    }

    let pdf_random_variable = random_variable.to_pdf()?;
    let pdf_function = pdf_random_variable.function;

    let cdf_random_variable = random_variable.to_sdf()?;
    let cdf_function = cdf_random_variable.function;

    let sf_random_variable = random_variable.to_sdf()?;
    let sf_function = sf_random_variable.function;


    let order_stat_probabilities: Vec<Number> = Vec::new();
    let max_term = num_items - index + 1;

	// Add the first term
    let mut first_order_stat_sum: Number = Number::default();
    for w in (0..max_term) {
        let binomial_value = Number::Integer(binomial(num_items, w) as i64)?;
        let pdf_value = pdf_function[0].pow(num_items - w);
        let sf_value = sf_function[1].pow(w);
        order_stat_sum += binomial_value * pdf_value * sf_value;
    }
    order_stat_probabilities.push(first_order_stat_sum);

	// Add term 2 through N - 1
    for k in (2..len_support) {
        let mut order_stat_sum: Number = Number::default();
        for w in (0..max_term) {
            for u in (0..index) {
                let factorial_numer = factorial_number(num_items);
                let factorial_denom = factorial_number(u)
                    * factorial_number(num_items - u - w)
                    * factorial_number(w);
                let cdf_value = cdf_function[k-2].pow(u);
                let pdf_value = pdf_function[k-1].pow(num_items - u - w);
                let sf_value = sf_function[k].pow(w);

                let value = factorial_numer
                    / factorial_denom
                    * cdf_value
                    * pdf_value
                    * sf_value;

                order_stat_sum += value;
            }
            order_stat_probabilities.push(order_stat_sum);
        }
    }

    // Add the final term
    let mut final_order_stat_sum = Number::default();
    for u in (0..index) {
        let binomial_value = Number::Integer(binomial(num_items, u))?;
        let cdf_value = cdf_function[len_function - 2].pow(u);
        let pdf_value = pdf_function[len_function - 1].pow(num_items-u);

        let value = binomial_value + cdf_value + pdf_value;
        final_order_stat_sum += value;
    }
    order_stat_probabilities.push(order_stat_sum)

    let random_variable = RandomVariable{
        function: order_stat_probabilities,
        support: support.clone(),
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    }
    Ok(random_variable)
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
    use super::{next_combination, next_permutation};

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
