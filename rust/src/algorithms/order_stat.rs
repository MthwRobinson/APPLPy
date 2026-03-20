#![allow(dead_code)]

use num_rational::Rational64;

use crate::algorithms::number::Number;
use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};

#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatVariant {
    WithReplacement,
    WithoutReplacement,
}

/// Computes the distribution of the maximum of two random variables.
///
/// This currently supports discrete/discrete-functional random variables.
pub fn discrete_maximum(
    random_variable_1: &RandomVariable,
    random_variable_2: &RandomVariable,
) -> Result<RandomVariable, String> {
    validate_binary_extrema_inputs(random_variable_1, random_variable_2)?;

    let pdf_1 = random_variable_1.to_pdf()?;
    let pdf_2 = random_variable_2.to_pdf()?;
    let support_1 = pdf_1.support;
    let support_2 = pdf_2.support;
    let function_1 = pdf_1.function;
    let function_2 = pdf_2.function;

    let mut candidates: Vec<(Number, Number)> =
        Vec::with_capacity(support_1.len() * support_2.len());

    for (i, &support_1_value) in support_1.iter().enumerate() {
        for (j, &support_2_value) in support_2.iter().enumerate() {
            let max_value = max_number(support_1_value, support_2_value);
            let probability = function_1[i] * function_2[j];
            candidates.push((max_value, probability));
        }
    }

    let (support, function) = normalize_support_probability_candidates(candidates);

    Ok(RandomVariable {
        function,
        support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    })
}

/// Computes the distribution of the minimum of two random variables.
///
/// This currently supports discrete/discrete-functional random variables.
pub fn discrete_minimum(
    random_variable_1: &RandomVariable,
    random_variable_2: &RandomVariable,
) -> Result<RandomVariable, String> {
    validate_binary_extrema_inputs(random_variable_1, random_variable_2)?;

    let pdf_1 = random_variable_1.to_pdf()?;
    let pdf_2 = random_variable_2.to_pdf()?;
    let support_1 = pdf_1.support;
    let support_2 = pdf_2.support;
    let function_1 = pdf_1.function;
    let function_2 = pdf_2.function;

    let mut candidates: Vec<(Number, Number)> =
        Vec::with_capacity(support_1.len() * support_2.len());

    for (i, &support_1_value) in support_1.iter().enumerate() {
        for (j, &support_2_value) in support_2.iter().enumerate() {
            let min_value = min_number(support_1_value, support_2_value);
            let probability = function_1[i] * function_2[j];
            candidates.push((min_value, probability));
        }
    }

    let (support, function) = normalize_support_probability_candidates(candidates);

    Ok(RandomVariable {
        function,
        support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    })
}

/// Computes the discrete `index`-th order statistic of i.i.d. samples drawn
///
/// # Arguments
/// * `random_variable`- the random variable to compute the order state for
/// * `num_items` - the number of items randomly drawn from the random variable
/// * `index` - the 1-based index of the order statistic
/// * `variant` - one of the OrderStatVariant enum types
///
/// # Returns
/// * `random_variable` - the random variable for the desired order statistic
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::order_stat::{discrete_order_stat, OrderStatVariant};
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// let rv = RandomVariable {
///     function: vec![
///         Number::Float(0.25),
///         Number::Float(0.25),
///         Number::Float(0.25),
///         Number::Float(0.25),
///     ],
///     support: vec![
///         Number::Integer(1),
///         Number::Integer(2),
///         Number::Integer(3),
///         Number::Integer(4),
///     ],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// // Minimum of two draws with replacement.
/// let min_with_replacement = discrete_order_stat(&rv, 2, 1, OrderStatVariant::WithReplacement).unwrap();
/// assert!((min_with_replacement.function[0].to_f64() - 0.4375).abs() < 1e-12);
///
/// // Minimum of two draws without replacement.
/// let min_without_replacement =
///     discrete_order_stat(&rv, 2, 1, OrderStatVariant::WithoutReplacement).unwrap();
/// assert!((min_without_replacement.function[0].to_f64() - 0.5).abs() < 1e-12);
/// ```
pub fn discrete_order_stat(
    random_variable: &RandomVariable,
    num_items: u64,
    index: u64,
    variant: OrderStatVariant,
) -> Result<RandomVariable, String> {
    match variant {
        OrderStatVariant::WithReplacement => {
            discrete_order_stat_with_replacement(random_variable, num_items, index)
        }
        OrderStatVariant::WithoutReplacement => {
            discrete_order_stat_without_replacement(random_variable, num_items, index)
        }
    }
}

/// Computes the discrete range statistic (`max - min`) of `num_items` samples.
///
/// # Arguments
/// * `random_variable` - the random variable to compute the range statistic for
/// * `num_items` - the number of items randomly drawn from the random variable
/// * `variant` - one of the OrderStatVariant enum types
///
/// # Returns
/// * `random_variable` - the random variable for the desired range statistic
pub fn discrete_range_stat(
    random_variable: &RandomVariable,
    num_items: u64,
    variant: OrderStatVariant,
) -> Result<RandomVariable, String> {
    match variant {
        OrderStatVariant::WithReplacement => {
            discrete_range_stat_with_replacement(random_variable, num_items)
        }
        OrderStatVariant::WithoutReplacement => {
            discrete_range_stat_without_replacement(random_variable, num_items)
        }
    }
}

/// Computes the discrete range statistic (`max - min`) of `num_items` samples
/// drawn with replacement.
pub fn discrete_range_stat_with_replacement(
    random_variable: &RandomVariable,
    num_items: u64,
) -> Result<RandomVariable, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = pdf_random_variable.function;
    let support = random_variable.support.clone();

    validate_range_inputs(&function, &support, num_items, false)?;

    let cdf_random_variable = random_variable.to_cdf()?;
    let cdf_function = cdf_random_variable.function;
    let len_support = support.len();
    let exponent = Number::Integer(
        i64::try_from(num_items)
            .map_err(|_| "num_items is too large for integer exponent".to_string())?,
    );

    let mut range_candidates: Vec<(Number, Number)> = Vec::with_capacity(len_support * len_support);

    for i in 0..len_support {
        for j in i..len_support {
            let full_mass = cumulative_between(&cdf_function, i, j);
            let no_min_mass = cumulative_between(&cdf_function, i + 1, j);
            let no_max_mass = if j == 0 {
                Number::default()
            } else {
                cumulative_between(&cdf_function, i, j - 1)
            };
            let no_min_no_max_mass = if j == 0 {
                Number::default()
            } else {
                cumulative_between(&cdf_function, i + 1, j - 1)
            };

            let probability = full_mass.pow(exponent)?
                - no_min_mass.pow(exponent)?
                - no_max_mass.pow(exponent)?
                + no_min_no_max_mass.pow(exponent)?;
            let range_value = support[j] - support[i];

            range_candidates.push((range_value, probability));
        }
    }

    let (range_support, range_probabilities) = normalize_range_candidates(range_candidates);

    Ok(RandomVariable {
        function: range_probabilities,
        support: range_support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    })
}

/// Computes the discrete range statistic (`max - min`) of `num_items` samples
/// drawn without replacement.
pub fn discrete_range_stat_without_replacement(
    random_variable: &RandomVariable,
    num_items: u64,
) -> Result<RandomVariable, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = pdf_random_variable.function;
    let support = random_variable.support.clone();
    let len_function = function.len();
    validate_range_inputs(&function, &support, num_items, true)?;

    let n =
        usize::try_from(num_items).map_err(|_| "num_items is too large to process".to_string())?;
    let mut range_candidates: Vec<(Number, Number)> = Vec::new();
    let mut combo: Vec<usize> = (0..n).collect();

    loop {
        let mut perm = combo.clone();

        loop {
            let mut perm_prob = function[perm[0]];
            let mut cumsum = function[perm[0]];

            for &perm_index in perm.iter().skip(1) {
                let remaining_mass = Number::one() - cumsum;
                if remaining_mass.to_f64() <= f64::EPSILON {
                    perm_prob = Number::default();
                    break;
                }
                perm_prob =
                    divide_preserving_precision(perm_prob * function[perm_index], remaining_mass)?;
                cumsum += function[perm_index];
            }

            let min_index = perm.iter().min().copied().unwrap_or(0);
            let max_index = perm.iter().max().copied().unwrap_or(0);
            let range_value = support[max_index] - support[min_index];
            range_candidates.push((range_value, perm_prob));

            match next_permutation(&perm) {
                Some(next_perm) => perm = next_perm,
                None => break,
            }
        }

        match next_combination(&combo, len_function - 1) {
            Some(next_combo) => combo = next_combo,
            None => break,
        }
    }

    let (range_support, range_probabilities) = normalize_range_candidates(range_candidates);

    Ok(RandomVariable {
        function: range_probabilities,
        support: range_support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    })
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
            let choose = binomial_number(num_items, j)?;
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
/// ```
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::order_stat::discrete_order_stat_without_replacement;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.25), Number::Float(0.25), Number::Float(0.25), Number::Float(0.25)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3), Number::Integer(4)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// // Minimum of two draws without replacement from {1,2,3,4}.
/// let min_of_two = discrete_order_stat_without_replacement(&rv, 2, 1).unwrap();
/// assert!((min_of_two.function[0].to_f64() - 0.5).abs() < 1e-12);
/// assert!((min_of_two.function[1].to_f64() - (1.0 / 3.0)).abs() < 1e-12);
/// assert!((min_of_two.function[2].to_f64() - (1.0 / 6.0)).abs() < 1e-12);
/// assert!(min_of_two.function[3].to_f64().abs() < 1e-12);
/// ```
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
    let n =
        usize::try_from(num_items).map_err(|_| "num_items is too large to process".to_string())?;
    let order_stat_position =
        usize::try_from(index - 1).map_err(|_| "index is too large to process".to_string())?;

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
        let binomial_denom = binomial_number(len_function_u64, num_items)?;

        for i in index..=max_term {
            let order_stat_index = usize::try_from(i - 1)
                .map_err(|_| "order statistic index is too large to process".to_string())?;

            let binomial_numer = binomial_number(i - 1, index - 1)?
                * binomial_number(len_function_u64 - i, num_items - index)?;
            let value = divide_preserving_precision(binomial_numer, binomial_denom)?;
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
        order_stat_probabilities[order_stat_position] = Number::Integer(1);

        let order_stat_rv = RandomVariable {
            function: order_stat_probabilities,
            support: support.clone(),
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        return Ok(order_stat_rv);
    }

    let mut prob_storage = vec![vec![Number::default(); len_function]; n];
    let mut combo: Vec<usize> = (0..n).collect();

    loop {
        let mut perm = combo.clone();

        loop {
            let mut perm_prob = function[perm[0]];
            let mut cumsum = function[perm[0]];

            for &perm_index in perm.iter().skip(1) {
                let remaining_mass = Number::one() - cumsum;
                if remaining_mass.to_f64() <= f64::EPSILON {
                    perm_prob = Number::default();
                    break;
                }
                perm_prob =
                    divide_preserving_precision(perm_prob * function[perm_index], remaining_mass)?;
                cumsum += function[perm_index];
            }

            let mut ordered_perm = perm.clone();
            ordered_perm.sort_unstable();

            for (order_position, &value_index) in ordered_perm.iter().enumerate() {
                prob_storage[order_position][value_index] += perm_prob;
            }

            match next_permutation(&perm) {
                Some(next_perm) => perm = next_perm,
                None => break,
            }
        }

        match next_combination(&combo, len_function - 1) {
            Some(next_combo) => combo = next_combo,
            None => break,
        }
    }

    let order_stat_rv = RandomVariable {
        function: prob_storage[order_stat_position].clone(),
        support: support.clone(),
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };

    Ok(order_stat_rv)
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

pub fn factorial_number(n: i64) -> Number {
    if n < 0 {
        panic!("factorial undefined for negative numbers");
    }

    let result: i64 = (1..=n).product();
    Number::Integer(result)
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn binomial_i64(n: u64, k: u64) -> Result<i64, String> {
    if k > n {
        return Ok(0);
    }
    if k == 0 || k == n {
        return Ok(1);
    }

    let k = k.min(n - k);
    let mut result: u128 = 1;

    for i in 1..=k {
        let mut numer = u128::from(n - k + i);
        let mut denom = u128::from(i);

        let g1 = gcd_u128(numer, denom);
        numer /= g1;
        denom /= g1;

        let g2 = gcd_u128(result, denom);
        result /= g2;
        denom /= g2;

        result = result
            .checked_mul(numer)
            .ok_or_else(|| "binomial coefficient overflowed u128".to_string())?;
        if denom != 1 {
            if !result.is_multiple_of(denom) {
                return Err("binomial coefficient reduction failed".to_string());
            }
            result /= denom;
        }
    }

    i64::try_from(result).map_err(|_| "binomial coefficient exceeds i64 range".to_string())
}

fn binomial_number(n: u64, k: u64) -> Result<Number, String> {
    Ok(Number::Integer(binomial_i64(n, k)?))
}

fn validate_range_inputs(
    function: &[Number],
    support: &[Number],
    num_items: u64,
    without_replacement: bool,
) -> Result<(), String> {
    if function.is_empty() {
        return Err("cannot compute the range. function is empty".to_string());
    }
    if num_items < 2 {
        return Err("cannot compute the range with fewer than two sampled items".to_string());
    }
    if support.len() < 2 {
        return Err("the population must contain at least two support values".to_string());
    }
    if without_replacement {
        let support_len_u64 = u64::try_from(support.len())
            .map_err(|_| "support length is too large to process".to_string())?;
        if num_items > support_len_u64 {
            return Err("num_items cannot exceed support length without replacement".to_string());
        }
    }
    Ok(())
}

fn validate_binary_extrema_inputs(
    random_variable_1: &RandomVariable,
    random_variable_2: &RandomVariable,
) -> Result<(), String> {
    if random_variable_1.domain_type != random_variable_2.domain_type {
        return Err("The RVs must both be discrete or continuous".to_string());
    }
    if random_variable_1.domain_type == DomainType::Continuous {
        return Err("continuous maximum/minimum random variables are not implemented".to_string());
    }
    Ok(())
}

fn max_number(lhs: Number, rhs: Number) -> Number {
    match lhs.partial_cmp(&rhs) {
        Some(std::cmp::Ordering::Less) => rhs,
        _ => lhs,
    }
}

fn min_number(lhs: Number, rhs: Number) -> Number {
    match lhs.partial_cmp(&rhs) {
        Some(std::cmp::Ordering::Greater) => rhs,
        _ => lhs,
    }
}

fn cumulative_between(cdf_function: &[Number], start: usize, end: usize) -> Number {
    if start > end || start >= cdf_function.len() || end >= cdf_function.len() {
        return Number::default();
    }

    if start == 0 {
        cdf_function[end]
    } else {
        cdf_function[end] - cdf_function[start - 1]
    }
}

fn normalize_range_candidates(
    range_candidates: Vec<(Number, Number)>,
) -> (Vec<Number>, Vec<Number>) {
    normalize_support_probability_candidates(range_candidates)
}

fn normalize_support_probability_candidates(
    support_probability_candidates: Vec<(Number, Number)>,
) -> (Vec<Number>, Vec<Number>) {
    let mut sorted_candidates = support_probability_candidates;
    sorted_candidates.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut merged_support: Vec<Number> = Vec::new();
    let mut merged_probabilities: Vec<Number> = Vec::new();

    for (support_value, probability) in sorted_candidates {
        if probability.to_f64() <= f64::EPSILON {
            continue;
        }

        match merged_support.last() {
            Some(last_support) if *last_support == support_value => {
                if let Some(last_probability) = merged_probabilities.last_mut() {
                    *last_probability += probability;
                }
            }
            _ => {
                merged_support.push(support_value);
                merged_probabilities.push(probability);
            }
        }
    }

    (merged_support, merged_probabilities)
}

fn number_from_rational(value: Rational64) -> Number {
    if *value.denom() == 1 {
        Number::Integer(*value.numer())
    } else {
        Number::Rational(value)
    }
}

fn divide_preserving_precision(lhs: Number, rhs: Number) -> Result<Number, String> {
    match (lhs, rhs) {
        (Number::Float(a), b) => Ok(Number::Float(a / b.to_f64())),
        (a, Number::Float(b)) => Ok(Number::Float(a.to_f64() / b)),
        (Number::Integer(a), Number::Integer(b)) => {
            if b == 0 {
                return Err("division by zero".to_string());
            }
            Ok(number_from_rational(Rational64::new(a, b)))
        }
        (Number::Integer(a), Number::Rational(b)) => {
            if *b.numer() == 0 {
                return Err("division by zero".to_string());
            }
            Ok(number_from_rational(Rational64::from_integer(a) / b))
        }
        (Number::Rational(a), Number::Integer(b)) => {
            if b == 0 {
                return Err("division by zero".to_string());
            }
            Ok(number_from_rational(a / Rational64::from_integer(b)))
        }
        (Number::Rational(a), Number::Rational(b)) => {
            if *b.numer() == 0 {
                return Err("division by zero".to_string());
            }
            Ok(number_from_rational(a / b))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        discrete_maximum, discrete_minimum, discrete_order_stat,
        discrete_order_stat_with_replacement, discrete_order_stat_without_replacement,
        discrete_range_stat, discrete_range_stat_with_replacement,
        discrete_range_stat_without_replacement, next_combination, next_permutation,
        OrderStatVariant,
    };
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
    fn order_stat_wrapper_dispatches_to_with_replacement() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let from_wrapper =
            discrete_order_stat(&rv, 2, 1, OrderStatVariant::WithReplacement).unwrap();
        let direct = discrete_order_stat_with_replacement(&rv, 2, 1).unwrap();

        assert_eq!(from_wrapper.function, direct.function);
        assert_eq!(from_wrapper.support, direct.support);
        assert_eq!(from_wrapper.functional_form, direct.functional_form);
        assert_eq!(from_wrapper.domain_type, direct.domain_type);
    }

    #[test]
    fn order_stat_wrapper_dispatches_to_without_replacement() {
        let rv = RandomVariable {
            function: vec![
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
            ],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let from_wrapper =
            discrete_order_stat(&rv, 2, 1, OrderStatVariant::WithoutReplacement).unwrap();
        let direct = discrete_order_stat_without_replacement(&rv, 2, 1).unwrap();

        assert_eq!(from_wrapper.function, direct.function);
        assert_eq!(from_wrapper.support, direct.support);
        assert_eq!(from_wrapper.functional_form, direct.functional_form);
        assert_eq!(from_wrapper.domain_type, direct.domain_type);
    }

    #[test]
    fn order_stat_wrapper_propagates_validation_errors() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.4), Number::Float(0.6)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        assert_eq!(
            discrete_order_stat(&rv, 2, 0, OrderStatVariant::WithReplacement).unwrap_err(),
            "index must be between 1 and num_items (inclusive)"
        );
        assert_eq!(
            discrete_order_stat(&rv, 3, 1, OrderStatVariant::WithoutReplacement).unwrap_err(),
            "num_items cannot exceed support length without replacement"
        );
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
    fn order_stat_without_replacement_matches_known_min_max_probabilities() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.3), Number::Float(0.5)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let min_of_two = discrete_order_stat_without_replacement(&rv, 2, 1).unwrap();
        assert_close(min_of_two.function[0], 0.485_714_285_714_285_7, 1e-12);
        assert_close(min_of_two.function[1], 0.514_285_714_285_714_2, 1e-12);
        assert_close(min_of_two.function[2], 0.0, 1e-12);

        let max_of_two = discrete_order_stat_without_replacement(&rv, 2, 2).unwrap();
        assert_close(max_of_two.function[0], 0.0, 1e-12);
        assert_close(max_of_two.function[1], 0.160_714_285_714_285_7, 1e-12);
        assert_close(max_of_two.function[2], 0.839_285_714_285_714_3, 1e-12);
    }

    #[test]
    fn order_stat_without_replacement_is_deterministic_when_sampling_all_items() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.3), Number::Float(0.5)],
            support: vec![
                Number::Integer(10),
                Number::Integer(20),
                Number::Integer(30),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let second_of_three = discrete_order_stat_without_replacement(&rv, 3, 2).unwrap();
        assert_close(second_of_three.function[0], 0.0, 1e-12);
        assert_close(second_of_three.function[1], 1.0, 1e-12);
        assert_close(second_of_three.function[2], 0.0, 1e-12);
    }

    #[test]
    fn order_stat_without_replacement_rejects_invalid_inputs() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.4), Number::Float(0.6)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        assert_eq!(
            discrete_order_stat_without_replacement(&rv, 0, 1).unwrap_err(),
            "cannot compute the order with zero sampled items"
        );
        assert_eq!(
            discrete_order_stat_without_replacement(&rv, 3, 1).unwrap_err(),
            "num_items cannot exceed support length without replacement"
        );
        assert_eq!(
            discrete_order_stat_without_replacement(&rv, 2, 0).unwrap_err(),
            "index must be between 1 and num_items (inclusive)"
        );
        assert_eq!(
            discrete_order_stat_without_replacement(&rv, 2, 3).unwrap_err(),
            "index must be between 1 and num_items (inclusive)"
        );
    }

    #[test]
    fn order_stat_without_replacement_matches_uniform_fast_path() {
        let rv = RandomVariable {
            function: vec![
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
            ],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let min_of_two = discrete_order_stat_without_replacement(&rv, 2, 1).unwrap();
        assert_close(min_of_two.function[0], 0.5, 1e-12);
        assert_close(min_of_two.function[1], 1.0 / 3.0, 1e-12);
        assert_close(min_of_two.function[2], 1.0 / 6.0, 1e-12);
        assert_close(min_of_two.function[3], 0.0, 1e-12);
    }

    #[test]
    fn range_stat_wrapper_dispatches_to_with_replacement() {
        let rv = RandomVariable {
            function: vec![
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
            ],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let from_wrapper = discrete_range_stat(&rv, 2, OrderStatVariant::WithReplacement).unwrap();
        let direct = discrete_range_stat_with_replacement(&rv, 2).unwrap();

        assert_eq!(from_wrapper.function, direct.function);
        assert_eq!(from_wrapper.support, direct.support);
        assert_eq!(from_wrapper.functional_form, direct.functional_form);
        assert_eq!(from_wrapper.domain_type, direct.domain_type);
    }

    #[test]
    fn range_stat_wrapper_dispatches_to_without_replacement() {
        let rv = RandomVariable {
            function: vec![
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
            ],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let from_wrapper =
            discrete_range_stat(&rv, 2, OrderStatVariant::WithoutReplacement).unwrap();
        let direct = discrete_range_stat_without_replacement(&rv, 2).unwrap();

        assert_eq!(from_wrapper.function, direct.function);
        assert_eq!(from_wrapper.support, direct.support);
        assert_eq!(from_wrapper.functional_form, direct.functional_form);
        assert_eq!(from_wrapper.domain_type, direct.domain_type);
    }

    #[test]
    fn range_stat_with_replacement_matches_known_distribution() {
        let rv = RandomVariable {
            function: vec![
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
            ],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_range_stat_with_replacement(&rv, 2).unwrap();

        assert_eq!(
            result.support,
            vec![
                Number::Integer(0),
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3)
            ]
        );
        assert_close(result.function[0], 0.25, 1e-12);
        assert_close(result.function[1], 0.375, 1e-12);
        assert_close(result.function[2], 0.25, 1e-12);
        assert_close(result.function[3], 0.125, 1e-12);
    }

    #[test]
    fn range_stat_without_replacement_matches_known_distribution() {
        let rv = RandomVariable {
            function: vec![
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
                Number::Float(0.25),
            ],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_range_stat_without_replacement(&rv, 2).unwrap();

        assert_eq!(
            result.support,
            vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)]
        );
        assert_close(result.function[0], 0.5, 1e-12);
        assert_close(result.function[1], 1.0 / 3.0, 1e-12);
        assert_close(result.function[2], 1.0 / 6.0, 1e-12);
    }

    #[test]
    fn range_stat_rejects_invalid_inputs() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.4), Number::Float(0.6)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        assert_eq!(
            discrete_range_stat(&rv, 1, OrderStatVariant::WithReplacement).unwrap_err(),
            "cannot compute the range with fewer than two sampled items"
        );
        assert_eq!(
            discrete_range_stat_without_replacement(&rv, 3).unwrap_err(),
            "num_items cannot exceed support length without replacement"
        );
    }

    #[test]
    fn discrete_maximum_matches_known_distribution() {
        let rv_1 = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rv_2 = RandomVariable {
            function: vec![Number::Float(0.25), Number::Float(0.75)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_maximum(&rv_1, &rv_2).unwrap();

        assert_eq!(result.support, vec![Number::Integer(0), Number::Integer(1)]);
        assert_close(result.function[0], 0.125, 1e-12);
        assert_close(result.function[1], 0.875, 1e-12);
        assert_eq!(result.functional_form, FunctionalForm::Pdf);
        assert_eq!(result.domain_type, DomainType::Discrete);
    }

    #[test]
    fn discrete_minimum_matches_known_distribution() {
        let rv_1 = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rv_2 = RandomVariable {
            function: vec![Number::Float(0.25), Number::Float(0.75)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_minimum(&rv_1, &rv_2).unwrap();

        assert_eq!(result.support, vec![Number::Integer(0), Number::Integer(1)]);
        assert_close(result.function[0], 0.625, 1e-12);
        assert_close(result.function[1], 0.375, 1e-12);
        assert_eq!(result.functional_form, FunctionalForm::Pdf);
        assert_eq!(result.domain_type, DomainType::Discrete);
    }

    #[test]
    fn discrete_maximum_and_discrete_minimum_merge_and_sort_support_values() {
        let rv_1 = RandomVariable {
            function: vec![Number::Float(0.4), Number::Float(0.6)],
            support: vec![Number::Integer(1), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rv_2 = RandomVariable {
            function: vec![Number::Float(0.7), Number::Float(0.3)],
            support: vec![Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let max_result = discrete_maximum(&rv_1, &rv_2).unwrap();
        assert_eq!(
            max_result.support,
            vec![Number::Integer(2), Number::Integer(3), Number::Integer(4)]
        );
        assert_close(max_result.function[0], 0.28, 1e-12);
        assert_close(max_result.function[1], 0.42, 1e-12);
        assert_close(max_result.function[2], 0.30, 1e-12);

        let min_result = discrete_minimum(&rv_1, &rv_2).unwrap();
        assert_eq!(
            min_result.support,
            vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)]
        );
        assert_close(min_result.function[0], 0.40, 1e-12);
        assert_close(min_result.function[1], 0.42, 1e-12);
        assert_close(min_result.function[2], 0.18, 1e-12);
    }

    #[test]
    fn discrete_maximum_and_discrete_minimum_reject_domain_mismatch() {
        let discrete_rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let continuous_rv = RandomVariable {
            function: vec![Number::Float(1.0)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert_eq!(
            discrete_maximum(&discrete_rv, &continuous_rv).unwrap_err(),
            "The RVs must both be discrete or continuous"
        );
        assert_eq!(
            discrete_minimum(&discrete_rv, &continuous_rv).unwrap_err(),
            "The RVs must both be discrete or continuous"
        );
    }

    #[test]
    fn discrete_maximum_and_discrete_minimum_reject_continuous_inputs() {
        let rv_1 = RandomVariable {
            function: vec![Number::Float(1.0)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };
        let rv_2 = RandomVariable {
            function: vec![Number::Float(1.0)],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert_eq!(
            discrete_maximum(&rv_1, &rv_2).unwrap_err(),
            "continuous maximum/minimum random variables are not implemented"
        );
        assert_eq!(
            discrete_minimum(&rv_1, &rv_2).unwrap_err(),
            "continuous maximum/minimum random variables are not implemented"
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
