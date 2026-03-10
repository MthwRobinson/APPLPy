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
#[allow(dead_code)]
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
