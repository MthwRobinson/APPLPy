// Given the previous combination, finds the next lexicographical combination.
pub fn next_combination(previous: &[usize], upper_bound: usize) -> Option<Vec<usize>> {
    let k = previous.len();

    if k == 0 || k > upper_bound + 1 {
        return None;
    }

    if previous.iter().any(|&v| v > upper_bound) || previous.windows(2).any(|w| w[0] >= w[1]) {
        return None;
    }

    let mut next = previous.to_vec();

    for i in (0..k).rev() {
        if next[i] < upper_bound + i + 1 - k {
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

#[cfg(test)]
mod tests {
    use super::next_combination;

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
}
