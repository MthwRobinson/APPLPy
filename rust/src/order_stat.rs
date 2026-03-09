// Given the previous combination, finds the next lexicographical combination
pub fn next_combination(previous: &[usize], n: usize) -> Vec<usize> {
    let num_elements: usize = previous.len();

    let mut next = Vec::with_capacity(num_elements);
    next.extend_from_slice(previous);

    // If the value in the final position is not the maximum value it
    // can attain, increment it by 1
    if next[n-1] != n {
        next[n-1] += 1;
    } else {
        let mut move_left: bool = true;
        for i in 1..n {
            let index = i - 1;
            let upper_limit = n + i - n;
            if next[index] < upper_limit {
                for j in 1..upper_limit {
                    next[index + j] = next[(index + j) - 1] + 1
                }
                move_left = false;
            }
            if !move_left {
                break;
            }
        }
    }

    next
}

#[cfg(test)]
mod tests {
    use super::next_combination;

    #[test]
    fn increments_nth_value_when_below_limit() {
        let previous = [1, 1];

        let next = next_combination(&previous, 2);

        assert_eq!(next, vec![1, 2]);
    }

    #[test]
    fn keeps_vector_unchanged_when_nth_value_is_at_limit() {
        let previous = [1, 2, 3];

        let next = next_combination(&previous, 3);

        assert_eq!(next, vec![1, 2, 3]);
    }

    #[test]
    fn updates_suffix_when_left_shift_condition_is_met() {
        let previous = [1, 1, 3];

        let next = next_combination(&previous, 3);

        assert_eq!(next, vec![1, 1, 2]);
    }

    #[test]
    fn only_changes_the_target_window_for_shorter_n() {
        let previous = [1, 1, 5, 8];

        let next = next_combination(&previous, 2);

        assert_eq!(next, vec![1, 2, 5, 8]);
    }
}
