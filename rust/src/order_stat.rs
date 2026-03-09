#![allow(clippy::useless_conversion)]
#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Given the previous combination, finds the next lexicographical combination.
pub fn next_combination(previous: &[usize], upper_bound: usize) -> Vec<usize> {
    let mut next = previous.to_vec();
    let num_elements = next.len();

    // If the value in the final position is not the maximum value it can
    // attain, increment it by 1.
    if next[num_elements - 1] != upper_bound {
        next[num_elements - 1] += 1;
    } else {
        // Otherwise, move left through the combination and increment the
        // first value that can advance, then rewrite the suffix.
        let mut move_left = true;
        for i in (1..num_elements).rev() {
            let index = i - 1;
            if next[index] < upper_bound + i - num_elements {
                next[index] += 1;
                for j in 1..(num_elements - i + 1) {
                    next[index + j] = next[index + j - 1] + 1;
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

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "next_combination", signature = (previous, n))]
pub fn next_combination_py(previous: &[usize], n: usize) -> PyResult<Vec<usize>> {
    if previous.is_empty() {
        return Err(PyValueError::new_err("Previous must not be empty"));
    }
    Ok(next_combination(previous, n))
}

#[cfg(test)]
mod tests {
    use super::next_combination;

    #[test]
    fn increments_last_value_when_below_upper_bound() {
        let previous = [1, 2, 4];

        let next = next_combination(&previous, 5);

        assert_eq!(next, vec![1, 2, 5]);
    }

    #[test]
    fn updates_suffix_when_rightmost_value_reaches_upper_bound() {
        let previous = [1, 4, 5];

        let next = next_combination(&previous, 5);

        assert_eq!(next, vec![2, 3, 4]);
    }

    #[test]
    fn keeps_vector_when_no_increment_is_possible() {
        let previous = [3, 4, 5];

        let next = next_combination(&previous, 5);

        assert_eq!(next, vec![3, 4, 5]);
    }
}
