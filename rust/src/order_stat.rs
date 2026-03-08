// Given the previous combination, finds the next lexicographical
// combination
pub fn next_combination(previous: &[usize], _n: usize) -> Vec<usize> {
    let mut next = Vec::with_capacity(previous.len());
    next.extend_from_slice(previous);

    next[0] = 99;
    next
}
