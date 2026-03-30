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

    if support.is_empty() {
        return Err("support and function cannot be empty".to_string());
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

/// De-duplicates the support by combining probabilities for values that already
/// appear in the support
pub fn deduplicate_support(
    support: Vec<Number>,
    function: Vec<Number>,
) -> Result<(Vec<Number>, Vec<Number>), String> {
    if support.len() != function.len() {
        return Err("support and function must be the same length".to_string());
    }

    if support.is_empty() {
        return Err("support and function cannot be empty".to_string());
    }

    let mut deduped_support = Vec::new();
    let mut deduped_function = Vec::new();

    for (&s, &probability) in support.iter().zip(function.iter()) {
        let support_index = deduped_support
            .iter()
            .position(|&x: &Number| x.to_f64() == s.to_f64());

        match support_index {
            Some(index) => {
                deduped_function[index] += probability;
            }
            None => {
                deduped_support.push(s);
                deduped_function.push(probability);
            }
        }
    }

    Ok((deduped_support, deduped_function))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    #[test]
    fn sort_by_support_rejects_empty_inputs() {
        let result = sort_by_support(vec![], vec![]);

        assert!(matches!(
            result,
            Err(msg) if msg == "support and function cannot be empty"
        ));
    }

    #[test]
    fn deduplicate_support_combines_duplicate_values() {
        let support = vec![
            Number::Integer(1),
            Number::Integer(1),
            Number::Integer(2),
            Number::Integer(2),
            Number::Integer(3),
        ];
        let function = vec![
            Number::Rational(Rational64::new(1, 10)),
            Number::Rational(Rational64::new(2, 10)),
            Number::Rational(Rational64::new(1, 5)),
            Number::Rational(Rational64::new(1, 10)),
            Number::Rational(Rational64::new(3, 10)),
        ];

        let (deduped_support, deduped_function) = deduplicate_support(support, function).unwrap();

        assert_eq!(
            deduped_support,
            vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)]
        );
        assert_eq!(
            deduped_function,
            vec![
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(3, 10)),
            ]
        );
    }

    #[test]
    fn deduplicate_support_preserves_first_seen_order() {
        let support = vec![
            Number::Integer(2),
            Number::Integer(1),
            Number::Integer(2),
            Number::Integer(3),
        ];
        let function = vec![
            Number::Rational(Rational64::new(1, 4)),
            Number::Rational(Rational64::new(1, 4)),
            Number::Rational(Rational64::new(1, 4)),
            Number::Rational(Rational64::new(1, 4)),
        ];

        let (deduped_support, deduped_function) = deduplicate_support(support, function).unwrap();

        assert_eq!(
            deduped_support,
            vec![Number::Integer(2), Number::Integer(1), Number::Integer(3)]
        );
        assert_eq!(
            deduped_function,
            vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
    }
}
