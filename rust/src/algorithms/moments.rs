#![allow(dead_code)]

use crate::algorithms::number::Number;
use crate::algorithms::rv::RandomVariable;

/// Computes the mean of a discrete random variable.
///
/// # Arguments
/// * `random_variable`: a discrete random variable
///
/// # Returns
/// * `mean`: the mean of the random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_mean;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(1), Number::Integer(3)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let mean = discrete_mean(&rv).unwrap();
/// assert_eq!(mean, Number::Rational(Rational64::new(2, 1)));
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_mean;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // CDF for support [1, 2, 4]: [0.2, 0.7, 1.0]
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Cdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let mean = discrete_mean(&rv).unwrap();
/// assert!((mean.to_f64() - 2.4).abs() < 1e-12);
/// ```
pub fn discrete_mean(random_variable: &RandomVariable) -> Result<Number, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = &pdf_random_variable.function;
    let support = &pdf_random_variable.support;

    let mean = function
        .iter()
        .zip(support.iter())
        .fold(Number::default(), |mean, (&f, &s)| mean + (f * s));
    Ok(mean)
}

/// Computes the variance of a discrete random variable using the relationships
/// Var(x) = E(X^2) - E(X)^2
///
/// # Arguments
/// * `random_variable`: a discrete random variable
///
/// # Returns
/// * `variance`: the variance of the random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_variance;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv = RandomVariable {
///     function: vec![Number::Rational(Rational64::new(1, 2)); 2],
///     support: vec![Number::Integer(1), Number::Integer(3)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let variance = discrete_variance(&rv).unwrap();
/// assert_eq!(variance, Number::Rational(Rational64::new(1, 1)));
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_variance;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // PDF [0.2, 0.5, 0.3] over support [1, 2, 4]
/// // E[X] = 2.4, E[X^2] = 7.0, so Var(X) = 1.24
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let variance = discrete_variance(&rv).unwrap();
/// assert!((variance.to_f64() - 1.24).abs() < 1e-12);
/// ```
pub fn discrete_variance(random_variable: &RandomVariable) -> Result<Number, String> {
    let pdf_random_variable = random_variable.to_pdf()?;
    // E(X) is the mean of X
    let mean = random_variable.mean()?;

    // Now find E(X^2)
    let two = Number::Integer(2);
    let expected_x_squared = discrete_expected_value(&pdf_random_variable, |x: Number| {
        x.pow(two).expect("failed to computer the square")
    })?;

    // Var(X) = E(X^2) - E(X)^2
    let variance = expected_x_squared - mean.pow(two)?;
    Ok(variance)
}

/// Computes the skewness of a discrete random variable
///
/// # Arguments
/// * `random_variable`: a discrete random variable
///
/// # Returns
/// * `skewness`: the skewness of the random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_skewness;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// // Fair die is symmetric, so skewness = 0
/// let rv = RandomVariable {
///     function: vec![Number::Rational(Rational64::new(1, 6)); 6],
///     support: vec![
///         Number::Integer(1),
///         Number::Integer(2),
///         Number::Integer(3),
///         Number::Integer(4),
///         Number::Integer(5),
///         Number::Integer(6),
///     ],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let skewness = discrete_skewness(&rv).unwrap();
/// assert!(skewness.to_f64().abs() < 1e-12);
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_skewness;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // PDF [0.2, 0.5, 0.3] over support [1, 2, 4]
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let skewness = discrete_skewness(&rv).unwrap();
/// assert!((skewness.to_f64() - 0.4692912730377045).abs() < 1e-12);
/// ```
pub fn discrete_skewness(random_variable: &RandomVariable) -> Result<Number, String> {
    let mean = discrete_mean(random_variable)?;
    let variance = discrete_variance(random_variable)?;
    let standard_deviation = variance.sqrt()?;

    let two = Number::Integer(2);
    let three = Number::Integer(3);

    let term_1 = discrete_expected_value(random_variable, |x| {
        x.pow(three).expect("failed to compute x^3")
    })?;
    let term_2 = three
        * mean
        * discrete_expected_value(random_variable, |x| {
            x.pow(two).expect("failed to compute x^2")
        })?;
    let term_3 = two * mean.pow(three).expect("failed to compute x^3");

    let numerator = term_1 - term_2 + term_3;
    let denominator = standard_deviation
        .pow(three)
        .expect("failed to compute x^3");

    let skewness = numerator / denominator;
    Ok(skewness)
}

/// Computes the kurtosis of a discrete random variable
///
/// # Arguments
/// * `random_variable`: a discrete random variable
///
/// # Returns
/// * `kurtosis`: the kurtosis of the random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_kurtosis;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// // Fair die: kurtosis = 303/175
/// let rv = RandomVariable {
///     function: vec![Number::Rational(Rational64::new(1, 6)); 6],
///     support: vec![
///         Number::Integer(1),
///         Number::Integer(2),
///         Number::Integer(3),
///         Number::Integer(4),
///         Number::Integer(5),
///         Number::Integer(6),
///     ],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let kurtosis = discrete_kurtosis(&rv).unwrap();
/// assert!((kurtosis.to_f64() - (303.0 / 175.0)).abs() < 1e-12);
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_kurtosis;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // PDF [0.2, 0.5, 0.3] over support [1, 2, 4]
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let kurtosis = discrete_kurtosis(&rv).unwrap();
/// assert!((kurtosis.to_f64() - 1.7866805411030178).abs() < 1e-12);
/// ```
pub fn discrete_kurtosis(random_variable: &RandomVariable) -> Result<Number, String> {
    let mean = discrete_mean(random_variable)?;
    let variance = discrete_variance(random_variable)?;
    let standard_deviation = variance.sqrt()?;

    let two = Number::Integer(2);
    let three = Number::Integer(3);
    let four = Number::Integer(4);
    let six = Number::Integer(6);

    let term_1 = discrete_expected_value(random_variable, |x| {
        x.pow(four).expect("failed to compute x^4")
    })?;
    let term_2 = four
        * mean
        * discrete_expected_value(random_variable, |x| {
            x.pow(three).expect("failed to compute x^3")
        })?;
    let term_3 = six
        * mean.pow(two).expect("failed to compute x^2")
        * discrete_expected_value(random_variable, |x| {
            x.pow(two).expect("failed to compute x^2")
        })?;
    let term_4 = three * mean.pow(four).expect("failed to compute x^4");

    let numerator = term_1 - term_2 + term_3 - term_4;
    let denominator = standard_deviation.pow(four).expect("failed to comput x^4");

    let kurtosis = numerator / denominator;
    Ok(kurtosis)
}

/// Computes the coefficient of variation for a discrete random variable
///
/// # Arguments
/// * `random_variable`: a discrete random variable
///
/// # Returns
/// * `variance`: the variance of the random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_coefficient_of_variation;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// // Fair die: CV = sqrt(35/12) / (7/2)
/// let rv = RandomVariable {
///     function: vec![Number::Rational(Rational64::new(1, 6)); 6],
///     support: vec![
///         Number::Integer(1),
///         Number::Integer(2),
///         Number::Integer(3),
///         Number::Integer(4),
///         Number::Integer(5),
///         Number::Integer(6),
///     ],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let cv = discrete_coefficient_of_variation(&rv).unwrap();
/// assert!((cv.to_f64() - 0.487950036474267).abs() < 1e-12);
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_coefficient_of_variation;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // PDF [0.2, 0.5, 0.3] over support [1, 2, 4]:
/// // mean = 2.4, variance = 1.24, so CV = sqrt(1.24)/2.4
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let cv = discrete_coefficient_of_variation(&rv).unwrap();
/// assert!((cv.to_f64() - 0.463980363569169).abs() < 1e-12);
/// ```
pub fn discrete_coefficient_of_variation(
    random_variable: &RandomVariable,
) -> Result<Number, String> {
    let mean = discrete_mean(random_variable)?;
    let standard_deviation = discrete_variance(random_variable)?.sqrt()?;
    let coefficient_of_variation = standard_deviation / mean;
    Ok(coefficient_of_variation)
}

/// Computes the expected value of a random variable with a transformation applied
///
/// # Arguments
/// * `random_variable`: a discrete random variable
/// * `transformation`: a function describing the transformation of X.
///   For example, |x| x.pow(2) for E(X^2)
///
/// # Returns
/// * `discrete_expected_value`: the expected value for the random variable with the
///   transformation applied
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_expected_value;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv = RandomVariable {
///     function: vec![Number::Rational(Rational64::new(1, 6)); 6],
///     support: vec![
///         Number::Integer(1),
///         Number::Integer(2),
///         Number::Integer(3),
///         Number::Integer(4),
///         Number::Integer(5),
///         Number::Integer(6),
///     ],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let expectation = discrete_expected_value(&rv, |x| x).unwrap();
/// assert_eq!(expectation, Number::Rational(Rational64::new(7, 2)));
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_expected_value;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // For PDF [0.2, 0.5, 0.3] over [1, 2, 4], E[X^2] = 7.0
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let two = Number::Integer(2);
/// let expectation = discrete_expected_value(&rv, |x| {
///     x.pow(two).expect("failed to compute square")
/// }).unwrap();
/// assert!((expectation.to_f64() - 7.0).abs() < 1e-12);
/// ```
///
pub fn discrete_expected_value<F>(
    random_variable: &RandomVariable,
    transformation: F,
) -> Result<Number, String>
where
    F: Fn(Number) -> Number,
{
    let pdf_random_variable = random_variable.to_pdf()?;
    let function = &pdf_random_variable.function;
    let support = &pdf_random_variable.support;

    let expectation =
        function
            .iter()
            .zip(support.iter())
            .fold(Number::default(), |ex, (&f, &s)| {
                let transformed_support_value = transformation(s);
                ex + (f * transformed_support_value)
            });

    Ok(expectation)
}

/// Computes the entropy of a discrete random variable
///
/// # Arguments
/// * `random_variable`: a discrete random variable
///
/// # Returns
/// * `entropy`: the entropy of the random variable
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::moments::discrete_entropy;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(1), Number::Integer(2)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let entropy = discrete_entropy(&rv).unwrap();
/// assert_eq!(entropy, Number::Rational(Rational64::new(1, 2)));
/// ```
///
/// ```
/// use applpy_rust::algorithms::moments::discrete_entropy;
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
///
/// // 0.2*log2(1) + 0.5*log2(2) + 0.3*log2(4) = 1.1
/// let rv = RandomVariable {
///     function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
///     support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let entropy = discrete_entropy(&rv).unwrap();
/// assert!((entropy.to_f64() - 1.1).abs() < 1e-12);
/// ```
pub fn discrete_entropy(random_variable: &RandomVariable) -> Result<Number, String> {
    let two = Number::Integer(2);
    let entropy = discrete_expected_value(random_variable, |x| {
        x.log(two).expect("failed to compute log_2(x)")
    })?;
    Ok(entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
    use num_rational::Rational64;

    #[test]
    fn discrete_mean_computes_discrete_expected_value_for_discrete_pdf() {
        // Fair die: E[X] = (1+2+3+4+5+6)/6 = 7/2
        let rv = RandomVariable {
            function: vec![Number::Rational(Rational64::new(1, 6)); 6],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let mean = discrete_mean(&rv).unwrap();
        assert_eq!(mean, Number::Rational(Rational64::new(7, 2)));
    }

    #[test]
    fn discrete_mean_works_for_cdf_input_by_converting_to_pdf() {
        // PDF from this CDF is [0.2, 0.5, 0.3], so E[X] = 0.2*1 + 0.5*2 + 0.3*4 = 2.4
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let mean = discrete_mean(&rv).unwrap();
        assert!((mean.to_f64() - 2.4).abs() < 1e-12);
    }

    #[test]
    fn discrete_mean_returns_error_when_pdf_conversion_fails() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_mean(&rv);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "cannot compute the pdf. function is empty"
        );
    }

    #[test]
    fn discrete_variance_computes_discrete_expected_value_for_discrete_pdf() {
        // Fair die: E[X] = 7/2, E[X^2] = 91/6, so Var(X) = 35/12
        let rv = RandomVariable {
            function: vec![Number::Rational(Rational64::new(1, 6)); 6],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let variance = discrete_variance(&rv).unwrap();
        assert_eq!(variance, Number::Rational(Rational64::new(35, 12)));
    }

    #[test]
    fn discrete_variance_computes_discrete_expected_value_for_float_pdf() {
        // E[X] = 2.4 and E[X^2] = 7.0 for this distribution
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let variance = discrete_variance(&rv).unwrap();
        assert!((variance.to_f64() - 1.24).abs() < 1e-12);
    }

    #[test]
    fn discrete_variance_returns_error_when_mean_computation_fails() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_variance(&rv);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "cannot compute the pdf. function is empty"
        );
    }

    #[test]
    fn discrete_kurtosis_computes_kurtosis_for_rational_pdf() {
        // Fair die: kurtosis = 303/175
        let rv = RandomVariable {
            function: vec![Number::Rational(Rational64::new(1, 6)); 6],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let kurtosis = discrete_kurtosis(&rv).unwrap();
        assert!((kurtosis.to_f64() - (303.0 / 175.0)).abs() < 1e-12);
    }

    #[test]
    fn discrete_kurtosis_computes_kurtosis_for_float_pdf() {
        // For PDF [0.2, 0.5, 0.3] over [1, 2, 4], kurtosis ~= 1.7866805411030178
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let kurtosis = discrete_kurtosis(&rv).unwrap();
        assert!((kurtosis.to_f64() - 1.7866805411030178).abs() < 1e-12);
    }

    #[test]
    fn discrete_kurtosis_returns_error_when_mean_computation_fails() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_kurtosis(&rv);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "cannot compute the pdf. function is empty"
        );
    }

    #[test]
    fn discrete_skewness_computes_skewness_for_rational_pdf() {
        // Fair die is symmetric, so skewness = 0
        let rv = RandomVariable {
            function: vec![Number::Rational(Rational64::new(1, 6)); 6],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let skewness = discrete_skewness(&rv).unwrap();
        assert!(skewness.to_f64().abs() < 1e-12);
    }

    #[test]
    fn discrete_skewness_computes_skewness_for_float_pdf() {
        // For PDF [0.2, 0.5, 0.3] over [1, 2, 4], skewness ~= 0.4692912730377045
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let skewness = discrete_skewness(&rv).unwrap();
        assert!((skewness.to_f64() - 0.4692912730377045).abs() < 1e-12);
    }

    #[test]
    fn discrete_skewness_returns_error_when_mean_computation_fails() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_skewness(&rv);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "cannot compute the pdf. function is empty"
        );
    }

    #[test]
    fn discrete_coefficient_of_variation_computes_discrete_expected_value_for_rational_pdf() {
        // Fair die: CV = sqrt(35/12) / (7/2)
        let rv = RandomVariable {
            function: vec![Number::Rational(Rational64::new(1, 6)); 6],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let coefficient_of_variation = discrete_coefficient_of_variation(&rv).unwrap();
        assert!((coefficient_of_variation.to_f64() - 0.487950036474267).abs() < 1e-12);
    }

    #[test]
    fn discrete_coefficient_of_variation_computes_discrete_expected_value_for_float_pdf() {
        // mean = 2.4, variance = 1.24, so CV = sqrt(1.24)/2.4
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let coefficient_of_variation = discrete_coefficient_of_variation(&rv).unwrap();
        assert!((coefficient_of_variation.to_f64() - 0.463980363569169).abs() < 1e-12);
    }

    #[test]
    fn discrete_coefficient_of_variation_returns_error_when_mean_computation_fails() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = discrete_coefficient_of_variation(&rv);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            "cannot compute the pdf. function is empty"
        );
    }

    #[test]
    fn discrete_expected_value_computes_mean_for_identity_transformation() {
        // Fair die: E[X] = (1+2+3+4+5+6)/6 = 7/2
        let rv = RandomVariable {
            function: vec![Number::Rational(Rational64::new(1, 6)); 6],
            support: vec![
                Number::Integer(1),
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(5),
                Number::Integer(6),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let expectation = discrete_expected_value(&rv, |x| x).unwrap();
        assert_eq!(expectation, Number::Rational(Rational64::new(7, 2)));
    }

    #[test]
    fn discrete_expected_value_applies_transformation_to_support() {
        // For PDF [0.2, 0.5, 0.3] over [1, 2, 4], E[X^2] = 0.2*1 + 0.5*4 + 0.3*16 = 7.0
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let two = Number::Integer(2);
        let expectation =
            discrete_expected_value(&rv, |x| x.pow(two).expect("failed to compute square"))
                .unwrap();
        assert!((expectation.to_f64() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn discrete_expected_value_returns_zero_for_empty_distribution() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let expectation = discrete_expected_value(&rv, |x| x).unwrap();
        assert_eq!(expectation, Number::default());
    }

    #[test]
    fn discrete_entropy_computes_expected_log_for_rational_pdf() {
        // 0.5*log2(1) + 0.5*log2(2) = 0.5
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let entropy = discrete_entropy(&rv).unwrap();
        assert_eq!(entropy, Number::Rational(Rational64::new(1, 2)));
    }

    #[test]
    fn discrete_entropy_computes_expected_log_for_float_pdf() {
        // 0.2*log2(1) + 0.5*log2(2) + 0.3*log2(4) = 1.1
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let entropy = discrete_entropy(&rv).unwrap();
        assert!((entropy.to_f64() - 1.1).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "failed to compute log_2(x)")]
    fn discrete_entropy_panics_when_log_cannot_be_computed() {
        let rv = RandomVariable {
            function: vec![Number::Integer(1)],
            support: vec![Number::Integer(0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let _ = discrete_entropy(&rv);
    }
}
