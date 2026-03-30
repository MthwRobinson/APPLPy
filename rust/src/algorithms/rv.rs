#![allow(dead_code)]

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_rational::Rational64;
use num_traits::cast::ToPrimitive;

use crate::algorithms::algebra;
use crate::algorithms::conversion;
use crate::algorithms::moments;
use crate::algorithms::number::Number;
use crate::algorithms::transform;

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionalForm {
    Cdf,
    Chf,
    Hf,
    Idf,
    Pdf,
    Sf,
}

impl fmt::Display for FunctionalForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            FunctionalForm::Cdf => "cdf",
            FunctionalForm::Chf => "chf",
            FunctionalForm::Hf => "hf",
            FunctionalForm::Idf => "idf",
            FunctionalForm::Pdf => "pdf",
            FunctionalForm::Sf => "sf",
        };

        write!(f, "{}", value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DomainType {
    Continuous,
    Discrete,
    DiscreteFunctional,
}

impl fmt::Display for DomainType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            DomainType::Continuous => "continuous",
            DomainType::Discrete => "discrete",
            DomainType::DiscreteFunctional => "discrete_functional",
        };

        write!(f, "{}", value)
    }
}

#[derive(Debug, Clone)]
pub struct RandomVariable {
    pub function: Vec<Number>,
    pub support: Vec<Number>,
    pub functional_form: FunctionalForm,
    pub domain_type: DomainType,
}

impl Add for RandomVariable {
    type Output = Result<RandomVariable, String>;

    fn add(self, rhs: Self) -> Self::Output {
        let sum_rv = algebra::convolution_discrete(&self, &rhs)?;
        Ok(sum_rv)
    }
}

impl AddAssign for RandomVariable {
    fn add_assign(&mut self, rhs: Self) {
        *self =
            algebra::convolution_discrete(self, &rhs).expect("failed to sum the random variables");
    }
}

impl Div for RandomVariable {
    type Output = Result<RandomVariable, String>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.support.iter().any(|value| match value {
            Number::Float(x) => *x == 0.0,
            Number::Integer(x) => *x == 0,
            Number::Rational(x) => *x.numer() == 0,
        }) {
            return Err("cannot divide by a random variable with zero in its support".to_string());
        }

        let min_support = rhs
            .support
            .first()
            .expect("failed to extract the first number");
        let max_support = rhs
            .support
            .last()
            .expect("failed to extract the last number");
        let transformation = transform::Transformation {
            mapping: |x| Number::Integer(1) / x,
            min_support: *min_support,
            max_support: *max_support,
        };
        let inverse_rhs = transform::transform_discrete(&rhs, &[transformation])?;
        let div_rv = algebra::product_discrete(&self, &inverse_rhs)?;
        Ok(div_rv)
    }
}

impl DivAssign for RandomVariable {
    fn div_assign(&mut self, rhs: Self) {
        let div_rv = self
            .clone()
            .div(rhs)
            .expect("failed to divide the random variables");
        *self = div_rv;
    }
}

/// Computes the difference of two independent discrete random variables.
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let rv1 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(1), Number::Integer(2)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let rv2 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(2), Number::Integer(3)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let difference = (rv1 - rv2).unwrap();
///
/// assert_eq!(
///     difference.support,
///     vec![Number::Integer(-2), Number::Integer(-1), Number::Integer(0)]
/// );
/// assert_eq!(
///     difference.function,
///     vec![
///         Number::Rational(Rational64::new(1, 4)),
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 4)),
///     ]
/// );
/// assert!(difference.verify_pdf(None).unwrap());
/// ```
impl Sub for RandomVariable {
    type Output = Result<RandomVariable, String>;

    fn sub(self, rhs: Self) -> Self::Output {
        let min_support = rhs
            .support
            .first()
            .expect("failed to extract the first number");
        let max_support = rhs
            .support
            .last()
            .expect("failed to extract the last number");
        let transformation = transform::Transformation {
            mapping: |x| x * Number::Integer(-1),
            min_support: *min_support,
            max_support: *max_support,
        };
        let negative_rhs = transform::transform_discrete(&rhs, &[transformation])?;
        let sub_rv = algebra::convolution_discrete(&self, &negative_rhs)?;
        Ok(sub_rv)
    }
}

/// Updates a random variable in place with the difference of two
/// independent discrete random variables.
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{DomainType, FunctionalForm, RandomVariable};
/// use num_rational::Rational64;
///
/// let mut rv1 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(1), Number::Integer(2)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// let rv2 = RandomVariable {
///     function: vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2)),
///     ],
///     support: vec![Number::Integer(2), Number::Integer(3)],
///     functional_form: FunctionalForm::Pdf,
///     domain_type: DomainType::Discrete,
/// };
///
/// rv1 -= rv2;
///
/// assert_eq!(
///     rv1.support,
///     vec![Number::Integer(-2), Number::Integer(-1), Number::Integer(0)]
/// );
/// assert_eq!(
///     rv1.function,
///     vec![
///         Number::Rational(Rational64::new(1, 4)),
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 4)),
///     ]
/// );
/// assert!(rv1.verify_pdf(None).unwrap());
/// ```
impl SubAssign for RandomVariable {
    fn sub_assign(&mut self, rhs: Self) {
        let sub_rv = self
            .clone()
            .sub(rhs)
            .expect("failed to subtract the random variables");
        *self = sub_rv.clone();
    }
}

impl Mul for RandomVariable {
    type Output = Result<RandomVariable, String>;

    fn mul(self, rhs: Self) -> Self::Output {
        let product_rv = algebra::product_discrete(&self, &rhs)?;
        Ok(product_rv)
    }
}

impl MulAssign for RandomVariable {
    fn mul_assign(&mut self, rhs: Self) {
        let product_rv = self
            .clone()
            .mul(rhs)
            .expect("failed to multiply the random variables");
        *self = product_rv;
    }
}

impl RandomVariable {
    pub fn verify_pdf(&self, tolerance: Option<f64>) -> Result<bool, String> {
        if self.functional_form != FunctionalForm::Pdf {
            return Err("verify_pdf only works for PDFs".to_string());
        }

        verify_pdf(&self.function, tolerance)
    }

    pub fn evaluate(&self, value: Number) -> Option<Number> {
        evaluate_rv(&self.function, &self.support, &self.functional_form, value)
    }

    pub fn to_pdf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => conversion::discrete_cdf_to_pdf(self),
            FunctionalForm::Chf => {
                let sf_random_variable = self.to_sf()?;
                sf_random_variable.to_pdf()
            }
            FunctionalForm::Hf => {
                let chf_random_variable = self.to_chf()?;
                chf_random_variable.to_pdf()
            }
            FunctionalForm::Pdf => Ok(self.clone()),
            FunctionalForm::Sf | FunctionalForm::Idf => {
                let cdf_random_variable = self.to_cdf()?;
                conversion::discrete_cdf_to_pdf(&cdf_random_variable)
            }
        }
    }

    pub fn to_cdf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => Ok(self.clone()),
            FunctionalForm::Chf => {
                let sf_random_variable = self.to_sf()?;
                sf_random_variable.to_cdf()
            }
            FunctionalForm::Hf => {
                let hf_random_variable = self.to_chf()?;
                hf_random_variable.to_cdf()
            }
            FunctionalForm::Idf => conversion::swap_discrete_cdf_and_idf(self),
            FunctionalForm::Pdf => conversion::discrete_pdf_to_cdf(self),
            FunctionalForm::Sf => conversion::swap_discrete_cdf_and_sf(self),
        }
    }

    pub fn to_sf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => conversion::swap_discrete_cdf_and_sf(self),
            FunctionalForm::Chf => conversion::discrete_chf_to_sf(self),
            FunctionalForm::Hf => {
                let chf_random_variable = conversion::discrete_hf_to_chf(self)?;
                chf_random_variable.to_sf()
            }
            FunctionalForm::Pdf | FunctionalForm::Idf => {
                let cdf_random_variable = self.to_cdf()?;
                conversion::swap_discrete_cdf_and_sf(&cdf_random_variable)
            }
            FunctionalForm::Sf => Ok(self.clone()),
        }
    }

    pub fn to_chf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Chf => Ok(self.clone()),
            FunctionalForm::Hf => conversion::discrete_hf_to_chf(self),
            _ => {
                let hf_function = conversion::discrete_rv_to_hf(self)?;
                hf_function.to_chf()
            }
        }
    }

    pub fn to_hf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Hf => Ok(self.clone()),
            _ => conversion::discrete_rv_to_hf(self),
        }
    }

    pub fn to_idf(&self) -> Result<RandomVariable, String> {
        match &self.functional_form {
            FunctionalForm::Cdf => conversion::swap_discrete_cdf_and_idf(self),
            FunctionalForm::Chf | FunctionalForm::Hf => {
                let cdf_random_variable = self.to_cdf()?;
                cdf_random_variable.to_idf()
            }
            FunctionalForm::Idf => Ok(self.clone()),
            FunctionalForm::Pdf | FunctionalForm::Sf => {
                let cdf_random_variable = self.to_cdf()?;
                conversion::swap_discrete_cdf_and_idf(&cdf_random_variable)
            }
        }
    }

    pub fn mean(&self) -> Result<Number, String> {
        moments::discrete_mean(self)
    }

    pub fn variance(&self) -> Result<Number, String> {
        moments::discrete_variance(self)
    }

    pub fn skewness(&self) -> Result<Number, String> {
        moments::discrete_skewness(self)
    }

    pub fn kurtosis(&self) -> Result<Number, String> {
        moments::discrete_kurtosis(self)
    }

    pub fn coefficient_of_variation(&self) -> Result<Number, String> {
        moments::discrete_coefficient_of_variation(self)
    }

    pub fn entropy(&self) -> Result<Number, String> {
        moments::discrete_entropy(self)
    }
}

/// Verifies that the area under the PDF of random variable sums to 1
///
/// # Arguments
/// * `function` - the probability mass functon of the RV
/// * `support` - the support of the RV
/// * `tolerance` - sets the tolerance for how far the result
///   can deviate from 1
///
/// # Returns
/// * `valid` - a boolean indicating if the PDF is valid
pub fn verify_pdf(function: &[Number], tolerance: Option<f64>) -> Result<bool, String> {
    let default_tolerance: f64 = 0.000001;
    let tolerance = tolerance.unwrap_or(default_tolerance);

    println!("Now checking for the area ...");
    let mut area: f64 = 0.0;
    let mut all_positive: bool = true;

    for function_value in function {
        let probability: f64 = match &function_value {
            Number::Float(x) => *x,
            Number::Integer(x) => *x as f64,
            Number::Rational(x) => x.to_f64().unwrap(),
        };

        if probability < 0.0 {
            all_positive = false;
        }

        area += probability;
    }
    println!("The area under f(x) is: {}", area);

    println!("Now checking for absolute value ...");
    if !all_positive {
        return Ok(false);
    }

    Ok((area > 1.0 - tolerance) && (area < 1.0 + tolerance))
}

/// Constructs a discrete random variable from an array of variates
///
/// # Arguments
/// * `variates` - a list of numbers, which represent data observations
///
/// # Returns
/// * `support` - the support of the random variable, which is a sorted
///   and deduplicated vector of variates
/// * `function` - the probability of each variate, assuming each is equally likely
///
/// # Examples
/// ```
/// use applpy_rust::algorithms::number::Number;
/// use applpy_rust::algorithms::rv::{bootstrap_rv, FunctionalForm};
/// use num_rational::Rational64;
///
/// let variates = vec![
///     Number::Integer(2),
///     Number::Integer(1),
///     Number::Integer(1),
///     Number::Integer(2),
/// ];
///
/// let rv = bootstrap_rv(&variates).unwrap();
///
/// assert_eq!(rv.support, vec![Number::Integer(1), Number::Integer(2)]);
/// assert_eq!(
///     rv.function,
///     vec![
///         Number::Rational(Rational64::new(1, 2)),
///         Number::Rational(Rational64::new(1, 2))
///     ]
/// );
/// assert_eq!(rv.functional_form, FunctionalForm::Pdf);
/// ```
pub fn bootstrap_rv(variates: &[Number]) -> Result<RandomVariable, String> {
    if variates.is_empty() {
        return Err("at least one variate is required to construct the bootstrap rv".to_string());
    }

    let mut sorted_variates = variates.to_vec();
    sorted_variates.sort_by(|a, b| {
        let first = a.to_f64();
        let second = b.to_f64();
        first.total_cmp(&second)
    });

    let num_items = sorted_variates.len();
    let num_observations: i64 = num_items
        .try_into()
        .expect("could not convert sorted variate length to number");
    let base_probability = Number::Rational(Rational64::new(1, num_observations));

    let mut function = Vec::new();
    let mut support = Vec::new();

    let mut current_probability = base_probability;
    let mut current_variate = sorted_variates[0];

    for &variate in sorted_variates.iter().skip(1) {
        if variate == current_variate {
            current_probability += base_probability;
        } else {
            support.push(current_variate);
            function.push(current_probability);
            current_variate = variate;
            current_probability = base_probability;
        }
    }

    function.push(current_probability);
    support.push(current_variate);

    let random_variable = RandomVariable {
        function,
        support,
        functional_form: FunctionalForm::Pdf,
        domain_type: DomainType::Discrete,
    };
    Ok(random_variable)
}

/// Evaluates a random variable at a specific value. Used to compute F(x) for
/// the random variable
///
/// # Arguments
/// * `function` - the probability mass functon of the RV
/// * `support` - the support of the RV
/// * `functional_form` - the functional form of the RV
/// * `value` - the value at which to evaulate the function
///
/// # Returns
/// * `output` - a Number representing F(x) at the
pub fn evaluate_rv(
    function: &[Number],
    support: &[Number],
    functional_form: &FunctionalForm,
    value: Number,
) -> Option<Number> {
    if function.is_empty() || support.is_empty() {
        return None;
    }

    let value_f64 = value.to_f64();
    let lower_bound = support[0].to_f64();

    if *functional_form == FunctionalForm::Idf {
        if value_f64 < lower_bound || value_f64 > 1.0 {
            return None;
        }
    } else {
        let upper_bound = support[support.len() - 1].to_f64();

        if value_f64 < lower_bound {
            return match functional_form {
                FunctionalForm::Cdf
                | FunctionalForm::Chf
                | FunctionalForm::Hf
                | FunctionalForm::Pdf => Some(Number::Integer(0)),
                FunctionalForm::Sf => Some(Number::Integer(1)),
                FunctionalForm::Idf => None,
            };
        }

        if value_f64 > upper_bound {
            return match functional_form {
                FunctionalForm::Cdf => Some(Number::Integer(1)),
                FunctionalForm::Sf | FunctionalForm::Pdf => Some(Number::Integer(0)),
                FunctionalForm::Chf | FunctionalForm::Hf => Some(Number::Float(f64::INFINITY)),
                FunctionalForm::Idf => None,
            };
        }
    }

    for (support_window, &function_value) in support.windows(2).zip(function.iter()) {
        let current_support = support_window[0].to_f64();
        let next_support = support_window[1].to_f64();
        if current_support <= value_f64 && next_support > value_f64 {
            return Some(function_value);
        }
    }

    function.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    #[test]
    fn add_returns_sum_distribution() {
        let lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = (lhs + rhs).unwrap();

        assert_eq!(
            result.support,
            vec![Number::Integer(3), Number::Integer(4), Number::Integer(5)]
        );
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn add_assign_updates_random_variable_in_place() {
        let mut lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        lhs += rhs;

        assert_eq!(
            lhs.support,
            vec![Number::Integer(3), Number::Integer(4), Number::Integer(5)]
        );
        assert_eq!(
            lhs.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(lhs.functional_form, FunctionalForm::Pdf));
        assert!(matches!(lhs.domain_type, DomainType::Discrete));
    }

    #[test]
    fn div_returns_quotient_distribution() {
        let lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![
                Number::Rational(Rational64::new(2, 1)),
                Number::Rational(Rational64::new(4, 1)),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = (lhs / rhs).unwrap();

        assert_eq!(
            result.support,
            vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 1)),
                Number::Rational(Rational64::new(2, 1)),
            ]
        );
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn div_assign_updates_random_variable_in_place() {
        let mut lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(4)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![
                Number::Rational(Rational64::new(2, 1)),
                Number::Rational(Rational64::new(4, 1)),
            ],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        lhs /= rhs;

        assert_eq!(
            lhs.support,
            vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 1)),
                Number::Rational(Rational64::new(2, 1)),
            ]
        );
        assert_eq!(
            lhs.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(lhs.functional_form, FunctionalForm::Pdf));
        assert!(matches!(lhs.domain_type, DomainType::Discrete));
    }

    #[test]
    fn div_returns_error_when_rhs_support_contains_zero() {
        let lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(0), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = lhs.clone();

        let err = (lhs / rhs).unwrap_err();

        assert_eq!(
            err,
            "cannot divide by a random variable with zero in its support"
        );
    }

    #[test]
    fn sub_returns_difference_distribution() {
        let lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = (lhs - rhs).unwrap();

        assert_eq!(
            result.support,
            vec![Number::Integer(-2), Number::Integer(-1), Number::Integer(0)]
        );
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn sub_assign_updates_random_variable_in_place() {
        let mut lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        lhs -= rhs;

        assert_eq!(
            lhs.support,
            vec![Number::Integer(-2), Number::Integer(-1), Number::Integer(0)]
        );
        assert_eq!(
            lhs.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(lhs.functional_form, FunctionalForm::Pdf));
        assert!(matches!(lhs.domain_type, DomainType::Discrete));
    }

    #[test]
    fn mul_returns_product_distribution() {
        let lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = (lhs * rhs).unwrap();

        assert_eq!(
            result.support,
            vec![
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(6),
            ]
        );
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn mul_assign_updates_random_variable_in_place() {
        let mut lhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };
        let rhs = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        lhs *= rhs;

        assert_eq!(
            lhs.support,
            vec![
                Number::Integer(2),
                Number::Integer(3),
                Number::Integer(4),
                Number::Integer(6),
            ]
        );
        assert_eq!(
            lhs.function,
            vec![
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
                Number::Rational(Rational64::new(1, 4)),
            ]
        );
        assert!(matches!(lhs.functional_form, FunctionalForm::Pdf));
        assert!(matches!(lhs.domain_type, DomainType::Discrete));
    }

    #[test]
    fn verify_pdf_returns_err_for_non_pdf_functional_form() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0)],
            support: vec![Number::Float(1.0)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Continuous,
        };

        let result = rv.verify_pdf(None);
        assert!(result.is_err());
    }

    #[test]
    fn verify_pdf_accepts_exact_unit_area() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.5)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_rejects_area_outside_default_tolerance() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.49)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(!rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_uses_custom_tolerance() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.5), Number::Float(0.49)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(rv.verify_pdf(Some(0.02)).unwrap());
    }

    #[test]
    fn verify_pdf_supports_rational_values() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 2)),
                Number::Rational(Rational64::new(1, 2)),
            ],
            support: vec![Number::Integer(1), Number::Integer(1)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };

        assert!(rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn verify_pdf_returns_false_with_negative_function_values() {
        let rv = RandomVariable {
            function: vec![Number::Float(-0.5), Number::Float(1.5)],
            support: vec![Number::Float(1.0), Number::Float(1.0)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Continuous,
        };
        assert!(!rv.verify_pdf(None).unwrap());
    }

    #[test]
    fn bootstrap_rv_returns_error_for_empty_input() {
        let result = bootstrap_rv(&[]);

        assert!(matches!(
            result,
            Err(msg) if msg == "at least one variate is required to construct the bootstrap rv"
        ));
    }

    #[test]
    fn bootstrap_rv_sorts_support_and_aggregates_duplicates() {
        let variates = vec![
            Number::Integer(3),
            Number::Integer(1),
            Number::Integer(3),
            Number::Integer(2),
            Number::Integer(1),
        ];

        let rv = bootstrap_rv(&variates).unwrap();

        assert_eq!(
            rv.support,
            vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)]
        );
        assert_eq!(
            rv.function,
            vec![
                Number::Rational(Rational64::new(2, 5)),
                Number::Rational(Rational64::new(1, 5)),
                Number::Rational(Rational64::new(2, 5)),
            ]
        );
        assert!(matches!(rv.functional_form, FunctionalForm::Pdf));
        assert!(matches!(rv.domain_type, DomainType::Discrete));
    }

    #[test]
    fn bootstrap_rv_handles_single_observation() {
        let rv = bootstrap_rv(&[Number::Integer(7)]).unwrap();

        assert_eq!(rv.support, vec![Number::Integer(7)]);
        assert_eq!(rv.function, vec![Number::Rational(Rational64::new(1, 1))]);
        assert!(matches!(rv.functional_form, FunctionalForm::Pdf));
        assert!(matches!(rv.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_pdf_returns_clone_when_already_pdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.8)],
            support: vec![Number::Integer(1), Number::Integer(2)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_pdf().unwrap();

        assert_eq!(result.function.len(), rv.function.len());
        assert!(matches!(result.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(result.function[1], Number::Float(x) if x == 0.8));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_pdf_converts_cdf_to_pdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_pdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.2).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.5).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.support[2], Number::Integer(3)));
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_pdf_converts_sf_to_pdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.8), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_pdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.2).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.5).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Pdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_pdf_propagates_conversion_error_for_empty_cdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_pdf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the pdf. function is empty"));
    }

    #[test]
    fn to_cdf_returns_clone_when_already_cdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_cdf().unwrap();

        assert_eq!(result.function.len(), rv.function.len());
        assert!(matches!(result.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(result.function[1], Number::Float(x) if x == 0.7));
        assert!(matches!(result.function[2], Number::Float(x) if x == 1.0));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.support[2], Number::Integer(3)));
        assert!(matches!(result.functional_form, FunctionalForm::Cdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_cdf_converts_pdf_to_cdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_cdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if x == 0.2));
        assert!(matches!(result.function[1], Number::Float(x) if x == 0.7));
        assert!(matches!(result.function[2], Number::Float(x) if x == 1.0));
        assert_eq!(result.support.len(), rv.support.len());
        assert!(matches!(result.support[0], Number::Integer(1)));
        assert!(matches!(result.support[1], Number::Integer(2)));
        assert!(matches!(result.support[2], Number::Integer(3)));
        assert!(matches!(result.functional_form, FunctionalForm::Cdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_cdf_converts_sf_to_cdf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.7), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_cdf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.7).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Cdf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_cdf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_cdf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn to_sf_returns_clone_when_already_sf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.7), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_sf().unwrap();

        assert_eq!(result.function, rv.function);
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Sf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_sf_converts_cdf_to_sf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_sf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.8).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Sf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_sf_converts_pdf_to_sf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_sf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert!(matches!(result.function[0], Number::Float(x) if (x - 0.8).abs() < 1e-12));
        assert!(matches!(result.function[1], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.function[2], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Sf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_sf_converts_idf_to_sf_starting_at_one_and_non_increasing() {
        let rv = RandomVariable {
            function: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            support: vec![Number::Integer(0), Number::Float(0.2), Number::Float(0.7)],
            functional_form: FunctionalForm::Idf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_sf().unwrap();

        assert_eq!(result.function.len(), 3);
        assert_eq!(result.function[0], Number::Integer(1));
        let values: Vec<f64> = result.function.iter().map(|value| value.to_f64()).collect();
        assert!(values.windows(2).all(|window| window[0] >= window[1]));
        assert_eq!(result.support, rv.function);
        assert!(matches!(result.functional_form, FunctionalForm::Sf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_sf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_sf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn to_idf_returns_clone_when_already_idf() {
        let rv = RandomVariable {
            function: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            support: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            functional_form: FunctionalForm::Idf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.function);
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_converts_cdf_to_idf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.7), Number::Float(1.0)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Cdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.support);
        assert_eq!(
            result.support,
            vec![Number::Integer(0), Number::Float(0.2), Number::Float(0.7)]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_converts_pdf_to_idf() {
        let rv = RandomVariable {
            function: vec![Number::Float(0.2), Number::Float(0.5), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.support);
        assert_eq!(
            result.support,
            vec![Number::Integer(0), Number::Float(0.2), Number::Float(0.7)]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_converts_sf_to_idf() {
        let rv = RandomVariable {
            function: vec![Number::Float(1.0), Number::Float(0.7), Number::Float(0.3)],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::DiscreteFunctional,
        };

        let result = rv.to_idf().unwrap();

        assert_eq!(result.function, rv.support);
        assert_eq!(result.support.len(), 3);
        assert!(matches!(result.support[0], Number::Integer(0)));
        assert!(matches!(result.support[1], Number::Float(x) if (x - 0.0).abs() < 1e-12));
        assert!(matches!(result.support[2], Number::Float(x) if (x - 0.3).abs() < 1e-12));
        assert!(matches!(result.functional_form, FunctionalForm::Idf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_chf_converts_sf_to_chf() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(9, 10)),
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(1, 10)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_chf().unwrap();

        assert_eq!(result.support, rv.support);
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 9)),
                Number::Rational(Rational64::new(19, 9)),
                Number::Rational(Rational64::new(37, 9)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Chf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_chf_returns_clone_when_already_chf() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 9)),
                Number::Rational(Rational64::new(19, 9)),
                Number::Rational(Rational64::new(37, 9)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Chf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_chf().unwrap();

        assert_eq!(result.function, rv.function);
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Chf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_chf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_chf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn to_hf_converts_sf_to_hf() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(9, 10)),
                Number::Rational(Rational64::new(3, 10)),
                Number::Rational(Rational64::new(1, 10)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Sf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_hf().unwrap();

        assert_eq!(result.support, rv.support);
        assert_eq!(
            result.function,
            vec![
                Number::Rational(Rational64::new(1, 9)),
                Number::Rational(Rational64::new(2, 1)),
                Number::Rational(Rational64::new(2, 1)),
            ]
        );
        assert!(matches!(result.functional_form, FunctionalForm::Hf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_hf_returns_clone_when_already_hf() {
        let rv = RandomVariable {
            function: vec![
                Number::Rational(Rational64::new(1, 9)),
                Number::Rational(Rational64::new(2, 1)),
                Number::Rational(Rational64::new(2, 1)),
            ],
            support: vec![Number::Integer(1), Number::Integer(2), Number::Integer(3)],
            functional_form: FunctionalForm::Hf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_hf().unwrap();

        assert_eq!(result.function, rv.function);
        assert_eq!(result.support, rv.support);
        assert!(matches!(result.functional_form, FunctionalForm::Hf));
        assert!(matches!(result.domain_type, DomainType::Discrete));
    }

    #[test]
    fn to_idf_propagates_conversion_error_for_empty_pdf() {
        let rv = RandomVariable {
            function: vec![],
            support: vec![],
            functional_form: FunctionalForm::Pdf,
            domain_type: DomainType::Discrete,
        };

        let result = rv.to_idf();
        assert!(matches!(result, Err(msg) if msg == "cannot compute the cdf. function is empty"));
    }

    #[test]
    fn evaluate_rv_returns_interval_value_for_interior_point() {
        let function = vec![Number::Float(0.1), Number::Float(0.4), Number::Float(0.9)];
        let support = vec![Number::Integer(1), Number::Integer(3), Number::Integer(5)];

        let result = evaluate_rv(
            &function,
            &support,
            &FunctionalForm::Cdf,
            Number::Integer(2),
        );
        assert_eq!(result, Some(Number::Float(0.1)));
    }

    #[test]
    fn evaluate_rv_returns_value_for_exact_support_point() {
        let function = vec![
            Number::Integer(10),
            Number::Integer(20),
            Number::Integer(30),
        ];
        let support = vec![Number::Integer(1), Number::Integer(3), Number::Integer(5)];

        let result = evaluate_rv(
            &function,
            &support,
            &FunctionalForm::Cdf,
            Number::Integer(3),
        );
        assert_eq!(result, Some(Number::Integer(20)));
    }

    #[test]
    fn evaluate_rv_uses_functional_form_specific_out_of_support_behavior() {
        let function = vec![
            Number::Integer(10),
            Number::Integer(20),
            Number::Integer(30),
        ];
        let support = vec![Number::Integer(1), Number::Integer(3), Number::Integer(5)];

        assert_eq!(
            evaluate_rv(
                &function,
                &support,
                &FunctionalForm::Cdf,
                Number::Integer(0)
            ),
            Some(Number::Integer(0))
        );
        assert_eq!(
            evaluate_rv(
                &function,
                &support,
                &FunctionalForm::Cdf,
                Number::Integer(9)
            ),
            Some(Number::Integer(1))
        );
        assert_eq!(
            evaluate_rv(&function, &support, &FunctionalForm::Sf, Number::Integer(0)),
            Some(Number::Integer(1))
        );
        assert_eq!(
            evaluate_rv(&function, &support, &FunctionalForm::Sf, Number::Integer(9)),
            Some(Number::Integer(0))
        );
        assert_eq!(
            evaluate_rv(
                &function,
                &support,
                &FunctionalForm::Pdf,
                Number::Integer(0)
            ),
            Some(Number::Integer(0))
        );
        assert_eq!(
            evaluate_rv(
                &function,
                &support,
                &FunctionalForm::Pdf,
                Number::Integer(9)
            ),
            Some(Number::Integer(0))
        );
        assert_eq!(
            evaluate_rv(
                &function,
                &support,
                &FunctionalForm::Idf,
                Number::Integer(0)
            ),
            None
        );
        assert_eq!(
            evaluate_rv(
                &function,
                &support,
                &FunctionalForm::Idf,
                Number::Integer(9)
            ),
            None
        );
        assert_eq!(
            evaluate_rv(&function, &support, &FunctionalForm::Hf, Number::Integer(0)),
            Some(Number::Integer(0))
        );
        assert!(matches!(
            evaluate_rv(&function, &support, &FunctionalForm::Hf, Number::Integer(9)),
            Some(Number::Float(x)) if x.is_infinite() && x.is_sign_positive()
        ));
        assert_eq!(
            evaluate_rv(
                &function,
                &support,
                &FunctionalForm::Chf,
                Number::Integer(0)
            ),
            Some(Number::Integer(0))
        );
        assert!(matches!(
            evaluate_rv(&function, &support, &FunctionalForm::Chf, Number::Integer(9)),
            Some(Number::Float(x)) if x.is_infinite() && x.is_sign_positive()
        ));
    }

    #[test]
    fn evaluate_rv_returns_none_for_empty_function() {
        let function: Vec<Number> = vec![];
        let support = vec![Number::Integer(1), Number::Integer(2)];

        let result = evaluate_rv(
            &function,
            &support,
            &FunctionalForm::Cdf,
            Number::Integer(1),
        );
        assert_eq!(result, None);
    }
}
