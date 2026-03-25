#![allow(dead_code)]

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Sub};
use std::{collections::BTreeMap, f64::consts::E};

use num_rational::Rational64;
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Number {
    Float(f64),
    Integer(i64),
    Rational(Rational64),
}

impl Number {
    pub fn one() -> Number {
        Number::Integer(1)
    }

    fn is_zero(self) -> bool {
        match self {
            Number::Float(x) => x == 0.0,
            Number::Integer(x) => x == 0,
            Number::Rational(x) => *x.numer() == 0,
        }
    }

    fn is_one(self) -> bool {
        match self {
            Number::Float(x) => x == 1.0,
            Number::Integer(x) => x == 1,
            Number::Rational(x) => x == Rational64::from_integer(1),
        }
    }

    fn is_positive(self) -> bool {
        match self {
            Number::Float(x) => x.is_finite() && x > 0.0,
            Number::Integer(x) => x > 0,
            Number::Rational(x) => x > Rational64::from_integer(0),
        }
    }

    fn to_positive_rational_for_log(self) -> Option<Rational64> {
        match self {
            Number::Integer(value) if value > 0 => Some(Rational64::from_integer(value)),
            Number::Rational(value) if value > Rational64::from_integer(0) => Some(value),
            _ => None,
        }
    }

    fn to_rational(self) -> Rational64 {
        match self {
            Number::Float(x) => Rational64::approximate_float(x)
                .expect("cannot convert non-finite float to Rational64"),
            Number::Integer(x) => Rational64::from_integer(x),
            Number::Rational(x) => x,
        }
    }

    pub fn to_f64(self) -> f64 {
        match self {
            Number::Float(x) => x,
            Number::Integer(x) => x as f64,
            Number::Rational(x) => x.to_f64().expect("cannot convert Rational64 to f64"),
        }
    }

    fn powi(self, exponent: i32) -> Result<Number, String> {
        match self {
            Number::Float(base) => Ok(Number::Float(base.powi(exponent))),
            Number::Integer(base) => {
                if base == 0 && exponent < 0 {
                    return Err("0 cannot be raised to a negative power".to_string());
                }

                if exponent >= 0 {
                    let n = exponent.unsigned_abs();
                    Ok(Number::Integer(base.pow(n)))
                } else {
                    Ok(Number::Rational(
                        Rational64::from_integer(base).pow(exponent),
                    ))
                }
            }
            Number::Rational(base) => {
                if *base.numer() == 0 && exponent < 0 {
                    return Err("0 cannot be raised to a negative power".to_string());
                }

                Ok(Number::Rational(base.pow(exponent)))
            }
        }
    }

    /// Raises a `Number` to a numeric exponent.
    ///
    /// ```
    /// use applpy_rust::algorithms::number::Number;
    ///
    /// let result = Number::Integer(3).pow(Number::Integer(2)).unwrap();
    /// assert_eq!(result, Number::Integer(9));
    /// ```
    pub fn pow(self, exponent: Number) -> Result<Number, String> {
        match exponent {
            Number::Integer(exp) => {
                let exp = i32::try_from(exp)
                    .map_err(|_| "integer exponent is too large for powi".to_string())?;
                self.powi(exp)
            }
            Number::Rational(exp) if *exp.denom() == 1 => {
                let exp = i32::try_from(*exp.numer())
                    .map_err(|_| "integer exponent is too large for powi".to_string())?;
                self.powi(exp)
            }
            other => {
                let base = self.to_f64();
                let exponent = other.to_f64();

                if self.is_zero() && exponent < 0.0 {
                    return Err("0 cannot be raised to a negative power".to_string());
                }

                if base < 0.0 {
                    return Err(
                        "fractional powers of negative numbers are not supported in real arithmetic"
                            .to_string(),
                    );
                }

                Ok(Number::Float(base.powf(exponent)))
            }
        }
    }

    fn number_from_rational(value: Rational64) -> Number {
        if *value.denom() == 1 {
            Number::Integer(*value.numer())
        } else {
            Number::Rational(value)
        }
    }

    fn exact_nth_root_u64(value: u64, n: u32) -> Option<u64> {
        if n == 0 {
            return None;
        }

        if n == 1 || value <= 1 {
            return Some(value);
        }

        let mut low = 0_u64;
        let mut high = value;

        while low <= high {
            let mid = low + ((high - low) / 2);
            match mid.checked_pow(n) {
                Some(pow) if pow == value => return Some(mid),
                Some(pow) if pow < value => low = mid + 1,
                _ => {
                    if mid == 0 {
                        break;
                    }
                    high = mid - 1;
                }
            }
        }

        None
    }

    fn exact_positive_root(&self, n: u32) -> Option<Number> {
        match *self {
            Number::Float(_) => None,
            Number::Integer(value) => {
                let is_negative = value < 0;
                if is_negative && n.is_multiple_of(2) {
                    return None;
                }

                let abs_root = Number::exact_nth_root_u64(value.unsigned_abs(), n)?;
                if is_negative {
                    if abs_root == (i64::MAX as u64) + 1 {
                        Some(Number::Integer(i64::MIN))
                    } else {
                        let root = i64::try_from(abs_root).ok()?;
                        Some(Number::Integer(-root))
                    }
                } else {
                    let root = i64::try_from(abs_root).ok()?;
                    Some(Number::Integer(root))
                }
            }
            Number::Rational(value) => {
                if *value.numer() < 0 && n.is_multiple_of(2) {
                    return None;
                }

                let numer_root = Number::exact_nth_root_u64(value.numer().unsigned_abs(), n)?;
                let denom_root = Number::exact_nth_root_u64(value.denom().unsigned_abs(), n)?;

                let denom = i64::try_from(denom_root).ok()?;
                let numer = if *value.numer() < 0 {
                    if numer_root == (i64::MAX as u64) + 1 {
                        i64::MIN
                    } else {
                        let root = i64::try_from(numer_root).ok()?;
                        -root
                    }
                } else {
                    i64::try_from(numer_root).ok()?
                };

                Some(Number::number_from_rational(Rational64::new(numer, denom)))
            }
        }
    }

    fn reciprocal(number: Number) -> Result<Number, String> {
        match number {
            Number::Float(value) => Ok(Number::Float(1.0 / value)),
            Number::Integer(value) => {
                if value == 0 {
                    Err("0 cannot be raised to a negative power".to_string())
                } else if value == 1 || value == -1 {
                    Ok(Number::Integer(1 / value))
                } else {
                    Ok(Number::Rational(Rational64::new(1, value)))
                }
            }
            Number::Rational(value) => {
                if *value.numer() == 0 {
                    Err("0 cannot be raised to a negative power".to_string())
                } else {
                    Ok(Number::number_from_rational(value.recip()))
                }
            }
        }
    }

    /// Computes the `n`th root of a `Number`.
    ///
    /// ```
    /// use applpy_rust::algorithms::number::Number;
    ///
    /// let result = Number::Integer(27).root(3).unwrap();
    /// assert_eq!(result, Number::Integer(3));
    /// ```
    pub fn root(&self, n: i32) -> Result<Number, String> {
        if n == 0 {
            return Err("cannot take zeroth root".to_string());
        }

        if self.is_zero() && n < 0 {
            return Err("0 cannot be raised to a negative power".to_string());
        }

        let abs_n = n.unsigned_abs();
        let base = self.to_f64();

        if base < 0.0 && abs_n.is_multiple_of(2) {
            return Err(
                "even roots of negative numbers are not supported in real arithmetic".to_string(),
            );
        }

        if let Some(exact) = self.exact_positive_root(abs_n) {
            if n < 0 {
                return Number::reciprocal(exact);
            }
            return Ok(exact);
        }

        let mut result = if base < 0.0 {
            -(-base).powf(1.0 / abs_n as f64)
        } else {
            base.powf(1.0 / abs_n as f64)
        };

        if n < 0 {
            result = 1.0 / result;
        }

        Ok(Number::Float(result))
    }

    /// Computes the square root of a `Number`.
    ///
    /// ```
    /// use applpy_rust::algorithms::number::Number;
    ///
    /// let result = Number::Integer(9).sqrt().unwrap();
    /// assert_eq!(result, Number::Integer(3));
    /// ```
    pub fn sqrt(&self) -> Result<Number, String> {
        self.root(2)
    }

    fn factor_u64(mut value: u64) -> BTreeMap<u64, i32> {
        let mut factors = BTreeMap::new();
        if value <= 1 {
            return factors;
        }

        let mut count = 0_i32;
        while value.is_multiple_of(2) {
            value /= 2;
            count += 1;
        }
        if count > 0 {
            factors.insert(2, count);
        }

        let mut divisor = 3_u64;
        while divisor <= value / divisor {
            let mut power = 0_i32;
            while value.is_multiple_of(divisor) {
                value /= divisor;
                power += 1;
            }
            if power > 0 {
                factors.insert(divisor, power);
            }
            divisor += 2;
        }

        if value > 1 {
            factors.insert(value, 1);
        }

        factors
    }

    fn rational_prime_exponents(value: Rational64) -> BTreeMap<u64, i32> {
        let mut exponents = BTreeMap::new();
        for (prime, power) in Number::factor_u64(value.numer().unsigned_abs()) {
            exponents.insert(prime, power);
        }
        for (prime, power) in Number::factor_u64(value.denom().unsigned_abs()) {
            *exponents.entry(prime).or_insert(0) -= power;
        }
        exponents.retain(|_, power| *power != 0);
        exponents
    }

    fn exact_log_rational(value: Rational64, base: Rational64) -> Option<Number> {
        if value == Rational64::from_integer(1) {
            return Some(Number::Integer(0));
        }

        let value_exponents = Number::rational_prime_exponents(value);
        let base_exponents = Number::rational_prime_exponents(base);

        if base_exponents.is_empty() {
            return None;
        }

        let mut numerator: Option<i32> = None;
        let mut denominator: Option<i32> = None;

        for (&prime, &base_power) in &base_exponents {
            let value_power = *value_exponents.get(&prime).unwrap_or(&0);
            if let (Some(num), Some(den)) = (numerator, denominator) {
                if i128::from(value_power) * i128::from(den)
                    != i128::from(base_power) * i128::from(num)
                {
                    return None;
                }
            } else {
                numerator = Some(value_power);
                denominator = Some(base_power);
            }
        }

        for (&prime, &value_power) in &value_exponents {
            if !base_exponents.contains_key(&prime) && value_power != 0 {
                return None;
            }
        }

        let ratio = Rational64::new(i64::from(numerator?), i64::from(denominator?));
        Some(Number::number_from_rational(ratio))
    }

    /// Computes the logarithm of a `Number` with a chosen base.
    ///
    /// ```
    /// use applpy_rust::algorithms::number::Number;
    ///
    /// let result = Number::Integer(8).log(Number::Integer(2)).unwrap();
    /// assert_eq!(result, Number::Integer(3));
    /// ```
    pub fn log(&self, base: Number) -> Result<Number, String> {
        if !self.is_positive() {
            return Err("logarithm is undefined for non-positive values".to_string());
        }
        if !base.is_positive() || base.is_one() {
            return Err("logarithm base must be positive and not equal to 1".to_string());
        }

        if self.is_one() {
            return Ok(Number::Integer(0));
        }

        if let (Some(value), Some(base)) = (
            (*self).to_positive_rational_for_log(),
            base.to_positive_rational_for_log(),
        ) {
            if let Some(exact) = Number::exact_log_rational(value, base) {
                return Ok(exact);
            }
        }

        Ok(Number::Float(self.to_f64().log(base.to_f64())))
    }

    /// Computes the natural logarithm (`log_e`) of a `Number`.
    ///
    /// ```
    /// use applpy_rust::algorithms::number::Number;
    ///
    /// let result = Number::Integer(1).ln().unwrap();
    /// assert_eq!(result, Number::Integer(0));
    /// ```
    pub fn ln(&self) -> Result<Number, String> {
        self.log(Number::Float(E))
    }

    fn promote(self, other: Self) -> (Self, Self) {
        match (&self, &other) {
            (Number::Float(_), _) | (_, Number::Float(_)) => {
                (Number::Float(self.to_f64()), Number::Float(other.to_f64()))
            }

            (Number::Rational(_), _) | (_, Number::Rational(_)) => (
                Number::Rational(self.to_rational()),
                Number::Rational(other.to_rational()),
            ),

            _ => (self, other),
        }
    }
}

impl Default for Number {
    fn default() -> Self {
        Number::Integer(0)
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Float(x) => write!(f, "{}", x),
            Number::Integer(x) => write!(f, "{}", x),
            Number::Rational(x) => {
                if *x.denom() == 1 {
                    write!(f, "{}", x.numer())
                } else {
                    write!(f, "{}/{}", x.numer(), x.denom())
                }
            }
        }
    }
}

impl Add for Number {
    type Output = Number;

    fn add(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a + b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a + b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a + b),
            _ => unreachable!(),
        }
    }
}

impl AddAssign for Number {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Number {
    type Output = Number;

    fn sub(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a - b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a - b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a - b),
            _ => unreachable!(),
        }
    }
}

impl Mul for Number {
    type Output = Number;

    fn mul(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a * b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a * b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a * b),
            _ => unreachable!(),
        }
    }
}

impl Div for Number {
    type Output = Number;

    fn div(self, rhs: Self) -> Self::Output {
        let (a, b) = self.promote(rhs);

        match (a, b) {
            (Number::Float(a), Number::Float(b)) => Number::Float(a / b),
            (Number::Rational(a), Number::Rational(b)) => Number::Rational(a / b),
            (Number::Integer(a), Number::Integer(b)) => Number::Integer(a / b),
            _ => unreachable!(),
        }
    }
}

impl Sum for Number {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Number::default(), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_zero_integer() {
        assert!(matches!(Number::default(), Number::Integer(0)));
    }

    #[test]
    fn mixed_addition_promotes_to_float() {
        let result = Number::Integer(2) + Number::Float(1.5);
        assert!(matches!(result, Number::Float(x) if x == 3.5));
    }

    #[test]
    fn mixed_addition_promotes_to_rational() {
        let result = Number::Integer(1) + Number::Rational(Rational64::new(1, 2));
        assert_eq!(result, Number::Rational(Rational64::new(3, 2)));
    }

    #[test]
    fn add_assign_works() {
        let mut value = Number::Integer(1);
        value += Number::Integer(2);
        assert!(matches!(value, Number::Integer(3)));
    }

    #[test]
    fn display_formats_each_variant() {
        assert_eq!(Number::Float(1.5).to_string(), "1.5");
        assert_eq!(Number::Integer(7).to_string(), "7");
        assert_eq!(Number::Rational(Rational64::new(3, 2)).to_string(), "3/2");
        assert_eq!(Number::Rational(Rational64::new(4, 1)).to_string(), "4");
    }

    #[test]
    fn integer_power_stays_exact() {
        let result = Number::Integer(3).pow(Number::Integer(2)).unwrap();
        assert_eq!(result, Number::Integer(9));
    }

    #[test]
    fn rational_base_with_integer_exponent_stays_rational() {
        let result = Number::Rational(Rational64::new(2, 3))
            .pow(Number::Rational(Rational64::new(2, 1)))
            .unwrap();
        assert_eq!(result, Number::Rational(Rational64::new(4, 9)));
    }

    #[test]
    fn negative_integer_exponent_returns_reciprocal() {
        let result = Number::Integer(2).pow(Number::Integer(-3)).unwrap();
        assert_eq!(result, Number::Rational(Rational64::new(1, 8)));
    }

    #[test]
    fn zero_to_negative_power_errors() {
        let err = Number::Integer(0).pow(Number::Integer(-1)).unwrap_err();
        assert_eq!(err, "0 cannot be raised to a negative power");
    }

    #[test]
    fn negative_base_fractional_power_errors() {
        let err = Number::Integer(-4)
            .pow(Number::Rational(Rational64::new(1, 2)))
            .unwrap_err();
        assert_eq!(
            err,
            "fractional powers of negative numbers are not supported in real arithmetic"
        );
    }

    #[test]
    fn fractional_exponents_fall_back_to_float() {
        let result = Number::Integer(9)
            .pow(Number::Rational(Rational64::new(1, 2)))
            .unwrap();
        assert_eq!(result, Number::Float(3.0));
    }

    #[test]
    fn square_root_works() {
        let result = Number::Integer(9).root(2).unwrap();
        assert_eq!(result, Number::Integer(3));
    }

    #[test]
    fn sqrt_delegates_to_second_root() {
        let result = Number::Integer(16).sqrt().unwrap();
        assert_eq!(result, Number::Integer(4));
    }

    #[test]
    fn cube_root_of_negative_number_works() {
        let result = Number::Integer(-8).root(3).unwrap();
        assert_eq!(result, Number::Integer(-2));
    }

    #[test]
    fn even_root_of_negative_number_errors() {
        let err = Number::Integer(-8).root(2).unwrap_err();
        assert_eq!(
            err,
            "even roots of negative numbers are not supported in real arithmetic"
        );
    }

    #[test]
    fn zeroth_root_errors() {
        let err = Number::Integer(8).root(0).unwrap_err();
        assert_eq!(err, "cannot take zeroth root");
    }

    #[test]
    fn negative_root_returns_reciprocal() {
        let result = Number::Integer(9).root(-2).unwrap();
        assert_eq!(result, Number::Rational(Rational64::new(1, 3)));
    }

    #[test]
    fn non_perfect_integer_root_falls_back_to_float() {
        let result = Number::Integer(2).root(2).unwrap();
        assert!(matches!(result, Number::Float(x) if x > 1.41 && x < 1.42));
    }

    #[test]
    fn exact_rational_root_returns_rational() {
        let result = Number::Rational(Rational64::new(1, 16)).root(2).unwrap();
        assert_eq!(result, Number::Rational(Rational64::new(1, 4)));
    }

    #[test]
    fn negative_exact_rational_root_returns_exact_reciprocal() {
        let result = Number::Rational(Rational64::new(1, 16)).root(-2).unwrap();
        assert_eq!(result, Number::Integer(4));
    }

    #[test]
    fn exact_integer_log_returns_integer() {
        let result = Number::Integer(8).log(Number::Integer(2)).unwrap();
        assert_eq!(result, Number::Integer(3));
    }

    #[test]
    fn exact_integer_log_can_return_rational() {
        let result = Number::Integer(2).log(Number::Integer(8)).unwrap();
        assert_eq!(result, Number::Rational(Rational64::new(1, 3)));
    }

    #[test]
    fn exact_rational_log_can_return_negative_integer() {
        let result = Number::Rational(Rational64::new(1, 8))
            .log(Number::Integer(2))
            .unwrap();
        assert_eq!(result, Number::Integer(-3));
    }

    #[test]
    fn non_exact_log_falls_back_to_float() {
        let result = Number::Integer(3).log(Number::Integer(2)).unwrap();
        assert!(matches!(result, Number::Float(x) if x > 1.58 && x < 1.59));
    }

    #[test]
    fn log_of_one_is_exact_zero() {
        let result = Number::Integer(1).log(Number::Float(10.0)).unwrap();
        assert_eq!(result, Number::Integer(0));
    }

    #[test]
    fn log_rejects_non_positive_values() {
        let err = Number::Integer(0).log(Number::Integer(2)).unwrap_err();
        assert_eq!(err, "logarithm is undefined for non-positive values");
    }

    #[test]
    fn log_rejects_invalid_bases() {
        let err = Number::Integer(4).log(Number::Integer(1)).unwrap_err();
        assert_eq!(err, "logarithm base must be positive and not equal to 1");
    }

    #[test]
    fn ln_of_one_is_exact_zero() {
        let result = Number::Integer(1).ln().unwrap();
        assert_eq!(result, Number::Integer(0));
    }

    #[test]
    fn ln_uses_float_for_non_exact_results() {
        let result = Number::Float(E.powi(2)).ln().unwrap();
        assert!(matches!(result, Number::Float(x) if x > 1.99 && x < 2.01));
    }
}
