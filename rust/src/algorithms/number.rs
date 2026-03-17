use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

use num_rational::Rational64;
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Number {
    Float(f64),
    Integer(i64),
    Rational(Rational64),
}

impl Number {
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

    pub fn one() -> Number {
        Number::Integer(1)
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
}
