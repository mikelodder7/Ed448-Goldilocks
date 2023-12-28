use std::{
    borrow::Borrow,
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{curve::scalar_mul::double_and_add, Scalar};

use super::DecafPoint;

/// Scalar Mul Operations
impl<'s, 'p> Mul<&'s Scalar> for &'p DecafPoint {
    type Output = DecafPoint;
    fn mul(self, scalar: &'s Scalar) -> DecafPoint {
        // XXX: We can do better than double and add
        DecafPoint(double_and_add(&self.0, &scalar))
    }
}
impl<'p, 's> Mul<&'p DecafPoint> for &'s Scalar {
    type Output = DecafPoint;
    fn mul(self, point: &'p DecafPoint) -> DecafPoint {
        DecafPoint(double_and_add(&point.0, self))
    }
}
impl Mul<DecafPoint> for Scalar {
    type Output = DecafPoint;
    fn mul(self, point: DecafPoint) -> DecafPoint {
        DecafPoint(double_and_add(&point.0, &self))
    }
}
impl Mul<Scalar> for DecafPoint {
    type Output = DecafPoint;
    fn mul(self, scalar: Scalar) -> DecafPoint {
        DecafPoint(double_and_add(&self.0, &scalar))
    }
}
impl<'s> MulAssign<&'s Scalar> for DecafPoint {
    fn mul_assign(&mut self, scalar: &'s Scalar) {
        *self = &*self * scalar;
    }
}
impl MulAssign<Scalar> for DecafPoint {
    fn mul_assign(&mut self, scalar: Scalar) {
        *self = &*self * &scalar;
    }
}

// Point addition

impl<'a, 'b> Add<&'a DecafPoint> for &'b DecafPoint {
    type Output = DecafPoint;
    fn add(self, other: &'a DecafPoint) -> DecafPoint {
        DecafPoint(self.0.to_extensible().add_extended(&other.0).to_extended())
    }
}
impl Add<DecafPoint> for &DecafPoint {
    type Output = DecafPoint;
    fn add(self, other: DecafPoint) -> DecafPoint {
        self + &other
    }
}
impl Add<&DecafPoint> for DecafPoint {
    type Output = DecafPoint;
    fn add(self, other: &DecafPoint) -> DecafPoint {
        &self + other
    }
}
impl Add<DecafPoint> for DecafPoint {
    type Output = DecafPoint;
    fn add(self, other: DecafPoint) -> DecafPoint {
        &self + &other
    }
}
impl AddAssign<&DecafPoint> for DecafPoint {
    fn add_assign(&mut self, other: &DecafPoint) {
        *self = &*self + other;
    }
}
impl AddAssign for DecafPoint {
    fn add_assign(&mut self, other: DecafPoint) {
        *self = &*self + &other;
    }
}

// Point Subtraction

impl<'a, 'b> Sub<&'a DecafPoint> for &'b DecafPoint {
    type Output = DecafPoint;
    fn sub(self, other: &'a DecafPoint) -> DecafPoint {
        DecafPoint(self.0.to_extensible().sub_extended(&other.0).to_extended())
    }
}
impl Sub<DecafPoint> for &DecafPoint {
    type Output = DecafPoint;
    fn sub(self, other: DecafPoint) -> DecafPoint {
        self - &other
    }
}
impl Sub<&DecafPoint> for DecafPoint {
    type Output = DecafPoint;
    fn sub(self, other: &DecafPoint) -> DecafPoint {
        &self - other
    }
}
impl Sub<DecafPoint> for DecafPoint {
    type Output = DecafPoint;
    fn sub(self, other: DecafPoint) -> DecafPoint {
        &self - &other
    }
}
impl SubAssign<&DecafPoint> for DecafPoint {
    fn sub_assign(&mut self, other: &DecafPoint) {
        *self = &*self - other;
    }
}
impl SubAssign for DecafPoint {
    fn sub_assign(&mut self, other: DecafPoint) {
        *self = &*self - &other;
    }
}

// Point Negation

impl<'b> Neg for &'b DecafPoint {
    type Output = DecafPoint;
    fn neg(self) -> DecafPoint {
        DecafPoint(self.0.negate())
    }
}
impl Neg for DecafPoint {
    type Output = DecafPoint;
    fn neg(self) -> DecafPoint {
        (&self).neg()
    }
}

impl<T> Sum<T> for DecafPoint
where
    T: Borrow<DecafPoint>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        iter.fold(Self::IDENTITY, |acc, item| acc + item.borrow())
    }
}
