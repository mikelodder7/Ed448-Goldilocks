#![allow(non_snake_case)]

use super::double_and_add;
use crate::curve::twedwards::extended::ExtendedPoint;
use crate::field::Scalar;
/// XXX: This is a very inefficient way to perform double-base scalar multiplication.
/// Replace it with Pornin's endomorphism or use NAF form.
/// Computes `aA + bB`, where `B` is the twisted Edwards basepoint.
pub(crate) fn double_base_scalar_mul(a: &Scalar, A: &ExtendedPoint, b: &Scalar) -> ExtendedPoint {
    let part_a = double_and_add(A, a);
    let part_b = double_and_add(&ExtendedPoint::GENERATOR, b);
    part_a.add(&part_b)
}
