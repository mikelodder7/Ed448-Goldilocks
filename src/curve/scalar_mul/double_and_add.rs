use crate::curve::twedwards::extended::ExtendedPoint;
use crate::field::Scalar;
use subtle::{Choice, ConditionallySelectable};

/// Traditional double-and-add algorithm.
pub(crate) fn double_and_add(point: &ExtendedPoint, s: &Scalar) -> ExtendedPoint {
    let mut result = ExtendedPoint::IDENTITY;

    // Note: We reverse here, so we are going from MSB to LSB.
    // XXX: It would be useful if subtle had a `From<u32>` implementation for `Choice`,
    // but perhaps that is not its purpose.
    for bit in s.bits().into_iter().rev() {
        result = result.double();

        let mut p = ExtendedPoint::IDENTITY;
        p.conditional_assign(point, Choice::from(bit as u8));
        result = result.add(&p);
    }

    result
}
