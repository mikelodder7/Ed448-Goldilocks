use crate::curve::edwards::EdwardsPoint;
use crate::field::FieldElement;
// Affine point on untwisted curve
// XXX: This is only really needed for convenience in extended.rs . Will remove it sooner or later
pub struct AffinePoint {
    pub(crate) x: FieldElement,
    pub(crate) y: FieldElement,
}

impl AffinePoint {
    pub const IDENTITY: AffinePoint = AffinePoint {
        x: FieldElement::ZERO,
        y: FieldElement::ONE,
    };

    pub fn to_extended(&self) -> EdwardsPoint {
        EdwardsPoint {
            X: self.x,
            Y: self.y,
            Z: FieldElement::ONE,
            T: self.x * self.y,
        }
    }
}
