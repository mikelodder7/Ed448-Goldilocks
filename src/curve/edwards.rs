/// This module contains the code for the Goldilocks curve.
/// The Goldilocks curve is the (untwisted) Edwards curve with affine equation x^2 + y^2 = 1 - 39081x^2y^2.
/// Scalar multiplication for this curve is predominantly delegated to the twisted Edwards variant using a (doubling) isogeny.
/// Passing the point back to the Goldilocks curve using the dual-isogeny clears the cofactor.
/// The small remainder of the scalar multiplication is computed on the untwisted curve.
/// See <https://www.shiftleft.org/papers/isogeny/isogeny.pdf> for details.
///
/// This isogeny strategy does not clear the cofactor on the Goldilocks curve unless the scalar is a multiple of 4
/// or the point is known to be in the q-torsion subgroup.
/// Hence, one will need to multiply by the cofactor to ensure it is cleared when using the Goldilocks curve.
/// If this is a problem, one can use a different isogeny strategy (Decaf/Ristretto).
pub(crate) mod affine;
pub(crate) mod extended;
pub use affine::AffinePoint;
pub use extended::{CompressedEdwardsY, EdwardsPoint};
