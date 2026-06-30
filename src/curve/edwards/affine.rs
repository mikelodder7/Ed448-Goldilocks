use crate::curve::edwards::EdwardsPoint;
use crate::field::FieldElement;
use crate::*;
use elliptic_curve::{
    Error,
    array::{Array as GenericArray, typenum::U57},
    common::Generate,
    ctutils::{CtEq, CtSelect},
    group::{CurveAffine, GroupEncoding},
    ops::MulVartime,
    point::NonIdentity,
};
use rand_core::TryCryptoRng;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

#[cfg(feature = "zeroize")]
use zeroize::DefaultIsZeroes;

/// Affine point on untwisted curve
#[derive(Copy, Clone, Debug)]
pub struct AffinePoint {
    pub(crate) x: FieldElement,
    pub(crate) y: FieldElement,
}

impl Default for AffinePoint {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ConstantTimeEq for AffinePoint {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.x.ct_eq(&other.x) & self.y.ct_eq(&other.y)
    }
}

impl CtEq for AffinePoint {
    fn ct_eq(&self, other: &Self) -> elliptic_curve::ctutils::Choice {
        ConstantTimeEq::ct_eq(self, other).into()
    }
}

impl ConditionallySelectable for AffinePoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self {
            x: FieldElement::conditional_select(&a.x, &b.x, choice),
            y: FieldElement::conditional_select(&a.y, &b.y, choice),
        }
    }
}

impl CtSelect for AffinePoint {
    fn ct_select(&self, other: &Self, choice: elliptic_curve::ctutils::Choice) -> Self {
        Self::conditional_select(self, other, choice.into())
    }
}

impl PartialEq for AffinePoint {
    fn eq(&self, other: &Self) -> bool {
        ConstantTimeEq::ct_eq(self, other).into()
    }
}

impl Eq for AffinePoint {}

impl GroupEncoding for AffinePoint {
    type Repr = GenericArray<u8, U57>;

    fn from_bytes(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        let mut value = [0u8; 57];
        value.copy_from_slice(bytes);
        CompressedEdwardsY(value)
            .decompress()
            .map(|point| point.to_affine())
    }

    fn from_bytes_unchecked(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        Self::from_bytes(bytes)
    }

    fn to_bytes(&self) -> Self::Repr {
        Self::Repr::from(self.to_edwards().compress().0)
    }
}

impl CurveAffine for AffinePoint {
    type Curve = EdwardsPoint;
    type Scalar = Scalar;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn generator() -> Self {
        EdwardsPoint::GENERATOR.to_affine()
    }

    fn is_identity(&self) -> Choice {
        ConstantTimeEq::ct_eq(self, &Self::IDENTITY)
    }

    fn to_curve(&self) -> Self::Curve {
        self.to_edwards()
    }
}

impl Generate for AffinePoint {
    fn try_generate_from_rng<R: TryCryptoRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        EdwardsPoint::try_generate_from_rng(rng).map(|point| point.to_affine())
    }
}

impl From<NonIdentity<AffinePoint>> for AffinePoint {
    fn from(point: NonIdentity<AffinePoint>) -> Self {
        point.to_point()
    }
}

impl TryFrom<AffinePoint> for NonIdentity<AffinePoint> {
    type Error = Error;

    fn try_from(point: AffinePoint) -> Result<Self, Self::Error> {
        Option::from(NonIdentity::new(point)).ok_or(Error)
    }
}

impl MulVartime<Scalar> for AffinePoint {
    fn mul_vartime(self, rhs: Scalar) -> Self::Output {
        self * rhs
    }
}

impl<'a> MulVartime<&'a Scalar> for AffinePoint {
    fn mul_vartime(self, rhs: &'a Scalar) -> Self::Output {
        self * rhs
    }
}

impl elliptic_curve::point::AffineCoordinates for AffinePoint {
    type FieldRepr = Ed448FieldBytes;

    fn from_coordinates(x: &Self::FieldRepr, y: &Self::FieldRepr) -> subtle::CtOption<Self> {
        let mut x_bytes = [0u8; 56];
        x_bytes.copy_from_slice(&x[..56]);
        let mut y_bytes = [0u8; 56];
        y_bytes.copy_from_slice(&y[..56]);
        let point = Self {
            x: FieldElement::from_bytes(&x_bytes),
            y: FieldElement::from_bytes(&y_bytes),
        };
        subtle::CtOption::new(point, point.to_edwards().is_on_curve())
    }

    fn x(&self) -> Self::FieldRepr {
        let mut repr = Ed448FieldBytes::default();
        repr[..56].copy_from_slice(&self.x.to_bytes());
        repr
    }

    fn y(&self) -> Self::FieldRepr {
        let mut repr = Ed448FieldBytes::default();
        repr[..56].copy_from_slice(&self.y.to_bytes());
        repr
    }

    fn x_is_odd(&self) -> Choice {
        self.x.is_negative()
    }

    fn y_is_odd(&self) -> Choice {
        self.y.is_negative()
    }
}

#[cfg(feature = "zeroize")]
impl DefaultIsZeroes for AffinePoint {}

impl AffinePoint {
    /// The identity point
    pub const IDENTITY: AffinePoint = AffinePoint {
        x: FieldElement::ZERO,
        y: FieldElement::ONE,
    };

    pub(crate) fn isogeny(&self) -> Self {
        let x = self.x;
        let y = self.y;
        let mut t0 = x.square(); // x^2
        let t1 = t0 + FieldElement::ONE; // x^2+1
        t0 -= FieldElement::ONE; // x^2-1
        let mut t2 = y.square(); // y^2
        t2 = t2.double(); // 2y^2
        let t3 = x.double(); // 2x

        let mut t4 = t0 * y; // y(x^2-1)
        t4 = t4.double(); // 2y(x^2-1)
        let xNum = t4.double(); // xNum = 4y(x^2-1)

        let mut t5 = t0.square(); // x^4-2x^2+1
        t4 = t5 + t2; // x^4-2x^2+1+2y^2
        let xDen = t4 + t2; // xDen = x^4-2x^2+1+4y^2

        t5 *= x; // x^5-2x^3+x
        t4 = t2 * t3; // 4xy^2
        let yNum = t4 - t5; // yNum = -(x^5-2x^3+x-4xy^2)

        t4 = t1 * t2; // 2x^2y^2+2y^2
        let yDen = t5 - t4; // yDen = x^5-2x^3+x-2x^2y^2-2y^2

        Self {
            x: xNum * xDen.invert(),
            y: yNum * yDen.invert(),
        }
    }

    /// Convert to edwards extended point
    pub fn to_edwards(&self) -> EdwardsPoint {
        EdwardsPoint {
            X: self.x,
            Y: self.y,
            Z: FieldElement::ONE,
            T: self.x * self.y,
        }
    }

    /// The X coordinate
    pub fn x(&self) -> [u8; 56] {
        self.x.to_bytes()
    }

    /// The Y coordinate
    pub fn y(&self) -> [u8; 56] {
        self.y.to_bytes()
    }
}
