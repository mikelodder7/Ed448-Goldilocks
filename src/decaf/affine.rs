use crate::curve::twedwards::affine::AffinePoint as InnerAffinePoint;
use crate::field::FieldElement;
use crate::{
    decaf::points::DecafPointRepr, CompressedDecaf, Decaf448FieldBytes, DecafPoint, Scalar,
};
use elliptic_curve::{
    common::Generate,
    ctutils::{CtEq, CtSelect},
    group::{Curve, CurveAffine, GroupEncoding},
    ops::MulVartime,
    point::NonIdentity,
    Error,
};
use rand_core::TryCryptoRng;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

#[cfg(feature = "zeroize")]
use zeroize::DefaultIsZeroes;

/// Affine point on the twisted curve
#[derive(Copy, Clone, Debug, Default)]
pub struct AffinePoint(pub(crate) InnerAffinePoint);

impl ConstantTimeEq for AffinePoint {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.x.ct_eq(&other.0.x) & self.0.y.ct_eq(&other.0.y)
    }
}

impl CtEq for AffinePoint {
    fn ct_eq(&self, other: &Self) -> elliptic_curve::ctutils::Choice {
        ConstantTimeEq::ct_eq(self, other).into()
    }
}

impl ConditionallySelectable for AffinePoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(InnerAffinePoint {
            x: FieldElement::conditional_select(&a.0.x, &b.0.x, choice),
            y: FieldElement::conditional_select(&a.0.y, &b.0.y, choice),
        })
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
    type Repr = DecafPointRepr;

    fn from_bytes(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        CompressedDecaf(*(bytes.as_ref()))
            .decompress()
            .map(|point| point.to_affine())
    }

    fn from_bytes_unchecked(bytes: &Self::Repr) -> subtle::CtOption<Self> {
        Self::from_bytes(bytes)
    }

    fn to_bytes(&self) -> Self::Repr {
        DecafPointRepr::from(self.to_decaf().compress().0)
    }
}

impl CurveAffine for AffinePoint {
    type Curve = DecafPoint;
    type Scalar = Scalar;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn generator() -> Self {
        DecafPoint::GENERATOR.to_affine()
    }

    fn is_identity(&self) -> Choice {
        ConstantTimeEq::ct_eq(self, &Self::IDENTITY)
    }

    fn to_curve(&self) -> Self::Curve {
        self.to_decaf()
    }
}

impl Generate for AffinePoint {
    fn try_generate_from_rng<R: TryCryptoRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        DecafPoint::try_generate_from_rng(rng).map(|point| point.to_affine())
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
    type FieldRepr = Decaf448FieldBytes;

    fn from_coordinates(x: &Self::FieldRepr, y: &Self::FieldRepr) -> subtle::CtOption<Self> {
        let mut x_bytes = [0u8; 56];
        x_bytes.copy_from_slice(&x[..56]);
        let mut y_bytes = [0u8; 56];
        y_bytes.copy_from_slice(&y[..56]);
        let point = Self(InnerAffinePoint {
            x: FieldElement::from_bytes(&x_bytes),
            y: FieldElement::from_bytes(&y_bytes),
        });
        let xx = point.0.x.square();
        let yy = point.0.y.square();
        let is_on_curve =
            (yy - xx).ct_eq(&(FieldElement::ONE + (FieldElement::TWISTED_D * xx * yy)));
        subtle::CtOption::new(point, is_on_curve)
    }

    fn x(&self) -> Self::FieldRepr {
        let mut repr = Decaf448FieldBytes::default();
        repr[..56].copy_from_slice(&self.x());
        repr
    }

    fn y(&self) -> Self::FieldRepr {
        let mut repr = Decaf448FieldBytes::default();
        repr[..56].copy_from_slice(&self.y());
        repr
    }

    fn x_is_odd(&self) -> Choice {
        self.0.x.is_negative()
    }

    fn y_is_odd(&self) -> Choice {
        self.0.y.is_negative()
    }
}

#[cfg(feature = "zeroize")]
impl DefaultIsZeroes for AffinePoint {}

impl AffinePoint {
    /// The identity point
    pub const IDENTITY: Self = Self(InnerAffinePoint::IDENTITY);

    /// Convert to DecafPoint
    pub fn to_decaf(&self) -> DecafPoint {
        DecafPoint(self.0.to_extended())
    }

    /// The X coordinate
    pub fn x(&self) -> [u8; 56] {
        self.0.x.to_bytes()
    }

    /// The Y coordinate
    pub fn y(&self) -> [u8; 56] {
        self.0.y.to_bytes()
    }
}
