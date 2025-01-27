use crate::{
    CompressedEdwardsY, EdwardsPoint, Scalar, ScalarBytes, SigningError, SECRET_KEY_LENGTH,
    SIGNATURE_LENGTH,
};

/// Ed448 signature as defined in [RFC8032 § 5.2.5]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Signature {
    pub(crate) r: CompressedEdwardsY,
    pub(crate) s: [u8; 57],
}

impl Default for Signature {
    fn default() -> Self {
        Self {
            r: CompressedEdwardsY::default(),
            s: [0u8; 57],
        }
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl TryFrom<Vec<u8>> for Signature {
    type Error = SigningError;

    fn try_from(value: Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_slice())
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl TryFrom<&Vec<u8>> for Signature {
    type Error = SigningError;

    fn try_from(value: &Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_slice())
    }
}

impl TryFrom<&[u8]> for Signature {
    type Error = SigningError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        if value.len() != SIGNATURE_LENGTH {
            return Err(SigningError::InvalidSignatureLength);
        }

        let mut r = [0u8; SECRET_KEY_LENGTH];
        r.copy_from_slice(&value[..SECRET_KEY_LENGTH]);
        let mut s = [0u8; SECRET_KEY_LENGTH];
        s.copy_from_slice(&value[SECRET_KEY_LENGTH..]);

        Ok(Self {
            r: CompressedEdwardsY(r),
            s,
        })
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl TryFrom<Box<[u8]>> for Signature {
    type Error = SigningError;

    fn try_from(value: Box<[u8]>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_ref())
    }
}

impl Signature {
    /// Converts [`Signature`] to a byte array.
    pub fn to_bytes(&self) -> [u8; SIGNATURE_LENGTH] {
        let mut bytes = [0u8; SIGNATURE_LENGTH];
        bytes[..57].copy_from_slice(self.r.as_bytes());
        bytes[57..].copy_from_slice(&self.s);
        bytes
    }

    /// Converts a byte array to a [`Signature`].
    pub fn from_bytes(bytes: &[u8; SIGNATURE_LENGTH]) -> Self {
        let mut r = [0u8; SECRET_KEY_LENGTH];
        r.copy_from_slice(&bytes[..SECRET_KEY_LENGTH]);
        let mut s = [0u8; SECRET_KEY_LENGTH];
        s.copy_from_slice(&bytes[SECRET_KEY_LENGTH..]);
        Self {
            r: CompressedEdwardsY(r),
            s,
        }
    }

    /// The `r` value of the signature.
    pub fn r(&self) -> CompressedEdwardsY {
        self.r
    }

    /// The `s` value of the signature.
    pub fn s(&self) -> &[u8; SECRET_KEY_LENGTH] {
        &self.s
    }
}

impl From<InnerSignature> for Signature {
    fn from(inner: InnerSignature) -> Self {
        let mut s = [0u8; SECRET_KEY_LENGTH];
        s.copy_from_slice(&inner.s.to_bytes_rfc_8032());
        Self {
            r: inner.r.compress(),
            s,
        }
    }
}

impl TryFrom<Signature> for InnerSignature {
    type Error = SigningError;

    fn try_from(signature: Signature) -> Result<Self, Self::Error> {
        let s_bytes = ScalarBytes::from_slice(&signature.s);
        let s = Option::from(Scalar::from_canonical_bytes(s_bytes))
            .ok_or(SigningError::InvalidSignatureSComponent)?;
        let r = Option::from(signature.r.decompress())
            .ok_or(SigningError::InvalidSignatureRComponent)?;
        Ok(Self { r, s })
    }
}

pub(crate) struct InnerSignature {
    pub(crate) r: EdwardsPoint,
    pub(crate) s: Scalar,
}
