//! Much of this code is borrowed from Thomas Pornin's [CRRL Project](https://github.com/pornin/crrl/blob/main/src/ed448.rs)
//! and adapted to mirror `ed25519-dalek`'s API.

use crate::curve::edwards::extended::PointBytes;
use crate::sign::expanded::ExpandedSecretKey;
use crate::{
    Context, Scalar, ScalarBytes, Signature, VerifyingKey, PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH,
};
use core::fmt::{self, Debug, Formatter};
use sha3::{
    digest::{
        consts::U64, crypto_common::BlockSizeUser, typenum::IsEqual, ExtendableOutput, FixedOutput,
        FixedOutputReset, HashMarker, Update, XofReader,
    },
    Digest,
};
use signature::Error;
use subtle::{Choice, ConstantTimeEq};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Ed448 secret key as defined in [RFC8032 § 5.2.5]
///
/// The private key is 57 octets (448 bits, 56 bytes) long.
pub type SecretKey = ScalarBytes;

/// Signing hash trait for Ed448ph
pub trait SigningHash {
    fn fill_bytes(&mut self, out: &mut [u8]);
}

/// Signing pre-hasher for Ed448ph with a fixed output size
#[derive(Debug)]
pub struct SigningPreHasherXmd<HashT>
where
    HashT: BlockSizeUser + Default + FixedOutput + FixedOutputReset + Update + HashMarker,
    HashT::OutputSize: IsEqual<U64>,
{
    hasher: HashT,
}

impl<HashT> From<HashT> for SigningPreHasherXmd<HashT>
where
    HashT: BlockSizeUser + Default + FixedOutput + FixedOutputReset + Update + HashMarker,
    HashT::OutputSize: IsEqual<U64>,
{
    fn from(hasher: HashT) -> Self {
        Self::new(hasher)
    }
}

impl<HashT> SigningPreHasherXmd<HashT>
where
    HashT: BlockSizeUser + Default + FixedOutput + FixedOutputReset + Update + HashMarker,
    HashT::OutputSize: IsEqual<U64>,
{
    pub fn new(hasher: HashT) -> Self {
        Self { hasher }
    }
}

impl<HashT> SigningHash for SigningPreHasherXmd<HashT>
where
    HashT: BlockSizeUser + Default + FixedOutput + FixedOutputReset + Update + HashMarker,
    HashT::OutputSize: IsEqual<U64>,
{
    fn fill_bytes(&mut self, out: &mut [u8]) {
        out.copy_from_slice(self.hasher.finalize_reset().as_slice());
    }
}

/// Signing pre-hasher for Ed448ph with a xof output
pub struct SigningPreHasherXof<HashT>
where
    HashT: Default + ExtendableOutput + Update,
{
    reader: <HashT as ExtendableOutput>::Reader,
}

impl<HashT> Debug for SigningPreHasherXof<HashT>
where
    HashT: Default + ExtendableOutput + Update,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigningPreHasherXof")
            .finish_non_exhaustive()
    }
}

impl<HashT> SigningHash for SigningPreHasherXof<HashT>
where
    HashT: Default + ExtendableOutput + Update,
{
    fn fill_bytes(&mut self, out: &mut [u8]) {
        self.reader.read(out);
    }
}

impl<HashT> From<HashT> for SigningPreHasherXof<HashT>
where
    HashT: Default + ExtendableOutput + Update,
{
    fn from(hasher: HashT) -> Self {
        Self::new(hasher)
    }
}

impl<HashT> SigningPreHasherXof<HashT>
where
    HashT: Default + ExtendableOutput + Update,
{
    pub fn new(hasher: HashT) -> Self {
        Self {
            reader: hasher.finalize_xof(),
        }
    }
}

/// Signing key for Ed448
#[derive(Clone)]
pub struct SigningKey {
    pub(crate) secret: ExpandedSecretKey,
}

impl Debug for SigningKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigningKey")
            .field("verifying_key", &self.secret.public_key)
            .finish_non_exhaustive()
    }
}

impl Zeroize for SigningKey {
    fn zeroize(&mut self) {
        self.secret.zeroize();
    }
}

impl Drop for SigningKey {
    fn drop(&mut self) {
        self.secret.zeroize();
    }
}

impl ZeroizeOnDrop for SigningKey {}

impl ConstantTimeEq for SigningKey {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.secret.seed.ct_eq(&other.secret.seed)
    }
}

impl Eq for SigningKey {}

impl PartialEq for SigningKey {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl From<SecretKey> for SigningKey {
    fn from(secret_scalar: SecretKey) -> Self {
        Self::from(&secret_scalar)
    }
}

impl From<&SecretKey> for SigningKey {
    fn from(secret_scalar: &SecretKey) -> Self {
        Self {
            secret: ExpandedSecretKey::from(secret_scalar),
        }
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl TryFrom<Vec<u8>> for SigningKey {
    type Error = &'static str;

    fn try_from(value: Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_slice())
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl TryFrom<&Vec<u8>> for SigningKey {
    type Error = &'static str;

    fn try_from(value: &Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_slice())
    }
}

impl TryFrom<&[u8]> for SigningKey {
    type Error = &'static str;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        if value.len() != SECRET_KEY_LENGTH {
            return Err("Invalid length for a signing key");
        }
        Ok(Self::from(ScalarBytes::from_slice(value)))
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl TryFrom<Box<[u8]>> for SigningKey {
    type Error = &'static str;

    fn try_from(value: Box<[u8]>) -> Result<Self, Self::Error> {
        Self::try_from(value.as_ref())
    }
}

impl<D> signature::DigestSigner<D, Signature> for SigningKey
where
    D: Digest,
{
    fn try_sign_digest(&self, digest: D) -> Result<Signature, Error> {
        let mut prehashed_message = [0u8; 64];
        prehashed_message.copy_from_slice(digest.finalize().as_slice());
        let sig = self.secret.sign_prehashed(&[], &prehashed_message)?;
        Ok(sig.into())
    }
}

impl signature::hazmat::PrehashSigner<Signature> for SigningKey {
    fn sign_prehash(&self, prehash: &[u8]) -> Result<Signature, Error> {
        let sig = self.secret.sign_prehashed(&[], prehash)?;
        Ok(sig.into())
    }
}

impl signature::Signer<Signature> for SigningKey {
    fn try_sign(&self, msg: &[u8]) -> Result<Signature, Error> {
        let sig = self.secret.sign_raw(msg)?;
        Ok(sig.into())
    }
}

impl<D> signature::DigestSigner<D, Signature> for Context<'_, '_, SigningKey>
where
    D: Digest,
{
    fn try_sign_digest(&self, digest: D) -> Result<Signature, Error> {
        let mut prehashed_message = [0u8; 64];
        prehashed_message.copy_from_slice(digest.finalize().as_slice());
        let sig = self
            .key
            .secret
            .sign_prehashed(self.value, &prehashed_message)?;
        Ok(sig.into())
    }
}

impl signature::hazmat::PrehashSigner<Signature> for Context<'_, '_, SigningKey> {
    fn sign_prehash(&self, prehash: &[u8]) -> Result<Signature, Error> {
        let sig = self.key.secret.sign_prehashed(self.value, prehash)?;
        Ok(sig.into())
    }
}

impl signature::Signer<Signature> for Context<'_, '_, SigningKey> {
    fn try_sign(&self, msg: &[u8]) -> Result<Signature, Error> {
        let sig = self.key.secret.sign_ctx(self.value, msg)?;
        Ok(sig.into())
    }
}

impl<D> signature::DigestVerifier<D, Signature> for SigningKey
where
    D: Digest,
{
    fn verify_digest(&self, msg: D, signature: &Signature) -> Result<(), Error> {
        <VerifyingKey as signature::DigestVerifier<D, Signature>>::verify_digest(
            &self.secret.public_key,
            msg,
            signature,
        )
    }
}

impl signature::Verifier<Signature> for SigningKey {
    fn verify(&self, msg: &[u8], signature: &Signature) -> Result<(), Error> {
        self.secret.public_key.verify_raw(signature, msg)
    }
}

#[cfg(feature = "pkcs8")]
/// The OID for Ed448 as defined in [RFC8410 §2]
pub const ALGORITHM_OID: pkcs8::ObjectIdentifier =
    pkcs8::ObjectIdentifier::new_unwrap("1.3.101.113");

#[cfg(feature = "pkcs8")]
pub const ALGORITHM_ID: pkcs8::AlgorithmIdentifierRef<'static> = pkcs8::AlgorithmIdentifierRef {
    oid: ALGORITHM_OID,
    parameters: None,
};

#[cfg(all(any(feature = "alloc", feature = "std"), feature = "pkcs8"))]
impl pkcs8::EncodePrivateKey for SigningKey {
    fn to_pkcs8_der(&self) -> pkcs8::Result<pkcs8::SecretDocument> {
        KeypairBytes::from(self).to_pkcs8_der()
    }
}

#[cfg(all(any(feature = "alloc", feature = "std"), feature = "pkcs8"))]
impl pkcs8::spki::DynSignatureAlgorithmIdentifier for SigningKey {
    fn signature_algorithm_identifier(
        &self,
    ) -> pkcs8::spki::Result<pkcs8::spki::AlgorithmIdentifierOwned> {
        // See https://datatracker.ietf.org/doc/html/rfc8410 for id-Ed448
        Ok(pkcs8::spki::AlgorithmIdentifier {
            oid: ALGORITHM_OID,
            parameters: None,
        })
    }
}

#[cfg(feature = "pkcs8")]
pub struct KeypairBytes {
    pub secret_key: PointBytes,
    pub verifying_key: Option<PointBytes>,
}

#[cfg(all(any(feature = "alloc", feature = "std"), feature = "pkcs8"))]
impl pkcs8::EncodePrivateKey for KeypairBytes {
    fn to_pkcs8_der(&self) -> pkcs8::Result<pkcs8::SecretDocument> {
        let mut private_key = [0u8; 2 + SECRET_KEY_LENGTH];
        private_key[0] = 0x04;
        private_key[1] = SECRET_KEY_LENGTH as u8;
        private_key[2..].copy_from_slice(self.secret_key.as_ref());

        let private_key_info = pkcs8::PrivateKeyInfo {
            algorithm: ALGORITHM_ID,
            private_key: &private_key,
            public_key: self.verifying_key.as_ref().map(|v| v.as_ref()),
        };
        let result = pkcs8::SecretDocument::encode_msg(&private_key_info)?;

        #[cfg(feature = "zeroize")]
        private_key.zeroize();

        Ok(result)
    }
}

#[cfg(feature = "pkcs8")]
impl TryFrom<pkcs8::PrivateKeyInfo<'_>> for KeypairBytes {
    type Error = pkcs8::Error;

    fn try_from(value: pkcs8::PrivateKeyInfo<'_>) -> Result<Self, Self::Error> {
        if value.algorithm.oid != ALGORITHM_OID {
            return Err(pkcs8::Error::KeyMalformed);
        }
        if value.private_key.len() != SECRET_KEY_LENGTH {
            return Err(pkcs8::Error::KeyMalformed);
        }
        let secret_key = PointBytes::from(value.private_key);
        let verifying_key = if let Some(public_key) = value.public_key {
            if public_key.len() != PUBLIC_KEY_LENGTH {
                return Err(pkcs8::Error::KeyMalformed);
            }
            Some(PointBytes::from(public_key))
        } else {
            None
        };
        Ok(KeypairBytes {
            secret_key,
            verifying_key,
        })
    }
}

#[cfg(feature = "pkcs8")]
impl TryFrom<KeypairBytes> for SigningKey {
    type Error = pkcs8::Error;

    fn try_from(value: KeypairBytes) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

#[cfg(feature = "pkcs8")]
impl TryFrom<&KeypairBytes> for SigningKey {
    type Error = pkcs8::Error;

    fn try_from(value: &KeypairBytes) -> Result<Self, Self::Error> {
        let signing_key = SigningKey::from(SecretKey::from_slice(value.secret_key.as_ref()));

        if let Some(public_bytes) = value.verifying_key {
            let verifying_key =
                VerifyingKey::from_bytes(&public_bytes).map_err(|_| pkcs8::Error::KeyMalformed)?;
            if !signing_key.verifying_key() != &verifying_key {
                return Err(pkcs8::Error::KeyMalformed);
            }
        }
        Ok(signing_key)
    }
}

#[cfg(feature = "pkcs8")]
impl From<&SigningKey> for KeypairBytes {
    fn from(signing_key: &SigningKey) -> Self {
        KeypairBytes {
            secret_key: PointBytes::from(signing_key.to_bytes()),
            verifying_key: Some(PointBytes::from(signing_key.verifying_key().to_bytes())),
        }
    }
}

#[cfg(feature = "pkcs8")]
impl TryFrom<pkcs8::PrivateKeyInfo<'_>> for SigningKey {
    type Error = pkcs8::Error;

    fn try_from(value: pkcs8::PrivateKeyInfo<'_>) -> Result<Self, Self::Error> {
        KeypairBytes::try_from(value)?.try_into()
    }
}

#[cfg(feature = "serde")]
impl serdect::serde::Serialize for SigningKey {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serdect::serde::Serializer,
    {
        serdect::array::serialize_hex_lower_or_bin(self.secret.seed.as_ref(), s)
    }
}

#[cfg(feature = "serde")]
impl<'de> serdect::serde::Deserialize<'de> for SigningKey {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serdect::serde::Deserializer<'de>,
    {
        let mut bytes = SecretKey::default();
        serdect::array::deserialize_hex_or_bin(&mut bytes, d)?;
        Ok(SigningKey::from(bytes))
    }
}

impl SigningKey {
    /// Generate a cryptographically random [`SigningKey`].
    pub fn generate(mut rng: impl rand_core::RngCore) -> Self {
        let mut secret_scalar = SecretKey::default();
        rng.fill_bytes(secret_scalar.as_mut());
        assert!(!secret_scalar.iter().all(|&v| v == 0));
        Self {
            secret: ExpandedSecretKey::from(&secret_scalar),
        }
    }

    /// Serialize this [`SigningKey`] as bytes.
    pub fn to_bytes(&self) -> SecretKey {
        self.secret.seed
    }

    /// Serialize this [`SigningKey`] as a byte reference.
    pub fn as_bytes(&self) -> &SecretKey {
        &self.secret.seed
    }

    /// Return the clamped [`Scalar`] for this [`SigningKey`].
    ///
    /// This is the scalar that is actually used for signing.
    /// Be warned, this is secret material that should be handled with care.
    pub fn to_scalar(&self) -> Scalar {
        self.secret.scalar
    }

    /// Get the [`VerifyingKey`] for this [`SigningKey`].
    pub fn verifying_key(&self) -> VerifyingKey {
        self.secret.public_key
    }

    /// Create a signing context that can be used for Ed448ph with
    /// [`DigestSigner`]
    pub fn with_context<'k, 'v>(&'k self, context: &'v [u8]) -> Context<'k, 'v, Self> {
        Context {
            key: self,
            value: context,
        }
    }

    /// Sign a `message` with this [`SigningKey`] using the Ed448 algorithm
    /// defined in [RFC8032 §5.2][rfc8032].
    pub fn sign_raw(&self, message: &[u8]) -> Result<Signature, Error> {
        let sig = self.secret.sign_raw(message)?;
        Ok(sig.into())
    }

    /// Sign a `message` in the given `context` with this [`SigningKey`] using the Ed448ph algorithm
    /// defined in [RFC8032 §5.2][rfc8032].
    pub fn sign_ctx(&self, context: &[u8], message: &[u8]) -> Result<Signature, Error> {
        let sig = self.secret.sign_ctx(context, message)?;
        Ok(sig.into())
    }

    /// Sign a `prehashed_message` with this [`SigningKey`] using the
    /// Ed448ph algorithm defined in [RFC8032 §5.2][rfc8032].
    pub fn sign_prehashed<D>(
        &self,
        mut prehashed_message: D,
        context: Option<&[u8]>,
    ) -> Result<Signature, Error>
    where
        D: SigningHash,
    {
        let mut m = [0u8; 64];
        prehashed_message.fill_bytes(&mut m);
        let sig = self
            .secret
            .sign_prehashed(context.unwrap_or_default(), &m)?;
        Ok(sig.into())
    }
}
