mod context;
mod error;
mod expanded;
mod signature;
mod signing_key;
mod verifying_key;

pub use context::*;
pub use error::*;
pub use signature::*;
pub use signing_key::*;
pub use verifying_key::*;

/// Length of a secret key in bytes
pub const SECRET_KEY_LENGTH: usize = 57;

/// Length of a public key in bytes
pub const PUBLIC_KEY_LENGTH: usize = 57;

/// Length of a signature in bytes
pub const SIGNATURE_LENGTH: usize = 114;

/// Constant string "SigEd448".
pub(crate) const HASH_HEAD: [u8; 8] = [0x53, 0x69, 0x67, 0x45, 0x64, 0x34, 0x34, 0x38];

#[cfg(feature = "pkcs8")]
/// The OID for Ed448 as defined in [RFC8410 §2]
pub const ALGORITHM_OID: pkcs8::ObjectIdentifier =
    pkcs8::ObjectIdentifier::new_unwrap("1.3.101.113");

#[cfg(feature = "pkcs8")]
pub const ALGORITHM_ID: pkcs8::AlgorithmIdentifierRef<'static> = pkcs8::AlgorithmIdentifierRef {
    oid: ALGORITHM_OID,
    parameters: None,
};
