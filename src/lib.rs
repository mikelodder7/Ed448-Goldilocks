//! This crate provides a pure Rust implementation of Curve448, Edwards, Decaf, and Ristretto.
//! It is intended to be portable, fast, and safe.
//!
//! # Usage
//! ```
//! use ed448_goldilocks_plus::{EdwardsPoint, CompressedEdwardsY, Scalar, elliptic_curve::hash2curve::ExpandMsgXof, sha3::Shake256};
//! use rand_core::OsRng;
//!
//! let secret_key = Scalar::TWO;
//! let public_key = EdwardsPoint::GENERATOR * &secret_key;
//!
//! assert_eq!(public_key, EdwardsPoint::GENERATOR + EdwardsPoint::GENERATOR);
//!
//! let secret_key = Scalar::random(&mut OsRng);
//! let public_key = EdwardsPoint::GENERATOR * &secret_key;
//! let compressed_public_key = public_key.compress();
//!
//! assert_eq!(compressed_public_key.to_bytes().len(), 57);
//!
//! let hashed_scalar = Scalar::hash::<ExpandMsgXof<Shake256>>(b"test", b"edwards448_XOF:SHAKE256_ELL2_RO_");
//! let input = hex_literal::hex!("c8c6c8f584e0c25efdb6af5ad234583c56dedd7c33e0c893468e96740fa0cf7f1a560667da40b7bde340a39252e89262fcf707d1180fd43400");
//! let expected_scalar = Scalar::from_canonical_bytes(&input.into()).unwrap();
//! assert_eq!(hashed_scalar, expected_scalar);
//!
//! let hashed_point = EdwardsPoint::hash::<ExpandMsgXof<Shake256>>(b"test", b"edwards448_XOF:SHAKE256_ELL2_RO_");
//! let expected = hex_literal::hex!("d15c4427b5c5611a53593c2be611fd3635b90272d331c7e6721ad3735e95dd8b9821f8e4e27501ce01aa3c913114052dce2e91e8ca050f4980");
//! let expected_point = CompressedEdwardsY(expected).decompress().unwrap();
//! assert_eq!(hashed_point, expected_point);
//!
//! let hashed_point = EdwardsPoint::hash_with_defaults(b"test");
//! assert_eq!(hashed_point, expected_point);
//! ```
//!
//! [`EdwardsPoint`] implements the [`elliptic_curve::Group`] and [`elliptic_curve::group::GroupEncoding`]
//! and [`Scalar`] implements [`elliptic_curve::Field`] and [`elliptic_curve::PrimeField`] traits.
// XXX: Change this to deny later on
#![warn(unused_attributes, unused_imports, unused_mut, unused_must_use)]
#![allow(non_snake_case)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// Internal macros. Must come first!
#[macro_use]
pub(crate) mod macros;

pub use elliptic_curve;
pub use rand_core;
pub use sha3;
pub use subtle;

// As usual, we will use this file to carefully define the API/ what we expose to the user
pub(crate) mod constants;
pub(crate) mod curve;
pub(crate) mod decaf;
pub(crate) mod field;
pub(crate) mod ristretto;

pub(crate) use field::{GOLDILOCKS_BASE_POINT, TWISTED_EDWARDS_BASE_POINT};

pub use curve::{
    AffinePoint, CompressedEdwardsY, EdwardsPoint, MontgomeryPoint, ProjectiveMontgomeryPoint,
};
pub use decaf::{CompressedDecaf, DecafPoint};
pub use field::{Scalar, ScalarBytes, WideScalarBytes};
pub use ristretto::{CompressedRistretto, RistrettoPoint};
