//! This crate provides a pure Rust implementation of Curve448, Edwards, Decaf, and Ristretto.
//! It is intended to be portable, fast, and safe.
//!
//! # Usage
//! ```
//! ```
// XXX: Change this to deny later on
#![warn(unused_attributes, unused_imports, unused_mut, unused_must_use)]
#![allow(non_snake_case)]

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
