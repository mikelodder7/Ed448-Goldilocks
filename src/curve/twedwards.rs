/// This module contains the EC arithmetic for the twisted Edwards form of Goldilocks,
/// with the following affine equation: -x^2 + y^2 = 1 - 39082x^2y^2.
/// This curve is used as a backend for Goldilocks, Ristretto, and Decaf through isogenies.
/// It will not be exposed in the public API.
pub(crate) mod affine;
pub(crate) mod extended;
pub(crate) mod extensible;
pub(crate) mod projective;
