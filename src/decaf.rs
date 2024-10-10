// This will be the module for Decaf over Ed448
// This is the newer version of the Decaf strategy, which looks simpler

mod ops;
pub mod points;
pub use points::{CompressedDecaf, DecafPoint};
