<p align="center">
<img src="resources/bear.png" width="400">
</p>

ed448-goldilocks-plus

[![Crate][crate-image]][crate-link]
[![Docs][docs-image]][docs-link]
![BSD-3 Licensed][license-image]
[![Downloads][downloads-image]][crate-link]
![build](https://github.com/mikelodder7/Ed448-Goldilocks/actions/workflows/rust.yml/badge.svg)
![MSRV][msrv-image]

THIS CODE HAS NOT BEEN AUDITED OR REVIEWED. USE AT YOUR OWN RISK.

## Field Choice

The field size is a Solinas trinomial prime, 2^448 - 2^224 - 1. This prime is called the Goldilocks prime.

## Curves

This repository implements three curves explicitly and another curve implicitly.

The three explicitly implemented curves are:

- Ed448-Goldilocks
- Curve448
- Twisted-Goldilocks

## Ed448-Goldilocks Curve

- The Goldilocks curve is an Edwards curve with affine equation x^2 + y^2 = 1 - 39081x^2y^2.
- This curve was defined by Mike Hamburg in <https://eprint.iacr.org/2015/625.pdf>.
- The cofactor of this curve over the Goldilocks prime is 4.

## Twisted-Goldilocks Curve

- The Twisted-Goldilocks curve is a twisted Edwards curve with affine equation y^2 - x^2 = 1 - 39082x^2y^2.
- This curve is also defined in <https://eprint.iacr.org/2015/625.pdf>.
- The cofactor of this curve over the Goldilocks prime is 4.

### Isogeny

- This curve is 2-isogenous to Ed448-Goldilocks. Details of the isogeny are available in the [isogeny paper](https://www.shiftleft.org/papers/isogeny/isogeny.pdf).

## Curve448

This curve is 2-isogenous to Ed448-Goldilocks. Details of Curve448 are available in [RFC 7748](https://tools.ietf.org/html/rfc7748).

The main usage of this curve is for X448.

Note: That document describes an Edwards curve that is birationally equivalent to Curve448 and has a large `d` value. This curve is not implemented and, to my knowledge, has no utility.

## Strategy

The main strategy for group arithmetic on Ed448-Goldilocks is to perform the 2-isogeny to map the point to the Twisted-Goldilocks curve, then use the faster twisted Edwards formulas to perform scalar multiplication. Computing the 2-isogeny and then the dual isogeny picks up a factor of 4 when the point is mapped back to the Ed448-Goldilocks curve, so the scalar must be adjusted by a factor of 4. Adjusting the scalar depends on the point and the scalar. More details are available in the isogeny paper linked above.

# Decaf

The [Decaf strategy](https://www.shiftleft.org/papers/decaf/decaf.pdf) is used to build a group of prime order from the Twisted-Goldilocks curve, which has faster formulas. Curve448 or Ed448-Goldilocks can also be used. Decaf takes advantage of an isogeny with a Jacobi quartic curve that is not explicitly defined. However, to my knowledge, there is no documentation for the Decaf protocol implemented in this repository, which is a modified version of the original Decaf protocol described in the paper.

## Completed Points vs. Extensible Points

Unlike Curve25519-Dalek, this library implements extensible points instead of completed points for the following reason:

- Switching from a `CompletedPoint` costs three or four field multiplications. Repeated doubling would therefore add this cost for every doubling in projective form. Section 3.2 of the [Fast and Faster paper](https://www.shiftleft.org/papers/fff/fff.pdf) provides more details about the `ExtensiblePoint`.

## Credits

The library design was taken from Dalek's design of Curve25519. The code for Montgomery curve arithmetic was also taken from Dalek's library.

The Go implementation of Ed448 and libdecaf were used as references.

Special thanks to Mike Hamburg for answering questions about Decaf and Goldilocks.

This library adds [hash_to_curve](https://datatracker.ietf.org/doc/rfc9380/) and serialization of structs.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the BSD-3-Clause
license, shall be dual-licensed as above, without any additional terms or
conditions.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/ed448-goldilocks-plus.svg
[crate-link]: https://crates.io/crates/ed448-goldilocks-plus
[docs-image]: https://docs.rs/ed448-goldilocks-plus/badge.svg
[docs-link]: https://docs.rs/ed448-goldilocks-plus/
[license-image]: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
[downloads-image]: https://img.shields.io/crates/d/ed448-goldilocks-plus.svg
[msrv-image]: https://img.shields.io/badge/rustc-1.85+-blue.svg
