#[cfg(feature = "fiat_u64_backend")]
pub mod fiat_u64;

#[cfg(feature = "u32_backend")]
pub mod u32;

// XXX: Currently we only have one implementation for Scalar
mod scalar;

use elliptic_curve::{
    bigint::U704,
    generic_array::{GenericArray, typenum::{U84, U88}},
    hash2curve::FromOkm,
};
pub use crate::field::scalar::Scalar;

#[cfg(feature = "u32_backend")]
pub type FieldElement = crate::field::u32::FieldElement28;

#[cfg(feature = "fiat_u64_backend")]
pub type FieldElement = crate::field::fiat_u64::FieldElement56;

use subtle::{Choice, ConstantTimeEq};
impl ConstantTimeEq for FieldElement {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.to_bytes().ct_eq(&other.to_bytes())
    }
}

impl PartialEq for FieldElement {
    fn eq(&self, other: &FieldElement) -> bool {
        self.ct_eq(&other).into()
    }
}
impl Eq for FieldElement {}

impl FromOkm for FieldElement {
    type Length = U84;
    
    fn from_okm(data: &GenericArray<u8, Self::Length>) -> Self {
        use elliptic_curve::bigint::Encoding;

        let mut tmp = GenericArray::<u8, U88>::default();
        tmp[4..].copy_from_slice(&data[..]);

        let mut num = U704::from_be_slice(&tmp[..]);
        let den = U704::from_be_hex("0000000000000000000000000000000000000000000000000000000000000000fffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        num = num.wrapping_rem(&den);
        let bytes = <[u8; 56]>::try_from(&num.to_le_bytes()[..56]).unwrap();
        FieldElement::from_bytes(&bytes)
    }
}

impl FieldElement {
    /// Checks if a field element is zero
    pub(crate) fn is_zero(&self) -> Choice {
        self.ct_eq(&FieldElement::ZERO)
    }
    /// Inverts a field element
    /// Previous chain length: 462, new length 460
    pub fn invert(&self) -> FieldElement {
        // Addition chain taken from https://github.com/mmcloughlin/addchain
        let _1 = self;
        let _10 = _1.square();
        let _11 = *_1 * _10;
        let _110 = _11.square();
        let _111 = *_1 * _110;
        let _111000 = _111.square_n(3);
        let _111111 = _111 * _111000;

        let x12 = _111111.square_n(6) * _111111;
        let x24 = x12.square_n(12) * x12;
        let i34 = x24.square_n(6);
        let x30 = _111111 * i34;
        let x48 = i34.square_n(18) * x24;
        let x96 = x48.square_n(48) * x48;
        let x192 = x96.square_n(96) * x96;
        let x222 = x192.square_n(30) * x30;
        let x223 = x222.square() * _1;

        (x223.square_n(223) * x222).square_n(2) * _1
    }
    /// Squares a field element  `n` times
    fn square_n(&self, mut n: u32) -> FieldElement {
        let mut result = self.square();

        // Decrease value by 1 since we just did a squaring
        n = n - 1;

        for _ in 0..n {
            result = result.square();
        }

        result
    }

    /// Computes the inverse square root of a field element
    /// Returns the result and a boolean to indicate whether self
    /// was a Quadratic residue
    pub(crate) fn inverse_square_root(&self) -> (FieldElement, Choice) {
        let (mut l0, mut l1, mut l2) = (
            FieldElement::ZERO,
            FieldElement::ZERO,
            FieldElement::ZERO,
        );

        l1 = self.square();
        l2 = l1 * self;
        l1 = l2.square();
        l2 = l1 * self;
        l1 = l2.square_n(3);
        l0 = l2 * l1;
        l1 = l0.square_n(3);
        l0 = l2 * l1;
        l2 = l0.square_n(9);
        l1 = l0 * l2;
        l0 = l1 * l1;
        l2 = l0 * self;
        l0 = l2.square_n(18);
        l2 = l1 * l0;
        l0 = l2.square_n(37);
        l1 = l2 * l0;
        l0 = l1.square_n(37);
        l1 = l2 * l0;
        l0 = l1.square_n(111);
        l2 = l1 * l0;
        l0 = l2.square();
        l1 = l0 * self;
        l0 = l1.square_n(223);
        l1 = l2 * l0;
        l2 = l1.square();
        l0 = l2 * self;

        let is_residue = l0.ct_eq(&FieldElement::ONE);
        (l1, is_residue)
    }

    /// Computes the square root ratio of two elements
    pub(crate) fn sqrt_ratio(u: &FieldElement, v: &FieldElement) -> (FieldElement, Choice) {
        // Compute sqrt(1/(uv))
        let x = *u * v;
        let (inv_sqrt_x, is_res) = x.inverse_square_root();
        // Return u * sqrt(1/(uv)) == sqrt(u/v). However, since this trick only works
        // for u != 0, check for that case explicitly (when u == 0 then inv_sqrt_x
        // will be zero, which is what we want, but is_res will be 0)
        let zero_u = u.ct_eq(&FieldElement::ZERO);
        (inv_sqrt_x * u, zero_u | is_res)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use elliptic_curve::hash2curve::{ExpandMsg, Expander, ExpandMsgXof};
    use hex_literal::hex;
    use sha3::Shake256;

    #[test]
    fn from_okm() {
        const DST: &[u8] = b"QUUX-V01-CS02-with-curve448_XOF:SHAKE256_ELL2_RO_";
        const MSGS: &[(&[u8], [u8; 56], [u8; 56])] = &[
            (b"", hex!("c704c7b3d3b36614cf3eedd0324fe6fe7d1402c50efd16cff89ff63f50938506280d3843478c08e24f7842f4e3ef45f6e3c4897f9d976148"), hex!("c25427dc97fff7a5ad0a78654e2c6c27b1c1127b5b53c7950cd1fd6edd2703646b25f341e73deedfebf022d1d3cecd02b93b4d585ead3ed7")),
            (b"abc", hex!("2dd95593dfee26fe0d218d3d9a0a23d9e1a262fd1d0b602483d08415213e75e2db3c69b0a5bc89e71bcefc8c723d2b6a0cf263f02ad2aa70"), hex!("272e4c79a1290cc6d2bc4f4f9d31bf7fbe956ca303c04518f117d77c0e9d850796fc3e1e2bcb9c75e8eaaded5e150333cae9931868047c9d")),
            (b"abcdef0123456789", hex!("6aab71a38391639f27e49eae8b1cb6b7172a1f478190ece293957e7cdb2391e7cc1c4261970d9c1bbf9c3915438f74fbd7eb5cd4d4d17ace"), hex!("c80b8380ca47a3bcbf76caa75cef0e09f3d270d5ee8f676cde11aedf41aaca6741bd81a86232bd336ccb42efad39f06542bc06a67b65909e")),
            (b"q128_qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq", hex!("cb5c27e51f9c18ee8ffdb6be230f4eb4f2c2481963b2293484f08da2241c1ff59f80978e6defe9d70e34abba2fcbe12dc3a1eb2c5d3d2e4a"), hex!("c895e8afecec5466e126fa70fc4aa784b8009063afb10e3ee06a9b22318256aa8693b0c85b955cf2d6540b8ed71e729af1b8d5ca3b116cd7")),
            (b"a512_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", hex!("8cba93a007bb2c801b1769e026b1fa1640b14a34cf3029db3c7fd6392745d6fec0f7870b5071d6da4402cedbbde28ae4e50ab30e1049a238"), hex!("4223746145069e4b8a981acc3404259d1a2c3ecfed5d864798a89d45f81a2c59e2d40eb1d5f0fe11478cbb2bb30246dd388cb932ad7bb330")),
        ];

        for (msg, expected_u0, expected_u1) in MSGS {
            let mut expander = ExpandMsgXof::<Shake256>::expand_message(&[msg], &[DST], 84*2).unwrap();
            let mut data = GenericArray::<u8, U84>::default();
            expander.fill_bytes(&mut data);
            let u0 = FieldElement::from_okm(&data);
            let mut e_u0 = *expected_u0;
            e_u0.reverse();
            let mut e_u1 = *expected_u1;
            e_u1.reverse();
            assert_eq!(u0.to_bytes(), e_u0);
            expander.fill_bytes(&mut data);
            let u1 = FieldElement::from_okm(&data);
            assert_eq!(u1.to_bytes(), e_u1);
        }

    }
}