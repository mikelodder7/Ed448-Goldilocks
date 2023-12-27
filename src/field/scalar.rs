use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use elliptic_curve::{
    ff::{helpers, Field},
    generic_array::{
        typenum::{U114, U57},
        GenericArray,
    },
    PrimeField,
};
use rand_core::{CryptoRng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::constants;

/// This is the scalar field
/// size = 4q = 2^446 - 0x8335dc163bb124b65129c96fde933d8d723a70aadc873d6d54a7bb0d
/// We can therefore use 14 saturated 32-bit limbs
#[derive(Debug, Copy, Clone)]
pub struct Scalar(pub(crate) [u32; 14]);

pub type ScalarBytes = GenericArray<u8, U57>;
pub type WideScalarBytes = GenericArray<u8, U114>;

pub(crate) const MODULUS: Scalar = constants::BASEPOINT_ORDER;

// Montgomomery R^2
const R2: Scalar = Scalar([
    0x049b9b60, 0xe3539257, 0xc1b195d9, 0x7af32c4b, 0x88ea1859, 0x0d66de23, 0x5ee4d838, 0xae17cf72,
    0xa3c47c44, 0x1a9cc14b, 0xe4d070af, 0x2052bcb7, 0xf823b729, 0x3402a939,
]);
const R: Scalar = Scalar([
    0x529eec34, 0x721cf5b5, 0xc8e9c2ab, 0x7a4cf635, 0x44a725bf, 0xeec492d9, 0xcd77058, 0x2, 0, 0,
    0, 0, 0, 0,
]);

impl ConstantTimeEq for Scalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.to_bytes().ct_eq(&other.to_bytes())
    }
}

impl ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut nums = [0u32; 14];
        for i in 0..14 {
            nums[i] = u32::conditional_select(&a.0[i], &b.0[i], choice);
        }
        Self(nums)
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Scalar) -> bool {
        self.ct_eq(other).into()
    }
}

impl Eq for Scalar {}

impl From<u8> for Scalar {
    fn from(a: u8) -> Self {
        Scalar([a as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }
}

impl From<u16> for Scalar {
    fn from(a: u16) -> Self {
        Scalar([a as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }
}

impl From<u32> for Scalar {
    fn from(a: u32) -> Scalar {
        Scalar([a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }
}

impl From<u64> for Scalar {
    fn from(a: u64) -> Self {
        Scalar([
            a as u32,
            (a >> 32) as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ])
    }
}

impl From<u128> for Scalar {
    fn from(a: u128) -> Self {
        Scalar([
            a as u32,
            (a >> 32) as u32,
            (a >> 64) as u32,
            (a >> 96) as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ])
    }
}

impl Index<usize> for Scalar {
    type Output = u32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Scalar {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// Trait implementations
impl Add<&Scalar> for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Self::Output {
        add(self, rhs)
    }
}

impl Add<&Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: &Scalar) -> Self::Output {
        add(&self, rhs)
    }
}

impl Add<Scalar> for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: Scalar) -> Self::Output {
        add(self, &rhs)
    }
}

impl Add<Scalar> for Scalar {
    type Output = Scalar;
    fn add(self, rhs: Scalar) -> Self::Output {
        add(&self, &rhs)
    }
}

impl AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl AddAssign<&Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &Scalar) {
        *self = *self + rhs
    }
}

impl Mul<&Scalar> for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        let unreduced = montgomery_multiply(self, rhs);
        montgomery_multiply(&unreduced, &R2)
    }
}

impl Mul<&Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        self * *rhs
    }
}

impl Mul<Scalar> for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        *self * rhs
    }
}

impl Mul<Scalar> for Scalar {
    type Output = Scalar;
    fn mul(self, rhs: Scalar) -> Self::Output {
        &self * &rhs
    }
}

impl MulAssign for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

impl MulAssign<&Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &Scalar) {
        *self = *self * rhs
    }
}

impl Sub<&Scalar> for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &Scalar) -> Self::Output {
        sub_extra(self, rhs, 0)
    }
}

impl Sub<&Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &Scalar) -> Self::Output {
        sub_extra(&self, rhs, 0)
    }
}

impl Sub<Scalar> for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Scalar) -> Self::Output {
        sub_extra(self, &rhs, 0)
    }
}

impl Sub<Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Scalar) -> Self::Output {
        sub_extra(&self, &rhs, 0)
    }
}

impl SubAssign for Scalar {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl SubAssign<&Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &Scalar) {
        *self = *self - rhs
    }
}

impl Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        Scalar::ZERO - self
    }
}

impl Default for Scalar {
    fn default() -> Scalar {
        Scalar::ZERO
    }
}

impl Sum for Scalar {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut acc = Scalar::ZERO;
        for s in iter {
            acc += s;
        }
        acc
    }
}

impl<'a> Sum<&'a Scalar> for Scalar {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut acc = Scalar::ZERO;
        for s in iter {
            acc += s;
        }
        acc
    }
}

impl Product for Scalar {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut acc = Scalar::ONE;
        for s in iter {
            acc *= s;
        }
        acc
    }
}

impl<'a> Product<&'a Scalar> for Scalar {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut acc = Scalar::ONE;
        for s in iter {
            acc *= s;
        }
        acc
    }
}

impl Field for Scalar {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    fn random(mut rng: impl RngCore) -> Self {
        let mut seed = WideScalarBytes::default();
        rng.fill_bytes(&mut seed);
        Scalar::from_bytes_mod_order_wide(&seed)
    }

    fn square(&self) -> Self {
        self.square()
    }

    fn double(&self) -> Self {
        self + self
    }

    fn invert(&self) -> CtOption<Self> {
        CtOption::new(self.invert(), !self.ct_eq(&Self::ZERO))
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        helpers::sqrt_ratio_generic(num, div)
    }
}

impl PrimeField for Scalar {
    type Repr = ScalarBytes;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        Self::from_canonical_bytes(repr)
    }

    fn to_repr(&self) -> Self::Repr {
        self.to_bytes_rfc_8032()
    }

    fn is_odd(&self) -> Choice {
        Choice::from((self.0[0] & 1) as u8)
    }

    const MODULUS: &'static str = "3fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f3";
    const NUM_BITS: u32 = 448;
    const CAPACITY: u32 = Self::NUM_BITS - 1;
    const TWO_INV: Self = Self([
        0x55ac227a, 0x91bc6149, 0x46e2c7aa, 0x10b66139, 0xd76b1b48, 0xe2276da4, 0xbe6511f4,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x1fffffff,
    ]);
    const MULTIPLICATIVE_GENERATOR: Self = Self([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const S: u32 = 1;
    const ROOT_OF_UNITY: Self = Self([
        0xbdc63ffe, 0xa3a90a90, 0x8330209f, 0x8990470e, 0x3f5c425d, 0x63ea6e70, 0xc2efc65d,
        0xf9b5c014, 0x1d478bc5, 0xb26d8405, 0x59cd81e2, 0x5a1f2fb9, 0x838ee01b, 0x2935b8ce,
    ]);
    const ROOT_OF_UNITY_INV: Self = Self([
        0x284d98cd, 0x37bc82e8, 0x2e8ae84c, 0xe42ddad7, 0xaea94041, 0x1a21435e, 0x1b644703,
        0x2c07bf6c, 0x330b2d96, 0x1f0163dc, 0x8b6172cd, 0x8925b1ee, 0x6717df40, 0x1f87f25a,
    ]);
    const DELTA: Self = Self([0x961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}

#[cfg(feature = "serde")]
impl serde::Serialize for Scalar {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;

        let bytes = self.to_bytes_rfc_8032();
        let mut tupler = s.serialize_tuple(bytes.len())?;
        for i in &bytes {
            tupler.serialize_element(i)?;
        }
        tupler.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Scalar {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{SeqAccess, Visitor};

        struct ScalarVisitor;

        impl<'de> Visitor<'de> for ScalarVisitor {
            type Value = Scalar;

            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "a sequence of 57 little endian bytes")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut scalar = [0u8; 57];
                for (i, v) in scalar.iter_mut().enumerate() {
                    *v = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &"expected 57 bytes"))?;
                }
                Scalar::from_canonical_bytes(scalar)
                    .ok_or_else(|| serde::de::Error::custom("scalar was not canonically encoded"))
            }
        }

        d.deserialize_tuple(57, ScalarVisitor)
    }
}

#[cfg(feature = "zeroize")]
impl zeroize::DefaultIsZeroes for Scalar {}

impl core::fmt::LowerHex for Scalar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tmp = self.to_bytes_rfc_8032();
        for &b in tmp.iter() {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl core::fmt::UpperHex for Scalar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tmp = self.to_bytes_rfc_8032();
        for &b in tmp.iter() {
            write!(f, "{:02X}", b)?;
        }
        Ok(())
    }
}

impl Scalar {
    pub const ONE: Scalar = Scalar([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    pub const TWO: Scalar = Scalar([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    pub const ZERO: Scalar = Scalar([0; 14]);

    pub fn is_zero(&self) -> Choice {
        let mut res = 0i32;
        let mut i = 0;
        while i < 14 {
            res |= self.0[i] as i32;
            i += 1;
        }
        Choice::from((((res | -res) >> 31) + 1) as u8)
    }

    /// Divides a scalar by four without reducing mod p
    /// This is used in the 2-isogeny when mapping points from Ed448-Goldilocks
    /// to Twisted-Goldilocks
    pub(crate) fn div_by_four(&mut self) {
        for i in 0..=12 {
            self.0[i] = (self.0[i + 1] << 30) | (self.0[i] >> 2);
        }
        self.0[13] >>= 2
    }
    // This method was modified from Curve25519-Dalek codebase. [scalar.rs]
    // We start with 14 u32s and convert them to 56 u8s.
    // We then use the code copied from Dalek to convert the 56 u8s to radix-16 and re-center the coefficients to be between [-16,16)
    // XXX: We can recode the scalar without converting it to bytes, will refactor this method to use this and check which is faster.
    pub(crate) fn to_radix_16(&self) -> [i8; 113] {
        let bytes = self.to_bytes();
        let mut output = [0i8; 113];

        // Step 1: change radix.
        // Convert from radix 256 (bytes) to radix 16 (nibbles)
        #[inline(always)]
        fn bot_half(x: u8) -> u8 {
            x & 15
        }
        #[inline(always)]
        fn top_half(x: u8) -> u8 {
            (x >> 4) & 15
        }

        // radix-16
        for i in 0..56 {
            output[2 * i] = bot_half(bytes[i]) as i8;
            output[2 * i + 1] = top_half(bytes[i]) as i8;
        }
        // re-center co-efficients to be between [-8, 8)
        for i in 0..112 {
            let carry = (output[i] + 8) >> 4;
            output[i] -= carry << 4;
            output[i + 1] += carry;
        }

        output
    }
    // XXX: Better if this method returns an array of 448 items
    pub fn bits(&self) -> Vec<bool> {
        let mut bits: Vec<bool> = Vec::with_capacity(14 * 32);
        // We have 14 limbs, each 32 bits
        // First we iterate each limb
        for limb in self.0.iter() {
            // Then we iterate each bit in the limb
            for j in 0..32 {
                bits.push(limb & (1 << j) != 0)
            }
        }

        // XXX :We are doing LSB first
        bits
    }

    pub fn from_bytes(bytes: [u8; 56]) -> Scalar {
        let load7 = |input: &[u8]| -> u64 {
            (input[0] as u64)
                | ((input[1] as u64) << 8)
                | ((input[2] as u64) << 16)
                | ((input[3] as u64) << 24)
        };

        let mut res = Scalar::ZERO;
        for i in 0..14 {
            // Load i'th 32 bytes
            let out = load7(&bytes[i * 4..]);
            res[i] = out as u32;
        }

        res
    }

    pub fn to_bytes(&self) -> [u8; 56] {
        let mut res = [0u8; 56];

        for i in 0..14 {
            let mut l = self.0[i];
            for j in 0..4 {
                res[4 * i + j] = l as u8;
                l >>= 8;
            }
        }
        res
    }

    pub fn square(&self) -> Scalar {
        montgomery_multiply(self, self)
    }

    pub fn invert(&self) -> Self {
        let mut pre_comp = [Scalar::ZERO; 8];
        let mut result = Scalar::ZERO;

        let scalar_window_bits = 3;
        let last = (1 << scalar_window_bits) - 1;

        // precompute [a^1, a^3,,..]
        pre_comp[0] = montgomery_multiply(self, &R2);

        if last > 0 {
            pre_comp[last] = montgomery_multiply(&pre_comp[0], &pre_comp[0]);
        }

        for i in 1..=last {
            pre_comp[i] = montgomery_multiply(&pre_comp[i - 1], &pre_comp[last])
        }

        // Sliding window
        let mut residue: usize = 0;
        let mut trailing: usize = 0;
        let mut started: usize = 0;

        // XXX: This can definitely be refactored to be readable
        let loop_start = -scalar_window_bits as isize;
        let loop_end = 446 - 1;
        for i in (loop_start..=loop_end).rev() {
            if started != 0 {
                result = result.square()
            }

            let mut w: u32;
            if i >= 0 {
                w = MODULUS[(i / 32) as usize];
            } else {
                w = 0;
            }

            if (0..32).contains(&i) {
                w -= 2
            }

            residue = (((residue as u32) << 1) | ((w >> ((i as u32) % 32)) & 1)) as usize;
            if residue >> scalar_window_bits != 0 {
                trailing = residue;
                residue = 0
            }

            if trailing > 0 && (trailing & ((1 << scalar_window_bits) - 1)) == 0 {
                if started != 0 {
                    result = montgomery_multiply(
                        &result,
                        &pre_comp[trailing >> (scalar_window_bits + 1)],
                    )
                } else {
                    result = pre_comp[trailing >> (scalar_window_bits + 1)];
                    started = 1
                }
                trailing = 0
            }
            trailing <<= 1
        }

        // de-montgomerize and return result

        montgomery_multiply(&result, &Scalar::ONE)
    }

    /// Halves a Scalar modulo the prime
    pub fn halve(&self) -> Self {
        let mut result = Scalar::ZERO;

        let mask = 0u32.wrapping_sub(self[0] & 1);
        let mut chain = 0u64;

        for i in 0..14 {
            chain += (self[i] as u64) + ((MODULUS[i] & mask) as u64);
            result[i] = chain as u32;
            chain >>= 32
        }

        for i in 0..13 {
            result[i] = (result[i] >> 1) | (result[i + 1] << 31);
        }
        result[13] = (result[13] >> 1) | ((chain << 31) as u32);

        result
    }

    /// Attempt to construct a `Scalar` from a canonical byte representation.
    ///
    /// # Return
    ///
    /// - `Some(s)`, where `s` is the `Scalar` corresponding to `bytes`,
    ///   if `bytes` is a canonical byte representation;
    /// - `None` if `bytes` is not a canonical byte representation.
    pub fn from_canonical_bytes(bytes: ScalarBytes) -> CtOption<Scalar> {
        // Check that the 10 high bits are not set
        let is_valid = is_zero(bytes[56]) | is_zero(bytes[55] >> 6);
        let bytes: [u8; 56] = core::array::from_fn(|i| bytes[i]);
        let candidate = Scalar::from_bytes(bytes);

        let reduced = sub_extra(&candidate, &MODULUS, 0);

        CtOption::new(candidate, candidate.ct_eq(&reduced) & is_valid)
    }

    /// Serialize the scalar into 57 bytes, per RFC 8032.
    /// Byte 56 will always be zero.
    pub fn to_bytes_rfc_8032(&self) -> ScalarBytes {
        let mut bytes = ScalarBytes::default();
        bytes[..56].copy_from_slice(&self.to_bytes());
        bytes
    }

    /// Construct a `Scalar` by reducing a 912-bit little-endian integer
    /// modulo the group order ℓ.
    pub fn from_bytes_mod_order_wide(input: &WideScalarBytes) -> Scalar {
        let lo: [u8; 56] = core::array::from_fn(|i| input[i]);
        let lo = Scalar::from_bytes(lo);
        // montgomery_multiply computes ((a*b)/R) mod ℓ, thus this computes
        // ((lo*R)/R) = lo mod ℓ
        let lo = montgomery_multiply(&lo, &R);

        let hi: [u8; 56] = core::array::from_fn(|i| input[i + 56]);
        let hi = Scalar::from_bytes(hi);
        // ((hi*R^2)/R) = hi * R mod ℓ
        let hi = montgomery_multiply(&hi, &R2);

        // There are only two bytes left, build an array with them and pad with zeroes
        let top: [u8; 56] = core::array::from_fn(|i| if i < 2 { input[i + 112] } else { 0 });
        let top = Scalar::from_bytes(top);
        // ((top*R^2)/R) = top * R mod ℓ
        let top = montgomery_multiply(&top, &R2);
        // (((top*R)*R^2)/R) = top * R^2 mod ℓ
        let top = montgomery_multiply(&top, &R2);

        // lo + hi*R + top*R^2 mod ℓ is the final result we want
        add(&lo, &hi).add(top)
    }

    /// Return a `Scalar` chosen uniformly at random using a user-provided RNG.
    ///
    /// # Inputs
    ///
    /// * `rng`: any RNG which implements the `RngCore + CryptoRng` interface.
    ///
    /// # Returns
    ///
    /// A random scalar within ℤ/lℤ.
    pub fn random<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
        let mut scalar_bytes = WideScalarBytes::default();
        rng.fill_bytes(&mut scalar_bytes);
        Scalar::from_bytes_mod_order_wide(&scalar_bytes)
    }
}

/// Computes a + b mod p
pub(crate) fn add(a: &Scalar, b: &Scalar) -> Scalar {
    // First add the two Scalars together
    // Since our limbs are saturated, the result of each
    // limb being added can be a 33-bit integer so we propagate the carry bit
    let mut result = Scalar::ZERO;

    // a + b
    let mut chain = 0u64;
    //XXX: Can unroll all of these for loops. They are mainly just ripple carry/borrow adders.
    for i in 0..14 {
        chain += (a[i] as u64) + (b[i] as u64);
        // Low 32 bits are the results
        result[i] = chain as u32;
        // 33rd bit is the carry
        chain >>= 32;
    }

    // Now reduce the results
    sub_extra(&result, &MODULUS, chain as u32)
}

/// Compute a - b mod p
/// Computes a - b and conditionally computes the modulus if the result was negative
fn sub_extra(a: &Scalar, b: &Scalar, carry: u32) -> Scalar {
    let mut result = Scalar::ZERO;

    // a - b
    let mut chain = 0i64;
    for i in 0..14 {
        chain += a[i] as i64 - b[i] as i64;
        // Low 32 bits are the results
        result[i] = chain as u32;
        // 33rd bit is the borrow
        chain >>= 32
    }

    // if the result of a-b was negative and carry was zero
    // then borrow will be 0xfff..fff and the modulus will be added conditionally to the result
    // If the carry was 1 and a-b was not negative, then the borrow will be 0x00000...001 ( this should not happen)
    // Since the borrow should never be more than 0, the carry should never be more than 1;
    // XXX: Explain why the case of borrow == 1 should never happen
    let borrow = chain + (carry as i64);

    chain = 0i64;
    for i in 0..14 {
        chain += (result[i] as i64) + ((MODULUS[i] as i64) & borrow);
        // Low 32 bits are the results
        result[i] = chain as u32;
        // 33rd bit is the carry
        chain >>= 32;
    }

    result
}

fn montgomery_multiply(x: &Scalar, y: &Scalar) -> Scalar {
    const MONTGOMERY_FACTOR: u32 = 0xae918bc5;

    let mut result = Scalar::ZERO;
    let mut carry = 0u32;

    // (a * b ) + c
    let mul_add = |a: u32, b: u32, c: u32| -> u64 { ((a as u64) * (b as u64)) + (c as u64) };

    for i in 0..14 {
        let mut chain = 0u64;
        for j in 0..14 {
            chain += mul_add(x[i], y[j], result[j]);
            result[j] = chain as u32;
            chain >>= 32;
        }

        let saved = chain as u32;
        let multiplicand = result[0].wrapping_mul(MONTGOMERY_FACTOR);
        chain = 0u64;

        for j in 0..14 {
            chain += mul_add(multiplicand, MODULUS[j], result[j]);
            if j > 0 {
                result[j - 1] = chain as u32;
            }
            chain >>= 32;
        }
        chain += (saved as u64) + (carry as u64);
        result[14 - 1] = chain as u32;
        carry = (chain >> 32) as u32;
    }

    sub_extra(&result, &MODULUS, carry)
}

fn is_zero(b: u8) -> Choice {
    let res = b as i8;
    Choice::from((((res | -res) >> 7) + 1) as u8)
}

#[cfg(test)]
mod test {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn test_basic_add() {
        let five = Scalar::from(5u8);
        let six = Scalar::from(6u8);

        assert_eq!(five + six, Scalar::from(11u8))
    }

    #[test]
    fn test_basic_sub() {
        let ten = Scalar::from(10u8);
        let five = Scalar::from(5u8);
        assert_eq!(ten - five, Scalar::from(5u8))
    }

    #[test]
    fn test_basic_mul() {
        let ten = Scalar::from(10u8);
        let five = Scalar::from(5u8);

        assert_eq!(ten * five, Scalar::from(50u8))
    }

    #[test]
    fn test_mul() {
        let a = Scalar([
            0xffb823a3, 0xc96a3c35, 0x7f8ed27d, 0x087b8fb9, 0x1d9ac30a, 0x74d65764, 0xc0be082e,
            0xa8cb0ae8, 0xa8fa552b, 0x2aae8688, 0x2c3dc273, 0x47cf8cac, 0x3b089f07, 0x1e63e807,
        ]);

        let b = Scalar([
            0xd8bedc42, 0x686eb329, 0xe416b899, 0x17aa6d9b, 0x1e30b38b, 0x188c6b1a, 0xd099595b,
            0xbc343bcb, 0x1adaa0e7, 0x24e8d499, 0x8e59b308, 0x0a92de2d, 0xcae1cb68, 0x16c5450a,
        ]);

        let exp = Scalar([
            0xa18d010a, 0x1f5b3197, 0x994c9c2b, 0x6abd26f5, 0x08a3a0e4, 0x36a14920, 0x74e9335f,
            0x07bcd931, 0xf2d89c1e, 0xb9036ff6, 0x203d424b, 0xfccd61b3, 0x4ca389ed, 0x31e055c1,
        ]);

        assert_eq!(a * b, exp)
    }
    #[test]
    fn test_basic_square() {
        let a = Scalar([
            0xcf5fac3d, 0x7e56a34b, 0xf640922b, 0x3fa50692, 0x1370f8b8, 0x6f08f331, 0x8dccc486,
            0x4bb395e0, 0xf22c6951, 0x21cc3078, 0xd2391f9d, 0x930392e5, 0x04b3273b, 0x31620816,
        ]);
        let expected_a_squared = Scalar([
            0x15598f62, 0xb9b1ed71, 0x52fcd042, 0x862a9f10, 0x1e8a309f, 0x9988f8e0, 0xa22347d7,
            0xe9ab2c22, 0x38363f74, 0xfd7c58aa, 0xc49a1433, 0xd9a6c4c3, 0x75d3395e, 0x0d79f6e3,
        ]);

        assert_eq!(a.square(), expected_a_squared)
    }

    #[test]
    fn test_sanity_check_index_mut() {
        let mut x = Scalar::ONE;
        x[0] = 2u32;
        assert_eq!(x, Scalar::from(2u8))
    }
    #[test]
    fn test_basic_halving() {
        let eight = Scalar::from(8u8);
        let four = Scalar::from(4u8);
        let two = Scalar::from(2u8);
        assert_eq!(eight.halve(), four);
        assert_eq!(four.halve(), two);
        assert_eq!(two.halve(), Scalar::ONE);
    }

    #[test]
    fn test_equals() {
        let a = Scalar::from(5u8);
        let b = Scalar::from(5u8);
        let c = Scalar::from(10u8);
        assert!(a == b);
        assert!(!(a == c))
    }

    #[test]
    fn test_basic_inversion() {
        // Test inversion from 2 to 100
        for i in 1..=100u8 {
            let x = Scalar::from(i);
            let x_inv = x.invert();
            assert_eq!(x_inv * x, Scalar::ONE)
        }

        // Inversion of zero is zero
        let zero = Scalar::ZERO;
        let expected_zero = zero.invert();
        assert_eq!(expected_zero, zero)
    }
    #[test]
    fn test_serialise() {
        let scalar = Scalar([
            0x15598f62, 0xb9b1ed71, 0x52fcd042, 0x862a9f10, 0x1e8a309f, 0x9988f8e0, 0xa22347d7,
            0xe9ab2c22, 0x38363f74, 0xfd7c58aa, 0xc49a1433, 0xd9a6c4c3, 0x75d3395e, 0x0d79f6e3,
        ]);
        let got = Scalar::from_bytes(scalar.to_bytes());
        assert_eq!(scalar, got)
    }
    #[test]
    fn test_debug() {
        let k = Scalar([
            200, 210, 250, 145, 130, 180, 147, 122, 222, 230, 214, 247, 203, 32,
        ]);
        let s = k;
        dbg!(&s.to_radix_16()[..]);
    }
    #[test]
    fn test_from_canonical_bytes() {
        // ff..ff should fail
        let mut bytes = ScalarBytes::clone_from_slice(&hex!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"));
        bytes.reverse();
        let s = Scalar::from_canonical_bytes(bytes);
        assert!(<Choice as Into<bool>>::into(s.is_none()));

        // n should fail
        let mut bytes = ScalarBytes::clone_from_slice(&hex!("003fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f3"));
        bytes.reverse();
        let s = Scalar::from_canonical_bytes(bytes);
        assert!(<Choice as Into<bool>>::into(s.is_none()));

        // n-1 should work
        let mut bytes = ScalarBytes::clone_from_slice(&hex!("003fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f2"));
        bytes.reverse();
        let s = Scalar::from_canonical_bytes(bytes);
        match Option::<Scalar>::from(s) {
            Some(s) => assert_eq!(s, Scalar::ZERO - Scalar::ONE),
            None => panic!("should not return None"),
        };
    }

    #[test]
    fn test_from_bytes_mod_order_wide() {
        // n should become 0
        let mut bytes = WideScalarBytes::clone_from_slice(&hex!("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f3"));
        bytes.reverse();
        let s = Scalar::from_bytes_mod_order_wide(&bytes);
        assert_eq!(s, Scalar::ZERO);

        // n-1 should stay the same
        let mut bytes = WideScalarBytes::clone_from_slice(&hex!("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f2"));
        bytes.reverse();
        let s = Scalar::from_bytes_mod_order_wide(&bytes);
        assert_eq!(s, Scalar::ZERO - Scalar::ONE);

        // n+1 should become 1
        let mut bytes = WideScalarBytes::clone_from_slice(&hex!("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f4"));
        bytes.reverse();
        let s = Scalar::from_bytes_mod_order_wide(&bytes);
        assert_eq!(s, Scalar::ONE);

        // 2^912-1 should become 0x2939f823b7292052bcb7e4d070af1a9cc14ba3c47c44ae17cf72c985bb24b6c520e319fb37a63e29800f160787ad1d2e11883fa931e7de81
        let mut bytes = WideScalarBytes::clone_from_slice(&hex!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"));
        bytes.reverse();
        let s = Scalar::from_bytes_mod_order_wide(&bytes);
        let mut bytes = ScalarBytes::clone_from_slice(&hex!("002939f823b7292052bcb7e4d070af1a9cc14ba3c47c44ae17cf72c985bb24b6c520e319fb37a63e29800f160787ad1d2e11883fa931e7de81"));
        bytes.reverse();
        let reduced = Scalar::from_canonical_bytes(bytes).unwrap();
        assert_eq!(s, reduced);
    }

    #[test]
    fn test_to_bytes_rfc8032() {
        // n-1
        let mut bytes: [u8; 57] = hex!("003fffffffffffffffffffffffffffffffffffffffffffffffffffffff7cca23e9c44edb49aed63690216cc2728dc58f552378c292ab5844f2");
        bytes.reverse();
        let x = Scalar::ZERO - Scalar::ONE;
        let candidate = x.to_bytes_rfc_8032();
        assert_eq!(&bytes[..], &candidate[..]);
    }
}
