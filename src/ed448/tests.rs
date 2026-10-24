use super::{KEYPAIR_LENGTH, SIGNATURE_LENGTH, Signature, SigningKey, VerifyingKey};
use crate::ed448::{PreHasherXof, SecretKey};
use shake::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};

#[test]
fn signs_and_verifies_rfc8032_vector_through_ed448_module() {
    let seed = SecretKey::from([
        0x6c, 0x82, 0xa5, 0x62, 0xcb, 0x80, 0x8d, 0x10, 0xd6, 0x32, 0xbe, 0x89, 0xc8, 0x51, 0x3e,
        0xbf, 0x6c, 0x92, 0x9f, 0x34, 0xdd, 0xfa, 0x8c, 0x9f, 0x63, 0xc9, 0x96, 0x0e, 0xf6, 0xe3,
        0x48, 0xa3, 0x52, 0x8c, 0x8a, 0x3f, 0xcc, 0x2f, 0x04, 0x4e, 0x39, 0xa3, 0xfc, 0x5b, 0x94,
        0x49, 0x2f, 0x8f, 0x03, 0x2e, 0x75, 0x49, 0xa2, 0x00, 0x98, 0xf9, 0x5b,
    ]);
    let public_key = [
        0x5f, 0xd7, 0x44, 0x9b, 0x59, 0xb4, 0x61, 0xfd, 0x2c, 0xe7, 0x87, 0xec, 0x61, 0x6a, 0xd4,
        0x6a, 0x1d, 0xa1, 0x34, 0x24, 0x85, 0xa7, 0x0e, 0x1f, 0x8a, 0x0e, 0xa7, 0x5d, 0x80, 0xe9,
        0x67, 0x78, 0xed, 0xf1, 0x24, 0x76, 0x9b, 0x46, 0xc7, 0x06, 0x1b, 0xd6, 0x78, 0x3d, 0xf1,
        0xe5, 0x0f, 0x6c, 0xd1, 0xfa, 0x1a, 0xbe, 0xaf, 0xe8, 0x25, 0x61, 0x80,
    ];
    let signing_key = SigningKey::from(&seed);
    let verifying_key = signing_key.verifying_key();

    assert_eq!(verifying_key.as_bytes(), &public_key);

    let signature = signing_key.sign_raw(&[]);
    let parsed_key = match VerifyingKey::from_bytes(&public_key) {
        Ok(key) => key,
        Err(err) => panic!("valid RFC8032 public key failed to parse: {err:?}"),
    };

    assert_eq!(parsed_key, verifying_key);
    assert!(verifying_key.verify_raw(&signature, &[]).is_ok());

    let keypair_bytes = signing_key.to_keypair_bytes();
    assert_eq!(keypair_bytes.len(), KEYPAIR_LENGTH);
    assert_eq!(&keypair_bytes[..57], &signing_key.as_bytes()[..]);
    assert_eq!(&keypair_bytes[57..], verifying_key.as_bytes());
    assert_eq!(
        SigningKey::from_keypair_bytes(&keypair_bytes)
            .expect("valid keypair bytes")
            .verifying_key(),
        verifying_key
    );
    assert_eq!(
        SigningKey::as_ref(&signing_key).as_bytes(),
        verifying_key.as_bytes()
    );

    let signature_bytes = signature.to_bytes();
    assert_eq!(signature_bytes.len(), SIGNATURE_LENGTH);
    assert_eq!(
        super::Signature::from_slice(&signature_bytes)
            .expect("valid signature slice")
            .to_bytes(),
        signature_bytes
    );
    assert_eq!(signature.r_bytes(), &signature_bytes[..57]);
    assert_eq!(signature.s_bytes(), &signature_bytes[57..]);
}

#[test]
fn signs_and_verifies_prehash_with_context_through_ed448_module() {
    let seed = SecretKey::from([
        0x83, 0x3f, 0xe6, 0x24, 0x09, 0x23, 0x7b, 0x9d, 0x62, 0xec, 0x77, 0x58, 0x75, 0x20, 0x91,
        0x1e, 0x9a, 0x75, 0x9c, 0xec, 0x1d, 0x19, 0x75, 0x5b, 0x7d, 0xa9, 0x01, 0xb9, 0x6d, 0xca,
        0x3d, 0x42, 0xef, 0x78, 0x22, 0xe0, 0xd5, 0x10, 0x41, 0x27, 0xdc, 0x05, 0xd6, 0xdb, 0xef,
        0xde, 0x69, 0xe3, 0xab, 0x2c, 0xec, 0x7c, 0x86, 0x7c, 0x6e, 0x2c, 0x49,
    ]);
    let message = b"abc";
    let context = b"foo";
    let signing_key = SigningKey::from(&seed);
    let verifying_key = signing_key.verifying_key();
    let signature = match signing_key.sign_prehashed::<PreHasherXof<Shake256>>(
        Some(context),
        Shake256::default().chain(message).into(),
    ) {
        Ok(signature) => signature,
        Err(err) => panic!("valid Ed448ph signing input failed: {err:?}"),
    };

    let mut reader = Shake256::default().chain(message).finalize_xof();
    let mut prehash = [0u8; 64];
    reader.read(&mut prehash);

    assert!(
        verifying_key
            .verify_prehashed::<PreHasherXof<Shake256>>(
                &signature,
                Some(context),
                Shake256::default().chain(message).into(),
            )
            .is_ok()
    );
    assert!(verifying_key.verify_raw(&signature, &prehash).is_err());
}

#[test]
fn wycheproof_ed448_unused_bits_in_r() {
    // Wycheproof ed448_test.json (testvectors_v1): tcId 1 is valid, tcIds 63, 64 and 65 set a
    // bit among bits 448 to 454 of R and are invalid.
    use hex_literal::hex;

    let public_key = hex!(
        "419610a534af127f583b04818cdb7f0ff300b025f2e01682bcae33fd691cee03"
        "9511df0cddc690ee978426e8b38e50ce5af7dcfba50f704c00"
    );
    let invalid_message = hex!("313233343030");
    let cases: [(&[u8], [u8; SIGNATURE_LENGTH], bool); 4] = [
        (
            &[],
            hex!(
                "cf7953007666e12f73af9ec92e3e018da5ee5a8d5b17f5100a354c58f1d5f4bb"
                "37ab835c52f72374c72d612689149cf6d36a70db6dc5a6c400b597348e0e31e5"
                "1e65bb144e63c892a367b4c055c036aa6cd7e728cdd2a098963bda863903e6dd"
                "025b5a5d891209f4e28537694804e50b0800"
            ),
            true,
        ),
        (
            &invalid_message,
            hex!(
                "5db94c53101f521f6c1f43b60ea4d7e06fbd49c2e8afaf4fcc289e645e0880a8"
                "7b8e55858df4cf2291a7303ffda446b82a117b4dd408cff2811adf92201088e0"
                "51ee48b57aecf46edfc68e5baeed5ae4910ba5681d370f75ab593811e18293ef"
                "0808581c254196bcbf2b4c454136a6711b00"
            ),
            false,
        ),
        (
            &invalid_message,
            hex!(
                "5db94c53101f521f6c1f43b60ea4d7e06fbd49c2e8afaf4fcc289e645e0880a8"
                "7b8e55858df4cf2291a7303ffda446b82a117b4dd408cff2825e06c3999e8308"
                "be439c40940b0075d3e4f65147c1608cbe6e9c432e33bed6686f9393ae2568f0"
                "ad60febcb4b6179c0d90d034e7c3c4681000"
            ),
            false,
        ),
        (
            &invalid_message,
            hex!(
                "5db94c53101f521f6c1f43b60ea4d7e06fbd49c2e8afaf4fcc289e645e0880a8"
                "7b8e55858df4cf2291a7303ffda446b82a117b4dd408cff2c02456bbd141df04"
                "8dbf1843be6d5fef402483314c2af547b361a09f3319489eaede43404df9faf6"
                "34c1298d678b5261c808b0be3726013e3900"
            ),
            false,
        ),
    ];

    let verifying_key = VerifyingKey::from_bytes(&public_key).expect("valid public key");
    for (message, signature, valid) in cases {
        // `Signature::from_bytes` decodes R, so an invalid R is refused at parse or at verify.
        let accepted = Signature::from_bytes(&signature)
            .ok()
            .is_some_and(|signature| verifying_key.verify_raw(&signature, message).is_ok());
        assert_eq!(accepted, valid);
    }
}
