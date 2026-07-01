//! Permanent regression fixtures for the pure-Rust zopfli port.
//!
//! These pinned hex blobs were generated when the FFI oracle was still
//! in place and confirmed byte-for-byte against `ZopfliCompress`. After
//! the cutover (Steps 21–24) they remain as the long-term guard against
//! drift — any change to the encoder that alters output bytes will
//! surface here without needing the C library.
//!
//! To regenerate after a deliberate encoder change, run
//! `GZIPPY_REGEN_FIXTURES=1 cargo test -p gzippy --release zopfli_pure::tests`
//! once; the test writes the new bytes to `fixtures/*.hex` and then
//! re-asserts.

use super::{compress, ZopfliFormat, ZopfliOptions};
use std::path::PathBuf;

fn opts_default() -> ZopfliOptions {
    ZopfliOptions::default()
}

fn opts_iter1() -> ZopfliOptions {
    ZopfliOptions {
        numiterations: 1,
        ..ZopfliOptions::default()
    }
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{:02x}", b).unwrap();
    }
    s
}

fn from_hex(s: &str) -> Vec<u8> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(s.len().is_multiple_of(2), "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("invalid hex"))
        .collect()
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("src/backends/zopfli_pure/fixtures");
    p.push(format!("{name}.hex"));
    p
}

/// Compare `got` to the pinned `fixtures/<name>.hex`. If
/// `GZIPPY_REGEN_FIXTURES` is set in the env, the file is rewritten
/// instead — useful after a deliberate encoder change.
fn assert_fixture(name: &str, got: &[u8]) {
    let path = fixture_path(name);
    let regen = std::env::var_os("GZIPPY_REGEN_FIXTURES").is_some();
    if regen {
        std::fs::write(&path, format!("{}\n", to_hex(got))).unwrap();
        eprintln!("regenerated {}", path.display());
        return;
    }
    let expected_hex = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing {}: {} (set GZIPPY_REGEN_FIXTURES=1 to seed)",
            path.display(),
            e
        )
    });
    let expected = from_hex(expected_hex.trim());
    if got != expected.as_slice() {
        panic!(
            "{}: encoder drift.\nexpected ({} bytes): {}\n     got ({} bytes): {}\n(if intentional, rerun with GZIPPY_REGEN_FIXTURES=1)",
            name,
            expected.len(),
            to_hex(&expected),
            got.len(),
            to_hex(got)
        );
    }
}

// Each test below pins one (input, options, format) combo. Inputs are
// kept tiny so the hex blob stays readable when reviewing diffs.

#[test]
fn fixture_empty_gzip_iter15() {
    let got = compress(&opts_default(), ZopfliFormat::Gzip, b"");
    assert_fixture("empty_gzip_iter15", &got);
}

#[test]
fn fixture_hello_world_gzip_iter15() {
    let got = compress(&opts_default(), ZopfliFormat::Gzip, b"hello world");
    assert_fixture("hello_world_gzip_iter15", &got);
}

#[test]
fn fixture_repeating_text_gzip_iter15() {
    let input = b"the quick brown fox jumps over the lazy dog\n".repeat(8);
    let got = compress(&opts_default(), ZopfliFormat::Gzip, &input);
    assert_fixture("repeating_text_gzip_iter15", &got);
}

#[test]
fn fixture_zeros_4k_gzip_iter15() {
    let input = vec![0u8; 4096];
    let got = compress(&opts_default(), ZopfliFormat::Gzip, &input);
    assert_fixture("zeros_4k_gzip_iter15", &got);
}

#[test]
fn fixture_hello_world_deflate_iter1() {
    let got = compress(&opts_iter1(), ZopfliFormat::Deflate, b"hello world");
    assert_fixture("hello_world_deflate_iter1", &got);
}

#[test]
fn fixture_alphabet_zlib_iter15() {
    let got = compress(
        &opts_default(),
        ZopfliFormat::Zlib,
        b"abcdefghijklmnopqrstuvwxyz",
    );
    assert_fixture("alphabet_zlib_iter15", &got);
}

/// Sanity round-trip: every gzip fixture must decode back to the input
/// via flate2's gunzip. This catches encoder bugs that would change the
/// hex blob in a way that still happens to match (vanishingly unlikely
/// but cheap to guard against).
#[test]
fn fixtures_roundtrip_via_flate2() {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let cases: &[(&str, Vec<u8>)] = &[
        ("empty", Vec::new()),
        ("hello", b"hello world".to_vec()),
        (
            "repeating",
            b"the quick brown fox jumps over the lazy dog\n".repeat(8),
        ),
        ("zeros_4k", vec![0u8; 4096]),
    ];
    for (name, input) in cases {
        let gz = compress(&opts_default(), ZopfliFormat::Gzip, input);
        let mut out = Vec::new();
        GzDecoder::new(&gz[..])
            .read_to_end(&mut out)
            .unwrap_or_else(|e| panic!("{}: gunzip failed: {}", name, e));
        assert_eq!(&out, input, "{}: roundtrip mismatch", name);
    }
}
