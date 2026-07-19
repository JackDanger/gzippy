//! Correctness net for the pure-Rust PARALLEL DEFLATE encoder (Increment 6:
//! `PipelinedGzEncoder::compress_buffer_pure`, gated on `pure-rust-encoder`).
//!
//! The parallel path splits the input on a deterministic, data-length-only
//! block grid, compresses each chunk independently with the previous chunk's
//! 32 KiB input tail seeded as a preset dictionary, closes every non-final
//! chunk with a byte-aligned sync-flush marker, and concatenates the results
//! into ONE standard single-member gzip stream (header + DEFLATE + combined
//! CRC32/ISIZE). This module pins:
//!
//!   1. DETERMINISM — output is byte-identical across thread counts (the grid
//!      does not depend on `num_threads`).
//!   2. 3-ORACLE roundtrip — flate2, libdeflate, and system `gzip -d` all
//!      reproduce the input byte-exact at L1/L6/L9/L12.
//!   3. proptest — tiny (<1 chunk), incompressible (each chunk stored-escapes),
//!      and cross-chunk-boundary match inputs stay valid.
//!   4. CRC32 + ISIZE trailer bytes are correct.
//!
//! Run: `cargo test --release --features pure-rust-encoder pure_parallel_encoder`.

#![cfg(feature = "pure-rust-encoder")]

use crate::compress::pipelined::PipelinedGzEncoder;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

/// Compress `data` at `level` with `threads` through the pure parallel path.
fn compress_pure(data: &[u8], level: u32, threads: usize) -> Vec<u8> {
    let encoder = PipelinedGzEncoder::new(level, threads);
    let mut out = Vec::new();
    encoder.compress_buffer_pure(data, &mut out).unwrap();
    out
}

// ---- the three independent decoders ----

fn decode_flate2(gz: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(gz)
        .read_to_end(&mut out)
        .expect("flate2 failed to decode our parallel gzip stream");
    out
}

fn decode_libdeflate(gz: &[u8], expected_len: usize) -> Vec<u8> {
    let mut decomp = libdeflater::Decompressor::new();
    let mut out = vec![0u8; expected_len.max(1)];
    let n = decomp
        .gzip_decompress(gz, &mut out)
        .expect("libdeflate failed to decode our parallel gzip stream");
    out.truncate(n);
    out
}

/// `gzip -dc`; `None` only when no `gzip` binary is on PATH.
fn decode_system_gzip(gz: &[u8]) -> Option<Vec<u8>> {
    let mut child = Command::new("gzip")
        .arg("-dc")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take().unwrap();
    let buf = gz.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&buf);
    });
    let mut out = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut out).unwrap();
    writer.join().unwrap();
    assert!(child.wait().unwrap().success(), "gzip -dc exited non-zero");
    Some(out)
}

/// All three oracles must reproduce `input` byte-exact from `gz`.
fn assert_three_oracle(gz: &[u8], input: &[u8], ctx: &str) {
    assert_eq!(decode_flate2(gz), input, "flate2 mismatch: {ctx}");
    assert_eq!(
        decode_libdeflate(gz, input.len()),
        input,
        "libdeflate mismatch: {ctx}"
    );
    if let Some(sys) = decode_system_gzip(gz) {
        assert_eq!(sys, input, "gzip -d mismatch: {ctx}");
    }
}

/// Deterministic mixed text+binary corpus of `len` bytes (multi-chunk when
/// `len` exceeds the ~128 KiB pipelined block size).
fn mixed_corpus(len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let phrases: [&[u8]; 5] = [
        b"the quick brown fox jumps over the lazy dog; ",
        b"cross-chunk back-references must resolve against the seeded window. ",
        b"lorem ipsum dolor sit amet consectetur adipiscing elit; ",
        b"pub fn compress_block_streaming(data, dict, level, is_last, out) {} ",
        b"0123456789abcdef 0123456789abcdef structure structure structure ",
    ];
    let mut i = 0usize;
    while v.len() < len {
        v.extend_from_slice(phrases[i % phrases.len()]);
        let x = (i.wrapping_mul(2654435761)) as u32;
        v.extend_from_slice(&x.to_le_bytes());
        i += 1;
    }
    v.truncate(len);
    v
}

/// Output is byte-identical across thread counts (grid is data-length-only).
#[test]
fn deterministic_across_thread_counts() {
    let input = mixed_corpus(1_200_000); // ~9 chunks at 128 KiB
    for level in [1u32, 6, 9, 12] {
        let t1 = compress_pure(&input, level, 1);
        let t4 = compress_pure(&input, level, 4);
        let t16 = compress_pure(&input, level, 16);
        assert_eq!(t1, t4, "T1 != T4 at L{level}");
        assert_eq!(t1, t16, "T1 != T16 at L{level}");
        // And it must still be a correct stream.
        assert_three_oracle(&t1, &input, &format!("determinism L{level}"));
    }
}

/// Multi-chunk silesia-style input roundtrips through all three oracles.
#[test]
fn three_oracle_multichunk() {
    let input = mixed_corpus(900_000);
    for level in [1u32, 6, 9, 12] {
        for threads in [2usize, 8] {
            let gz = compress_pure(&input, level, threads);
            assert_three_oracle(&gz, &input, &format!("L{level} T{threads}"));
        }
    }
}

/// CRC32 + ISIZE trailer bytes are exactly the combined CRC of the input and
/// its length mod 2^32.
#[test]
fn crc_and_isize_trailer_correct() {
    let input = mixed_corpus(700_000);
    let gz = compress_pure(&input, 6, 4);
    let n = gz.len();
    let crc = u32::from_le_bytes(gz[n - 8..n - 4].try_into().unwrap());
    let isize = u32::from_le_bytes(gz[n - 4..n].try_into().unwrap());
    assert_eq!(crc, crc32fast::hash(&input), "trailer CRC32 wrong");
    assert_eq!(isize, input.len() as u32, "trailer ISIZE wrong");
}

/// Empty input yields a valid empty gzip member.
#[test]
fn empty_input_valid() {
    let gz = compress_pure(&[], 6, 4);
    assert_three_oracle(&gz, &[], "empty");
}

/// Input smaller than one chunk still routes cleanly (single is_last chunk).
#[test]
fn sub_chunk_input_valid() {
    for len in [1usize, 37, 4096, 65_000] {
        let input = mixed_corpus(len);
        for level in [1u32, 9, 12] {
            let gz = compress_pure(&input, level, 4);
            assert_three_oracle(&gz, &input, &format!("subchunk len={len} L{level}"));
        }
    }
}

mod prop {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 96, max_shrink_iters: 2000, ..ProptestConfig::default() })]

        /// Tiny (<1 chunk) inputs stay valid.
        #[test]
        fn tiny_inputs(data in proptest::collection::vec(any::<u8>(), 0..600)) {
            for level in [1u32, 6, 12] {
                let gz = compress_pure(&data, level, 4);
                prop_assert_eq!(decode_flate2(&gz), data.clone());
            }
        }

        /// Incompressible multi-chunk inputs (each chunk stored-escapes) stay
        /// valid single-member streams.
        #[test]
        fn incompressible_multichunk(seed in any::<u64>()) {
            // ~400 KiB of PRNG bytes → several chunks, all stored-escaping.
            let mut s = seed | 1;
            let mut data = vec![0u8; 400_000];
            for b in data.iter_mut() {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                *b = s as u8;
            }
            for level in [1u32, 6, 12] {
                let gz = compress_pure(&data, level, 8);
                prop_assert_eq!(decode_flate2(&gz), data.clone());
            }
        }

        /// A repeated motif spanning chunk boundaries exercises cross-chunk
        /// dictionary back-references (offset <= 32768).
        #[test]
        fn cross_boundary_matches(
            motif in proptest::collection::vec(any::<u8>(), 8..4000),
            reps in 200usize..1200,
        ) {
            let mut data = Vec::with_capacity(motif.len() * reps);
            for _ in 0..reps {
                data.extend_from_slice(&motif);
            }
            for level in [1u32, 6, 12] {
                let gz = compress_pure(&data, level, 8);
                prop_assert_eq!(decode_flate2(&gz), data.clone());
            }
        }
    }
}
