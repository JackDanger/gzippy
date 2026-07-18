//! Three-oracle differential testing for the inflate path.
//!
//! We cannot validate correctness of any alternative inflate without
//! agreement from multiple independent reference implementations.
//!
//! ## Oracles
//!
//! - **libdeflate** (via `libdeflate-sys`) — Eric Biggers' modern
//!   inflate; ~30-50% faster than zlib for in-memory ops.
//! - **zlib-ng** (via `libz-ng-sys`) — Mark Adler's reference zlib
//!   lineage with SIMD optimizations. Different code path from
//!   libdeflate; an algorithmic disagreement between them is real.
//! - **gzippy** under test (pure-Rust inflate, current production
//!   path through `decompress::decompress_gzip_libdeflate`).
//!
//! A third oracle (`rapidgzip` via C-ABI shim) is deferred until
//! needed. Two oracles catch algorithmic divergences; the third adds
//! independence-of-lineage but doesn't change the verdict for routine
//! deflate.
//!
//! ## Methodology
//!
//! For each test input:
//! 1. Compress via flate2 (or accept raw gzip).
//! 2. Decode via libdeflate-sys.
//! 3. Decode via libz-ng-sys.
//! 4. Decode via gzippy.
//! 5. Three-way assert byte-for-byte equality.
//!
//! A two-way disagreement is a bug. Three-way agreement is high
//! confidence. We surface the FIRST disagreement found; downstream
//! cargo-fuzz will explore broader inputs.

#![cfg(test)]

use std::io::Write;

// ── Oracle 1: libdeflate via libdeflate-sys ─────────────────────────────────

/// Decode a complete gzip stream via libdeflate. Returns the decoded
/// bytes or panics with the libdeflate error code (intentional: this
/// is an oracle, divergence is a test bug not a runtime path).
fn decode_via_libdeflate(gz: &[u8], expected_size: usize) -> Vec<u8> {
    use std::ptr::NonNull;

    let decompressor = unsafe { libdeflate_sys::libdeflate_alloc_decompressor() };
    let decompressor = NonNull::new(decompressor).expect("libdeflate_alloc_decompressor");

    let mut out = vec![0u8; expected_size];
    let mut out_nbytes = 0usize;
    let mut in_nbytes = 0usize;
    let result = unsafe {
        libdeflate_sys::libdeflate_gzip_decompress_ex(
            decompressor.as_ptr(),
            gz.as_ptr() as *const _,
            gz.len(),
            out.as_mut_ptr() as *mut _,
            out.len(),
            &mut in_nbytes,
            &mut out_nbytes,
        )
    };
    unsafe { libdeflate_sys::libdeflate_free_decompressor(decompressor.as_ptr()) };

    assert_eq!(
        result,
        0,
        "libdeflate_gzip_decompress_ex failed: code {result}, input {} bytes",
        gz.len()
    );
    out.truncate(out_nbytes);
    out
}

// ── Oracle 2: zlib-ng via libz-ng-sys ───────────────────────────────────────

/// Decode a complete gzip stream via zlib-ng. Uses
/// `inflateInit2(31)` for gzip wrapper auto-detection.
fn decode_via_zlibng(gz: &[u8], expected_size_hint: usize) -> Vec<u8> {
    use libz_ng_sys as zng;
    use std::mem;
    use std::ptr;

    let mut strm: zng::z_stream = unsafe {
        let mut m = mem::MaybeUninit::<zng::z_stream>::uninit();
        ptr::write_bytes(m.as_mut_ptr(), 0, 1);
        m.assume_init()
    };

    // windowBits=31 = gzip-only wrapper (16 + 15 bits window).
    let ret = unsafe {
        zng::inflateInit2_(
            &mut strm,
            31,
            ptr::null(),
            mem::size_of::<zng::z_stream>() as i32,
        )
    };
    assert_eq!(ret, zng::Z_OK, "zlib-ng inflateInit2_(31) failed: {ret}");

    strm.next_in = gz.as_ptr() as *mut _;
    strm.avail_in = gz.len() as u32;

    // Grow output buffer as needed; start with hint and double on overflow.
    let mut output = vec![0u8; expected_size_hint.max(4096)];
    let mut out_pos = 0usize;

    loop {
        if out_pos >= output.len() {
            output.resize(output.len() * 2, 0);
        }
        strm.next_out = unsafe { output.as_mut_ptr().add(out_pos) };
        strm.avail_out = (output.len() - out_pos) as u32;

        let ret = unsafe { zng::inflate(&mut strm, zng::Z_NO_FLUSH) };
        let written = (output.len() - out_pos) - strm.avail_out as usize;
        out_pos += written;

        if ret == zng::Z_STREAM_END {
            break;
        }
        if ret == zng::Z_OK || ret == zng::Z_BUF_ERROR {
            if written == 0 && strm.avail_in == 0 {
                break;
            }
            continue;
        }
        panic!("zlib-ng inflate returned error {ret}");
    }

    unsafe { zng::inflateEnd(&mut strm) };
    output.truncate(out_pos);
    output
}

// ── Oracle 3: gzippy under test ─────────────────────────────────────────────

/// Decode via gzippy's pure-rust production path. Uses
/// `decompress_gzip_libdeflate` which is the single-threaded
/// entry that routes through the libdeflate-inner inflate.
///
/// We deliberately do NOT use the parallel-SM path here because it
/// has its own routing tests; this oracle's purpose is to validate
/// the INFLATE primitive, not the chunking dispatcher.
fn decode_via_gzippy(gz: &[u8]) -> Vec<u8> {
    // Use the single-threaded libdeflate path (which is what
    // pure-rust-inflate substitutes for in the alternative build).
    // This exercises the same inflate primitives the bake-off Routes
    // are competing to replace.
    let mut out = Vec::new();
    // Single-threaded path (num_threads=1) — exercises the
    // libdeflate-inner inflate primitive without parallel-SM chunking.
    crate::decompress::decompress_bytes(gz, &mut out, 1).expect("gzippy decode");
    out
}

// ── Differential harness ────────────────────────────────────────────────────

/// Read ISIZE from the gzip trailer for an exact output-size hint.
/// ISIZE is uncompressed size mod 2^32; for our test inputs (all
/// well under 4 GiB) this is the true size.
fn isize_from_trailer(gz: &[u8]) -> usize {
    assert!(gz.len() >= 8, "gzip stream shorter than trailer");
    let tail = &gz[gz.len() - 4..];
    u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as usize
}

/// Run the three-oracle differential on a single gzip input.
/// Panics with a structured message on disagreement.
fn assert_three_oracle_agree(gz: &[u8], label: &str) {
    // Exact size from ISIZE — libdeflate needs the right hint.
    let exact = isize_from_trailer(gz);

    let lib_out = decode_via_libdeflate(gz, exact);
    let zng_out = decode_via_zlibng(gz, exact);
    let gzp_out = decode_via_gzippy(gz);

    if lib_out != zng_out {
        // Pure C-vs-C oracle disagreement — extremely rare; either
        // a libdeflate vs zlib-ng difference or a malformed input.
        panic!(
            "[{label}] ORACLE DISAGREEMENT (libdeflate ≠ zlib-ng): \
             lib_out.len()={}, zng_out.len()={}, first divergence at byte {}",
            lib_out.len(),
            zng_out.len(),
            lib_out
                .iter()
                .zip(zng_out.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(lib_out.len().min(zng_out.len()))
        );
    }

    if gzp_out != lib_out {
        let divergence = gzp_out
            .iter()
            .zip(lib_out.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(gzp_out.len().min(lib_out.len()));
        panic!(
            "[{label}] gzippy DISAGREES with oracles (which agree): \
             gzippy.len()={}, oracle.len()={}, divergence at byte {} \
             (gzippy={:02x}, oracle={:02x})",
            gzp_out.len(),
            lib_out.len(),
            divergence,
            gzp_out.get(divergence).copied().unwrap_or(0),
            lib_out.get(divergence).copied().unwrap_or(0),
        );
    }

    // All three agree.
}

fn compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

// ── Marker-decoder (marker_inflate::Block) fuzz differential ──────────────────
//
// `decode_via_gzippy` above uses num_threads=1 and deliberately avoids the
// parallel-SM / marker path, so the window-absent marker decoder
// (`marker_inflate::Block`) had ZERO fuzz coverage (only one silesia md5). This
// decodes raw DEFLATE through `Block` directly (from offset 0, empty window →
// all output is clean, no markers emitted) and asserts byte-exactness vs both
// the libdeflate oracle and the original payload, over fuzzed inputs.
#[cfg(pure_inflate_decode)]
fn decode_via_deflate_block(gz: &[u8]) -> Vec<u8> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    use crate::decompress::parallel::marker_inflate::Block;
    use crate::decompress::parallel::replace_markers::MARKER_BASE;

    let (_h, header) =
        crate::decompress::parallel::gzip_format::read_header(gz).expect("parse gzip header");
    let raw = &gz[header..gz.len() - 8];
    let mut block = Block::new();
    block.reset(None, None);
    let mut bits = Bits::new(raw);
    let mut out16: Vec<u16> = Vec::new();
    loop {
        if block.read_header(&mut bits, false).is_err() {
            break;
        }
        while !block.eob() {
            block
                .read(&mut bits, &mut out16, usize::MAX)
                .expect("deflate_block read");
        }
        if block.is_last_block() {
            break;
        }
    }
    // From offset 0 with no predecessor window, no back-ref can reach before
    // the start → no markers → every value is a literal byte.
    out16
        .iter()
        .map(|&v| {
            assert!(
                v < MARKER_BASE,
                "offset-0 decode emitted a marker (={v:#x})"
            );
            v as u8
        })
        .collect()
}

#[cfg(pure_inflate_decode)]
#[test]
fn deflate_block_marker_decoder_fuzz_200_cases() {
    let mut rng_seed: u64 = 0x1234_5678_9abc_def0;
    for case in 0..200 {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let size = ((rng_seed >> 32) as usize) % (96 * 1024) + 1;
        let mut rng = rng_seed ^ 0xFEED_FACE_CAFE_BABE;
        let mut payload = Vec::with_capacity(size);
        // Mix random bytes with periodic repetitive runs so flate2 emits real
        // back-references/matches (exercises the match-copy in the Block loop),
        // not just literal/stored blocks.
        for i in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 40) & 7 == 0 {
                payload.push((i % 37) as u8);
            } else {
                payload.push((rng >> 24) as u8);
            }
        }
        let level = ((rng_seed >> 8) as u32) % 10;
        let gz = compress(&payload, level);
        let exact = isize_from_trailer(&gz);
        let oracle = decode_via_libdeflate(&gz, exact);
        let dblock = decode_via_deflate_block(&gz);
        assert_eq!(
            dblock, oracle,
            "deflate_block marker-decoder fuzz case {case} (size {size}, L{level}) disagrees with libdeflate"
        );
        assert_eq!(
            dblock, payload,
            "deflate_block marker-decoder fuzz case {case} (size {size}, L{level}) != original payload"
        );
    }
}

// ── Coverage tests: corpora across all DEFLATE block-type frequencies ──

#[test]
fn three_oracle_empty_payload() {
    let gz = compress(b"", 6);
    assert_three_oracle_agree(&gz, "empty");
}

#[test]
fn three_oracle_single_byte() {
    let gz = compress(b"x", 6);
    assert_three_oracle_agree(&gz, "single_byte");
}

#[test]
fn three_oracle_short_text() {
    let gz = compress(b"the quick brown fox jumps over the lazy dog", 6);
    assert_three_oracle_agree(&gz, "short_text");
}

#[test]
fn three_oracle_long_repetitive() {
    let payload = b"abcdefghijklmnopqrstuvwxyz".repeat(10_000);
    let gz = compress(&payload, 9);
    assert_three_oracle_agree(&gz, "long_repetitive");
}

#[test]
fn three_oracle_random_l0_stored_blocks() {
    // L0 forces stored blocks.
    let mut rng: u64 = 0xcafef00d_deadbeef;
    let mut data = Vec::with_capacity(256 * 1024);
    for _ in 0..(256 * 1024) {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        data.push((rng >> 24) as u8);
    }
    let gz = compress(&data, 0);
    assert_three_oracle_agree(&gz, "random_l0");
}

#[test]
fn three_oracle_random_l1_fixed_huffman() {
    // L1 on random data → mostly fixed-Huffman, which exercises the
    // RFC 1951 reserved-symbol handling.
    let mut rng: u64 = 0xcafef00d_deadbeef;
    let mut data = Vec::with_capacity(64 * 1024);
    for _ in 0..(64 * 1024) {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        data.push((rng >> 24) as u8);
    }
    let gz = compress(&data, 1);
    assert_three_oracle_agree(&gz, "random_l1_fixed");
}

#[test]
fn three_oracle_mixed_entropy_l9_dynamic_huffman() {
    let phrases: &[&[u8]] = &[b"abc", b"foo bar ", b"the quick brown ", b"hello ", b"xyz "];
    let mut rng: u64 = 0xb0bd1ec0de;
    let mut data = Vec::with_capacity(256 * 1024);
    while data.len() < 256 * 1024 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng >> 32) % 100 < 70 {
            data.push((rng >> 16) as u8);
        } else {
            let phrase = phrases[(rng as usize) % phrases.len()];
            let take = phrase.len().min(256 * 1024 - data.len());
            data.extend_from_slice(&phrase[..take]);
        }
    }
    let gz = compress(&data, 9);
    assert_three_oracle_agree(&gz, "mixed_l9_dynamic");
}

#[test]
fn three_oracle_multi_block_crossref() {
    // Large payload with long-distance cross-block back-references —
    // the shape that caught `decode_block` API multi-block back-ref bug.
    let mut payload = Vec::new();
    for i in 0..20_000 {
        payload.extend_from_slice(format!("entry-{i:08}-with-some-shared-suffix\n").as_bytes());
    }
    let shift = payload[..50_000].to_vec();
    payload.extend_from_slice(&shift);
    let gz = compress(&payload, 6);
    assert_three_oracle_agree(&gz, "multi_block_crossref");
}

// ── Real-corpus test: silesia, if available ───────────────────────────────

#[test]
fn three_oracle_silesia_if_available() {
    let candidates = [
        "benchmark_data/silesia-large.gz",
        "benchmark_data/silesia.tar.gz",
        "benchmark_data/silesia-gzip.tar.gz",
    ];
    let Some(gz) = candidates.iter().find_map(|p| std::fs::read(p).ok()) else {
        eprintln!("[three-oracle silesia] no silesia corpus available, skipping");
        return;
    };
    assert_three_oracle_agree(&gz, "silesia");
    eprintln!(
        "[three-oracle silesia] all three oracles agreed on {} bytes input",
        gz.len()
    );
}

// ── Property-based fuzz over random inputs ────────────────────────────────

/// Mini-fuzz: generate N random payloads with N different seeds, vary
/// compression level, three-oracle each. Bounded so `cargo test`
/// completes in seconds. The full ≥72h fuzz lives in a cargo-fuzz
/// workspace (deferred).
#[test]
fn three_oracle_mini_fuzz_50_cases() {
    let mut rng_seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for case in 0..50 {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let size = ((rng_seed >> 32) as usize) % (64 * 1024) + 1;
        let mut rng = rng_seed ^ 0xC0DE_FEED_DEAD_BEEF;
        let mut payload = Vec::with_capacity(size);
        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            payload.push((rng >> 24) as u8);
        }
        let level = ((rng_seed >> 8) as u32) % 10;
        let gz = compress(&payload, level);
        assert_three_oracle_agree(&gz, &format!("fuzz_case_{case}_size{size}_L{level}"));
    }
}

/// Extended fuzz soak — proxy for a 72h fuzz requirement. We can't run
/// 72h in a test session, but we CAN run a high-case-count batch (10,000
/// cases) to materially raise statistical confidence vs. the 50-case
/// mini-fuzz above.
///
/// The full 72h soak is a release-gate requirement that must run
/// pre-release in CI, not in unit-test budget. This test stands in for
/// the pre-release validation shape; zero disagreements across this scale
/// is necessary-but-not-sufficient for the full 72h gate.
///
/// Skipped by default (run with --ignored to opt in).
#[test]
#[ignore = "extended fuzz — run with --ignored"]
fn three_oracle_extended_fuzz_10k_cases() {
    let mut rng_seed: u64 = 0xB5C0_FACE_DEAD_BEEF;
    let mut max_size = 0usize;
    let mut total_bytes = 0usize;
    for case in 0..10_000 {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Bias toward smaller cases (more cases per minute) but
        // include occasional larger ones (more entropy coverage).
        let size = if case % 100 == 0 {
            ((rng_seed >> 32) as usize) % (1024 * 1024) + 1
        } else {
            ((rng_seed >> 32) as usize) % (64 * 1024) + 1
        };
        max_size = max_size.max(size);
        total_bytes += size;
        let mut rng = rng_seed ^ 0xC0DE_FEED_DEAD_BEEF;
        let mut payload = Vec::with_capacity(size);
        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            payload.push((rng >> 24) as u8);
        }
        let level = ((rng_seed >> 8) as u32) % 10;
        let gz = compress(&payload, level);
        assert_three_oracle_agree(&gz, &format!("ext_fuzz_{case}_size{size}_L{level}"));
        if case % 1000 == 999 {
            eprintln!(
                "extended_fuzz: case {} / 10000, max_size {} bytes, total {} MB",
                case + 1,
                max_size,
                total_bytes / 1_048_576
            );
        }
    }
    eprintln!(
        "extended_fuzz: 10000 / 10000 passed, max_size {} bytes, total {} MB",
        max_size,
        total_bytes / 1_048_576
    );
}
