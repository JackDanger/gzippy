//! PROBE (JOB 2 diagnosis): does ISA-L's END_OF_BLOCK stopping point record
//! boundaries on STORED (BTYPE=00) and FIXED-Huffman (BTYPE=01) blocks?
//!
//! The gzip_chunk.rs decline (the `end_bit<=stop_hint` BFINAL-only accept added
//! in 19add96c) fires when `decompress_deflate_from_bit_with_boundaries` returns
//! ZERO boundaries at-or-past the stop hint on stored/fixed input. This probe
//! constructs deflate streams dominated by stored and fixed blocks and reports
//! the boundary count, so the actual mechanism is observed first-hand (not
//! inferred). x86_64 + isal-compression only (ISA-L FFI).
#![cfg(all(test, feature = "isal-compression", target_arch = "x86_64"))]

use crate::backends::isal_decompress::{
    decompress_deflate_from_bit_into, decompress_deflate_from_bit_with_boundaries,
};

/// Build a raw-deflate stream of N stored blocks, each carrying `chunk` bytes
/// of payload, the last with BFINAL=1. Stored blocks are byte-aligned, so this
/// is the simplest multi-block deflate to construct by hand.
fn stored_blocks(payloads: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, p) in payloads.iter().enumerate() {
        let last = i + 1 == payloads.len();
        // Stored block: byte-aligned. First byte: BFINAL (bit0), BTYPE=00 (bits1-2).
        out.push(if last { 0x01 } else { 0x00 });
        let len = p.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(p);
    }
    out
}

#[test]
fn probe_stored_block_boundaries() {
    // 6 stored blocks, distinct payloads so boundaries are unambiguous.
    let p0 = b"AAAAAAAAAAAAAAAA".as_slice();
    let p1 = b"BBBBBBBBBBBBBBBB".as_slice();
    let p2 = b"CCCCCCCCCCCCCCCC".as_slice();
    let p3 = b"DDDDDDDDDDDDDDDD".as_slice();
    let p4 = b"EEEEEEEEEEEEEEEE".as_slice();
    let p5 = b"FFFFFFFFFFFFFFFF".as_slice();
    let payloads = [p0, p1, p2, p3, p4, p5];
    let deflate = stored_blocks(&payloads);
    let expected: Vec<u8> = payloads.iter().flat_map(|p| p.iter().copied()).collect();

    let mut crc = crc32fast::Hasher::new();
    let res = decompress_deflate_from_bit_with_boundaries(
        &deflate,
        0,
        &[],
        expected.len() + 64,
        &mut crc,
    );

    match res {
        Some((out, end_bit, bounds)) => {
            eprintln!(
                "[probe stored] decoded {} bytes (expected {}), end_bit={}, boundaries={}",
                out.len(),
                expected.len(),
                end_bit,
                bounds.len()
            );
            for (i, b) in bounds.iter().enumerate() {
                eprintln!(
                    "   boundary[{i}] bit_offset={} output_offset={}",
                    b.bit_offset, b.output_offset
                );
            }
            assert_eq!(out, expected, "[probe stored] bytes must match");
            eprintln!(
                "[probe stored] RESULT: {} boundaries recorded for 6 stored blocks",
                bounds.len()
            );
        }
        None => {
            eprintln!("[probe stored] RESULT: returned None (declined) on 6 stored blocks");
            panic!("ISA-L declined on a simple stored-block stream");
        }
    }
}

/// First deflate block byte low 3 bits => BFINAL(bit0) + BTYPE(bits1-2).
fn first_block_btype(raw: &[u8]) -> u8 {
    (raw[0] >> 1) & 0x03
}

fn deflate_raw(plain: &[u8], level: u32) -> Vec<u8> {
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut e = DeflateEncoder::new(Vec::new(), Compression::new(level));
    e.write_all(plain).unwrap();
    e.finish().unwrap()
}

#[test]
fn probe_large_stored_stream() {
    let block = vec![0x5Au8; 65535];
    let refs: Vec<&[u8]> = (0..64).map(|_| block.as_slice()).collect();
    let deflate = stored_blocks(&refs);
    let expected_len = 65535usize * 64;

    let mut crc = crc32fast::Hasher::new();
    let res =
        decompress_deflate_from_bit_with_boundaries(&deflate, 0, &[], expected_len + 64, &mut crc);
    match res {
        Some((out, end_bit, bounds)) => {
            eprintln!(
                "[probe large-stored] decoded {} (expected {}), end_bit={}, boundaries={}",
                out.len(),
                expected_len,
                end_bit,
                bounds.len()
            );
            assert_eq!(out.len(), expected_len);
        }
        None => panic!("[probe large-stored] declined"),
    }

    let mut out = vec![0u8; expected_len + 64];
    let res2 = decompress_deflate_from_bit_into(&deflate, 0, &[], &mut out);
    match res2 {
        Some((written, end_bit, bounds)) => {
            eprintln!(
                "[probe large-stored _into] written={} end_bit={} boundaries={}",
                written,
                end_bit,
                bounds.len()
            );
        }
        None => eprintln!("[probe large-stored _into] returned None (under-reserve/decline)"),
    }
}

#[test]
fn probe_fixed_huffman_boundaries() {
    for (label, plain, level) in [
        ("tiny-ascii L1", b"hello world hello world".to_vec(), 1u32),
        ("zeros-1k L1", vec![0u8; 1024], 1u32),
        ("zeros-256k L1", vec![0u8; 256 * 1024], 1u32),
        (
            "rand-ish L0",
            (0..4096u32)
                .map(|i| (i.wrapping_mul(2654435761u32) >> 24) as u8)
                .collect(),
            0u32,
        ),
    ] {
        let raw = deflate_raw(&plain, level);
        let bt = first_block_btype(&raw);
        let mut crc = crc32fast::Hasher::new();
        let res =
            decompress_deflate_from_bit_with_boundaries(&raw, 0, &[], plain.len() + 1024, &mut crc);
        match res {
            Some((out, end_bit, bounds)) => {
                let ok = out == plain;
                eprintln!(
                    "[probe fixed:{label}] first_btype={bt} raw_len={} decoded={} bytes_ok={ok} end_bit={end_bit} boundaries={}",
                    raw.len(), out.len(), bounds.len()
                );
            }
            None => eprintln!(
                "[probe fixed:{label}] first_btype={bt} raw_len={} => DECLINED (None)",
                raw.len()
            ),
        }
    }
}

/// Mirror the repo's make_btype01_heavy_data (routing.rs:1141) — 70% random /
/// 30% short-phrase, which flate2 DEFAULT (level 6) compresses with many
/// FIXED-Huffman (BTYPE=01) blocks (the data is near-incompressible so dynamic
/// tables don't pay). Used by test_coalesce_fixed_huffman_multithread_byte_exact.
fn make_btype01_heavy_data(size: usize) -> Vec<u8> {
    let phrases: &[&[u8]] = &[b"abc", b"foo bar ", b"the quick brown ", b"hello ", b"xyz "];
    let mut data = Vec::with_capacity(size);
    let mut rng: u64 = 0xb0bd1ec0de;
    while data.len() < size {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng >> 32) % 100 < 70 {
            data.push((rng >> 16) as u8);
        } else {
            let phrase = phrases[(rng as usize) % phrases.len()];
            let to_take = phrase.len().min(size - data.len());
            data.extend_from_slice(&phrase[..to_take]);
        }
    }
    data.truncate(size);
    data
}

/// Materialize the GENUINE fixed-Huffman fixture (flate2 default, as the repo's
/// fixed-Huffman coalesce test does) as a gzip member on /dev/shm so the release
/// gzippy-isal binary can be run on it to read isal_chunks/isal_fallbacks.
/// `--ignored` (writes a 40 MB file). Run:
///   cargo test --release --features gzippy-isal materialize_fixed_huffman_gz -- --ignored --nocapture
#[test]
#[ignore = "materializes /dev/shm/fixed_huffman.gz for the production coverage probe"]
fn materialize_fixed_huffman_gz() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let original = make_btype01_heavy_data(40 * 1024 * 1024);
    // flate2 DEFAULT = level 6 — what the repo's fixed-Huffman test uses.
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&original).unwrap();
    let gz = enc.finish().unwrap();
    // Report the first deflate block's BTYPE so we can confirm fixed-Huffman.
    // skip gzip header (no extra/name with flate2): 10-byte header.
    let bt = (gz[10] >> 1) & 0x03;
    eprintln!(
        "[materialize] orig={} gz={} (>{:.1} MiB gate), first_btype={} (1=FIXED)",
        original.len(),
        gz.len(),
        gz.len() as f64 / 1048576.0,
        bt
    );
    std::fs::write("/dev/shm/fixed_huffman.gz", &gz).unwrap();
    std::fs::write("/dev/shm/fixed_huffman.raw", &original).unwrap();
    eprintln!("[materialize] wrote /dev/shm/fixed_huffman.gz + .raw");
}

/// ADVERSARIAL fixture (advisor-owed): MANY TINY fixed/stored blocks via frequent
/// SYNC_FLUSH. Each flush ends a deflate block + inserts an empty stored block,
/// so a chunk's stop_hint rarely coincides EXACTLY with a recorded boundary —
/// the regime that maximizes the `until_exact` exact-match decline. ~40 MB raw of
/// low-redundancy data flushed every ~2 KiB => tens of thousands of blocks.
/// Run: cargo test --release --features gzippy-isal materialize_tiny_block_gz -- --ignored --nocapture
#[test]
#[ignore = "materializes /dev/shm/tinyblocks.gz — adversarial many-tiny-block stress"]
fn materialize_tiny_block_gz() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let original = make_btype01_heavy_data(40 * 1024 * 1024);
    // Level 1 = most fixed/stored emissions; flush every 2 KiB to force a block
    // boundary at non-round output offsets.
    let mut enc = GzEncoder::new(Vec::new(), Compression::new(1));
    let mut off = 0usize;
    const STEP: usize = 2048;
    while off < original.len() {
        let end = (off + STEP).min(original.len());
        enc.write_all(&original[off..end]).unwrap();
        enc.flush().unwrap(); // Z_SYNC_FLUSH -> ends the block, empty stored block
        off = end;
    }
    let gz = enc.finish().unwrap();
    eprintln!(
        "[materialize-tiny] orig={} gz={} ({:.1} MiB), ~{} flushed blocks",
        original.len(),
        gz.len(),
        gz.len() as f64 / 1048576.0,
        original.len() / STEP
    );
    std::fs::write("/dev/shm/tinyblocks.gz", &gz).unwrap();
    std::fs::write("/dev/shm/tinyblocks.raw", &original).unwrap();
    eprintln!("[materialize-tiny] wrote /dev/shm/tinyblocks.gz + .raw");
}

/// ROOT-CAUSE disambiguation (pass-2 advisor-owed): on the ACTUAL tiny-block
/// SYNC_FLUSH stream, does `decompress_deflate_from_bit_with_boundaries` record
/// per-block boundaries (=> decline is accept-logic) or ZERO boundaries (=> decline
/// is absent-boundaries, the production comment's claim)? Decode the raw deflate of
/// /dev/shm/tinyblocks.gz from bit 0 with a generous cap and report the boundary
/// count + whether boundaries land at the SYNC_FLUSH cadence.
/// Run: cargo test --release --features gzippy-isal probe_tinyblocks_boundaries -- --ignored --nocapture
#[test]
#[ignore = "reads /dev/shm/tinyblocks.gz (materialize_tiny_block_gz first)"]
fn probe_tinyblocks_boundaries() {
    let gz = std::fs::read("/dev/shm/tinyblocks.gz").expect("run materialize_tiny_block_gz first");
    // flate2 GzEncoder header is 10 bytes (no extra/name); raw deflate follows,
    // trailer is last 8 bytes (CRC32 + ISIZE).
    let raw = &gz[10..gz.len() - 8];
    let mut crc = crc32fast::Hasher::new();
    let res = decompress_deflate_from_bit_with_boundaries(raw, 0, &[], 64 * 1024 * 1024, &mut crc);
    match res {
        Some((out, end_bit, bounds)) => {
            eprintln!(
                "[probe tinyblocks] raw_deflate={} decoded={} end_bit={} BOUNDARIES={}",
                raw.len(),
                out.len(),
                end_bit,
                bounds.len()
            );
            // Show the first 8 boundary spacings (output bytes between boundaries).
            let mut prev = 0usize;
            for (i, b) in bounds.iter().take(8).enumerate() {
                eprintln!(
                    "   boundary[{i}] bit={} out_off={} (+{} bytes since prev)",
                    b.bit_offset,
                    b.output_offset,
                    b.output_offset - prev
                );
                prev = b.output_offset;
            }
            // KEY: if BOUNDARIES ~= number of SYNC_FLUSH blocks (tens of thousands)
            // => boundaries ARE recorded on this stream (decline is accept-logic).
            // If BOUNDARIES is ~0/tiny => boundaries are ABSENT (production comment).
            eprintln!(
                "[probe tinyblocks] VERDICT: {} boundaries for a ~20480-flush stream",
                bounds.len()
            );
        }
        None => eprintln!("[probe tinyblocks] returned None (declined at wrapper)"),
    }
}
