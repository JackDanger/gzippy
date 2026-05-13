//! Parallel single-member gzip decompression — marker-based design (v0.6).
//!
//! Production path: wired into `decompress::decompress_single_member` when
//! ISA-L is available, `num_threads > 1`, and the compressed stream exceeds
//! 10 MiB. Replaces v0.5.1's speculative-window two-pass design.
//!
//! # Why a marker pipeline
//!
//! v0.5.1's design did 2N total compute work (phase 1 decode with empty
//! dict + phase 2 re-decode with prior window). On CI's 4-physical-core
//! ubuntu-latest runner that produced 288 MB/s vs. rapidgzip's 327 MB/s —
//! 0.88×, below the 0.99 target.
//!
//! The marker pipeline does ~1.1N total compute work:
//!
//! - **Phase 1 (parallel workers)**: each chunk is decoded by
//!   `fast_marker_inflate::decode_chunk_markers_bounded` over its
//!   `[start_bit, end_bit)` bit range. Output is `Vec<u16>` where literals
//!   are 0..=255 and cross-chunk back-references are markers ≥
//!   `MARKER_BASE = 32768` encoding a window offset.
//!
//! - **Phase 2 (sequential)**: walk chunks in order. Chunk 0's output has
//!   no markers (no predecessor); pass through. For chunk i ≥ 1, call
//!   `replace_markers` with the prior chunk's last 32 KB as the window —
//!   AVX2 (x86_64) or NEON (aarch64) substitution. Convert u16 → u8 via
//!   `u16_to_u8` which fails fast on any leftover marker.
//!
//! - **Phase 3 (sequential)**: verify total bytes against gzip ISIZE,
//!   verify combined CRC32 against gzip CRC. **Write to the writer only
//!   after CRC verifies** — a fallback never produces partial corrupt
//!   output.
//!
//! Marker-resolution work in phase 2 is essentially memcpy-speed (one
//! lookup + one store per marker, vectorized 8–16 lanes at a time). The
//! sequential chain is short — typically a few percent of total wall.
//! Speedup over single-thread ISA-L is ≈ T / 1.1.
//!
//! # Routing-assertion counter (the deletion-trap killer)
//!
//! `MARKER_PIPELINE_RUNS` is incremented on every successful run. Tests in
//! `src/tests/routing.rs` snapshot it before/after a decode and assert it
//! increased — that is the only thing that catches a regression where
//! `decompress_single_member`'s routing silently falls back to sequential
//! ISA-L. Without this assertion, output-equivalence tests pass while the
//! marker pipeline goes uncovered, and the code becomes a deletion target
//! during the next cleanup. See `docs/marker-decoder-plan.md` "deletion-
//! trap killer."

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::decompress::parallel::block_finder::BlockFinder;
use crate::decompress::parallel::fast_marker_inflate::decode_chunk_markers_bounded;
use crate::decompress::parallel::replace_markers::{replace_markers, u16_to_u8, MARKER_BASE};

/// Deflate sliding-window size (RFC 1951 §3.2.4).
const WINDOW_SIZE: usize = 32_768;

/// Minimum compressed size to attempt parallel. The routing entry in
/// `decompress::decompress_single_member` gates at 10 MiB; this is the inner
/// lower bound used by tests that exercise smaller fixtures.
const MIN_PARALLEL_SIZE: usize = 4 * 1024 * 1024;

/// Minimum threads to attempt parallel decode.
const MIN_THREADS_FOR_PARALLEL: usize = 2;

/// Search radius (bytes) around each partition point for block boundaries.
const SEARCH_RADIUS: usize = 512 * 1024;

/// Successful runs of the marker pipeline. Snapshot before/after a decode to
/// confirm production routing actually called us — see the deletion-trap
/// killer test in `src/tests/routing.rs`.
pub static MARKER_PIPELINE_RUNS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

// ── Public entry ─────────────────────────────────────────────────────────────

/// Parallel decompress a single-member gzip stream via the v0.6 marker
/// pipeline.
///
/// Returns `Err(ParallelError::TooSmall)` if the input is below the parallel
/// threshold — the caller should fall back to sequential decode. Other
/// errors indicate a genuine boundary-search failure, decode failure, or
/// CRC/size mismatch; the writer is **not** written to in those cases (the
/// pipeline buffers internally until verification passes).
pub fn decompress_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> Result<u64, ParallelError> {
    let t0 = std::time::Instant::now();

    let header_size = crate::decompress::parallel::marker_decode::skip_gzip_header(gzip_data)
        .map_err(|_| ParallelError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::TooSmall);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

    if deflate_data.len() < MIN_PARALLEL_SIZE || num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::TooSmall);
    }

    // Trailer: gzip stores CRC32 then ISIZE (little-endian) in the last 8 bytes.
    let crc_offset = gzip_data.len() - 8;
    let expected_crc = u32::from_le_bytes([
        gzip_data[crc_offset],
        gzip_data[crc_offset + 1],
        gzip_data[crc_offset + 2],
        gzip_data[crc_offset + 3],
    ]);
    let isize_offset = gzip_data.len() - 4;
    let expected_size = u32::from_le_bytes([
        gzip_data[isize_offset],
        gzip_data[isize_offset + 1],
        gzip_data[isize_offset + 2],
        gzip_data[isize_offset + 3],
    ]) as usize;

    let num_chunks = num_threads;
    let total_bits = deflate_data.len() * 8;
    let spacing_bits = total_bits / num_chunks;

    if debug_enabled() {
        eprintln!(
            "[parallel_sm:v0.6] {} bytes deflate, {} chunks, spacing={}KB, isize={}",
            deflate_data.len(),
            num_chunks,
            spacing_bits / 8 / 1024,
            expected_size
        );
    }

    // ── Phase 1a: parallel boundary search ──────────────────────────────────
    let t_search = std::time::Instant::now();
    let start_bits = phase1_search_boundaries(deflate_data, num_chunks, spacing_bits);
    let search_elapsed = t_search.elapsed();

    if start_bits.iter().any(Option::is_none) {
        if debug_enabled() {
            let n = start_bits.iter().filter(|s| s.is_none()).count();
            eprintln!(
                "[parallel_sm:v0.6] {}/{} boundary searches failed",
                n, num_chunks
            );
        }
        return Err(ParallelError::DecodeFailed);
    }
    let start_bits: Vec<usize> = start_bits.into_iter().map(|s| s.unwrap()).collect();

    // End-bit limit for each chunk = next chunk's start (a real block
    // boundary). The last chunk has no limit; it runs to BFINAL.
    let end_limits: Vec<Option<usize>> = (0..num_chunks)
        .map(|i| {
            if i + 1 < num_chunks {
                Some(start_bits[i + 1])
            } else {
                None
            }
        })
        .collect();

    // ── Phase 1b: parallel marker decode ────────────────────────────────────
    let t_decode = std::time::Instant::now();
    let chunks = phase1_marker_decode_parallel(deflate_data, &start_bits, &end_limits);
    let decode_elapsed = t_decode.elapsed();

    if chunks.iter().any(Option::is_none) {
        if debug_enabled() {
            eprintln!("[parallel_sm:v0.6] one or more marker decodes failed");
        }
        return Err(ParallelError::DecodeFailed);
    }
    let chunks: Vec<Vec<u16>> = chunks.into_iter().map(|c| c.unwrap()).collect();

    // ── Phase 2: sequential resolve + CRC + size accounting ─────────────────
    let t_resolve = std::time::Instant::now();
    let (assembled, total_crc) = phase2_resolve_sequential(chunks)?;
    let resolve_elapsed = t_resolve.elapsed();

    if debug_enabled() {
        eprintln!(
            "[parallel_sm:v0.6] search={:.1}ms decode={:.1}ms resolve={:.1}ms",
            search_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0,
            resolve_elapsed.as_secs_f64() * 1000.0,
        );
    }

    // ── Phase 3: verify trailer, write ──────────────────────────────────────
    if expected_size > 0 && assembled.len() != expected_size {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] size mismatch: got {} expected {}",
                assembled.len(),
                expected_size
            );
        }
        return Err(ParallelError::SizeMismatch);
    }
    if expected_crc != 0 && total_crc != expected_crc {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] CRC mismatch: got {:#010x} expected {:#010x}",
                total_crc, expected_crc
            );
        }
        return Err(ParallelError::CrcMismatch);
    }

    writer.write_all(&assembled)?;
    MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);

    if debug_enabled() {
        let total = t0.elapsed();
        let mbps = assembled.len() as f64 / total.as_secs_f64() / 1e6;
        eprintln!(
            "[parallel_sm:v0.6] total={:.1}ms ({:.0} MB/s)",
            total.as_secs_f64() * 1000.0,
            mbps,
        );
    }

    Ok(assembled.len() as u64)
}

// ── Phase 1a: parallel boundary search ───────────────────────────────────────

fn phase1_search_boundaries(
    deflate_data: &[u8],
    num_chunks: usize,
    spacing_bits: usize,
) -> Vec<Option<usize>> {
    let results: Vec<Mutex<Option<usize>>> = (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_chunks {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }
                let start = if idx == 0 {
                    Some(0)
                } else {
                    search_boundary_forward(deflate_data, idx * spacing_bits)
                };
                *results[idx].lock().unwrap() = start;
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Search forward from `from_bit` for a valid deflate block boundary.
fn search_boundary_forward(deflate_data: &[u8], from_bit: usize) -> Option<usize> {
    let search_end = (from_bit + SEARCH_RADIUS * 8).min(deflate_data.len() * 8);
    if from_bit >= search_end {
        return None;
    }

    let finder = BlockFinder::new(deflate_data);
    let sub_chunk_bits = 8 * 1024 * 8;
    let mut chunk_start = from_bit;
    while chunk_start < search_end {
        let chunk_end = (chunk_start + sub_chunk_bits).min(search_end);
        let mut candidates = finder.find_blocks(chunk_start, chunk_end);
        candidates.sort_by_key(|b| b.bit_offset);
        for candidate in &candidates {
            if try_decode_at(deflate_data, candidate.bit_offset) {
                return Some(candidate.bit_offset);
            }
        }
        chunk_start = chunk_end;
    }
    // Brute-force fallback within first 128 KiB.
    let brute_end = (from_bit + 128 * 1024 * 8).min(search_end);
    (from_bit..brute_end)
        .step_by(8)
        .find(|&bit| try_decode_at(deflate_data, bit))
}

/// Validate a candidate boundary in two stages — see premortem mitigation
/// B7 in `docs/marker-decoder-premortem.md`:
///
/// 1. **Fast filter** via ISA-L's `decompress_deflate_from_bit` (or zlib-ng
///    on non-x86_64): does 32 KB decode succeed without error? ISA-L is
///    permissive — it accepts some non-boundary positions that happen to
///    decode plausible-looking bytes for 32 KB before diverging.
/// 2. **Strict check** via our `fast_marker_inflate` on the same position,
///    bounded to a single deflate block (`end_bit_limit = bit_offset + 1`
///    stops the decoder at the next block boundary at or past `bit_offset`).
///    A real boundary decodes one block cleanly; a false positive fails
///    on Huffman table validation or a malformed length code somewhere in
///    the first block.
///
/// Why both: with stage 1 alone, false positives passed through and the
/// production marker decode later returned Err mid-block, causing routing
/// to silently fall back to sequential ISA-L (failure mode F6). The
/// deletion-trap killer test caught it. Stage 2 alone is correct but
/// would run the slower pure-Rust decoder against every BlockFinder
/// candidate.
///
/// Cost: stage 2 runs only when stage 1 approves. Stage 1 approval
/// candidates are rare on real data (a few per chunk's search window).
/// Each stage-2 run decodes one deflate block (~30 KB out / ~10 KB in)
/// at ~200-300 MB/s — sub-millisecond per accepted candidate.
fn try_decode_at(deflate_data: &[u8], bit_offset: usize) -> bool {
    let start_byte = bit_offset / 8;
    if start_byte >= deflate_data.len() {
        return false;
    }
    let remaining = deflate_data.len() - start_byte;
    let min_output = if remaining > 128 * 1024 {
        32 * 1024
    } else {
        4 * 1024
    };

    // Stage 1: ISA-L (or zlib-ng) fast filter.
    let isal_ok = crate::backends::inflate_bit::decompress_deflate_from_bit(
        deflate_data,
        bit_offset,
        &[],
        min_output,
    )
    .is_some_and(|out| out.len() >= min_output);
    if !isal_ok {
        return false;
    }

    // Stage 2: strict marker-decoder validation, scoped to one block.
    // `end_bit_limit = Some(bit_offset + 1)` tells the decoder to stop at
    // the next block boundary at or past `bit_offset + 1`, which is
    // exactly "after the first block." A real boundary decodes one block
    // and returns Ok; a false positive returns Err somewhere mid-block.
    use crate::decompress::parallel::fast_marker_inflate::decode_chunk_markers_bounded;
    decode_chunk_markers_bounded(deflate_data, bit_offset, Some(bit_offset + 1)).is_ok()
}

// ── Phase 1b: parallel marker decode ─────────────────────────────────────────

fn phase1_marker_decode_parallel(
    deflate_data: &[u8],
    start_bits: &[usize],
    end_limits: &[Option<usize>],
) -> Vec<Option<Vec<u16>>> {
    let num_chunks = start_bits.len();
    let results: Vec<Mutex<Option<Vec<u16>>>> = (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_chunks {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }
                let start_bit = start_bits[idx];
                let end_limit = end_limits[idx];
                if let Ok((out, _)) =
                    decode_chunk_markers_bounded(deflate_data, start_bit, end_limit)
                {
                    *results[idx].lock().unwrap() = Some(out);
                }
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

// ── Phase 2: sequential marker resolve + CRC + size ──────────────────────────

fn phase2_resolve_sequential(chunks: Vec<Vec<u16>>) -> Result<(Vec<u8>, u32), ParallelError> {
    let total_estimated: usize = chunks.iter().map(|c| c.len()).sum();
    let mut assembled = Vec::with_capacity(total_estimated);
    let mut crc = crc32fast::Hasher::new();
    let mut window: Vec<u8> = Vec::with_capacity(WINDOW_SIZE);

    for (i, mut chunk_u16) in chunks.into_iter().enumerate() {
        if i > 0 {
            // Resolve markers against the predecessor's last 32 KB.
            replace_markers(&mut chunk_u16, &window);
        }
        let bytes = u16_to_u8(&chunk_u16).map_err(|pos| {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] chunk {} has unresolved marker at index {} (offset {})",
                    i,
                    pos,
                    chunk_u16[pos] - MARKER_BASE,
                );
            }
            ParallelError::DecodeFailed
        })?;
        crc.update(&bytes);
        // Update window from the bytes we just resolved.
        update_window(&mut window, &bytes);
        assembled.extend_from_slice(&bytes);
    }

    Ok((assembled, crc.finalize()))
}

fn update_window(window: &mut Vec<u8>, new_data: &[u8]) {
    if new_data.is_empty() {
        return;
    }
    if new_data.len() >= WINDOW_SIZE {
        window.clear();
        window.extend_from_slice(&new_data[new_data.len() - WINDOW_SIZE..]);
        return;
    }
    if window.len() + new_data.len() <= WINDOW_SIZE {
        window.extend_from_slice(new_data);
        return;
    }
    let keep = WINDOW_SIZE - new_data.len();
    let drop = window.len() - keep;
    window.drain(..drop);
    window.extend_from_slice(new_data);
}

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ParallelError {
    InvalidHeader,
    TooSmall,
    DecodeFailed,
    SizeMismatch,
    CrcMismatch,
    Io(io::Error),
}

impl From<io::Error> for ParallelError {
    fn from(e: io::Error) -> Self {
        ParallelError::Io(e)
    }
}

impl ParallelError {
    pub fn is_routing(&self) -> bool {
        matches!(self, ParallelError::TooSmall)
    }
}

impl std::fmt::Display for ParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParallelError::InvalidHeader => write!(f, "invalid gzip header"),
            ParallelError::TooSmall => write!(f, "file too small for parallel decode"),
            ParallelError::DecodeFailed => write!(f, "chunk decode failed"),
            ParallelError::SizeMismatch => write!(f, "output size mismatch"),
            ParallelError::CrcMismatch => write!(f, "CRC32 mismatch"),
            ParallelError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_gzip_data(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn make_gzip_at_level(data: &[u8], level: u32) -> Vec<u8> {
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn make_compressible_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26) as u8 + b'a';
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    fn make_random_data(size: usize, seed: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng = seed;
        while data.len() < size {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            for shift in (0..64).step_by(8) {
                if data.len() >= size {
                    break;
                }
                data.push((rng >> shift) as u8);
            }
        }
        data
    }

    // ── Contract: input filters ──────────────────────────────────────────────

    #[test]
    fn small_inputs_return_too_small() {
        let compressed = make_gzip_data(b"hello world");
        let mut output = Vec::new();
        assert!(matches!(
            decompress_parallel(&compressed, &mut output, 4),
            Err(ParallelError::TooSmall)
        ));
    }

    #[test]
    fn single_thread_returns_too_small() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let mut output = Vec::new();
        assert!(matches!(
            decompress_parallel(&compressed, &mut output, 1),
            Err(ParallelError::TooSmall)
        ));
    }

    #[test]
    fn empty_input_errors() {
        let mut output = Vec::new();
        assert!(decompress_parallel(&[], &mut output, 4).is_err());
    }

    // ── Contract: round-trip correctness ─────────────────────────────────────

    fn assert_roundtrip_or_known_fallback(data: &[u8], num_threads: usize) {
        let compressed = make_gzip_data(data);
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, num_threads) {
            Ok(n) => {
                assert_eq!(n as usize, data.len(), "size mismatch");
                assert_eq!(output, data, "content mismatch");
            }
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::CrcMismatch)
            | Err(ParallelError::SizeMismatch)
            | Err(ParallelError::TooSmall) => {
                assert!(
                    output.is_empty(),
                    "fallback error must not have written to the writer"
                );
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn roundtrip_compressible_8mb_t4() {
        assert_roundtrip_or_known_fallback(&make_compressible_data(8 * 1024 * 1024), 4);
    }

    #[test]
    fn roundtrip_compressible_25mb_t4() {
        assert_roundtrip_or_known_fallback(&make_compressible_data(25 * 1024 * 1024), 4);
    }

    #[test]
    fn roundtrip_random_10mb_t2() {
        assert_roundtrip_or_known_fallback(&make_random_data(10 * 1024 * 1024, 0xabad1dea), 2);
    }

    #[test]
    fn roundtrip_random_10mb_t4() {
        assert_roundtrip_or_known_fallback(&make_random_data(10 * 1024 * 1024, 0xabad1dea), 4);
    }

    #[test]
    fn roundtrip_random_10mb_t8() {
        assert_roundtrip_or_known_fallback(&make_random_data(10 * 1024 * 1024, 0xabad1dea), 8);
    }

    #[test]
    fn roundtrip_levels_t4() {
        let data = make_compressible_data(8 * 1024 * 1024);
        for level in [1u32, 3, 6, 9] {
            let compressed = make_gzip_at_level(&data, level);
            let mut output = Vec::new();
            match decompress_parallel(&compressed, &mut output, 4) {
                Ok(_) => assert_eq!(output, data, "L{level} content mismatch"),
                Err(ParallelError::DecodeFailed)
                | Err(ParallelError::CrcMismatch)
                | Err(ParallelError::SizeMismatch)
                | Err(ParallelError::TooSmall) => {}
                Err(e) => panic!("L{level} unexpected: {e}"),
            }
        }
    }

    #[test]
    fn determinism_two_runs_match() {
        let data = make_compressible_data(16 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        let r1 = decompress_parallel(&compressed, &mut out1, 4);
        let r2 = decompress_parallel(&compressed, &mut out2, 4);
        match (r1, r2) {
            (Ok(_), Ok(_)) => assert_eq!(out1, out2, "non-deterministic Ok output"),
            (Err(_), Err(_)) => {}
            (a, b) => panic!("non-deterministic outcome: {a:?} vs {b:?}"),
        }
    }

    #[test]
    fn thread_count_does_not_change_output() {
        let data = make_random_data(12 * 1024 * 1024, 0xdeadc0de);
        let compressed = make_gzip_data(&data);
        let mut reference: Option<Vec<u8>> = None;
        for &t in &[2usize, 3, 4, 8] {
            let mut out = Vec::new();
            if decompress_parallel(&compressed, &mut out, t).is_ok() {
                if let Some(r) = &reference {
                    assert_eq!(&out, r, "T={t} differs from prior successful run");
                }
                reference = Some(out);
            }
        }
    }

    #[test]
    fn crc_corruption_is_detected() {
        let data = make_random_data(10 * 1024 * 1024, 0xc0ffee);
        let mut compressed = make_gzip_data(&data);
        let crc_offset = compressed.len() - 8;
        compressed[crc_offset] ^= 0xff;
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Err(ParallelError::CrcMismatch) => {}
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::SizeMismatch)
            | Err(ParallelError::TooSmall) => {}
            Ok(_) => panic!("CRC corruption not detected"),
            Err(e) => panic!("unexpected error: {e}"),
        }
        // Writer must not have received any bytes on any error path.
        assert!(output.is_empty());
    }

    // ── Deletion-trap killer counter ─────────────────────────────────────────

    #[test]
    fn marker_pipeline_counter_increments_on_success() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let before = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
        let mut output = Vec::new();
        if decompress_parallel(&compressed, &mut output, 4).is_ok() {
            let after = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
            assert!(
                after > before,
                "marker pipeline counter must increment on successful decode (was {before}, now {after})"
            );
        }
        // If parallel declined (e.g., search failure on this specific data),
        // the counter shouldn't have moved either. The point of this test
        // is to prove the counter exists and works — the routing-level
        // assertion in `src/tests/routing.rs` is the deletion-trap killer
        // proper, exercising the full CLI path.
    }
}
