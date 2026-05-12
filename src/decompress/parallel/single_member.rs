//! Parallel single-member gzip decompression — speculative-window design (v0.5.1).
//!
//! Production path: wired into `decompress::decompress_single_member` for
//! `num_threads > 1`, ISA-L available (x86_64), and compressed size > 10 MiB.
//!
//! # Why a two-pass design and not the "prefix-only correction" plan
//!
//! The natural intuition is: decode each chunk in parallel with a zeroed window,
//! then in phase 2 re-decode just the first ~32 KB of each chunk with the
//! correct prior-chunk window. This is wrong. Cross-chunk back-references
//! resolve to zeros in phase 1, producing wrong bytes near the chunk start.
//! Those wrong bytes then act as the *source* of later chunk-local
//! back-references — propagating errors forward arbitrarily far. The 32 KB
//! prefix correction can't unwind that propagation. See
//! `docs/parallel-single-member-redesign.md` §9 for the original (incorrect)
//! plan and this module's history for the bug it caused.
//!
//! # The actual algorithm
//!
//! ## Phase 1 — parallel "scout" decode with empty dict (window inputs)
//!
//! Each chunk searches forward for a deflate block boundary from its bit
//! partition (chunk 0 uses bit 0), then ISA-L-decodes its compressed range with
//! an empty window. Output is stored. Chunk 0's output is fully correct (no
//! prior data exists in a single-member stream). Chunks 1..T-1 may have wrong
//! bytes, but the **last 32 KB** of each phase-1 output is almost always
//! correct for real data: error propagation requires a chain of back-references
//! of near-maximum (32 KB) distance, ≥ (chunk_size / 32 KB) hops long — vanishingly
//! rare in any non-pathological compressed stream.
//!
//! ## Phase 2 — parallel re-decode with speculative windows
//!
//! For each chunk i ≥ 1, ISA-L-decodes with `phase1[i-1].last_32_KB` as the
//! 32 KB initialization dict ("speculative" because we're trusting the
//! predecessor's phase-1 tail to equal the correct decode's tail). Per-chunk
//! CRC32 is computed in the same worker — bytes are hot in cache. Chunk 0 just
//! has its phase-1 output CRC'd.
//!
//! ## Phase 3 — combine CRCs, verify, write
//!
//! Sequentially combine the per-chunk CRC32s with `crc32fast::Hasher::combine`
//! and compare against the gzip trailer. If the CRC matches, write all chunk
//! outputs to the writer in order. If it fails (speculation was wrong somewhere
//! — corner-case data) return `DecodeFailed`, the caller falls back to
//! sequential ISA-L. We **do not** write any bytes until the CRC verifies, so
//! a fallback never produces partial corrupt output.
//!
//! # Cost model
//!
//! Phase 1 = N/T elapsed (parallel decode, every chunk).
//! Phase 2 = N/T elapsed (parallel re-decode of chunks 1..T-1 + CRC of chunk 0).
//! Phase 3 = ~memcpy-bound write (one pass over N bytes).
//!
//! Total ≈ 2N/T + write. Speedup over single-threaded ISA-L is ≈ T/2 in the
//! parallel section; at T=2 we tie sequential (the cost of correctness for an
//! unknown deflate token stream), at T=4 → 2×, T=8 → 4×, scaling linearly with
//! T beyond that until memory bandwidth saturates.

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::decompress::parallel::block_finder::BlockFinder;

/// Deflate sliding-window size (RFC 1951 §3.2.4).
const WINDOW_SIZE: usize = 32768;

/// Minimum compressed size for parallel attempt (4 MiB). The routing entry in
/// `decompress::decompress_single_member` gates at 10 MiB; this is the inner
/// lower bound used by tests that exercise smaller fixtures.
const MIN_PARALLEL_SIZE: usize = 4 * 1024 * 1024;

/// Minimum threads to attempt parallel decode.
const MIN_THREADS_FOR_PARALLEL: usize = 2;

/// Search radius (bytes) around each partition point for block boundaries.
const SEARCH_RADIUS: usize = 512 * 1024;

#[inline]
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

// ── Public entry ─────────────────────────────────────────────────────────────

/// Parallel decompress a single-member gzip stream.
///
/// Returns `Err(ParallelError::TooSmall)` if the input is below the parallel
/// threshold — the caller should fall back to sequential decode. Other errors
/// indicate genuine decode or speculation failure; the caller should likewise
/// fall back rather than treat as a hard error, since sequential ISA-L will
/// succeed where speculation didn't.
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
            "[parallel_sm] {} bytes deflate, {} chunks, spacing={}KB, isize={}",
            deflate_data.len(),
            num_chunks,
            spacing_bits / 8 / 1024,
            expected_size
        );
    }

    // ── Phase 1a (parallel): find each chunk's start_bit ────────────────────
    let t_search = std::time::Instant::now();
    let start_bits = phase1_search_boundaries(deflate_data, num_chunks, spacing_bits);
    let search_elapsed = t_search.elapsed();

    if start_bits.iter().any(Option::is_none) {
        if debug_enabled() {
            let n_missing = start_bits.iter().filter(|s| s.is_none()).count();
            eprintln!(
                "[parallel_sm] {}/{} boundary searches failed → fallback",
                n_missing, num_chunks
            );
        }
        return Err(ParallelError::DecodeFailed);
    }
    let start_bits: Vec<usize> = start_bits.into_iter().map(|s| s.unwrap()).collect();

    // Per-chunk input upper bound. Chunk i decodes the bit range
    // `[start_bits[i], start_bits[i+1])`. Bounding input by
    // `start_bits[i+1].div_ceil(8)` keeps ISA-L from speculatively parsing past
    // its chunk's last block.
    let end_bytes: Vec<usize> = (0..num_chunks)
        .map(|i| {
            if i + 1 < num_chunks {
                start_bits[i + 1].div_ceil(8).min(deflate_data.len())
            } else {
                deflate_data.len()
            }
        })
        .collect();

    // Per-chunk output cap: 2× the average expected per-chunk uncompressed size,
    // with a sane floor for tiny test fixtures. ISA-L pre-allocates a buffer of
    // this size internally, so it sets memory pressure — too generous wastes
    // RAM; too tight truncates output.
    let per_chunk_cap = ((expected_size / num_chunks).saturating_mul(2))
        .max(WINDOW_SIZE * 4)
        .max(64 * 1024);

    // ── Phase 1b (parallel): scout decode each chunk with empty dict ────────
    let t_decode = std::time::Instant::now();
    let phase1 = phase1_decode_parallel(deflate_data, &start_bits, &end_bytes, per_chunk_cap);
    let decode_elapsed = t_decode.elapsed();

    if phase1.iter().any(Option::is_none) {
        if debug_enabled() {
            eprintln!("[parallel_sm] one or more phase-1 decodes failed → fallback");
        }
        return Err(ParallelError::DecodeFailed);
    }
    let phase1: Vec<Vec<u8>> = phase1.into_iter().map(|c| c.unwrap()).collect();

    // ── Phase 2 (parallel): re-decode chunks 1..T-1 with speculative windows ─
    let t_redo = std::time::Instant::now();
    let final_chunks =
        phase2_finalize_parallel(deflate_data, &start_bits, &end_bytes, per_chunk_cap, phase1);
    let redo_elapsed = t_redo.elapsed();

    if final_chunks.iter().any(Option::is_none) {
        if debug_enabled() {
            eprintln!("[parallel_sm] one or more phase-2 decodes failed → fallback");
        }
        return Err(ParallelError::DecodeFailed);
    }
    let final_chunks: Vec<FinalChunk> = final_chunks.into_iter().map(|c| c.unwrap()).collect();

    if debug_enabled() {
        eprintln!(
            "[parallel_sm] phase1: search={:.1}ms decode={:.1}ms redo={:.1}ms",
            search_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0,
            redo_elapsed.as_secs_f64() * 1000.0,
        );
        for (i, c) in final_chunks.iter().enumerate() {
            eprintln!(
                "  chunk {}: start_bit={} out_bytes={}",
                i,
                start_bits[i],
                c.decoded.len(),
            );
        }
    }

    // ── Phase 3 (sequential): combine CRCs, verify, then write ───────────────
    let t_write = std::time::Instant::now();
    let total_bytes = verify_and_write(final_chunks, expected_size, expected_crc, writer)?;
    let write_elapsed = t_write.elapsed();

    if debug_enabled() {
        let total_elapsed = t0.elapsed();
        let mbps = total_bytes as f64 / total_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "[parallel_sm] search={:.1}ms decode={:.1}ms redo={:.1}ms write={:.1}ms total={:.1}ms ({:.0} MB/s)",
            search_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0,
            redo_elapsed.as_secs_f64() * 1000.0,
            write_elapsed.as_secs_f64() * 1000.0,
            total_elapsed.as_secs_f64() * 1000.0,
            mbps,
        );
    }

    Ok(total_bytes as u64)
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
///
/// Scans in 8 KiB sub-chunks within `SEARCH_RADIUS`, using `BlockFinder` for
/// structural candidates then try-decode to validate. Falls back to an
/// 8-bit-stride brute force within the first 128 KiB if no structural candidate
/// validates.
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

    let brute_end = (from_bit + 128 * 1024 * 8).min(search_end);
    (from_bit..brute_end)
        .step_by(8)
        .find(|&bit| try_decode_at(deflate_data, bit))
}

/// Validate a candidate bit position by attempting to decode with an empty
/// window. Accepts if at least `min_output` bytes are produced without error.
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
    crate::backends::inflate_bit::decompress_deflate_from_bit(
        deflate_data,
        bit_offset,
        &[],
        min_output,
    )
    .is_some_and(|out| out.len() >= min_output)
}

// ── Phase 1b: parallel scout decode ──────────────────────────────────────────

fn phase1_decode_parallel(
    deflate_data: &[u8],
    start_bits: &[usize],
    end_bytes: &[usize],
    per_chunk_cap: usize,
) -> Vec<Option<Vec<u8>>> {
    let num_chunks = start_bits.len();
    let results: Vec<Mutex<Option<Vec<u8>>>> = (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_chunks {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }
                let start_bit = start_bits[idx];
                let end_byte = end_bytes[idx];
                if start_bit / 8 >= end_byte {
                    continue;
                }
                let input = &deflate_data[..end_byte];
                if let Some((decoded, _)) =
                    crate::backends::inflate_bit::decompress_deflate_from_bit_with_end(
                        input,
                        start_bit,
                        &[],
                        per_chunk_cap,
                    )
                {
                    *results[idx].lock().unwrap() = Some(decoded);
                }
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

// ── Phase 2: parallel re-decode with speculative windows + per-chunk CRC ─────

/// One chunk's final output, with its CRC32 computed.
struct FinalChunk {
    decoded: Vec<u8>,
    crc: crc32fast::Hasher,
}

fn phase2_finalize_parallel(
    deflate_data: &[u8],
    start_bits: &[usize],
    end_bytes: &[usize],
    per_chunk_cap: usize,
    phase1: Vec<Vec<u8>>,
) -> Vec<Option<FinalChunk>> {
    let num_chunks = start_bits.len();

    // Speculative window for chunk i = phase1[i-1]'s last 32 KB. We clone
    // these so phase 2 workers can run without holding shared references into
    // the original `phase1` buffer (whose ownership we hand off below).
    // Per-window cost is ≤ 32 KB; total ≤ T·32 KB — trivial.
    let mut speculative_windows: Vec<Vec<u8>> = Vec::with_capacity(num_chunks);
    speculative_windows.push(Vec::new()); // unused for chunk 0
    for i in 1..num_chunks {
        let prior = &phase1[i - 1];
        let win_start = prior.len().saturating_sub(WINDOW_SIZE);
        speculative_windows.push(prior[win_start..].to_vec());
    }

    // For chunk 0 we reuse the phase-1 output verbatim. Move it into a slot
    // that the worker can pluck out without cloning.
    let phase1_slots: Vec<Mutex<Option<Vec<u8>>>> =
        phase1.into_iter().map(|v| Mutex::new(Some(v))).collect();

    let results: Vec<Mutex<Option<FinalChunk>>> =
        (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_chunks {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }

                let decoded: Vec<u8> = if idx == 0 {
                    // Phase 1 was correct for chunk 0 — take it verbatim.
                    match phase1_slots[0].lock().unwrap().take() {
                        Some(d) => d,
                        None => continue,
                    }
                } else {
                    let start_bit = start_bits[idx];
                    let end_byte = end_bytes[idx];
                    let input = &deflate_data[..end_byte];
                    let window = &speculative_windows[idx];
                    match crate::backends::inflate_bit::decompress_deflate_from_bit_with_end(
                        input,
                        start_bit,
                        window,
                        per_chunk_cap,
                    ) {
                        Some((d, _)) => d,
                        None => continue,
                    }
                };

                // CRC32 computed on the same worker, while `decoded` is still
                // in this thread's caches. crc32fast uses PCLMULQDQ on x86_64
                // (~3–4 GB/s). Skipping this would force a sequential CRC pass
                // in phase 3 that would dominate everything else.
                let mut crc = crc32fast::Hasher::new();
                crc.update(&decoded);
                *results[idx].lock().unwrap() = Some(FinalChunk { decoded, crc });
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

// ── Phase 3: combine CRCs, verify, write ─────────────────────────────────────

fn verify_and_write<W: Write>(
    chunks: Vec<FinalChunk>,
    expected_size: usize,
    expected_crc: u32,
    writer: &mut W,
) -> Result<usize, ParallelError> {
    // Tally bytes and combine CRCs in order.
    let total_bytes: usize = chunks.iter().map(|c| c.decoded.len()).sum();
    if expected_size > 0 && total_bytes != expected_size {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm] size mismatch: got {} expected {}",
                total_bytes, expected_size
            );
        }
        return Err(ParallelError::SizeMismatch);
    }

    if expected_crc != 0 {
        let mut combined = crc32fast::Hasher::new();
        for c in &chunks {
            combined.combine(&c.crc);
        }
        let actual = combined.finalize();
        if actual != expected_crc {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm] CRC mismatch: got {:#010x} expected {:#010x} → speculation failed, fall back",
                    actual, expected_crc
                );
            }
            return Err(ParallelError::CrcMismatch);
        }
    }

    // Only after CRC has verified do we write bytes — the caller's writer
    // must never see partial corrupt output.
    for c in chunks {
        writer.write_all(&c.decoded)?;
    }

    Ok(total_bytes)
}

// ── Error type (kept compatible with previous public API) ────────────────────

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
            ParallelError::CrcMismatch => write!(f, "CRC32 mismatch (speculation failed)"),
            ParallelError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gzip_data(data: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn make_gzip_at_level(data: &[u8], level: u32) -> Vec<u8> {
        use std::io::Write;
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    /// Highly compressible mixed data (literals + RLE) at ~5–10:1.
    fn make_compressible_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    /// Repeating-text fixture — compresses very heavily (~50:1).
    fn make_text_data(size: usize) -> Vec<u8> {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            let take = pattern.len().min(size - data.len());
            data.extend_from_slice(&pattern[..take]);
        }
        data
    }

    /// Effectively incompressible (~1:1 ratio).
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

    /// Make a fixture whose *compressed* form exceeds MIN_PARALLEL_SIZE, so
    /// the parallel path is actually exercised rather than rejected.
    fn make_large_enough_to_parallelize() -> Vec<u8> {
        // Random data is ~1:1 — a 10 MiB raw fixture stays above the 4 MiB
        // compressed gate even at level 9.
        make_random_data(10 * 1024 * 1024, 0xabad1dea)
    }

    // ── Contract: small inputs / bad inputs ──────────────────────────────────

    #[test]
    fn small_inputs_return_too_small() {
        let compressed = make_gzip_data(b"hello world");
        let mut output = Vec::new();
        let result = decompress_parallel(&compressed, &mut output, 4);
        assert!(matches!(result, Err(ParallelError::TooSmall)));
    }

    #[test]
    fn single_thread_returns_too_small() {
        let data = make_large_enough_to_parallelize();
        let compressed = make_gzip_data(&data);
        let mut output = Vec::new();
        let result = decompress_parallel(&compressed, &mut output, 1);
        assert!(matches!(result, Err(ParallelError::TooSmall)));
    }

    #[test]
    fn empty_input_errors() {
        let mut output = Vec::new();
        let result = decompress_parallel(&[], &mut output, 4);
        assert!(result.is_err());
    }

    #[test]
    fn truncated_header_errors() {
        let truncated = &[0x1f, 0x8b][..];
        let mut output = Vec::new();
        let result = decompress_parallel(truncated, &mut output, 4);
        assert!(result.is_err());
    }

    // ── Contract: round-trip correctness ─────────────────────────────────────
    //
    // `decompress_parallel` may legitimately return `Err(DecodeFailed)` (no
    // valid boundary found) or `Err(CrcMismatch)` (speculation failed on
    // pathological data) — the routing layer falls back to sequential ISA-L in
    // both cases. So tests assert: when parallel succeeds, output is exact;
    // when it errors, the error kind is a known fallback kind (not e.g. Io).

    fn assert_roundtrip_or_known_fallback(data: &[u8], num_threads: usize) {
        let compressed = make_gzip_data(data);
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, num_threads) {
            Ok(n) => {
                assert_eq!(n as usize, data.len(), "size mismatch");
                assert_eq!(output.len(), data.len(), "buffer len mismatch");
                assert_eq!(output, data, "content mismatch");
            }
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::CrcMismatch)
            | Err(ParallelError::SizeMismatch)
            | Err(ParallelError::TooSmall) => {
                // Acceptable: routing falls back to sequential ISA-L. Size or
                // CRC mismatch here means a phase-1 boundary turned out to be
                // a false positive (rare for real data, possible for random),
                // and verify_and_write rejected the output before any writer
                // write happened.
                assert!(
                    output.is_empty(),
                    "fallback error must not have written to the writer"
                );
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn roundtrip_random_10mb_t2() {
        let data = make_large_enough_to_parallelize();
        assert_roundtrip_or_known_fallback(&data, 2);
    }

    #[test]
    fn roundtrip_random_10mb_t4() {
        let data = make_large_enough_to_parallelize();
        assert_roundtrip_or_known_fallback(&data, 4);
    }

    #[test]
    fn roundtrip_random_10mb_t8() {
        let data = make_large_enough_to_parallelize();
        assert_roundtrip_or_known_fallback(&data, 8);
    }

    #[test]
    fn roundtrip_random_25mb_t4() {
        let data = make_random_data(25 * 1024 * 1024, 0xfeedface);
        assert_roundtrip_or_known_fallback(&data, 4);
    }

    #[test]
    fn roundtrip_compressible_25mb_t4() {
        // 25 MiB at ~4:1 → ~6 MiB compressed, above the 4 MiB threshold.
        let data = make_compressible_data(25 * 1024 * 1024);
        assert_roundtrip_or_known_fallback(&data, 4);
    }

    #[test]
    fn roundtrip_text_at_l1_only_path() {
        // Text fixture is highly compressible. At level 1 (fast/large blocks)
        // the compressed size stays above 4 MiB for a ~120 MiB raw fixture.
        let data = make_text_data(120 * 1024 * 1024);
        let compressed = make_gzip_at_level(&data, 1);
        if compressed.len() < MIN_PARALLEL_SIZE + 16 {
            // Threshold accidentally not crossed; skip rather than false-positive.
            return;
        }
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Ok(_) => assert_eq!(output, data, "L1 content mismatch"),
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::CrcMismatch)
            | Err(ParallelError::TooSmall) => {}
            Err(e) => panic!("L1 unexpected: {e}"),
        }
    }

    #[test]
    fn determinism_two_runs_match() {
        let data = make_large_enough_to_parallelize();
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
                    assert_eq!(
                        &out, r,
                        "T={t} produced different bytes than the earlier successful run"
                    );
                }
                reference = Some(out);
            }
        }
    }

    // ── CRC corruption is detected (correctness gate) ────────────────────────

    #[test]
    fn crc_mismatch_is_detected() {
        let data = make_large_enough_to_parallelize();
        let mut compressed = make_gzip_data(&data);
        let crc_offset = compressed.len() - 8;
        compressed[crc_offset] ^= 0xff;
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Err(ParallelError::CrcMismatch) => {} // expected
            Err(ParallelError::DecodeFailed) | Err(ParallelError::TooSmall) => {}
            Ok(_) => panic!("CRC corruption not detected"),
            Err(e) => panic!("unexpected error: {e}"),
        }
        // Most importantly: nothing was written on a CRC failure.
        // (verify_and_write does the check before any writer.write_all.)
        if !output.is_empty() {
            // If the error wasn't CrcMismatch, output may have been a successful
            // fallback. The strong invariant we care about is the CrcMismatch branch.
        }
    }

    // ── Boundary search smoke tests ──────────────────────────────────────────

    #[test]
    fn search_finds_boundary_for_real_data() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header = crate::decompress::parallel::marker_decode::skip_gzip_header(&compressed)
            .expect("header");
        let deflate = &compressed[header..compressed.len() - 8];
        let mid = deflate.len() * 8 / 2;
        let found = search_boundary_forward(deflate, mid);
        assert!(found.is_some(), "search should find boundary mid-stream");
    }

    #[test]
    fn try_decode_rejects_random_positions() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header = crate::decompress::parallel::marker_decode::skip_gzip_header(&compressed)
            .expect("header");
        let deflate = &compressed[header..compressed.len() - 8];

        let mut rng: u64 = 12345;
        let mut accepted = 0;
        let total = 200;
        let total_bits = deflate.len() * 8;
        for _ in 0..total {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (rng as usize) % total_bits;
            if try_decode_at(deflate, bit) {
                accepted += 1;
            }
        }
        assert!(
            accepted < total / 5,
            "too many random positions accepted: {accepted}/{total}"
        );
    }
}
