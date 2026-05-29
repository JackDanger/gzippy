//! Gzip decompression engine — pure bytes-in / bytes-out.
//!
//! Entry points for the I/O layer are in `io`. This module handles:
//! classify → route → decompress. `classify_gzip`, `decompress_bytes`, and
//! `decompress_gzip_to_writer` are `pub` for use by the library API in
//! `lib.rs`; all other functions are `pub(crate)`.

pub mod bgzf;
pub mod block_walker;
pub mod combined_lut;
pub mod deflate64;
pub mod format;
pub mod index;
pub mod inflate;
pub mod inflate_tables;
pub mod io;
pub mod mmap_writer;
pub mod packed_lut;
pub mod parallel;
pub mod scan_inflate;
pub mod simd_copy;
pub mod simd_huffman;
pub mod two_level_table;
pub mod ultra_fast_inflate;

use std::io::Write;

use crate::decompress::format::{has_bgzf_markers, is_likely_multi_member, read_gzip_isize};
use crate::error::{GzippyError, GzippyResult};

const STREAM_BUFFER_SIZE: usize = 1024 * 1024;

#[cfg(target_os = "macos")]
const CACHE_LINE_SIZE: usize = 128;
#[cfg(not(target_os = "macos"))]
const CACHE_LINE_SIZE: usize = 64;

#[inline]
fn alloc_aligned_buffer(size: usize) -> Vec<u8> {
    let aligned = (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
    vec![0u8; aligned]
}

// =============================================================================
// Routing
// =============================================================================

/// The decompression path selected for a given input.
///
/// This is the canonical routing table. `classify_gzip` returns one of these;
/// `decompress_gzip_libdeflate` dispatches on it. To add a new path: add a variant
/// here, a condition in `classify_gzip`, and a dispatch arm below.
///
/// Current paths (in priority order):
///   GzippyParallel   — gzippy-produced multi-block files ("GZ" FEXTRA subfield)
///   MultiMemberPar   — pigz-style multi-member, Tmax threads
///   MultiMemberSeq   — pigz-style multi-member, T1
///   IsalParallelSM   — x86_64 single-member ≥ 10 MiB w/ T>1 — parallel marker pipeline
///   IsalSingle       — x86_64 single-member via ISA-L (one-shot, T=1 or small input)
///   StreamingSingle  — single-member > 1GB, no ISA-L (avoids huge allocation)
///   LibdeflateSingle — default single-member via libdeflate one-shot (no ISA-L)
///
/// The single-member sub-paths are now fully decided here. The body of
/// `decompress_single_member` is a pure dispatcher — no `if-let-Some-else`
/// fall-through, no silent retry against a different backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    GzippyParallel,
    MultiMemberPar,
    MultiMemberSeq,
    IsalParallelSM,
    IsalSingle,
    StreamingSingle,
    LibdeflateSingle,
}

/// Inputs ≥ this size on a multi-threaded host with ISA-L take the
/// parallel single-member pipeline. Below this, single-shot libdeflate
/// (or ISA-L on x86_64) wins on wall-clock because the parallel
/// pipeline's per-chunk fixed overhead dominates. This is a *routing*
/// decision (visible at the classifier level); it is never used as a
/// silent in-body fallback.
pub(crate) const MIN_PARALLEL_COMPRESSED: usize = 10 * 1024 * 1024;

/// Compression ratio (uncompressed / compressed) below which the speculative
/// parallel single-member pipeline is a NET LOSS and we route to the one-shot
/// path instead.
///
/// Stored-dominated / incompressible data has very sparse deflate-block
/// boundaries, so the block finder's spacing-aligned arithmetic guesses never
/// land on a real boundary: every speculative chunk fails to validate (header
/// / body speculation failures) and decode serializes — measured 3–4× SLOWER
/// than a single ISA-L pass. (2026-05-29 matrix: random100 collapses from T1
/// 7263 MB/s to T8 1963 MB/s; GZIPPY_VERBOSE showed 228 header + 69 body
/// speculation failures.) The ratio cleanly separates incompressible (~1.00)
/// from compressible silesia (~3.1); 1.15 leaves margin.
const PARALLEL_SM_MIN_RATIO_NUM: u64 = 115;
const PARALLEL_SM_MIN_RATIO_DEN: u64 = 100;

/// True when a single-member stream is too incompressible for the speculative
/// parallel pipeline to pay off (see [`PARALLEL_SM_MIN_RATIO_NUM`]).
///
/// Uses the gzip ISIZE trailer (uncompressed size mod 2^32). For the rare
/// single-member stream whose true uncompressed size ≥ 4 GiB *and* is highly
/// compressible, ISIZE wraps and this may mis-fire — a perf-only miss (it
/// picks the correct, just non-parallel, one-shot decoder), never a
/// correctness issue.
fn parallel_sm_unprofitable(data: &[u8]) -> bool {
    match read_gzip_isize(data) {
        Some(isize_bytes) => {
            (isize_bytes as u64) * PARALLEL_SM_MIN_RATIO_DEN
                < (data.len() as u64) * PARALLEL_SM_MIN_RATIO_NUM
        }
        None => false,
    }
}

/// Classify a gzip input into the optimal `DecodePath`.
///
/// Single source of truth for routing. All classification logic lives here;
/// `decompress_gzip_libdeflate` and `decompress_single_member` only
/// dispatch on the result.
pub fn classify_gzip(data: &[u8], num_threads: usize) -> DecodePath {
    if has_bgzf_markers(data) {
        return DecodePath::GzippyParallel;
    }
    if is_likely_multi_member(data) {
        return if num_threads > 1 {
            DecodePath::MultiMemberPar
        } else {
            DecodePath::MultiMemberSeq
        };
    }
    if crate::decompress::parallel::sm_cfg::PARALLEL_SM
        && num_threads > 1
        && data.len() > MIN_PARALLEL_COMPRESSED
        && !parallel_sm_unprofitable(data)
    {
        return DecodePath::IsalParallelSM;
    }
    if crate::backends::isal_decompress::is_available() {
        return DecodePath::IsalSingle;
    }
    if data.len() > 1024 * 1024 * 1024 {
        return DecodePath::StreamingSingle;
    }
    DecodePath::LibdeflateSingle
}

// =============================================================================
// Public entry point
// =============================================================================

/// Decompress gzip data to an arbitrary writer. Used by --test mode and
/// callers that need direct access without going through the file I/O layer.
pub fn decompress_gzip_to_writer<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    decompress_gzip_libdeflate(data, writer, num_threads)
}

// =============================================================================
// Library API
// =============================================================================

/// Decompress gzip data with an explicit thread count.
///
/// Uses gzippy's full routing table (parallel bgzf, parallel multi-member,
/// ISA-L single-member, libdeflate one-shot) based on the input and
/// the requested thread count.
#[allow(dead_code)] // called from lib.rs; unused in the binary
pub fn decompress_bytes<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    decompress_gzip_libdeflate(data, writer, num_threads.max(1))
}

// =============================================================================
// Core engine — all pub(crate) so tests can reach them directly
// =============================================================================

/// Route and decompress a gzip byte slice to `writer`.
pub(crate) fn decompress_gzip_libdeflate<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(0);
    }

    let path = classify_gzip(data, num_threads);

    if crate::utils::debug_enabled() {
        eprintln!(
            "[gzippy] path={:?} threads={} bytes={}",
            path,
            num_threads,
            data.len()
        );
    }

    match path {
        DecodePath::GzippyParallel => {
            let bytes =
                crate::decompress::bgzf::decompress_bgzf_parallel(data, writer, num_threads)?;
            Ok(bytes)
        }
        DecodePath::MultiMemberPar => {
            match crate::decompress::bgzf::decompress_multi_member_parallel(
                data,
                writer,
                num_threads,
            ) {
                Ok(bytes) => Ok(bytes),
                // Parallel scan can fail on random/stored-block data with false gzip
                // header sequences; sequential path handles all multi-member files.
                Err(_) => decompress_multi_member_sequential(data, writer),
            }
        }
        DecodePath::MultiMemberSeq => decompress_multi_member_sequential(data, writer),
        DecodePath::IsalParallelSM
        | DecodePath::IsalSingle
        | DecodePath::StreamingSingle
        | DecodePath::LibdeflateSingle => {
            decompress_single_member_for(path, data, writer, num_threads)
        }
    }
}

/// Decompress gzip to an owned Vec. Used by the I/O layer for parallel paths
/// that benefit from a Vec intermediate (e.g. multi-member Tmax).
pub(crate) fn decompress_gzip_to_vec(data: &[u8], num_threads: usize) -> GzippyResult<Vec<u8>> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(Vec::new());
    }
    if has_bgzf_markers(data) {
        return Ok(crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(
            data,
            num_threads,
        )?);
    }
    if num_threads > 1 && is_likely_multi_member(data) {
        match crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(data, num_threads) {
            Ok(v) => return Ok(v),
            // scan_member_boundaries_fast can fail on random/stored-block data where
            // the compressed bytes contain false 0x1f 0x8b 0x08 sequences; fall back
            // to the sequential path which handles all multi-member files correctly.
            Err(_) => {
                let mut out = Vec::new();
                decompress_multi_member_sequential(data, &mut out)?;
                return Ok(out);
            }
        }
    }
    let mut output = Vec::new();
    decompress_gzip_libdeflate(data, &mut output, num_threads)?;
    Ok(output)
}

/// Route a single-member input. Pure dispatcher — classifies once and
/// hands off to exactly one backend. **No fallback.** Each backend
/// either succeeds or returns `Err(GzippyError::Decompression(_))`.
pub(crate) fn decompress_single_member<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    let path = classify_gzip(data, num_threads);
    debug_assert!(
        matches!(
            path,
            DecodePath::IsalParallelSM
                | DecodePath::IsalSingle
                | DecodePath::StreamingSingle
                | DecodePath::LibdeflateSingle
        ),
        "decompress_single_member called on non-single-member input: {path:?}"
    );
    if crate::utils::debug_enabled() {
        eprintln!(
            "[gzippy] decompress_single_member path={:?} threads={} bytes={}",
            path,
            num_threads,
            data.len()
        );
    }
    decompress_single_member_for(path, data, writer, num_threads)
}

/// Test-only counter incremented every time a single-member call reaches
/// the libdeflate one-shot backend. Snapshot before/after a decode to
/// verify the no-fallback invariant — any increment from a call that
/// *should* have routed parallel is a bug.
#[cfg(test)]
pub(crate) static LIBDEFLATE_SM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Test-only counter incremented every time a single-member call reaches
/// the ISA-L one-shot backend. Together with `LIBDEFLATE_SM_CALLS`
/// these prove that a parallel-eligible input never silently took a
/// sequential backend.
#[cfg(test)]
pub(crate) static ISAL_STREAM_SM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Hard dispatcher. Each arm is terminal — success or `Err`.
fn decompress_single_member_for<W: Write>(
    path: DecodePath,
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    match path {
        DecodePath::IsalParallelSM => {
            // The parallel pipeline runs and verifies CRC + ISIZE — or
            // returns an error. No fallback. This is the production
            // hot path on x86_64 + ISA-L for inputs ≥ MIN_PARALLEL_COMPRESSED
            // with T > 1.
            let n = crate::decompress::parallel::single_member::decompress_parallel(
                data,
                writer,
                num_threads,
            )
            .map_err(|e| GzippyError::decompression(format!("parallel SM: {e}")))?;
            writer.flush()?;
            Ok(n)
        }
        DecodePath::IsalSingle => {
            #[cfg(test)]
            ISAL_STREAM_SM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let bytes = crate::backends::isal_decompress::decompress_gzip_stream(data, writer)
                .ok_or_else(|| {
                    GzippyError::decompression("ISA-L sequential decompress failed".to_string())
                })?;
            writer.flush()?;
            Ok(bytes)
        }
        DecodePath::StreamingSingle => {
            if crate::utils::debug_enabled() {
                eprintln!("[gzippy] streaming zlib-ng decode: {} bytes", data.len());
            }
            decompress_single_member_streaming(data, writer)
        }
        DecodePath::LibdeflateSingle => {
            #[cfg(test)]
            LIBDEFLATE_SM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            decompress_single_member_libdeflate(data, writer)
        }
        // unreachable on well-formed callers — `decompress_gzip_libdeflate`
        // routes multi-member / bgzf paths before calling here.
        other => Err(GzippyError::decompression(format!(
            "decompress_single_member_for called with non-single-member path: {other:?}"
        ))),
    }
}

/// Streaming single-member decompress via flate2/zlib-ng. Fixed 1MB buffer —
/// avoids the page-fault overhead of allocating a full-size output buffer.
pub(crate) fn decompress_single_member_streaming<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(data);
    let mut buf = vec![0u8; STREAM_BUFFER_SIZE];
    let mut total = 0u64;
    loop {
        let n = decoder.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        total += n as u64;
    }
    writer.flush()?;
    Ok(total)
}

/// Single-member decompress via libdeflate FFI (fastest path).
/// Uses the ISIZE trailer hint for initial buffer sizing; grows and retries
/// on InsufficientSpace.
pub(crate) fn decompress_single_member_libdeflate<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    use crate::backends::libdeflate::{DecompressError, DecompressorEx};
    let mut decompressor = DecompressorEx::new();
    let isize_hint = read_gzip_isize(data).unwrap_or(0) as usize;
    let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(64 * 1024)
    };
    let mut output = alloc_aligned_buffer(initial_size);
    loop {
        match decompressor.gzip_decompress_ex(data, &mut output) {
            Ok(result) => {
                writer.write_all(&output[..result.output_size])?;
                writer.flush()?;
                return Ok(result.output_size as u64);
            }
            Err(DecompressError::InsufficientSpace) => {
                let new_size = output.len().saturating_mul(2);
                output.resize(new_size, 0);
            }
            Err(DecompressError::BadData) => {
                return Err(GzippyError::invalid_argument(
                    "invalid gzip data".to_string(),
                ));
            }
        }
    }
}

/// Sequential multi-member decompress via libdeflate. Uses `gzip_decompress_ex`
/// which returns `input_consumed` so we can step through members without
/// re-scanning.
pub(crate) fn decompress_multi_member_sequential<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    use crate::backends::libdeflate::{DecompressError, DecompressorEx};
    let mut decompressor = DecompressorEx::new();
    let mut total_bytes = 0u64;
    let mut offset = 0;
    let mut member_count = 0u32;
    let isize_hint = read_gzip_isize(data).unwrap_or(0) as usize;
    let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(256 * 1024)
    };
    let mut output_buf = alloc_aligned_buffer(initial_size);

    while offset < data.len() {
        if data.len() - offset < 10 {
            break;
        }
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }
        let remaining = &data[offset..];
        let min_size = remaining.len().max(128 * 1024);
        if output_buf.len() < min_size {
            output_buf.resize(min_size, 0);
        }
        let mut success = false;
        loop {
            match decompressor.gzip_decompress_ex(remaining, &mut output_buf) {
                Ok(result) => {
                    member_count += 1;
                    if crate::utils::debug_enabled() {
                        eprintln!(
                            "[gzippy] sequential member {}: in_consumed={} out_size={} offset={}/{}",
                            member_count,
                            result.input_consumed,
                            result.output_size,
                            offset,
                            data.len()
                        );
                    }
                    writer.write_all(&output_buf[..result.output_size])?;
                    total_bytes += result.output_size as u64;
                    offset += result.input_consumed;
                    success = true;
                    break;
                }
                Err(DecompressError::InsufficientSpace) => {
                    let new_size = output_buf.len().saturating_mul(2);
                    output_buf.resize(new_size, 0);
                }
                Err(DecompressError::BadData) => {
                    break;
                }
            }
        }
        if !success {
            break;
        }
    }
    writer.flush()?;
    Ok(total_bytes)
}

/// Decompress zlib data (2-byte header + deflate + 4-byte Adler32).
pub(crate) fn decompress_zlib_turbo<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    if data.len() < 6 {
        return Err(GzippyError::invalid_argument(
            "Zlib data too short".to_string(),
        ));
    }
    let deflate_data = &data[2..data.len() - 4];
    let mut output_buf = vec![0u8; data.len().saturating_mul(4).max(64 * 1024)];
    match crate::decompress::bgzf::inflate_into_pub(deflate_data, &mut output_buf) {
        Ok(size) => {
            writer.write_all(&output_buf[..size])?;
            writer.flush()?;
            Ok(size as u64)
        }
        Err(e) => Err(GzippyError::invalid_argument(format!(
            "zlib decompression failed: {}",
            e
        ))),
    }
}

/// Decompress a raw DEFLATE stream (RFC 1951) — no gzip header or trailer.
///
/// Uses libdeflate with a growing output buffer, falling back to flate2/zlib-ng
/// streaming when the output exceeds 1 GiB.
#[allow(dead_code)] // called from lib.rs; unused in the binary
pub fn decompress_raw_bytes(data: &[u8]) -> GzippyResult<Vec<u8>> {
    let mut decompressor = libdeflater::Decompressor::new();
    const CAP: usize = 1 << 30; // 1 GiB
    let mut estimate = data.len().saturating_mul(4).clamp(4096, CAP);

    loop {
        let mut out = vec![0u8; estimate];
        match decompressor.deflate_decompress(data, &mut out) {
            Ok(n) => {
                out.truncate(n);
                return Ok(out);
            }
            Err(libdeflater::DecompressionError::InsufficientSpace) if estimate < CAP => {
                estimate = (estimate * 2).min(CAP);
            }
            Err(libdeflater::DecompressionError::InsufficientSpace) => break,
            Err(_) => return Err(GzippyError::decompression("invalid raw DEFLATE data")),
        }
    }

    // Output exceeds 1 GiB — stream through flate2/zlib-ng
    use flate2::read::DeflateDecoder;
    use std::io::Read;
    let mut dec = DeflateDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out)
        .map_err(|_| GzippyError::decompression("raw DEFLATE decompression failed"))?;
    Ok(out)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    #[cfg(all(
        target_arch = "x86_64",
        any(feature = "isal-compression", feature = "pure-rust-inflate")
    ))]
    use std::sync::atomic::Ordering;

    /// Build a moderately-compressible single-member gzip fixture whose
    /// compressed size clears `MIN_PARALLEL_COMPRESSED` AND whose ratio clears
    /// the `parallel_sm_unprofitable` gate (~2:1), so it routes to the parallel
    /// SM path. (Incompressible fixtures now correctly route to one-shot, so
    /// parallel-path tests must use compressible data.) Returns (original,
    /// compressed). 32 MiB of 4-bit-entropy bytes → ~16 MiB compressed.
    fn compressible_parallel_fixture() -> (Vec<u8>, Vec<u8>) {
        let mut original = vec![0u8; 32 * 1024 * 1024];
        let mut state = 0xc0ffeeu64;
        for b in &mut original {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((state >> 32) as u8) & 0x0F;
        }
        let mut compressed = Vec::new();
        {
            let mut enc = GzEncoder::new(&mut compressed, Compression::default());
            enc.write_all(&original).unwrap();
            enc.finish().unwrap();
        }
        assert!(
            compressed.len() > MIN_PARALLEL_COMPRESSED,
            "fixture must clear parallel gate (got {} bytes)",
            compressed.len()
        );
        assert!(
            !parallel_sm_unprofitable(&compressed),
            "fixture must be compressible enough to route parallel"
        );
        (original, compressed)
    }

    /// The classifier — not an in-body fallback — is the only place that
    /// decides whether parallel SM runs. An input that satisfies the
    /// gate must classify to `IsalParallelSM` on x86_64+ISA-L hosts and
    /// to `IsalSingle`/`LibdeflateSingle` elsewhere — never silently
    /// switch backends inside `decompress_single_member`.
    #[test]
    fn test_classify_routes_at_classifier_not_in_body() {
        let (_raw, payload) = compressible_parallel_fixture();
        let path = classify_gzip(&payload, 4);
        #[cfg(all(
            target_arch = "x86_64",
            any(feature = "isal-compression", feature = "pure-rust-inflate")
        ))]
        assert_eq!(
            path,
            DecodePath::IsalParallelSM,
            "12 MiB compressed input on x86_64+ISA-L must classify parallel"
        );
        #[cfg(not(all(
            target_arch = "x86_64",
            any(feature = "isal-compression", feature = "pure-rust-inflate")
        )))]
        assert_ne!(
            path,
            DecodePath::IsalParallelSM,
            "no-ISA-L host must not classify parallel"
        );
    }

    /// Incompressible single-member input above the size gate must NOT take
    /// the speculative parallel pipeline — on stored-dominated data the block
    /// finder's spacing guesses never hit a real boundary and the pipeline is
    /// 3-4× slower than a single ISA-L pass (2026-05-29 matrix). It must route
    /// to the one-shot path instead via `parallel_sm_unprofitable`.
    #[test]
    fn test_incompressible_single_member_bails_to_one_shot() {
        // 16 MiB of high-entropy bytes -> compressed ~= raw, ratio ~1.0.
        let mut original = vec![0u8; 16 * 1024 * 1024];
        let mut state = 0xdeadbeefu64;
        for b in &mut original {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 32) as u8;
        }
        let mut compressed = Vec::new();
        {
            let mut enc = GzEncoder::new(&mut compressed, Compression::default());
            enc.write_all(&original).unwrap();
            enc.finish().unwrap();
        }
        assert!(
            compressed.len() > MIN_PARALLEL_COMPRESSED,
            "fixture clears the size gate (got {} bytes)",
            compressed.len()
        );
        assert!(
            parallel_sm_unprofitable(&compressed),
            "incompressible input must be flagged unprofitable for parallel SM"
        );
        // Even at T=4 it must NOT classify parallel.
        assert_ne!(
            classify_gzip(&compressed, 4),
            DecodePath::IsalParallelSM,
            "incompressible single-member must bail off the speculative pipeline"
        );
        // And it must still decode byte-perfectly through whatever path it took.
        let mut out = Vec::new();
        decompress_single_member(&compressed, &mut out, 4).expect("must decode");
        assert_eq!(out, original, "byte-perfect output required after bail");
    }

    /// On a parallel-eligible input the libdeflate one-shot backend must
    /// never be called from the single-member dispatcher. This is the
    /// no-fallback invariant the user asked for: under no circumstances
    /// can a silent libdeflate retry mask a parallel-pipeline failure.
    #[cfg(all(
        target_arch = "x86_64",
        any(feature = "isal-compression", feature = "pure-rust-inflate")
    ))]
    #[test]
    fn test_no_libdeflate_fallback_ever_fires_from_sm_path() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // Compressible fixture above MIN_PARALLEL_COMPRESSED so the
        // classifier returns IsalParallelSM (incompressible would now bail
        // to one-shot via parallel_sm_unprofitable).
        let (original, compressed) = compressible_parallel_fixture();
        let before_lib = LIBDEFLATE_SM_CALLS.load(Ordering::Relaxed);
        let mut out = Vec::new();
        decompress_single_member(&compressed, &mut out, 4).expect("must succeed");
        assert_eq!(out, original, "byte-perfect output required");
        let after_lib = LIBDEFLATE_SM_CALLS.load(Ordering::Relaxed);
        assert_eq!(
            after_lib, before_lib,
            "libdeflate SM backend must not be called from the parallel-eligible SM path \
             (would indicate a silent fallback)"
        );
        // ISA-L sequential counter is incremented by other unit tests that
        // call `decompress_single_member(..., T=1)` in parallel; only the
        // libdeflate counter is a reliable no-fallback signal here.
    }

    /// A corrupted CRC must surface as `GzippyError::Decompression`, not
    /// produce silent `Ok` via a libdeflate retry.
    #[cfg(all(
        target_arch = "x86_64",
        any(feature = "isal-compression", feature = "pure-rust-inflate")
    ))]
    #[test]
    fn test_parallel_sm_propagates_errors_not_fallbacks() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let (_original, mut compressed) = compressible_parallel_fixture();
        // Flip a byte deep in the deflate stream; CRC check at the end
        // of the parallel SM path catches the corruption.
        let mid = compressed.len() / 2;
        compressed[mid] ^= 0xFF;
        let mut out = Vec::new();
        let res = decompress_single_member(&compressed, &mut out, 4);
        assert!(
            matches!(res, Err(GzippyError::Decompression(_))),
            "corrupt input must propagate Err(Decompression(_)), got {:?}",
            res.as_ref().err()
        );
    }

    /// An input *below* the parallel gate routes deterministically. This
    /// is a routing decision visible at the classifier — not a silent
    /// in-body fallback. Document it explicitly.
    #[test]
    fn test_parallel_sm_routes_below_10mib() {
        let small: Vec<u8> = vec![0u8; 1024 * 1024]; // 1 MiB original
        let mut compressed = Vec::new();
        {
            let mut enc = GzEncoder::new(&mut compressed, Compression::default());
            enc.write_all(&small).unwrap();
            enc.finish().unwrap();
        }
        // Compressed size of all-zeros 1 MiB is tiny — well below 10 MiB.
        assert!(compressed.len() < MIN_PARALLEL_COMPRESSED);
        let path = classify_gzip(&compressed, 4);
        assert_ne!(
            path,
            DecodePath::IsalParallelSM,
            "below-gate input must not classify parallel"
        );
        // Whatever sub-path the classifier picks, it must decode.
        let mut out = Vec::new();
        decompress_single_member(&compressed, &mut out, 4).expect("below-gate must still decode");
        assert_eq!(out, small);
    }

    /// Type-level fence: WindowMap's stored type is
    /// `Arc<CompressedVector>`, decompressed lazily on `get`. The test
    /// is structural — if someone reverts to `Arc<[u8;32768]>` storage
    /// this will fail to compile.
    #[test]
    fn test_window_map_uses_compressed_vector() {
        use crate::decompress::parallel::compressed_vector::CompressionType;
        use crate::decompress::parallel::window_map::WindowMap;
        let m = WindowMap::new();
        assert_eq!(
            m.compression(),
            CompressionType::Zlib,
            "default compression must be Zlib to match rapidgzip's WindowMap"
        );
    }

    #[test]
    fn test_decompress_multi_member_file() {
        let part1: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let part2: Vec<u8> = (0..100_000).map(|i| ((i + 50) % 256) as u8).collect();

        let mut enc1 = GzEncoder::new(Vec::new(), Compression::default());
        enc1.write_all(&part1).unwrap();
        let compressed1 = enc1.finish().unwrap();

        let mut enc2 = GzEncoder::new(Vec::new(), Compression::default());
        enc2.write_all(&part2).unwrap();
        let compressed2 = enc2.finish().unwrap();

        let mut multi = compressed1;
        multi.extend_from_slice(&compressed2);

        let mut output = Vec::new();
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        decompress_gzip_libdeflate(&multi, &mut output, num_threads).unwrap();

        let mut expected = part1;
        expected.extend_from_slice(&part2);
        assert_eq!(output, expected);
    }
}
