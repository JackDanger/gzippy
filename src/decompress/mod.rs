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
///   ParallelSM   — single-member ≥ 10 MiB w/ T≥4 — parallel marker pipeline
///                      (x86_64: ISA-L or pure-Rust; aarch64: pure-Rust inner decoder).
///                      Gated by `parallel::sm_cfg::PARALLEL_SM`; name is historical.
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
    // Constructed only under `parallel_sm` (the pure-rust-inflate build);
    // dead in the legacy default build, alive in production.
    #[allow(dead_code)]
    ParallelSM,
    /// Stored-block-dominated (incompressible) single-member, decoded in
    /// parallel WITHOUT speculation by splitting on explicit stored-block
    /// `LEN` fields. See [`crate::decompress::parallel::stored_split`]. The
    /// `parallel_sm_unprofitable` ratio gate routes incompressible input here
    /// (instead of single-thread libdeflate) when its first block is stored —
    /// stored streams are trivially, bandwidth-bound parallelizable.
    #[allow(dead_code)] // constructed only under parallel_sm (production)
    StoredParallel,
    /// Single-threaded (`num_threads <= 1`) single-member decode on the
    /// gzippy-isal build (`isal_clean_tail`): the WHOLE stream through ONE
    /// ISA-L `isal_inflate` call (`isal_decompress::decompress_gzip_stream`),
    /// no chunking. The 16-chunk ParallelSM pipeline buys ZERO parallelism at
    /// one thread — each chunk waits the prior's 32 KiB tail window before it
    /// can ISA-L-decode, so it runs fully serial WITH handoff/ring/window-map
    /// overhead (DIS-15: that lifecycle costs ~247 ms / ~24% of the T1 wall;
    /// single-shot = 1.197x rapidgzip vs ParallelSM 0.905x, byte-exact —
    /// `decompress_gzip_stream` verifies the gzip CRC32+ISIZE via ISA-L's
    /// `IGZIP_GZIP` crc_flag, a non-zero `isal_inflate` ret => terminal Err,
    /// no fallback). Constructed ONLY under `isal_clean_tail` for true
    /// single-member streams (multi-member/BGZF are classified earlier and
    /// never reach here); on the gzippy-native build T1 stays ParallelSM.
    #[allow(dead_code)] // constructed only under isal_clean_tail (production)
    IsalSingleShot,
    // Only constructed in the legacy `not(parallel_sm)` build; under
    // `parallel_sm` (production) single-member never routes to C-FFI.
    #[allow(dead_code)]
    StreamingSingle,
    #[allow(dead_code)]
    LibdeflateSingle,
}

/// Inputs ≥ this size on a multi-threaded host with ISA-L take the
/// parallel single-member pipeline. Below this, single-shot libdeflate
/// (or ISA-L on x86_64) wins on wall-clock because the parallel
/// pipeline's per-chunk fixed overhead dominates. This is a *routing*
/// decision (visible at the classifier level); it is never used as a
/// silent in-body fallback.
#[allow(dead_code)] // referenced by tests + the legacy not(parallel_sm) gate
pub(crate) const MIN_PARALLEL_COMPRESSED: usize = 10 * 1024 * 1024;

// (Removed 2026-06-04) `MIN_PARALLEL_SM_THREADS` / `parallel_sm_min_threads`:
// the parallel-SM engine is now the SOLE single-member decode path at every
// thread count and size (task #8), so there is no thread floor below which a
// C-FFI one-shot is chosen — there is no C-FFI one-shot in the decode graph.

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
#[allow(dead_code)] // used by classify's parallel_sm branch + tests
const PARALLEL_SM_MIN_RATIO_NUM: u64 = 115;
#[allow(dead_code)] // used by classify's parallel_sm branch + tests
const PARALLEL_SM_MIN_RATIO_DEN: u64 = 100;

/// True when a single-member stream is too incompressible for the speculative
/// parallel pipeline to pay off (see [`PARALLEL_SM_MIN_RATIO_NUM`]).
///
/// Uses the gzip ISIZE trailer (uncompressed size mod 2^32). For the rare
/// single-member stream whose true uncompressed size ≥ 4 GiB *and* is highly
/// compressible, ISIZE wraps and this may mis-fire — a perf-only miss (it
/// picks the correct, just non-parallel, one-shot decoder), never a
/// correctness issue.
#[allow(dead_code)] // used by classify's parallel_sm branch + tests
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
    // SINGLE-MEMBER. When the pure-Rust engine is compiled (`parallel_sm`,
    // i.e. the `pure-rust-inflate` build), single-member always routes to the
    // ParallelSM pipeline — never a C-FFI one-shot dispatch (libdeflate/zlib-ng
    // one-shot). On the gzippy-native build the ParallelSM pipeline is pure-Rust
    // end-to-end (NO C-FFI in the decode graph, /goal part 1, task #8). On the
    // gzippy-isal build (x86_64 + `isal-compression`, sets `isal_clean_tail`)
    // the SAME ParallelSM pipeline decodes the clean tail through REAL ISA-L FFI
    // (gzip_chunk.rs `finish_decode_chunk_impl` -> `decompress_deflate_from_bit_into`,
    // faithful rapidgzip WITH_ISAL) — so on that build C-FFI IS on the decode
    // graph by design. The one-shot C-FFI backends remain compiled ONLY in the
    // `not(parallel_sm)` legacy build and behind the dev/oracle feature as fuzz
    // oracles; production single-member never routes to a one-shot.
    #[cfg(parallel_sm)]
    {
        // T1 ROUTE (gzippy-isal only): single-threaded single-member decode
        // goes to ONE ISA-L call, not the 16-chunk ParallelSM pipeline. At one
        // thread the pipeline's chunking buys NO parallelism (each chunk waits
        // the prior's 32 KiB tail window before it can ISA-L-decode → fully
        // serial) yet still pays ring/window-map/CRC/handoff — ~247 ms / ~24%
        // of the T1 wall (DIS-15). Single-shot ISA-L (same igzip kernel) =
        // 1.197x rapidgzip vs the pipeline's 0.905x, byte-exact. This is the
        // gzippy-isal charter literally ("hand off to ISA-L at the right spot"
        // — at T1 the right spot is one ISA-L call). gzippy-NATIVE has no ISA-L
        // (`isal_clean_tail` false) so it stays ParallelSM at every T.
        // Multi-member/BGZF are classified above and never reach here; and
        // `decompress_gzip_stream` itself loops over trailing gzip members, so
        // a multi-member stream that slipped past the detection window still
        // decodes fully (no truncation).
        #[cfg(isal_clean_tail)]
        {
            if num_threads <= 1 {
                return DecodePath::IsalSingleShot;
            }
        }
        // No thread/size floor: correctness-wise the ParallelSM pipeline
        // handles any size/T (verified byte-exact for tiny / T1 /
        // incompressible / stored), and the C-FFI one-shot is no longer an
        // option, so every remaining single-member stream routes to the
        // ParallelSM pipeline (pure-Rust on gzippy-native; ISA-L clean tail on
        // gzippy-isal, whose T1 took the single-shot route above).
        // Stored-dominated streams have sparse deflate boundaries; the
        // speculative marker pipeline thrashes on them, so use the explicit
        // stored-block parallel decoder (also pure-Rust). It re-validates
        // and, if not actually 100% stored, the dispatch falls back to the
        // pure-Rust marker pipeline (see `decompress_single_member_for`).
        if parallel_sm_unprofitable(data)
            && crate::decompress::parallel::stored_split::first_block_is_stored(data)
        {
            return DecodePath::StoredParallel;
        }
        DecodePath::ParallelSM
    }
    #[cfg(not(parallel_sm))]
    {
        let _ = num_threads;
        if data.len() > 1024 * 1024 * 1024 {
            return DecodePath::StreamingSingle;
        }
        DecodePath::LibdeflateSingle
    }
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
        DecodePath::ParallelSM
        | DecodePath::StoredParallel
        | DecodePath::IsalSingleShot
        | DecodePath::StreamingSingle
        | DecodePath::LibdeflateSingle => {
            decompress_single_member_for(path, data, writer, None, num_threads)
        }
    }
}

/// Like [`decompress_single_member`] but threads an output fd for zero-copy
/// `writev` on the parallel-SM consumer path (file/stdout sinks).
pub(crate) fn decompress_single_member_fd<W: Write>(
    data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    num_threads: usize,
) -> GzippyResult<u64> {
    let path = classify_gzip(data, num_threads);
    if crate::utils::debug_enabled() {
        eprintln!(
            "[gzippy] decompress_single_member path={:?} threads={} bytes={}",
            path,
            num_threads,
            data.len()
        );
    }
    match path {
        DecodePath::MultiMemberSeq | DecodePath::MultiMemberPar | DecodePath::GzippyParallel => {
            decompress_multi_member_sequential(data, writer)
        }
        DecodePath::ParallelSM
        | DecodePath::StoredParallel
        | DecodePath::IsalSingleShot
        | DecodePath::StreamingSingle
        | DecodePath::LibdeflateSingle => {
            decompress_single_member_for(path, data, writer, out_fd, num_threads)
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
    if crate::utils::debug_enabled() {
        eprintln!(
            "[gzippy] decompress_single_member path={:?} threads={} bytes={}",
            path,
            num_threads,
            data.len()
        );
    }
    match path {
        // A multi-member or BGZF stream can reach this non-parallel CLI entry
        // (small files, or `-p1`) — the io dispatcher routes here whenever it
        // can't parallelize, for ANY format. Decode it sequentially
        // member-by-member rather than erroring; the parallel branches handle
        // these when threads/size allow. (Previously a release-compiled-out
        // `debug_assert!` let these fall through to
        // `decompress_single_member_for`, which rejects non-single-member
        // paths -> empty output / terminal error. Caught by the multi-member
        // `-p1` audit: `cat a.gz a.gz | gzippy -d -p1` produced 0 bytes.)
        DecodePath::MultiMemberSeq | DecodePath::MultiMemberPar | DecodePath::GzippyParallel => {
            decompress_multi_member_sequential(data, writer)
        }
        DecodePath::ParallelSM
        | DecodePath::StoredParallel
        | DecodePath::IsalSingleShot
        | DecodePath::StreamingSingle
        | DecodePath::LibdeflateSingle => {
            decompress_single_member_for(path, data, writer, None, num_threads)
        }
    }
}

/// Test-only counter incremented every time a single-member call reaches
/// the libdeflate one-shot backend. Snapshot before/after a decode to
/// verify the no-fallback invariant — any increment from a call that
/// *should* have routed parallel is a bug.
#[cfg(test)]
pub(crate) static LIBDEFLATE_SM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Hard dispatcher. Each arm is terminal — success or `Err`.
fn decompress_single_member_for<W: Write>(
    path: DecodePath,
    data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    num_threads: usize,
) -> GzippyResult<u64> {
    match path {
        DecodePath::ParallelSM => {
            // The parallel pipeline runs and verifies CRC + ISIZE — or
            // returns an error. No fallback. This is the production
            // hot path on x86_64 + ISA-L for inputs ≥ MIN_PARALLEL_COMPRESSED
            // with T > 1.
            let n = crate::decompress::parallel::single_member::decompress_parallel(
                data,
                writer,
                out_fd,
                num_threads,
            )
            .map_err(|e| GzippyError::decompression(format!("parallel SM: {e}")))?;
            writer.flush()?;
            Ok(n)
        }
        DecodePath::StoredParallel => {
            // Non-speculative parallel stored-block decode. The classifier's
            // `first_block_is_stored` heuristic sent us here; if the stream is
            // not actually 100% stored, the decoder reports `NotStoredDominated`
            // WITHOUT writing, and we decode via the safe one-shot path. This is
            // a router correction, NOT a silent backend retry masking a failure:
            // any genuine corruption (CRC / size / LEN mismatch) is terminal.
            use crate::decompress::parallel::stored_split::{
                decompress_stored_parallel, StoredSplitError,
            };
            match decompress_stored_parallel(data, writer, num_threads) {
                Ok(n) => Ok(n),
                Err(StoredSplitError::NotStoredDominated) => {
                    if crate::utils::debug_enabled() {
                        eprintln!(
                            "[gzippy] StoredParallel declined (not pure-stored) → pure-Rust SM"
                        );
                    }
                    // ParallelSM marker pipeline (NOT a C-FFI one-shot). On
                    // gzippy-native the decode graph stays FFI-free; on
                    // gzippy-isal the clean tail uses ISA-L FFI.
                    #[cfg(parallel_sm)]
                    {
                        let n = crate::decompress::parallel::single_member::decompress_parallel(
                            data,
                            writer,
                            out_fd,
                            num_threads,
                        )
                        .map_err(|e| GzippyError::decompression(format!("parallel SM: {e}")))?;
                        writer.flush()?;
                        Ok(n)
                    }
                    #[cfg(not(parallel_sm))]
                    {
                        decompress_single_member_one_shot(data, writer)
                    }
                }
                Err(e) => Err(GzippyError::decompression(format!("stored parallel: {e}"))),
            }
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
        // T1 single-threaded single-member on the gzippy-isal build: ONE ISA-L
        // `isal_inflate` over the whole stream (no chunking pipeline). Verifies
        // the gzip CRC32 + ISIZE via ISA-L's IGZIP_GZIP crc_flag — a non-zero
        // ret => `decompress_gzip_stream` returns None => terminal Err here,
        // NO fallback (rule 5). Constructed only under `isal_clean_tail`; on
        // other builds the variant is never produced by `classify_gzip`, and
        // `decompress_gzip_stream` resolves to the no-op stub. `out_fd` is
        // unused (the writer is the sink; single-shot needs no zero-copy
        // writev plumbing at one thread).
        DecodePath::IsalSingleShot => {
            let _ = out_fd;
            if crate::utils::debug_enabled() {
                eprintln!(
                    "[gzippy] IsalSingleShot: one ISA-L call, {} bytes",
                    data.len()
                );
            }
            let n = crate::backends::isal_decompress::decompress_gzip_stream(data, writer)
                .ok_or_else(|| {
                    GzippyError::decompression("isal single-shot decode failed".to_string())
                })?;
            writer.flush()?;
            Ok(n)
        }
        // unreachable on well-formed callers — `decompress_gzip_libdeflate`
        // routes multi-member / bgzf paths before calling here.
        other => Err(GzippyError::decompression(format!(
            "decompress_single_member_for called with non-single-member path: {other:?}"
        ))),
    }
}

/// Safe one-shot single-member decode — the path the classifier would pick if
/// the parallel/stored fast-paths were unavailable. Used as the correction
/// target when `StoredParallel`'s heuristic mis-fires on a non-stored stream
/// (`NotStoredDominated`). Mirrors the tail of [`classify_gzip`]: ISA-L when
/// available, zlib-ng streaming for >1 GiB, else libdeflate one-shot.
///
/// Legacy-build only: under `parallel_sm` (the default/production build) the
/// ParallelSM pipeline is the sole single-member path and this C-FFI one-shot
/// is off the decode graph. (The pipeline is pure-Rust on gzippy-native; on
/// gzippy-isal its clean tail uses ISA-L FFI — but never this one-shot.)
#[cfg(not(parallel_sm))]
fn decompress_single_member_one_shot<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    if data.len() > 1024 * 1024 * 1024 {
        return decompress_single_member_streaming(data, writer);
    }
    #[cfg(test)]
    LIBDEFLATE_SM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    decompress_single_member_libdeflate(data, writer)
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
                let mut total = result.output_size as u64;
                // Multi-member gzip (RFC 1952 §2.2; `cat a.gz b.gz`, pigz, log
                // rotation): `gzip_decompress_ex` decodes ONE member and
                // reports `input_consumed`. If another member follows, decode
                // the remainder — otherwise we'd SILENTLY TRUNCATE to the first
                // member (classify() can land a multi-member file here when its
                // 2nd member starts past the 16 MiB detection window). Zero cost
                // on a true single member: input_consumed == data.len().
                let consumed = result.input_consumed;
                if consumed < data.len()
                    && data.len() - consumed >= 2
                    && data[consumed] == 0x1f
                    && data[consumed + 1] == 0x8b
                {
                    total += decompress_multi_member_sequential(&data[consumed..], writer)?;
                    return Ok(total);
                }
                writer.flush()?;
                return Ok(total);
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
    #[cfg(parallel_sm)]
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
    /// gate must classify to `ParallelSM` wherever the parallel-SM
    /// pipeline is compiled in (`PARALLEL_SM`) and to
    /// `SingleMember`/`LibdeflateSingle` otherwise — never silently switch
    /// backends inside `decompress_single_member`.
    ///
    /// `PARALLEL_SM` is true on x86_64 (isal-compression | pure-rust-inflate)
    /// AND on aarch64 (pure-rust-inflate) — the latter being the arm64
    /// pure-Rust parallel path. Asserting against the const keeps this test
    /// in lockstep with the gate instead of duplicating the cfg predicate.
    #[test]
    fn test_classify_routes_at_classifier_not_in_body() {
        let (_raw, payload) = compressible_parallel_fixture();
        let path = classify_gzip(&payload, 4);
        if crate::decompress::parallel::sm_cfg::PARALLEL_SM {
            assert_eq!(
                path,
                DecodePath::ParallelSM,
                "size+ratio-eligible input must classify parallel where PARALLEL_SM is on"
            );
        } else {
            assert_ne!(
                path,
                DecodePath::ParallelSM,
                "host without the parallel-SM pipeline must not classify parallel"
            );
        }
    }

    /// The speculative parallel single-member pipeline must engage only at
    /// T≥4 (its per-core throughput is below the ISA-L one-shot until ~4
    /// threads; 2026-05-29 matrix: one-shot T1 1074 > parallel T2 863). At T2/T3
    /// a compressible, size-eligible input must route to the one-shot path.
    #[cfg(parallel_sm)]
    #[test]
    fn test_parallel_sm_thread_gate() {
        let (_raw, payload) = compressible_parallel_fixture();
        // ParallelSM is the single-member path at EVERY thread count on
        // gzippy-native (no C-FFI one-shot to gate toward, task #8). On the
        // gzippy-isal build (`isal_clean_tail`) T1 routes to single-shot ISA-L
        // (DIS-15: the 16-chunk pipeline buys no parallelism at one thread),
        // while T>=2 stays ParallelSM. (Legacy `not(parallel_sm)` build keeps
        // the T>=4 gate that fed the FFI one-shot below it.)
        #[cfg(all(parallel_sm, isal_clean_tail))]
        {
            assert_eq!(
                classify_gzip(&payload, 1),
                DecodePath::IsalSingleShot,
                "T=1 on gzippy-isal must take single-shot ISA-L"
            );
            for t in [2usize, 3, 4] {
                assert_eq!(
                    classify_gzip(&payload, t),
                    DecodePath::ParallelSM,
                    "T={t} on gzippy-isal must take the ParallelSM pipeline"
                );
            }
        }
        #[cfg(all(parallel_sm, not(isal_clean_tail)))]
        for t in [1usize, 2, 3, 4] {
            assert_eq!(
                classify_gzip(&payload, t),
                DecodePath::ParallelSM,
                "T={t} must take the pure-Rust pipeline (sole single-member path)"
            );
        }
        #[cfg(not(parallel_sm))]
        {
            assert_ne!(classify_gzip(&payload, 2), DecodePath::ParallelSM);
            assert_ne!(classify_gzip(&payload, 3), DecodePath::ParallelSM);
            assert_eq!(classify_gzip(&payload, 4), DecodePath::ParallelSM);
        }
    }

    /// Incompressible single-member input above the size gate must NOT take the
    /// SPECULATIVE parallel pipeline — on stored-dominated data the block
    /// finder's spacing guesses never hit a real boundary and that pipeline is
    /// 3-4× slower than a single ISA-L pass (2026-05-29 matrix). The
    /// `parallel_sm_unprofitable` ratio gate keeps it off the speculative path;
    /// instead such input (a stored stream) is routed to the NON-speculative
    /// `StoredParallel` split path on x86_64 + ISA-L/pure-rust, which decodes
    /// the explicit stored-block lengths in parallel. Either way the speculative
    /// `ParallelSM` path is never chosen, and the output is byte-exact.
    #[test]
    fn test_incompressible_single_member_avoids_speculative_pipeline() {
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
            "incompressible input must be flagged unprofitable for the speculative SM"
        );
        // It must NEVER take the speculative pipeline (the slow path on stored
        // data), at any thread count.
        assert_ne!(
            classify_gzip(&compressed, 4),
            DecodePath::ParallelSM,
            "incompressible single-member must avoid the speculative pipeline"
        );
        // On parallel-SM builds it takes the non-speculative stored split
        // instead of single-thread libdeflate.
        #[cfg(parallel_sm)]
        assert_eq!(
            classify_gzip(&compressed, 4),
            DecodePath::StoredParallel,
            "stored incompressible input must take the parallel stored-split path"
        );
        // And it must decode byte-perfectly through whatever path it took.
        let mut out = Vec::new();
        decompress_single_member(&compressed, &mut out, 4).expect("must decode");
        assert_eq!(out, original, "byte-perfect output required");
    }

    /// On a parallel-eligible input the libdeflate one-shot backend must
    /// never be called from the single-member dispatcher. This is the
    /// no-fallback invariant the user asked for: under no circumstances
    /// can a silent libdeflate retry mask a parallel-pipeline failure.
    #[cfg(parallel_sm)]
    #[test]
    fn test_no_libdeflate_fallback_ever_fires_from_sm_path() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // Compressible fixture above MIN_PARALLEL_COMPRESSED so the
        // classifier returns ParallelSM (incompressible would now bail
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
    #[cfg(parallel_sm)]
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

    /// A single-member input *below* the old 10 MiB parallel gate routes
    /// deterministically. Under `parallel_sm` (the production build) the
    /// ParallelSM pipeline is the SOLE single-member path — below-gate inputs
    /// route to it, NEVER to a C-FFI one-shot (task #8). That pipeline is
    /// pure-Rust on gzippy-native; on gzippy-isal its clean tail uses ISA-L
    /// FFI. Under the legacy `not(parallel_sm)` build they route to the C-FFI
    /// one-shot.
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
        #[cfg(parallel_sm)]
        assert!(
            matches!(path, DecodePath::ParallelSM | DecodePath::StoredParallel),
            "below-gate single-member must route pure-Rust under parallel_sm, got {path:?}"
        );
        #[cfg(not(parallel_sm))]
        assert_ne!(
            path,
            DecodePath::ParallelSM,
            "legacy build routes below-gate to the C-FFI one-shot"
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

    /// Regression guard for the stored-block false-positive consumer bug (2026-06-12).
    ///
    /// **Root cause (fixed):** `chunk_fetcher.rs` consumer loop skipped stale
    /// spacing-guess blocks (`!block_is_confirmed && next_block_offset <
    /// furthest_decoded_bit`) but did NOT skip confirmed blocks in the same
    /// condition.  `GzipBlockFinder` scans ALL input bits — including stored
    /// block raw bytes — for BTYPE=10 dynamic Huffman patterns.  When stored
    /// block raw data contains bit patterns that pass precode + full code-table
    /// validation, a false-positive confirmed block is inserted.  After the
    /// stored chunk is decoded (inserted=1), the consumer's index advances by 1
    /// to the false-positive X.  Because `block_is_confirmed=true`, the stale-
    /// block skip did not fire even though `X < furthest_decoded_bit`.  Decoding
    /// at X produced a clean chunk (raw bytes of the stored block interpreted as
    /// literals) that was written to output as garbage → CRC mismatch.
    ///
    /// **Fix:** Remove `!block_is_confirmed &&` so ANY block (confirmed or guess)
    /// at a bit position behind `furthest_decoded_bit` is skipped.  Safe by
    /// invariant: a legitimate confirmed block is ALWAYS at or after
    /// `furthest_decoded_bit` when it reaches `next_unprocessed_block_index`,
    /// because `consumer_append_subchunks_vendor`'s `inserted` count advances the
    /// index past every subchunk within the decoded chunk.
    ///
    /// This test exercises the stored-block code path in ParallelSM using a
    /// mixed compressible/incompressible payload.  The compressible prefix ensures
    /// the file routes to ParallelSM (not StoredParallel) and provides a dynamic
    /// Huffman first block for the block-finder bootstrap.  The full regression
    /// requires `monorepo.tar.gz` from squishy (9.8 MB, triggers actual false
    /// positives); this test catches general stored-block decode regressions.
    #[cfg(parallel_sm)]
    #[test]
    fn test_stored_block_parallel_sm_byte_exact() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        // Compressible prefix: 4-bit entropy (LCG), 200 KiB.
        // Compresses ~2:1 at level 1 → dynamic Huffman blocks, clears the
        // parallel_sm_unprofitable ratio gate (1.15x) for the overall file.
        let mut compressible = vec![0u8; 200 * 1024];
        let mut s: u64 = 0xC0FFEE_1234_5678;
        for b in &mut compressible {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((s >> 32) as u8) & 0x0F;
        }

        // Incompressible suffix: 8-bit entropy, 40 KiB (> MAX_WINDOW_SIZE=32768).
        // Level-1 gzip uses stored blocks for incompressible runs, which is the
        // block type that triggers the false-positive false-positive scanner issue.
        let mut incompressible = vec![0u8; 40 * 1024];
        let mut s2: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for b in &mut incompressible {
            s2 = s2.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (s2 >> 32) as u8;
        }

        let mut original = compressible;
        original.extend_from_slice(&incompressible);

        // Compress at level 1 (fast; often produces stored blocks for high-entropy data).
        let mut compressed = Vec::new();
        {
            let mut enc = GzEncoder::new(&mut compressed, Compression::new(1));
            enc.write_all(&original).unwrap();
            enc.finish().unwrap();
        }

        // Sanity: routing must land on ParallelSM (not StoredParallel or one-shot).
        // The compressible prefix ensures the first block is dynamic Huffman, and
        // the overall ratio should clear 1.15x.
        let path = classify_gzip(&compressed, 4);
        assert!(
            matches!(path, DecodePath::ParallelSM | DecodePath::StoredParallel),
            "must route to parallel path, got {path:?}"
        );

        // Single-threaded reference decode.
        let mut ref_out = Vec::new();
        decompress_single_member(&compressed, &mut ref_out, 1).expect("T=1 decode must succeed");
        assert_eq!(ref_out, original, "T=1 must decode byte-exact");

        // Multi-threaded decode must be byte-identical.
        let mut par_out = Vec::new();
        decompress_single_member(&compressed, &mut par_out, 4).expect("T=4 decode must succeed");
        assert_eq!(
            par_out, original,
            "T=4 parallel decode must be byte-identical to T=1; \
             CRC mismatch here indicates the stored-block false-positive consumer bug"
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
