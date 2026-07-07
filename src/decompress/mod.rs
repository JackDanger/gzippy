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

use std::io::Write;

use crate::decompress::format::{has_bgzf_markers, is_likely_multi_member, read_gzip_isize};
use crate::error::{GzippyError, GzippyResult};

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
///   MultiMemberPar   — pigz-style multi-member, Tmax threads (pure-Rust inflate)
///   MultiMemberSeq   — pigz-style multi-member, T1 (pure-Rust inflate)
///   ParallelSM       — single-member parallel marker pipeline (pure-Rust DEFLATE
///                      engine, every arch). The SOLE single-member decode path.
///   StoredParallel   — stored-block-dominated single-member, non-speculative
///                      parallel split (pure-Rust).
///
/// The single-member sub-paths are fully decided here. The body of
/// `decompress_single_member` is a pure dispatcher — no `if-let-Some-else`
/// fall-through, no silent retry against a different backend, and NO C-FFI in
/// the decode graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    GzippyParallel,
    MultiMemberPar,
    /// Cross-member-capable path: each member is inflated by the full
    /// within-member parallel single-member engine and streamed in member order
    /// (pure-Rust, per-member CRC32 + ISIZE verified). Used as the DETERMINISTIC
    /// upfront route for a MIXED concatenation (a "GZ" first member followed by a
    /// plain member — see [`crate::decompress::bgzf::gz_coverage_is_pure`]), which
    /// the BGZF fast path would truncate. See
    /// [`crate::decompress::parallel::single_member::decompress_multi_member_chunked`].
    ///
    /// NOTE (2026-07-05): this reuses the per-member parallel engine (member walk).
    /// It is NOT routed for plain dominant/few-member distributions: that was
    /// MEASURED to REGRESS on M1 (per-member pipeline spinup + thread
    /// oversubscription at high T). The located dominant-member plateau needs the
    /// rapidgzip-faithful whole-file-block-finder cross-member port, not this
    /// member-walk shortcut — see the gate-phase plan.
    MultiMemberChunked,
    MultiMemberSeq,
    ParallelSM,
    /// Stored-block-dominated (incompressible) single-member, decoded in
    /// parallel WITHOUT speculation by splitting on explicit stored-block
    /// `LEN` fields. See [`crate::decompress::parallel::stored_split`]. The
    /// `parallel_sm_unprofitable` ratio gate routes incompressible input here
    /// when its first block is stored — stored streams are trivially,
    /// bandwidth-bound parallelizable (still pure-Rust).
    StoredParallel,
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
        // A first-member "GZ" FEXTRA hit means gzippy's own parallel format —
        // UNLESS the file is a mixed concatenation (GZ members ++ a plain gzip
        // member, or vice-versa). The GZ fast path (`decompress_bgzf_parallel`)
        // assumes EVERY member carries the subfield; a plain member past the
        // first would desync it. A member-to-member coverage walk over the GZ
        // subfield's whole-member size decides deterministically at classify
        // time (no in-body fallback): pure-GZ (every member carries GZ AND the
        // walk ends exactly at file end) → GzippyParallel (unchanged); any
        // shortfall → the cross-member-capable chunked path. [R2-#3]
        if !crate::decompress::bgzf::gz_coverage_is_pure(data) {
            return DecodePath::MultiMemberChunked;
        }
        return DecodePath::GzippyParallel;
    }
    if is_likely_multi_member(data) {
        // Plain multi-member streams keep the member-per-worker split
        // ([`MultiMemberPar`]) at T>1 / sequential at T1.
        //
        // STAGE-2 STATUS (design §1.2 / [R1-#8]): the routing predicate
        // [`crate::decompress::bgzf::fast_path_ok`] (greedy-LPT/dominance sim
        // over member sizes) is BUILT AND UNIT-TESTED, but it is NOT wired to
        // flip production routing yet. Routing dominant/few-member distributions
        // to today's [`MultiMemberChunked`] body — the per-member MEMBER-WALK —
        // was MEASURED to REGRESS on M1 (per-member pipeline spinup + thread
        // oversubscription at high T; few-large T8 156ms chunked vs 45ms
        // member-per-worker), and that revert is locked by
        // `tests::multi_member_chunked` (uneven/balanced/few_large ⇒
        // MultiMemberPar). The flip is a STAGE-2b action: it becomes correct
        // and non-regressing only once `MultiMemberChunked` IS the
        // rapidgzip-faithful whole-file block-finder cross-member GRID (one
        // pool, one chunk grid spanning members), at which point
        // `fast_path_ok` selects it for dominant streams, gated box-side
        // (OQ-1/OQ-2, stage 3). Until then the predicate stays dead-code-ready.
        return if num_threads > 1 {
            DecodePath::MultiMemberPar
        } else {
            DecodePath::MultiMemberSeq
        };
    }
    // SINGLE-MEMBER. The pure-Rust ParallelSM pipeline is the SOLE single-member
    // decode path at every thread count and size — NO C-FFI one-shot (libdeflate /
    // zlib-ng / ISA-L) anywhere in the decode graph. Correctness-wise it handles
    // any size/T (verified byte-exact for tiny / T1 / incompressible / stored).
    //
    // Stored-dominated streams have sparse deflate boundaries; the speculative
    // marker pipeline thrashes on them, so they take the explicit stored-block
    // parallel decoder (also pure-Rust). It re-validates and, if not actually
    // 100% stored, dispatch falls back to the pure-Rust marker pipeline (see
    // `decompress_single_member_for`).
    let _ = num_threads;
    if parallel_sm_unprofitable(data)
        && crate::decompress::parallel::stored_split::first_block_is_stored(data)
    {
        return DecodePath::StoredParallel;
    }
    DecodePath::ParallelSM
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
/// pure-Rust parallel single-member) based on the input and the requested
/// thread count. The decode graph is pure-Rust end-to-end — no C-FFI.
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
        DecodePath::MultiMemberChunked => {
            decompress_multi_member_chunked(data, writer, num_threads)
        }
        DecodePath::MultiMemberSeq => decompress_multi_member_sequential(data, writer),
        DecodePath::ParallelSM | DecodePath::StoredParallel => {
            decompress_single_member_for(path, data, writer, None, num_threads)
        }
    }
}

/// Dispatch a [`DecodePath::MultiMemberChunked`] stream: each member is inflated
/// by the full within-member parallel single-member engine, streamed in member
/// order (per-member CRC32 + ISIZE verified). Pure-Rust; no C-FFI. On the
/// non-`parallel_sm` legacy build the engine is unavailable, so this decodes
/// sequentially member-by-member (still correct, just not within-member
/// parallel) — same as [`decompress_multi_member_sequential`].
fn decompress_multi_member_chunked<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    #[cfg(parallel_sm)]
    {
        crate::decompress::parallel::single_member::decompress_multi_member_chunked(
            data,
            writer,
            num_threads.max(1),
        )
        .map_err(|e| GzippyError::decompression(format!("multi-member chunked: {e}")))
    }
    #[cfg(not(parallel_sm))]
    {
        let _ = num_threads;
        decompress_multi_member_sequential(data, writer)
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
        DecodePath::MultiMemberChunked => {
            decompress_multi_member_chunked(data, writer, num_threads)
        }
        DecodePath::MultiMemberSeq | DecodePath::MultiMemberPar | DecodePath::GzippyParallel => {
            decompress_multi_member_sequential(data, writer)
        }
        DecodePath::ParallelSM | DecodePath::StoredParallel => {
            let r = decompress_single_member_for(path, data, writer, out_fd, num_threads);
            if crate::utils::debug_enabled() {
                eprintln!(
                    "[gzippy] inject_fired={} (route-B sub-region injector non-inert proof)",
                    crate::decompress::inflate::consume_first_decode::INJECT_FIRED
                        .load(std::sync::atomic::Ordering::Relaxed)
                );
            }
            r
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
        if crate::decompress::bgzf::gz_coverage_is_pure(data) {
            return Ok(crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(
                data,
                num_threads,
            )?);
        }
        // Mixed "GZ" ++ plain concatenation: the deterministic chunked route
        // (matches `classify_gzip`). The BGZF fast path would truncate it.
        let mut out = Vec::new();
        decompress_multi_member_chunked(data, &mut out, num_threads)?;
        return Ok(out);
    }
    if num_threads > 1 && is_likely_multi_member(data) {
        // Plain multi-member keeps the member-per-worker fast path (STAGE-2
        // status: the `fast_path_ok` flip to chunked is deferred to stage-2b —
        // see `classify_gzip`).
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
        DecodePath::MultiMemberChunked => {
            decompress_multi_member_chunked(data, writer, num_threads)
        }
        DecodePath::MultiMemberSeq | DecodePath::MultiMemberPar | DecodePath::GzippyParallel => {
            decompress_multi_member_sequential(data, writer)
        }
        DecodePath::ParallelSM | DecodePath::StoredParallel => {
            let r = decompress_single_member_for(path, data, writer, None, num_threads);
            if crate::utils::debug_enabled() {
                eprintln!(
                    "[gzippy] inject_fired={} (route-B sub-region injector non-inert proof)",
                    crate::decompress::inflate::consume_first_decode::INJECT_FIRED
                        .load(std::sync::atomic::Ordering::Relaxed)
                );
            }
            r
        }
    }
}

/// Test-only pure-Rust single-member decode helper. After the C-FFI decode
/// backends (libdeflate / zlib-ng one-shot) were removed, the unit tests that
/// previously called those helpers route here instead, exercising the SAME
/// pure-Rust single-member production path (`decompress_single_member`, T=1 →
/// ParallelSM). It preserves the historical error semantics those tests assert
/// (corrupt / truncated single-member input → terminal `Err`).
#[cfg(test)]
pub(crate) fn decompress_single_member_pure<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    decompress_single_member(data, writer, 1)
}

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
            // Gate-0 non-inert proof for the engine-A clean-path fastloop
            // wire-in: dump the calls/bytes it actually decoded (process-global).
            #[cfg(all(
                pure_inflate_decode,
                not(all(feature = "asm-kernel", target_arch = "x86_64"))
            ))]
            if crate::utils::debug_enabled() {
                use std::sync::atomic::Ordering::Relaxed;
                eprintln!(
                    "[gzippy] flat_contig calls={} bytes={} careful_calls={} clean_lut_builds={}",
                    crate::decompress::inflate::consume_first_decode::FLAT_CONTIG_CALLS
                        .load(Relaxed),
                    crate::decompress::inflate::consume_first_decode::FLAT_CONTIG_BYTES
                        .load(Relaxed),
                    crate::decompress::inflate::consume_first_decode::FLAT_CAREFUL_CALLS
                        .load(Relaxed),
                    crate::decompress::inflate::consume_first_decode::CLEAN_LUT_BUILDS
                        .load(Relaxed),
                );
            }
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
                Ok(n) => {
                    if crate::utils::debug_enabled() {
                        eprintln!(
                            "[gzippy] StoredParallel ok: {n} bytes; pure-stored chunked-streaming \
                             runs (no monolithic buffer)={}",
                            crate::decompress::parallel::stored_split::STORED_STREAM_RUNS
                                .load(std::sync::atomic::Ordering::Relaxed),
                        );
                    }
                    Ok(n)
                }
                Err(StoredSplitError::NotStoredDominated) => {
                    if crate::utils::debug_enabled() {
                        eprintln!(
                            "[gzippy] StoredParallel declined (not pure-stored) → pure-Rust SM"
                        );
                    }
                    // ParallelSM marker pipeline (pure-Rust; NO C-FFI in the
                    // decode graph).
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
                Err(e) => Err(GzippyError::decompression(format!("stored parallel: {e}"))),
            }
        }
        // unreachable on well-formed callers — `classify_gzip` only produces
        // `ParallelSM` / `StoredParallel` for single-member streams, and
        // `decompress_gzip_libdeflate` routes multi-member / bgzf paths before
        // calling here.
        other => Err(GzippyError::decompression(format!(
            "decompress_single_member_for called with non-single-member path: {other:?}"
        ))),
    }
}

/// Sequential multi-member decompress via the pure-Rust inflate engine
/// (RFC 1952 §2.2: `cat a.gz b.gz`, pigz, log rotation). NO C-FFI: each member's
/// gzip header is parsed in Rust, the raw deflate body is decoded with
/// `inflate_consume_first_bits` (which byte-aligns at BFINAL), and the per-member
/// CRC32 + ISIZE trailer is verified before stepping to the next member.
///
/// Stops cleanly (returning the bytes decoded so far) on the first non-gzip /
/// truncated / corrupt trailing bytes — matching the historical libdeflate-based
/// behavior, where trailing garbage after a valid member terminated the walk
/// without erroring the already-written output.
pub(crate) fn decompress_multi_member_sequential<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    use crate::decompress::format::parse_gzip_header_size;
    use crate::decompress::inflate::consume_first_decode::{inflate_consume_first_bits, Bits};

    let isize_hint = read_gzip_isize(data).unwrap_or(0) as usize;
    let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(256 * 1024)
    };
    let mut output_buf = alloc_aligned_buffer(initial_size);

    let mut total_bytes = 0u64;
    let mut offset = 0usize;
    let mut member_count = 0u32;

    // Minimum gzip member: 10-byte header + ≥2-byte deflate + 8-byte trailer.
    while offset + 18 <= data.len() {
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }
        let header = match parse_gzip_header_size(&data[offset..]) {
            Some(h) if h > 0 && offset + h < data.len() => h,
            _ => break,
        };
        let body_start = offset + header;
        let body = &data[body_start..];

        // DoS/OOM guard: a valid DEFLATE member expands by at most
        // MAX_DEFLATE_EXPANSION:1 (~1032:1). On a malformed body the inflate bit
        // reader zero-pads past end-of-input and fabricates phantom literals
        // forever, so the WriteZero growth loop below doubles `output_buf`
        // without limit — a ~50-byte input drives a multi-GB allocation before
        // OOM-kill. `body` is the whole residual (this member's deflate + its
        // trailer + any following members), so this ceiling over-counts and can
        // never trip on a legitimate high-ratio multi-member stream; it only
        // fires when the decoder is producing implausibly more than the residual
        // input could ever expand to. Mirrors the parallel-SM path's
        // `set_output_ceiling_for_input` (chunk_data.rs).
        const OOM_GUARD_SLACK: usize = 64 * 1024; // one 32 KiB window + headroom
        let output_ceiling = body
            .len()
            .saturating_mul(crate::decompress::parallel::chunk_data::MAX_DEFLATE_EXPANSION)
            .saturating_add(OOM_GUARD_SLACK);

        // Decode the raw deflate body, growing the output buffer on overflow.
        let (out_size, deflate_len) = loop {
            let mut bits = Bits::new(body);
            match inflate_consume_first_bits(&mut bits, &mut output_buf) {
                Ok(n) => break (n, bits.bit_position().div_ceil(8)),
                Err(e) if e.kind() == std::io::ErrorKind::WriteZero => {
                    if output_buf.len() >= output_ceiling {
                        // Runaway: output already exceeds the plausible ceiling
                        // for this member's compressed length — reject as
                        // malformed (terminal Err) instead of OOMing.
                        return Err(GzippyError::decompression(format!(
                            "multi-member: decoded output {} exceeds plausible ceiling {} \
                             for {}-byte residual input — malformed stream",
                            output_buf.len(),
                            output_ceiling,
                            body.len()
                        )));
                    }
                    let new_size = output_buf
                        .len()
                        .saturating_mul(2)
                        .max(256 * 1024)
                        .min(output_ceiling);
                    output_buf.resize(new_size, 0);
                    continue;
                }
                // Corrupt / truncated trailing bytes: stop the walk (parity with
                // the prior libdeflate `BadData => break`).
                Err(_) => {
                    writer.flush()?;
                    return Ok(total_bytes);
                }
            }
        };

        // Verify the 8-byte gzip trailer (CRC32 + ISIZE) for this member.
        let trailer = body_start + deflate_len;
        if trailer + 8 > data.len() {
            break; // truncated trailer
        }
        let expected_crc = u32::from_le_bytes([
            data[trailer],
            data[trailer + 1],
            data[trailer + 2],
            data[trailer + 3],
        ]);
        let expected_isize = u32::from_le_bytes([
            data[trailer + 4],
            data[trailer + 5],
            data[trailer + 6],
            data[trailer + 7],
        ]);
        if (out_size as u32) != expected_isize {
            return Err(GzippyError::decompression(format!(
                "multi-member: ISIZE mismatch (got {out_size}, expected {expected_isize})"
            )));
        }
        let actual_crc = crc32fast::hash(&output_buf[..out_size]);
        if actual_crc != expected_crc {
            return Err(GzippyError::decompression(
                "multi-member: CRC32 mismatch".to_string(),
            ));
        }

        member_count += 1;
        if crate::utils::debug_enabled() {
            eprintln!(
                "[gzippy] sequential member {member_count}: out_size={out_size} \
                 deflate_len={deflate_len} offset={offset}/{}",
                data.len()
            );
        }
        writer.write_all(&output_buf[..out_size])?;
        total_bytes += out_size as u64;
        offset = trailer + 8;
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
/// Pure-Rust (`inflate_consume_first`) with a growing output buffer — NO C-FFI.
#[allow(dead_code)] // called from lib.rs; unused in the binary
pub fn decompress_raw_bytes(data: &[u8]) -> GzippyResult<Vec<u8>> {
    use crate::decompress::inflate::consume_first_decode::inflate_consume_first;
    const CAP: usize = 1 << 31; // 2 GiB output ceiling
    let mut estimate = data.len().saturating_mul(4).clamp(4096, CAP);

    loop {
        let mut out = vec![0u8; estimate];
        match inflate_consume_first(data, &mut out) {
            Ok(n) => {
                out.truncate(n);
                return Ok(out);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WriteZero && estimate < CAP => {
                estimate = (estimate * 2).min(CAP);
            }
            Err(_) => return Err(GzippyError::decompression("invalid raw DEFLATE data")),
        }
    }
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

    /// The pure-Rust ParallelSM pipeline is the SOLE single-member decode path
    /// at every thread count — there is no C-FFI one-shot to gate toward, so a
    /// compressible, size-eligible input must classify `ParallelSM` at T1–T4.
    #[test]
    fn test_parallel_sm_thread_gate() {
        let (_raw, payload) = compressible_parallel_fixture();
        for t in [1usize, 2, 3, 4] {
            assert_eq!(
                classify_gzip(&payload, t),
                DecodePath::ParallelSM,
                "T={t} must take the pure-Rust pipeline (sole single-member path)"
            );
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

    /// The single-member dispatcher decodes a parallel-eligible input purely
    /// through the pure-Rust ParallelSM pipeline. The historical "no silent
    /// libdeflate fallback" invariant is now enforced at COMPILE TIME: there is
    /// no C-FFI one-shot decode backend left in the decode graph to fall back to
    /// (the libdeflate/zlib-ng one-shot paths were deleted). This test asserts
    /// byte-exact decode through the sole pure-Rust path.
    #[cfg(parallel_sm)]
    #[test]
    fn test_no_libdeflate_fallback_ever_fires_from_sm_path() {
        let _guard = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let (original, compressed) = compressible_parallel_fixture();
        assert_eq!(classify_gzip(&compressed, 4), DecodePath::ParallelSM);
        let mut out = Vec::new();
        decompress_single_member(&compressed, &mut out, 4).expect("must succeed");
        assert_eq!(out, original, "byte-perfect output required");
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
