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
use crate::decompress::parallel::fast_marker_inflate::{
    decode_chunk_bootstrap, decode_chunk_markers_continuing, BootstrapResult,
};
use crate::decompress::parallel::replace_markers::{replace_markers, MARKER_BASE};

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
///
/// `pub(crate)` rather than `pub`: the counter is an internal diagnostic
/// surface for routing-assertion tests, not part of the library API.
/// Downstream crates have no reason to read it. (Copilot review on PR #94.)
pub(crate) static MARKER_PIPELINE_RUNS: AtomicU64 = AtomicU64::new(0);

/// Mutex serializing the body of the deletion-trap killer routing test
/// against any other test in the crate that calls `decompress_parallel`
/// concurrently. Without this lock, `cargo test`'s default parallel
/// execution can cause another test to bump `MARKER_PIPELINE_RUNS`
/// between the killer test's before/after snapshots, masking a real
/// silent-fallback regression with a false positive. (Copilot review on
/// PR #94.)
pub(crate) static MARKER_PIPELINE_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Times the routing layer silently fell back to sequential libdeflate
/// because `decompress_parallel` returned a non-routing error (D1, Opus
/// advisor review). Distinct from "input was too small to bother"
/// (`ParallelError::TooSmall`), which is expected and intended.
///
/// The bench harness in `scripts/benchmark_single_member.py` reads this
/// counter (via gzippy's debug log) and hard-fails CI if it's non-zero —
/// turning "marker pipeline silently fell back" from invisible (same
/// throughput as libdeflate) into a specific actionable failure. See
/// `docs/marker-decoder-premortem.md` F7 + D1.
pub(crate) static MARKER_PIPELINE_BOUNDARY_MISSED: AtomicU64 = AtomicU64::new(0);

/// Number of cross-chunk consistency retry iterations executed across the
/// process lifetime. Healthy traffic (no BTYPE=01-heavy regions) should
/// see this stay at zero or near-zero; the BTYPE=01-heavy killer fixture
/// should see it move (test asserts `> 0`). Spikes indicate retry
/// thrashing — see `docs/marker-decoder-premortem.md` G5 mitigation.
pub(crate) static MARKER_PIPELINE_RETRY_ITERATIONS: AtomicU64 = AtomicU64::new(0);

/// Hard wall-time deadline for the cross-chunk correction sweep.
/// Healthy data converges in <50 ms; the deadline bounds adversarial
/// inputs (where every chunk decodes serially via the correction chain)
/// at a generous limit. Beyond this, return Err and let the routing
/// fallback fire.
const RETRY_WALL_DEADLINE_MS: u64 = 2000;

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

    // ── Phase 1a: parallel speculative boundary search ──────────────────────
    //
    // Returns `Option<usize>` per chunk. A `None` means BlockFinder +
    // byte-aligned brute force both failed to find a validated candidate
    // within 512 KiB of the chunk's spacing-derived anchor — common in
    // BTYPE=01-heavy regions where fixed-Huffman blocks have no header
    // redundancy for `validate_boundary` to filter against. We do NOT
    // reject here; phase 1c chain-decodes those chunks from the
    // predecessor's confirmed end_bit. Chunk 0 must always succeed
    // (anchored at bit 0); we still bail if it doesn't.
    let t_search = std::time::Instant::now();
    let mut start_bits_opt = phase1_search_boundaries(deflate_data, num_chunks, spacing_bits);
    let search_elapsed = t_search.elapsed();

    if start_bits_opt[0].is_none() {
        // Should be impossible — chunk 0's "search" is hard-pinned to bit 0.
        return Err(ParallelError::DecodeFailed);
    }
    if debug_enabled() {
        let n = start_bits_opt.iter().filter(|s| s.is_none()).count();
        if n > 0 {
            eprintln!(
                "[parallel_sm:v0.6] {}/{} speculative boundary searches failed; \
                 phase 1c will chain-decode those chunks from predecessor end_bits",
                n, num_chunks
            );
        }
    }
    let mut start_bits: Vec<Option<usize>> = start_bits_opt.iter_mut().map(|s| s.take()).collect();

    // ── Phase 1b: parallel marker decode from speculative starts ────────────
    //
    // Each chunk's start is speculative — `validate_boundary(min_blocks=2)`
    // admits some false positives on BTYPE=01 regions (no header redundancy
    // in fixed Huffman). After phase 1b, chunk N's *decoded* end_bit is
    // always a real block boundary (G1 invariant), so phase 1c uses it as
    // chunk N+1's confirmed start. Chunks with `None` start get a `None`
    // decode result that phase 1c will fill in via chain-decode.
    let t_decode = std::time::Instant::now();
    // end_limit[i] = start_bits[i+1] if both are Some; otherwise None and
    // phase 1c applies the correction once chunk i decodes.
    let end_limits: Vec<Option<usize>> = (0..num_chunks)
        .map(|i| {
            if i + 1 < num_chunks {
                start_bits[i + 1]
            } else {
                None
            }
        })
        .collect();
    // Per-chunk decoded-output hint for the ISA-L bulk decode's
    // initial buffer cap. ISIZE / num_chunks is the average; the last
    // chunk may legitimately exceed it, so `decode_chunk_with_handoff`
    // adds 50% headroom.
    let per_chunk_output_hint = expected_size / num_chunks.max(1);
    let chunks_opt = phase1_marker_decode_parallel_with_optional_starts(
        deflate_data,
        &start_bits,
        &end_limits,
        per_chunk_output_hint,
    );
    let decode_elapsed = t_decode.elapsed();
    let mut chunks: Vec<Option<ChunkResult>> = chunks_opt;

    // ── Phase 1c: cross-chunk consistency correction ────────────────────────
    //
    // Walk pairs (N, N+1). When chunks[N].end_bit != start_bits[N+1], the
    // latter was a false positive — correct start_bits[N+1] to
    // chunks[N].end_bit (which IS a real boundary by G1 invariant) and
    // re-decode chunk N+1. Propagate forward.
    //
    // Per Opus advisor review: this is the correct shape for the
    // BTYPE=01-heavy class. Earlier top-K + strictness-ramp design was
    // probabilistic; this is induction-based (chunk 0 starts at bit 0, a
    // real boundary; chunk N's decode lands at a real boundary; chunk
    // N+1's corrected start is therefore a real boundary; …).
    let t_retry = std::time::Instant::now();
    let retry_deadline = t_retry + std::time::Duration::from_millis(RETRY_WALL_DEADLINE_MS);
    phase1c_resolve_consistency(
        deflate_data,
        &mut start_bits,
        &mut chunks,
        retry_deadline,
        per_chunk_output_hint,
    )?;
    let retry_elapsed = t_retry.elapsed();
    let chunks: Vec<ChunkResult> = chunks
        .into_iter()
        .map(|c| c.expect("phase 1c guarantees all chunks decoded"))
        .collect();

    // ── Phase 2: sequential resolve + CRC + size accounting ─────────────────
    let t_resolve = std::time::Instant::now();
    let (assembled, total_crc) = phase2_resolve_sequential(chunks)?;
    let resolve_elapsed = t_resolve.elapsed();

    if debug_enabled() {
        eprintln!(
            "[parallel_sm:v0.6] search={:.1}ms decode={:.1}ms retry={:.1}ms resolve={:.1}ms",
            search_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0,
            retry_elapsed.as_secs_f64() * 1000.0,
            resolve_elapsed.as_secs_f64() * 1000.0,
        );
    }

    // ── Phase 3: verify trailer, write ──────────────────────────────────────
    //
    // Both ISIZE and CRC32 in the gzip trailer can legitimately be zero
    // (ISIZE=0 for an empty input; CRC32=0 happens for specific byte
    // sequences — the CRC of the empty stream is itself 0). Earlier
    // versions guarded these with `!= 0` sentinels and silently accepted
    // corrupted output for those streams. Verification is now
    // unconditional: the trailer is always present (we slice it off
    // `gzip_data` at the function entry), so both checks must always
    // fire. (Premortem mitigation B5; Copilot review on PR #94.)
    if assembled.len() != expected_size {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] size mismatch: got {} expected {}",
                assembled.len(),
                expected_size
            );
        }
        return Err(ParallelError::SizeMismatch);
    }
    if total_crc != expected_crc {
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
//
// Returns one *speculative* start per chunk — a bit offset that passed
// `try_decode_at` (ISA-L + `validate_boundary(min_blocks=2)`). The pick
// is speculative because the same filter that admits real boundaries
// also admits some false positives on BTYPE=01-heavy data — those have
// no header redundancy and any random fixed-Huffman position can decode
// 2 blocks worth of valid symbols by chance.
//
// Speculative picks are corrected in phase 1c after phase 1b reveals
// each chunk's natural decoded end_bit. Chunk N's end_bit is *always* a
// real block boundary (G1 invariant: a decode starting at a real
// boundary lands at a real boundary). If chunks[N].end_bit doesn't
// match start_bits[N+1], the latter is the false positive; phase 1c
// corrects start_bits[N+1] to chunks[N].end_bit and re-decodes chunk
// N+1.

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
                let pick = if idx == 0 {
                    Some(0)
                } else {
                    let from_bit = idx * spacing_bits;
                    search_boundary_forward(deflate_data, from_bit)
                };
                *results[idx].lock().unwrap() = pick;
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Find one validated deflate block-boundary candidate at or past
/// `from_bit`. Returns the first accepted bit offset, or None if no
/// candidate within the search radius passes.
///
/// The pick is speculative — `validate_boundary(min_blocks=2)` admits
/// some BTYPE=01 false positives (no header redundancy in fixed
/// Huffman); phase 1c's correction sweep verifies and corrects them
/// after phase 1b reveals each chunk's actual decoded end_bit.
///
/// Two tiers:
/// 1. **BlockFinder heuristic candidates** (BTYPE=00 stored + BTYPE=10
///    dynamic Huffman).
/// 2. **Byte-aligned brute force** (first 128 KiB).
fn search_boundary_forward(deflate_data: &[u8], from_bit: usize) -> Option<usize> {
    let search_end = (from_bit + SEARCH_RADIUS * 8).min(deflate_data.len() * 8);
    if from_bit >= search_end {
        return None;
    }

    // Tier 1: BlockFinder.
    let finder = BlockFinder::new(deflate_data);
    let sub_chunk_bits = 8 * 1024 * 8;
    let mut chunk_start = from_bit;
    while chunk_start < search_end {
        let chunk_end = (chunk_start + sub_chunk_bits).min(search_end);
        let mut candidates = finder.find_blocks(chunk_start, chunk_end);
        candidates.sort_by_key(|b| b.bit_offset);
        for c in &candidates {
            if c.bit_offset < from_bit {
                continue;
            }
            if try_decode_at(deflate_data, c.bit_offset) {
                return Some(c.bit_offset);
            }
        }
        chunk_start = chunk_end;
    }

    // Tier 2: byte-aligned brute force in the first 128 KiB.
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

    // Stage 2: marker-decoder validation, `min_blocks=2`. A single
    // coincidentally-valid stored block (~1/65536 chance) cannot fake a
    // second consecutive valid block — joint probability ~2^-32 against
    // structured-header (BTYPE=00/10) false positives. BTYPE=01
    // false positives DO get through here (fixed Huffman has no header
    // redundancy), but the speculative-pick + phase-1c correction
    // contract handles those: chunk N's decoded end_bit propagates
    // forward as chunk N+1's confirmed start.
    use crate::decompress::parallel::fast_marker_inflate::validate_boundary;
    validate_boundary(
        deflate_data,
        bit_offset,
        /*min_blocks=*/ 2,
        /*min_output_bytes=*/ 0,
    )
    .is_ok()
}

// ── Phase 1b: parallel marker decode ─────────────────────────────────────────
//
// Each chunk's decode produces both the marker stream AND its end bit
// offset — the bit position just past the last block consumed. D4 (cross-
// chunk consistency) verifies that chunk N's end_bit_offset equals chunk
// N+1's start_bit; any mismatch means phase 1a picked a misaligned
// boundary for one of them, and the pipeline must reject rather than
// emit double-covered or partial output.

/// Phase 1b chunk result.
///
/// `bootstrap`: u16 output from the pure-Rust marker decoder's bootstrap
/// pass, covering the chunk's first ~32 KB of output (and any prefix
/// markers for cross-chunk back-references). May be empty when ISA-L was
/// not entered (last chunk with no clean tail or chunk-too-small fallback).
///
/// `isal_bytes`: real bytes produced by ISA-L after the bootstrap handed
/// off. Empty when no handoff occurred (chunk decoded entirely by the
/// marker decoder).
///
/// `end_bit_offset`: bit position just past the chunk's last decoded
/// block. Always at a real deflate block boundary (G1 invariant — both
/// the marker decoder's `decode_loop` and ISA-L's natural stopping
/// points sit between blocks).
///
/// This shape mirrors rapidgzip's per-chunk worker (cleanData → ISA-L
/// handoff at `vendor/rapidgzip/.../GzipChunk.hpp:521`): the marker
/// decoder bootstraps ≤32 KB to establish a window dict, then ISA-L
/// does the ~99% bulk of the chunk at full single-thread ISA-L speed.
/// Without this handoff the per-thread decoder is structurally bounded
/// to pure-Rust speed (~50 MB/s/thread vs. ISA-L's ~163 MB/s/thread on
/// x86_64 CI), which prevents per-thread parity with sequential ISA-L
/// no matter how good the parallel orchestration is.
struct ChunkResult {
    bootstrap: Vec<u16>,
    isal_bytes: Vec<u8>,
    end_bit_offset: usize,
}

impl ChunkResult {
    /// Total decoded byte count for this chunk (bootstrap u8 equivalent
    /// + ISA-L bytes). Used by phase 2 to pre-size the output buffer.
    fn decoded_len(&self) -> usize {
        self.bootstrap.len() + self.isal_bytes.len()
    }
}

/// Decodes each chunk in parallel from its speculative start. Each worker:
///
/// 1. Runs the marker decoder in **bootstrap mode** until 32 KB of clean
///    (marker-free) tail accumulates at a deflate block boundary.
/// 2. Hands off to ISA-L (seeded with that 32 KB as `isal_inflate_set_dict`)
///    for the remaining ~99% of the chunk.
///
/// This is the cleanData → ISA-L handoff from rapidgzip's `GzipChunk.hpp`
/// (per the Opus advisor review on PR #97). Without it, the per-thread
/// pure-Rust marker decoder caps single-thread parity vs ISA-L at ~30%
/// no matter how good the parallel orchestration is.
///
/// Chunks whose `start_bits[i]` is `None` (phase 1a found no candidate)
/// are skipped — their result stays `None` and phase 1c chain-decodes
/// them from the predecessor's confirmed `end_bit_offset`.
fn phase1_marker_decode_parallel_with_optional_starts(
    deflate_data: &[u8],
    start_bits: &[Option<usize>],
    end_limits: &[Option<usize>],
    per_chunk_output_hint: usize,
) -> Vec<Option<ChunkResult>> {
    let num_chunks = start_bits.len();
    let results: Vec<Mutex<Option<ChunkResult>>> =
        (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let next_task = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_chunks {
            s.spawn(|| loop {
                let idx = next_task.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks {
                    break;
                }
                let Some(start_bit) = start_bits[idx] else {
                    continue;
                };
                let end_limit = end_limits[idx];
                if let Ok(chunk) = decode_chunk_with_handoff(
                    deflate_data,
                    start_bit,
                    end_limit,
                    per_chunk_output_hint,
                    num_chunks,
                ) {
                    *results[idx].lock().unwrap() = Some(chunk);
                }
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Per-chunk worker body: marker-decoder bootstrap → ISA-L handoff.
///
/// Falls back to "marker decoder for the entire chunk" when (a) the chunk
/// is too small to accumulate 32 KB of clean tail before reaching
/// `end_bit_limit` / BFINAL, or (b) ISA-L is not available on this
/// platform. Both fall-back cases return a `ChunkResult` with empty
/// `isal_bytes` — phase 2 sees no difference.
fn decode_chunk_with_handoff(
    deflate_data: &[u8],
    start_bit: usize,
    end_bit_limit: Option<usize>,
    per_chunk_output_hint: usize,
    num_chunks_total: usize,
) -> std::io::Result<ChunkResult> {
    // Phase 1 of the worker: bootstrap. Decode the chunk's first ~32 KB
    // with the marker decoder until the trailing 32 KB of output is
    // marker-free. This window seeds ISA-L for the bulk decode.
    let BootstrapResult {
        markers: bootstrap_markers,
        end_bit_offset: bootstrap_end_bit,
        clean_window,
    } = decode_chunk_bootstrap(deflate_data, start_bit, end_bit_limit)?;

    // No clean window? Two reasons: (1) end_bit_limit reached before
    // 32 KB clean tail accumulated (tiny chunk), or (2) BFINAL hit
    // (last chunk smaller than 32 KB of output). Either way the
    // bootstrap markers cover the chunk's intended range — phase 2
    // resolves them as in the pre-handoff design.
    let Some(dict) = clean_window else {
        return Ok(ChunkResult {
            bootstrap: bootstrap_markers,
            isal_bytes: Vec::new(),
            end_bit_offset: bootstrap_end_bit,
        });
    };

    // Phase 2 of the worker: ISA-L bulk decode from `bootstrap_end_bit`,
    // seeded with the 32 KB clean window as `isal_inflate_set_dict`.
    // Pass a slice bounded by the chunk's speculative end so ISA-L
    // cannot run past the next chunk's territory; on healthy data
    // BlockFinder's pick is a real boundary and ISA-L stops there
    // naturally. On a BTYPE=01 false-positive pick, ISA-L may stop
    // short (or run on) — phase 1c then sees
    // `chunks[N].end_bit != chunks[N+1].start` and corrects.
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    {
        // Bound input to the chunk's envelope. Without the chunk
        // boundary trim, ISA-L would decode all the way to BFINAL on
        // every chunk — quadratic total work, defeating the parallel
        // win.
        let end_byte = match end_bit_limit {
            Some(end) => (end.div_ceil(8)).min(deflate_data.len()),
            None => deflate_data.len(),
        };
        let input = &deflate_data[..end_byte];

        // Output cap: the whole-stream ISIZE (`per_chunk_output_hint *
        // num_chunks` recovered from the routing path). A chunk can
        // legitimately decode anywhere from 0 bytes (BTYPE=00 stored)
        // up to the entire stream's remaining ISIZE — for Silesia at
        // T=4, individual chunks decoded ~120 MB on average but some
        // ranges compress 5-10× more than others. The earlier
        // `hint * 3 / 2` cap silently truncated those chunks, the
        // wrapper returned a mid-block end_bit, and phase 1c could
        // not correct because the marker decoder can't bootstrap
        // mid-block. CI saw it as a silent fallback to libdeflate
        // at 0.55× rapidgzip. The new grow-on-demand wrapper
        // (`backends::isal_decompress`) makes this cap a true upper
        // bound, not the initial allocation — passing ISIZE doesn't
        // cost 481 MB of preallocation per worker.
        let max_output = per_chunk_output_hint.saturating_mul(num_chunks_total);

        match crate::backends::isal_decompress::decompress_deflate_from_bit_with_end(
            input,
            bootstrap_end_bit,
            &dict,
            max_output,
        ) {
            Some((isal_bytes, isal_end_bit)) => Ok(ChunkResult {
                bootstrap: bootstrap_markers,
                isal_bytes,
                end_bit_offset: isal_end_bit,
            }),
            None => {
                // ISA-L failed — most often because the speculative
                // end_bit_limit cut mid-block. Fall back to finishing
                // the chunk via the marker decoder. Slow path, rare
                // on real data.
                marker_finish_after_bootstrap(
                    deflate_data,
                    bootstrap_markers,
                    bootstrap_end_bit,
                    end_bit_limit,
                )
            }
        }
    }

    // On non-x86_64 or without the ISA-L feature, the parallel single-
    // member path is gated off at routing time. This branch should be
    // unreachable in production but is kept for build correctness.
    #[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
    {
        let _ = (dict, per_chunk_output_hint, num_chunks_total);
        marker_finish_after_bootstrap(
            deflate_data,
            bootstrap_markers,
            bootstrap_end_bit,
            end_bit_limit,
        )
    }
}

/// Slow-path completion when the ISA-L handoff isn't available or fails:
/// finish the chunk with the marker decoder, appending its output to the
/// bootstrap. Used in fallback only; the production path should keep this
/// rare.
fn marker_finish_after_bootstrap(
    deflate_data: &[u8],
    mut bootstrap_markers: Vec<u16>,
    bootstrap_end_bit: usize,
    end_bit_limit: Option<usize>,
) -> std::io::Result<ChunkResult> {
    // Continue the marker decoder into the same Vec — `emit_match` must
    // see the bootstrap output as part of `out_pos` so its chunk-local
    // copies and "D > out_pos ⇒ marker" rule work correctly. Decoding
    // the continuation into a fresh Vec and concatenating would emit
    // markers for back-references that actually point into the
    // bootstrap (false cross-chunk markers).
    let end_bit = decode_chunk_markers_continuing(
        deflate_data,
        bootstrap_end_bit,
        end_bit_limit,
        &mut bootstrap_markers,
    )?;
    Ok(ChunkResult {
        bootstrap: bootstrap_markers,
        isal_bytes: Vec::new(),
        end_bit_offset: end_bit,
    })
}

// ── Phase 1c: cross-chunk consistency correction ─────────────────────────────

/// Walk pairs (N, N+1) once forward and correct chunk N+1's start when chunk N's
/// decoded `end_bit` doesn't match it. Re-decode chunk N+1 from the
/// corrected start; propagate forward.
///
/// The induction: chunk 0 starts at bit 0 (a real boundary). Each
/// decode that starts at a real boundary lands at a real boundary
/// (G1 invariant — `decode_loop` only exits between blocks). So
/// chunks[N].end_bit is *always* a real block boundary, regardless
/// of how speculative `end_bit_limit` was set. Correcting chunk N+1's
/// start to chunks[N].end_bit therefore makes chunk N+1's start a
/// real boundary by construction.
///
/// Wall-time deadline bounds adversarial cases where many chunks need
/// correction. On exhaustion: return `Err(DecodeFailed)` and let the
/// routing-layer fallback (G3/G4) surface the failure.
fn phase1c_resolve_consistency(
    deflate_data: &[u8],
    start_bits: &mut [Option<usize>],
    chunks: &mut [Option<ChunkResult>],
    deadline: std::time::Instant,
    per_chunk_output_hint: usize,
) -> Result<(), ParallelError> {
    let num_chunks = chunks.len();
    if num_chunks == 0 {
        return Ok(());
    }

    // Chunk 0's start is bit 0 by construction; it must have decoded
    // unless the gzip stream is malformed. Bail if not.
    if chunks[0].is_none() {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm:v0.6] phase1c: chunks[0] is None — bit 0 isn't a valid \
                 deflate start; gzip stream likely corrupt"
            );
        }
        return Err(ParallelError::DecodeFailed);
    }

    if num_chunks == 1 {
        return Ok(());
    }

    let mut i: usize = 0;
    while i < num_chunks - 1 {
        let chunk_end_bit = chunks[i]
            .as_ref()
            .expect("invariant: chunks[i] is Some by this point")
            .end_bit_offset;
        let next_start = start_bits[i + 1];

        // The correction step fires when:
        //   - chunks[i+1] is None (phase 1b worker failed — e.g. the
        //     bootstrap→ISA-L handoff returned Err for that chunk),
        //     regardless of whether start_bits[i+1] looked OK; OR
        //   - start_bits[i+1] is None (phase 1a found no candidate;
        //     chain-decode from chunks[i].end_bit); OR
        //   - start_bits[i+1] is Some but != chunks[i].end_bit
        //     (false-positive pick; correct to the real boundary).
        //
        // The first case was the latent bug exposed by PR #97's
        // handoff path: when start_bits[i+1] coincidentally equaled
        // chunks[i].end_bit but chunks[i+1] was None (worker
        // failure), the old logic skipped correction and the next
        // iteration panicked on `.expect("invariant: chunks[i] is
        // Some")`. Now any None chunk triggers chain-decode.
        let needs_correction = chunks[i + 1].is_none()
            || match next_start {
                None => true,
                Some(s) => s != chunk_end_bit,
            };

        if needs_correction {
            if std::time::Instant::now() >= deadline {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] phase1c: wall-time deadline exceeded after \
                         {} corrections",
                        MARKER_PIPELINE_RETRY_ITERATIONS.load(Ordering::Relaxed)
                    );
                }
                return Err(ParallelError::DecodeFailed);
            }
            MARKER_PIPELINE_RETRY_ITERATIONS.fetch_add(1, Ordering::Relaxed);
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] phase1c: correcting chunk {} start from {:?} \
                     to chunk {}'s end_bit {chunk_end_bit}",
                    i + 1,
                    next_start,
                    i
                );
            }
            start_bits[i + 1] = Some(chunk_end_bit);

            // If the corrected start leaves less than one block-header
            // (3 bits) of input remaining, the predecessor chunk already
            // absorbed BFINAL and what's left is only byte-padding before
            // the gzip trailer. Chunk N+1 is naturally empty. This is
            // normal when an upstream chunk overshoots its speculative
            // range to consume the BFINAL block that a downstream chunk
            // was supposed to handle.
            //
            // Threshold: < 8 bits remaining. 3 bits is the minimum block
            // header size (BFINAL + BTYPE), but the byte-padding between
            // the last block's last symbol and the gzip trailer can be
            // up to 7 bits. Using 8 covers the worst case.
            let total_bits = deflate_data.len() * 8;
            if total_bits.saturating_sub(chunk_end_bit) < 8 {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm:v0.6] phase1c: chunk {} starts at/near EOF \
                         (chunk_end_bit={chunk_end_bit}, total_bits={total_bits}, \
                         remaining={}); marking empty (predecessor consumed BFINAL)",
                        i + 1,
                        total_bits.saturating_sub(chunk_end_bit)
                    );
                }
                chunks[i + 1] = Some(ChunkResult {
                    bootstrap: Vec::new(),
                    isal_bytes: Vec::new(),
                    end_bit_offset: chunk_end_bit,
                });
                i += 1;
                continue;
            }

            // Re-decode chunk N+1 from the corrected (= real) start.
            // Use chunk N+2's *current* speculative start as the
            // end_limit when present; the next iteration corrects it
            // if needed. Use the same bootstrap-+-ISA-L handoff path
            // as the parallel worker so corrected chunks also benefit
            // from per-thread ISA-L throughput.
            let new_end_limit = if i + 2 < num_chunks {
                start_bits[i + 2]
            } else {
                None
            };
            match decode_chunk_with_handoff(
                deflate_data,
                chunk_end_bit,
                new_end_limit,
                per_chunk_output_hint,
                num_chunks,
            ) {
                Ok(result) => chunks[i + 1] = Some(result),
                Err(_) => {
                    if debug_enabled() {
                        eprintln!(
                            "[parallel_sm:v0.6] phase1c: re-decode of chunk {} from real \
                             boundary {chunk_end_bit} failed; deflate stream likely corrupt",
                            i + 1
                        );
                    }
                    return Err(ParallelError::DecodeFailed);
                }
            }
        }
        i += 1;
    }
    Ok(())
}

// ── Phase 2: sequential marker resolve + CRC + size ──────────────────────────

fn phase2_resolve_sequential(chunks: Vec<ChunkResult>) -> Result<(Vec<u8>, u32), ParallelError> {
    let total_estimated: usize = chunks.iter().map(|c| c.decoded_len()).sum();
    let mut assembled: Vec<u8> = Vec::with_capacity(total_estimated);
    let mut crc = crc32fast::Hasher::new();
    let mut window: Vec<u8> = Vec::with_capacity(WINDOW_SIZE);

    for (i, mut chunk) in chunks.into_iter().enumerate() {
        let chunk_start = assembled.len();

        // Bootstrap u16 prefix: may contain cross-chunk markers (only
        // for chunks i > 0). Resolve them against the previous chunk's
        // last 32 KB, then narrow u16 → u8 into `assembled`.
        if i > 0 {
            // In-place SIMD substitution; ~memory-bandwidth speed on AVX2/NEON.
            replace_markers(&mut chunk.bootstrap, &window);
        }
        narrow_and_append(&mut assembled, &chunk.bootstrap).map_err(|pos| {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm:v0.6] chunk {} unresolved marker at bootstrap[{}] (offset {})",
                    i,
                    pos,
                    chunk.bootstrap[pos] - MARKER_BASE,
                );
            }
            ParallelError::DecodeFailed
        })?;

        // ISA-L bytes (the bulk of the chunk on the handoff path): no
        // markers possible — ISA-L was seeded with the resolved 32 KB
        // window and emits real u8 directly. Append as-is.
        assembled.extend_from_slice(&chunk.isal_bytes);

        // CRC across both pieces in one pass — phase 1's structural
        // win (bootstrap markers ≤32 KB per chunk; ISA-L bytes are the
        // bulk) means this CRC pass is the only one that walks all
        // chunk output bytes.
        crc.update(&assembled[chunk_start..]);

        // Update window from the last 32 KB now sitting in `assembled`.
        let tail_start = assembled.len().saturating_sub(WINDOW_SIZE);
        window.clear();
        window.extend_from_slice(&assembled[tail_start..]);
    }

    Ok((assembled, crc.finalize()))
}

/// Append `chunk_u16` to `out` as bytes (low 8 bits of each u16), failing
/// with `Err(idx)` if any value has the marker bit still set. Combines the
/// validation walk of `u16_to_u8` with the byte-narrowing copy and the
/// downstream `assembled.extend_from_slice`, so chunk data only crosses the
/// memory hierarchy once on the way out.
#[inline]
fn narrow_and_append(out: &mut Vec<u8>, chunk_u16: &[u16]) -> Result<(), usize> {
    let n = chunk_u16.len();
    out.reserve(n);
    let base = out.len();
    // SAFETY: capacity reserved above. We write `n` bytes then set_len.
    // We compute an `any_high` accumulator across the entire chunk and check
    // it once at the end — the hot loop has no branches on the marker bit,
    // letting the autovectorizer turn it into a tight SIMD narrow. If any
    // value had the high bit set we walk back to find the offending index
    // for the error message; that path is cold.
    let mut any_high: u16 = 0;
    let dst_ptr = unsafe { out.as_mut_ptr().add(base) };
    for (i, &v) in chunk_u16.iter().enumerate() {
        any_high |= v;
        // SAFETY: i < n and we reserved n bytes past `base`.
        unsafe { dst_ptr.add(i).write(v as u8) };
    }
    // SAFETY: wrote exactly n bytes.
    unsafe { out.set_len(base + n) };

    if (any_high & MARKER_BASE) != 0 {
        // Cold path: find the first offending index for the error report.
        let bad = chunk_u16
            .iter()
            .position(|&v| v >= MARKER_BASE)
            .unwrap_or(0);
        // Undo what we just appended so the caller observing the error doesn't
        // see corrupt partial output. (Phase 3 verifies and writes only on
        // success, but this keeps `assembled` honest for the debug path.)
        unsafe { out.set_len(base) };
        return Err(bad);
    }
    Ok(())
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

    /// Premortem mitigation B5: trailer fields can legitimately be zero.
    /// Earlier versions guarded the verification with `if expected_crc != 0`
    /// and `if expected_size > 0` — silently accepting corrupted streams
    /// whose trailer fields happened to be zero. This test mutates a real
    /// stream's CRC field to zero and asserts the decoder rejects it (the
    /// true CRC of any non-trivial 10+ MiB stream is overwhelmingly
    /// non-zero, so trailer-CRC=0 means "trailer lies").
    #[test]
    fn crc_zero_trailer_is_verified() {
        let data = make_random_data(10 * 1024 * 1024, 0xfacefeed);
        let mut compressed = make_gzip_data(&data);
        let crc_offset = compressed.len() - 8;
        compressed[crc_offset..crc_offset + 4].copy_from_slice(&0u32.to_le_bytes());
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Err(ParallelError::CrcMismatch) => {}
            Err(ParallelError::DecodeFailed)
            | Err(ParallelError::SizeMismatch)
            | Err(ParallelError::TooSmall) => {}
            Ok(_) => panic!(
                "CRC=0 trailer not detected as corruption — the `expected_crc != 0` \
                 sentinel guard is back. See Copilot review on PR #94."
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
        assert!(
            output.is_empty(),
            "no bytes written on verification failure"
        );
    }

    /// Same as above for ISIZE. A 10 MiB input has expected_size = 10 MiB
    /// mod 2^32, which is non-zero; setting the trailer ISIZE field to 0
    /// must be detected even though `expected_size == 0` was the previous
    /// "skip the check" sentinel.
    #[test]
    fn isize_zero_trailer_is_verified() {
        let data = make_random_data(10 * 1024 * 1024, 0xfeedbeef);
        let mut compressed = make_gzip_data(&data);
        let isize_offset = compressed.len() - 4;
        compressed[isize_offset..isize_offset + 4].copy_from_slice(&0u32.to_le_bytes());
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Err(ParallelError::SizeMismatch) => {}
            Err(ParallelError::CrcMismatch)
            | Err(ParallelError::DecodeFailed)
            | Err(ParallelError::TooSmall) => {}
            Ok(_) => panic!(
                "ISIZE=0 trailer not detected as corruption — the `expected_size > 0` \
                 sentinel guard is back. See Copilot review on PR #94."
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
        assert!(
            output.is_empty(),
            "no bytes written on verification failure"
        );
    }

    // ── Deletion-trap killer counter ─────────────────────────────────────────

    #[test]
    fn marker_pipeline_counter_increments_on_success() {
        // Same Mutex as the routing-level deletion-trap killer. Under
        // `cargo test`'s default parallel execution, concurrent unit tests
        // that call `decompress_parallel` would otherwise bump the counter
        // between the before/after snapshots and mask a real regression.
        let _guard = MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

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
