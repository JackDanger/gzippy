//! Parallel single-member gzip decompression — rapidgzip-shaped port.
//!
//! Production path on x86_64/arm64 with `pure-rust-inflate` when the classifier
//! returns [`crate::decompress::DecodePath::ParallelSM`] (parallel SM
//! enabled and compressed size > `MIN_PARALLEL_COMPRESSED`). Routing lives in
//! [`crate::decompress::classify_gzip`]; this module never makes its
//! own routing decisions — every error variant is terminal.
//!
//! This module is a thin driver. It parses the gzip header and trailer,
//! delegates to [`crate::decompress::parallel::chunk_fetcher::drive`]
//! for parallel decode, and verifies the CRC32 + ISIZE.
//!
//! The architecture (worker pool, prefetch loop, shared WindowMap, fast
//! and slow decode paths, async re-dispatch on speculative mismatch)
//! lives in [`crate::decompress::parallel::chunk_fetcher`].
//!
//! Streaming-write trade-off: bytes flow to the writer as each chunk
//! resolves, so a CRC/ISIZE mismatch at the end leaves partial output
//! behind. The routing layer treats CRC failures as terminal corruption.
//! There is **no fallback**: if `decompress_parallel` returns Err the
//! caller surfaces it; the silent libdeflate retry that used to follow
//! has been removed.

use std::io::{self, Write};
use std::sync::atomic::AtomicU64;
#[cfg(parallel_sm)]
use std::sync::atomic::Ordering;
use std::sync::Mutex;

// (Removed 2026-06-04, task #8) `MIN_PARALLEL_SIZE`: was a 4 MiB floor below
// which a C-FFI one-shot decoded small inputs. The ParallelSM pipeline is now
// the SOLE single-member path at any size (verified byte-exact for tiny /
// incompressible / stored at T1+T4), so there is no floor and no one-shot FFI
// fallback. (That pipeline is pure-Rust on gzippy-native; on gzippy-isal its
// clean tail decodes via ISA-L FFI — see chunk_decode.rs `finish_decode_chunk_impl`.)
// 1 (was 2, 2026-05-31): the parallel-SM engine is the production path at EVERY
// thread count (MIN_PARALLEL_SM_THREADS=0, user directive). At num_threads=1 the
// pool has one worker and the consumer runs on the calling thread (2 OS threads,
// no worker==consumer deadlock), so the engine runs single-threaded rather than
// erroring "input below parallel SM minimum (routing bug)". This is what lets us
// measure the engine we are optimizing at T=1 instead of a libdeflate confound.
const MIN_THREADS_FOR_PARALLEL: usize = 1;
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
const TARGET_COMPRESSED_CHUNK_BYTES: usize = 4 * 1024 * 1024;
/// T1-only default compressed-chunk target (1 MiB vs the 4 MiB T>1 default).
///
/// At T1 the pipeline is inline (no workers, no prefetch) and the only thing
/// chunk size changes is the size of each chunk's materialized output buffer.
/// A SMALLER per-chunk output buffer stays under the allocator's mmap
/// threshold, so when the in-order-drain recycles it (depth-1) the next chunk
/// decodes into the SAME warm, already-faulted pages instead of a fresh mmap →
/// fewer page faults and a warmer working set.
/// SCOPED TO T1: at T>1 the finer granularity REGRESSES the parallel pipeline
/// because more/smaller chunks add block-finder + scheduling overhead, so T>1
/// keeps the 4 MiB default.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
const T1_TARGET_COMPRESSED_CHUNK_BYTES: usize = 1024 * 1024;
/// T1 OUTPUT-RESIDENT chunk sizing target (decoded bytes per chunk).
///
/// The T1 win documented on `T1_TARGET_COMPRESSED_CHUNK_BYTES` is purely a
/// function of the per-chunk *output* (decoded) buffer staying warm/cache- and
/// fault-friendly when the depth-1 in-order drain recycles it. A *fixed
/// compressed* stride (1 MiB) makes that output buffer scale with the corpus
/// expansion ratio: a high-ratio corpus (monorepo 5.2×, nasa 9.9×) gets a
/// 5–10 MiB per-chunk output buffer (cold, fault-heavy), while a low-ratio one
/// (silesia 3.1×) sits near the sweet spot. So instead we target a fixed
/// *decoded* size and derive the compressed stride from the gzip-trailer ISIZE
/// expansion ratio, keeping the output buffer near this size across corpora.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
const T1_OUTPUT_RESIDENT_TARGET_BYTES: usize = 2_560 * 1024;
/// Lower clamp on the ISIZE-derived T1 compressed stride: a pathological ratio
/// (all-zeros corpus, or an ISIZE field wrapped on a >4 GiB member) must not
/// pick a degenerate sub-256-KiB stride that thrashes the per-chunk loop. Upper
/// clamp is `T1_TARGET_COMPRESSED_CHUNK_BYTES` (the legacy 1 MiB default), so
/// low-ratio / stored corpora keep exactly the prior behaviour.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
const T1_MIN_COMPRESSED_CHUNK_BYTES: usize = 256 * 1024;
/// Floor on the adjusted chunk size when the file is small.
/// Mirror of `512_Ki` literal at vendor's ParallelGzipReader.hpp:305.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
const MIN_ADJUSTED_CHUNK_BYTES: usize = 512 * 1024;

/// Literal port of vendor's small-file chunk-size adjustment at
/// `ParallelGzipReader.hpp:294-306`:
///
/// ```cpp
/// if (fileSize && (m_chunkSizeInBytes * 2U * parallelization > *fileSize)) {
///     m_chunkSizeInBytes =
///         std::max(512_Ki,
///                  ceilDiv(ceilDiv(*fileSize, 3U * parallelization), 512_Ki) * 512_Ki);
/// }
/// ```
///
/// Without this, gzippy decompresses small-to-medium files with the
/// static 4 MiB chunk size, capping effective parallelism at
/// `fileSize / 4 MiB` chunks. On the 221 MB / 10.7 MB-compressed
/// fixture: 4 MiB chunks → 4 chunks → ~4-way parallelism on a 16-core
/// machine. Vendor uses 21 chunks (`--verbose` stats: "Total Fetched:
/// 21") and runs at 4.79 CPUs utilized. After this adjustment, gzippy
/// has the same effective chunk count.
///
/// Formula: when default `chunk_size * 2 * num_threads > file_size`,
/// shrink the chunk size to spread the work across `~3 *
/// num_threads` chunks (vendor's "give the thread pool more time to
/// be filled out" — chosen empirically per the comment), with a
/// 512 KiB floor (block-finder overhead would dominate below that).
/// MID-RATIO COARSENING BAND (AMD/Zen2, T>8) — the low edge. A stream whose
/// ISIZE/deflate ratio is in `[LOW, HIGH)` skips the finer `adjusted_chunk_size_amd`
/// (`file/T²`, 512 KiB floor) formula and falls through to the VENDOR (rapidgzip)
/// partition instead. The finer AMD formula over-chunks a LARGE MID-ratio stream
/// (e.g. `weights.safetensors`, ~83 MiB deflate, ratio 1.09), and its per-chunk
/// CONSUMER marker-resolution + apply-window cost then sets the wall; the vendor
/// formula's coarser grid (matching rapidgzip's) avoids that. The band is bracketed
/// structurally: BELOW it a PURE-STORED stream (incompressible, NO window markers)
/// prefers the finer path, and ABOVE it a COMPRESSIBLE stream's slow Huffman decode
/// repays finer chunks for load balance. The separating mechanism is window-marker
/// density × decode-cost: mid-ratio has markers but fast decode, so consumer
/// overhead dominates when over-chunked. AMD/Zen2 + T>8 ONLY (the sole regime that
/// takes the finer formula); every other arch/T is byte-identically routed.
const MID_RATIO_COARSE_LOW: f64 = 1.05;
/// MID-RATIO COARSENING BAND high edge — see [`MID_RATIO_COARSE_LOW`]. Set below
/// the lowest compressible corpus that PREFERS finer chunks (squishy 2.30,
/// silesia 3.11) and above `weights` (1.09), so only the mid-ratio band
/// coarsens.
const MID_RATIO_COARSE_HIGH: f64 = 2.0;

#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
pub(crate) fn adjusted_chunk_size_bytes(
    file_size: usize,
    num_threads: usize,
    default_chunk_size: usize,
    isize_deflate_ratio: f64,
) -> usize {
    adjusted_chunk_size_bytes_with(
        file_size,
        num_threads,
        default_chunk_size,
        isize_deflate_ratio,
        cpu_is_amd(),
        physical_core_count,
    )
}

/// Host-parameterized core of [`adjusted_chunk_size_bytes`]. The CPU vendor and
/// the physical-core-count probe are PARAMETERS (`is_amd`, `physical_cores`) so
/// the selector is a pure function of its inputs: tests pin explicit values and
/// assert deterministic, host-independent outputs (a `cpu_is_amd()` +
/// `physical_core_count()`-dependent assertion is a hardware-fingerprint flake —
/// it fired only on AMD CI runners). Production calls it through the thin wrapper
/// above with the real vendor + topology; `physical_cores` stays a lazily-invoked
/// fn pointer so the one-time `/sys` topology scan still only happens on the
/// large-incompressible branch that needs it (see
/// [`INCOMPRESSIBLE_CAP_MIN_FILE_BYTES`]). Behavior is byte-identical to the
/// pre-refactor code.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
fn adjusted_chunk_size_bytes_with(
    file_size: usize,
    num_threads: usize,
    default_chunk_size: usize,
    isize_deflate_ratio: f64,
    is_amd: bool,
    physical_cores: fn() -> usize,
) -> usize {
    let threads = num_threads.max(1);
    // AMD/Zen2 VERY-HIGH-THREAD finer-chunk dispatch: Zen's wide CCX pool
    // wins with MORE, smaller chunks once the worker count is high — finer
    // granularity flips T16 from a loss to a win. Restricted to threads > 8:
    //   • T2/T4 REGRESS badly with finer chunks,
    //   • T8 is a TIE (the finer effect flips sign across runs), so it stays on
    //     the coarse baseline,
    // and ALL of T1–T8 keep the coarse vendor spacing. It is ALSO restricted to AMD
    // because on Intel ANY chunk <= 1 MiB triggers a 3–4× wall BLOWUP (silesia
    // T8/T16) while the vendor default (>= 1.5 MiB) wins
    // every T — so Intel / other x86 / aarch64 keep the vendor formula unchanged
    // (byte-identical wall). Dispatch is by `cpu_is_amd()` (cached cpuid vendor
    // string) and only moves chunk BOUNDARIES — decoded output is byte-identical.
    if is_amd && threads > 8 {
        // MID-RATIO COARSENING: a large mid-ratio stream (ratio ∈ [LOW, HIGH))
        // over-chunks under the finer `file/T²` formula and its per-chunk consumer
        // marker-resolution cost sets the wall. Skip the finer formula and fall
        // through to the VENDOR (rapidgzip) partition instead. See
        // [`MID_RATIO_COARSE_LOW`] for the mechanism.
        // Byte-transparent (only chunk boundaries move); output CRC32/ISIZE-verified.
        let in_coarse_band =
            (MID_RATIO_COARSE_LOW..MID_RATIO_COARSE_HIGH).contains(&isize_deflate_ratio);
        if !in_coarse_band {
            // INCOMPRESSIBLE HIGH-THREAD CHUNK-COUNT CAP (ratio < LOW; T > physical
            // cores). At T > physical the `file/T²` refinement over-chunks a LARGE
            // incompressible stream into very many tiny chunks and loses to
            // rapidgzip's T-independent uniform grid, while at T = physical the same
            // `file/threads²` grid WINS. Capping the refinement divisor at the
            // physical core count makes T > physical reuse the WINNING T = physical
            // grid; every requested worker still spawns, only the chunk COUNT stops
            // growing past T = physical. threads <= physical is byte-IDENTICAL
            // (refine == threads). The COMPRESSIBLE arm (ratio >= HIGH) is UNTOUCHED
            // — its real decode-time variance repays finer chunks at every T, so it
            // keeps the full `threads` divisor. Byte-transparent (only moves chunk
            // boundaries; CRC32 + ISIZE still verified by the caller).
            // Only probe the physical-core count (and apply the cap) once the file
            // is large enough that the cap can actually change the chunk size:
            // below `file/physical² > floor` the `file/threads²` result is ALREADY
            // at the 512 KiB floor, so `threads.min(physical)` yields the identical
            // floored size — a no-op. Gating on file size keeps small incompressible
            // streams (movie.mp4, data.sqlite, tool.bin) byte-AND-cost-identical to
            // base: they never call `physical_core_count()` (whose one-time
            // `/sys` topology scan is a per-process tax on a tiny decode,
            // negligible on the 500 MiB+ files the cap targets). 32 MiB is a safe
            // lower bound (`floor · physical²` for physical≥8).
            let refine = incompressible_refine_divisor(
                isize_deflate_ratio,
                file_size,
                threads,
                physical_cores,
            );
            return adjusted_chunk_size_amd(file_size, refine, default_chunk_size);
        }
    }
    let vendor = adjusted_chunk_size_vendor(file_size, threads, default_chunk_size);
    // AMD/Zen2 MID-THREAD (2..=8): cap the chunk size by the per-T schedule so the
    // per-chunk decode-TIME variance spreads across more workers as T grows (see
    // `amd_midt_chunk_cap`). threads>8 already refines via the AMD branch above;
    // threads==1 is the serial T1 path. `min` preserves the vendor small-file
    // shrink (never coarsens); only shrinks large-file chunks.
    if is_amd && (2..=8).contains(&threads) {
        return vendor.min(amd_midt_chunk_cap(threads));
    }
    vendor
}

/// Vendor small-file chunk adjustment (Intel / other x86 / aarch64, and AMD at
/// `threads <= 8`). Literal port of `ParallelGzipReader.hpp:294-306` — see the
/// rationale block above `adjusted_chunk_size_bytes`.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
fn adjusted_chunk_size_vendor(
    file_size: usize,
    threads: usize,
    default_chunk_size: usize,
) -> usize {
    if default_chunk_size.saturating_mul(2).saturating_mul(threads) <= file_size {
        return default_chunk_size;
    }
    let denom = 3 * threads;
    let inner = file_size.div_ceil(denom);
    let aligned = inner.div_ceil(MIN_ADJUSTED_CHUNK_BYTES) * MIN_ADJUSTED_CHUNK_BYTES;
    aligned.max(MIN_ADJUSTED_CHUNK_BYTES)
}

/// AMD/Zen2 high-thread finer-chunk sizing: spread the work across ~`threads²`
/// chunks (chunks-per-worker grows with the worker count), refining the chunk size
/// DOWN from the caller's coarse `default_chunk_size` toward `file_size / threads²`,
/// floored at [`MIN_ADJUSTED_CHUNK_BYTES`]. On the 68 MB silesia fixture this yields
/// ≈4 MiB @ T4 (capped at the default), ≈1 MiB @ T8, and the 512 KiB floor @ T16.
/// `min`-then-`max` (not `clamp`) so a sub-floor
/// `default_chunk_size` can never panic; the floor always wins. Byte-transparent —
/// only moves chunk boundaries, not decoded output.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
fn adjusted_chunk_size_amd(file_size: usize, threads: usize, default_chunk_size: usize) -> usize {
    let target = file_size / threads.saturating_mul(threads).max(1);
    target.min(default_chunk_size).max(MIN_ADJUSTED_CHUNK_BYTES)
}

/// Minimum compressed file size for the incompressible SMT-oversubscription cap to
/// engage (see [`adjusted_chunk_size_bytes`]). Below this the finer `file/threads²`
/// size is already at the [`MIN_ADJUSTED_CHUNK_BYTES`] floor, so capping the divisor
/// at the physical-core count is a no-op — gating on file size avoids the one-time
/// `physical_core_count()` `/sys` probe on small decodes (a per-process tax that is
/// measurable on a ~12 MiB file, negligible on the 500 MiB+ files the cap targets).
/// 32 MiB is a conservative lower bound (`floor · physical²` for physical ≥ 8).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
const INCOMPRESSIBLE_CAP_MIN_FILE_BYTES: usize = 32 * 1024 * 1024;

/// Cold, out-of-line divisor selector for the incompressible SMT-oversubscription
/// cap (see [`adjusted_chunk_size_bytes`]). For a large incompressible stream at
/// more threads than physical cores, caps the finer `file/threads²` refinement
/// divisor at the physical-core count (reusing the winning T=physical grid);
/// otherwise returns `threads` unchanged (byte-identical grid). Isolating the added
/// const+topology-probe logic behind a `#[cold] #[inline(never)]` boundary keeps it
/// out of the hot chunk-grid path's codegen unit so it cannot perturb the
/// surrounding decode code's compiled layout. Byte-transparent.
#[cfg_attr(not(parallel_sm), allow(dead_code))]
#[cold]
#[inline(never)]
fn incompressible_refine_divisor(
    isize_deflate_ratio: f64,
    file_size: usize,
    threads: usize,
    physical_cores: fn() -> usize,
) -> usize {
    if isize_deflate_ratio < MID_RATIO_COARSE_LOW && file_size > INCOMPRESSIBLE_CAP_MIN_FILE_BYTES {
        threads.min(physical_cores())
    } else {
        threads
    }
}

/// Physical (not logical / SMT) core count of the host, cached. Caps the
/// incompressible-stream finer-chunk refinement divisor in
/// [`incompressible_refine_divisor`]: SMT-sibling workers past the physical cores
/// add no real parallel decode to balance a cheap-decode incompressible stream's
/// (absent) straggler variance, so refining `file/threads²` for them only
/// multiplies the in-order consumer's per-chunk round-trips. Falls back to 1 if topology is
/// unavailable. Cached in a `OnceLock` (topology is invariant for the process).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
#[cold]
#[inline(never)]
fn physical_core_count() -> usize {
    use std::sync::OnceLock;
    static PHYS: OnceLock<usize> = OnceLock::new();
    *PHYS.get_or_init(|| num_cpus::get_physical().max(1))
}

/// AMD/Zen2 MID-THREAD (2..=8) per-T chunk-size CAP (byte-transparent).
///
/// MECHANISM (not divisibility). The consumer streams chunks IN ORDER
/// while a `pool_size`-worker pull-queue decodes ahead. The makespan is set by a
/// STRAGGLER worker, and the straggler is a SLOW CHUNK, not an "extra" chunk from
/// a count that fails to divide `T`: two workers with the SAME chunk load can
/// differ several-fold in decode time (per-chunk decode-time variance).
/// Rounding the chunk COUNT to a multiple of `T` does NOT remove the tail and
/// regresses the wall (it detunes the chunk size). The lever that WORKS is the
/// chunk SIZE: as `T` grows, smaller chunks spread the decode-time variance across
/// more workers so no single straggler dominates the tail — while staying coarse
/// enough to dodge the low-worker throughput / per-chunk-overhead penalty.
///
/// The schedule is the per-T optimum on silesia/Zen2: T3/T4 want the 4 MiB
/// default (finer REGRESSES them), T5/T6 want 3 MiB, T7/T8 want 2.5 MiB.
/// Note the T3/T4 PLATEAU: the optimum is not a smooth per-worker formula
/// (a `file/(T·k)` form would shrink T4 below 4 MiB and regress it), so the
/// cap is keyed on `T`. Callers `min` this with the vendor size, so the small-file
/// vendor shrink is preserved and chunks are only ever made FINER, never coarser.
/// Only moves chunk boundaries — decoded output is byte-identical (CRC32 + ISIZE
/// still verified by the caller). Scoped to AMD threads 2..=8; Intel/aarch64 and
/// AMD threads>8 are untouched (a <1 MiB chunk 3–4×'s Intel; threads>8 already
/// refines via `adjusted_chunk_size_amd`).
fn amd_midt_chunk_cap(threads: usize) -> usize {
    match threads {
        ..=4 => TARGET_COMPRESSED_CHUNK_BYTES, // 4 MiB (T3/T4 plateau; T2 default)
        5..=6 => 3 * 1024 * 1024,              // 3 MiB
        _ => 2560 * 1024,                      // 2.5 MiB (T7/T8)
    }
}

/// T1 OUTPUT-RESIDENT compressed stride: derive the per-chunk *compressed*
/// stride so the per-chunk *decoded* output buffer is ≈[`T1_OUTPUT_RESIDENT_TARGET_BYTES`],
/// using the gzip-trailer ISIZE expansion ratio (decoded / compressed). Keeps
/// the per-chunk output buffer warm/fault-friendly across corpus ratios instead
/// of letting a fixed compressed stride blow it up on high-ratio corpora. See
/// [`T1_OUTPUT_RESIDENT_TARGET_BYTES`] for the rationale.
///
/// Falls back to the legacy [`T1_TARGET_COMPRESSED_CHUNK_BYTES`] when ISIZE is
/// implausible (ISIZE < compressed ⇒ near-incompressible / stored / wrapped
/// field), and clamps the result to
/// `[T1_MIN_COMPRESSED_CHUNK_BYTES, T1_TARGET_COMPRESSED_CHUNK_BYTES]` so the
/// stride can never go below 256 KiB or above the prior 1 MiB default — output
/// is byte-identical for any stride (the thin-T1 driver just lands chunk
/// boundaries elsewhere; CRC32 + ISIZE are still verified by the caller).
#[cfg_attr(not(parallel_sm), allow(dead_code))] // used by the parallel-SM T1 path
pub(crate) fn t1_output_resident_chunk(gzip_data: &[u8], deflate_data_len: usize) -> usize {
    let n = gzip_data.len();
    if n < 4 || deflate_data_len == 0 {
        return T1_TARGET_COMPRESSED_CHUNK_BYTES;
    }
    // ISIZE = original (decoded) size modulo 2^32 — gzip trailer, last 4 bytes LE.
    let isize_field = u32::from_le_bytes([
        gzip_data[n - 4],
        gzip_data[n - 3],
        gzip_data[n - 2],
        gzip_data[n - 1],
    ]) as u64;
    let deflate_len = deflate_data_len as u64;
    if isize_field < deflate_len {
        // Near-incompressible / stored / wrapped ISIZE: keep the safe default.
        return T1_TARGET_COMPRESSED_CHUNK_BYTES;
    }
    // stride_compressed = target_decoded × compressed / decoded  (one division,
    // no double truncation). u64 math: target(2.5 MiB) × deflate_len(≤4 GiB) fits.
    let stride = (T1_OUTPUT_RESIDENT_TARGET_BYTES as u64).saturating_mul(deflate_len) / isize_field;
    (stride as usize).clamp(
        T1_MIN_COMPRESSED_CHUNK_BYTES,
        T1_TARGET_COMPRESSED_CHUNK_BYTES,
    )
}

/// Default crossover-margin multiplier for the serial-clean cost-model selector
/// (see [`effective_parallel_threads`]). The predicted parallel work-inflation
/// W ≈ ISIZE/deflate ratio; parallel only repays at `T >= ceil(W * margin)`.
/// `margin = 1.0` reproduces the per-corpus crossovers (silesia ratio
/// 2.75 → T3, monorepo → T6, storedheavy → T7-8). Frozen (the env override
/// was removed); `margin = 0`
/// still disables the selector (legacy always-parallel-below-ratio_max
/// behaviour) when passed explicitly to [`effective_parallel_threads_with`].
/// (aarch64 disables the selector entirely — see [`arch_crossover_margin_default`] —
/// so this Intel-default constant is unconsumed there.)
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
const PARALLEL_CROSSOVER_MARGIN_DEFAULT: f64 = 1.0;

/// AMD/Zen crossover-margin default. Zen2's per-chunk parallel overhead is higher
/// than Raptor Lake, so the marginal-parallelism crossover sits one notch higher;
/// `margin = 1.6` (crossover `ceil(ratio·1.6)`) is the value that erases the
/// Zen2 monorepo-T6/T8 default-constant regression. Selected
/// at runtime by [`cpu_is_amd`]. Frozen (the env override was removed).
/// (aarch64 disables the selector entirely — see [`arch_crossover_margin_default`] —
/// so this Zen-default constant is unconsumed there.)
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
const PARALLEL_CROSSOVER_MARGIN_AMD: f64 = 1.6;

/// Output-size threshold (bytes) at/above which the cost-model crossover is
/// lowered by one notch (see [`effective_parallel_threads`]). The parallel
/// pipeline carries a roughly FIXED per-decode overhead (thread spawn, pipeline
/// setup, the serial Amdahl tail). On a LARGE output that fixed cost is a small
/// fraction of the serial work (B/S → 0), so the real crossover sits ~1 thread
/// BELOW the ISIZE-ratio proxy; on a SMALL output the same fixed cost is a large
/// fraction (B/S large) and pushes the crossover ABOVE the proxy. The pure
/// ISIZE-ratio model can't separate these (silesia ratio 3.1 and monorepo ratio
/// 5.18 both round to a too-high crossover for silesia while monorepo genuinely
/// needs the high one). For example:
///   silesia  212 MiB out, ratio 3.1  → parallel first beats serial at T3 (proxy
///                                       said 4) → 1-notch bonus routes T3 parallel
///   squishy  480 MiB out, ratio 2.76 → first beats serial at T2 (proxy said 3)
///   monorepo  51 MiB out, ratio 5.18 → first beats serial at T7 (proxy said 6);
///                                       BELOW the threshold ⇒ NO bonus (correct)
/// Conservative by construction: the bonus only fires for clearly-large outputs
/// where the parallel arm at `crossover-1` still beats single-thread igzip by a
/// wide margin, so
/// it can never manufacture an igzip regression. Frozen (the env override
/// was removed; `0` still disables the bonus when
/// passed explicitly to [`effective_parallel_threads_with`]).
pub(crate) const PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT: u64 = 128 * 1024 * 1024;

/// Number of crossover notches the large-output bonus subtracts (default/Intel).
/// One notch (silesia crossover 4→3, squishy 3→2).
/// Frozen (the env override was removed).
const PARALLEL_LARGE_OUTPUT_NOTCH_DEFAULT: u64 = 1;

/// AMD/Zen large-output bonus depth: TWO notches. The Zen2 margin (1.6) is needed
/// for SMALL outputs (monorepo, ratio 5.18 → crossover 9 → parallel never beats
/// serial through T8); but for LARGE outputs the same 1.6 over-inflates the
/// crossover (silesia ratio 3.1 → ceil(3.1·1.6)=5; squishy 2.76 → 4), pushing the
/// marginal-parallelism knee one notch too high. A single notch leaves silesia-T3
/// SERIAL (regresses that cell); a SECOND notch lands silesia at crossover 3
/// (T3 parallel) and squishy at 2 (T2 parallel). Margin
/// 1.6 + 2-notch erases the monorepo-T6/T8 + squishy-T2 regressions WITHOUT
/// regressing silesia-T3 / squishy-T3. Selected at runtime by [`cpu_is_amd`].
/// Frozen (the env override was removed).
const PARALLEL_LARGE_OUTPUT_NOTCH_AMD: u64 = 2;

/// SMALL-OUTPUT SERIAL FLOOR: decoded-output size below which the parallel
/// single-member pipeline is capped to one thread (the fast inline T1 path).
/// This is the ALL-MARKER-BLOWUP guard (arch-independent — fires on every arch).
///
/// MECHANISM: a small, compressible single-member stream
/// with LOW deflate-block-boundary density — e.g. `markup.xml` (7.6 MiB XML, a
/// handful of huge dynamic blocks) — has no findable block start inside a chunk,
/// so EVERY speculatively-decoded chunk stays window-absent all-marker. The
/// parallel path then runs much slower than the serial inline path — a large LOSS
/// to rapidgzip at T≥6, WORSE than gz's own T1, while T1/T4 (routed serial by the
/// crossover) WIN.
/// The ISIZE ratio does NOT separate this input (markup 3.71 ≈ silesia 3.10); the
/// separating signal is OUTPUT SIZE. Below a few MiB the parallel pipeline's fixed
/// overhead is unamortizable AND the all-marker re-decode dominates, while the
/// serial inline path already BEATS rapidgzip on every squishy item at T1 (the
/// smallest squishy item where parallel reliably helps is ≥14 MiB). 8 MiB sits
/// above markup (7.60 MiB) and below the next squishy item (aozora 11.4 MiB), so
/// only markup-and-smaller (all TIE-or-WIN at serial) route serial — no genuine
/// parallel win is forfeited. A COMPRESSIBILITY gate (ratio ≥ 1.5) excludes
/// near-incompressible small files (photo.jpg, ratio 1.005): they are
/// stored-block-dominated, carry no marker pathology, and DO parallelize, so they
/// stay parallel.
/// Byte-transparent: only the effective thread count changes; CRC32 + ISIZE are
/// still verified by the caller. Frozen (the env override was removed;
/// `0` still disables when passed explicitly to
/// [`effective_parallel_threads_with`]).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
const PARALLEL_MIN_OUTPUT_BYTES_DEFAULT: u64 = 8 * 1024 * 1024;

/// SMALL-COMPRESSED (few-chunk) SERIAL FLOOR: compressed (deflate) size below
/// which a HIGHLY-COMPRESSIBLE (ratio ≥ 2.5) stream is capped to one thread,
/// regardless of decoded-output size. This is the companion to
/// [`PARALLEL_MIN_OUTPUT_BYTES_DEFAULT`] (which keys on OUTPUT size) and closes the
/// Intel-hybrid high-ratio-MEDIUM-file regression.
///
/// The cap keys on COMPRESSED size because — for this corpus — compressed size is
/// the signal that separates the two caught loss cells (access.log 2.66 MiB /
/// 25 MiB out, data.json 1.58 MiB / 14 MiB out), which run several-fold slower
/// parallel than serial on the Intel hybrid, from the must-stay-parallel
/// neighbours at equal OUTPUT (data.csv 3.45 MiB, ecoli 4.48 MiB, both ~25 MiB
/// out). The OUTPUT-size floor (`PARALLEL_MIN_OUTPUT_BYTES_DEFAULT`) does NOT catch
/// these (25/14 MiB ≫ 8 MiB), which is why a second, compressed-size guard exists.
/// The cap is UNIVERSAL (no arch dispatch): serial is independently
/// competitive-or-better for both caught cells on AMD too, so the same threshold
/// improves AMD while fixing Intel.
///
/// FRONTIER: the nearest CATCH is access.log (compressed 2.53–2.66 MiB, serial
/// WINS both arches at every T≥8) and the nearest EXCLUDE is data.csv (compressed
/// 3.33–3.45 MiB, parallel WIN / serial-neutral); the floor sits at their midpoint
/// with ~12% margin each side. Files just above it MUST stay parallel —
/// serializing them is a LOSS on ≥1 arch: ecoli (4.4 MiB), aozora (4.0 MiB),
/// dickens (4.5 MiB), monorepo (9.8 MiB), data.sqlite (12.9 MiB), tool.bin
/// (20.9 MiB), nasa (37 MiB). All are ≥ 3.3 MiB compressed and correctly excluded.
/// Byte-transparent (only the thread count changes; CRC32 + ISIZE still verified
/// by the caller). Bounded to MEDIUM output (< the large-output threshold) so a
/// genuinely huge output with a tiny compressed size (extreme-ratio stored/RLE) is
/// not force-serialized. `0` disables when passed explicitly to
/// [`effective_parallel_threads_with`].
#[cfg_attr(not(parallel_sm), allow(dead_code))]
const PARALLEL_MEDIUM_COMPRESSED_FLOOR_DEFAULT: u64 = 2_900 * 1024;

/// Runtime CPU-vendor detection for the arch-dispatched selector constants.
/// AMD (Zen) needs a higher crossover margin (for small outputs) AND a deeper
/// large-output bonus (2 notches), both for Zen2 — see
/// [`PARALLEL_CROSSOVER_MARGIN_AMD`] / [`PARALLEL_LARGE_OUTPUT_NOTCH_AMD`]. Intel
/// and every other arch keep the Raptor-Lake `margin = 1.0` + 1-notch bonus
/// (arm64 decode CI-green with those). The dispatch is by VENDOR (`AuthenticAMD`),
/// the deterministic signal; it only changes the parallel THREAD-COUNT routing
/// (byte-identical output). `cpuid` leaf 0 is read once and cached.
#[cfg(target_arch = "x86_64")]
pub(crate) fn cpu_is_amd() -> bool {
    use std::sync::OnceLock;
    static IS_AMD: OnceLock<bool> = OnceLock::new();
    *IS_AMD.get_or_init(|| {
        // SAFETY: cpuid leaf 0 (vendor string) is available on every x86_64 CPU.
        // `#[allow(unused_unsafe)]`: `__cpuid` is `unsafe fn` on some toolchains
        // and a safe fn on others (stdarch has moved cpuid-class intrinsics in
        // and out of `unsafe` across Rust versions); keep the block for the
        // toolchains that still require it and silence the resulting
        // `unused_unsafe` lint on toolchains (like this one) where it doesn't.
        #[allow(unused_unsafe)]
        let v = unsafe { std::arch::x86_64::__cpuid(0) };
        // Vendor string lives in EBX, EDX, ECX (in that order) — 12 bytes.
        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&v.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&v.edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&v.ecx.to_le_bytes());
        &vendor == b"AuthenticAMD"
    })
}

/// Non-x86_64 arches (aarch64) keep the Intel/Raptor-Lake defaults — the selector
/// is arch-independent and arm64 decode is CI-green with `margin = 1.0` + bonus on.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn cpu_is_amd() -> bool {
    false
}

/// Arch-dispatched default crossover margin.
///
/// aarch64 (Apple Silicon): the serial-clean cost-model selector + large-output
/// bonus are x86/Zen2-tuned — their crossover constants were tuned for Intel
/// Raptor-Lake (margin 1.0 + 1-notch) and AMD Zen2 (margin 1.6 + 2-notch). On
/// arm64 they MISFIRE: silesia (ratio ~3.1) → crossover ceil(3.1)=4, minus the
/// 1-notch large-output bonus = 3, which routes silesia T2 to the SERIAL floor
/// where the prestack always-parallel-below-ratio_max routing ran parallel — a
/// REGRESSION. Return margin 0.0 to DISABLE the cost-model
/// selector on aarch64, restoring the prestack routing (whose high-ratio
/// serialization is the aarch64-only [`AARCH64_PRESTACK_RATIO_MAX`] cap in
/// [`effective_parallel_threads`] — removing that cap regresses M1 on
/// logs/software-class at T2). x86_64 codegen is byte-identical
/// (this branch compiles out off-aarch64).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
fn arch_crossover_margin_default() -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        0.0
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if cpu_is_amd() {
            PARALLEL_CROSSOVER_MARGIN_AMD
        } else {
            PARALLEL_CROSSOVER_MARGIN_DEFAULT
        }
    }
}

/// Arch-dispatched large-output bonus depth in crossover notches. AMD
/// subtracts 2; Intel/other subtract 1.
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub(crate) fn arch_large_output_notch_default() -> u64 {
    if cpu_is_amd() {
        PARALLEL_LARGE_OUTPUT_NOTCH_AMD
    } else {
        PARALLEL_LARGE_OUTPUT_NOTCH_DEFAULT
    }
}

/// AMD MID-BAND graduated-amortization LOW bound (byte-transparent). Outputs in
/// `[this, PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT)` — i.e. 48–128 MiB — earn the
/// same crossover reduction the >128 MiB tier grants, but FLOORED at
/// [`PARALLEL_MID_BAND_MIN_CROSSOVER`] (never split below that T) and never
/// RAISED above the un-amortized crossover.
///
/// WHY: the >128 MiB large-output tier
/// was the sole fixed-overhead amortization credit, so a mid-size output like
/// `tool.bin` (59.6 MiB, ratio 2.99) got the full un-amortized crossover
/// `ceil(2.99·1.6)=5` and routed SERIAL at T2/T3/T4 while its own
/// per-chunk kernel BEATS rapidgzip when it DOES split. The step
/// "splits at T8 not T4" was exactly `SERIAL_CLEAN_FLOOR_APPLIED` firing for
/// T<5. A 59.6 MiB output amortizes the pipeline's fixed cost as well as
/// silesia does; the 128 MiB tier boundary was over-conservative for the
/// 48–128 MiB band. The FLOOR (min crossover 4) is why
/// storedmix (100 MiB, ratio 1.98, base crossover 4) is BYTE-IDENTICALLY routed
/// — it keeps its serial T2 (forcing storedmix-T2 parallel REGRESSES it),
/// and `min(base, …)` keeps low-ratio
/// weights (ratio 1.09, crossover 2) unchanged. AMD/Zen2 only (Intel margin 1.0
/// already lands tool.bin at crossover 3; aarch64 disables the selector).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
const PARALLEL_MID_BAND_LOW_AMD: u64 = 48 * 1024 * 1024;

/// Floor for the mid-band graduated reduction: a mid-band file never routes
/// parallel below T=4 (the first thread count at which mid-size parallelism
/// repays; below it the per-chunk pipeline overhead is
/// unamortized — the storedmix-T2 regression). See [`PARALLEL_MID_BAND_LOW_AMD`].
const PARALLEL_MID_BAND_MIN_CROSSOVER: f64 = 4.0;

/// Arch-dispatched mid-band low bound: AMD/Zen2 enables the 48–128 MiB
/// graduated-amortization tier; every other arch disables it (0 ⇒ the
/// `mid_band_low > 0` guard in [`effective_parallel_threads_with`] is never
/// taken, so routing is byte-identical off-AMD).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub(crate) fn arch_mid_band_low_default() -> u64 {
    #[cfg(not(target_arch = "aarch64"))]
    {
        if cpu_is_amd() {
            return PARALLEL_MID_BAND_LOW_AMD;
        }
    }
    0
}

/// Counter proving the serial-clean cost-model selector fired on a real decode
/// (deletion-trap discipline; read by the routing test).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub static SERIAL_CLEAN_FLOOR_APPLIED: AtomicU64 = AtomicU64::new(0);

/// aarch64-only prestack ISIZE-ratio cap (see the block comment in
/// [`effective_parallel_threads`]): with the cost-model selector disabled on
/// aarch64 (margin 0.0), this cap is the arch's ONLY high-ratio serialization,
/// and removing it regresses Apple M1 (logs-class and software-class at T2).
/// x86_64 deletes the cap
/// entirely (the T-aware selector owns high-ratio routing there). Knob-free: no env override.
#[cfg(target_arch = "aarch64")]
const AARCH64_PRESTACK_RATIO_MAX: u64 = 8;

/// Counter proving the aarch64 prestack cap fired on a real decode
/// (deletion-trap discipline).
#[cfg(target_arch = "aarch64")]
pub static AARCH64_PRESTACK_CAP_APPLIED: AtomicU64 = AtomicU64::new(0);

/// Counter proving the small-output serial floor (the all-marker-blowup guard,
/// see [`PARALLEL_MIN_OUTPUT_BYTES_DEFAULT`]) fired on a real decode
/// (deletion-trap discipline; read by the routing test).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub static SMALL_OUTPUT_SERIAL_FLOOR_APPLIED: AtomicU64 = AtomicU64::new(0);

/// Production entry point: [`effective_parallel_threads_with`] fed the
/// selector defaults (with every parallel-selector env knob removed). The former
/// work-per-thread cap (over-threading guard) and
/// its `GZIPPY_MIN_BYTES_PER_THREAD` override were REMOVED (2026-07-13): no
/// nonzero floor is a clean no-regress win across arches
/// (the movie.mp4 high-T regression it was built for is gone at HEAD; the only
/// residual over-threading — access.log — already beats rapidgzip), so per the
/// no-env-vars-in-prod-path rule the cap was deleted rather than baked to a
/// nonzero default.
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub(crate) fn effective_parallel_threads(
    gzip_data: &[u8],
    deflate_data_len: usize,
    num_threads: usize,
) -> usize {
    effective_parallel_threads_with(
        gzip_data,
        deflate_data_len,
        num_threads,
        PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
        arch_crossover_margin_default(),
        PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
        arch_large_output_notch_default(),
        arch_mid_band_low_default(),
        PARALLEL_MEDIUM_COMPRESSED_FLOOR_DEFAULT,
    )
}

/// Content-derived effective thread count for the parallel single-member
/// pipeline. Two guards, in order (each byte-transparent — only the thread
/// count changes; CRC32 + ISIZE are still verified by the caller):
///   1. SMALL-OUTPUT SERIAL FLOOR — small compressible streams (markup.xml)
///      degenerate to an all-marker re-decode; below the floor there is no real
///      parallel win to forfeit. See [`PARALLEL_MIN_OUTPUT_BYTES_DEFAULT`].
///   2. SERIAL-CLEAN COST-MODEL SELECTOR — the T-aware crossover
///      `T >= ceil(ratio × margin)`: below it the parallel pipeline's
///      OUTPUT-proportional overhead (marker resolution + apply_window +
///      per-chunk scaffold ≈ ratio× more total work than serial-clean) is not
///      repaid, so route serial; at/above it, parallel. High-ratio streams get
///      proportionally high crossovers (silesia 3.1 → T3, nasa 9.9 → T10-14,
///      software 29.8 → ≥30 i.e. serial at every practical T), which is what
///      protects the extreme-ratio corpora now that the former T-blind hard
///      ratio cap (`ratio >= 8 → serial at EVERY T`) is deleted — that cap
///      pre-empted this selector and forfeited high-T wins.
/// (A former 3rd guard, the WORK-PER-THREAD CAP over-threading guard, was
/// removed 2026-07-13 — it was frozen disabled and had no clean job left; see
/// [`effective_parallel_threads`].)
/// ISIZE wrap (>4 GiB output) or near-incompressible (isize<deflate) yields a
/// low ratio → low/no crossover → keep the requested threads (parallel is
/// correct there).
///
/// Pure and parameterized (no env reads, no arch dispatch) — the production
/// values live in [`effective_parallel_threads`]; tests call this directly
/// with explicit parameter values so selector-decision coverage stays
/// deterministic across hosts without process-global env mutation.
#[cfg_attr(not(parallel_sm), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn effective_parallel_threads_with(
    gzip_data: &[u8],
    deflate_data_len: usize,
    num_threads: usize,
    min_output: u64,
    margin: f64,
    large_output_bytes: u64,
    notch: u64,
    mid_band_low: u64,
    medium_compressed_floor: u64,
) -> usize {
    if num_threads <= 1 || deflate_data_len == 0 || gzip_data.len() < 4 {
        return num_threads;
    }

    // SMALL-OUTPUT SERIAL FLOOR (x86_64) — the all-marker-blowup guard. A small,
    // compressible, low-block-boundary-density stream (markup.xml) degenerates into
    // an all-marker re-decode under the parallel pipeline (a large LOSS to rapidgzip
    // at T≥6) while the serial inline path BEATS rapidgzip; below the floor there is
    // no real parallel win to forfeit. Cap to serial. See
    // [`PARALLEL_MIN_OUTPUT_BYTES_DEFAULT`] for the mechanism.
    // Byte-transparent (only the thread count changes). Guarded by a COMPRESSIBILITY
    // gate (ratio >= 2.5) so LOW-ratio small files (photo.jpg ratio 1.005; dovi.tar
    // ratio 2.46) — which parallelize to a WIN — stay parallel, AND by
    // `deflate < floor` (a true >4 GiB member whose ISIZE wrapped to a small value
    // still has a huge COMPRESSED size and correctly stays parallel).
    //
    // WHY A RATIO GATE AT ALL: the guard is the downstream AMORTIZATION SELECTOR
    // (below, "SERIAL-CLEAN COST-MODEL SELECTOR"). The parallel pipeline does
    // W ∝ (ISIZE/deflate ratio) more marker-resolution + apply_window work than the
    // serial-clean driver, so it only repays at T >= crossover = ceil(ratio · margin)
    // — a relation monotonic in ratio. At these SMALL output sizes the selector's
    // fixed-overhead proxy under-amortizes, so a small compressible file is faster
    // SERIAL up to a higher T than the proxy predicts; this ratio gate is the
    // small-output pre-filter that keeps such files serial.
    //
    // The gate is 2.5: the [1.5, 2.5) band stays parallel (dovi.tar ratio 2.46 WINS
    // parallel), while at/above 2.5 the file is faster serial at these sizes (markup
    // 3.71 regresses when forced parallel).
    //
    // ARCH: the small-output serial demotion applies on every arch — AMD/Zen2 (routed
    // parallel at T≥6 by the crossover) AND aarch64/Apple-M1 (selector disabled ⇒
    // routed parallel at T≥2), so the floor caps both.
    {
        let n = gzip_data.len();
        let isize_field = u32::from_le_bytes([
            gzip_data[n - 4],
            gzip_data[n - 3],
            gzip_data[n - 2],
            gzip_data[n - 1],
        ]) as u64;
        let deflate_len = deflate_data_len as u64;
        // COMPRESSIBILITY GATE (ratio >= 2.5, i.e. 2·isize >= 5·deflate). Justified by
        // the downstream AMORTIZATION SELECTOR (crossover = ceil(ratio · margin),
        // W ∝ ratio) — see the block comment above. A LOW-ratio small file
        // (photo.jpg 1.005 stored-dominated; dovi.tar 2.46) PARALLELIZES to a WIN, so
        // it must stay parallel; HIGH-ratio small files (markup 3.71, source tars
        // 3.9-4.1, minjs 3.31, …) are faster serial at these sizes and stay demoted.
        // The [1.5, 2.5) band stays parallel while the ≥2.5 band is demoted. The
        // 2·isize >= 5·deflate form uses integer math (isize <= 4 GiB ⇒ ×2 fits u64).
        // The lower `isize >= deflate` bound is subsumed (ratio 2.5 > 1.0), which also
        // rejects a wrapped >4 GiB ISIZE (isize < deflate).
        // COMPRESSIBILITY GATE — shared by both serial-floor guards below.
        let compressible = isize_field.saturating_mul(2) >= deflate_len.saturating_mul(5);
        // GUARD A — SMALL-OUTPUT serial floor (markup.xml-shaped): the decoded output
        // itself is below `min_output`, so the parallel pipeline's fixed overhead is
        // unamortizable AND the all-marker re-decode dominates. (Unchanged behaviour.)
        let small_output = min_output > 0 && isize_field < min_output && deflate_len < min_output;
        // GUARD B — SMALL-COMPRESSED serial floor: a highly-compressible MEDIUM-output
        // stream whose COMPRESSED size is below the empirical Intel-envelope threshold
        // (access.log / data.json). Output size does NOT separate these from the
        // must-stay-parallel files (access.log/data.csv/ecoli all ~25 MiB out);
        // COMPRESSED size is the signal that does. Bounded to medium output
        // (< `large_output_bytes`) so an extreme-ratio HUGE output is not serialized.
        // See [`PARALLEL_MEDIUM_COMPRESSED_FLOOR_DEFAULT`] for the frontier.
        // UNIVERSAL (serial is competitive-or-better on AMD too), byte-transparent
        // (only the thread count changes).
        let small_compressed = medium_compressed_floor > 0
            && deflate_len < medium_compressed_floor
            && (large_output_bytes == 0 || isize_field < large_output_bytes);
        if compressible && (small_output || small_compressed) {
            SMALL_OUTPUT_SERIAL_FLOOR_APPLIED.fetch_add(1, Ordering::Relaxed);
            return 1;
        }
    }

    // ISIZE/deflate ratio (used by the aarch64 prestack cap and the cost-model
    // selector + large-output bonus below). ISIZE is the gzip trailer's
    // little-endian uncompressed size (mod 2^32); `deflate_len` is the
    // compressed member length.
    let n = gzip_data.len();
    let isize_field = u32::from_le_bytes([
        gzip_data[n - 4],
        gzip_data[n - 3],
        gzip_data[n - 2],
        gzip_data[n - 1],
    ]) as u64;
    let deflate_len = deflate_data_len as u64;

    // NOTE (2026-07-05): the former T-BLIND hard ratio cap that lived here
    // (`if isize >= 8*deflate { return 1 }` at EVERY T, env-controlled)
    // is DELETED on x86_64. It pre-empted the
    // T-aware cost-model selector below on every high-ratio stream, forfeiting
    // the parallel wins the selector grants at high T (nasa/bignasa T16 recover
    // large parallel wins with the cap lifted).
    // The selector's `crossover = ceil(ratio × margin)` scales the serial floor
    // with the SAME ratio the cap used, so extreme ratios (logs_i32 11.35 →
    // crossover 17, logs_r3 19.9 → 30) still route serial at every practical T.
    // A content-based replacement (marker-density / block-boundary probe) does
    // not separate these streams: higher-ratio corpora have DENSER dynamic-block
    // boundaries than silesia, and the window-absent marker fraction is ~1.0 for
    // ALL corpora (back-refs only reach 32 KiB), so the ISIZE ratio via the
    // selector is the correct — and only — separating signal.
    //
    // aarch64 KEEPS the prestack cap: its cost-model selector is DISABLED
    // (margin 0.0, see [`arch_crossover_margin_default`]) so nothing else would
    // serialize high-ratio streams there, and the bare deletion REGRESSES Apple
    // M1 (logs-class and software-class slower at T2/T8). The cap IS
    // aarch64's prestack routing — byte-transparent, knob-free (no env
    // override; the old ratio-cap env knob is deleted per the
    // no-env-vars-in-prod-path rule). x86_64 codegen compiles this out.
    #[cfg(target_arch = "aarch64")]
    if isize_field >= AARCH64_PRESTACK_RATIO_MAX.saturating_mul(deflate_len) {
        AARCH64_PRESTACK_CAP_APPLIED.fetch_add(1, Ordering::Relaxed);
        return 1;
    }

    // SERIAL-CLEAN COST-MODEL SELECTOR (the low-T monotonicity fix).
    //
    // The parallel pipeline is NOT free: it does
    // W ≈ (ISIZE/deflate ratio)× more total CPU work than the serial-clean
    // thin-T1 driver, because marker-RESOLUTION + apply_window + per-chunk
    // SCAFFOLD are OUTPUT-proportional and under-amortized at low T (parallel
    // does ~2–5× more total work than serial-clean; the decode kernel itself is
    // K≈1.0).
    // Predicted parallel speedup ≈ T / W, so parallel only repays once
    //     T >= ceil(W * margin)         (W taken as the ISIZE/deflate ratio).
    // `ceil(ratio)` reproduces the per-corpus crossovers (silesia
    // ratio 2.75 → first wins at T3; monorepo → T6; storedheavy → T7-8; nasa /
    // low-ratio → wins at every T≥2), and is conservative by construction
    // (rounds UP → stays serial one notch longer). Below the crossover we route
    // to the serial-clean thin-T1 driver (return 1 → the `parallelization<=1`
    // path in `read_parallel_sm`), which returns to ~1.02-1.07× of gz-T1 — and
    // gz-T1 already BEATS single-thread igzip, so the floor is a WIN, not a tie.
    //
    // The asymmetry licenses the conservatism: misrouting a cell to PARALLEL
    // below its crossover REGRESSES it below T1 (loses to single-thread igzip);
    // misrouting to SERIAL at worst ties gz-T1 (still a win). `margin` is the
    // tunable safety knob (param, arch-dispatched by the production
    // caller); `0` disables this selector entirely.
    if margin > 0.0 {
        // ratio as f64 (deflate_len > 0 guaranteed above); crossover = ceil(ratio*margin).
        let ratio = isize_field as f64 / deflate_len as f64;
        let mut crossover = (ratio * margin).ceil();
        // LARGE-OUTPUT BONUS: a large output amortizes the parallel pipeline's
        // FIXED overhead (B/S → 0), so its real crossover sits BELOW the
        // ISIZE-ratio proxy. Lower the crossover by `notch` for outputs at/above
        // the threshold (kept ≥ 1). The small-output case (where fixed overhead is
        // a large fraction → the proxy under-estimates the crossover) is left
        // as-is, which is exactly what the design wants (monorepo keeps its
        // higher crossover). The notch DEPTH is arch-dispatched: 1 on Raptor Lake,
        // 2 on Zen2 (the Zen2 margin 1.6 over-inflates large-output crossovers; the
        // 2nd notch lands silesia/squishy back at the marginal-parallelism knee).
        // See [`PARALLEL_LARGE_OUTPUT_NOTCH_DEFAULT`] / [`PARALLEL_LARGE_OUTPUT_NOTCH_AMD`].
        if large_output_bytes > 0 && isize_field >= large_output_bytes && notch > 0 {
            crossover = (crossover - notch as f64).max(1.0);
        } else if mid_band_low > 0
            && isize_field >= mid_band_low
            && large_output_bytes > 0
            && isize_field < large_output_bytes
            && notch > 0
        {
            // MID-BAND GRADUATED AMORTIZATION (48–128 MiB, AMD/Zen2). Grant the
            // same `notch` reduction the >128 MiB tier grants, but (a) FLOORED at
            // `PARALLEL_MID_BAND_MIN_CROSSOVER` so a mid-size file never splits
            // below T=4 (protects the low-T over-threading regime that regressed
            // storedmix-T2), and (b) `min(crossover, …)` so it can only LOWER a
            // high crossover, never RAISE a low one (protects low-ratio mid files
            // like weights whose base crossover is already < the floor). This is
            // the fix for the tool-T4 partition step-function; see
            // [`PARALLEL_MID_BAND_LOW_AMD`] for the rationale.
            let reduced = (crossover - notch as f64).max(PARALLEL_MID_BAND_MIN_CROSSOVER);
            crossover = crossover.min(reduced);
        }
        if (num_threads as f64) < crossover {
            SERIAL_CLEAN_FLOOR_APPLIED.fetch_add(1, Ordering::Relaxed);
            return 1;
        }
    }

    // (A former WORK-PER-THREAD CAP over-threading guard lived here — capping
    // effective-T so each worker got a minimum of COMPRESSED bytes to amortize
    // per-chunk serial cost. It was frozen disabled and removed 2026-07-13: at
    // HEAD the movie.mp4 high-T regression it targeted is gone, and no nonzero
    // floor is a clean no-regress win across the corpus. See
    // [`effective_parallel_threads`].)
    num_threads
}

/// Counter incremented every time `adjusted_chunk_size_bytes` returns a
/// value strictly less than the default. Mirror of the
/// `PREFETCH_NEXT_FILESIZE_ACCEPT` / `UNSPLIT_BLOCKS_EMPLACED`
/// deletion-trap pattern — proves the adjustment branch is reached on
/// real production decodes.
#[cfg_attr(not(parallel_sm), allow(dead_code))] // incremented on the x86 SM path; routing traps read it under the same cfg
pub static ADJUSTED_CHUNK_SIZE_APPLIED: AtomicU64 = AtomicU64::new(0);

/// Successful runs of the parallel pipeline. Snapshot before/after a
/// decode to confirm production routing actually called us — see the
/// deletion-trap killer test in `src/tests/routing.rs`.
///
/// `pub(crate)` rather than `pub`: internal diagnostic surface, not a
/// library API.
#[allow(dead_code)] // incremented by the x86_64+isal-compression decompress_parallel path; read by tests
pub(crate) static MARKER_PIPELINE_RUNS: AtomicU64 = AtomicU64::new(0);

/// Deletion-trap counter for the FALSE-SINGLE late-detection re-entry:
/// a stream the classifier called single-member that actually
/// carries a trailing member (empty-first member, or first member compressed
/// past the 16 MiB detection window) errors at member 1's boundary and re-enters
/// the multi-member driver. A routing test asserts this FIRES on the
/// detection-evading fixtures and stays 0 across the single-member corpus.
#[allow(dead_code)] // incremented on the misroute path; read by tests
pub(crate) static MISROUTE_REENTRY_APPLIED: AtomicU64 = AtomicU64::new(0);

/// Mutex serializing routing tests that snapshot `MARKER_PIPELINE_RUNS`
/// against each other. Without this, `cargo test`'s default parallel
/// execution can mask a real silent-fallback regression with a false
/// positive.
#[allow(dead_code)] // wired by #[cfg(test)] consumers in src/tests/routing.rs + src/decompress/mod.rs
pub(crate) static MARKER_PIPELINE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
#[inline]
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

/// Parse the gzip header and return the byte offset where the deflate
/// stream starts. Thin wrapper over `gzip_format::read_header`
/// (literal port of `rapidgzip::gzip::readHeader`); drops the parsed
/// `Header` since the driver currently doesn't need it. Multi-stream
/// support reads subsequent headers via `gzip_format::read_header` too.
pub(crate) fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    crate::decompress::parallel::gzip_format::read_header(data).map(|(_h, off)| off)
}

/// Cheap, sound guard for the multi-member resume path: does a SECOND gzip
/// member actually begin right after the first member's deflate body + trailer?
///
/// Walks the first member's deflate stream to its byte-aligned `BFINAL` end
/// (pure-Rust, bounded memory), then checks whether `[member1_end + 8..]` starts
/// with a gzip magic. Returns `false` for a genuinely-corrupt single member (the
/// walk errors) so the caller surfaces the original decode error instead of
/// attempting a pointless resume. Only invoked on the error path (rare).
#[cfg(parallel_sm)]
fn trailing_member_after_first(gzip_data: &[u8]) -> bool {
    let header_size = match skip_gzip_header(gzip_data) {
        Ok(h) => h,
        Err(_) => return false,
    };
    if gzip_data.len() < header_size + 8 {
        return false;
    }
    let deflate_len =
        match crate::decompress::scan_inflate::deflate_stream_byte_len(&gzip_data[header_size..]) {
            Ok(n) => n,
            Err(_) => return false,
        };
    let next = header_size + deflate_len + 8;
    next + 2 <= gzip_data.len() && gzip_data[next] == 0x1f && gzip_data[next + 1] == 0x8b
}

pub fn decompress_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    num_threads: usize,
    verbose: bool,
) -> Result<u64, ParallelError> {
    let t0 = std::time::Instant::now();

    // Routing-eligibility gate — classifier upstream guarantees these
    // bounds; reaching this point with bad inputs is a routing bug
    // surfaced as a hard error. There is no silent retry.
    let header_size = skip_gzip_header(gzip_data).map_err(|_| ParallelError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let _deflate_data_len = gzip_data.len().saturating_sub(header_size + trailer_size);
    // No size floor (task #8: the ParallelSM pipeline is the sole single-member
    // path at any size — pure-Rust on gzippy-native, ISA-L clean tail on
    // gzippy-isal). Only num_threads is gated — T=0 is a caller bug.
    if num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::InvalidGzipFormat);
    }

    #[cfg(parallel_sm)]
    {
        use crate::decompress::parallel::sm_driver::{
            read_parallel_sm_capturing, read_parallel_sm_resume_multi, ReadParallelSmError,
        };

        // Production driver: `sm_driver::read_parallel_sm` → `chunk_fetcher::drive`.
        // `single_member::decompress_parallel` is now a thin classifier-
        // routed wrapper: it owns the routing-eligibility gate and the
        // `MARKER_PIPELINE_RUNS` counter; the trailer parsing + CRC /
        // ISIZE verification + chunk_fetcher::drive orchestration all
        // live in the new driver (mirror of vendor's
        // `ParallelGzipReader::read` at ParallelGzipReader.hpp:553-646).
        // Thread-aware default chunk target: T1 uses the smaller 1 MiB target
        // (warm output-buffer recycling; see
        // T1_TARGET_COMPRESSED_CHUNK_BYTES), T>1 keeps the 4 MiB
        // TARGET_COMPRESSED_CHUNK_BYTES target (finer granularity regresses the
        // parallel pipeline).
        // Compressibility thread-cap: on highly-compressible single-member
        // streams the parallel pipeline regresses below the inline T1 path
        // (output-proportional marker overhead > parallel decode benefit), so
        // cap to one thread there. Shadows num_threads so chunk sizing and the
        // driver both use the effective count. See `effective_parallel_threads`.
        let num_threads = effective_parallel_threads(gzip_data, _deflate_data_len, num_threads);
        if debug_enabled() {
            eprintln!("[parallel_sm] effective_threads={num_threads} (compressibility cap)");
        }
        let thread_default_chunk = if num_threads <= 1 {
            // T1 OUTPUT-RESIDENT stride: keep the per-chunk decoded buffer near a
            // fixed cache/fault-friendly size by deriving the compressed stride
            // from the ISIZE expansion ratio (see the helper doc).
            t1_output_resident_chunk(gzip_data, _deflate_data_len)
        } else {
            TARGET_COMPRESSED_CHUNK_BYTES
        };
        // ISIZE/deflate ratio (gzip trailer's LE uncompressed size mod 2^32 over
        // the compressed member length) — drives the mid-ratio coarsening band in
        // `adjusted_chunk_size_bytes` (see its doc). Guarded len above.
        let chunk_ratio = {
            let n = gzip_data.len();
            let isize_field = u32::from_le_bytes([
                gzip_data[n - 4],
                gzip_data[n - 3],
                gzip_data[n - 2],
                gzip_data[n - 1],
            ]) as f64;
            isize_field / (_deflate_data_len.max(1) as f64)
        };
        let chunk_size = adjusted_chunk_size_bytes(
            gzip_data.len(),
            num_threads,
            thread_default_chunk,
            chunk_ratio,
        );
        if chunk_size < TARGET_COMPRESSED_CHUNK_BYTES {
            ADJUSTED_CHUNK_SIZE_APPLIED.fetch_add(1, Ordering::Relaxed);
        }
        if debug_enabled() {
            // LOCATE instrumentation (byte-transparent, debug-gated): report
            // the partition decision so the "splits at T8 not T4" step-function is
            // deterministic. `chunks≈deflate/chunk_size` is the target grid; the
            // block finder validates boundaries near each stride.
            let n = gzip_data.len();
            let isize_field = u32::from_le_bytes([
                gzip_data[n - 4],
                gzip_data[n - 3],
                gzip_data[n - 2],
                gzip_data[n - 1],
            ]) as u64;
            let ratio = isize_field as f64 / (_deflate_data_len.max(1) as f64);
            let est_chunks = (_deflate_data_len as u64).div_ceil(chunk_size.max(1) as u64);
            eprintln!(
                "[parallel_sm] STEP0 deflate_len={} isize={} ratio={:.4} \
                 eff_threads={} chunk_size={} est_chunks={} \
                 serial_clean_floor={} small_out_floor={}",
                _deflate_data_len,
                isize_field,
                ratio,
                num_threads,
                chunk_size,
                est_chunks,
                SERIAL_CLEAN_FLOOR_APPLIED.load(Ordering::Relaxed),
                SMALL_OUTPUT_SERIAL_FLOOR_APPLIED.load(Ordering::Relaxed),
            );
        }

        // No pool pre-warm here. A prior experiment touched pool pages
        // on the consumer thread before workers spawn; it regressed the SM path
        // because every fresh
        // CLI process paid the pre-touch cost without amortization. The
        // page-fault gap vs vendor (40% gzippy vs 17% rapidgzip) needs a
        // real per-Vec allocator (allocator-api2 + rpmalloc-rs) or
        // daemon-mode CLI to close; not a pre-touch loop. See module
        // docs at `chunk_buffer_pool.rs:57-77`.
        // Decode as a single member, capturing the bytes streamed so far so the
        // multi-member resume can pick up past them on a boundary error.
        let mut bytes_written = 0usize;
        let sm_result = read_parallel_sm_capturing(
            gzip_data,
            writer,
            out_fd,
            num_threads,
            chunk_size,
            &mut bytes_written,
            verbose,
        );

        let result = match sm_result {
            Ok(r) => r,
            Err(e) => {
                if debug_enabled() {
                    eprintln!("[parallel_sm] driver error: {e}");
                }
                // A decode/size/CRC failure on a stream the classifier called
                // single-member is the multi-member-misroute signature: the
                // second member begins past the 16 MiB detection window, so the
                // single-stream finder cannot cross member 1's gzip footer.
                // Resume the remaining members (pure-Rust, per-member CRC+ISIZE
                // verified), skipping the validated prefix already streamed.
                // `InvalidHeader`/`InvalidFormat` are genuine malformation — not
                // resumable. (Truly corrupt single-member input also lands here;
                // the resume then finds no further valid member and surfaces the
                // original failure, never silently truncating.)
                let resumable = matches!(
                    e,
                    ReadParallelSmError::DecodeFailed(_)
                        | ReadParallelSmError::SizeMismatch { .. }
                        | ReadParallelSmError::CrcMismatch { .. }
                );
                if resumable && trailing_member_after_first(gzip_data) {
                    MISROUTE_REENTRY_APPLIED.fetch_add(1, Ordering::Relaxed);
                    if debug_enabled() {
                        eprintln!(
                            "[parallel_sm] single-member decode failed at multi-member \
                             boundary; resuming members past {bytes_written} streamed bytes"
                        );
                    }
                    read_parallel_sm_resume_multi(
                        gzip_data,
                        writer,
                        bytes_written,
                        num_threads,
                        chunk_size,
                        verbose,
                    )
                    .map_err(|me| {
                        ParallelError::DecodeFailed(format!("multi-member resume: {me}"))
                    })?
                } else {
                    return Err(match e {
                        ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
                        ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
                        ReadParallelSmError::DecodeFailed(detail) => {
                            ParallelError::DecodeFailed(detail)
                        }
                        ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
                        ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
                    });
                }
            }
        };

        MARKER_PIPELINE_RUNS.fetch_add(1, Ordering::Relaxed);
        if debug_enabled() {
            let total = t0.elapsed();
            let mbps = result.total_size as f64 / total.as_secs_f64() / 1e6;
            eprintln!(
                "[parallel_sm:v0.6] total={:.1}ms isize={} ({:.0} MB/s)",
                total.as_secs_f64() * 1000.0,
                result.total_size,
                mbps,
            );
        }
        Ok(result.total_size as u64)
    }
    #[cfg(not(parallel_sm))]
    {
        let _ = (writer, out_fd, t0, _deflate_data_len);
        Err(ParallelError::UnsupportedPlatform)
    }
}

/// Production entry for [`crate::decompress::DecodePath::MultiMemberChunked`].
/// Walks each member and inflates it with the full within-member parallel
/// engine, streaming output in member order (per-member CRC32 + ISIZE verified,
/// pure-Rust, no C-FFI). Routed for MIXED "GZ" ++ plain concatenations (see the
/// `DecodePath::MultiMemberChunked` docs for why it is not used for plain
/// dominant/few-member distributions).
#[cfg(parallel_sm)]
/// STAGE-2d whole-file MULTI-MEMBER GRID entry
/// ([`crate::decompress::DecodePath::MultiMemberGrid`]). Decodes the whole
/// multi-member stream as ONE chunk grid (not a member walk), streaming with the
/// zero-copy `out_fd` path when available and running per-member CRC32 + ISIZE
/// verification inside the consumer. See
/// [`crate::decompress::parallel::sm_driver::read_parallel_sm_grid`].
pub fn decompress_multi_member_grid<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    out_fd: Option<i32>,
    num_threads: usize,
    verbose: bool,
) -> Result<u64, ParallelError> {
    use crate::decompress::parallel::sm_driver::{read_parallel_sm_grid, ReadParallelSmError};

    if gzip_data.len() < 18 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let num_threads = num_threads.max(1);
    let chunk_size = adjusted_chunk_size_bytes(
        gzip_data.len(),
        num_threads,
        TARGET_COMPRESSED_CHUNK_BYTES,
        0.0,
    );
    let r = read_parallel_sm_grid(gzip_data, writer, out_fd, num_threads, chunk_size, verbose)
        .map_err(|e| match e {
            ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
            ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
            ReadParallelSmError::DecodeFailed(detail) => ParallelError::DecodeFailed(detail),
            ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
            ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
        })?;
    writer
        .flush()
        .map_err(|_| ParallelError::InvalidGzipFormat)?;
    Ok(r.total_size as u64)
}

pub fn decompress_multi_member_chunked<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
    verbose: bool,
) -> Result<u64, ParallelError> {
    use crate::decompress::parallel::sm_driver::{read_parallel_sm_multi, ReadParallelSmError};

    if gzip_data.len() < 18 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let num_threads = num_threads.max(1);
    let chunk_size = adjusted_chunk_size_bytes(
        gzip_data.len(),
        num_threads,
        TARGET_COMPRESSED_CHUNK_BYTES,
        0.0,
    );
    let r = read_parallel_sm_multi(gzip_data, writer, num_threads, chunk_size, verbose).map_err(
        |e| match e {
            ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
            ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
            ReadParallelSmError::DecodeFailed(detail) => ParallelError::DecodeFailed(detail),
            ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
            ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
        },
    )?;
    writer
        .flush()
        .map_err(|_| ParallelError::InvalidGzipFormat)?;
    Ok(r.total_size as u64)
}

/// A `Write` that forwards to `inner` and counts the bytes streamed. Used by the
/// T1 multi-member fast path to learn how many validated output bytes the grid
/// attempt streamed before an error, so the member-walk fallback can resume past
/// them (a `SkipWriter` drops exactly that prefix).
#[cfg(parallel_sm)]
struct CountWriter<'a, W: Write> {
    inner: &'a mut W,
    count: usize,
}

#[cfg(parallel_sm)]
impl<W: Write> Write for CountWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(parallel_sm)]
fn map_read_sm_err(
    e: crate::decompress::parallel::sm_driver::ReadParallelSmError,
) -> ParallelError {
    use crate::decompress::parallel::sm_driver::ReadParallelSmError;
    match e {
        ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
        ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
        ReadParallelSmError::DecodeFailed(detail) => ParallelError::DecodeFailed(detail),
        ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
        ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
    }
}

/// T1 fast path for [`crate::decompress::DecodePath::MultiMemberSeq`] (mmiso
/// 2026-07-06). Decode a plain multi-member gzip stream at parallelization=1
/// through the ParallelSM chunk kernel instead of the legacy scalar
/// `inflate_consume_first_bits` — recovering the located ~3× T1 deficit
/// (fewer insn/byte and lower RSS than the scalar path).
///
/// Strategy: try the WHOLE-FILE grid first (one inflate pass, member boundaries
/// discovered inline, per-member CRC32 + ISIZE verified in the consumer). The
/// grid's speculative block finder can fail on a sparse-boundary member or on
/// trailing garbage where a bounded per-member walk succeeds; on such a failure
/// we RESUME the sequential member-walk past the validated prefix already
/// streamed (a `SkipWriter` drops those bytes). This is the SAME correction
/// [`decompress_parallel`] applies to a misrouted single-member stream — NOT a
/// silent backend retry masking a failure: the member-walk re-verifies every
/// member's CRC32 + ISIZE, so a genuinely corrupt stream surfaces here as a
/// terminal error and trailing garbage clean-stops (gzip(1) parity). Respects
/// `-p1`: FASTER engine at effective-T=1, never a parallelized request.
#[cfg(parallel_sm)]
pub fn decompress_multi_member_seq_fast<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    verbose: bool,
) -> Result<u64, ParallelError> {
    use crate::decompress::parallel::sm_driver::{
        read_parallel_sm_grid, read_parallel_sm_resume_multi, ReadParallelSmError,
    };

    if gzip_data.len() < 18 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let chunk_size =
        adjusted_chunk_size_bytes(gzip_data.len(), 1, TARGET_COMPRESSED_CHUNK_BYTES, 0.0);

    let mut counter = CountWriter {
        inner: writer,
        count: 0,
    };
    let grid = read_parallel_sm_grid(gzip_data, &mut counter, None, 1, chunk_size, verbose);
    let streamed = counter.count;
    let inner = counter.inner;

    match grid {
        Ok(r) => {
            inner
                .flush()
                .map_err(|_| ParallelError::InvalidGzipFormat)?;
            Ok(r.total_size as u64)
        }
        // A malformed FIRST header / too-short stream is genuine malformation the
        // member-walk cannot recover — propagate it directly.
        Err(e @ (ReadParallelSmError::InvalidHeader | ReadParallelSmError::InvalidFormat)) => {
            Err(map_read_sm_err(e))
        }
        // Decode / boundary / trailer failure: resume the member-walk past the
        // validated prefix. If the stream is genuinely corrupt, the walk fails
        // again → terminal error (never silent truncation).
        Err(_) => {
            let r =
                read_parallel_sm_resume_multi(gzip_data, inner, streamed, 1, chunk_size, verbose)
                    .map_err(map_read_sm_err)?;
            inner
                .flush()
                .map_err(|_| ParallelError::InvalidGzipFormat)?;
            Ok(r.total_size as u64)
        }
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

/// Every variant is terminal — the classifier filters
/// parallel-eligibility upstream, so reaching this module with bad
/// inputs is a routing bug surfaced as a hard error rather than a
/// silent fallback opportunity.
#[derive(Debug)]
pub enum ParallelError {
    /// Bytes don't start with a valid gzip header (FHCRC/FNAME/etc.
    /// fields malformed or truncated).
    InvalidHeader,
    /// Header parses but the byte range available to the worker pool
    /// is too short for the parallel pipeline's invariants (e.g. the
    /// classifier sent a stream below `MIN_PARALLEL_SIZE`). Treat as
    /// a routing bug — the dispatcher must have classified this as
    /// `SingleMember`, not `ParallelSM`.
    InvalidGzipFormat,
    /// One or more chunk decodes failed inside the worker pool.
    /// Carries `sm_driver` / `chunk_fetcher` `Debug` detail (e.g.
    /// `Decode(InflateFailed(InvalidBlock))`).
    #[allow(dead_code)] // constructed on the x86+isal SM path only
    DecodeFailed(String),
    /// Output size doesn't match the gzip ISIZE trailer — corruption.
    #[allow(dead_code)] // constructed on the x86+isal SM path only
    SizeMismatch,
    /// CRC32 doesn't match the gzip CRC trailer — corruption.
    #[allow(dead_code)] // constructed on the x86+isal SM path only
    CrcMismatch,
    /// Build doesn't support the parallel pipeline on this platform
    /// (no x86_64 + ISA-L). The classifier never routes here on
    /// unsupported builds; this exists only as the cfg-stubbed body's
    /// guaranteed error path.
    #[allow(dead_code)] // non-SM-build cfg stub; constructed only off the x86+isal path
    UnsupportedPlatform,
    Io(io::Error),
}

impl From<io::Error> for ParallelError {
    fn from(e: io::Error) -> Self {
        ParallelError::Io(e)
    }
}

impl std::fmt::Display for ParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParallelError::InvalidHeader => write!(f, "invalid gzip header"),
            ParallelError::InvalidGzipFormat => {
                write!(f, "input below parallel SM minimum (routing bug)")
            }
            ParallelError::DecodeFailed(detail) => write!(f, "chunk decode failed: {detail}"),
            ParallelError::SizeMismatch => write!(f, "output size mismatch"),
            ParallelError::CrcMismatch => write!(f, "CRC32 mismatch"),
            ParallelError::UnsupportedPlatform => {
                write!(f, "parallel SM unsupported on this build")
            }
            ParallelError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// ISIZE/deflate ratio gate: highly-compressible single-member caps to one
    /// thread (the inline T1 path) so Tmax never regresses below T1; moderately
    /// compressible keeps the requested threads. Builds a minimal blob whose
    /// last 4 bytes are the gzip-trailer ISIZE the helper reads.
    fn blob_with_isize(isize_val: u32) -> Vec<u8> {
        let mut v = vec![0u8; 64];
        v[60..64].copy_from_slice(&isize_val.to_le_bytes());
        v
    }

    /// HIGH-RATIO ROUTING (2026-07-05):
    /// x86_64 — NO T-blind ratio cap: a high-ISIZE-ratio stream is serialized
    /// ONLY below its T-aware crossover (`ceil(ratio × margin)`), never at every
    /// T. The former hard cap (`ratio >= 8 → 1 thread at EVERY T`) forfeited
    /// high-T parallel wins and is DELETED there.
    /// aarch64 — the prestack cap REMAINS (selector disabled; removing the cap
    /// regresses M1 on logs/software-class at T2).
    /// Neutral params that isolate a SINGLE guard: floor disabled (`min_output=0`),
    /// large-output bonus disabled (`large_output_bytes=0`). Tests override
    /// only the dimension(s) they're exercising.
    const NO_FLOOR: u64 = 0;
    const NO_BONUS_BYTES: u64 = 0;

    /// HIGH-RATIO ROUTING (2026-07-05):
    /// x86_64 — NO T-blind ratio cap: a high-ISIZE-ratio stream is serialized
    /// ONLY below its T-aware crossover (`ceil(ratio × margin)`), never at every
    /// T. The former hard cap (`ratio >= 8 → 1 thread at EVERY T`) forfeited
    /// high-T parallel wins and is DELETED there.
    /// aarch64 — the prestack cap REMAINS (selector disabled; removing the cap
    /// regresses M1 on logs/software-class at T2).
    ///
    /// Calls [`effective_parallel_threads_with`] directly with an explicit
    /// margin=1.0 (vendor-independent — the production arch default is 1.6 on
    /// AMD) and the floor disabled, so the assertions below isolate the
    /// high-ratio crossover logic from both.
    #[test]
    fn high_ratio_routing_selector_owns_x86_prestack_cap_aarch64() {
        let deflate_len = 1_000_000usize;
        let high = blob_with_isize(20_000_000); // ratio 20
        let nasa_like = blob_with_isize(10_000_000); // ratio 10
        let low = blob_with_isize(3_000_000); // ratio 3
        let call = |gz: &[u8], t: usize| {
            effective_parallel_threads_with(
                gz,
                deflate_len,
                t,
                NO_FLOOR,
                1.0,
                PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
                arch_large_output_notch_default(),
                0, // mid_band_low disabled — this test isolates the base selector
                0, // medium_compressed_floor disabled — isolate other guards
            )
        };
        // Common to every arch: high-ratio serial at low T; low-ratio parallel;
        // T1 never altered.
        assert_eq!(call(&high, 8), 1);
        assert_eq!(call(&nasa_like, 8), 1);
        assert_eq!(call(&low, 8), 8);
        assert_eq!(call(&high, 1), 1);
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Selector owns: ratio 20 → crossover 20 → serial through T16 but
            // parallel at T20 (impossible under the old blind cap); nasa-shaped
            // ratio 10 → crossover 10 → parallel at T16 (the freed win cell).
            assert_eq!(call(&high, 16), 1);
            assert_eq!(call(&high, 20), 20);
            assert_eq!(call(&nasa_like, 16), 16);
        }
        #[cfg(target_arch = "aarch64")]
        {
            // Prestack cap: ratio >= 8 serial at EVERY T (selector is disabled
            // on aarch64; the cap is its only high-ratio serialization).
            let before = AARCH64_PRESTACK_CAP_APPLIED.load(Ordering::Relaxed);
            assert_eq!(call(&high, 16), 1);
            assert_eq!(call(&high, 20), 1);
            assert_eq!(call(&nasa_like, 16), 1);
            let after = AARCH64_PRESTACK_CAP_APPLIED.load(Ordering::Relaxed);
            assert!(after > before, "aarch64 prestack cap counter must fire");
            // Just under the cap (ratio 7.99) falls through (selector disabled
            // by default margin 0.0 → parallel; margin pinned 1.0 here → xover 8).
            let under = blob_with_isize(7_990_000);
            assert_eq!(call(&under, 8), 8);
        }
    }

    /// SMALL-OUTPUT SERIAL FLOOR (x86_64): a small compressible stream (markup.xml-
    /// shaped: 7.6 MiB out / 2.1 MiB deflate) caps to serial to dodge the all-marker
    /// blowup, while a stream at/above the 8 MiB floor keeps its threads. The floor
    /// must NOT fire on a near-incompressible / ISIZE-wrapped stream (isize < deflate)
    /// nor when the compressed size is large (a real >4 GiB member whose ISIZE wrapped).
    /// Arch-independent (the floor now fires on every arch — AMD/Zen2 + Apple-M1).
    #[test]
    fn small_output_serial_floor_caps_markup_shaped_only() {
        let call = |gz: &[u8], deflate_len: usize, t: usize, min_output: u64, margin: f64| {
            effective_parallel_threads_with(
                gz,
                deflate_len,
                t,
                min_output,
                margin,
                PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
                arch_large_output_notch_default(),
                0, // mid_band_low disabled — this test isolates the base selector
                0, // medium_compressed_floor disabled — isolate other guards
            )
        };
        // markup.xml-shaped: 7.6 MiB decoded, 2.1 MiB deflate → below the 8 MiB
        // floor, isize >= deflate → floor fires (serial) at every requested T.
        let markup = blob_with_isize(7_974_912);
        assert_eq!(
            call(
                &markup,
                2_147_400,
                6,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                1.0
            ),
            1
        );
        assert_eq!(
            call(
                &markup,
                2_147_400,
                8,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                1.0
            ),
            1
        );
        assert_eq!(
            call(
                &markup,
                2_147_400,
                16,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                1.0
            ),
            1
        );
        // T1 request is never altered.
        assert_eq!(
            call(
                &markup,
                2_147_400,
                1,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                1.0
            ),
            1
        );
        // aozora-shaped: 11.4 MiB decoded (>= 8 MiB floor) → floor does NOT fire;
        // margin 0 (crossover selector off) isolates the floor from the crossover.
        let aozora = blob_with_isize(12_003_648);
        assert_eq!(
            call(
                &aozora,
                3_995_900,
                8,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                0.0
            ),
            8
        );
        // Near-incompressible small stream (isize < deflate): floor must NOT fire.
        let incompressible = blob_with_isize(1_000_000);
        assert_eq!(
            call(
                &incompressible,
                2_000_000,
                8,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                0.0
            ),
            8
        );
        // photo.jpg-shaped: 6.5 MiB out / 6.48 MiB deflate → ratio 1.005 < 2.5
        // compressibility gate → floor must NOT fire (parallelizes fine as stored
        // blocks; the low ratio routes it below the amortization crossover).
        let photo = blob_with_isize(6_511_067);
        assert_eq!(
            call(&photo, 6_481_121, 8, PARALLEL_MIN_OUTPUT_BYTES_DEFAULT, 0.0),
            8
        );
        // dovi.tar-shaped: 5.888 MiB out / 2.391 MiB deflate → ratio 2.46 < 2.5
        // compressibility gate → floor must NOT fire. (The parallel pipeline WINS on
        // dovi, so demoting it to serial would forfeit a real parallel win; the gate
        // frees the low-ratio [1.5, 2.5) band while markup/tars, ratio 3.7-4.1, stay
        // demoted.)
        let dovi = blob_with_isize(5_888_000);
        assert_eq!(
            call(&dovi, 2_390_860, 8, PARALLEL_MIN_OUTPUT_BYTES_DEFAULT, 0.0),
            8
        );
        // Just below the 2.5 gate (2·isize < 5·deflate) → floor must NOT fire.
        let just_below = blob_with_isize(4_999_999);
        assert_eq!(
            call(
                &just_below,
                2_000_000,
                8,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                0.0
            ),
            8
        );
        // Right at the 2.5 gate (2·isize == 5·deflate) → fires (compressible enough).
        let at_gate = blob_with_isize(5_000_000);
        assert_eq!(
            call(
                &at_gate,
                2_000_000,
                8,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                0.0
            ),
            1
        );
        // min_output=0 disables the floor: markup-shaped now reaches the crossover.
        assert_eq!(call(&markup, 2_147_400, 8, NO_FLOOR, 0.0), 8);
    }

    /// SMALL-COMPRESSED (few-chunk) SERIAL FLOOR (Guard B): a highly-compressible
    /// stream whose COMPRESSED size is below the few-chunk floor is capped to serial
    /// even when its decoded OUTPUT is well above the small-output floor. This is the
    /// Intel-hybrid high-ratio-MEDIUM-file fix. The floor keys on COMPRESSED size
    /// (≈ chunk count), the only signal that separates access.log (catch) from
    /// data.csv / ecoli (must-stay-parallel, ~equal output). See
    /// [`PARALLEL_MEDIUM_COMPRESSED_FLOOR_DEFAULT`] for the frontier.
    #[test]
    fn medium_compressed_serial_floor_caps_high_ratio_small_compressed_only() {
        // margin 0.0 isolates Guard B from the cost-model crossover; min_output at the
        // 8 MiB production value keeps Guard A (small OUTPUT) OFF for these 14–26 MiB
        // outputs, so only the COMPRESSED-size guard can fire.
        let floor = PARALLEL_MEDIUM_COMPRESSED_FLOOR_DEFAULT;
        let call = |gz: &[u8], deflate_len: usize, t: usize, med_floor: u64| {
            effective_parallel_threads_with(
                gz,
                deflate_len,
                t,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                0.0,
                PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
                0,
                0,
                med_floor,
            )
        };
        // access.log-shaped: 25 MiB out (> 8 MiB small-output floor), 2.66 MiB deflate
        // (< 2.9 MiB few-chunk floor), ratio ~9.86 → Guard B fires: serial at every T.
        let access = blob_with_isize(26_214_398);
        assert_eq!(call(&access, 2_659_220, 16, floor), 1);
        assert_eq!(call(&access, 2_659_220, 24, floor), 1);
        assert_eq!(call(&access, 2_659_220, 32, floor), 1);
        // T1 request is never altered.
        assert_eq!(call(&access, 2_659_220, 1, floor), 1);
        // data.json-shaped: 13.6 MiB out, 1.58 MiB deflate, ratio ~8.97 → fires.
        let djson = blob_with_isize(14_215_394);
        assert_eq!(call(&djson, 1_584_494, 16, floor), 1);
        // data.csv-shaped: 25 MiB out, 3.45 MiB deflate (> 2.9 MiB), ratio ~7.68 →
        // MUST NOT fire (parallel WIN on both arches; nearest exclude in the frontier).
        let dcsv = blob_with_isize(26_500_039);
        assert_eq!(call(&dcsv, 3_450_111, 16, floor), 16);
        // ecoli-shaped: 25 MiB out, 4.48 MiB deflate, ratio ~5.85 → MUST NOT fire
        // (Intel-T16 serial LOSES — needs parallel).
        let ecoli = blob_with_isize(26_214_271);
        assert_eq!(call(&ecoli, 4_478_915, 16, floor), 16);
        // LOW-ratio (< 2.5) small-compressed stream: 2 MiB deflate, 4 MiB out (ratio 2)
        // → compressibility gate rejects → MUST NOT fire (both arches).
        let lowratio = blob_with_isize(4_000_000);
        assert_eq!(call(&lowratio, 2_000_000, 16, floor), 16);
        // The next two exclude cases are ratio ≥ 8, so on aarch64 the (independent)
        // prestack cap serializes them regardless of Guard B; assert Guard B's
        // non-firing only where it is observable (x86_64, cap deleted there).
        #[cfg(not(target_arch = "aarch64"))]
        {
            // nasa-shaped: 356 MiB out, 37 MiB deflate → deflate ≫ floor → MUST NOT fire.
            let nasa = blob_with_isize(373_056_138);
            assert_eq!(call(&nasa, 37_309_343, 16, floor), 16);
            // BOUND: a HUGE output (> 128 MiB large-output threshold) with a tiny
            // deflate (extreme ratio) is NOT force-serialized by Guard B — it is not a
            // "medium" file. 200 MiB out, 1 MiB deflate.
            let huge = blob_with_isize(200 * 1024 * 1024);
            assert_eq!(call(&huge, 1_000_000, 16, floor), 16);
        }
        // med_floor = 0 disables Guard B: access.log-shaped now keeps its threads on
        // x86_64 (margin 0 → cost selector off → parallel). On aarch64 the independent
        // prestack cap (ratio 9.86 ≥ 8) still serializes it, so Guard B's disabling is
        // not observable there.
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(call(&access, 2_659_220, 16, 0), 16);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(call(&access, 2_659_220, 16, 0), 1);
    }

    #[test]
    fn serial_clean_selector_crossover_and_margin() {
        let deflate_len = 1_000_000usize;
        let call = |gz: &[u8], t: usize, margin: f64| {
            effective_parallel_threads_with(
                gz,
                deflate_len,
                t,
                NO_FLOOR,
                margin,
                PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
                arch_large_output_notch_default(),
                0, // mid_band_low disabled — this test isolates the base selector
                0, // medium_compressed_floor disabled — isolate other guards
            )
        };

        // Margin 1.0, ratio 3× (< hard cap 8): crossover = ceil(3.0) = 3.
        let blob = blob_with_isize(3_000_000);
        // T below crossover → serial-clean floor (1 thread).
        assert_eq!(call(&blob, 2, 1.0), 1);
        // T at/above crossover → parallel keeps the requested threads.
        assert_eq!(call(&blob, 3, 1.0), 3);
        assert_eq!(call(&blob, 8, 1.0), 8);

        // Non-integer ratio rounds UP (conservative): ratio 2.75× → crossover 3.
        let blob275 = blob_with_isize(2_750_000);
        assert_eq!(call(&blob275, 2, 1.0), 1);
        assert_eq!(call(&blob275, 3, 1.0), 3);

        // margin 0 disables the selector → keep threads (legacy below-cap behaviour).
        assert_eq!(call(&blob, 2, 0.0), 2);

        // Larger margin pushes the crossover up (more conservative): margin 2.0,
        // ratio 3 → crossover 6 → T4 now serial, T6 parallel.
        assert_eq!(call(&blob, 4, 2.0), 1);
        assert_eq!(call(&blob, 6, 2.0), 6);
    }

    #[test]
    fn serial_clean_selector_large_output_bonus() {
        let call =
            |gz: &[u8], deflate_len: usize, t: usize, large_output_bytes: u64, notch: u64| {
                effective_parallel_threads_with(
                    gz,
                    deflate_len,
                    t,
                    NO_FLOOR,
                    1.0,
                    large_output_bytes,
                    notch,
                    0, // mid_band_low disabled — this test isolates the large-output bonus
                    0, // medium_compressed_floor disabled — isolate the large-output bonus
                )
            };

        // LARGE output (200 MiB), ratio 3.1 (< hard cap 8). Proxy crossover =
        // ceil(3.1) = 4, but the large-output bonus lowers it to 3 → T3 parallel,
        // T2 serial. (silesia-shaped: 212 MiB out, ratio 3.1 → first wins at T3.)
        let big_out = 200 * 1024 * 1024u32;
        let big_deflate = (big_out as f64 / 3.1) as usize; // ratio 3.1
        let big = blob_with_isize(big_out);
        assert_eq!(call(&big, big_deflate, 2, 134_217_728, 1), 1); // serial
        assert_eq!(call(&big, big_deflate, 3, 134_217_728, 1), 3); // parallel
        assert_eq!(call(&big, big_deflate, 4, 134_217_728, 1), 4);

        // SMALL output (50 MiB), same ratio 3.1: below the threshold → NO bonus →
        // crossover stays 4 → T3 still serial, T4 parallel. (monorepo-shaped: the
        // fixed overhead is a large fraction, so the proxy crossover is correct.)
        let small_out = 50 * 1024 * 1024u32;
        let small_deflate = (small_out as f64 / 3.1) as usize;
        let small = blob_with_isize(small_out);
        assert_eq!(call(&small, small_deflate, 3, 134_217_728, 1), 1); // serial
        assert_eq!(call(&small, small_deflate, 4, 134_217_728, 1), 4); // parallel

        // Bonus disabled (large_output_bytes=0) → large output behaves like the
        // proxy again.
        assert_eq!(call(&big, big_deflate, 3, NO_BONUS_BYTES, 1), 1); // serial (no bonus)

        // 2-notch bonus (the AMD depth): crossover 4 → 2 → T2 parallel.
        assert_eq!(call(&big, big_deflate, 2, 134_217_728, 2), 2); // parallel (4-2=2)
        assert_eq!(call(&big, big_deflate, 1, 134_217_728, 2), 1); // never alters T1
    }

    /// MID-BAND GRADUATED AMORTIZATION (48–128 MiB, AMD margin 1.6 + 2-notch,
    /// floor 4). The fix for the tool-T4 partition step-function: a 48–128
    /// MiB output earns the 2-notch reduction the >128 MiB tier grants, floored at
    /// crossover 4 (never split below T4) and never raised above the un-amortized
    /// proxy. Uses real corpus shapes.
    #[test]
    fn mid_band_graduated_amortization_tool_and_monorepo_only() {
        const LARGE: u64 = 128 * 1024 * 1024;
        let mid = PARALLEL_MID_BAND_LOW_AMD;
        // (gz, deflate_len, t, mid_band_low) with AMD production params margin 1.6,
        // large=128 MiB, notch=2.
        let call = |gz: &[u8], deflate_len: usize, t: usize, mid_low: u64| {
            effective_parallel_threads_with(gz, deflate_len, t, NO_FLOOR, 1.6, LARGE, 2, mid_low, 0)
        };

        // tool.bin-shaped: 59.6 MiB out, 20.87 MiB deflate, ratio 2.99 → base
        // crossover ceil(2.99·1.6)=5. Mid-band → max(5-2,4)=4: T3 SERIAL (was
        // already serial at base too, base 5), T4 PARALLEL (was serial at base 5).
        let tool = blob_with_isize(62_480_352);
        assert_eq!(call(&tool, 20_874_333, 3, mid), 1); // T3 serial
        assert_eq!(call(&tool, 20_874_333, 4, mid), 4); // T4 PARALLEL — the fix
        assert_eq!(call(&tool, 20_874_333, 2, mid), 1); // T2 stays serial
                                                        // Base (mid disabled): T4 still serial (crossover 5) — reproduces the bug.
        assert_eq!(call(&tool, 20_874_333, 4, 0), 1);

        // monorepo-shaped: 48.6 MiB out, 9.82 MiB deflate, ratio 5.18 → base
        // crossover 9. Mid-band → max(9-2,4)=7: T6 serial, T7 PARALLEL (split-win).
        let mono = blob_with_isize(50_915_328);
        assert_eq!(call(&mono, 9_819_846, 6, mid), 1); // T6 serial
        assert_eq!(call(&mono, 9_819_846, 7, mid), 7); // T7 PARALLEL — split-win
        assert_eq!(call(&mono, 9_819_846, 4, mid), 1); // T4 stays serial
        assert_eq!(call(&mono, 9_819_846, 7, 0), 1); // base: still serial at T7

        // storedmix-shaped: 100 MiB out, 52.9 MiB deflate, ratio 1.98 → base
        // crossover 4. Mid-band FLOOR keeps it at 4 (min(4,max(2,4))=4) → BYTE-
        // IDENTICAL routing: T2/T3 serial (regression-protected), T4 parallel.
        let smix = blob_with_isize(104_857_600);
        assert_eq!(call(&smix, 52_890_462, 2, mid), 1); // T2 serial (protected)
        assert_eq!(call(&smix, 52_890_462, 2, 0), 1); // == base
        assert_eq!(call(&smix, 52_890_462, 4, mid), 4);
        assert_eq!(call(&smix, 52_890_462, 4, 0), 4); // == base

        // weights-shaped: 86.7 MiB out, 83.1 MiB deflate, ratio 1.09 → base
        // crossover 2. `min(base, …)` never RAISES it: mid-band leaves it at 2 →
        // T2 parallel unchanged (low-ratio mid file must not be de-parallelized).
        let weights = blob_with_isize(90_868_376);
        assert_eq!(call(&weights, 83_099_652, 2, mid), 2);
        assert_eq!(call(&weights, 83_099_652, 2, 0), 2); // == base

        // Below the band (40 MiB): untouched by the mid-band tier.
        let below = blob_with_isize(40 * 1024 * 1024);
        let below_deflate = (40.0 * 1024.0 * 1024.0 / 2.99) as usize;
        assert_eq!(
            call(&below, below_deflate, 4, mid),
            call(&below, below_deflate, 4, 0)
        );
    }

    #[test]
    fn arch_dispatch_defaults_match_vendor() {
        // The arch-dispatched defaults must follow the detected CPU vendor: AMD
        // gets the Zen2 margin 1.6 + 2-notch bonus; Intel/other x86 keep the
        // Raptor-Lake margin 1.0 + 1-notch bonus. Detection is deterministic
        // (cpuid leaf 0), so this is a stable invariant on whatever host runs it.
        // aarch64 (Apple Silicon) DISABLES the selector (margin 0.0) — the x86/Zen2
        // crossover tune misfires on arm64 (silesia T2/T3 regression), so arm64
        // restores the prestack always-parallel-below-ratio_max routing.
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(arch_crossover_margin_default(), 0.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            if cpu_is_amd() {
                assert_eq!(
                    arch_crossover_margin_default(),
                    PARALLEL_CROSSOVER_MARGIN_AMD
                );
                assert_eq!(
                    arch_large_output_notch_default(),
                    PARALLEL_LARGE_OUTPUT_NOTCH_AMD
                );
                assert_eq!(arch_large_output_notch_default(), 2); // deeper bonus on Zen
            } else {
                assert_eq!(
                    arch_crossover_margin_default(),
                    PARALLEL_CROSSOVER_MARGIN_DEFAULT
                );
                assert_eq!(
                    arch_large_output_notch_default(),
                    PARALLEL_LARGE_OUTPUT_NOTCH_DEFAULT
                );
            }
        }
    }

    /// REGRESSION GUARD (aarch64): on Apple Silicon the x86/Zen2-tuned serial-clean
    /// selector + large-output bonus are DISABLED (arch_crossover_margin_default = 0),
    /// so a silesia-shaped large output (ISIZE ~212 MiB, ratio ~3.1) routes T2/T3/T4
    /// to the requested parallel thread count — matching the prestack
    /// always-parallel-below-ratio_max routing. On x86 the selector would collapse
    /// T2 (and below) to the serial floor; that x86 tune misfired on arm64 (a
    /// silesia T2/T3 regression). No env override here:
    /// this asserts the SHIPPED arm64 default. The hard ISIZE-ratio cap (ratio >= 8)
    /// is unaffected — see `compressibility_cap_serializes_high_ratio_keeps_low_ratio`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_large_output_stays_parallel_at_low_t() {
        // silesia-shaped: 212 MiB output, ~67 MiB deflate → ratio ~3.14 (< hard cap 8),
        // output >= the 128 MiB large-output threshold (would trigger the x86 bonus).
        let deflate_len = 67_585_052usize;
        let big = blob_with_isize(211_968_000); // 212 MiB ISIZE
                                                // On aarch64 the selector is OFF → every requested T below the hard cap is kept.
        assert_eq!(effective_parallel_threads(&big, deflate_len, 2), 2);
        assert_eq!(effective_parallel_threads(&big, deflate_len, 3), 3);
        assert_eq!(effective_parallel_threads(&big, deflate_len, 4), 4);
    }

    // NOTE: these three cover the VENDOR formula (Intel / other x86 / aarch64, and
    // AMD at threads <= 8). They call `adjusted_chunk_size_vendor` directly so they
    // are deterministic regardless of the test runner's CPU vendor (the dispatcher
    // `adjusted_chunk_size_bytes` would take the AMD branch on a Zen box). The AMD
    // very-high-thread branch (threads > 8) is covered separately below.
    #[test]
    fn adjusted_chunk_size_keeps_default_on_large_files() {
        // File big enough that chunkSize * 2 * threads <= fileSize.
        // 4 MiB * 2 * 16 = 128 MiB; pick fileSize = 256 MiB.
        let file_size = 256 * 1024 * 1024;
        let got = adjusted_chunk_size_vendor(file_size, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(got, TARGET_COMPRESSED_CHUNK_BYTES);
    }

    #[test]
    fn adjusted_chunk_size_shrinks_for_221mb_fixture() {
        // The bench fixture: 10.7 MB compressed, 16 threads.
        // Vendor formula: max(512 KiB, ceilDiv(ceilDiv(10726414, 48), 512 KiB) * 512 KiB)
        // ceilDiv(10726414, 48) = 223468
        // ceilDiv(223468, 524288) = 1
        // 1 * 524288 = 524288 = 512 KiB
        let got = adjusted_chunk_size_vendor(10_726_414, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(got, 512 * 1024, "expected 512 KiB floor for 10.7 MB / T=16");
        // 10.7 MB / 512 KiB = ~20.5 chunks (matches vendor's "Total Fetched: 21").
    }

    #[test]
    fn adjusted_chunk_size_scales_with_threads() {
        // For a fixed file size, more threads should produce smaller chunks
        // (more parallelism), but never below the 512 KiB floor.
        let file_size = 80 * 1024 * 1024;
        let t4 = adjusted_chunk_size_vendor(file_size, 4, TARGET_COMPRESSED_CHUNK_BYTES);
        let t16 = adjusted_chunk_size_vendor(file_size, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        let t64 = adjusted_chunk_size_vendor(file_size, 64, TARGET_COMPRESSED_CHUNK_BYTES);
        assert!(t4 >= t16, "more threads → smaller (or equal) chunks");
        assert!(t16 >= t64);
        assert!(t64 >= 512 * 1024, "never below 512 KiB floor");
    }

    // --- AMD/Zen2 mid-ratio coarsening band (T>8): a large MID-ratio stream
    //     (weights-shaped, ratio ~1.09) skips the finer file/T² formula and takes
    //     rapidgzip's VENDOR partition; pure-stored (~1.0) and compressible
    //     (>=2.0) streams keep the finer path. Byte-transparent; scoped to Zen2. ---
    #[test]
    fn mid_ratio_band_coarsens_only_mid_ratio_on_amd_high_t() {
        // 83 MiB deflate (weights.safetensors-shaped). At T16 the finer AMD formula
        // yields ~159 chunks (512 KiB floor); the vendor formula yields ~40 (2 MiB).
        let file = 83_099_652usize;
        let vendor = adjusted_chunk_size_vendor(file, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        let amd_fine = adjusted_chunk_size_amd(file, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        assert!(
            vendor > amd_fine,
            "vendor must be coarser than the finer AMD size"
        );

        // Band constants bracket weights (1.09) between pure-stored (1.0) and the
        // lowest finer-preferring compressible corpus (silesia 3.11 / squishy 2.30).
        // Compile-time checked (both operands are consts).
        const {
            assert!(MID_RATIO_COARSE_LOW > 1.0 && MID_RATIO_COARSE_LOW < 1.09);
            assert!(MID_RATIO_COARSE_HIGH > 1.09 && MID_RATIO_COARSE_HIGH <= 2.30);
        }

        // The ratio arg only bites on AMD + T>8. Pin the host parameters
        // explicitly (`is_amd = true`, 16 physical cores) so the assertions are
        // deterministic on EVERY runner — the previous `if cpu_is_amd()` form
        // asserted through the runtime dispatcher, whose incompressible-cap arm
        // reads the HOST's physical-core count: on an AMD CI runner with fewer
        // than 16 physical cores the pure-stored expectation silently changed
        // (a hardware-fingerprint flake, not a code regression).
        let mid = adjusted_chunk_size_bytes_with(
            file,
            16,
            TARGET_COMPRESSED_CHUNK_BYTES,
            1.09,
            true,
            || 16,
        );
        let stored = adjusted_chunk_size_bytes_with(
            file,
            16,
            TARGET_COMPRESSED_CHUNK_BYTES,
            1.003,
            true,
            || 16,
        );
        let compressible = adjusted_chunk_size_bytes_with(
            file,
            16,
            TARGET_COMPRESSED_CHUNK_BYTES,
            3.11,
            true,
            || 16,
        );
        // mid-ratio → vendor (coarse); the other two → finer AMD formula.
        // (T=16 <= 16 physical cores, so the incompressible cap is an identity
        // here and pure-stored keeps the full file/T² refinement.)
        assert_eq!(mid, vendor, "mid-ratio must take the vendor partition");
        assert_eq!(
            stored, amd_fine,
            "pure-stored must keep the finer AMD partition"
        );
        assert_eq!(
            compressible, amd_fine,
            "compressible must keep the finer AMD partition"
        );
        assert!(mid > stored, "mid-ratio coarser than pure-stored");

        // On an SMT-oversubscribed host (8 physical cores, T=16) the pure-stored
        // stream instead reuses the winning T=physical grid — the incompressible
        // high-T chunk-COUNT cap (see `incompressible_refine_divisor`).
        let stored_smt = adjusted_chunk_size_bytes_with(
            file,
            16,
            TARGET_COMPRESSED_CHUNK_BYTES,
            1.003,
            true,
            || 8,
        );
        assert_eq!(
            stored_smt,
            adjusted_chunk_size_amd(file, 8, TARGET_COMPRESSED_CHUNK_BYTES),
            "pure-stored at T>physical must reuse the T=physical AMD grid"
        );

        // Below T>8 the band is never consulted (only the T>8 branch reads it),
        // so T8 is ratio-invariant.
        let t8_mid = adjusted_chunk_size_bytes_with(
            file,
            8,
            TARGET_COMPRESSED_CHUNK_BYTES,
            1.09,
            true,
            || 16,
        );
        let t8_hi = adjusted_chunk_size_bytes_with(
            file,
            8,
            TARGET_COMPRESSED_CHUNK_BYTES,
            3.11,
            true,
            || 16,
        );
        assert_eq!(
            t8_mid, t8_hi,
            "T<=8 must be byte-identically routed for any ratio"
        );

        // Non-AMD hosts never consult the ratio at any T (vendor partition, with
        // no AMD mid-T cap): pinned is_amd = false.
        let non_amd = adjusted_chunk_size_bytes_with(
            file,
            16,
            TARGET_COMPRESSED_CHUNK_BYTES,
            1.09,
            false,
            || 16,
        );
        assert_eq!(non_amd, vendor, "non-AMD hosts take the vendor partition");
    }

    // --- Shipped lever (commit 4062efe3): the incompressible high-T chunk-COUNT cap.
    //     Once the worker count exceeds the PHYSICAL cores, a large incompressible
    //     stream reuses the winning T=physical grid instead of the finer file/threads²
    //     grid (the bbb/storedheavy-T32 SMT-over-subscription fix). Locks the pure
    //     divisor selector + the >32 MiB file-size gate + the AMD call-site routing. ---
    #[test]
    fn incompressible_cap_reuses_physical_grid_above_physical_cores() {
        // Pin the physical-core count (12) so the expectations are deterministic
        // on every host — the selector takes it as an explicit parameter.
        let phys = 12usize;
        let phys_fn: fn() -> usize = || 12;
        let large = 723 * 1024 * 1024usize; // > INCOMPRESSIBLE_CAP_MIN_FILE_BYTES
        let stored = 1.003_f64; // ratio < MID_RATIO_COARSE_LOW (incompressible)
        let compressible = 3.11_f64; // ratio >= MID_RATIO_COARSE_LOW

        // threads <= physical → the divisor is the identity (byte-identical grid).
        assert_eq!(incompressible_refine_divisor(stored, large, 1, phys_fn), 1);
        assert_eq!(
            incompressible_refine_divisor(stored, large, phys, phys_fn),
            phys
        );

        // threads > physical on a large incompressible stream → capped at physical.
        let over = phys + 1;
        assert_eq!(
            incompressible_refine_divisor(stored, large, over, phys_fn),
            phys,
            "incompressible T>physical must reuse the T=physical divisor"
        );
        // The cap only ever SHRINKS the divisor (coarser grid), never grows it.
        assert!(incompressible_refine_divisor(stored, large, over, phys_fn) <= over);

        // Compressible stream (ratio >= LOW) → the cap arm is NOT taken; full divisor.
        assert_eq!(
            incompressible_refine_divisor(compressible, large, over, phys_fn),
            over,
            "compressible streams keep the finer file/threads² grid at every T"
        );

        // File-size gate is a strict `>`: at or below 32 MiB the cap is a no-op even
        // for T>physical (skips the physical_core_count /sys probe on small decodes);
        // just above it, the cap engages.
        assert_eq!(
            incompressible_refine_divisor(stored, INCOMPRESSIBLE_CAP_MIN_FILE_BYTES, over, phys_fn),
            over,
            "at exactly the gate size the cap is a no-op (strict >)"
        );
        assert_eq!(
            incompressible_refine_divisor(
                stored,
                INCOMPRESSIBLE_CAP_MIN_FILE_BYTES - 1,
                over,
                phys_fn
            ),
            over
        );
        assert_eq!(
            incompressible_refine_divisor(
                stored,
                INCOMPRESSIBLE_CAP_MIN_FILE_BYTES + 1,
                over,
                phys_fn
            ),
            phys,
            "just above the gate the cap engages"
        );

        // AMD call-site integration: is_amd && threads>8 is the only path that
        // reads the ratio. A large stored stream at T>8 AND T>physical sizes
        // chunks by the T=physical grid — coarser than (or equal to) the
        // uncapped file/threads² grid. Host parameters pinned → deterministic
        // everywhere (no cpu_is_amd() / topology dependence).
        let threads = core::cmp::max(9, phys + 1); // > 8 (call-site gate) and > physical
        let capped = adjusted_chunk_size_bytes_with(
            large,
            threads,
            TARGET_COMPRESSED_CHUNK_BYTES,
            stored,
            true,
            phys_fn,
        );
        let physical_grid = adjusted_chunk_size_amd(large, phys, TARGET_COMPRESSED_CHUNK_BYTES);
        let uncapped_grid = adjusted_chunk_size_amd(large, threads, TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(
            capped, physical_grid,
            "AMD incompressible T>physical must size chunks by the T=physical grid"
        );
        assert!(
            capped >= uncapped_grid,
            "capped grid must be coarser-or-equal to the uncapped file/threads² grid"
        );
    }

    // --- AMD/Zen2 mid-thread (2..=8) per-T chunk-size cap: shrink the chunk size
    //     as T grows so per-chunk decode-time variance spreads across workers ---
    #[test]
    fn amd_midt_chunk_cap_is_the_gated_per_t_schedule() {
        // Per-T schedule (silesia/Zen2): T3/T4 4 MiB,
        // T5/T6 3 MiB, T7/T8 2.5 MiB. T2 stays at 4 MiB (default; finer hurts T2).
        assert_eq!(amd_midt_chunk_cap(2), TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(amd_midt_chunk_cap(3), TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(amd_midt_chunk_cap(4), TARGET_COMPRESSED_CHUNK_BYTES);
        assert_eq!(amd_midt_chunk_cap(5), 3 * 1024 * 1024);
        assert_eq!(amd_midt_chunk_cap(6), 3 * 1024 * 1024);
        assert_eq!(amd_midt_chunk_cap(7), 2560 * 1024);
        assert_eq!(amd_midt_chunk_cap(8), 2560 * 1024);
        // Monotone non-increasing in T (never coarsens as workers are added).
        for t in 2..8usize {
            assert!(amd_midt_chunk_cap(t) >= amd_midt_chunk_cap(t + 1));
        }
        // Never below the 512 KiB floor / sub-MiB (Intel blowup / throughput hit).
        for t in 2..=8usize {
            assert!(amd_midt_chunk_cap(t) >= MIN_ADJUSTED_CHUNK_BYTES);
        }
    }

    // --- AMD/Zen2 high-thread finer-chunk formula (chunk = file / threads²,
    //     refined DOWN from `default`, floored at 512 KiB) ---
    #[test]
    fn adjusted_chunk_size_amd_matches_per_t_optima_on_silesia() {
        // 68 MB silesia.gz fixture, 4 MiB default. Per-T sizing:
        // T4 coarse ≈4 MiB, T8 ≈1 MiB,
        // T16 512 KiB floor. (T4/T8 do NOT take this branch in the dispatcher —
        // threads > 8 — but the pure helper still caps/floors correctly.)
        let file_size = 68_229_982usize;
        let t4 = adjusted_chunk_size_amd(file_size, 4, TARGET_COMPRESSED_CHUNK_BYTES);
        let t8 = adjusted_chunk_size_amd(file_size, 8, TARGET_COMPRESSED_CHUNK_BYTES);
        let t16 = adjusted_chunk_size_amd(file_size, 16, TARGET_COMPRESSED_CHUNK_BYTES);
        // T4: 68 MB / 16 ≈ 4.07 MiB → capped at the 4 MiB default.
        assert_eq!(t4, TARGET_COMPRESSED_CHUNK_BYTES);
        // T8: 68 MB / 64 ≈ 1.02 MiB (finer than default, above the floor).
        assert_eq!(t8, file_size / 64);
        assert!(t8 > 512 * 1024 && t8 < TARGET_COMPRESSED_CHUNK_BYTES);
        // T16: 68 MB / 256 ≈ 260 KiB → clamped up to the 512 KiB floor.
        assert_eq!(t16, 512 * 1024);
    }

    #[test]
    fn adjusted_chunk_size_amd_never_below_floor_or_above_default() {
        let file_size = 68_229_982usize;
        for t in [5usize, 8, 12, 16, 32, 64] {
            let got = adjusted_chunk_size_amd(file_size, t, TARGET_COMPRESSED_CHUNK_BYTES);
            assert!(got >= 512 * 1024, "T{t}: never below the 512 KiB floor");
            assert!(
                got <= TARGET_COMPRESSED_CHUNK_BYTES,
                "T{t}: never above default"
            );
        }
    }

    #[test]
    fn adjusted_chunk_size_amd_finer_than_or_equal_vendor_at_high_t() {
        // At high T the AMD branch must not be COARSER than the vendor formula.
        let file_size = 68_229_982usize;
        for t in [8usize, 16] {
            let amd = adjusted_chunk_size_amd(file_size, t, TARGET_COMPRESSED_CHUNK_BYTES);
            let vendor = adjusted_chunk_size_vendor(file_size, t, TARGET_COMPRESSED_CHUNK_BYTES);
            assert!(
                amd <= vendor,
                "T{t}: AMD chunk {amd} should be <= vendor {vendor}"
            );
        }
    }

    #[test]
    fn adjusted_chunk_size_amd_subfloor_default_does_not_panic() {
        // A sub-floor `default_chunk_size` must not panic (min-then-max, not
        // clamp) — the floor wins.
        let got = adjusted_chunk_size_amd(68_229_982, 16, 256 * 1024);
        assert_eq!(got, 512 * 1024);
    }

    #[test]
    fn small_input_returns_hard_error() {
        let small = [0u8; 100];
        let mut out = Vec::new();
        let err = decompress_parallel(&small, &mut out, None, 4, false).unwrap_err();
        // Either InvalidHeader (no gzip magic) or InvalidGzipFormat
        // (too short for a deflate body). Both are terminal —
        // `TooSmall` is gone.
        assert!(matches!(
            err,
            ParallelError::InvalidGzipFormat | ParallelError::InvalidHeader
        ));
    }

    // Gated on `parallel_sm`: this asserts a SUCCESSFUL decode, which only
    // the `parallel_sm` body of `decompress_parallel` can produce. In the
    // default (`not(parallel_sm)`) build the function is a cfg-stub that
    // returns `UnsupportedPlatform` by design (no pure-Rust engine compiled
    // in), so an ungated test would fail on every default `cargo test` even
    // though nothing is wrong. The correctness assertions (byte-exact output,
    // ISIZE) are unchanged — they run in full wherever the engine exists.
    #[cfg(parallel_sm)]
    #[test]
    fn single_thread_decodes_small_input() {
        // Pure-Rust-sole (task #8): the engine is the ONLY single-member
        // decode path at every size/T. A small input at num_threads=1
        // DECODES — it used to hard-error below the 4 MiB floor (when a
        // C-FFI one-shot existed to catch it); that floor and that
        // fallback are both gone, so the engine handles it directly.
        use std::io::Write as _;
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&vec![0u8; 5_000_000]).unwrap();
        let gz = enc.finish().unwrap();
        let mut out = Vec::new();
        let n = decompress_parallel(&gz, &mut out, None, 1, false)
            .expect("pure-Rust SM decodes a small input at T=1");
        assert_eq!(n, 5_000_000);
        assert_eq!(out, vec![0u8; 5_000_000]);
    }
}
