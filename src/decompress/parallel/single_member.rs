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
/// At T1 the pipeline is inline (no workers, no prefetch — see the night4
/// in-flight-depth note) and the only thing chunk size changes is the size of
/// each chunk's materialized output buffer. A SMALLER per-chunk output buffer
/// stays under the allocator's mmap threshold, so when the in-order-drain
/// recycles it (depth-1) the next chunk decodes into the SAME warm, already-
/// faulted pages instead of a fresh mmap → faults drop ~67% (nasa
/// 11381→3769) / ~42% (silesia 13911→8095) and the warm working set cuts
/// cyc/byte (GATED paired N=15: nasa −0.125 cyc/B / −5.69% p=0.0007; silesia
/// −0.079 cyc/B / −1.37% p=0.0011; both CI-excl-0, byte-exact).
/// SCOPED TO T1: at T>1 the finer granularity REGRESSES the parallel pipeline
/// (silesia T4 wall +20%, measured) because more/smaller chunks add
/// block-finder + scheduling overhead, so T>1 keeps the 4 MiB default.
/// Intel-LXC NOT-YET-LAW (AMD/Zen2 replication owed).
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
/// expansion ratio. GATED (Intel, `fulcrum abmeasure`, paired N=15 interleaved,
/// load-immune sign-test): monorepo gz/igzip 1.00→0.98 (15/15 faster, p=6e-5),
/// nasa →0.95 (15/15, p=6e-5), silesia NEUTRAL (derived ≈824 KiB, 8/7 TIE —
/// silesia's flat optimum is already ≈1 MiB so it does not over-shrink). 2.5 MiB
/// lands monorepo at ≈494 KiB (confirmed win zone 384–512 KiB) and silesia at
/// ≈824 KiB (confirmed flat zone 768–1024 KiB). Intel NOT-YET-LAW (AMD owed).
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
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
pub(crate) fn adjusted_chunk_size_bytes(
    file_size: usize,
    num_threads: usize,
    default_chunk_size: usize,
) -> usize {
    let threads = num_threads.max(1);
    // AMD/Zen2 VERY-HIGH-THREAD finer-chunk dispatch (GATED silesia T16, N=15,
    // load-immune `fulcrum abmeasure`, llama-safe quiet window): Zen's wide CCX pool
    // wins with MORE, smaller chunks once the worker count is high — finer
    // granularity flips T16 from a loss to a win (gz/rg 1.035 → 0.956, after/base
    // 15/0 paired, Δ +7.7%). Restricted to threads > 8:
    //   • T2/T4 REGRESS badly with finer chunks (silesia T4 gz/rg 1.04 → 1.27 @ 1 MiB),
    //   • T8 is a measured TIE — the finer effect flips sign across interleaved runs
    //     (sweep 0.99 vs quiet-window gate 1.04), so it stays on the coarse baseline,
    // and ALL of T1–T8 keep the coarse vendor spacing. It is ALSO restricted to AMD
    // because on Intel ANY chunk <= 1 MiB triggers a 3–4× wall BLOWUP (silesia
    // T8/T16, N=15, all 15/15 slower) while the vendor default (>= 1.5 MiB) wins
    // every T — so Intel / other x86 / aarch64 keep the vendor formula unchanged
    // (byte-identical wall). Dispatch is by `cpu_is_amd()` (cached cpuid vendor
    // string) and only moves chunk BOUNDARIES — decoded output is byte-identical.
    if cpu_is_amd() && threads > 8 {
        return adjusted_chunk_size_amd(file_size, threads, default_chunk_size);
    }
    let vendor = adjusted_chunk_size_vendor(file_size, threads, default_chunk_size);
    // AMD/Zen2 MID-THREAD (2..=8): cap the chunk size by the per-T schedule so the
    // per-chunk decode-TIME variance spreads across more workers as T grows (see
    // `amd_midt_chunk_cap`). threads>8 already refines via the AMD branch above;
    // threads==1 is the serial T1 path. `min` preserves the vendor small-file
    // shrink (never coarsens); only shrinks large-file chunks.
    if cpu_is_amd() && (2..=8).contains(&threads) {
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
/// ≈4 MiB @ T4 (capped at the default), ≈1 MiB @ T8, and the 512 KiB floor @ T16 —
/// matching the measured per-T optima. `min`-then-`max` (not `clamp`) so a sub-floor
/// `default_chunk_size` can never panic; the floor always wins. Byte-transparent —
/// only moves chunk boundaries, not decoded output.
#[allow(dead_code)] // used by the x86_64+isal-compression decompress_parallel path
fn adjusted_chunk_size_amd(file_size: usize, threads: usize, default_chunk_size: usize) -> usize {
    let target = file_size / threads.saturating_mul(threads).max(1);
    target.min(default_chunk_size).max(MIN_ADJUSTED_CHUNK_BYTES)
}

/// AMD/Zen2 MID-THREAD (2..=8) per-T chunk-size CAP (byte-transparent).
///
/// MECHANISM (measured, not divisibility). The consumer streams chunks IN ORDER
/// while a `pool_size`-worker pull-queue decodes ahead. The makespan is set by a
/// STRAGGLER worker, and the straggler is a SLOW CHUNK, not an "extra" chunk from
/// a count that fails to divide `T`: at silesia-T7 the GZIPPY_TIMELINE shows two
/// workers with the SAME 2-chunk load differing 76 ms vs 127 ms (1.67× per-chunk
/// decode-time variance). Rounding the chunk COUNT to a multiple of `T` does NOT
/// remove the tail (balanced 14-chunk T7 still measured a 53 ms tail) and
/// regresses the wall (it detunes the chunk size). The lever that WORKS is the
/// chunk SIZE: as `T` grows, smaller chunks spread the decode-time variance across
/// more workers so no single straggler dominates the tail — while staying coarse
/// enough to dodge the low-worker throughput / per-chunk-overhead penalty.
///
/// The schedule is the GATED per-T optimum on silesia/Zen2 (`fulcrum abmeasure`,
/// N=11–15, load-immune, base/rg): T3/T4 want the 4 MiB default (finer REGRESSES
/// them — 0.99/0.97 vs 1.03/1.05 at 3 MiB), T5/T6 want 3 MiB (0.967/0.913 vs
/// 1.006/0.986 at 4 MiB), T7/T8 want 2.5 MiB (0.958/0.903 vs 1.022/0.995 at
/// 4 MiB). Note the T3/T4 PLATEAU: the optimum is not a smooth per-worker formula
/// (a `file/(T·k)` form would shrink T4 below 4 MiB and regress its win), so the
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
/// [`T1_OUTPUT_RESIDENT_TARGET_BYTES`] for the gated rationale + measurements.
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
/// `margin = 1.0` reproduces the gated per-corpus crossovers (silesia ratio
/// 2.75 → T3, monorepo → T6, storedheavy → T7-8). Frozen (was
/// `GZIPPY_PARALLEL_CROSSOVER_MARGIN`, unset in production); `margin = 0`
/// still disables the selector (legacy always-parallel-below-ratio_max
/// behaviour) when passed explicitly to [`effective_parallel_threads_with`].
/// (aarch64 disables the selector entirely — see [`arch_crossover_margin_default`] —
/// so this Intel-default constant is unconsumed there.)
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
const PARALLEL_CROSSOVER_MARGIN_DEFAULT: f64 = 1.0;

/// AMD/Zen crossover-margin default. Zen2's per-chunk parallel overhead is higher
/// than Raptor Lake, so the marginal-parallelism crossover sits one notch higher;
/// `margin = 1.6` (crossover `ceil(ratio·1.6)`) is the GATED value that erases the
/// Zen2 monorepo-T6/T8 default-constant regression (AMD/Zen2, N=11, load-immune
/// interleaved paired ratios, plans/XARCH-CONCURRENCY-LAW-2026-06-26.md). Selected
/// at runtime by [`cpu_is_amd`]. Frozen (was `GZIPPY_PARALLEL_CROSSOVER_MARGIN`).
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
/// needs the high one). Gated (Intel, N≥11, parallel-vs-serial-vs-rapidgzip):
///   silesia  212 MiB out, ratio 3.1  → parallel first beats serial at T3 (proxy
///                                       said 4) → 1-notch bonus routes T3 parallel
///   squishy  480 MiB out, ratio 2.76 → first beats serial at T2 (proxy said 3)
///   monorepo  51 MiB out, ratio 5.18 → first beats serial at T7 (proxy said 6);
///                                       BELOW the threshold ⇒ NO bonus (correct)
/// Conservative by construction: the bonus only fires for clearly-large outputs
/// where the parallel arm at `crossover-1` still beats single-thread igzip by a
/// wide margin (silesia T3 516 ≪ igzip ~686; squishy T2 1184 ≪ igzip ~1215), so
/// it can never manufacture an igzip regression. Frozen (was
/// `GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES`; `0` still disables the bonus when
/// passed explicitly to [`effective_parallel_threads_with`]).
pub(crate) const PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT: u64 = 128 * 1024 * 1024;

/// Number of crossover notches the large-output bonus subtracts (default/Intel).
/// One notch reproduces the Raptor-Lake gate (silesia crossover 4→3, squishy 3→2).
/// Frozen (was `GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH`).
const PARALLEL_LARGE_OUTPUT_NOTCH_DEFAULT: u64 = 1;

/// AMD/Zen large-output bonus depth: TWO notches. The Zen2 margin (1.6) is needed
/// for SMALL outputs (monorepo, ratio 5.18 → crossover 9 → parallel never beats
/// serial through T8); but for LARGE outputs the same 1.6 over-inflates the
/// crossover (silesia ratio 3.1 → ceil(3.1·1.6)=5; squishy 2.76 → 4), pushing the
/// marginal-parallelism knee one notch too high. A single notch leaves silesia-T3
/// SERIAL (loses rg 1.061 — regresses the north-star cell); a SECOND notch lands
/// silesia at crossover 3 (T3 parallel, ties rg 0.99) and squishy at 2 (T2 parallel
/// — still beats igzip 0.985 + rg 0.901). GATED (AMD/Zen2, N≥9, load-immune
/// interleaved paired ratios; plans/ARCH-DISPATCH-ZEN2-T3-2026-06-26.md): margin
/// 1.6 + 2-notch erases the monorepo-T6/T8 + squishy-T2 regressions WITHOUT
/// regressing silesia-T3 / squishy-T3. Selected at runtime by [`cpu_is_amd`].
/// Frozen (was `GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH`).
const PARALLEL_LARGE_OUTPUT_NOTCH_AMD: u64 = 2;

/// SMALL-OUTPUT SERIAL FLOOR: decoded-output size below which the parallel
/// single-member pipeline is capped to one thread (the fast inline T1 path).
/// This is the ALL-MARKER-BLOWUP guard (arch-independent — fires on every arch).
///
/// MECHANISM (gated AMD/Zen2, N≥15, load-immune, marker-stat instrument
/// `feat/marker-stat-instrument`): a small, compressible single-member stream
/// with LOW deflate-block-boundary density — e.g. `markup.xml` (7.6 MiB XML, a
/// handful of huge dynamic blocks) — has no findable block start inside a chunk,
/// so EVERY speculatively-decoded chunk stays window-absent all-marker (measured
/// `marker_byte_frac = 0.9997` vs silesia's 0.45, with 23 742 failed header
/// probes vs silesia's 20). The parallel path then runs at ~20 MB/s vs the serial
/// inline path's ~600 MB/s — a ~13× LOSS to rapidgzip at T≥6 (388 ms vs 30 ms),
/// WORSE than gz's own T1, while T1/T4 (routed serial by the crossover) WIN.
/// The ISIZE ratio does NOT separate this input (markup 3.71 ≈ silesia 3.10); the
/// separating signal is OUTPUT SIZE. Below a few MiB the parallel pipeline's fixed
/// overhead is unamortizable AND the all-marker re-decode dominates, while the
/// serial inline path already BEATS rapidgzip on every squishy item at T1 (markup
/// 13.5 ms vs rg 21.9 ms; the smallest squishy item where parallel reliably helps
/// is ≥14 MiB). 8 MiB sits above markup (7.60 MiB) and below the next squishy item
/// (aozora 11.4 MiB, a healthy TIE), so only markup-and-smaller (all TIE-or-WIN at
/// serial) route serial — no genuine parallel win is forfeited. A COMPRESSIBILITY
/// gate (ratio ≥ 1.5) excludes near-incompressible small files (photo.jpg, ratio
/// 1.005): they are stored-block-dominated, carry no marker pathology, and DO
/// parallelize (20 ms parallel vs 28 ms serial), so they stay parallel.
/// Byte-transparent: only the effective thread count changes; CRC32 + ISIZE are
/// still verified by the caller. Frozen (was `GZIPPY_PARALLEL_MIN_OUTPUT_BYTES`;
/// `0` still disables when passed explicitly to
/// [`effective_parallel_threads_with`]).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
const PARALLEL_MIN_OUTPUT_BYTES_DEFAULT: u64 = 8 * 1024 * 1024;

/// Runtime CPU-vendor detection for the arch-dispatched selector constants.
/// AMD (Zen) needs a higher crossover margin (for small outputs) AND a deeper
/// large-output bonus (2 notches), both GATED on Zen2 — see
/// [`PARALLEL_CROSSOVER_MARGIN_AMD`] / [`PARALLEL_LARGE_OUTPUT_NOTCH_AMD`]. Intel
/// and every other arch keep the Raptor-Lake-gated `margin = 1.0` + 1-notch bonus
/// (arm64 decode CI-green with those). The dispatch is by VENDOR (`AuthenticAMD`),
/// the deterministic signal; it only changes the parallel THREAD-COUNT routing
/// (byte-identical output). `cpuid` leaf 0 is read once and cached.
#[cfg(target_arch = "x86_64")]
pub(crate) fn cpu_is_amd() -> bool {
    use std::sync::OnceLock;
    static IS_AMD: OnceLock<bool> = OnceLock::new();
    *IS_AMD.get_or_init(|| {
        // SAFETY: cpuid leaf 0 (vendor string) is available on every x86_64 CPU.
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
/// bonus are x86/Zen2-tuned — their crossover constants were GATED only on Intel
/// Raptor-Lake (margin 1.0 + 1-notch) and AMD Zen2 (margin 1.6 + 2-notch). On
/// arm64 they MISFIRE: silesia (ratio ~3.1) → crossover ceil(3.1)=4, minus the
/// 1-notch large-output bonus = 3, which routes silesia T2 to the SERIAL floor
/// where the prestack always-parallel-below-ratio_max routing ran parallel —
/// measured REGRESSION (fulcrum wall --steady, M1, silesia T2 +5.7% / T3 +1.2%
/// vs reimplement-isa-l@e1f0c99d). Return margin 0.0 to DISABLE the cost-model
/// selector on aarch64, restoring the prestack routing (whose high-ratio
/// serialization is the aarch64-only [`AARCH64_PRESTACK_RATIO_MAX`] cap in
/// [`effective_parallel_threads`] — removing that cap regresses M1 2.1-2.6× on
/// logs/software-class at T2, paired N=15). x86_64 codegen is byte-identical
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

/// Counter proving the serial-clean cost-model selector fired on a real decode
/// (deletion-trap discipline; read by the routing test).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub static SERIAL_CLEAN_FLOOR_APPLIED: AtomicU64 = AtomicU64::new(0);

/// aarch64-only prestack ISIZE-ratio cap (see the block comment in
/// [`effective_parallel_threads`]): with the cost-model selector disabled on
/// aarch64 (margin 0.0), this cap is the arch's ONLY high-ratio serialization,
/// and removing it measurably regresses Apple M1 (logs-class T2 2.11×,
/// software-class T2 2.62×; paired N=15, 0-1/15 wins). x86_64 deletes the cap
/// entirely (the T-aware selector owns high-ratio routing there — gated wins on
/// nasa/bignasa/logs-class T16, both x86 boxes). Knob-free: no env override.
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

/// Counter proving the work-per-thread cap (the over-threading guard) fired on a
/// real decode (deletion-trap discipline; read by the routing test).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub static WORK_PER_THREAD_CAP_APPLIED: AtomicU64 = AtomicU64::new(0);

/// Minimum COMPRESSED bytes of work each parallel worker must be given before
/// spawning another. Over-threading guard: on a SMALL near-incompressible stream
/// (movie.mp4 12.9 MB, ratio ~1.0) the vendor chunk-size shrink floors chunks at
/// [`MIN_ADJUSTED_CHUNK_BYTES`] (512 KiB) once `file/(3·T)` drops below it, so at
/// high T the file is split into ~2× more chunks than at its knee (movie: 13
/// chunks @ T8 → 25 @ T16) — doubling every per-chunk serial cost (bootstrap scan,
/// window seed, block-finder validation, consumer coordination, pool-pick
/// contention) while the short decode cannot hide it. Result: NEGATIVE thread
/// scaling past the knee (movie T16 +9.64% SIG slower than T8, Intel, paired N=41).
/// Capping effective-T to `deflate_len / this` keeps each worker's chunk coarse
/// enough to stay on the amortized side of the knee. `0` disables the cap (the
/// A/B baseline). Env override: `GZIPPY_MIN_BYTES_PER_THREAD` (the one live
/// lever left in this selector — see [`effective_parallel_threads`]).
const MIN_COMPRESSED_BYTES_PER_THREAD_DEFAULT: u64 = 0;

/// Frozen default for the work-per-thread cap's physical-core floor (was
/// `GZIPPY_MIN_THREADS_FLOOR`, unset in production). See the mechanism note
/// on the `floor` local in [`effective_parallel_threads_with`].
pub(crate) const MIN_THREADS_FLOOR_DEFAULT: u64 = 1;

/// Production entry point: [`effective_parallel_threads_with`] fed the frozen
/// selector defaults (the campaign-measured values with every
/// `GZIPPY_PARALLEL_*` / `GZIPPY_MIN_THREADS_FLOOR` knob unset), plus the one
/// live lever `GZIPPY_MIN_BYTES_PER_THREAD`.
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub(crate) fn effective_parallel_threads(
    gzip_data: &[u8],
    deflate_data_len: usize,
    num_threads: usize,
) -> usize {
    let min_bpt = std::env::var("GZIPPY_MIN_BYTES_PER_THREAD")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(MIN_COMPRESSED_BYTES_PER_THREAD_DEFAULT);
    effective_parallel_threads_with(
        gzip_data,
        deflate_data_len,
        num_threads,
        PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
        arch_crossover_margin_default(),
        PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
        arch_large_output_notch_default(),
        min_bpt,
        MIN_THREADS_FLOOR_DEFAULT,
    )
}

/// Content-derived effective thread count for the parallel single-member
/// pipeline. Three guards, in order (each byte-transparent — only the thread
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
///      pre-empted this selector and forfeited high-T wins (nasa Intel T16:
///      capped 220 ms = 1.20 LOSS vs rg → selector-routed 150 ms = 0.82 WIN).
///   3. WORK-PER-THREAD CAP — over-threading guard for small files at high T.
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
    min_bpt: u64,
    threads_floor: u64,
) -> usize {
    if num_threads <= 1 || deflate_data_len == 0 || gzip_data.len() < 4 {
        return num_threads;
    }

    // SMALL-OUTPUT SERIAL FLOOR (x86_64) — the all-marker-blowup guard. A small,
    // compressible, low-block-boundary-density stream (markup.xml) degenerates into
    // an all-marker re-decode under the parallel pipeline (~13× LOSS to rapidgzip at
    // T≥6) while the serial inline path BEATS rapidgzip; below the floor there is no
    // real parallel win to forfeit. Cap to serial. See
    // [`PARALLEL_MIN_OUTPUT_BYTES_DEFAULT`] for the gated mechanism + measurements.
    // Byte-transparent (only the thread count changes). Guarded by a COMPRESSIBILITY
    // gate (ratio >= 1.5) so near-incompressible small files (photo.jpg) — which are
    // stored-block-dominated, have no marker pathology, and parallelize fine — stay
    // parallel, AND by `deflate < floor` (a true >4 GiB member whose ISIZE wrapped to
    // a small value still has a huge COMPRESSED size and correctly stays parallel).
    // ARCH-INDEPENDENT: the all-marker blowup reproduces on BOTH AMD/Zen2 (routed
    // parallel at T≥6 by the crossover) AND aarch64/Apple-M1 (selector disabled ⇒
    // routed parallel at T≥2; measured T1 33 ms → T4/T8/T16 ~180 ms, ~5× blowup), so
    // the floor applies on every arch (gated on AMD/Zen2 + Apple-M1; Intel owed, same
    // x86_64 code path as AMD).
    {
        let n = gzip_data.len();
        let isize_field = u32::from_le_bytes([
            gzip_data[n - 4],
            gzip_data[n - 3],
            gzip_data[n - 2],
            gzip_data[n - 1],
        ]) as u64;
        let deflate_len = deflate_data_len as u64;
        // COMPRESSIBILITY GATE (ratio >= 1.5, i.e. 2·isize >= 3·deflate): the
        // all-marker blowup REQUIRES compressibility — markers come from back-
        // references, which near-incompressible data (stored-block-dominated) does
        // not produce. A small INcompressible file (photo.jpg: 6.5 MiB, ratio 1.005)
        // decodes as clean stored blocks and PARALLELIZES fine (measured 20 ms
        // parallel vs 28 ms serial), so it must stay parallel; only compressible
        // small files (markup 3.71, minjs 3.31, …) risk the marker pathology and are
        // also faster serial at these sizes. The 2·isize >= 3·deflate form uses
        // integer math (isize <= 4 GiB ⇒ ×2 fits u64). The lower `isize >= deflate`
        // bound is subsumed (ratio 1.5 > 1.0), which also rejects a wrapped >4 GiB
        // ISIZE (isize < deflate).
        if min_output > 0
            && isize_field.saturating_mul(2) >= deflate_len.saturating_mul(3)
            && isize_field < min_output
            && deflate_len < min_output
        {
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
    // (`if isize >= 8*deflate { return 1 }` at EVERY T, env
    // `GZIPPY_PARALLEL_RATIO_MAX`) is DELETED on x86_64. It pre-empted the
    // T-aware cost-model selector below on every high-ratio stream, forfeiting
    // the parallel wins the selector grants at high T. Changed-cell gate (both
    // x86 boxes, single binary, arm B = cap-lifted; N=15 paired interleaved,
    // /dev/null, sha==igzip, A/A noise <0.2%):
    //   Intel  nasa    T16  220 ms → 150 ms  (artifact re-confirm 0.698, 15/15)
    //   Intel  bignasa T16  950 ms → 524 ms  (artifact re-confirm 0.563, 15/15)
    //   AMD    nasa    T16  116 ms → 101 ms  (artifact re-confirm 0.906, 15/15)
    //   AMD    logs_i100/i400/t0 T16          (0.892–0.924, ≥14/15)
    // The selector's `crossover = ceil(ratio × margin)` scales the serial floor
    // with the SAME ratio the cap used, so extreme ratios (logs_i32 11.35 →
    // crossover 17, logs_r3 19.9 → 30) still route serial at every practical T.
    // A content-based replacement (marker-density / block-boundary probe) was
    // designed and FALSIFIED by measurement (probe-a: logs/software have DENSER
    // dynamic-block boundaries — 5-6 KiB mean gap — than silesia's 25 KiB;
    // probe-b: window-absent marker fraction is ~1.0 for ALL corpora since
    // back-refs only reach 32 KiB), so the ISIZE ratio via the selector is the
    // correct — and only — separating signal.
    //
    // aarch64 KEEPS the prestack cap: its cost-model selector is DISABLED
    // (margin 0.0, see [`arch_crossover_margin_default`]) so nothing else would
    // serialize high-ratio streams there, and the bare deletion MEASURABLY
    // REGRESSES Apple M1 (paired interleaved N=15, sha-verified, /dev/null:
    // logs-class ratio 23 → T2 2.11× / T8 1.08× slower, 0-1/15 wins;
    // software-class ratio 43 → T2 2.62× / T8 1.45× slower, 0/15). The cap IS
    // aarch64's prestack routing — byte-transparent, knob-free (no env
    // override; the old `GZIPPY_PARALLEL_RATIO_MAX` knob is deleted per the
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
    // SCAFFOLD are OUTPUT-proportional and under-amortized at low T (gated,
    // T2-MECHANISM-2026-06-26, Intel: parallel does 2.25× silesia – 4.75×
    // monorepo more work than serial-clean; the decode kernel itself is K≈1.0).
    // Predicted parallel speedup ≈ T / W, so parallel only repays once
    //     T >= ceil(W * margin)         (W taken as the ISIZE/deflate ratio).
    // `ceil(ratio)` reproduces the measured per-corpus crossovers (silesia
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
    // gate-tunable safety knob (param, arch-dispatched by the production
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
        // as-is, which is exactly what the gated data wants (monorepo keeps its
        // higher crossover). The notch DEPTH is arch-dispatched: 1 on Raptor Lake,
        // 2 on Zen2 (the Zen2 margin 1.6 over-inflates large-output crossovers; the
        // 2nd notch lands silesia/squishy back at the marginal-parallelism knee).
        // See [`PARALLEL_LARGE_OUTPUT_NOTCH_DEFAULT`] / [`PARALLEL_LARGE_OUTPUT_NOTCH_AMD`].
        if large_output_bytes > 0 && isize_field >= large_output_bytes && notch > 0 {
            crossover = (crossover - notch as f64).max(1.0);
        }
        if (num_threads as f64) < crossover {
            SERIAL_CLEAN_FLOOR_APPLIED.fetch_add(1, Ordering::Relaxed);
            return 1;
        }
    }

    // WORK-PER-THREAD CAP (over-threading guard). Above the serial floors the file
    // IS worth parallelizing, but a SMALL stream can be split into more chunks than
    // there is decode work to amortize their per-chunk serial cost (see
    // [`MIN_COMPRESSED_BYTES_PER_THREAD_DEFAULT`] for the mechanism + the paired
    // movie T8-vs-T16 measurement). Cap effective-T so each worker gets at least
    // `min_bpt` COMPRESSED bytes: `eff = deflate_len / min_bpt` (floored at 1,
    // never raised above the requested `num_threads`). This is a GENERAL
    // work-per-thread law — it binds ONLY when `deflate_len < min_bpt*num_threads`
    // (small file at high T); a large file (silesia/weights/nasa) clears
    // `min_bpt*T` and keeps every requested thread. Byte-transparent: only the
    // thread count changes. `min_bpt == 0` disables (production leaves this on via
    // `GZIPPY_MIN_BYTES_PER_THREAD`, the one live env lever; see
    // [`effective_parallel_threads`]).
    if min_bpt > 0 {
        // FLOOR the cap at the physical-core count of the run's affinity set. The
        // corpus no-regress gate (Intel trainer, paired) proved a cap that drops
        // BELOW ~physical-cores REGRESSES compressible mid-size files (monorepo 9.8 MB
        // ratio 5.2× → capped to 6 was +16.7% SIG slower; its own knee is T8 too) and
        // small incompressible files at moderate T (photo → capped to 4 was +20% SIG).
        // The over-threading is confined to the SMT-sibling tier (threads > physical
        // P-cores): a short compute-bound decode saturates the physical cores, so the
        // 2nd SMT thread per core only adds coordination (chunk-count doubling at the
        // 512 KiB chunk floor). Large files still repay SMT (silesia/weights T16 win)
        // and clear `min_bpt*T` so they are never capped. Flooring at the physical
        // count changes ONLY the sub-floor cases — exactly the regressing ones — so it
        // keeps the movie/tool.bin wins while erasing the monorepo/photo regressions.
        // The floor is TOPOLOGY-derived, not a portable constant; frozen to
        // [`MIN_THREADS_FLOOR_DEFAULT`] (was `GZIPPY_MIN_THREADS_FLOOR`, unset).
        let eff = (deflate_len / min_bpt)
            .max(threads_floor)
            .min(num_threads as u64);
        if eff < num_threads as u64 {
            WORK_PER_THREAD_CAP_APPLIED.fetch_add(1, Ordering::Relaxed);
            return eff as usize;
        }
    }
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

/// Deletion-trap counter for the FALSE-SINGLE late-detection re-entry
/// (§1.4 / R2-#1): a stream the classifier called single-member that actually
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
        // (warm output-buffer recycling win, gated; see
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
            // from the ISIZE expansion ratio (gated win; see the helper doc).
            t1_output_resident_chunk(gzip_data, _deflate_data_len)
        } else {
            TARGET_COMPRESSED_CHUNK_BYTES
        };
        let chunk_size =
            adjusted_chunk_size_bytes(gzip_data.len(), num_threads, thread_default_chunk);
        if chunk_size < TARGET_COMPRESSED_CHUNK_BYTES {
            ADJUSTED_CHUNK_SIZE_APPLIED.fetch_add(1, Ordering::Relaxed);
        }

        // No pool pre-warm here. A prior experiment touched pool pages
        // on the consumer thread before workers spawn; 20-trial bench
        // on neurotic measured a -50% SM regression because every fresh
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
) -> Result<u64, ParallelError> {
    use crate::decompress::parallel::sm_driver::{read_parallel_sm_grid, ReadParallelSmError};

    if gzip_data.len() < 18 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let num_threads = num_threads.max(1);
    let chunk_size =
        adjusted_chunk_size_bytes(gzip_data.len(), num_threads, TARGET_COMPRESSED_CHUNK_BYTES);
    let r =
        read_parallel_sm_grid(gzip_data, writer, out_fd, num_threads, chunk_size).map_err(|e| {
            match e {
                ReadParallelSmError::InvalidHeader => ParallelError::InvalidHeader,
                ReadParallelSmError::InvalidFormat => ParallelError::InvalidGzipFormat,
                ReadParallelSmError::DecodeFailed(detail) => ParallelError::DecodeFailed(detail),
                ReadParallelSmError::SizeMismatch { .. } => ParallelError::SizeMismatch,
                ReadParallelSmError::CrcMismatch { .. } => ParallelError::CrcMismatch,
            }
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
) -> Result<u64, ParallelError> {
    use crate::decompress::parallel::sm_driver::{read_parallel_sm_multi, ReadParallelSmError};

    if gzip_data.len() < 18 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let num_threads = num_threads.max(1);
    let chunk_size =
        adjusted_chunk_size_bytes(gzip_data.len(), num_threads, TARGET_COMPRESSED_CHUNK_BYTES);
    let r = read_parallel_sm_multi(gzip_data, writer, num_threads, chunk_size).map_err(
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
/// (measured on a silesia multi-member: 23.0 → 8.9 insn/byte, 0.66 → 0.22 s,
/// 760 → 106 MiB RSS on M1).
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
) -> Result<u64, ParallelError> {
    use crate::decompress::parallel::sm_driver::{
        read_parallel_sm_grid, read_parallel_sm_resume_multi, ReadParallelSmError,
    };

    if gzip_data.len() < 18 || gzip_data[0] != 0x1f || gzip_data[1] != 0x8b {
        return Err(ParallelError::InvalidGzipFormat);
    }
    let chunk_size = adjusted_chunk_size_bytes(gzip_data.len(), 1, TARGET_COMPRESSED_CHUNK_BYTES);

    let mut counter = CountWriter {
        inner: writer,
        count: 0,
    };
    let grid = read_parallel_sm_grid(gzip_data, &mut counter, None, 1, chunk_size);
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
            let r = read_parallel_sm_resume_multi(gzip_data, inner, streamed, 1, chunk_size)
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

    /// DIAGNOSTIC (manual, ignored): `GZIPPY_PROBE_FILE=<path.gz> cargo test \
    /// --release --features pure-rust-inflate marker_density_measure -- \
    /// --ignored --nocapture`. Prints, for a real gzip file: the dynamic-block
    /// boundary density, the window-absent marker fraction of an interior chunk,
    /// and the per-T selector routing.
    ///
    /// This is the instrument that FALSIFIED the marker-density-predicate design
    /// (2026-07-05): logs/software (ratio 15-43×) have DENSER dynamic-block
    /// boundaries (mean gap 5-6 KiB) than silesia (25 KiB), and the window-absent
    /// marker fraction is ~1.0 for ALL corpora (back-refs only reach 32 KiB), so
    /// neither content signal separates "marker-heavy" from "marker-dormant" —
    /// the ISIZE ratio via the T-aware selector is the correct separating signal.
    /// Kept so the falsification is re-runnable on any corpus.
    #[test]
    #[ignore]
    fn marker_density_measure() {
        use crate::decompress::parallel::blockfinder_validation::DeflateBlockValidator;
        use crate::decompress::parallel::chunk_data::ChunkConfiguration;
        use crate::decompress::parallel::chunk_decode::decode_chunk_window_absent;
        let path = match std::env::var("GZIPPY_PROBE_FILE") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("set GZIPPY_PROBE_FILE=<file.gz>");
                return;
            }
        };
        let gz = std::fs::read(&path).expect("read GZIPPY_PROBE_FILE");
        let header = skip_gzip_header(&gz).expect("gzip header");
        let deflate = &gz[header..gz.len() - 8];
        let dlen = deflate.len();
        let v = DeflateBlockValidator::new(deflate);
        let total_dyn = v.count_dynamic_boundaries_capped(0, dlen * 8, usize::MAX);
        let mean_gap = if total_dyn > 0 {
            dlen / total_dyn
        } else {
            dlen
        };
        eprintln!(
            "PROBE {path}: deflate={} MiB dyn_blocks={total_dyn} mean_gap={} KiB",
            dlen / (1024 * 1024),
            mean_gap / 1024,
        );
        // Window-absent marker fraction of one interior chunk (512 KiB + 4 MiB).
        let mid = dlen / 2;
        let scan = (4 * 1024 * 1024usize).min(dlen - mid) * 8;
        for chunk_kib in [512usize, 4096] {
            if let Some(start_bit) = v.find_first_candidate(mid * 8, scan) {
                let stop = (start_bit + chunk_kib * 1024 * 8).min(dlen * 8);
                match decode_chunk_window_absent(
                    deflate,
                    start_bit,
                    stop,
                    ChunkConfiguration::default(),
                ) {
                    Ok(chunk) => {
                        let mk = chunk.data_with_markers.len();
                        let d = chunk.data.len().saturating_sub(chunk.data_prefix_len);
                        let frac = mk as f64 / (mk + d).max(1) as f64;
                        eprintln!(
                            "  MARKERFRAC[{chunk_kib}KiB comp]: u16_markers={mk} u8_data={d} frac={frac:.4}"
                        );
                    }
                    Err(e) => eprintln!("  MARKERFRAC[{chunk_kib}KiB]: decode err {e:?}"),
                }
            }
        }
        for t in [2usize, 4, 8, 16] {
            eprintln!(
                "  T{t} -> effective {}",
                effective_parallel_threads(&gz, dlen, t)
            );
        }
    }

    /// HIGH-RATIO ROUTING (2026-07-05):
    /// x86_64 — NO T-blind ratio cap: a high-ISIZE-ratio stream is serialized
    /// ONLY below its T-aware crossover (`ceil(ratio × margin)`), never at every
    /// T. The former hard cap (`ratio >= 8 → 1 thread at EVERY T`) forfeited
    /// high-T parallel wins (gated: Intel nasa T16 220→150 ms 15/15, bignasa T16
    /// 950→524 ms 15/15; AMD nasa T16 0.871 15/15, logs_i100/i400/t0 T16
    /// 0.892-0.924) and is DELETED there.
    /// aarch64 — the prestack cap REMAINS (selector disabled; removing the cap
    /// regresses M1 2.1-2.6× on logs/software-class at T2, paired N=15).
    /// Neutral params that isolate a SINGLE guard: floor disabled (`min_output=0`),
    /// large-output bonus disabled (`large_output_bytes=0`), work-per-thread cap
    /// disabled (`min_bpt=0`, `threads_floor` irrelevant then). Tests override
    /// only the dimension(s) they're exercising.
    const NO_FLOOR: u64 = 0;
    const NO_BONUS_BYTES: u64 = 0;
    const NO_BPT_CAP: u64 = 0;

    /// HIGH-RATIO ROUTING (2026-07-05):
    /// x86_64 — NO T-blind ratio cap: a high-ISIZE-ratio stream is serialized
    /// ONLY below its T-aware crossover (`ceil(ratio × margin)`), never at every
    /// T. The former hard cap (`ratio >= 8 → 1 thread at EVERY T`) forfeited
    /// high-T parallel wins (gated: Intel nasa T16 220→150 ms 15/15, bignasa T16
    /// 950→524 ms 15/15; AMD nasa T16 0.871 15/15, logs_i100/i400/t0 T16
    /// 0.892-0.924) and is DELETED there.
    /// aarch64 — the prestack cap REMAINS (selector disabled; removing the cap
    /// regresses M1 2.1-2.6× on logs/software-class at T2, paired N=15).
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
                NO_BPT_CAP,
                MIN_THREADS_FLOOR_DEFAULT,
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
                NO_BPT_CAP,
                MIN_THREADS_FLOOR_DEFAULT,
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
        // photo.jpg-shaped: 6.5 MiB out / 6.48 MiB deflate → ratio 1.005 < 1.5
        // compressibility gate → floor must NOT fire (parallelizes fine as stored
        // blocks; the marker pathology needs compressibility).
        let photo = blob_with_isize(6_511_067);
        assert_eq!(
            call(&photo, 6_481_121, 8, PARALLEL_MIN_OUTPUT_BYTES_DEFAULT, 0.0),
            8
        );
        // Right at the 1.5 gate (isize = 1.5·deflate) → fires (compressible enough).
        let at_gate = blob_with_isize(6_000_000);
        assert_eq!(
            call(
                &at_gate,
                4_000_000,
                8,
                PARALLEL_MIN_OUTPUT_BYTES_DEFAULT,
                0.0
            ),
            1
        );
        // min_output=0 disables the floor: markup-shaped now reaches the crossover.
        assert_eq!(call(&markup, 2_147_400, 8, NO_FLOOR, 0.0), 8);
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
                NO_BPT_CAP,
                MIN_THREADS_FLOOR_DEFAULT,
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
                    NO_BPT_CAP,
                    MIN_THREADS_FLOOR_DEFAULT,
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

    #[test]
    fn arch_dispatch_defaults_match_vendor() {
        // The arch-dispatched defaults must follow the detected CPU vendor: AMD
        // gets the Zen2-gated margin 1.6 + 2-notch bonus; Intel/other x86 keep the
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
    /// T2 (and below) to the serial floor; that x86 tune misfired on arm64 (measured
    /// silesia T2 +5.7% / T3 +1.2% steady-wall regression). No env override here:
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

    // --- AMD/Zen2 mid-thread (2..=8) per-T chunk-size cap: shrink the chunk size
    //     as T grows so per-chunk decode-time variance spreads across workers ---
    #[test]
    fn amd_midt_chunk_cap_is_the_gated_per_t_schedule() {
        // Gated silesia/Zen2 optima (fulcrum abmeasure, base/rg): T3/T4 4 MiB,
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
        // 68 MB silesia.gz fixture, 4 MiB default. Measured per-T optima
        // (fulcrum abmeasure, N=15, load-immune): T4 coarse ≈4 MiB, T8 ≈1 MiB,
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
        let err = decompress_parallel(&small, &mut out, None, 4).unwrap_err();
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
        let n = decompress_parallel(&gz, &mut out, None, 1)
            .expect("pure-Rust SM decodes a small input at T=1");
        assert_eq!(n, 5_000_000);
        assert_eq!(out, vec![0u8; 5_000_000]);
    }
}
