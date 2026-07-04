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
/// Intel-LXC NOT-YET-LAW (AMD/Zen2 replication owed). The explicit
/// `GZIPPY_CHUNK_KIB` env override still wins over this default.
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
/// `default_chunk_size` (e.g. a small `GZIPPY_CHUNK_KIB`) can never panic; the floor
/// always wins. Byte-transparent — only moves chunk boundaries, not decoded output.
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

/// Default ISIZE/deflate ratio at or above which the parallel single-member
/// pipeline is capped to one thread (the fast inline T1 path). Override with
/// `GZIPPY_PARALLEL_RATIO_MAX` (used to lock this constant from a gate-hardware
/// sweep). Rationale + measurements in [`effective_parallel_threads`].
const PARALLEL_RATIO_MAX_DEFAULT: u64 = 8;

/// Default crossover-margin multiplier for the serial-clean cost-model selector
/// (see [`effective_parallel_threads`]). The predicted parallel work-inflation
/// W ≈ ISIZE/deflate ratio; parallel only repays at `T >= ceil(W * margin)`.
/// `margin = 1.0` reproduces the gated per-corpus crossovers (silesia ratio
/// 2.75 → T3, monorepo → T6, storedheavy → T7-8). Override with
/// `GZIPPY_PARALLEL_CROSSOVER_MARGIN`; `0` disables the selector (legacy
/// always-parallel-below-ratio_max behaviour, for A/B).
/// (aarch64 disables the selector entirely — see [`arch_crossover_margin_default`] —
/// so this Intel-default constant is unconsumed there.)
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
const PARALLEL_CROSSOVER_MARGIN_DEFAULT: f64 = 1.0;

/// AMD/Zen crossover-margin default. Zen2's per-chunk parallel overhead is higher
/// than Raptor Lake, so the marginal-parallelism crossover sits one notch higher;
/// `margin = 1.6` (crossover `ceil(ratio·1.6)`) is the GATED value that erases the
/// Zen2 monorepo-T6/T8 default-constant regression (AMD/Zen2, N=11, load-immune
/// interleaved paired ratios, plans/XARCH-CONCURRENCY-LAW-2026-06-26.md). Selected
/// at runtime by [`cpu_is_amd`]; overridable via `GZIPPY_PARALLEL_CROSSOVER_MARGIN`.
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
/// it can never manufacture an igzip regression. Override with
/// `GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES` (gate-tunable; `0` disables the bonus).
const PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT: u64 = 128 * 1024 * 1024;

/// Number of crossover notches the large-output bonus subtracts (default/Intel).
/// One notch reproduces the Raptor-Lake gate (silesia crossover 4→3, squishy 3→2).
/// Override with `GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH`.
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
/// regressing silesia-T3 / squishy-T3. Selected at runtime by [`cpu_is_amd`];
/// overridable via `GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH`.
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
/// still verified by the caller. Override with `GZIPPY_PARALLEL_MIN_OUTPUT_BYTES`
/// (`0` disables).
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

/// Arch-dispatched default crossover margin (env override takes precedence).
///
/// aarch64 (Apple Silicon): the serial-clean cost-model selector + large-output
/// bonus are x86/Zen2-tuned — their crossover constants were GATED only on Intel
/// Raptor-Lake (margin 1.0 + 1-notch) and AMD Zen2 (margin 1.6 + 2-notch). On
/// arm64 they MISFIRE: silesia (ratio ~3.1) → crossover ceil(3.1)=4, minus the
/// 1-notch large-output bonus = 3, which routes silesia T2 to the SERIAL floor
/// where the prestack always-parallel-below-ratio_max routing ran parallel —
/// measured REGRESSION (fulcrum wall --steady, M1, silesia T2 +5.7% / T3 +1.2%
/// vs reimplement-isa-l@e1f0c99d). Return margin 0.0 to DISABLE the cost-model
/// selector on aarch64, restoring EXACTLY the prestack routing (the hard
/// ISIZE-ratio cap above still fires for high-ratio corpora like logs). The
/// `GZIPPY_PARALLEL_CROSSOVER_MARGIN` env override still takes precedence. x86_64
/// codegen is byte-identical (this branch compiles out off-aarch64).
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

/// Arch-dispatched large-output bonus depth in crossover notches (env override
/// takes precedence). AMD subtracts 2; Intel/other subtract 1.
#[cfg_attr(not(parallel_sm), allow(dead_code))]
fn arch_large_output_notch_default() -> u64 {
    if cpu_is_amd() {
        PARALLEL_LARGE_OUTPUT_NOTCH_AMD
    } else {
        PARALLEL_LARGE_OUTPUT_NOTCH_DEFAULT
    }
}

/// Counter proving the serial-clean cost-model selector fired on a real decode
/// (deletion-trap discipline; read by the routing test). Distinct from the hard
/// ratio cap so the two effects can be told apart in a trace.
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub static SERIAL_CLEAN_FLOOR_APPLIED: AtomicU64 = AtomicU64::new(0);

/// Counter proving the compressibility thread-cap fired on a real decode
/// (deletion-trap discipline; read by the routing test).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub static COMPRESSIBILITY_THREAD_CAP_APPLIED: AtomicU64 = AtomicU64::new(0);

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
/// A/B baseline). Env override: `GZIPPY_MIN_BYTES_PER_THREAD`.
const MIN_COMPRESSED_BYTES_PER_THREAD_DEFAULT: u64 = 0;

/// Compressibility-gated effective thread count for the parallel single-member
/// pipeline.
///
/// The pipeline's per-chunk marker-resolution + window-application cost scales
/// with OUTPUT bytes, while the parallelizable decode work scales with
/// COMPRESSED bytes. On a highly-compressible single-member stream (high
/// ISIZE/deflate ratio) the output-proportional overhead overwhelms the
/// parallel speedup, so Tmax REGRESSES below T1. Measured (gzippy-vs-itself,
/// /dev/null, interleaved; crossover transfers across arch):
///   silesia  ratio 2.75x → p8 = 1.94x of T1   (parallel HELPS)
///   logs     ratio 15.2x → p8 = 0.61x of T1   (parallel HURTS — even p2 = 0.59x)
///   software ratio 29.8x → p8 = 0.51x of T1   (parallel HURTS)
/// So when the ratio is at/above the crossover, cap to ONE thread — the inline
/// T1 path — guaranteeing Tmax is never slower than T1. ISIZE wrap (>4 GiB
/// output) or near-incompressible (isize<deflate) yields a low/!computable
/// ratio → keep the requested threads (parallel is correct there).
#[cfg_attr(not(parallel_sm), allow(dead_code))]
pub(crate) fn effective_parallel_threads(
    gzip_data: &[u8],
    deflate_data_len: usize,
    num_threads: usize,
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
        let min_output = std::env::var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(PARALLEL_MIN_OUTPUT_BYTES_DEFAULT);
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

    let ratio_max = std::env::var("GZIPPY_PARALLEL_RATIO_MAX")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(PARALLEL_RATIO_MAX_DEFAULT);
    if ratio_max == 0 {
        return num_threads; // explicit disable of the cap
    }
    let n = gzip_data.len();
    let isize_field = u32::from_le_bytes([
        gzip_data[n - 4],
        gzip_data[n - 3],
        gzip_data[n - 2],
        gzip_data[n - 1],
    ]) as u64;
    let deflate_len = deflate_data_len as u64;
    // ratio = isize/deflate >= ratio_max  ⟺  isize >= ratio_max*deflate (no division)
    if isize_field >= ratio_max.saturating_mul(deflate_len) {
        COMPRESSIBILITY_THREAD_CAP_APPLIED.fetch_add(1, Ordering::Relaxed);
        return 1;
    }

    // SERIAL-CLEAN COST-MODEL SELECTOR (the low-T monotonicity fix).
    //
    // Below the hard ratio cap the parallel pipeline is still NOT free: it does
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
    // gate-tunable safety knob; `0` disables this selector entirely (A/B).
    let margin = std::env::var("GZIPPY_PARALLEL_CROSSOVER_MARGIN")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|m| m.is_finite() && *m >= 0.0)
        .unwrap_or_else(arch_crossover_margin_default);
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
        let large_output_bytes = std::env::var("GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT);
        let notch = std::env::var("GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(arch_large_output_notch_default);
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
    // thread count changes. `min_bpt == 0` disables (A/B baseline).
    let min_bpt = std::env::var("GZIPPY_MIN_BYTES_PER_THREAD")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(MIN_COMPRESSED_BYTES_PER_THREAD_DEFAULT);
    if min_bpt > 0 {
        let eff = (deflate_len / min_bpt).max(1);
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

        // Phase-timing instrument (GZIPPY_PHASE_TIMING=1, byte-transparent, NO rg
        // counterpart). Marks the SERIAL phase boundaries to locate the AMD-T2
        // serial-wrapper excess. reset()+main_start fire in main() so the PRE
        // (main_start->decode_entry) and POST (crc_verified->main_end) wrappers
        // OUTSIDE this region are captured; report() runs in main().
        crate::decompress::parallel::phase_timing::mark("decode_entry");

        // Production driver: `sm_driver::read_parallel_sm` → `chunk_fetcher::drive`.
        // `single_member::decompress_parallel` is now a thin classifier-
        // routed wrapper: it owns the routing-eligibility gate and the
        // `MARKER_PIPELINE_RUNS` counter; the trailer parsing + CRC /
        // ISIZE verification + chunk_fetcher::drive orchestration all
        // live in the new driver (mirror of vendor's
        // `ParallelGzipReader::read` at ParallelGzipReader.hpp:553-646).
        // Granularity probe (2026-05-29): GZIPPY_CHUNK_KIB overrides the
        // 4 MiB default chunk target so a T=16 chunk-count sweep can
        // discriminate "T16 regression is straggler/granularity" from
        // "T16 regression is HT microarchitecture" without a rebuild per
        // size. Falls back to TARGET_COMPRESSED_CHUNK_BYTES when unset.
        // Thread-aware default: T1 uses the smaller 1 MiB target (warm
        // output-buffer recycling win, gated; see T1_TARGET_COMPRESSED_CHUNK_BYTES),
        // T>1 keeps the 4 MiB target (finer granularity regresses the parallel
        // pipeline). Explicit GZIPPY_CHUNK_KIB always overrides.
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
        let default_chunk = std::env::var("GZIPPY_CHUNK_KIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|kib| kib * 1024)
            .unwrap_or(thread_default_chunk);
        let chunk_size = adjusted_chunk_size_bytes(gzip_data.len(), num_threads, default_chunk);
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
        if crate::decompress::parallel::chunk_data::rss_split_enabled() {
            crate::decompress::parallel::chunk_data::rss_split_report(result.total_size as u64);
        }
        if crate::decompress::parallel::chunk_data::lifecycle_enabled() {
            crate::decompress::parallel::chunk_data::lifecycle_report();
        }
        if crate::decompress::parallel::segmented_markers::free_markers_enabled() {
            use crate::decompress::parallel::segmented_markers::{
                MARKER_FREE_BYTES, MARKER_FREE_FIRED,
            };
            let bytes = MARKER_FREE_BYTES.load(Ordering::Relaxed);
            let fired = MARKER_FREE_FIRED.load(Ordering::Relaxed);
            eprintln!(
                "[free_markers] MADV_DONTNEED freed={:.2}MiB chunks_fired={} (non_inert={})",
                bytes as f64 / (1024.0 * 1024.0),
                fired,
                bytes > 0 && fired > 0,
            );
        }
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

    /// Serializes the two tests that read/mutate `GZIPPY_PARALLEL_CROSSOVER_MARGIN`
    /// (process-global) so they never observe each other's writes.
    static SELECTOR_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// ISIZE/deflate ratio gate: highly-compressible single-member caps to one
    /// thread (the inline T1 path) so Tmax never regresses below T1; moderately
    /// compressible keeps the requested threads. Builds a minimal blob whose
    /// last 4 bytes are the gzip-trailer ISIZE the helper reads.
    fn blob_with_isize(isize_val: u32) -> Vec<u8> {
        let mut v = vec![0u8; 64];
        v[60..64].copy_from_slice(&isize_val.to_le_bytes());
        v
    }

    #[test]
    fn compressibility_cap_serializes_high_ratio_keeps_low_ratio() {
        let _guard = SELECTOR_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Pin the selector margin so the assertions are arch-independent (the
        // arch-dispatched default is 1.6 on AMD vs 1.0 on Intel — this test
        // exercises the HARD ratio cap, not the per-arch crossover tune).
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "1.0");
        // Disable the small-output serial floor so this test isolates the HARD
        // ratio cap (its 1–20 MB blobs sit below the 8 MiB floor and would else
        // all route serial before reaching the ratio cap).
        std::env::set_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES", "0");
        let deflate_len = 1_000_000usize;
        // ratio 20x (>= default 8) → cap to 1 thread.
        let high = blob_with_isize(20_000_000);
        assert_eq!(effective_parallel_threads(&high, deflate_len, 8), 1);
        // ratio 3x (< 8) → keep all threads.
        let low = blob_with_isize(3_000_000);
        assert_eq!(effective_parallel_threads(&low, deflate_len, 8), 8);
        // single-thread request is never altered.
        assert_eq!(effective_parallel_threads(&high, deflate_len, 1), 1);
        // exactly at the threshold (8x) caps (>= boundary).
        let at = blob_with_isize(8_000_000);
        assert_eq!(effective_parallel_threads(&at, deflate_len, 8), 1);
        // just under (7.99x) keeps.
        let under = blob_with_isize(7_990_000);
        assert_eq!(effective_parallel_threads(&under, deflate_len, 8), 8);
        std::env::remove_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES");
        std::env::remove_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN");
    }

    /// SMALL-OUTPUT SERIAL FLOOR (x86_64): a small compressible stream (markup.xml-
    /// shaped: 7.6 MiB out / 2.1 MiB deflate) caps to serial to dodge the all-marker
    /// blowup, while a stream at/above the 8 MiB floor keeps its threads. The floor
    /// must NOT fire on a near-incompressible / ISIZE-wrapped stream (isize < deflate)
    /// nor when the compressed size is large (a real >4 GiB member whose ISIZE wrapped).
    /// Arch-independent (the floor now fires on every arch — AMD/Zen2 + Apple-M1).
    #[test]
    fn small_output_serial_floor_caps_markup_shaped_only() {
        let _guard = SELECTOR_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // markup.xml-shaped: 7.6 MiB decoded, 2.1 MiB deflate → below the 8 MiB
        // floor, isize >= deflate → floor fires (serial) at every requested T.
        let markup = blob_with_isize(7_974_912);
        assert_eq!(effective_parallel_threads(&markup, 2_147_400, 6), 1);
        assert_eq!(effective_parallel_threads(&markup, 2_147_400, 8), 1);
        assert_eq!(effective_parallel_threads(&markup, 2_147_400, 16), 1);
        // T1 request is never altered.
        assert_eq!(effective_parallel_threads(&markup, 2_147_400, 1), 1);
        // aozora-shaped: 11.4 MiB decoded (>= 8 MiB floor) → floor does NOT fire;
        // pin margin low so it stays parallel (isolates the floor from the crossover).
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "0");
        let aozora = blob_with_isize(12_003_648);
        assert_eq!(effective_parallel_threads(&aozora, 3_995_900, 8), 8);
        // Near-incompressible small stream (isize < deflate): floor must NOT fire.
        let incompressible = blob_with_isize(1_000_000);
        assert_eq!(effective_parallel_threads(&incompressible, 2_000_000, 8), 8);
        // photo.jpg-shaped: 6.5 MiB out / 6.48 MiB deflate → ratio 1.005 < 1.5
        // compressibility gate → floor must NOT fire (parallelizes fine as stored
        // blocks; the marker pathology needs compressibility).
        let photo = blob_with_isize(6_511_067);
        assert_eq!(effective_parallel_threads(&photo, 6_481_121, 8), 8);
        // Right at the 1.5 gate (isize = 1.5·deflate) → fires (compressible enough).
        let at_gate = blob_with_isize(6_000_000);
        assert_eq!(effective_parallel_threads(&at_gate, 4_000_000, 8), 1);
        std::env::remove_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN");
        // Env override 0 disables the floor: markup-shaped now reaches the crossover.
        std::env::set_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES", "0");
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "0");
        assert_eq!(effective_parallel_threads(&markup, 2_147_400, 8), 8);
        std::env::remove_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN");
        std::env::remove_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES");
    }

    #[test]
    fn serial_clean_selector_crossover_and_margin() {
        let _guard = SELECTOR_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Single sequential test: the margin env var is process-global, so all
        // env-dependent assertions live here (no concurrent test reads it).
        // Pin margin=1.0 so the crossover assertions are arch-independent (the
        // default is 1.6 on AMD).
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "1.0");
        // Isolate the crossover selector from the small-output serial floor
        // (the 1–3 MB blobs here sit below the 8 MiB floor).
        std::env::set_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES", "0");
        let deflate_len = 1_000_000usize;

        // Margin 1.0, ratio 3× (< hard cap 8): crossover = ceil(3.0) = 3.
        let blob = blob_with_isize(3_000_000);
        // T below crossover → serial-clean floor (1 thread).
        assert_eq!(effective_parallel_threads(&blob, deflate_len, 2), 1);
        // T at/above crossover → parallel keeps the requested threads.
        assert_eq!(effective_parallel_threads(&blob, deflate_len, 3), 3);
        assert_eq!(effective_parallel_threads(&blob, deflate_len, 8), 8);

        // Non-integer ratio rounds UP (conservative): ratio 2.75× → crossover 3.
        let blob275 = blob_with_isize(2_750_000);
        assert_eq!(effective_parallel_threads(&blob275, deflate_len, 2), 1);
        assert_eq!(effective_parallel_threads(&blob275, deflate_len, 3), 3);

        // margin 0 disables the selector → keep threads (legacy below-cap behaviour).
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "0");
        assert_eq!(effective_parallel_threads(&blob, deflate_len, 2), 2);

        // Larger margin pushes the crossover up (more conservative): margin 2.0,
        // ratio 3 → crossover 6 → T4 now serial, T6 parallel.
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "2.0");
        assert_eq!(effective_parallel_threads(&blob, deflate_len, 4), 1);
        assert_eq!(effective_parallel_threads(&blob, deflate_len, 6), 6);

        std::env::remove_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES");
        std::env::remove_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN");
    }

    #[test]
    fn serial_clean_selector_large_output_bonus() {
        let _guard = SELECTOR_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Pin margin=1.0 + 128 MiB threshold + 1-notch so the assertions are
        // arch-independent (AMD defaults are margin 1.6 + 2-notch).
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "1.0");
        std::env::set_var("GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES", "134217728");
        std::env::set_var("GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH", "1");

        // LARGE output (200 MiB), ratio 3.1 (< hard cap 8). Proxy crossover =
        // ceil(3.1) = 4, but the large-output bonus lowers it to 3 → T3 parallel,
        // T2 serial. (silesia-shaped: 212 MiB out, ratio 3.1 → first wins at T3.)
        let big_out = 200 * 1024 * 1024u32;
        let big_deflate = (big_out as f64 / 3.1) as usize; // ratio 3.1
        let big = blob_with_isize(big_out);
        assert_eq!(effective_parallel_threads(&big, big_deflate, 2), 1); // serial
        assert_eq!(effective_parallel_threads(&big, big_deflate, 3), 3); // parallel
        assert_eq!(effective_parallel_threads(&big, big_deflate, 4), 4);

        // SMALL output (50 MiB), same ratio 3.1: below the threshold → NO bonus →
        // crossover stays 4 → T3 still serial, T4 parallel. (monorepo-shaped: the
        // fixed overhead is a large fraction, so the proxy crossover is correct.)
        let small_out = 50 * 1024 * 1024u32;
        let small_deflate = (small_out as f64 / 3.1) as usize;
        let small = blob_with_isize(small_out);
        assert_eq!(effective_parallel_threads(&small, small_deflate, 3), 1); // serial
        assert_eq!(effective_parallel_threads(&small, small_deflate, 4), 4); // parallel

        // Bonus disabled via env (0) → large output behaves like the proxy again.
        std::env::set_var("GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES", "0");
        assert_eq!(effective_parallel_threads(&big, big_deflate, 3), 1); // serial (no bonus)

        // 2-notch bonus (the AMD depth): crossover 4 → 2 → T2 parallel.
        std::env::set_var("GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES", "134217728");
        std::env::set_var("GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH", "2");
        assert_eq!(effective_parallel_threads(&big, big_deflate, 2), 2); // parallel (4-2=2)
        assert_eq!(effective_parallel_threads(&big, big_deflate, 1), 1); // never alters T1

        std::env::remove_var("GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH");
        std::env::remove_var("GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES");
        std::env::remove_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN");
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
        let _guard = SELECTOR_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN");
        std::env::remove_var("GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH");
        std::env::remove_var("GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES");
        // silesia-shaped: 212 MiB output, ~67 MiB deflate → ratio ~3.14 (< hard cap 8),
        // output >= the 128 MiB large-output threshold (would trigger the x86 bonus).
        let deflate_len = 67_585_052usize;
        let big = blob_with_isize(211_968_000); // 212 MiB ISIZE
                                                // On aarch64 the selector is OFF → every requested T below the hard cap is kept.
        assert_eq!(effective_parallel_threads(&big, deflate_len, 2), 2);
        assert_eq!(effective_parallel_threads(&big, deflate_len, 3), 3);
        assert_eq!(effective_parallel_threads(&big, deflate_len, 4), 4);
    }

    #[test]
    fn arch_dispatch_env_override_wins_over_arch_default() {
        let _guard = SELECTOR_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Even on AMD (default margin 1.6), an explicit env override must take
        // precedence — proves the env knob still works as the documented override.
        let deflate_len = 1_000_000usize;
        let blob = blob_with_isize(3_000_000); // ratio 3.0
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "1.0");
        // Disable the small-output serial floor (this 3 MB blob sits below the
        // 8 MiB floor) so the assertion isolates the crossover override.
        std::env::set_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES", "0");
        // margin 1.0, ratio 3 → crossover 3 → T3 parallel regardless of host arch.
        assert_eq!(effective_parallel_threads(&blob, deflate_len, 3), 3);
        std::env::remove_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES");
        std::env::remove_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN");
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
        // A sub-floor `default_chunk_size` (e.g. GZIPPY_CHUNK_KIB=256) must not
        // panic (min-then-max, not clamp) — the floor wins.
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
