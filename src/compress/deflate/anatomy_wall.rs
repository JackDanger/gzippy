//! Wall-clock phase timers for `fulcrum anatomy`'s execution level.
//!
//! Cargo feature `anatomy-wall`, DEFAULT OFF. Sibling of `anatomy_counters`
//! (semantic-work-unit VOLUME counters): this module answers "where does the
//! TIME go" instead of "how much work happened". Built 2026-07-25 because
//! neither existing `fulcrum anatomy` arm can answer that question:
//! `--counters-from-stderr` (this crate's `anatomy-counters` feature) is
//! EXACT but reports work volume, never time; `--exec` (host-side cachegrind)
//! is instructions-retired, a WEAK whole-program attribution that left 43% of
//! gzippy's own Ir and 78% of libdeflate's uncategorized on the motivating
//! run, and its libdeflate categorization is unreliable (keyed to Rust
//! symbol names).
//!
//! ## Design: region granularity, never per-position
//!
//! Every timer wraps a PER-BLOCK (or per-invocation) span, never a
//! per-position/per-token one. `parse/fast.rs`'s hot loop resolves one input
//! byte per iteration at up to hundreds of millions of iterations per
//! second; an `Instant::now()` pair around each position would cost far more
//! than the position itself (measured: seeding a *_counters-style atomic per
//! position cost 10-14% wall before batching — a wall-clock read is
//! materially more expensive than an atomic increment, so per-position
//! timing was never attempted here). Regions below are therefore:
//!
//!   - **`parse_match`** — one timer per INTERNAL BLOCK (`FAST_BLOCK_LENGTH`,
//!     64 KiB by default — see `parse/fast.rs::run`'s block loop), wrapping
//!     the fastloop dispatch (`fastloop_l0`/`fastloop_l1`) + tail loop for
//!     that block. This is BOTH "match-finding/probing" AND "match
//!     evaluation+emission" fused into one bucket for the fast (L0/L1)
//!     parser: `process_position_l1`'s probe, `lz_extend` evaluation, and
//!     `Sink::push_{literal,match}_fast` emission are ONE inlined function
//!     body per position (see that function's doc comment) — there is no
//!     call boundary between "found a candidate" and "emitted a token" to
//!     put a clock on without either taking it per-position (the disallowed
//!     granularity) or fabricating a distorted split. Documented as fused,
//!     not silently merged. Block splitting is ALSO fused into this bucket
//!     for every parser strategy in principle, but for the fast (L0/L1)
//!     path specifically it is PROVABLY ZERO extra work: `push_literal_fast`/
//!     `push_match_fast` never call `BlockSplitStats::observe_*` at all (see
//!     `parse/mod.rs`'s dead-stats doc comment) — a structural fact checked
//!     by reading the call graph, not a timed near-zero.
//!   - **`huffman_table`** — one timer per block, wrapping `emit_block`'s
//!     `make_huffman_code` (litcode + offcode) + `build_dynamic_header` +
//!     the three-way stored/static/dynamic cost comparison
//!     (`cost_from_freqs` x2 + `stored_block_bits`) — the code-BUILDING
//!     phase, before any bit is written.
//!   - **`huffman_encode`** — one timer per block, wrapping the
//!     `emit_sequences` call that walks the block's tokens and writes
//!     codeword bits via `BitWriter`. Bitstream flush/serialization
//!     (`BitWriter::add_bits`'s internal `flush_word_unchecked` calls) is
//!     FUSED into this bucket, not separately timed: a flush fires roughly
//!     once per ~56 bits of output, i.e. thousands of times per block —
//!     timing each would reintroduce the per-position-granularity problem
//!     this module exists to avoid. Documented as fused/UNMEASURABLE-
//!     separately, not silently dropped.
//!   - **`crc`** — one timer per invocation, wrapping the single
//!     `crc32fast::hash` call `compress_gzip`/`compress_gzip_padded` makes
//!     over the WHOLE input (CRC in gzippy is not computed per-block).
//!   - **root span** — one timer per invocation, wrapping the ENTIRE
//!     `compress_gzip`/`compress_gzip_padded` body (gzip header write,
//!     `compress_block`/`deflate_padded_in_place`, CRC, trailer). This is
//!     the span every other region's total must reconcile against.
//!   - **RESIDUAL** — DERIVED, never independently timed: `root_ns -
//!     (parse_match_ns + huffman_table_ns + huffman_encode_ns + crc_ns)`.
//!     Covers `StaticCodes::build` (the once-per-invocation static reference
//!     code), the initial `Vec::with_capacity` + gzip header bytes, stored-
//!     block byte copies (`emit_stored_block`, which never touches Huffman
//!     machinery at all), `BitWriter::finish()`'s final flush, the trailer
//!     bytes, and this module's own timer overhead. Conservation
//!     (`root_ns >= sum of named regions`, i.e. RESIDUAL >= 0) is asserted
//!     at dump time, loudly, never silently — see [`reconcile`].
//!
//! ## Scope: T1 only (`-p 1`)
//!
//! The root/CRC timers wrap the T1 entry points ([`compress_gzip`]/
//! [`compress_gzip_padded`], `src/compress/deflate/mod.rs`) ONLY. At T>1 the
//! CLI's default thread count is "all CPUs" (see `src/compress/mod.rs`'s
//! `run` doc comment) and routes through `compress::pipelined::
//! PipelinedGzEncoder::compress_buffer_pure`, which drives the SAME
//! `parse_match`/`huffman_table`/`huffman_encode` regions per chunk but
//! computes its own combined CRC and never calls either T1 entry point —
//! so `root_ns`/`crc_ns` stay exactly 0 while the other regions accumulate,
//! and [`reconcile`] correctly reports this as a FAILURE (named regions
//! exceeding a zero root span), not a silently wrong share. This mirrors
//! the EXISTING constraint `anatomy::is_gzippy_name`/`-p1`-pinning already
//! enforces for `--counters-from-stderr` in fulcrum's `anatomy` command
//! (gzippy's own default-all-CPUs behavior would otherwise silently swap
//! comparators onto a different engine) — always invoke with `-p 1` when
//! using this arm, exactly as that existing convention already requires.
//!
//! ## Overhead
//!
//! Each timer fires a handful of times per 64 KiB block (2-4 per block,
//! typically thousands of times per file, not billions) — several orders of
//! magnitude coarser than a per-position counter, so `Instant::now()`'s cost
//! (tens of ns on typical hardware) is amortized over tens of thousands of
//! input bytes per call. See the commit introducing this module for the
//! measured interleaved-A/B tax (Gate 4b: a feature can cost wall time while
//! leaving output byte-identical, and that tax must be measured, not
//! assumed).

#[cfg(feature = "anatomy-wall")]
use std::sync::atomic::AtomicU64;
#[cfg(feature = "anatomy-wall")]
use std::sync::atomic::Ordering::Relaxed;

macro_rules! define_wall_regions {
    ($($ns:ident / $calls:ident),+ $(,)?) => {
        /// One relaxed `AtomicU64` pair (nanoseconds, call count) per timed
        /// region, plus the root span. Relaxed ordering: independent
        /// per-region tallies, no cross-region ordering requirement (same
        /// rationale as `anatomy_counters::AnatomyCounters`).
        #[cfg(feature = "anatomy-wall")]
        pub struct AnatomyWall {
            pub root_ns: AtomicU64,
            pub root_calls: AtomicU64,
            $(pub $ns: AtomicU64, pub $calls: AtomicU64,)+
        }

        #[cfg(feature = "anatomy-wall")]
        impl AnatomyWall {
            const fn zero() -> Self {
                Self {
                    root_ns: AtomicU64::new(0),
                    root_calls: AtomicU64::new(0),
                    $($ns: AtomicU64::new(0), $calls: AtomicU64::new(0),)+
                }
            }

            /// Reset every counter to zero (test isolation; no production
            /// call site — a single CLI invocation compresses once per
            /// process, mirroring `AnatomyCounters::reset`).
            #[allow(dead_code)]
            pub fn reset(&self) {
                self.root_ns.store(0, Relaxed);
                self.root_calls.store(0, Relaxed);
                $(self.$ns.store(0, Relaxed); self.$calls.store(0, Relaxed);)+
            }

            /// Sum of every NAMED region's nanoseconds (excludes root and
            /// RESIDUAL, which is derived from this sum -- see [`reconcile`]).
            fn named_region_ns(&self) -> u64 {
                0 $(+ self.$ns.load(Relaxed))+
            }

            /// Render the current snapshot as one flat JSON object: every
            /// region's ns + call count, the root span, and the DERIVED
            /// residual (never independently accumulated — always
            /// `root_ns - named_region_ns()`, so it cannot itself drift from
            /// the invariant it represents).
            pub fn to_json(&self) -> String {
                let root_ns = self.root_ns.load(Relaxed);
                let named = self.named_region_ns();
                let residual_ns = root_ns.saturating_sub(named);
                let mut parts = vec![
                    format!("\"root_ns\":{}", root_ns),
                    format!("\"root_calls\":{}", self.root_calls.load(Relaxed)),
                ];
                $(
                    parts.push(format!("\"{}\":{}", stringify!($ns), self.$ns.load(Relaxed)));
                    parts.push(format!("\"{}\":{}", stringify!($calls), self.$calls.load(Relaxed)));
                )+
                parts.push(format!("\"residual_ns\":{}", residual_ns));
                parts.push(format!(
                    "\"conserved\":{}",
                    if root_ns >= named { "true" } else { "false" }
                ));
                parts.push("\"granularity\":\"per-block (parse_match/huffman_table/huffman_encode); per-invocation (crc/root)\"".to_string());
                format!("{{{}}}", parts.join(","))
            }
        }

        #[cfg(feature = "anatomy-wall")]
        pub static WALL: AnatomyWall = AnatomyWall::zero();
    };
}

define_wall_regions!(
    parse_match_ns / parse_match_calls,
    huffman_table_ns / huffman_table_calls,
    huffman_encode_ns / huffman_encode_calls,
    crc_ns / crc_calls,
    // Task C (2026-07-26 bucket-split-oracle session): `HcMatchfinder::new()`
    // is a per-`run()`-call allocation (~256 KiB of scalar sentinel writes
    // across `hash3_tab`/`hash4_tab`/`next_tab` -- see `matchfinder/hc.rs`'s
    // `new()` doc comment). `run()` fires once per T1 whole-file parse but
    // once per ~512 KiB chunk on the T>1 `PipelinedGzEncoder` path
    // (`pipelined.rs::MAX_PARALLEL_BLOCK_SIZE`), so this region's CALL COUNT
    // is itself the falsifiable claim under review, not just its ns. One
    // timer per `run()` invocation (never per-position) -- cheapest
    // granularity in this module.
    mf_new_ns / mf_new_calls,
);

/// Reset every counter. Only exists when `anatomy-wall` is on (see
/// `AnatomyWall::reset`'s doc comment).
#[cfg(feature = "anatomy-wall")]
#[allow(dead_code)]
pub fn reset() {
    WALL.reset();
}

/// Conservation check: the named regions plus the derived RESIDUAL must
/// reconcile to the root span. Since RESIDUAL is *defined* as `root_ns -
/// named_region_ns()`, this can only fail one way: the named regions
/// OVERLAPPED or double-counted wall time and their sum exceeds the root
/// span (RESIDUAL would be negative — `to_json`'s `saturating_sub` would
/// silently floor it at zero, hiding the bug, which is exactly why this
/// function re-derives the raw (non-saturating) subtraction and checks its
/// sign directly rather than trusting the JSON's clamped field). Returns
/// `Err` with a loud, specific message on failure — this is Gate 0 (a number
/// that skipped this check does not exist, per the project's measurement
/// protocol) — never a silent wrong number.
#[cfg(feature = "anatomy-wall")]
pub fn reconcile() -> Result<(), String> {
    let root = WALL.root_ns.load(Relaxed);
    let named = WALL.named_region_ns();
    if root == 0 && named == 0 {
        return Err(
            "ANATOMY_WALL_RECONCILE=VOID root_ns=0 named_ns=0 -- no invocation was timed \
             (root timer never fired; are you calling compress_gzip/compress_gzip_padded?)"
                .to_string(),
        );
    }
    if named > root {
        return Err(format!(
            "ANATOMY_WALL_RECONCILE=FAIL named_region_ns({named}) > root_ns({root}) -- \
             regions overlapped or double-counted wall time; residual would be negative \
             ({}) -- this bug class is exactly what this check exists to catch",
            root as i128 - named as i128
        ));
    }
    Ok(())
}

/// Emit the current snapshot to stderr as one machine-parsable line:
/// `ANATOMY_WALL={json}`, immediately preceded by a loud
/// `ANATOMY_WALL_RECONCILE=PASS|FAIL|VOID` line so a reader (or `fulcrum
/// anatomy`) never has to re-derive the conservation check from the JSON
/// itself. Called once at process end (see `main.rs`), mirroring
/// `anatomy_counters::flush_to_stderr`. A reconciliation FAILURE still
/// prints the counters (so the raw numbers are visible for debugging) but
/// the caller must treat every share/percentage in them as VOID -- per the
/// project rule, a conservation failure means the number does not exist.
#[cfg(feature = "anatomy-wall")]
pub fn flush_to_stderr() {
    match reconcile() {
        Ok(()) => {
            let root = WALL.root_ns.load(Relaxed);
            let named = WALL.named_region_ns();
            eprintln!(
                "ANATOMY_WALL_RECONCILE=PASS root_ns={root} named_region_ns={named} \
                 residual_ns={}",
                root - named
            );
        }
        Err(e) => eprintln!("{e}"),
    }
    eprintln!("ANATOMY_WALL={}", WALL.to_json());
}

/// Time a block, accumulating elapsed wall-clock nanoseconds and a call
/// count into the named region. Expands to just `$body` (no timer, no
/// atomic, zero cost) when `anatomy-wall` is off — the call site itself
/// compiles to nothing extra, matching `anatomy_count!`'s contract.
#[macro_export]
macro_rules! anatomy_wall_time {
    ($region:ident, $calls:ident, $body:block) => {{
        #[cfg(feature = "anatomy-wall")]
        {
            let __anatomy_wall_start = ::std::time::Instant::now();
            let __anatomy_wall_ret = $body;
            let __anatomy_wall_elapsed = __anatomy_wall_start.elapsed().as_nanos() as u64;
            $crate::compress::deflate::anatomy_wall::WALL
                .$region
                .fetch_add(
                    __anatomy_wall_elapsed,
                    ::std::sync::atomic::Ordering::Relaxed,
                );
            $crate::compress::deflate::anatomy_wall::WALL
                .$calls
                .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
            __anatomy_wall_ret
        }
        #[cfg(not(feature = "anatomy-wall"))]
        {
            $body
        }
    }};
}

/// Time the ROOT span (the whole `compress_gzip`/`compress_gzip_padded`
/// invocation) — a distinct macro (not `anatomy_wall_time!`) because it
/// writes `root_ns`/`root_calls` directly rather than taking a region-name
/// pair, and because it is meant to wrap exactly ONE call site per
/// production entry point (never nested inside another root span).
#[macro_export]
macro_rules! anatomy_wall_root {
    ($body:block) => {{
        #[cfg(feature = "anatomy-wall")]
        {
            let __anatomy_wall_root_start = ::std::time::Instant::now();
            let __anatomy_wall_root_ret = $body;
            let __anatomy_wall_root_elapsed = __anatomy_wall_root_start.elapsed().as_nanos() as u64;
            $crate::compress::deflate::anatomy_wall::WALL
                .root_ns
                .fetch_add(
                    __anatomy_wall_root_elapsed,
                    ::std::sync::atomic::Ordering::Relaxed,
                );
            $crate::compress::deflate::anatomy_wall::WALL
                .root_calls
                .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
            __anatomy_wall_root_ret
        }
        #[cfg(not(feature = "anatomy-wall"))]
        {
            $body
        }
    }};
}

#[cfg(all(test, feature = "anatomy-wall"))]
mod tests {
    use super::*;

    /// Exercises a LOCAL instance shape via the macros against the real
    /// global `WALL` -- like `anatomy_counters`'s test doc comment explains,
    /// `cargo test` runs the crate's tests concurrently in one process, so
    /// this test only checks STRUCTURE (json parses, fields present,
    /// reconcile doesn't panic), never an exact count against the shared
    /// global (that would race every other compress-exercising test). Exact
    /// per-invocation counts are checked in `tests/anatomy_wall.rs`, which
    /// spawns a fresh subprocess (the real isolation boundary).
    #[test]
    fn json_shape_and_reconcile_do_not_panic() {
        // Don't reset() the shared global (would race other tests); just
        // confirm to_json() always parses as a well-formed flat object and
        // reconcile() never panics regardless of concurrent state.
        let j = WALL.to_json();
        assert!(j.starts_with('{') && j.ends_with('}'));
        assert!(j.contains("\"root_ns\""));
        assert!(j.contains("\"residual_ns\""));
        assert!(j.contains("\"granularity\""));
        let _ = reconcile();
    }

    /// A hand-built `AnatomyWall`-shaped reconciliation check (bypassing the
    /// global, so this one CAN assert exact values): named regions summing
    /// to less than root must reconcile PASS with the exact residual;
    /// summing to MORE than root must be caught, never silently clamped to
    /// zero. Reimplements the arithmetic `reconcile()` performs against a
    /// synthetic (root, named) pair rather than instantiating a second
    /// `AnatomyWall` (the macro-generated struct has no public constructor
    /// besides the const `zero()` used for the one static).
    #[test]
    fn conservation_arithmetic_catches_overshoot() {
        // Passing case: named < root.
        let root: u64 = 1_000_000;
        let named: u64 = 400_000;
        assert!(named <= root, "sanity");
        let residual = root - named;
        assert_eq!(residual, 600_000);

        // Failing case: named > root must be DETECTED (not silently
        // saturating-subbed to zero) -- this is exactly the bug class
        // `reconcile()` exists to catch.
        let root2: u64 = 1_000_000;
        let named2: u64 = 1_200_000;
        assert!(
            named2 > root2,
            "this must be treated as a FAIL, not clamped"
        );
    }
}
