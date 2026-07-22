//! Semantic-work-unit counters for `fulcrum anatomy`'s execution level.
//!
//! Cargo feature `anatomy-counters`, DEFAULT OFF. When off, every counter,
//! the static table, and every call site collapses to nothing (see
//! `crate::anatomy_count!`'s expansion) — the default build carries zero
//! bytes of this module in the binary, matching the repo's existing
//! measurement-feature precedent (`coz`, `phase-timing`-style probes in
//! `src/decompress/parallel/`; `MARKER_PIPELINE_RUNS` for the always-on
//! decode-side counter this mirrors on the compress side).
//!
//! This is the gzippy-side half of the calibration recipe fulcrum's
//! `src/anatomy/exec.rs` module doc lays out (`## SPEC: gzippy-side
//! anatomy-counters`, fulcrum @ `dc89bc3`): whole-program cachegrind Ir-share
//! attribution is a HYPOTHESIS-tier signal (a probe that misses after one
//! compare and one that walks a long chain land in the same instruction
//! bucket). These counters are the EXACT execution-side ground truth that
//! closes that gap for gzippy specifically — every count below is a single
//! semantic event (a chain-node compare, a table read/write, a token pushed),
//! not an instruction tally.
//!
//! ## Overhead: batch the hot chain-walk into one flush per call
//!
//! An atomic `fetch_add` at every visited chain node in `hc.rs::longest_match`
//! (one per `hc_probe_attempts`/`hc_probe_outcome_*`/`hc_chain_table_reads`
//! event — O(`max_search_depth`) events per call at L6-L9, where chains run
//! deep) measured at ~10-14% wall overhead on `dickens` (hyperfine, T1,
//! interleaved, `/dev/null` sink) before batching. `HcLocalCounters`
//! accumulates the SAME events in plain locals through the whole call and
//! flushes with ONE `fetch_add` per counter at the single return point —
//! same exact final counts (pure batching, not a different count), overhead
//! now within run-to-run noise (measured ~0-1%, i.e. a tie) at L1/L6/L9 on
//! the same corpus. `bt.rs`'s per-iteration atomics were measured WITHOUT
//! this batching (L12, `dd79_text6`) and also came in within noise (~0-1%)
//! — the near-optimal parse's own cost model dominates there, so the same
//! batching was not (yet) applied to `bt.rs`; revisit if a future corpus/
//! level shows otherwise.
//!
//! ## Counter → call-site map (spec deviations noted inline where the spec's
//! literal citation didn't survive contact with the actual call graph)
//!
//! - **`hc_*`** — `matchfinder/hc.rs`'s `HcMatchfinder::longest_match` /
//!   `skip_bytes`, faithfully at the spec's cited sites.
//! - **`bt_*`** — `matchfinder/bt.rs`'s `BtMatchfinder::advance::<REC>` /
//!   `skip_byte`. The spec left the exact bt compare/extend sites to the
//!   worker ("cite the specific compare/extend call once located"); see the
//!   doc comments at each call site in `bt.rs` for the interpretation chosen
//!   (the tree-descent shape doesn't map onto hc's simple probe/miss/accept
//!   model 1:1, so `bt_probe_outcome_*` buckets the single-byte equality
//!   check + `lz_extend` pair, not a 4-byte prefilter compare).
//! - **`literals_emitted`/`matches_emitted`(`_fast`)/`histogram_updates`** —
//!   `parse/mod.rs`'s `Sink::push_{literal,match}[_fast]`, exactly as spec'd.
//! - **`block_split_observations`** — `block_split.rs`'s
//!   `BlockSplitStats::observe_{literal,match}`, exactly as spec'd.
//! - **`blocks_emitted_{fixed,dynamic}`** — `parse/mod.rs`'s `emit_block` /
//!   `emit_block_static_or_stored`, exactly as spec'd (`fixed` = the DEFLATE
//!   BTYPE=01 "fixed Huffman" block, gzippy's `static_bits` branch).
//!   **`blocks_emitted_stored` DEVIATES from the spec's cited site**: a
//!   first cut counted it inside `emit_block`/`emit_block_static_or_stored`'s
//!   "stored wins" branches (as spec'd) and UNDERCOUNTED — the closed-loop
//!   check against fulcrum's token-level block count (`fulcrum anatomy
//!   --counters-from-stderr`, `dd79_text6` L1) found 15 stored blocks at the
//!   token level vs 0 executed here, because the T>1 pipelined path
//!   (`compress::pipelined`) emits its OWN per-chunk sync-flush stored
//!   block directly from `deflate::deflate_into` (`deflate/mod.rs`),
//!   bypassing the parser's cost-comparison branches entirely — so THIS
//!   counter never fired for them. Moved to `deflate/mod.rs`'s
//!   `write_stored_subblock` — the ONE physical-BTYPE=00-block emission
//!   site every stored block (parser-chosen OR sync-flush OR the
//!   empty-input special case) funnels through, including the sub-block
//!   SPLIT for inputs >65535 bytes (each physical sub-block is its own
//!   count, matching how the token-level side counts them) — removed from
//!   the two `emit_block*` call sites to avoid double-counting through
//!   `emit_stored_block`. See that function's doc comment for the full
//!   account; this is the ONE real bug this counter-building exercise
//!   caught, and catching it is the point of building the closed loop.
//! - **`huffman_tree_nodes_visited`/`huffman_length_limited_calls`** —
//!   `huffman/optimal.rs`'s `boundary_pm`/`boundary_pm_final`/
//!   `length_limited_code_lengths`, exactly as spec'd. **DEVIATION (load-
//!   bearing, not cosmetic):** `length_limited_code_lengths` is the EXACT
//!   package-merge builder used only by `parse::ultra` (the crown engine,
//!   `-F`/`-I`/`-J`). The ordinary greedy/lazy/fast/near_optimal path that
//!   `emit_block` calls for EVERY block (dynamic candidate always built for
//!   the stored/static/dynamic cost comparison, win or lose) uses the
//!   APPROXIMATE builder `huffman::fast::make_huffman_code` instead — so on
//!   the mission's own L1 text6 target, `huffman_length_limited_calls` is
//!   always 0 and the spec's "`2 * blocks_emitted_dynamic`" cross-check is
//!   vacuous for that path. Added `huffman_make_code_calls` (see below) so
//!   the mission's closed-loop reconciliation has a non-vacuous huffman-build
//!   counter on the actual target corpus/level; `huffman_length_limited_calls`
//!   is kept for when `parse::ultra` is exercised.
//! - **`huffman_make_code_calls`** — ADDED (not in the spec's literal list):
//!   `huffman::fast::make_huffman_code`'s call count. `emit_block` calls it
//!   twice per invocation (litcode, offcode) for the dynamic-candidate cost
//!   probe regardless of which block type wins, `build_dynamic_header` calls
//!   it once more (the 19-symbol precode), and `StaticCodes::build` calls it
//!   twice at parse start (the static-Huffman reference code). See the
//!   reconciliation test in this module for the exact expected count.
//! - **`match_length_bytes_total`** — ADDED (not in the spec's literal list):
//!   sum of match lengths pushed via `push_match`/`push_match_fast`. Needed
//!   for the "tokens emitted == extract count" invariant the mission asks
//!   for: `literals_emitted + match_length_bytes_total == total input bytes
//!   parsed` (every input byte is covered by exactly one literal or exactly
//!   one position of exactly one match — the LZ77 parse invariant).
//! - **`alloc_events`/`alloc_bytes`** — the `Vec::with_capacity` sites in
//!   `deflate/mod.rs` (`:56` `compress_oneshot`, `:123`/`:133` the two padded-
//!   working-buffer shapes, `:195`/`:218` the two gzip-wrapper output
//!   buffers) and `huffman/header.rs:129` (combined code-length buffer).
//!   **DEVIATION:** the spec's `deflate/mod.rs:273`/`:294` sites are inside
//!   `#[cfg(test)] mod streaming_tests`'s `mixed_corpus`/`wrap_gzip` test
//!   helpers (the spec's line numbers land there at HEAD) — never compiled
//!   into a production/release build, so instrumenting them would count
//!   test-fixture allocations, not production ones. Skipped; documented here
//!   rather than silently dropped.

#[cfg(feature = "anatomy-counters")]
use std::sync::atomic::AtomicU64;
#[cfg(feature = "anatomy-counters")]
use std::sync::atomic::Ordering::Relaxed;

macro_rules! define_counters {
    ($($name:ident),+ $(,)?) => {
        /// One relaxed `AtomicU64` per semantic-work-unit counter. Relaxed
        /// ordering is sufficient (and intentional): these are independent
        /// per-event tallies with no cross-counter ordering requirement, and
        /// counting must not itself perturb what it counts materially — a
        /// fence/acquire-release protocol would add cross-core synchronization
        /// cost the spec explicitly warns against ("<10% overhead").
        #[cfg(feature = "anatomy-counters")]
        pub struct AnatomyCounters {
            $(pub $name: AtomicU64,)+
        }

        #[cfg(feature = "anatomy-counters")]
        impl AnatomyCounters {
            const fn zero() -> Self {
                Self { $($name: AtomicU64::new(0),)+ }
            }

            /// Reset every counter to zero (test isolation across cases in the
            /// same process). No production call site today (a single CLI
            /// invocation compresses once per process); kept as public API for
            /// this module's own tests and any future harness that drives
            /// multiple compressions per process (in-process fuzz/bench loops).
            #[allow(dead_code)]
            pub fn reset(&self) {
                $(self.$name.store(0, Relaxed);)+
            }

            /// Render the current snapshot as one flat JSON object, field
            /// names matching this struct 1:1 (fulcrum anatomy's declared
            /// output shape).
            pub fn to_json(&self) -> String {
                let parts = [
                    $(format!("\"{}\":{}", stringify!($name), self.$name.load(Relaxed)),)+
                ];
                format!("{{{}}}", parts.join(","))
            }
        }

        #[cfg(feature = "anatomy-counters")]
        pub static COUNTERS: AnatomyCounters = AnatomyCounters::zero();
    };
}

define_counters!(
    // Hash-chain matchfinder (matchfinder/hc.rs).
    hc_probe_attempts,
    hc_probe_outcome_miss,
    hc_probe_outcome_too_short,
    hc_probe_outcome_accepted,
    hc_hash_computations,
    hc_head_table_reads,
    hc_head_table_writes,
    hc_chain_table_reads,
    hc_positions_skipped,
    // Binary-tree matchfinder (matchfinder/bt.rs).
    bt_probe_attempts,
    bt_probe_outcome_miss,
    bt_probe_outcome_too_short,
    bt_probe_outcome_accepted,
    bt_hash_computations,
    bt_head_table_reads,
    bt_head_table_writes,
    bt_child_table_reads,
    bt_child_table_writes,
    bt_positions_skipped,
    // Token emission / histogram updates (parse/mod.rs Sink).
    literals_emitted,
    literals_emitted_fast,
    matches_emitted,
    matches_emitted_fast,
    histogram_updates,
    match_length_bytes_total,
    // Block-split observations (block_split.rs).
    block_split_observations,
    // Block emission (parse/mod.rs).
    blocks_emitted_stored,
    blocks_emitted_fixed,
    blocks_emitted_dynamic,
    // Huffman table build (huffman/optimal.rs, huffman/fast.rs).
    huffman_tree_nodes_visited,
    huffman_length_limited_calls,
    huffman_make_code_calls,
    // Allocation events (deflate/mod.rs, huffman/header.rs).
    alloc_events,
    alloc_bytes,
);

/// Reset every counter. Only exists when `anatomy-counters` is on. No
/// production call site (see `AnatomyCounters::reset`'s doc); kept for this
/// module's own tests and any future multi-run-per-process harness.
#[cfg(feature = "anatomy-counters")]
#[allow(dead_code)]
pub fn reset() {
    COUNTERS.reset();
}

/// Emit the current counter snapshot to stderr as one machine-parsable line:
/// `ANATOMY_COUNTERS={json}`. Called once at process end (see `main.rs`).
/// Only exists when `anatomy-counters` is on (no env var, no behavior change
/// — consistent with "NO env vars in the production path": this entire
/// module does not exist in a production/default build).
#[cfg(feature = "anatomy-counters")]
pub fn flush_to_stderr() {
    eprintln!("ANATOMY_COUNTERS={}", COUNTERS.to_json());
}

/// Increment a named counter by `$n` (or by 1 if omitted). Expands to nothing
/// when `anatomy-counters` is off — not a runtime branch, the call site
/// itself compiles to zero bytes, matching the `#[cfg(feature = "counters")]`
/// pattern this repo already uses for decode-side hot-path counters.
#[macro_export]
macro_rules! anatomy_count {
    ($name:ident) => {
        $crate::anatomy_count!($name, 1u64)
    };
    ($name:ident, $n:expr) => {
        #[cfg(feature = "anatomy-counters")]
        {
            $crate::compress::deflate::anatomy_counters::COUNTERS
                .$name
                .fetch_add(($n) as u64, ::std::sync::atomic::Ordering::Relaxed);
        }
    };
}

#[cfg(all(test, feature = "anatomy-counters"))]
mod tests {
    use super::*;

    /// Exercises `AnatomyCounters::to_json`/`reset` on a LOCAL instance (never
    /// the global `COUNTERS` static): `cargo test` runs every test in this
    /// binary concurrently by default, and any other test module that
    /// exercises compression (the `matchfinder`/`parse` proptests etc.) is
    /// simultaneously incrementing the real `COUNTERS` in another thread — a
    /// unit test asserting an EXACT count against the shared global would be
    /// flaky by construction. See `tests/anatomy_counters.rs` (an integration
    /// test that spawns the real CLI binary as a fresh subprocess) for the
    /// reconciliation invariants against a real compression: process
    /// isolation, not a mutex, is the actual fix — a mutex only serializes
    /// tests IN this module, not every other module that also touches the
    /// one process-wide static.
    #[test]
    fn json_round_trips_shape() {
        let c = AnatomyCounters::zero();
        c.hc_probe_attempts.fetch_add(3, Relaxed);
        c.literals_emitted.fetch_add(1, Relaxed);
        let json = c.to_json();
        assert!(json.contains("\"hc_probe_attempts\":3"));
        assert!(json.contains("\"literals_emitted\":1"));
        c.reset();
        assert!(c.to_json().contains("\"hc_probe_attempts\":0"));
    }
}
