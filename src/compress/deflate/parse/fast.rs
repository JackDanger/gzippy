//! Level-0/1 igzip-class one-pass FAST parser.
//!
//! Shared by `Strategy::Fast0` (L0) and `Strategy::Fast` (L1): both use the
//! IDENTICAL chainless single-probe matchfinder below; they differ only in
//! `use_dynamic` at the block-emit step (L1 evaluates a per-block dynamic
//! Huffman code, L0 does not — see `super::emit_block_static_or_stored`'s doc
//! comment for why that is the L0/L1 cost/ratio trade).
//!
//! The match finder is a port of igzip's level-0/1 deflate body
//! (`vendor/isa-l/igzip/igzip_base.c:isal_deflate_body_base`, :27-113): a
//! chainless single-probe hash with `LIMIT_HASH_UPDATE`, the igzip-class speed
//! lever. Where this path DIFFERS from the greedy/lazy/near-optimal parsers is
//! only in the finder — it does one probe per position, no chains, no depth
//! loop — NOT in how blocks are coded.
//!
//! For coding it reuses the SHARED L2-9 backend ([`Sink`] + [`emit_block`]):
//! tokens stream into the sequence buffer while litlen + offset frequency
//! histograms accumulate as-you-go (no extra pass), a block is flushed every
//! [`FAST_BLOCK_LENGTH`] input bytes (and at end-of-input), and each flush picks
//! the cheapest of a per-block DYNAMIC Huffman code, the RFC-1951 static code, or
//! a STORED block. The old fast path direct-emitted one whole-input static block:
//! fast, but 1.2-1.3x larger than `gzip -1` on some corpora, and it could not
//! escape to STORED on incompressible data. Per-block dynamic Huffman + the
//! stored escape recover competitive ratio while the single-probe finder keeps
//! the speed.
//!
//! The mechanisms ported from igzip (finder only):
//!   1. **Chainless single-probe hash** (`igzip_base.c:60-64`): one head table
//!      storing the last position per hash; overwrite on collision; ONE
//!      candidate per position — no chains, no depth loop. **DEVIATION from
//!      igzip:** the table is `1 << 16` (64K) slots, not igzip's `1 << 13` (8K).
//!      Because there is only ONE probe, a wider table is the cheapest way to
//!      cut hash collisions — the single candidate is far less often an
//!      unrelated position — and it closes the last ~1% ratio gap vs `pigz -1`
//!      on `text`/`bin` at ~zero speed cost (still one load + one compare per
//!      position). See [`HASH_BITS`].
//!   2. **LIMIT_HASH_UPDATE** (`igzip_base.c:71-86`): over an accepted match,
//!      insert the hash for only the first ~3 interior positions, then jump the
//!      cursor by the whole match length (skip the interior stores).
//!   3. **compare258 match-extend** (`huffman.h:260-314`): 8-byte XOR +
//!      trailing-zero-count, reusing Increment 1's [`lz_extend`].
//!
//! Block emission (mechanisms 4-5: full-length-codeword LUT, 4-literals-per-
//! flush, branchless word-store FLUSH, cheapest-of-{dynamic,static,stored}) all
//! lives in the shared [`emit_block`]/`emit_tokens` machinery — this file does
//! not duplicate it.

use super::super::bitstream::BitWriter;
use super::super::matchfinder::common::{load_u32, lz_extend, lz_hash, prefetch_write};
use super::super::tables::DEFLATE_MAX_MATCH_LEN;
use super::{bsr32, emit_block, emit_block_static_or_stored, Sink, StaticCodes};

/// Per-`run()`-call local accumulator for the fast-path anatomy counters
/// (`anatomy-counters` feature only; see `anatomy_counters.rs`'s `fast_*` doc
/// comment for the full accounting story). Unlike `hc.rs`'s `HcLocalCounters`
/// (which lives entirely inside one function, `longest_match`), this finder's
/// hot logic spans `process_position_l1`/`fastloop_l0`/`fastloop_l1` and
/// `run`'s own tail/seed loops — so one instance is created in `run`, threaded
/// through as an extra parameter present ONLY when the feature is on
/// (`#[cfg(feature = "anatomy-counters")]` on the parameter declaration and on
/// the argument at every call site — a true zero-parameter signature when
/// off, not an unused one), and flushed once at `run`'s return, after every
/// internal block the call covers.
///
/// Two fields are DERIVED sums rather than directly-tracked totals — a
/// second overhead pass (hyperfine, M1 Pro, `/dev/null`, interleaved, L1)
/// found the direct-tracking version cost `bin6` ~13% (vs a ~6% PRE-EXISTING
/// baseline from the Sink's own un-batched `push_*_fast` atomics — see the
/// commit message for the full before/after table), and both derivations
/// remove an unconditional field write from the hottest (most frequent)
/// per-position branches without losing any information, because each is a
/// genuine structural identity of this parser, not a guess:
///
/// - `fast_positions_processed = positions_processed_matches +
///   probe_outcome_miss + probe_outcome_too_short + probe_outcome_deferred +
///   no_probe_literals`: every miss/too-short/deferred outcome consumes
///   EXACTLY one byte (by construction — it is coded as a single literal),
///   and `no_probe_literals` covers the one case with no outcome bucket at
///   all (the tail loop's near-EOF `max_len < SHORTEST_MATCH`, where no
///   probe is attempted). This is true across `process_position_l1`,
///   `fastloop_l0`, and the tail loop uniformly, so no site needs its own
///   unconditional `+= 1`.
/// - `fast_head_table_writes = probe_attempts + interior_writes +
///   dict_seed_writes`: the primary `head[h] = pos` insert is unconditional
///   at the top of every REAL probe (never at a no-probe near-EOF literal)
///   and fires EXACTLY once per probe attempt, so `probe_attempts` (already
///   tracked for its own sake) already equals that contribution exactly;
///   `interior_writes` is the separate LIMIT_HASH_UPDATE contribution
///   (accepted matches only) and `dict_seed_writes` the preset-dictionary
///   seed loop's (outside the coded span entirely).
#[cfg(feature = "anatomy-counters")]
#[derive(Default)]
struct FastLocalCounters {
    positions_processed_matches: u64,
    no_probe_literals: u64,
    positions_skipped: u64,
    hash_computations: u64,
    head_table_reads: u64,
    interior_writes: u64,
    dict_seed_writes: u64,
    probe_attempts: u64,
    probe_outcome_miss: u64,
    probe_outcome_too_short: u64,
    probe_outcome_accepted: u64,
    probe_outcome_deferred: u64,
    lazy_peek_events: u64,
    lazy_peek_defers: u64,
    k2_batch_iterations: u64,
}

#[cfg(feature = "anatomy-counters")]
impl FastLocalCounters {
    #[inline(always)]
    fn flush(self) {
        crate::anatomy_count!(
            fast_positions_processed,
            self.positions_processed_matches
                + self.probe_outcome_miss
                + self.probe_outcome_too_short
                + self.probe_outcome_deferred
                + self.no_probe_literals
        );
        crate::anatomy_count!(fast_positions_skipped, self.positions_skipped);
        crate::anatomy_count!(fast_hash_computations, self.hash_computations);
        crate::anatomy_count!(fast_head_table_reads, self.head_table_reads);
        crate::anatomy_count!(
            fast_head_table_writes,
            self.probe_attempts + self.interior_writes + self.dict_seed_writes
        );
        crate::anatomy_count!(fast_probe_attempts, self.probe_attempts);
        crate::anatomy_count!(fast_probe_outcome_miss, self.probe_outcome_miss);
        crate::anatomy_count!(fast_probe_outcome_too_short, self.probe_outcome_too_short);
        crate::anatomy_count!(fast_probe_outcome_accepted, self.probe_outcome_accepted);
        crate::anatomy_count!(fast_probe_outcome_deferred, self.probe_outcome_deferred);
        crate::anatomy_count!(fast_lazy_peek_events, self.lazy_peek_events);
        crate::anatomy_count!(fast_lazy_peek_defers, self.lazy_peek_defers);
        crate::anatomy_count!(fast_k2_batch_iterations, self.k2_batch_iterations);
    }
}

/// Env-var-overridable runtime knobs for the L1-band ratio-close-out config-
/// space search (2026-07-22 campaign; `l1-tune` Cargo feature, OFF by
/// default). Every field defaults to the EXISTING shipped const (see each
/// field's paired const below) so a feature-on build with NO env vars set
/// behaves identically to a feature-off build; `examples/l1_search.rs` sweeps
/// these via env vars across many process invocations (no rebuild per
/// candidate — the whole point of threading them as runtime values instead of
/// consts). Consumed ONLY by [`process_position_l1`]/[`fastloop_l1`]/[`run`]
/// when `l1-tune` is compiled in; the default build has none of this code.
#[cfg(feature = "l1-tune")]
pub mod tune {
    use std::sync::{OnceLock, RwLock};

    #[derive(Clone, Copy, Debug)]
    pub struct L1Tune {
        /// Overrides [`super::LAZY_PEEK_MAX_LEN`].
        pub lazy_peek_max_len: u32,
        /// Overrides [`super::LAZY_PEEK_MIN_DIST`].
        pub lazy_peek_min_dist: usize,
        /// Overrides [`super::LIMIT_HASH_UPDATE_INSERTS_L1`].
        pub insert_depth: usize,
        /// Overrides [`super::FAST_BLOCK_LENGTH`].
        pub block_length: usize,
        /// New lever (not in the shipped path): a conditional second-bucket
        /// probe (`head2`, one generation behind `head`) consulted ONLY when
        /// the primary probe already produced an ACCEPTED match no longer
        /// than `bucket2_gate_max_len` — the "short-match acceptance" gate
        /// named in the L1-band mission brief. Off by default (`false`);
        /// when on, replaces the accepted match with the second candidate's
        /// match iff it is both valid and strictly longer.
        pub bucket2_enabled: bool,
        /// Gate paired with `bucket2_enabled` (see its doc comment).
        pub bucket2_gate_max_len: u32,
    }

    fn env_u32(name: &str, default: u32) -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    fn env_bool(name: &str, default: bool) -> bool {
        std::env::var(name)
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(default)
    }

    impl L1Tune {
        fn from_env() -> Self {
            L1Tune {
                lazy_peek_max_len: env_u32(
                    "GZIPPY_L1TUNE_LAZY_PEEK_MAX_LEN",
                    super::LAZY_PEEK_MAX_LEN,
                ),
                lazy_peek_min_dist: env_usize(
                    "GZIPPY_L1TUNE_LAZY_PEEK_MIN_DIST",
                    super::LAZY_PEEK_MIN_DIST,
                ),
                insert_depth: env_usize(
                    "GZIPPY_L1TUNE_INSERT_DEPTH",
                    super::LIMIT_HASH_UPDATE_INSERTS_L1,
                ),
                block_length: env_usize("GZIPPY_L1TUNE_BLOCK_LENGTH", super::FAST_BLOCK_LENGTH),
                bucket2_enabled: env_bool("GZIPPY_L1TUNE_BUCKET2", false),
                bucket2_gate_max_len: env_u32("GZIPPY_L1TUNE_BUCKET2_GATE_MAX_LEN", 8),
            }
        }
    }

    fn cell() -> &'static RwLock<L1Tune> {
        static CELL: OnceLock<RwLock<L1Tune>> = OnceLock::new();
        CELL.get_or_init(|| RwLock::new(L1Tune::from_env()))
    }

    /// Current tune parameters (env-var defaults unless overridden by
    /// [`set`] in this process). `L1Tune` is `Copy`, so this is cheap: one
    /// `RwLock::read` + a struct copy, called once per `run()` call (not per
    /// position).
    pub fn get() -> L1Tune {
        *cell().read().unwrap()
    }

    /// Override the tune parameters for every subsequent `run()` call in
    /// THIS process. Search-only API: `examples/l1_search.rs` sweeps configs
    /// by calling this between candidates — one process, no rebuild and no
    /// respawn per candidate (the env-var path alone can't do this: it is
    /// read once and cached, by design, so a single process can't change it
    /// via `std::env::set_var` after the first `get()`).
    pub fn set(t: L1Tune) {
        *cell().write().unwrap() = t;
    }
}

/// Log2 of the head-table size. igzip's level-0 hash table is
/// `IGZIP_LVL0_HASH_SIZE = 8 * 1024 = 1 << 13` (`igzip_lib.h:121-125`); we widen
/// it to `1 << 16` (64K) because the finder is single-probe — a wider table
/// spreads the 4-byte keys over 8× more slots, so the ONE candidate we keep per
/// hash is far less likely to be an unrelated collision, and it recovers the
/// last ~1% ratio gap vs pigz-1 on `text`/`bin` at near-zero speed cost (same
/// one load, one compare per position). See Lever 1.
const HASH_BITS: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;

/// Software-pipeline distance for the head-table prefetch (SF1-C). Each fastloop
/// iteration prefetches the head slot for the position it will probe `PF_DIST`
/// steps ahead, so the dependent `head[h]` load — cachegrind-named as 69% of the
/// L1 fast path's D1 read misses, and perf-confirmed as the IPC collapse vs igzip
/// on binary data (IPC 1.32 vs 2.46) — is already warm when consumed. Pure hint:
/// it warms a cache line, never changes a value the finder reads, so output stays
/// byte-identical. Same technique as the hc.rs chain-walk prefetch.
const PF_DIST: usize = 4;

/// LIMIT_HASH_UPDATE: number of match-interior positions whose hash is inserted
/// into the head table before the cursor jumps over the rest of the match
/// (`igzip_base.c:71-86`, igzip uses ~3). Denser interior inserts seed more
/// candidates for later matches (better ratio) at the cost of more hash
/// stores; `usize::MAX` means "insert EVERY interior position" (zlib-ng
/// style). Passed as a runtime `run()` parameter (not a `const`) because L0
/// and L1 want DIFFERENT values sharing the SAME finder code: bumping it
/// helps ratio on both, but the extra hash stores are pure overhead for L0
/// (whose bar is speed + beating igzip -0, already cleared with room to
/// spare) — a shared constant would make L0 pay for an L1-only ratio lever
/// (measured: INSERTS=4 already cost L1 alone ~14.5% wall on `text6` for a
/// ~2.3% ratio gain — igzip's own ~3 turned out to be close to the true
/// speed/ratio knee for THIS finder; higher values were tried and reverted,
/// see the commit message).
pub(super) const LIMIT_HASH_UPDATE_INSERTS_L0: usize = 2;
/// L1 gets one step denser than L0 (matches igzip's own "~3"): measured
/// near-zero wall cost (`text6` 22.9ms→22.6ms, `bin6` 37.7ms→38.3ms, both
/// within run-to-run noise) for a real ratio win (`text6` -1.7%, `bin6`
/// -0.3% vs `LIMIT_HASH_UPDATE_INSERTS_L0`). Higher values (4, 8, MAX) give
/// more ratio but blow well past a 10% L1 wall budget — not shipped, see the
/// commit message for the measured numbers.
pub(super) const LIMIT_HASH_UPDATE_INSERTS_L1: usize = 3;

/// Sentinel head-table entry meaning "no position stored yet". Any position we
/// store is `< in_end <= u32::MAX`, so the sentinel never collides with a real
/// index, and its computed distance always fails the window test.
const NO_POS: u32 = u32::MAX;

/// L0-only search acceleration (the `ACCEL` const generic on [`run`]), an
/// LZ4-`LZ4_compress_fast`-style scan-step ramp: no vendor DEFLATE encoder
/// counterpart, a novel technique for this chainless single-probe finder.
/// The scan step is `1 + (consecutive_misses >> ACCEL_SHIFT)`, capped at
/// `ACCEL_MAX_STEP`: once `ACCEL_ARM_THRESHOLD` consecutive-miss positions
/// have been seen, the per-position hash lookup/insert is skipped for a
/// growing number of subsequent positions — those bytes are coded as literals
/// directly with no finder work at all. Any match resets the miss counter
/// (and so the step) back to 1. This trades some missed matches (ratio) for
/// skipping finder work outright (speed) — but ONLY on long literal runs,
/// which is exactly where L1's exhaustive per-position search is least
/// likely to pay off. `run::<false>` (L1 / `Strategy::Fast`) monomorphizes
/// with this whole mechanism compiled away — this is strictly an L0
/// (`Strategy::Fast0`) lever.
///
/// `ACCEL_SHIFT = 0` (was 1): once armed, the step grows by 1 per additional
/// consecutive miss (not 1 per 2 misses) — the growth-rate half of the ramp
/// was measured to cost ~nothing extra on `text6`/`sil40` once decoupled from
/// the ARM point (see `ACCEL_ARM_THRESHOLD`): `ACCEL_SHIFT=0` alone at the
/// OLD `ACCEL_ARM_THRESHOLD=2` blew the igzip-0 size budget on `text6`
/// (arms too eagerly on a corpus with short natural literal runs), but paired
/// with the higher threshold below it stays under budget on text AND still
/// closes a real chunk of the wall gap on the low-redundancy corpora where the
/// ramp actually gets to run (measured on Apple M1, `-p1 -0`, N=15
/// interleaved /dev/null, ~2026-07 gzippy-encoder campaign): `bin6` wall
/// -7.2% (20.40ms med → 18.94ms), `sil40` wall -4.6% (119.42ms → 113.95ms),
/// `text6` a noise-level tie (27.01ms → 26.98ms, matches are dense enough on
/// text that the ramp rarely arms either way). Sizes stayed within the
/// igzip-0 size-ratio budget on all three corpus classes (see the commit
/// message for the exact numbers) — DIRECTIONAL on this box; re-gate on
/// `scripts/measure.sh` / `fulcrum` before banking as a cross-arch finding.
const ACCEL_SHIFT: u32 = 0;
/// Consecutive-miss count at which the ramp arms (below this, `step` stays 1
/// and the per-literal cost is just the counter increment + one
/// well-predicted comparison — see the arming check's doc comment in
/// [`run`]). Decoupled from `ACCEL_SHIFT` (was `1 << ACCEL_SHIFT` = 2) so the
/// arm point and the post-arm growth rate can be tuned independently —
/// arming too eagerly on COMPRESSIBLE corpora (text/silesia) costs ratio
/// budget for negligible wall gain (most literal runs there are short), so a
/// slightly higher threshold (3, up from the old implied 2) buys back ratio
/// margin at near-zero wall cost while `ACCEL_SHIFT=0` keeps the post-arm
/// ramp aggressive for corpora where it actually engages.
const ACCEL_ARM_THRESHOLD: u32 = 3;
/// Cap on the accelerated scan step (bytes skipped per ramped-up jump).
/// Bounded well under the `DEFLATE_MAX_MATCH_LEN` (258) fastloop safety
/// margin so the skip can never run the cursor past `in_end`. Measured: raising
/// this past 8 (16, 32) gave no further wall win at `ACCEL_ARM_THRESHOLD=3`
/// (the remaining per-skipped-literal cost is the histogram bump / litrun
/// counter, which `ACCEL` does NOT remove — only the matchfinder touch is
/// skipped — so a bigger cap just means a bigger inner copy loop per
/// activation, not less total work); kept at the original value.
const ACCEL_MAX_STEP: usize = 8;

/// igzip `SHORTEST_MATCH` (`huff_codes.h:89`): the fast path only emits matches
/// of length >= 4 (its hash keys 4 bytes), coding anything shorter as literals.
const SHORTEST_MATCH: u32 = 4;

/// L1-only (`ACCEL == false`) one-position lazy peek, gated to short accepted
/// matches at a far distance (see [`LAZY_PEEK_MIN_DIST`]).
///
/// BACKGROUND (L1-ratio-gap campaign, 2026-07): libdeflate's ACTUAL level-1
/// matchfinder (vendor `lib/ht_matchfinder.h`, `HT_MATCHFINDER_BUCKET_SIZE ==
/// 2`) keeps a SECOND candidate per hash slot and takes the better of the
/// two on every lookup — not (as first hypothesized from the ratio-gap
/// frontier analysis) a distance-cost accept/reject threshold, which lives
/// only in libdeflate's HC-based `deflate_compress_greedy` (levels >= 2) and
/// measurably does not fire here: a length-3 candidate from this finder's
/// existing single 4-byte-hash probe is, by direct measurement, almost never
/// a TRUE length-3 match (a length-3 accept/reject rule changed 0-3 bytes
/// across text6/bin6/sil40 — falsified, not shipped).
///
/// A prior gzippy attempt ported the 2-way-bucket idea literally (a packed
/// 2-way head table, second slot probed on EVERY lookup) and recovered real
/// ratio (text6 under the libdeflate-1 target) but cost L1 +33.6% wall on
/// text60 and +24.2% on bin60 (see `git log -1 e0e4c44d`'s commit message) —
/// probing a second candidate on every lookup (hit AND miss) is the tax, and
/// no gating threshold got it below ~15-20%.
///
/// This lever tries the SAME idea far more narrowly: instead of a second
/// candidate at THIS position on every lookup, peek at the SAME single-probe
/// candidate ONE position ahead (`pos + 1`) — but ONLY after a match is
/// already ACCEPTED and it is both short (`length <= LAZY_PEEK_MAX_LEN`) and
/// far (`dist > LAZY_PEEK_MIN_DIST`, see below). Even this narrower gate is
/// NOT cheap: on these corpora, most accepted matches ARE short (confirming
/// the ratio-gap frontier's own finding that most of the byte gap sits in
/// 3-16-byte match decisions), so "peek on every short match" measured
/// +9-15% wall on text6/sil40 (length-only gate, any threshold 4-8) — a
/// SECOND independent confirmation, via a differently-shaped mechanism, of
/// the same wall-cost floor the reverted 2-way-bucket attempt hit. Adding the
/// distance gate cuts the peek rate further (most short matches here are
/// near, not far) at the cost of most of the ratio recovery; the values below
/// are the swept trade point that stays safely under a 5% wall budget
/// (interleaved paired-diff N=25-30 on M1, `/dev/null` sink: text6 ~2.9-4.2%,
/// sil40 ~1.8-3.0%, bin6 ~0.1-1.6%, all within noise-to-small) while still
/// improving size with ZERO regressions on text6/bin6/sil40 (and the 19-file
/// breadth corpus, see the commit message). This does NOT close the L1-vs-
/// libdeflate-1 gap to parity — it closes roughly a tenth to a third of it
/// per corpus; the residual is an HONEST, not a closed, gap (see the commit
/// message for the exact before/after/ld1 numbers). Reuses the SAME cost
/// tie-break as this codebase's existing HC lazy parser (`parse/lazy.rs`'s
/// `better_match`, threshold 2):
/// `4*(next_len-cur_len) + (bsr32(cur_offset)-bsr32(next_offset)) > 2`.
const LAZY_PEEK_MAX_LEN: u32 = 4;

/// Distance gate paired with [`LAZY_PEEK_MAX_LEN`] (see its doc comment for
/// the full story): only peek when the ALREADY-ACCEPTED match's distance
/// exceeds this. A short match at a NEAR distance is already cheap (small
/// distance code, few extra bits) and rarely worth deferring; the expensive
/// case worth the extra probe is a short match at a FAR distance (many
/// distance extra-bits paid for few covered bytes). Swept empirically
/// (0/256/1024/4096/8192/16384): lower values recover more ratio but cost
/// more wall (0 costs +9-15%; 16384 is wall-safe but the smallest ratio
/// recovery); 8192 is the chosen trade point.
const LAZY_PEEK_MIN_DIST: usize = 8192;

/// DEFLATE sliding-window size — the largest legal back-reference distance.
const WINDOW: usize = 32768;

/// Input bytes covered per DEFLATE block in the fast path (L1 value —
/// [`super::compress`] passes a caller-chosen `block_length`, see
/// [`FAST0_BLOCK_LENGTH`] for the L0 value).
///
/// A per-block dynamic Huffman code adapts to LOCAL statistics, so a moderate
/// block (vs one whole-input block) improves ratio on heterogeneous input; at
/// 64 KiB the ~dozens-of-bytes dynamic header amortizes to well under 1%. This
/// is the fast path's one ratio/speed tuning knob — it does not affect
/// correctness (any block boundary roundtrips).
pub(super) const FAST_BLOCK_LENGTH: usize = 1 << 16;

/// L0's block length. The per-block dynamic-Huffman build (canonical code +
/// length-limiting over up to 288+32 symbols) costs roughly the SAME whether
/// the block covers 64 KiB or 1 MiB — it is a function of alphabet size, not
/// byte count — so widening the block cuts the number of builds (and their
/// bit-cost evaluations / header emissions) roughly proportionally with
/// little further ratio loss beyond less-local adaptation. 16x L1's block
/// gives L0 ~1/16th the per-block overhead while keeping most of L1's ratio
/// (measured: far closer to L1 than the static-Huffman-only alternative,
/// which gave up ~20%+ on text).
pub(super) const FAST0_BLOCK_LENGTH: usize = 1 << 20;

/// `l1-tune`-only lever (b) from the L1-band ratio-close-out mission brief
/// (2026-07-22 campaign): a conditional second-bucket probe. `head2[h]` holds
/// the position ONE GENERATION behind `head[h]` (see the `cand2` capture at
/// the top of [`process_position_l1`]); this helper is consulted ONLY when
/// the primary probe already produced an ACCEPTED match no longer than
/// `tune.bucket2_gate_max_len` (the "short-match acceptance" gate — never on
/// every position, which is the always-2-bucket shape the mission brief
/// measured too costly). Returns `(length, dist)` upgraded to the second
/// candidate's match iff it is both a valid, in-window distance AND strictly
/// longer than the primary; otherwise returns the inputs unchanged. Does not
/// exist in the shipped path (`l1-tune` is a non-default Cargo feature).
#[cfg(feature = "l1-tune")]
#[inline(always)]
fn bucket2_upgrade(
    pos: usize,
    buf: &[u8],
    cand2: u32,
    length: u32,
    dist: usize,
    tune: tune::L1Tune,
) -> (u32, usize) {
    if !tune.bucket2_enabled || length > tune.bucket2_gate_max_len {
        return (length, dist);
    }
    let dist2 = pos.wrapping_sub(cand2 as usize);
    if (1..=WINDOW).contains(&dist2) {
        let length2 = lz_extend(buf, pos, cand2 as usize, 0, DEFLATE_MAX_MATCH_LEN);
        if length2 >= SHORTEST_MATCH && length2 > length {
            return (length2, dist2);
        }
    }
    (length, dist)
}

/// Process ONE L1 position given its hash `h` and an ALREADY-LOADED candidate
/// `cand` (the caller is responsible for having read `head[h]` — this
/// function never re-reads it, only writes the insert). Returns the new
/// cursor after consuming this position: `pos + 1` for a plain literal
/// (including a lazy-peek DEFER, which is also a plain one-byte literal), or
/// `pos + length` for an accepted match.
///
/// L1-ONLY (no `ACCEL` ramp — see [`fastloop_l1`]'s doc comment for why L0
/// does not share this function at all, even behind a dead branch).
/// Factoring this out of the fastloop is what makes the two-position
/// software pipeline in [`fastloop_l1`] possible without duplicating the
/// ~70-line match/literal body: the SAME per-position logic runs whether
/// `cand` came from a fresh `head[h]` read (the boundary case) or from a
/// batch-issued read that raced ahead of the insert (SF2). A free function
/// rather than a closure: a closure capturing `head`/`sink` mutably for its
/// whole lifetime would prevent the caller from also touching `head`
/// directly between calls (the SF2 batch's own two `head[h]` reads) — the
/// borrow checker requires ordinary `&mut` parameters passed per call
/// instead.
///
/// Callers MUST uphold `pos < fast_end` (the same invariant the pre-SF2
/// single-position loop required): every load inside — the primary 4-byte
/// hash load's caller, the up-to-258-byte `lz_extend`, the lazy peek's OWN
/// 4-byte load + 258-byte `lz_extend` at `pos + 1`, and the interior
/// LIMIT_HASH_UPDATE loads — stays in bounds because
/// `fast_end <= in_end - DEFLATE_MAX_MATCH_LEN` (see [`run`]'s `fast_end`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn process_position_l1(
    pos: usize,
    h: usize,
    cand: u32,
    buf: &[u8],
    base: *const u8,
    head: &mut [u32],
    sink: &mut Sink,
    limit_hash_update_inserts: usize,
    #[cfg(feature = "anatomy-counters")] local: &mut FastLocalCounters,
    #[cfg(feature = "l1-tune")] head2: &mut [u32],
    #[cfg(feature = "l1-tune")] tune: tune::L1Tune,
) -> usize {
    // SAFETY: `lz_hash(_, HASH_BITS)` output is `< 2^16 == HASH_SIZE`.
    unsafe { *head.get_unchecked_mut(h) = pos as u32 };
    // `head_table_writes`/`probe_attempts`'s "1 per call" contribution is
    // counted by the CALLER instead of here (see `fastloop_l1`'s call
    // sites) — the caller already knows statically how many times it is
    // about to invoke this function, so folding these two unconditional
    // per-call bumps out of the hottest inlined body trims two field writes
    // from the common (miss/too-short) path without losing any count.

    // `l1-tune` bucket2 lever (search-only, OFF the shipped path): `head2`
    // holds the position ONE GENERATION behind `head` per hash slot. Read the
    // current occupant (the candidate a bucket2 probe would consult) then
    // shift `cand` — the value `head[h]` is about to be overwritten WITH —
    // down into `head2[h]`, keeping the one-generation-behind invariant.
    // Zero cost when the feature is off (the whole binding doesn't exist).
    #[cfg(feature = "l1-tune")]
    let cand2 = if tune.bucket2_enabled {
        // SAFETY: `h < HASH_SIZE == head2.len()`.
        let c2 = unsafe { *head2.get_unchecked(h) };
        unsafe { *head2.get_unchecked_mut(h) = cand };
        c2
    } else {
        NO_POS
    };

    // `pos - cand`; a wrapping sub keeps a sentinel/stale entry out of
    // the window range instead of panicking on underflow.
    let dist = pos.wrapping_sub(cand as usize);
    if (1..=WINDOW).contains(&dist) {
        let cand_pos = cand as usize;
        // Byte-exact extend (never trusts the hash): a spurious
        // candidate simply yields length < SHORTEST_MATCH -> literal.
        let length = lz_extend(buf, pos, cand_pos, 0, DEFLATE_MAX_MATCH_LEN);
        if length >= SHORTEST_MATCH {
            // `l1-tune` bucket2 upgrade (search-only lever (b) from the
            // L1-band mission brief): consult the SECOND candidate ONLY on a
            // short-match ACCEPTANCE (this `if` is already gated on
            // `length >= SHORTEST_MATCH`), never on every position (that is
            // the always-2-bucket approach the mission brief says was
            // measured too costly). Shadows `length`/`dist` with the
            // upgraded pair when bucket2 finds something longer; a no-op
            // (returns the inputs unchanged) when the feature is off, the
            // lever is disabled, or the gate/candidate doesn't pay off.
            #[cfg(feature = "l1-tune")]
            let (length, dist) = bucket2_upgrade(pos, buf, cand2, length, dist, tune);
            // Lazy peek (see `LAZY_PEEK_MAX_LEN`'s doc comment): gated to
            // short accepted matches only, so this branch is rare (most
            // matches are longer, or the position is a miss and never
            // reaches here).
            #[cfg(not(feature = "l1-tune"))]
            let (peek_max_len, peek_min_dist) = (LAZY_PEEK_MAX_LEN, LAZY_PEEK_MIN_DIST);
            #[cfg(feature = "l1-tune")]
            let (peek_max_len, peek_min_dist) = (tune.lazy_peek_max_len, tune.lazy_peek_min_dist);
            if length <= peek_max_len && dist > peek_min_dist {
                #[cfg(feature = "anatomy-counters")]
                {
                    local.lazy_peek_events += 1;
                }
                // SAFETY: caller-upheld `pos < fast_end <= in_end - 258`,
                // so `pos + 1 + 4 <= in_end` and `pos + 1 + 258 <= in_end`:
                // both the 4-byte load and the up-to-258-byte
                // `lz_extend` below stay in bounds.
                let seq1 = unsafe { load_u32(base, pos + 1) };
                let h1 = lz_hash(seq1, HASH_BITS) as usize;
                #[cfg(feature = "anatomy-counters")]
                {
                    local.hash_computations += 1;
                }
                // SAFETY: same as the primary lookup above; this is a
                // READ ONLY -- the peek does not insert `pos + 1` into
                // `head`, so the caller's next position (whether this
                // match is deferred or taken) is not a double-insert.
                let cand1 = unsafe { *head.get_unchecked(h1) };
                #[cfg(feature = "anatomy-counters")]
                {
                    local.head_table_reads += 1;
                }
                let dist1 = (pos + 1).wrapping_sub(cand1 as usize);
                if (1..=WINDOW).contains(&dist1) {
                    let next_len =
                        lz_extend(buf, pos + 1, cand1 as usize, 0, DEFLATE_MAX_MATCH_LEN);
                    if next_len >= length
                        && 4 * (next_len as i32 - length as i32)
                            + (bsr32(dist as u32) as i32 - bsr32(dist1 as u32) as i32)
                            > 2
                    {
                        // Defer: a meaningfully better match starts one
                        // byte later. Emit `pos` as a literal; the
                        // caller discovers `pos + 1`'s match fresh
                        // (including its own head-table insert).
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.lazy_peek_defers += 1;
                            local.probe_outcome_deferred += 1;
                        }
                        // SAFETY: `pos < fast_end <= in_end`.
                        sink.push_literal_fast(unsafe { *buf.get_unchecked(pos) });
                        return pos + 1;
                    }
                }
            }

            #[cfg(feature = "anatomy-counters")]
            {
                local.probe_outcome_accepted += 1;
                local.positions_processed_matches += length as u64;
            }
            // LIMIT_HASH_UPDATE (see the tail loop for the full note).
            let match_end = pos + length as usize;
            let insert_end = if limit_hash_update_inserts == usize::MAX {
                match_end
            } else {
                (pos + 1 + limit_hash_update_inserts).min(match_end)
            };
            let mut nh = pos + 1;
            while nh < insert_end {
                // SAFETY: nh < match_end = pos+length <= in_end, and
                // buf's pad covers the 4-byte load past in_end.
                let s = unsafe { load_u32(base, nh) };
                // SAFETY: `lz_hash` output `< HASH_SIZE`, as above.
                unsafe { *head.get_unchecked_mut(lz_hash(s, HASH_BITS) as usize) = nh as u32 };
                nh += 1;
            }
            // Counted ONCE for the whole interior loop (not per iteration —
            // `insert_end - (pos + 1)` is exactly the iteration count above,
            // known without re-walking): this branch only runs on an
            // ACCEPTED match, already the less-frequent outcome, so this is
            // a smaller win than the miss/too-short derivations above, but
            // free to take.
            #[cfg(feature = "anatomy-counters")]
            {
                let interior = (insert_end - (pos + 1)) as u64;
                local.hash_computations += interior;
                local.interior_writes += interior;
            }

            sink.push_match_fast(length, dist as u32);
            return pos + length as usize;
        }
        #[cfg(feature = "anatomy-counters")]
        {
            local.probe_outcome_too_short += 1;
        }
    } else {
        #[cfg(feature = "anatomy-counters")]
        {
            local.probe_outcome_miss += 1;
        }
    }

    // Literal (miss or too-short — `positions_processed`'s +1 for this
    // position is DERIVED at flush from the outcome bucket just set above,
    // see `FastLocalCounters::flush`'s doc comment).
    // SAFETY: `pos < fast_end <= in_end <= buf.len()`.
    sink.push_literal_fast(unsafe { *buf.get_unchecked(pos) });
    pos + 1
}

/// L0's fastloop (`Strategy::Fast0`, `ACCEL == true`): the ORIGINAL
/// SF1-C-only single-position loop, verbatim — a dedicated, non-generic
/// function (not a `run::<true>` monomorphization sharing source with
/// [`fastloop_l1`]) so L0's codegen can NEVER be perturbed by anything about
/// L1's SF2 batching, even a provably-dead branch.
///
/// BACKGROUND: an earlier version of this lever put both fastloops in the
/// SAME `run<const ACCEL: bool>` body gated by `if !ACCEL { batch } else {
/// scalar }` — logically dead for `run::<true>`, and Apple M1 confirmed L0
/// was a true no-op there (instruction count delta -0.02% to +0.05%, noise).
/// But the AMD EPYC (x86_64) box showed a REPRODUCIBLE +1-4% instruction and
/// wall regression for L0 with that shared-function shape, confirmed stable
/// across repeated runs AND with the co-resident `llama-server` fully
/// SIGSTOPped (ruling out co-tenant contention) — i.e. a genuine, if small,
/// x86_64-specific codegen interaction between L0's monomorphization and the
/// dead L1 branch living in the same source function. Splitting into two
/// physically separate functions removes any such channel entirely.
#[allow(clippy::too_many_arguments)]
fn fastloop_l0(
    mut pos: usize,
    fast_end: usize,
    buf: &[u8],
    base: *const u8,
    head: &mut [u32],
    sink: &mut Sink,
    limit_hash_update_inserts: usize,
    #[cfg(feature = "anatomy-counters")] local: &mut FastLocalCounters,
) -> usize {
    // Consecutive-miss counter driving the `ACCEL` scan-step ramp (see
    // `ACCEL_SHIFT`'s doc comment).
    let mut literal_run: u32 = 0;

    while pos < fast_end {
        // SF1-C software-pipeline: warm the head-table line for the position
        // PF_DIST ahead. Speculative (a match jumps the cursor past it); a
        // wrong prefetch only wastes bandwidth. SAFETY: `pos < fast_end <=
        // in_end - 258` and `PF_DIST` is small, so `pos + PF_DIST + 4 <=
        // in_end`, in bounds for the 4-byte load; the prefetch address is a
        // pure hint that never faults. (Not counted as a `fast_hash_computations`
        // event — see anatomy_counters.rs's fast_* doc comment: this hash is
        // speculative and the same position's hash is computed again for real
        // below.)
        unsafe {
            let fseq = load_u32(base, pos + PF_DIST);
            let fh = lz_hash(fseq, HASH_BITS) as usize;
            prefetch_write(head.as_ptr().add(fh) as *const u8);
        }
        // SAFETY: `pos < fast_end <= in_end - 258`, so `pos + 4 <= in_end`.
        let seq = unsafe { load_u32(base, pos) };
        let h = lz_hash(seq, HASH_BITS) as usize;
        #[cfg(feature = "anatomy-counters")]
        {
            local.hash_computations += 1;
        }
        // SAFETY: `lz_hash(_, HASH_BITS)` output is `< 2^16 == HASH_SIZE`.
        let cand = unsafe { *head.get_unchecked(h) };
        unsafe { *head.get_unchecked_mut(h) = pos as u32 };
        #[cfg(feature = "anatomy-counters")]
        {
            local.head_table_reads += 1;
            local.probe_attempts += 1;
        }

        // `pos - cand`; a wrapping sub keeps a sentinel/stale entry out of
        // the window range instead of panicking on underflow.
        let dist = pos.wrapping_sub(cand as usize);
        if (1..=WINDOW).contains(&dist) {
            let cand_pos = cand as usize;
            // Byte-exact extend (never trusts the hash): a spurious
            // candidate simply yields length < SHORTEST_MATCH -> literal.
            let length = lz_extend(buf, pos, cand_pos, 0, DEFLATE_MAX_MATCH_LEN);
            if length >= SHORTEST_MATCH {
                #[cfg(feature = "anatomy-counters")]
                {
                    local.probe_outcome_accepted += 1;
                    local.positions_processed_matches += length as u64;
                }
                // LIMIT_HASH_UPDATE (see the tail loop for the full note).
                let match_end = pos + length as usize;
                let insert_end = if limit_hash_update_inserts == usize::MAX {
                    match_end
                } else {
                    (pos + 1 + limit_hash_update_inserts).min(match_end)
                };
                let mut nh = pos + 1;
                while nh < insert_end {
                    // SAFETY: nh < match_end = pos+length <= in_end, and
                    // buf's pad covers the 4-byte load past in_end.
                    let s = unsafe { load_u32(base, nh) };
                    // SAFETY: `lz_hash` output `< HASH_SIZE`, as above.
                    unsafe { *head.get_unchecked_mut(lz_hash(s, HASH_BITS) as usize) = nh as u32 };
                    nh += 1;
                }
                // Counted once for the whole loop — see `process_position_l1`'s
                // matching comment.
                #[cfg(feature = "anatomy-counters")]
                {
                    let interior = (insert_end - (pos + 1)) as u64;
                    local.hash_computations += interior;
                    local.interior_writes += interior;
                }

                sink.push_match_fast(length, dist as u32);
                pos += length as usize;
                literal_run = 0;
                continue;
            }
            #[cfg(feature = "anatomy-counters")]
            {
                local.probe_outcome_too_short += 1;
            }
        } else {
            #[cfg(feature = "anatomy-counters")]
            {
                local.probe_outcome_miss += 1;
            }
        }

        // Literal (miss or too-short — see `FastLocalCounters::flush`'s doc
        // comment for the derivation that covers this position's count).
        // SAFETY: `pos < fast_end <= in_end <= buf.len()`.
        sink.push_literal_fast(unsafe { *buf.get_unchecked(pos) });

        // Common-case-cheap ramp: below `ACCEL_ARM_THRESHOLD` consecutive
        // misses, the ONLY added cost is the `literal_run += 1` and a
        // (well-predicted-not-taken, on any corpus with normal match
        // density) comparison — the shift/min/extra-literal-copy work only
        // runs once we're actually inside a long literal run, which is the
        // ONLY time it can pay for itself.
        literal_run += 1;
        let mut step = 1usize;
        if literal_run >= ACCEL_ARM_THRESHOLD {
            // Scan-step ramp: skip the hash lookup/insert for `step - 1`
            // further positions, coding them as literals directly.
            // `pos + step <= pos + ACCEL_MAX_STEP < fast_end +
            // ACCEL_MAX_STEP <= in_end` (fast_end's 258-byte margin dwarfs
            // ACCEL_MAX_STEP), so every extra literal index stays in bounds.
            step = (1 + (literal_run >> ACCEL_SHIFT) as usize).min(ACCEL_MAX_STEP);
            let mut i = 1;
            while i < step {
                // SAFETY: see the bounds note above.
                sink.push_literal_fast(unsafe { *buf.get_unchecked(pos + i) });
                i += 1;
            }
            #[cfg(feature = "anatomy-counters")]
            {
                local.positions_skipped += (step - 1) as u64;
            }
        }
        pos += step;
    }
    pos
}

/// L1's fastloop (`Strategy::Fast`, `ACCEL == false`): SF2 two-position
/// software pipeline. Issues BOTH head-table reads for `pos` and `pos + 1`
/// before consuming EITHER, so their independent cache-miss latency can
/// overlap in the OOO window instead of serializing behind position `pos`'s
/// insert+compare+branch — the mechanism SF1-C's prefetch already targeted,
/// pushed one step further (MLP instead of a hint).
///
/// Measured on M1 + AMD EPYC (2026-07): CPI dropped 7-14% on every L1 cell
/// (confirming the latency-overlap mechanism fires on both arches), netting
/// a real (if modest) wall win: -1.1% to -4.8% across both arches, all 3
/// corpora (text6/bin6/sil40), both boxes. See [`fastloop_l0`]'s doc comment
/// for why L0 does NOT share this function even behind a dead branch.
#[allow(clippy::too_many_arguments)]
fn fastloop_l1(
    mut pos: usize,
    fast_end: usize,
    buf: &[u8],
    base: *const u8,
    head: &mut [u32],
    sink: &mut Sink,
    limit_hash_update_inserts: usize,
    #[cfg(feature = "anatomy-counters")] local: &mut FastLocalCounters,
    #[cfg(feature = "l1-tune")] head2: &mut [u32],
    #[cfg(feature = "l1-tune")] tune: tune::L1Tune,
) -> usize {
    while pos < fast_end {
        // SF1-C software-pipeline: warm the head-table line for the
        // position PF_DIST ahead. Speculative (a match jumps the cursor
        // past it); a wrong prefetch only wastes bandwidth. SAFETY:
        // `pos < fast_end <= in_end - 258` and `PF_DIST` is small, so
        // `pos + PF_DIST + 4 <= in_end`, in bounds for the 4-byte load;
        // the prefetch address is a pure hint that never faults.
        unsafe {
            let fseq0 = load_u32(base, pos + PF_DIST);
            let fh0 = lz_hash(fseq0, HASH_BITS) as usize;
            prefetch_write(head.as_ptr().add(fh0) as *const u8);
        }

        // SF2 (this lever): gated to `pos + 1 < fast_end` so position
        // `pos + 1` ALSO independently satisfies the `pos < fast_end` bound
        // `process_position_l1` requires (its own lazy peek reads as far as
        // `pos + 2`) — see `fast_end`'s derivation in [`run`]. The lone
        // leftover position when `pos + 1 == fast_end` falls through to the
        // scalar path below, byte-for-byte the pre-SF2 loop body.
        if pos + 1 < fast_end {
            // Second half of the prefetch pair: this branch processes up to
            // TWO positions per outer iteration (vs one before), so without
            // this the `pos + 1 + PF_DIST` bucket would only get a prefetch
            // every OTHER position instead of every position — a coverage
            // gap vs the pre-SF2 density. Same bounds reasoning as above
            // (`pos + 1 <= fast_end`).
            unsafe {
                let fseq1 = load_u32(base, pos + 1 + PF_DIST);
                let fh1 = lz_hash(fseq1, HASH_BITS) as usize;
                prefetch_write(head.as_ptr().add(fh1) as *const u8);
            }
            // SAFETY: `pos + 1 < fast_end <= in_end - 258`, so both
            // `pos + 4 <= in_end` and `pos + 1 + 4 <= in_end`.
            let seq0 = unsafe { load_u32(base, pos) };
            let h0 = lz_hash(seq0, HASH_BITS) as usize;
            let seq1 = unsafe { load_u32(base, pos + 1) };
            let h1 = lz_hash(seq1, HASH_BITS) as usize;
            // SAFETY: `lz_hash(_, HASH_BITS)` output is always < HASH_SIZE.
            // Two independent loads, issued back to back with no
            // intervening write, so a superscalar core can have both
            // outstanding misses in flight at once (unlike the serial
            // read-modify-write-then-read-next-slot the single-position
            // loop forced).
            let cand0 = unsafe { *head.get_unchecked(h0) };
            let cand1_raw = unsafe { *head.get_unchecked(h1) };
            #[cfg(feature = "anatomy-counters")]
            {
                local.hash_computations += 2;
                local.head_table_reads += 2;
                local.k2_batch_iterations += 1;
            }

            let after0 = process_position_l1(
                pos,
                h0,
                cand0,
                buf,
                base,
                head,
                sink,
                limit_hash_update_inserts,
                #[cfg(feature = "anatomy-counters")]
                local,
                #[cfg(feature = "l1-tune")]
                head2,
                #[cfg(feature = "l1-tune")]
                tune,
            );
            #[cfg(feature = "anatomy-counters")]
            {
                local.probe_attempts += 1;
            }
            if after0 == pos + 1 {
                // Position `pos` resolved to a plain one-byte literal (no
                // match, including a lazy-peek DEFER) — exactly the case
                // where the ORIGINAL serial loop's next iteration would
                // process `pos + 1` next, so `cand1_raw` is reusable
                // PROVIDED it still reflects `head[h1]` as of right now.
                // `process_position_l1(pos, h0, cand0, ...)` performed
                // exactly one write, `head[h0] = pos`. If `h1 != h0` that
                // write didn't touch bucket `h1`, so `cand1_raw` (read
                // BEFORE the write, in parallel with `cand0`) still equals
                // `head[h1]` — identical to what a fresh read would return.
                // If `h1 == h0`, the write just set `head[h1]` to `pos`, so
                // the up-to-date value is `pos` itself, not the stale
                // pre-write `cand1_raw`.
                let cand1 = if h1 == h0 { pos as u32 } else { cand1_raw };
                pos = process_position_l1(
                    after0,
                    h1,
                    cand1,
                    buf,
                    base,
                    head,
                    sink,
                    limit_hash_update_inserts,
                    #[cfg(feature = "anatomy-counters")]
                    local,
                    #[cfg(feature = "l1-tune")]
                    head2,
                    #[cfg(feature = "l1-tune")]
                    tune,
                );
                #[cfg(feature = "anatomy-counters")]
                {
                    local.probe_attempts += 1;
                }
            } else {
                // Position `pos` was a match: it consumed `pos + 1` (or
                // jumped past it), so the batch's `cand1_raw` was never
                // valid for a *fresh* position `pos + 1` and must be
                // discarded, not corrected — the next outer iteration
                // re-issues both loads for whatever position `after0`
                // actually is.
                pos = after0;
            }
            continue;
        }

        // Boundary: exactly one position left in the fast region
        // (`pos + 1 == fast_end`) — process it scalar, byte-for-byte the
        // pre-SF2 loop body (fresh single `head[h]` read).
        // SAFETY: `pos < fast_end <= in_end - 258`, so `pos + 4 <= in_end`.
        let seq = unsafe { load_u32(base, pos) };
        let h = lz_hash(seq, HASH_BITS) as usize;
        #[cfg(feature = "anatomy-counters")]
        {
            local.hash_computations += 1;
        }
        // SAFETY: `lz_hash(_, HASH_BITS)` output is `< 2^16 == HASH_SIZE`.
        let cand = unsafe { *head.get_unchecked(h) };
        #[cfg(feature = "anatomy-counters")]
        {
            local.head_table_reads += 1;
        }
        pos = process_position_l1(
            pos,
            h,
            cand,
            buf,
            base,
            head,
            sink,
            limit_hash_update_inserts,
            #[cfg(feature = "anatomy-counters")]
            local,
            #[cfg(feature = "l1-tune")]
            head2,
            #[cfg(feature = "l1-tune")]
            tune,
        );
        #[cfg(feature = "anatomy-counters")]
        {
            local.probe_attempts += 1;
        }
    }
    pos
}

/// Run the one-pass fast encoder over `buf[data_start..in_end]`, appending one
/// or more DEFLATE blocks to `bw`.
///
/// `buf` MUST carry at least [`super::BUF_PAD`] trailing pad bytes beyond
/// `in_end` (upheld by the caller in `deflate::compress_block`) so the
/// speculative 4-byte hash loads and 8-byte match-extend loads never read out of
/// bounds. `buf[..data_start]` is an optional preset dictionary: its positions
/// are seeded into the head table so matches may reference it, but it is not
/// coded.
///
/// `block_length` is the input-byte span covered by each internal block
/// (L1 passes [`FAST_BLOCK_LENGTH`], L0 passes [`FAST0_BLOCK_LENGTH`]).
/// `use_dynamic` selects the block emitter: `true` evaluates a per-block
/// dynamic Huffman code (cheapest of dynamic/static/stored, [`emit_block`]);
/// `false` skips that build entirely (cheapest of static/stored only,
/// [`emit_block_static_or_stored`]).
///
/// `ACCEL` is a CONST generic (not a runtime `bool`) so L1's call
/// (`run::<false>`) monomorphizes to code with the accel state/arithmetic
/// compiled away entirely — L1's fastloop is exactly the code that existed
/// before the accel lever was added, not "the same logic with a runtime
/// branch". Only the `run::<true>` instantiation (L0) carries the ramp.
pub(super) fn run<const ACCEL: bool>(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
    block_length: usize,
    use_dynamic: bool,
    limit_hash_update_inserts: usize,
) {
    debug_assert!(in_end > data_start, "empty data handled by the caller");
    debug_assert!(buf.len() >= in_end + super::BUF_PAD);

    // Chainless head table: one slot per hash, holding the most recent position.
    let mut head = vec![NO_POS; HASH_SIZE];
    let base = buf.as_ptr();

    // `l1-tune` bucket2 lever's second table (search-only; see `tune`'s doc
    // comment). `ACCEL` is a const generic so `if !ACCEL` folds away at each
    // monomorphization: L0's (`ACCEL == true`) instantiation never allocates
    // this (empty `Vec`, zero cost), matching how `fastloop_l0` never
    // receives it. `tune::get()` reads env vars ONCE per `run()` call (cached
    // after the first call via its own `OnceLock`).
    #[cfg(feature = "l1-tune")]
    let mut head2: Vec<u32> = if !ACCEL {
        vec![NO_POS; HASH_SIZE]
    } else {
        Vec::new()
    };
    #[cfg(feature = "l1-tune")]
    let l1_tune = tune::get();

    // One accumulator for the WHOLE `run()` call (every internal block this
    // call emits), flushed once at the end — see `FastLocalCounters`'s doc
    // comment for why (spans multiple functions, so it can't live as a plain
    // local the way `hc.rs`'s `HcLocalCounters` does).
    #[cfg(feature = "anatomy-counters")]
    let mut local = FastLocalCounters::default();

    // Seed the preset dictionary (positions < data_start) into the head table.
    // Each has >= 4 readable bytes because data follows the dict in `buf`.
    // NOT part of `fast_positions_processed`/`fast_positions_skipped` — the
    // dictionary prefix is seeded but never coded, so it is outside the
    // "input bytes consumed by the parse" the reconciliation invariant covers
    // (see anatomy_counters.rs's fast_* doc comment).
    let mut p = 0usize;
    while p < data_start {
        // SAFETY: p < data_start <= in_end, and buf has BUF_PAD >= 16 bytes past
        // in_end, so [p, p+4) is in bounds.
        let seq = unsafe { load_u32(base, p) };
        head[lz_hash(seq, HASH_BITS) as usize] = p as u32;
        #[cfg(feature = "anatomy-counters")]
        {
            local.hash_computations += 1;
            local.dict_seed_writes += 1;
        }
        p += 1;
    }

    // Per-block accumulator: tokens + litlen/offset histograms built as-you-go.
    let mut sink = Sink::new();
    let mut pos = data_start;

    loop {
        // Start a new block. It ends after `block_length` input bytes (a match
        // straddling the boundary is allowed to overrun it slightly) or at EOF.
        let block_begin = pos;
        sink.begin();
        let block_end_target = (block_begin + block_length).min(in_end);

        // FASTLOOP / TAIL split (igzip loop2 shape). While
        // `pos < in_end - DEFLATE_MAX_MATCH_LEN`, `remaining` strictly exceeds
        // the longest possible match, so `max_len == DEFLATE_MAX_MATCH_LEN` is a
        // constant and the per-position `remaining`/`max_len`/`>= SHORTEST_MATCH`
        // computations fold away; folding the block-end break into the loop
        // condition removes the second per-token branch. Token equivalence with
        // the previous single loop: a token starting at `p` was processed iff
        // `p < block_end_target` (the break was checked AFTER each token, and
        // `block_end_target <= in_end`), which is exactly the pre-checked loop
        // condition here; positions below `fast_end` additionally satisfied
        // `max_len == DEFLATE_MAX_MATCH_LEN`. Identical token decisions ⇒
        // identical bytes.
        let fast_end = block_end_target.min(in_end.saturating_sub(DEFLATE_MAX_MATCH_LEN as usize));

        // Dispatch to a fully separate, non-generic fastloop per level
        // rather than sharing one `if ACCEL {...} else {...}`-gated body:
        // see [`fastloop_l0`]'s doc comment for why (a measured x86_64-only
        // codegen interaction through the shared-function shape, absent on
        // aarch64). `ACCEL` is still a compile-time constant here, so this
        // `if` itself compiles to a direct, unconditional call at each of
        // `run`'s two monomorphizations — no runtime dispatch either.
        pos = if ACCEL {
            fastloop_l0(
                pos,
                fast_end,
                buf,
                base,
                &mut head,
                &mut sink,
                limit_hash_update_inserts,
                #[cfg(feature = "anatomy-counters")]
                &mut local,
            )
        } else {
            fastloop_l1(
                pos,
                fast_end,
                buf,
                base,
                &mut head,
                &mut sink,
                limit_hash_update_inserts,
                #[cfg(feature = "anatomy-counters")]
                &mut local,
                #[cfg(feature = "l1-tune")]
                &mut head2,
                #[cfg(feature = "l1-tune")]
                l1_tune,
            )
        };

        // TAIL: the last <= DEFLATE_MAX_MATCH_LEN bytes of input (or of the
        // block), where `max_len` must be clamped per position.
        while pos < block_end_target {
            let remaining = in_end - pos;
            let max_len = if remaining > DEFLATE_MAX_MATCH_LEN as usize {
                DEFLATE_MAX_MATCH_LEN
            } else {
                remaining as u32
            };

            if max_len >= SHORTEST_MATCH {
                // SAFETY: max_len >= 4 implies pos + 4 <= in_end, in bounds.
                let seq = unsafe { load_u32(base, pos) };
                let h = lz_hash(seq, HASH_BITS) as usize;
                #[cfg(feature = "anatomy-counters")]
                {
                    local.hash_computations += 1;
                }
                let cand = head[h];
                head[h] = pos as u32;
                #[cfg(feature = "anatomy-counters")]
                {
                    local.head_table_reads += 1;
                    local.probe_attempts += 1;
                }

                // `pos - cand`; a wrapping sub keeps a sentinel/stale entry out of
                // the window range instead of panicking on underflow.
                let dist = pos.wrapping_sub(cand as usize);
                if (1..=WINDOW).contains(&dist) {
                    let cand_pos = cand as usize;
                    // Byte-exact extend (never trusts the hash): a spurious
                    // candidate simply yields length < SHORTEST_MATCH -> literal.
                    let length = lz_extend(buf, pos, cand_pos, 0, max_len);
                    if length >= SHORTEST_MATCH {
                        #[cfg(feature = "anatomy-counters")]
                        {
                            local.probe_outcome_accepted += 1;
                            local.positions_processed_matches += length as u64;
                        }
                        // LIMIT_HASH_UPDATE: insert the hash for the first
                        // LIMIT_HASH_UPDATE_INSERTS match-interior positions
                        // (igzip inserts ~3), then jump the cursor over the whole
                        // match. usize::MAX means "insert every interior position"
                        // (zlib-ng style). length >= 4 guarantees at least the
                        // first interior positions are inside the match; the
                        // `.min(match_end)` clamp keeps every insert inside it.
                        let match_end = pos + length as usize;
                        let insert_end = if limit_hash_update_inserts == usize::MAX {
                            match_end
                        } else {
                            (pos + 1 + limit_hash_update_inserts).min(match_end)
                        };
                        let mut nh = pos + 1;
                        while nh < insert_end {
                            // SAFETY: nh < match_end = pos+length <= in_end, and
                            // buf's pad covers the 4-byte load past in_end.
                            let s = unsafe { load_u32(base, nh) };
                            head[lz_hash(s, HASH_BITS) as usize] = nh as u32;
                            #[cfg(feature = "anatomy-counters")]
                            {
                                local.hash_computations += 1;
                                local.interior_writes += 1;
                            }
                            nh += 1;
                        }

                        sink.push_match_fast(length, dist as u32);
                        pos += length as usize;
                        continue;
                    }
                    #[cfg(feature = "anatomy-counters")]
                    {
                        local.probe_outcome_too_short += 1;
                    }
                } else {
                    #[cfg(feature = "anatomy-counters")]
                    {
                        local.probe_outcome_miss += 1;
                    }
                }
            }

            // Literal. When `max_len >= SHORTEST_MATCH`, this position was
            // probed and already landed in `probe_outcome_miss`/`_too_short`
            // above — its "+1 processed" is DERIVED at flush (see
            // `FastLocalCounters::flush`'s doc comment), no explicit count
            // needed here. The one case with NO outcome bucket at all is
            // near-EOF `max_len < SHORTEST_MATCH` (not enough lookahead to
            // probe, the `if` above never ran) — `no_probe_literals` is
            // exactly that case, and only that case.
            #[cfg(feature = "anatomy-counters")]
            if max_len < SHORTEST_MATCH {
                local.no_probe_literals += 1;
            }
            sink.push_literal_fast(buf[pos]);
            pos += 1;
        }

        // The fast-path pushes skip per-push `block_length` bookkeeping; the
        // covered length is exactly the cursor distance walked this block.
        sink.block_length = pos - block_begin;

        // Flush the block. The BFINAL bit is set only on the last internal
        // block AND only when this chunk is the last chunk of the whole
        // stream (`is_last`). A non-final chunk's blocks stay BFINAL=0 so the
        // sync-flush marker the caller appends can close the chunk on a clean
        // boundary.
        let is_final = is_last && pos == in_end;
        if use_dynamic {
            // L1: cheapest of per-block dynamic / static / stored.
            emit_block(bw, buf, block_begin, &sink, statics, is_final);
        } else {
            // L0: cheapest of static / stored only — no per-block dynamic
            // Huffman build (see `emit_block_static_or_stored`'s doc comment
            // for why this is the L0-vs-L1 cost/ratio trade).
            emit_block_static_or_stored(bw, buf, block_begin, &sink, statics, is_final);
        }
        if pos == in_end {
            break;
        }
    }

    // One flush for the whole call — see `FastLocalCounters`'s doc comment.
    #[cfg(feature = "anatomy-counters")]
    local.flush();
}
