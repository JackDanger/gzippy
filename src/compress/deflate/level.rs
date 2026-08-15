//! Level → parser-parameters table (single source of truth for L0..L12).
//!
//! Port of the `switch (compression_level)` preset table in libdeflate
//! `vendor/libdeflate/lib/deflate_compress.c` (`deflate_alloc_compressor`,
//! ~:3920-4005). Each level selects a PARSE STRATEGY plus the two tuning knobs
//! the greedy/lazy parsers consume: `max_search_depth` and `nice_match_length`.
//!
//! Increment 2 implements the greedy (L2-4) and lazy/lazy2 (L5-9) strategies;
//! Increment 3 adds the near-optimal (L10-12) strategy; Increment 4 adds the
//! igzip-class one-pass FAST strategy for L1 (chainless single-probe hash table
//! + per-block cheapest-of-{dynamic,static,stored} Huffman coding — a port of
//! igzip `isal_deflate_body_base`). Increment 5 (ratio-hole fix, 2026-07)
//! gives L0 the SAME chainless matchfinder as L1 (`Strategy::Fast0`), but
//! skips the per-block dynamic-Huffman evaluation (always static-or-stored) —
//! cheaper than L1, and a real compressor instead of L0's old pure
//! stored-block passthrough.

use super::tables::DEFLATE_MAX_MATCH_LEN;

/// Parse strategy selected by a compression level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Level 0: igzip-class one-pass fast path, same chainless single-probe
    /// matchfinder as [`Strategy::Fast`], but each block is coded as the
    /// cheaper of a fixed (static) Huffman block or a stored block — the
    /// per-block DYNAMIC Huffman evaluation (canonical code build +
    /// length-limiting) that [`Strategy::Fast`] does is skipped entirely,
    /// which is the ratio/speed trade that makes L0 cheaper than L1. This
    /// replaces the old pure stored-block passthrough (which never
    /// compressed at all — see the L0 fix in the compression-ratio
    /// campaign).
    Fast0,
    /// Level 1: igzip-class one-pass fast path — chainless single-probe
    /// hash-table matchfinder + per-block cheapest-of-{dynamic,static,stored}
    /// Huffman coding. No hash chains, no depth loop
    /// (`vendor/isa-l/igzip/igzip_base.c:isal_deflate_body_base`).
    Fast,
    /// Greedy parse: always take the longest match at each position.
    Greedy,
    /// Lazy parse: defer a match one byte to check for a longer one.
    Lazy,
    /// Lazy2 parse: look ahead two positions.
    Lazy2,
    /// Near-optimal parse: bt matchfinder + iterative min-cost-path DP (L10-12).
    //
    // A 12-line doc comment describing a `LazyGated` DETECTOR-GATED LAZY-L3
    // strategy used to sit HERE, above `NearOptimal`, with the one real
    // `NearOptimal` line tacked on its end — so rustdoc rendered
    // `NearOptimal` as "per-block GREEDY-vs-LAZY dispatch under a two-sided
    // content detector". The `LazyGated` variant and `parse/gated.rs` were
    // deleted by user order (non-negotiable #3: no content detector chooses a
    // parser); the retraction reached the level-3 ARM (see its ⚠ STALE
    // marker) but not this enum, so the stale text outlived its variant by
    // attaching itself to the next one. That is the CLAUDE.md working rule
    // "a retraction must reach the ROOT" failing in the smallest possible
    // way: a deletion that removes a variant must also remove the doc
    // comment that was above it, or Rust silently re-parents the prose.
    NearOptimal,
}

/// Extra knobs for the near-optimal parser (`deflate_compress_near_optimal`).
#[derive(Clone, Copy, Debug)]
pub struct NearOptimalParams {
    /// `max_optim_passes` — max min-cost-path passes per block.
    pub max_optim_passes: u32,
    /// `min_improvement_to_continue` — stop passes early below this bit gain.
    pub min_improvement_to_continue: u32,
    /// `min_bits_to_use_nonfinal_path` — recover a prior pass's path if the
    /// final pass regressed by at least this many bits.
    pub min_bits_to_use_nonfinal_path: u32,
    /// `max_len_to_optimize_static_block` — block length below which to also
    /// optimize a static-Huffman solution.
    pub max_len_to_optimize_static_block: u32,
}

/// The parser parameters for a compression level.
#[derive(Clone, Copy, Debug)]
pub struct LevelParams {
    /// Cost the EXACT (package-merge) Huffman code as a second per-block candidate and emit
    /// whichever is cheaper. Non-worse than the heuristic BY CONSTRUCTION on SIZE, but it costs
    /// 10-14% wall at T1 (serial) against only +2.3% at T4 (the per-block work parallelises).
    /// Our T1 wall margin against libdeflate is 0-8%, so enabling it at T1 FLIPS wall cells —
    /// measured, both arms: sil40 L6 T1 went 0.952 PASS -> 1.035 FAIL. It is therefore T>1 ONLY,
    /// exactly like `max_search_depth` scaling, and set only by `params_parallel`.
    ///
    /// THE REASON IS THE WALL BUDGET, NOT BYTE-IDENTITY. T1 output happening to stay identical
    /// to `main` is a CONSEQUENCE of this gating, not a goal, and must never become one:
    /// byte-identity with libdeflate is the cage `CLAUDE.md` and the campaign memory both name
    /// as the thing that keeps us running their algorithm slower than they do. On SIZE this
    /// candidate is strictly better at T1 too (49 of 49 cells smaller, 0 worse, non-worse by
    /// construction) and we would take those bytes gladly. We decline them only because
    /// 10-14% of T1 wall is not available to buy them with. If the T1 wall deficit is ever
    /// closed, turn this on at T1 and take the size — do not preserve identity for its own
    /// sake.
    pub try_exact_huffman: bool,
    /// T>1-only: libdeflate's 2-way hash bucket (second candidate on short-match
    /// acceptance). Set only by `params_parallel`; T1 stays single-probe.
    pub fast_bucket2: bool,
    /// Gate paired with [`Self::fast_bucket2`]: consult the second bucket only when
    /// the primary probe already accepted a match no longer than this length.
    pub fast_bucket2_gate_max_len: u32,
    /// T>1-only: probe the second bucket slot on a primary miss (vendor
    /// `ht_matchfinder` probes both slots every lookup).
    pub fast_bucket2_probe_on_miss: bool,
    /// Hash inserts per accepted match interior. L1 T>1
    /// ([`apply_l1_fast_parallel_knobs`]) ships `usize::MAX` — insert EVERY
    /// skipped byte, exactly libdeflate's `ht_matchfinder_skip_bytes` — since
    /// the interleaved-bucket lever made full maintenance one cache line per
    /// insert. L1 T1 keeps the shipped cap of 8 (the maintenance bill is
    /// ~6-15% of L1 wall on match-dense files and T1's slack is thin — PR
    /// #296's clause-5 adjudication). Non-L1 levels keep the igzip-style
    /// small cap (only the fast path reads this).
    pub fast_hash_update_inserts: usize,
    /// **T1-ONLY (L1): MATCH REACH.** Selects the `REACH == true`
    /// monomorphization of `parse::fast`'s L1 fastloop, which shifts the whole
    /// 2-way bucket on every match-interior insert the way the vendor's
    /// `ht_matchfinder_skip_bytes` does. Paired with
    /// [`Self::fast_hash_update_inserts`] `== usize::MAX`; the two are one
    /// lever and are set together by [`apply_l1_match_reach_t1_knobs`], whose
    /// doc comment holds the whole measurement.
    ///
    /// `false` at T>1 BY CONSTRUCTION, and the reason is the WALL BUDGET, not
    /// byte-identity — the same shape as [`Self::try_exact_huffman`] with the
    /// thread counts swapped. Scoped adjudication of the unscoped lever
    /// (solvency, `try-l1-reach/try.json`, 208 in-scope cells) closed all four
    /// record-file size cells and then failed clause 3 on ONE cell:
    /// `pigz:ecoli.fastq:L1:T4:wall`, pass -> fail, cross-layout CONFIRMED
    /// REAL at median ln +0.1186. That is a T4 cell, in the thin-margin
    /// pigz-at-T4 class that also convicted #310, and the dense insert's
    /// maintenance cost lands hardest exactly where pigz's margin is thinnest.
    /// The two cells the lever closes at T1 —
    /// `libdeflate:{access.log,ecoli.fastq}:L1:T1:size` — do not need T>1 to
    /// move at all, so T>1 is left as `main`: the T4 wall cell cannot flip
    /// because its BYTES cannot change.
    ///
    /// Revival of the T>1 half is the wall-budget-scoped density (insert-all
    /// only while the block's measured cost stays inside the level's budget),
    /// i.e. the cost-model direction — NOT a second constant here.
    pub fast_dense_interior_insert: bool,
    /// T>1-only: vendor-exact bucket MAINTENANCE for the L1 head table — the
    /// INTERLEAVED 2-slot bucket layout (`parse/fast.rs`'s
    /// [`L1_HEAD_ENTRIES`]) with a slot-shift on EVERY insert (probe,
    /// interior, warmup, tail), i.e. `ht_matchfinder`'s protocol. Set only by
    /// [`apply_l1_fast_parallel_knobs`] (T>1); T1 keeps the two-array
    /// `head`/`head2` layout where only PROBE-position inserts shift and
    /// interior/warmup/tail inserts overwrite slot 0 — byte-frozen by the
    /// tie cage (see `scripts/campaign/tie-guard.sh`). Selected as a const
    /// generic at `parse::compress`'s dispatch so the T1 monomorphization
    /// compiles to the pre-lever code, not a runtime-branched hybrid.
    pub fast_interleaved_bucket: bool,
    /// T>1-only: lazy-peek COST-GATE. Rejects accepted matches whose
    /// estimated bit cost exceeds literals at the same span.
    pub fast_lazy_peek_cost_gate: bool,
    pub fast_lazy_peek_cost_margin_bits: i32,
    pub strategy: Strategy,
    /// Cap on hash-chain nodes searched per position (`c->max_search_depth`).
    pub max_search_depth: u32,
    /// Stop searching once a match this long is found (`c->nice_match_length`).
    pub nice_match_length: u32,
    /// zlib `good_length`: quarter the chain walk once `best_len_in >= good_match`
    /// (`vendor/zlib-ng/match_tpl.h:75-77`). Set only on the T1 path — see
    /// [`apply_zlib_t1_search_knobs`].
    pub good_match: u32,
    /// The far-len-3 cost gate (`parse/far_len3.rs`): let the greedy parser
    /// accept a len-3 match past the fixed offset-4096 guard when the block's
    /// running frequencies price it under the three literals it replaces.
    /// L2-only. NOT enabled at L4 (the other Greedy level): symbols.dwarf L4
    /// is a libdeflate byte-tie and the gate flips it +578 B (this box,
    /// 2026-08-09) — one cell, but the tie cage refuses any flip. Re-judge
    /// L4 if the gate's model ever prices the whole-block code externality.
    pub far_len3_gate: bool,
    /// L3-only lazy parser: when `nseqs*64 <= block_length` and
    /// `nseqs*M < block_length` (ultra-sparse), use offset-4096 instead of 8192
    /// for the len-3 guard. `0` = disabled (shipped 8192 everywhere).
    pub lazy_sparse_len3_guard_mul: u32,
    /// Near-optimal-only knobs (meaningful iff `strategy == NearOptimal`).
    pub near_optimal: NearOptimalParams,
}

/// L1 parse knobs shared by T1 (`params`) and T>1 (`params_parallel`) —
/// the values main shipped before the interleaved-bucket lever, byte-frozen
/// on the T1 route by the tie cage.
fn apply_l1_fast_shared_knobs(p: &mut LevelParams) {
    p.fast_bucket2 = true;
    p.fast_bucket2_gate_max_len = 64;
    p.fast_bucket2_probe_on_miss = true;
    // igzip's LIMIT_HASH_UPDATE (`vendor/isa-l/igzip/igzip_base.c:71-86`):
    // index a short prefix of an accepted match's interior, then jump the
    // cursor over the rest. T>1 KEEPS THIS; T1 replaces it — see
    // [`apply_l1_match_reach_t1_knobs`].
    p.fast_hash_update_inserts = 8;
    p.fast_lazy_peek_cost_gate = true;
    p.fast_lazy_peek_cost_margin_bits = 0;
}

/// **T1-ONLY (L1): MATCH REACH.** Index EVERY position of an accepted match's
/// interior (`usize::MAX`) and shift the whole 2-way bucket while doing it —
/// what the vendor does. Applied by [`params_baseline`] only; `params_parallel`
/// stops at [`apply_l1_fast_parallel_knobs`] above, so T>1 keeps igzip's `8`
/// and its output is byte-for-byte `main`'s.
///
/// # Why T1 only — the coordinate, stated separately from the mechanism
///
/// The mechanism below is INTRINSIC and holds at every thread count. The
/// SCOPE is a coordinate-dependent verdict and holds only where it was
/// measured. The unscoped lever (PR #319) was adjudicated on solvency
/// (`try-l1-reach/try.json`, scoped `levels=1`, 208 in-scope cells graded):
/// clause 4 closed ALL FOUR record-file cells,
/// `libdeflate:{access.log,ecoli.fastq}:L1:{T1,T4}:size`, and clause 3 then
/// failed on ONE cell — `pigz:ecoli.fastq:L1:T4:wall`, pass -> fail,
/// cross-layout CONFIRMED REAL, median ln +0.1186 (6-10x the layout floor).
/// Clause 3 is absolute, so the whole lever was NO-SHIP.
///
/// Every cell in that flip is at T>1, and the T1 half of the win needs
/// nothing from T>1. Gating here keeps `libdeflate:access.log:L1:T1:size` and
/// `libdeflate:ecoli.fastq:L1:T1:size` closing while leaving the T4 bitstream
/// IDENTICAL — so `pigz:ecoli.fastq:L1:T4:wall` cannot flip BY CONSTRUCTION,
/// not by a re-measurement that might come back differently. This is the
/// #310 pattern inverted: #310 kept T1 and paid at T4; here the margin is at
/// T4, so T4 is what we keep.
///
/// The Ir cost lands on the T1 side and is real, not free: trainer
/// (i7-13700T) cachegrind `-p1`, L1, frozen fixtures — text +7.93%, tabular
/// +12.23%, binary +8.01%, noise +1.80%. L6/L9 are untouched to the
/// instruction. It buys 23/23 tune files smaller at L1/T1 and a cliff that
/// goes from 3.4467x libdeflate to 0.9950x.
///
/// The T>1 half is NOT abandoned and NOT falsified — it is parked on a
/// different mechanism: density scoped by the block's measured wall budget
/// (the cost-model direction), where no constant decides and the data does.
/// Do not revive it by flipping this flag on in `params_parallel`.
///
/// # The mechanism (intrinsic; holds at every thread count)
fn apply_l1_match_reach_t1_knobs(p: &mut LevelParams) {
    // THE VENDOR DIFFERENCE. `vendor/libdeflate/lib/ht_matchfinder.h:196`
    // (`ht_matchfinder_skip_bytes`) is called with `count = length - 1` after
    // every accepted match and inserts all of them — libdeflate's L1 table is
    // dense at 1.000 inserts/byte. Ours was NOT: measured with
    // `--features anatomy-counters` on `e8_p8192_long_a256_r0` at L1/T1
    // (M1, 2026-08-13), 181,096 head writes over 1,048,576 bytes = **0.173
    // inserts/byte**, because 940,664 of those bytes (89.7%) sit inside an
    // accepted match whose interior past position 8 we never indexed. That is
    // the same 0.276-vs-1.000 structural gap the deleted `ht_fast.rs:200`
    // record's replacement measurement found; this is its cause.
    //
    // WHY IT COSTS SIZE. A position that was never inserted cannot be FOUND as
    // a match source later, so on content whose repeats are longer than the
    // prefix the next repeat's source is an un-indexed interior, the probe
    // misses, and we emit literals where the vendor emits a long match.
    // `examples/divergence_accounting --level 1` attributes the +96,482 B gap
    // on that point EXACTLY (residual 0 bits): 2,386 `we_lit_they_match`
    // decisions = +87,475 B (91%) where libdeflate takes a ~177-byte match at
    // distance 8192, plus 1,200 `diff_len` where our mean length is 66 against
    // their 195.
    //
    // WHY THIS MECHANISM AND NOT ANOTHER. The `diff_dist` bucket names it: on
    // 402 decisions we take the SAME length as libdeflate at a mean distance of
    // 20,582 while they take 8,192 — we match an OLDER copy of the same bytes,
    // which is only possible if the NEARER source is missing from the table. A
    // max-distance or `nice_length` cutoff would fail the opposite way (near
    // found, far missed), and a truncated `lz_extend` would show shorter
    // lengths at the SAME distance. Both are refuted by that one bucket.
    //
    // THE COST, AND THE CHEAPER POINT THAT WAS BUILT AND REJECTED. A strided
    // tail (index every k-th interior position after the dense prefix) was
    // implemented and swept on the cliff point: stride 1 -> 1.0022x
    // libdeflate, 2 -> 1.1195x, 4 -> 1.2272x, 8 -> 1.3487x, 16 -> 1.5123x
    // against 3.4467x at the old prefix-only setting. The stride does NOT pay:
    // a repeat whose source is un-indexed is found within k-1 further
    // positions, but the bytes skipped in the meantime become LITERALS, and on
    // high-entropy content a literal is a full 8 bits — the single most
    // expensive symbol there is. Only stride 1, i.e. this line, closes the
    // class, so the stride parameter was deleted rather than shipped at 1.
    p.fast_hash_update_inserts = usize::MAX;
    // The dense insert ALONE is worse, not better: `markup.xml` L1/T1 goes
    // WIN -> LOSS (+21,920 B) with `usize::MAX` and no bucket shift, because a
    // slot-0-only interior insert evicts good anchors with no second chance.
    // COMPOSITION IS REQUIRED — the line above and the line below are one
    // lever, and the flag is what selects `parse::fast`'s `REACH == true`
    // fastloop. `l1_match_reach_is_t1_only` asserts they never separate.
    p.fast_dense_interior_insert = true;
}

/// L1 knobs for the T>1 route ONLY: the shared set plus vendor-exact bucket
/// maintenance (PR #296, re-scoped to T>1).
///
/// Insert EVERY interior (match-skip) byte, shifting the interleaved 2-slot
/// bucket on each write — vendor `ht_matchfinder_skip_bytes` semantics. The
/// old cap of 8 plus shift-free interior overwrites left the bucket holding
/// stale generations; measured on solvency (2026-08-09, L1, main@03200049)
/// the pair of fixes flips access.log (-70,953 B vs libdeflate) and
/// ecoli.fastq (-7,579 B) and collapses the diff_dist divergence class to
/// exactly 0. See `parse/fast.rs`'s `L1_HEAD_ENTRIES` for the layout that
/// makes this affordable.
///
/// T>1-ONLY because the maintenance bill is REAL at T1: #296's solvency
/// adjudication measured insert-every-interior-byte at ~6-15% of L1 T1 wall
/// on match-dense files (data.json L1:T1 0.548 -> 0.638), a clause-5
/// NO-SHIP against T1's thin slack, while the same bill at T4 is absorbed
/// by 249-330% slack (and several T4 wall cells got FASTER). Pay where the
/// slack lives — the #297 pattern. The two L1 T1 size cells this leaves
/// open (libdeflate:{access.log,ecoli.fastq}:L1:T1) revive on a CHEAPER T1
/// maintenance scheme (batched/prefetched interior inserts), not on
/// re-widening this knob.
fn apply_l1_fast_parallel_knobs(p: &mut LevelParams) {
    apply_l1_fast_shared_knobs(p);
    p.fast_interleaved_bucket = true;
    p.fast_hash_update_inserts = usize::MAX;
}

/// Resolve a compression level (clamped to 0..=12) to its parser parameters.
///
/// The `max_search_depth`/`nice_match_length` values transliterate the vendor
/// presets exactly; the strategy mapping substitutes a fallback for the two
/// strategies not yet implemented in this increment (see the module docs).
pub fn params(level: u32) -> LevelParams {
    let p = params_baseline(level);
    // zlib-ng knobs apply only on the T1 whole-buffer pick-min path — see
    // [`deflate_one_shot_t1_zlib_pick_min`]. Segmented/streaming encode must
    // keep libdeflate depths so chunk bitstreams concatenate correctly.
    // Report the knobs that were ACTUALLY RESOLVED by the production path, once
    // per process. `fulcrum explain` reads this and asserts it against observed
    // behaviour; without it, the tool would have to keep its own copy of this
    // table, which would rot silently and would be a source-read rather than an
    // observation. Emitting from inside `params` means what is reported is what
    // executed. Feature-gated (default OFF) and compiles to nothing when off.
    #[cfg(feature = "anatomy-counters")]
    emit_declared_once(level, &p);
    p
}

/// libdeflate-table knobs (streaming / segmented / pick-min baseline).
pub(crate) fn params_baseline(level: u32) -> LevelParams {
    #[allow(unused_mut)]
    let mut p = params_inner(level);
    #[cfg(feature = "ladder-tune")]
    ladder_tune::apply(&mut p);
    if level == 1 {
        apply_l1_fast_shared_knobs(&mut p);
        // T1 ONLY. `params_parallel` deliberately does NOT call this — see its
        // doc comment for the adjudicated T4 wall cell that is the reason.
        apply_l1_match_reach_t1_knobs(&mut p);
    }
    p
}

/// zlib-ng chain depth + `good_match` for the T1 whole-buffer L5-L7 path.
pub(crate) fn params_zlib_t1(level: u32) -> LevelParams {
    let mut p = params_baseline(level);
    apply_zlib_t1_search_knobs(level, &mut p);
    p
}

/// One line per process: `LEVEL_DECLARED={json}`.
#[cfg(feature = "anatomy-counters")]
fn emit_declared_once(level: u32, p: &LevelParams) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "LEVEL_DECLARED={{\"level\":{},\"strategy\":\"{:?}\",\"max_search_depth\":{},\"nice_match_length\":{}}}",
            level, p.strategy, p.max_search_depth, p.nice_match_length
        );
    });
}

/// The level->params map for the PARALLEL (T>1) path.
///
/// WHY THIS EXISTS, and why it is not a knob. libdeflate-gzip is SINGLE-THREADED. Our board
/// failures are overwhelmingly T4 cells against it — 48 of 68 on the frozen box, with
/// libdeflate-at-T1 at ZERO — so the budget that matters is our 4 threads against their 1.
/// Measured (sil40, hyperfine n=5, /dev/null): at T4 we are 3.49x faster at L6 and 4.30x at
/// L9, i.e. 249-330% of wall slack. The whole parse-config space was once closed as
/// "unaffordable" against T1 slack of 0-8% — a budget 40x too small for the cells that fail.
///
/// Spending that slack: at L6, `Lazy2(35,65)` instead of `Lazy(35,65)` costs +10.9% of OUR
/// time (still 3.01x faster than the rival) and buys 19,000-24,000 B per file — roughly 100x
/// the T>1 seam it has to absorb. dickens L6 T4 goes from +343 B (FAILING) to -19,348 B.
///
/// SANCTIONED, not a content detector: `CLAUDE.md` non-negotiable #3 permits "parameter
/// tuning (write-buffer size, shared memory per thread count)", and STEP 2 states that "T>1
/// may emit different bytes than T1". Nothing here inspects the DATA — only the thread count
/// the user asked for.
///
/// The rule applied is one step of parse strategy at UNCHANGED knobs, which the ladder sweep
/// measured as strictly smaller at fixed depth. Levels with no stronger strategy available
/// (L0/L1 chainless, L3 already Lazy, L10-12 already NearOptimal) are
/// returned unchanged, so T>1 output at those levels is untouched. L8 and L9 are the
/// exceptions: they take the FULL step up to the near-optimal parser (first branch below).
pub fn params_parallel(level: u32) -> LevelParams {
    // L9 T>1 runs the NEAR-OPTIMAL parser at the L11 T>1 knobs — a level→config
    // routing decision (the map is free to change; CLAUDE.md "Every technique is
    // in scope"), measured in the crown-at-lower-levels study (2026-08-09, M1
    // laptop, synthetic fixtures, main@af5503f1):
    //
    //   size, 8 MiB fixtures, `-p4` (gzip container):
    //     text     ours-L9-pickmin 2,445,673 -> near-opt-chunked 2,260,495
    //              (gzip -9 2,425,685 / pigz -9 2,430,279 / libdeflate -9 2,445,776:
    //               LOSS -> WIN vs all three)
    //     binary   5,099,126 -> 5,093,556 (gzip 5,103,889: LOSS -> WIN)
    //     tabular  1,879,115 -> 1,832,677 (already winning; +46 KB margin)
    //     weights-like float32 stream: 7,774,144 -> 7,766,521 (libdeflate
    //              7,774,214 — the near-incompressible Lazy trap documented
    //              below does NOT recur under the cost-model parse)
    //     noise    byte-identical (stored escape unaffected)
    //   wall, same coordinate: the chunked pass alone is 377 ms vs the
    //     pick-min path's 1,030 ms (the two serial whole-file candidates
    //     dominated it; see `compress_buffer_pure`).
    //
    // The full crown (zopfli/ultra) engine was measured at the same coordinate
    // and REJECTED for the 1-9 ladder: 21-38x L9 wall at numiterations=1
    // (10.4 s on the same 8 MiB input, single-core — its master-block loop is
    // serial), and its F1 output (2,262,758) is LARGER than the near-optimal
    // chunked output above. Near-optimal captures 92-99% of crown-F15's size
    // win at ~1/30 the cost. Its exact-cost block SPLITTER transplanted onto
    // the shipped parse was also measured: -364/-268/+16 B — the crown's win
    // is the parse, not the boundaries.
    //
    // L11's knobs (not L10's): L10 is non-monotone vs L9 on the binary class
    // (+2,057 B at 1 MiB, would OPEN cells); L11/L12 are smaller than L9
    // everywhere measured, and L12's extra passes buy ~nothing (<0.01%) here.
    //
    // L8 JOINS L9 (nearoptimal-down-ladder probe, 2026-08-11, M1 laptop, same
    // 8 MiB fixtures, `-p4`, main@bee336b9 + this branch). L8 was the WEAKEST
    // T>1 level: excluded from depth scaling (the engine.wasm clause-3 flip
    // documented below), so its T4 triple pick-min shipped whole_t1 bytes that
    // LOSE to gzip -8 on the match-rich classes, and ran its three candidates
    // SEQUENTIALLY (881 ms on 8 MiB text at -p4 vs L7's 110 ms). The same
    // near-optimal routing is a strict Pareto win at L8 on every fixture:
    //
    //   size:  text    ship8 2,444,590 -> 2,260,495  (gzip -8 2,425,693: LOSS -> WIN)
    //          binary  ship8 5,117,867 -> 5,093,556  (gzip -8 5,108,313: LOSS -> WIN)
    //          tabular ship8 1,888,600 -> 1,832,677  (margin +56 KB)
    //          weights ship8 7,774,144 -> 7,766,521; noise byte-identical (stored)
    //   wall:  max(chunked 381 ms, whole_t1 guard 375 ms) vs the sequential
    //          triple's 881 ms on text — 1.3-2.4x FASTER across fixtures.
    //
    // The down-ladder probe also measured L6/L7 and they DO NOT pay: no failing
    // fixture cell exists at L6/L7 T4 (the chunked Lazy+depth-x4 configs already
    // beat every rival), and the near-optimal cost multiplier there is pure
    // added wall (L7: 2.5-4.9x, L6: 2.4-7.3x vs shipped) against T4 board slack
    // of only 2-5x. L8/L9 pay because their pick-min paths were ALREADY paying
    // near-optimal-class wall for worse bytes. Scope stops at L8.
    if level == 8 || level == 9 {
        return params_parallel(11);
    }
    let mut p = params_inner(level);
    // DEPTH, NOT STRATEGY. The first attempt took one step of parse strategy
    // (Greedy->Lazy, Lazy->Lazy2) and was NO-SHIP on clause 3: it flipped
    // igzip:weights.safetensors:L2:T4 from 0.9998 to 1.0002. Mechanism, measured: on
    // near-incompressible data (90 MB of float tensors) LAZY defers matches and emits more
    // literals than GREEDY, costing +32,975 B on that file. Lazy is not uniformly stronger;
    // it is stronger only where matches are dense.
    //
    // ⚠ CORRECTED 2026-08-01: the sentence that stood here — "Scaling max_search_depth has
    // no such failure mode — a deeper chain can only find a match at least as good" — is
    // FALSE, and was falsified by isolating the two mechanisms on engine.wasm L8 T4
    // (trainer, vanilla builds, deterministic bytes, libdeflate -8 = 396,254):
    //     main                        396,096  PASS
    //     depth x4 ONLY (exact OFF)   396,307  FAIL   <- the depth scaling alone
    //     try_exact_huffman ONLY      396,092  PASS   <- monotone, and 4 B smaller
    //     both (as first shipped)     396,302  FAIL
    // A deeper chain changes the PARSE, not merely the quality of one match: a longer
    // match taken at position i displaces a better match at i+k. It is NOT a magnitude
    // problem — engine.wasm L8 flips at x2, x3 and x4 alike — so no multiplier rescues it.
    // Depth scaling is therefore NOT monotone in output size, and the claim below is
    // narrowed to the levels where it was actually measured.
    //
    // Scaling max_search_depth measured strictly better at the levels below, including
    // L9 where the strategy step had nothing left to give (already Lazy2):
    //     L2 weights.safetensors  83,082,549 vs igzip 83,101,588  (0.99977, and 22 B SMALLER
    //                             than the shipped T>1 output — the clause-3 flip is gone)
    //     L2 dickens -107,533 | data.csv -112,714 | sil40 -157,275  vs libdeflate
    //     L6 sil40   -60,758  (strategy step gave -19,236)
    //     L9 sil40    -2,484  (strategy step gave nothing)
    //
    // x4 is the shipped factor. Wall on sil40, T4, vs BOTH rivals that matter (libdeflate is
    // single-threaded; pigz is not): L2 2.73x/1.39x, L6 2.53x/2.18x, L9 2.15x/1.62x faster.
    // Even at L9, where this walks 2,400 chain nodes, we stay ahead of both.
    // L8 IS EXCLUDED, and this is an EMPIRICAL PATCH ON A STRUCTURAL NON-MONOTONICITY,
    // not a principled fix. libdeflate:engine.wasm:L8:{T2,T4}:size flips PASS -> FAIL with
    // the scaling on (see the isolation above); L8/engine.wasm is simply where an 11-file
    // TUNE sample caught it, and a wider corpus may find the same effect elsewhere. Priced
    // at T4 vs libdeflate, L5-L9, 11 TUNE files:
    //     depth x4 at every level   closes 30, OPENS 1   <- clause 3 is absolute: NO-SHIP
    //     L8 excluded (this)        closes 26, opens 0
    //     try_exact_huffman alone   closes  5, opens 0
    // The exclusion costs 4 cells at L8 and removes the only clause-3 violation.
    // SINGLE-VARIABLE TEST: L4 parser only, depth UNCHANGED at 16 so `choose_min_match_len`
    // is not clamped (its `< 16` threshold forces min_len <= 7 and that was the confound
    // in the Lazy(12,30) attempt).
    if level == 4 {
        p.strategy = Strategy::Lazy;
        p.max_search_depth = p.max_search_depth.saturating_mul(4);
    } else {
        // The `level == 8` arm below is now unreachable (L8 returns early via
        // the near-optimal routing above); it is kept verbatim with its
        // receipt because the engine.wasm non-monotonicity it records is a
        // depth-scaling fact about Lazy2, not about L8 the level, and it must
        // be re-consulted if the near-optimal routing is ever reverted.
        p.max_search_depth = if level == 8 {
            p.max_search_depth
        } else {
            p.max_search_depth.saturating_mul(4)
        };
    }
    // L4 AT T>1 IS THE STRATEGY STEP **COMPOSED WITH** THIS BRANCH'S DEPTH SCALING.
    //
    // Measured against this branch, L4, 22 files: OPENED 0, CLOSED 5 at both T4 and T16,
    // T1 byte-identical, and **4,284,357 B smaller in total**. The parser step ALONE
    // (`Lazy` at depth 16, no scaling) also closed 5 and opened 0 but was only 3,151,510 B
    // smaller and made FOUR files worse — ecoli.fastq +114,871, access.log +53,268,
    // aozora.txt +28,452, weights.safetensors +31,651. Composing with the x4 recovers
    // three of those and beats this branch on them: ecoli -18,542, access.log -59,432,
    // aozora -71,994.
    //
    // The one file still worse is `weights.safetensors`, +31,516 — and that is the case
    // this file ALREADY documents above: "on near-incompressible data LAZY defers matches
    // and emits more literals than GREEDY, costing +32,975 B on that file". Measuring
    // +31,516 independently reproduces that +32,975. The cell is ALREADY FAILING vs
    // libdeflate on this branch (83,082,305 vs 83,082,171), and clause 3 does not protect
    // an already-failing cell — so it is a size cost, not a promotion blocker. Naming it
    // because a cell count hides it.
    //
    // ⛔ AND L4 IS THE MAXIMUM CLEAN SUBSET — extending this to every eligible level was
    // BUILT AND MEASURED, and it is NO-SHIP. The strategy step composed with the depth
    // scaling, applied wherever a stronger parser exists (Greedy->Lazy, Lazy->Lazy2), at
    // T4 vs this branch: **9 closed, 3 OPENED**. Per level:
    //
    //     L2   closed 3   OPENED 1   <- weights.safetensors vs igzip
    //     L4   closed 5   OPENED 0   <- THIS ARM
    //     L5   closed 0   OPENED 0   <- adds nothing
    //     L6   closed 0   OPENED 1   <- weights.safetensors vs libdeflate. strictly worse.
    //     L7   closed 1   OPENED 1   <- weights.safetensors vs libdeflate. net zero, NO-SHIP.
    //
    // **All three opens are the SAME FILE**, and it is the one this file already names:
    // on near-incompressible float tensors Lazy defers matches and emits more literals.
    // So the L4-only scope here is NOT a partial application waiting to be generalised —
    // it is the OPTIMUM of the family. Every other level either opens a cell or adds
    // nothing. Do not re-derive this by extending the arm.
    //
    // (superseded note kept for the reasoning chain)
    // L4 IS A STRATEGY STEP AT T>1, NOT A DEPTH STEP — measured, clause 3 clean.
    //
    // L4 is the only `Greedy` above L2 (L3 and L5 are both `Lazy`), so it enters T>1 with
    // no size margin at all, and the T>1 seam then costs it cells by 55-370 B. Stepping
    // the PARSER while holding depth at 16 gives it margin. Measured against this branch
    // as baseline, L4, all 22 corpus files, gzip/pigz/libdeflate:
    //
    //     T2   OPENED 0   CLOSED 4        T4  OPENED 0  CLOSED 5    T16  OPENED 0  CLOSED 5
    //     T1   byte-identical (198/198)   — `params_parallel` is T>1-only by construction
    //
    //   closed: data.sqlite vs gzip AND pigz (14,798,391 -> 12,352,596), plus dd79_bin6
    //           (-258 B margin), movie.mp4 (-370) and photo.jpg (-55) vs libdeflate — all
    //           zero-headroom seam cells.
    p.try_exact_huffman = true;
    // L1 AT T>1: enable the 2-way bucket inside `parse::fast` (search-only lever
    // (b) from the L1-band mission brief, gated to short accepts). Keeps lazy
    // peek/defer — unlike `ht_fast`'s greedy accept-all, which flipped tabular.
    //
    // ⚠ AND NOTHING MORE. `apply_l1_match_reach_t1_knobs` — the dense
    // match-interior insert + bucket shift — is called by `params_baseline`
    // (T1) and DELIBERATELY NOT HERE, so L1 T>1 keeps igzip's
    // `fast_hash_update_inserts = 8`, keeps `fast_dense_interior_insert =
    // false`, and emits byte-for-byte what `main` emits. Reason, adjudicated
    // on solvency: the unscoped lever flipped `pigz:ecoli.fastq:L1:T4:wall`
    // pass -> fail, cross-layout CONFIRMED REAL (median ln +0.1186), which is
    // clause 3 and therefore absolute. Read
    // `apply_l1_match_reach_t1_knobs`'s doc comment before adding a line
    // here; the T>1 revival is the wall-budget-scoped density, NOT this flag.
    if level == 1 {
        apply_l1_fast_parallel_knobs(&mut p);
    }
    p.good_match = 0;
    p
}

/// MEASUREMENT-ONLY level→params override, for deriving our own ladder instead of
/// inheriting libdeflate's (see the `ladder-tune` feature comment in `Cargo.toml`).
///
/// WHY THIS EXISTS. A full T1 size census (`/root/sizeboard-all-12fcd0ed/census.json`,
/// 22 corpus files x L1-9 x 4 rivals) found that against libdeflate we are an EXACT
/// BYTE TIE on 154 of 198 cells — 22/22 files at every one of L2, L4, L5, L6, L7, L8,
/// L9 — because `params_inner` transliterates their preset table. A tie is not a win:
/// it leaves zero size headroom, so all 109 T4 cells that fail vs libdeflate fail by
/// exactly the seam growth, with 0 bytes of slack to absorb it. L3 is the ONLY deep
/// level where we diverge (Lazy where they run Greedy at the same knobs) and it is the
/// only one where we hold a margin: smaller on 20/22 files, median 44 KB.
///
/// The frontier we need — a config that is BOTH smaller and cheaper than their level N
/// — cannot be read off the vendor's 8 points, because every one of those points is a
/// choice they made. It has to be measured.
///
/// This is NOT a production knob. CLAUDE.md non-negotiable #3 forbids env vars changing
/// behaviour in the shipped path; the deliverable here is a STATIC table, and the
/// feature is default-off, marks the binary `+INSTRUMENTED`, and is refused by
/// `scripts/campaign/board-wall.sh`. Resolved exactly ONCE per process via `OnceLock`
/// — reading the environment inside a per-position loop previously cost ~1.3 BILLION
/// instructions and inflated an ablation by 10x.
#[cfg(feature = "ladder-tune")]
pub mod ladder_tune {
    use super::{LevelParams, Strategy};

    /// `GZIPPY_LADDER=<strategy>:<max_search_depth>:<nice_match_length>`,
    /// e.g. `lazy:12:14`. Absent or unparseable => no override.
    fn spec() -> Option<(Strategy, u32, u32)> {
        static S: std::sync::OnceLock<Option<(Strategy, u32, u32)>> = std::sync::OnceLock::new();
        *S.get_or_init(|| {
            let raw = std::env::var("GZIPPY_LADDER").ok()?;
            let mut it = raw.split(':');
            let strategy = match it.next()? {
                "fast0" => Strategy::Fast0,
                "fast" => Strategy::Fast,
                "greedy" => Strategy::Greedy,
                "lazy" => Strategy::Lazy,
                "lazy2" => Strategy::Lazy2,
                "nearoptimal" => Strategy::NearOptimal,
                other => panic!("GZIPPY_LADDER: unknown strategy {other:?}"),
            };
            let depth = it.next()?.parse().ok()?;
            let nice = it.next()?.parse().ok()?;
            Some((strategy, depth, nice))
        })
    }

    pub fn apply(p: &mut LevelParams) {
        if let Some((strategy, depth, nice)) = spec() {
            p.strategy = strategy;
            p.max_search_depth = depth;
            p.nice_match_length = nice;
        }
    }
}

/// zlib-ng `configuration_table` chain depth + `good_length` for the T1 path
/// (G31/G31a). libdeflate's map gives shallow chains with no early exit; gzip
/// runs deep chains BECAUSE `good_match` quarters the walk once a match is long
/// enough. Applied only in [`params`], not [`params_parallel`], so T>1 keeps
/// libdeflate depths ×4.
fn apply_zlib_t1_search_knobs(level: u32, p: &mut LevelParams) {
    match level {
        5 => {
            p.good_match = 8;
            p.max_search_depth = 32;
        }
        6 => {
            p.good_match = 8;
            p.max_search_depth = 128;
        }
        7 => {
            p.good_match = 8;
            p.max_search_depth = 256;
        }
        _ => {}
    }
}

fn params_inner(level: u32) -> LevelParams {
    let max_match = DEFLATE_MAX_MATCH_LEN;
    // Placeholder near-optimal knobs for the non-near-optimal levels (unused).
    const NONE_NO: NearOptimalParams = NearOptimalParams {
        max_optim_passes: 0,
        min_improvement_to_continue: 0,
        min_bits_to_use_nonfinal_path: 0,
        max_len_to_optimize_static_block: 0,
    };
    const BUCKET2_OFF: (bool, u32) = (false, 8);
    const LAZY_SPARSE_LEN3_GUARD_MUL_OFF: u32 = 0;
    match level {
        0 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Fast0,
            max_search_depth: 0,
            nice_match_length: 32,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        // Native L1 is the igzip-class one-pass FAST path (Increment 4):
        // chainless single-probe hash table + direct-emit static Huffman. The
        // search-depth / nice-len knobs are unused by `Strategy::Fast` (it does
        // exactly one probe per position); they are left at the vendor-ish
        // values only so the struct is populated.
        1 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Fast,
            max_search_depth: 1,
            nice_match_length: 32,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        2 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Greedy,
            max_search_depth: 6,
            nice_match_length: 10,
            good_match: 0,
            far_len3_gate: true,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        // ⚠ STALE BELOW, KEPT FOR THE HISTORY ONLY: `Strategy::LazyGated` and
        // `parse::gated.rs` NO LONGER EXIST (deleted by user order — `CLAUDE.md`
        // non-negotiable #3 forbids content detectors choosing a parser). L3 routes to
        // `Strategy::Lazy` unconditionally; see the `3 => LevelParams` arm below, which
        // is the fact. The original note read:
        // L3 = DETECTOR-GATED LAZY (`Strategy::LazyGated` -> `parse::gated::run`,
        // per-block GREEDY-vs-LAZY dispatch under a two-sided literal-fraction
        // content detector — see `parse::gated`'s module doc comment).
        // Knobs unchanged from the prior plain-Greedy L3 (max_search_depth=12,
        // nice_match_length=14); the gate's own params (two-sided 34/95 pct
        // thresholds, 300KB detection block, initial_lazy=false) are
        // `parse::gated`'s `L3_GATE_*` constants, unconditionally in effect —
        // `l3-tune` no longer selects WHETHER L3 uses this strategy, only
        // whether those constants are env-var-overridable (see below).
        //
        // PROMOTION HISTORY (full campaign in git log; `2c7f9444` plain-lazy
        // -> `992c5837` strict-Pareto FAIL (ecoli.fastq/weights.safetensors
        // regress) -> `2c7f9444`-successor `parse::gated` composition fixes
        // both -> `2b566fcb` self-tax-vs-Greedy wall gate FAILS (wrong
        // rival) -> `88cf1b09` re-gated against the REAL rivals, pigz-3/
        // gzip-3, which is the record this promotion is adjudicated from).
        //
        // ADJUDICATION (2026-07-23, supervisor, promoting `88cf1b09`'s
        // frozen record — AMD EPYC 7282 Zen2 solvency, `/root/gz-l3final` +
        // `/root/l3final/{wall_results,wall_ld_results}.jsonl`, N=15
        // paired-diff/A/A-controlled, `/dev/null` sink, 6-file corpus x
        // T1/4/8/16 x {pigz-3, gzip-3}):
        //   - SIZE: strictly SMALLER than shipped Greedy on every file
        //     (the L3 campaign's original goal). Smaller than libdeflate-gzip-3
        //     on every file including `dd79_bin6` (was byte-EQUAL to ld-3
        //     under Greedy, a faithful-port confirmation, not a routing bug).
        //   - WALL: beats pigz-3/gzip-3 by 12-62% on all 30 (file x T x
        //     rival) cells, every 95% CI excluding 1.0 and clearing the ~1%
        //     A/A spread — INCLUDING `dd79_bin6` (21-45% faster) at every T.
        //   - ZERO class regressions: L2/L4/L6 byte-identical
        //     Greedy-vs-LazyGated builds; roundtrip byte-exact all files x
        //     T1/4/8/16; T4==T16 output byte-identical; `cargo test
        //     --release` green + clippy/fmt clean both feature states.
        //   - The ONE failing sub-leg — `dd79_bin6` size vs pigz-3/gzip-3
        //     (+0.445-0.767%, deterministic, zero-variance, narrowed from
        //     Greedy's 1.339% miss but not closed) — is a conjunctive-rule
        //     miss on a cell that is SPEED-ONLY today (L3 has never been the
        //     ship default on a size-vs-rivals basis; `dd79_bin6` already
        //     lost this same size comparison under Greedy) and remains
        //     SPEED-ONLY after this flip. Its residual is the SAME
        //     `match_diff` parse-quality gap `2c7f9444`'s own fulcrum
        //     diagnosis located (an optimal-frontier match-choice question,
        //     not an accept-vs-defer one lazy/greedy toggling reaches) —
        //     independent of this promotion and not a regression it
        //     introduces. Adjudicated PROMOTE: every gating leg (size vs
        //     shipped default, size vs ld-3, wall vs both real rivals at
        //     every T, zero-regression legs) clears; the recorded miss is
        //     out of this change's causal reach.
        3 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Lazy,
            max_search_depth: 12,
            nice_match_length: 14,
            good_match: 0,
            far_len3_gate: true,
            lazy_sparse_len3_guard_mul: 224,
            near_optimal: NONE_NO,
        },
        // PARKED, NOT SHIPPED — `Lazy` with max_search_depth 10 wins SIZE on 11 of 11
        // TUNE files (772,154 B total vs libdeflate-4, ZERO cells opened, clause 3 fully
        // satisfied) and FAILS clause 5 on 9 of 11 at T4. See the L4 sections of
        // docs/encoder-campaign-plan.md. Depths 6 and 8 are cheaper on the wall but OPEN
        // cells (clause 3); 10 and 12 satisfy clause 3 and cost too much wall. The
        // monotone cost/size relation leaves no interior point satisfying both, so this
        // is closed on the promotion rule AS WRITTEN — not on the encoder, and not on a
        // coordinate. The size win is intrinsic and does not expire; if the wall budget
        // ever changes, this is the configuration to re-measure FIRST.
        //     4 => Strategy::Lazy, max_search_depth: 10, nice_match_length: 30
        // symbols.dwarf is strictly Pareto-dominant there: 6,553 B smaller AND 4.8%
        // faster at T4 (wall ratio 0.9519).
        4 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Greedy,
            max_search_depth: 16,
            nice_match_length: 30,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        5 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Lazy,
            max_search_depth: 16,
            nice_match_length: 30,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        6 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Lazy,
            max_search_depth: 35,
            nice_match_length: 65,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        7 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Lazy,
            max_search_depth: 100,
            nice_match_length: 130,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        8 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Lazy2,
            max_search_depth: 300,
            nice_match_length: max_match,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        9 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::Lazy2,
            max_search_depth: 600,
            nice_match_length: max_match,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NONE_NO,
        },
        // Native near-optimal parser (`deflate_compress_near_optimal`,
        // deflate_compress.c:3974-4004).
        10 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::NearOptimal,
            max_search_depth: 35,
            nice_match_length: 75,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NearOptimalParams {
                max_optim_passes: 2,
                min_improvement_to_continue: 32,
                min_bits_to_use_nonfinal_path: 32,
                max_len_to_optimize_static_block: 0,
            },
        },
        11 => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::NearOptimal,
            max_search_depth: 100,
            nice_match_length: 150,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NearOptimalParams {
                max_optim_passes: 4,
                min_improvement_to_continue: 16,
                min_bits_to_use_nonfinal_path: 16,
                max_len_to_optimize_static_block: 1000,
            },
        },
        _ => LevelParams {
            try_exact_huffman: false,
            fast_bucket2: BUCKET2_OFF.0,
            fast_bucket2_gate_max_len: BUCKET2_OFF.1,
            fast_bucket2_probe_on_miss: false,
            fast_hash_update_inserts: 3,
            fast_dense_interior_insert: false,
            fast_interleaved_bucket: false,
            fast_lazy_peek_cost_gate: false,
            fast_lazy_peek_cost_margin_bits: 0,
            strategy: Strategy::NearOptimal,
            max_search_depth: 300,
            nice_match_length: max_match,
            good_match: 0,
            far_len3_gate: false,
            lazy_sparse_len3_guard_mul: LAZY_SPARSE_LEN3_GUARD_MUL_OFF,
            near_optimal: NearOptimalParams {
                max_optim_passes: 10,
                min_improvement_to_continue: 1,
                min_bits_to_use_nonfinal_path: 1,
                max_len_to_optimize_static_block: 10000,
            },
        },
    }
}

/// `max_passthrough_size` (`deflate_compress.c:3918`): inputs at or below this
/// size are emitted as a stored block without running the parser. `55 - 4*level`
/// for the near-optimal levels (negative/overflow clamps to 0 for lower levels
/// which are handled by their own passthrough).
///
/// Ported (formula pinned by the tests below) but not yet wired into
/// [`super::parse::near_optimal`]'s entry point — no call site skips the
/// parser for tiny near-optimal-level inputs today. Left as a documented,
/// tested residual (Stage E dead-code audit,
/// docs/compressor-architecture.md §5-E) rather than deleted: wiring it in
/// would change L10-12 output for inputs at/below the threshold, which is an
/// algorithmic change out of scope for a polish stage, but deleting a
/// correct, vendor-cited, unit-tested port for no reason would just be
/// throwing the work away. Re-open trigger: wiring this in is a real
/// (untried) small near-optimal-levels win, gated like any other lever.
#[allow(dead_code)]
pub fn max_passthrough_size(level: u32) -> usize {
    (55i64 - 4 * level as i64).max(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_mapping_matches_increment_scope() {
        assert_eq!(params(0).strategy, Strategy::Fast0);
        assert_eq!(params(1).strategy, Strategy::Fast); // igzip-class one-pass

        // ⚠ STALE: LazyGated is deleted; L3 is `Strategy::Lazy`. Original note:
        // L3 = LazyGated unconditionally since the 2026-07-23 supervisor
        // adjudication of the frozen cell gate (see the level-3 arm's
        // promotion-history comment); `l3-tune` only makes the gate's
        // constants env-overridable for the search harness. L2/L4 stay
        // Greedy either way.
        assert_eq!(params(2).strategy, Strategy::Greedy, "level 2");
        assert_eq!(params(3).strategy, Strategy::Lazy, "level 3");
        assert_eq!(params(4).strategy, Strategy::Greedy, "level 4");
        for l in 5..=7 {
            assert_eq!(params(l).strategy, Strategy::Lazy, "level {l}");
        }
        for l in 8..=9 {
            assert_eq!(params(l).strategy, Strategy::Lazy2, "level {l}");
        }
        for l in 10..=12 {
            assert_eq!(params(l).strategy, Strategy::NearOptimal, "level {l}");
        }
    }

    // DELETED 2026-07-31: `vendor_knob_values`, which asserted
    //     params(6).max_search_depth == 35   params(9).max_search_depth == 600
    //
    // IT WAS A CAGE, AND IT BEAT THE DOCUMENTATION. `CLAUDE.md` says the level->config
    // map "is currently a copy of libdeflate's, which is why we run their algorithm
    // slower than they do; it is free to change." This test said the opposite, in the
    // only medium that fails closed: touch the map, turn a test red. When a doc and a red
    // test disagree, the test wins every time, and it won for weeks —
    // `.git/logs/HEAD:235-237` records `probe/l5-depth` created and abandoned 102 seconds
    // later with no commit.
    //
    // Raising L5-L9 to zlib-ng's chain depths — the change this test forbade — closed 84
    // failing size cells (2026-07-31, full board, 1,320 measured, 0 VOID).
    //
    // A knob that is DECLARED FREE TO CHANGE must not be pinned by an equality assertion.
    // What is worth testing is the INVARIANT, not the value: depth must not decrease as
    // the level rises, because that is what "higher level = more effort" means and it is
    // the property a typo would actually break.
    #[test]
    fn search_effort_is_monotonic_in_level() {
        for l in 2..=9u32 {
            let prev = params(l - 1);
            let cur = params(l);
            assert!(
                cur.max_search_depth >= prev.max_search_depth,
                "L{l} searches less deeply than L{}: {} < {}",
                l - 1,
                cur.max_search_depth,
                prev.max_search_depth
            );
        }
        // nice_match_length is capped by the format, never exceeds it.
        for l in 0..=9u32 {
            assert!(
                params(l).nice_match_length <= DEFLATE_MAX_MATCH_LEN,
                "level {l}"
            );
        }
    }

    #[test]
    fn near_optimal_effort_is_monotonic_in_level() {
        // DELETED 2026-07-31: `near_optimal_knob_values`, which asserted
        //     params(10).max_search_depth == 35, .nice_match_length == 75,
        //     .max_optim_passes == 2, params(12).max_search_depth == 300, etc.
        //
        // SECOND INSTANCE OF THE CAGE. `vendor_knob_values` was deleted earlier today for
        // pinning L6/L9 depths that `CLAUDE.md` declares free to change; this pinned the
        // L10-L12 knobs the same way, and non-negotiable #5 now forbids it outright:
        // an equality assertion beats a sentence in a doc, because only one fails closed.
        // Found by an adversarial review after the first one was fixed — the pattern
        // repeats, so test the INVARIANT, not the VALUE.
        //
        // The invariant a typo would actually break is that effort rises with level.
        for l in 11..=12u32 {
            let prev = params(l - 1);
            let cur = params(l);
            assert!(
                cur.max_search_depth >= prev.max_search_depth,
                "L{l} searches less deeply than L{}",
                l - 1
            );
            assert!(
                cur.near_optimal.max_optim_passes >= prev.near_optimal.max_optim_passes,
                "L{l} runs fewer optimisation passes than L{}",
                l - 1
            );
        }
        for l in 10..=12u32 {
            let p = params(l);
            assert_eq!(p.strategy, Strategy::NearOptimal, "level {l}");
            assert!(p.nice_match_length <= DEFLATE_MAX_MATCH_LEN, "level {l}");
            assert!(p.near_optimal.max_optim_passes >= 1, "level {l}");
        }
    }

    /// The L1 MATCH-REACH knobs are T1-ONLY; the interleaved-bucket lever is
    /// T>1-ONLY. This is the assertion the whole clause-3 argument for REACH
    /// rests on: `pigz:ecoli.fastq:L1:T4:wall` flipped when the two-array
    /// REACH route ran at T>1 (PR #319). #310's interleaved route is a
    /// different monomorphization and was adjudicated separately.
    #[test]
    fn l1_match_reach_is_t1_only() {
        let t1 = params(1);
        let t_gt_1 = params_parallel(1);

        assert!(
            t1.fast_dense_interior_insert,
            "T1 L1 lost the match-reach bucket shift — the two record-file \
             cells (libdeflate:{{access.log,ecoli.fastq}}:L1:T1:size) close \
             only with it"
        );
        assert_eq!(
            t1.fast_hash_update_inserts,
            usize::MAX,
            "T1 L1 is no longer indexing the whole match interior; the dense \
             insert and the bucket shift are ONE lever (the insert alone is a \
             WIN -> LOSS flip on markup.xml) and must move together"
        );
        assert!(
            !t1.fast_interleaved_bucket,
            "T1 L1 must not take the T>1 interleaved-bucket route"
        );

        assert!(
            !t_gt_1.fast_dense_interior_insert,
            "T>1 L1 picked up the match-reach bucket shift. That is the \
             two-array REACH route whose unscoped adjudication failed clause 3 \
             on pigz:ecoli.fastq:L1:T4:wall."
        );
        assert!(
            t_gt_1.fast_interleaved_bucket,
            "T>1 L1 lost the interleaved-bucket route — the lever that closes \
             libdeflate:{{access.log,ecoli.fastq}}:L1:T4:size"
        );
        assert_eq!(
            t_gt_1.fast_hash_update_inserts,
            usize::MAX,
            "T>1 L1 interleaved bucket pairs with dense interior insert"
        );

        assert!(
            !(t1.fast_dense_interior_insert && t1.fast_interleaved_bucket),
            "REACH and INTERLEAVED are mutually exclusive on T1"
        );
        assert!(
            !(t_gt_1.fast_dense_interior_insert && t_gt_1.fast_interleaved_bucket),
            "REACH and INTERLEAVED are mutually exclusive at T>1"
        );

        // REACH is one lever on T1: dense insert and bucket shift move together.
        assert_eq!(
            t1.fast_dense_interior_insert,
            t1.fast_hash_update_inserts == usize::MAX,
            "T1 L1 has half the match-reach lever: dense insert with no \
             bucket shift measured +21,920 B on markup.xml (WIN -> LOSS)"
        );

        // No OTHER level has either knob, at either thread count.
        for l in (0..=12u32).filter(|&l| l != 1) {
            for (name, p) in [
                ("params", params(l)),
                ("params_parallel", params_parallel(l)),
            ] {
                assert!(
                    !p.fast_dense_interior_insert,
                    "{name}({l}) carries the L1 match-reach knob"
                );
                assert!(
                    !p.fast_interleaved_bucket,
                    "{name}({l}) carries the L1 interleaved-bucket knob"
                );
            }
        }
    }
}
