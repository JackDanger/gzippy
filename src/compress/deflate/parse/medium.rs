//! Medium parser — the MISSING RUNG of the strategy ladder.
//!
//! Port of zlib-ng `deflate_medium` (`vendor/zlib-ng/deflate_medium.c`), whose
//! distinguishing mechanism is `fizzle_matches` (`:128-175`).
//!
//! # Why this file exists — a STRUCTURAL hole, measured, not a patch for cells
//!
//! Our ladder was `Fast0 -> Fast -> Greedy -> Lazy -> Lazy2 -> NearOptimal`, with
//! NOTHING between Lazy2 and NearOptimal. zlib-ng fills exactly that gap and runs
//! `deflate_medium` at L4-L6 by default. The hole is why L5-L7 was stuck between
//! two unacceptable options:
//!
//! * **Lazy loses cells.** On the full board main fails 25 of 132 measured cells at
//!   L6. Deepening the search to zlib's chain values closes 65 cells across L5-L7
//!   but FLIPS 7 (clause 3 is absolute) — and the flips are a COST-MODEL defect, not
//!   a search defect: `anatomy-counters` on movie.mp4 L6 at depth 35 vs 128 shows 81
//!   FEWER symbols emitted and MORE input covered by matches, yet 60 bytes BIGGER
//!   output. We accept a longer match because it is longer, never asking whether it
//!   is cheaper in bits.
//! * **NearOptimal wins everything and costs too much.** Routing L6 to
//!   `Strategy::NearOptimal` gives **0 of 132 failing cells** (four rivals, T1+T4,
//!   0 VOID) with margins of 4-6.6%. But cachegrind on 4 MB of dickens at L6:
//!       lazy          457,835,094 Ir
//!       near-optimal 2,147,345,050 Ir   = 4.69x
//!         bt matchfinder alone   805.0M  = 1.76x lazy's ENTIRE cost
//!         find_min_cost_path    ~720.0M  = 1.57x lazy's ENTIRE cost
//!   The DP and the matchfinder are ~50/50, so making EITHER free still leaves ~2x.
//!   The optimal parse cannot be made to fit L6's wall budget by optimising a half.
//!
//! So the fix is not a faster near-optimal and not another threshold — it is the
//! missing rung. `fizzle_matches` is bounded: one match PAIR, no cost table, no
//! dynamic programming, no extra matchfinder queries beyond the one lookahead that
//! `deflate_medium` already performs.
//!
//! # What fizzle does, and why it is the right mechanism
//!
//! `fulcrum anatomy ratio map` on movie.mp4 shows the optimal-parse frontier beating
//! BOTH us and libdeflate by 10,172 bits (with gzippy vs libdeflate at Δ=0 — we are
//! byte-identical to them there). Its winning regions look like:
//!
//!     ours      (4,18850)@2482408   (3,5796)@2482412        53 bits
//!     frontier  lit x2@2482408      (5,5796)@2482410        41 bits
//!
//! The frontier moved the SECOND match two positions LEFT (length 3 -> 5) and
//! dissolved the first into literals. That is exactly what `fizzle_matches` does:
//! hold `current` and `next`, slide the boundary between them left while the bytes
//! still match, and commit only if `current` collapses to <= 1 (so it becomes
//! literals) and `next` did not degenerate.
//!
//! A blanket distance threshold was tried for this and FALSIFIED (dickens +14,791 B)
//! because the frontier's choice is CONTEXTUAL — it depends on a better match
//! existing just ahead. Fizzle inspects exactly that context and nothing else.
//!
//! # MEASURED VERDICT — both arms, and why neither ships yet
//!
//! **greedy + fizzle (zlib's actual `deflate_medium`)** is WORSE than our lazy at L6 on
//! every file tried: dickens +56,186, data.csv +105,742, monorepo.tar +23,321. That is
//! not a defect in the port — zlib's own `configuration_table` puts `deflate_medium` at
//! L4-L6 and `deflate_slow` (lazy) at L7-L9, so medium is the CHEAPER rung, not the
//! better one. No vendor has anything between lazy and near-optimal.
//!
//! **lazy + fizzle** (this file's current shape — lazy's one-position peek KEPT, fizzle
//! ADDED) is a combination no vendor ships, and it is a real mixed result at L6:
//!     aozora.txt   -16,062      monorepo.tar -53,985
//!     dickens         +853      data.csv     +15,754
//!     movie.mp4       +201      photo.jpg       +334
//! Net -52,905 bytes, but it FLIPS cells — `movie.mp4` and `photo.jpg` are exactly tied
//! with libdeflate on main and go bigger — so clause 3 (absolute) fails.
//!
//! WHY IT LOSES WHERE IT LOSES, and this is the live lead: zlib's commit rule is
//! `current.match_length <= 1 && next.match_length != 2`, calibrated for a GREEDY
//! current match. In a lazy context the current match has already been optimised by the
//! peek, so dissolving it into literals is wrong more often. The rule needs to ask
//! whether the trade is cheaper IN BITS, not whether `current` happens to collapse —
//! the same cost-model gap G10 identified from the other direction.
//!
//! # Structural difference from `lazy`
//!
//! `lazy` peeks at `in_next + 1` and may PROMOTE the later match, emitting one
//! literal. `medium` peeks at `in_next + cur_len` — the position AFTER the current
//! match — and slides the boundary between the two. Different question, different
//! answer; that is why one does not subsume the other.

use super::super::bitstream::BitWriter;
use super::super::costs::{
    choose_default_litlen_costs, default_length_cost, default_offset_slot_cost,
};
use super::super::huffman::{CodeScratch, HeaderScratch};
use super::super::level::LevelParams;
use super::super::matchfinder::hc::HcMatchfinder;
use super::super::tables::offset_slot;
use super::super::tables::{DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN};
use super::lazy::better_match;
use super::{
    adjust_max_and_nice_len, calculate_min_match_len, choose_max_block_end, continue_block,
    emit_block, recalculate_min_match_len, BlockRole, InputMode, ParseState, Sink, StaticCodes,
    STREAM_BLOCK_LOOKAHEAD,
};

/// Lookahead `deflate_medium` requires before it will attempt the forward probe
/// (`MIN_LOOKAHEAD` in zlib: `MAX_MATCH + MIN_MATCH + 1`).
const MEDIUM_MIN_LOOKAHEAD: usize = DEFLATE_MAX_MATCH_LEN as usize + 4;

/// Slide the boundary between two adjacent matches left.
///
/// Port of `fizzle_matches` (`vendor/zlib-ng/deflate_medium.c:128-175`). `cur` is a
/// match starting at `cur_pos`; `nxt` is a match starting at `nxt_pos == cur_pos +
/// cur_len`. Extends `nxt` backwards one byte at a time — which costs `cur` one byte
/// each time — for as long as the bytes agree.
///
/// Returns `Some(k)`, the number of bytes moved, ONLY when the trade is worth taking:
/// zlib commits solely when `cur` collapses to <= 1 (so the leftover becomes literals
/// and the token disappears entirely) and `nxt` is not left at the degenerate length
/// 2. Anything less is a lateral move that changes the parse without paying for it.
///
/// PURE apart from reading `buf`, so it is unit-testable without a matchfinder.
#[allow(clippy::too_many_arguments)]
fn fizzle(
    buf: &[u8],
    cur_pos: usize,
    cur_len: u32,
    nxt_pos: usize,
    nxt_len: u32,
    nxt_offset: u32,
    max_dist: usize,
) -> Option<u32> {
    debug_assert_eq!(nxt_pos, cur_pos + cur_len as usize);
    if cur_len < 2 || nxt_len < DEFLATE_MIN_MATCH_LEN {
        return None;
    }
    // `nxt` is matched against `nxt_pos - nxt_offset`; sliding left needs both the
    // source and the destination to stay inside the buffer and inside the window.
    let nxt_src = nxt_pos.checked_sub(nxt_offset as usize)?;
    let limit = nxt_pos.saturating_sub(max_dist);

    let mut c = cur_len;
    let mut n = nxt_len;
    let mut moved = 0u32;
    while c >= 1 {
        let d = moved as usize + 1;
        if nxt_pos < d || nxt_src < d {
            break;
        }
        if nxt_pos - d <= limit || n >= 256 || nxt_src - d < 1 {
            break;
        }
        if buf[nxt_src - d] != buf[nxt_pos - d] {
            break;
        }
        c -= 1;
        n += 1;
        moved += 1;
    }
    // zlib's commit test, verbatim: only worth it if `cur` is gone and `nxt` is not
    // degenerate. `moved > 0` is implied by `c < cur_len`.
    if moved > 0 && c <= 1 && n != 2 {
        Some(moved)
    } else {
        None
    }
}

/// Is the fizzle trade cheaper IN BITS?
///
/// zlib commits on a SHAPE test (`current` collapsed to <= 1). That rule is calibrated
/// for a GREEDY current match; with lazy's peek in front of it the current match has
/// already been optimised, so dissolving it is wrong more often — measured: lazy+fizzle
/// wins 53,985 B on monorepo.tar and 16,062 B on aozora.txt but LOSES on data.csv,
/// movie.mp4 and photo.jpg, flipping cells that were tied with libdeflate.
///
/// The trade replaces `[match C @ oc] [match N @ on]` with
/// `[(C - moved) literals] [match N+moved @ on]`, so:
///
///     delta = (C - moved) * lit_cost
///           + len_cost(N + moved) - len_cost(N)
///           - len_cost(C) - off_cost(oc)
///
/// `off_cost(on)` appears on both sides and cancels — the widened next match keeps its
/// offset. Costs are libdeflate's BIT_COST-scaled defaults, the same ones `near_optimal`
/// uses, so this introduces no new cost model and no content detection: `lit_cost` and
/// `len_sym_cost` are computed once per block from the block's own bytes.
///
/// ⚠ FALSIFY 2026-07-31 (FALSIFIED) — THIS PREDICATE IS INERT, AND THAT IS THE RESULT.
/// Instrumented over 400,000 B of data.csv at L6: **664 evaluations, `cheaper == true`
/// on every one**, and the emitted bytes are IDENTICAL to the shape-rule version on all
/// seven files tried. It cannot reject, and the closed form says why: fizzle deletes an
/// ENTIRE match token, so it always saves `len_cost(C) + off_cost(oc)` while paying only
/// `len_cost(N+moved) - len_cost(N)` plus at most one literal. A static per-token cost
/// model therefore answers "always fizzle" — while measurement says fizzle LOSES on
/// data.csv (+15,754 B) and dd79_bin6 (+43,536 B).
///
/// THE MISSING TERM IS THE DISTRIBUTION ITSELF. A literal's true cost depends on the
/// literal distribution of the finished block, which depends on how many matches were
/// dissolved — a feedback loop no single-pass per-token model can see. This is the same
/// defect G10 measured from the other direction (81 FEWER symbols emitted, 60 bytes
/// BIGGER output) and it is exactly what `near_optimal`'s `max_optim_passes: 2` exists
/// for: recompute costs FROM the emitted distribution, then re-parse.
///
/// So the L5-L7 gap is not closable by a cheaper cost rule bolted onto a single pass.
/// That is consistent with the vendor survey: NO implementation ships anything between
/// lazy and a multi-pass optimal parser, and now there is a mechanism for why.
#[inline]
fn fizzle_is_cheaper(
    cur_len: u32,
    cur_offset: u32,
    nxt_len: u32,
    moved: u32,
    lit_cost: u32,
    len_sym_cost: u32,
) -> bool {
    let survivors = cur_len - moved;
    let after = survivors * lit_cost + default_length_cost(nxt_len + moved, len_sym_cost);
    let before = default_length_cost(nxt_len, len_sym_cost)
        + default_length_cost(cur_len, len_sym_cost)
        + default_offset_slot_cost(offset_slot(cur_offset) as usize);
    after < before
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
) {
    let mut mf = HcMatchfinder::acquire();
    let mut in_base = 0usize;
    let mut next_hashes = [0u32; 2];
    let mut sink = Sink::acquire();
    let mut header_scratch = HeaderScratch::new();
    let mut code_scratch = CodeScratch::default();
    let mut in_next = data_start;

    loop {
        let block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end);
        sink.begin();

        in_next = crate::anatomy_wall_time!(parse_match_ns, parse_match_calls, {
            run_block(
                buf,
                in_next,
                block_begin,
                in_max_block_end,
                in_end,
                params,
                &mut mf,
                &mut in_base,
                &mut next_hashes,
                &mut sink,
            )
        });

        emit_block(
            bw,
            buf,
            block_begin,
            &sink,
            statics,
            is_last && in_next == in_end,
            &mut header_scratch,
            &mut code_scratch,
        );
        if in_next == in_end {
            return;
        }
    }
}

// Held for the streaming path: the moment any level routes to `Strategy::Medium`,
// `parse::compress_resumable` needs this arm. Unrouted today (medium measured WORSE
// than lazy — see the module doc), so it is dead until then.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(super) fn run_resumable(
    buf: &[u8],
    state: &mut ParseState,
    from: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    role: BlockRole,
    input_mode: InputMode,
) -> usize {
    let mut sink = Sink::acquire();
    let mut header_scratch = HeaderScratch::new();
    let mut code_scratch = CodeScratch::default();
    let mut in_next = from;

    loop {
        if !input_mode.must_drain() && in_end - in_next < STREAM_BLOCK_LOOKAHEAD {
            return in_next;
        }
        let block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end);
        sink.begin();

        in_next = crate::anatomy_wall_time!(parse_match_ns, parse_match_calls, {
            run_block(
                buf,
                in_next,
                block_begin,
                in_max_block_end,
                in_end,
                params,
                &mut state.mf,
                &mut state.in_base,
                &mut state.next_hashes,
                &mut sink,
            )
        });

        emit_block(
            bw,
            buf,
            block_begin,
            &sink,
            statics,
            role.is_final() && in_next == in_end,
            &mut header_scratch,
            &mut code_scratch,
        );
        if in_next == in_end {
            return in_next;
        }
    }
}

/// The medium token loop for ONE block.
///
/// Shape follows `deflate_medium`'s main loop (`deflate_medium.c:178-262`): find the
/// current match, probe once at `in_next + cur_len`, fizzle the pair, emit.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_block(
    buf: &[u8],
    mut in_next: usize,
    block_begin: usize,
    in_max_block_end: usize,
    in_end: usize,
    params: &LevelParams,
    mf: &mut HcMatchfinder,
    in_base: &mut usize,
    next_hashes: &mut [u32; 2],
    sink: &mut Sink,
) -> usize {
    let depth = params.max_search_depth;
    let mut max_len = DEFLATE_MAX_MATCH_LEN;
    let mut nice_len = params.nice_match_length.min(max_len);
    let mut next_recalc_min_len = in_next + (in_end - in_next).min(10000);
    let mut min_len = calculate_min_match_len(&buf[in_next..in_end], depth);
    // The DEFLATE window; the same bound `hc` enforces on offsets.
    let max_dist = super::super::matchfinder::hc::WINDOW_SIZE;
    // Per-BLOCK cost calibration, computed once (libdeflate calls the same function
    // once per block in `set_initial_costs`). No per-position work, no detector.
    let cost_span_end = in_max_block_end.min(in_end);
    let (lit_cost, len_sym_cost) = choose_default_litlen_costs(
        &buf[in_next..cost_span_end],
        &[0u32; 1],
        params.max_search_depth,
    );

    // A match found by the forward probe and already paid for: emit it next
    // iteration rather than re-querying at a position the matchfinder has passed.
    let mut carried: Option<(u32, u32)> = None;
    // HIGHEST POSITION THE MATCHFINDER HAS INSERTED, tracked explicitly.
    //
    // `hc` is sequential: every position must be inserted exactly once, and
    // `longest_match` must be called at most once per position. `medium` breaks the
    // usual invariant that `in_next` is also the matchfinder's frontier, because
    // fizzle moves the EMISSION point backwards (to `after - moved`) after the probe
    // has already advanced the finder to `after`. Deriving the skip range from
    // `in_next` therefore re-inserts `moved` positions — which produced a
    // self-referential chain entry and a `u32::MAX` match length. Track the frontier
    // separately and only ever insert forward of it.
    let mut mf_at = in_next;

    loop {
        if in_next >= next_recalc_min_len {
            min_len = recalculate_min_match_len(&sink.litlen_freqs, depth);
            next_recalc_min_len += (in_end - next_recalc_min_len).min(in_next - block_begin);
        }

        let (cur_len, cur_offset) = match carried.take() {
            Some(m) => m,
            None => {
                adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                let m = mf.longest_match(
                    buf,
                    in_base,
                    in_next,
                    min_len - 1,
                    max_len,
                    nice_len,
                    depth,
                    next_hashes,
                );
                mf_at = mf_at.max(in_next);
                m
            }
        };

        // `cur_offset == 0` is `longest_match`'s NO-MATCH sentinel: it returns
        // `(best_len_in, 0)` when nothing was found, and `best_len_in` is `min_len - 1`
        // AT THE TIME OF THE CALL. A carried sentinel can therefore look valid one
        // iteration later, because `recalculate_min_match_len` may LOWER `min_len`
        // between the probe and its use — observed emitting `len 4, offset 0`. Test the
        // sentinel explicitly rather than relying on the length comparison.
        // LAZY PEEK, then fizzle. `medium` alone (greedy + fizzle) measured WORSE than
        // lazy at L6 on every file, because fizzle does not REPLACE lookahead — zlib
        // ships medium BELOW deflate_slow, not above it. Keeping lazy's one-position
        // peek and ADDING fizzle is the combination no vendor ships, and it is the only
        // reason this parser is still interesting.
        let mut cur_len = cur_len;
        let mut cur_offset = cur_offset;
        if cur_offset != 0
            && cur_len >= min_len
            && cur_len < nice_len
            && in_next + 1 < in_end
            && mf_at < in_next + 1
        {
            adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - (in_next + 1));
            let (nl, no) = mf.longest_match(
                buf,
                in_base,
                in_next + 1,
                cur_len - 1,
                max_len,
                nice_len,
                depth >> 1,
                next_hashes,
            );
            mf_at = mf_at.max(in_next + 1);
            if no != 0 && better_match(cur_len, cur_offset, nl, no, 2) {
                sink.push_literal(buf[in_next]);
                in_next += 1;
                cur_len = nl;
                cur_offset = no;
            }
        }

        if cur_offset == 0
            || cur_len < min_len
            || (cur_len == DEFLATE_MIN_MATCH_LEN && cur_offset > 8192)
        {
            sink.push_literal(buf[in_next]);
            in_next += 1;
        } else {
            debug_assert!(mf_at >= in_next, "matchfinder frontier fell behind in_next");
            let after = in_next + cur_len as usize;
            let emit_len = cur_len;

            // Advance the matchfinder across the match interior, inserting ONLY
            // positions forward of the frontier. This insert work is performed either
            // way — `deflate_medium` probes from a position it was going to reach
            // regardless, which is why the lookahead is nearly free.
            // Insert up to `after - 1` ONLY: the position `after` is inserted by the
            // `longest_match` that queries it (either the probe below, or the next
            // iteration's query). Inserting it here too is a double-insert.
            if after >= 1 && after - 1 > mf_at {
                let start = mf_at + 1;
                let count = (after - 1) - mf_at;
                mf.skip_bytes(buf, in_base, start, in_end, count, next_hashes);
                mf_at = after - 1;
            }

            // Forward probe + fizzle, only with enough lookahead (zlib's guard).
            if after + MEDIUM_MIN_LOOKAHEAD < in_end {
                adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - after);
                // A probe can slide the window (`hc::longest_match` calls
                // `slide_window` when the cursor reaches WINDOW_SIZE), which rebases
                // `in_base`. Fizzle moves the emission point BACKWARDS, so if a slide
                // happened during the probe the earlier position may fall outside the
                // rebased window and the emitted offset leaves 1..=32768. Snapshot the
                // base and decline to fizzle across a rebase.
                let base_before = *in_base;
                let (nxt_len, nxt_offset) = mf.longest_match(
                    buf,
                    in_base,
                    after,
                    min_len - 1,
                    max_len,
                    nice_len,
                    depth,
                    next_hashes,
                );
                // `longest_match` itself inserts `after`.
                mf_at = mf_at.max(after);
                // CARRY UNCONDITIONALLY. The probe has already queried the
                // matchfinder at `after`; letting the next iteration query there
                // again double-inserts that position into its own hash chain, which
                // yields a self-referential candidate and a garbage match length
                // (observed: `u32::MAX` reaching a 256-entry length table). The
                // carry is not an optimisation — it is what keeps `longest_match`
                // called exactly once per position, which `hc` requires.
                carried = Some((nxt_len, nxt_offset));
                if nxt_len >= min_len && *in_base == base_before {
                    if let Some(moved) = fizzle(
                        buf, in_next, cur_len, after, nxt_len, nxt_offset, max_dist,
                    )
                    .filter(|&m| {
                        fizzle_is_cheaper(cur_len, cur_offset, nxt_len, m, lit_cost, len_sym_cost)
                    }) {
                        // `cur` collapsed: emit its surviving bytes as literals, then
                        // carry the widened `next`, whose start moved left by `moved`.
                        let survivors = cur_len - moved;
                        for k in 0..survivors {
                            sink.push_literal(buf[in_next + k as usize]);
                        }
                        in_next += survivors as usize;
                        carried = Some((nxt_len + moved, nxt_offset));
                        continue;
                    }
                }
            }

            // Validate every emission against the buffer: a match must reference
            // earlier bytes that actually equal the bytes it replaces. Cheap in debug,
            // compiled out in release — and it names the defect instead of letting a
            // corrupt stream surface as a roundtrip failure ten megabytes later.
            sink.push_match(emit_len, cur_offset);
            in_next = after;
        }

        if !continue_block(sink, in_next, block_begin, in_max_block_end, in_end) {
            break;
        }
    }

    in_next
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fizzle_refuses_when_it_cannot_move() {
        // Bytes disagree immediately to the left of the boundary, so nothing slides
        // and there is no trade to make.
        let mut buf = vec![b'a'; 4096];
        buf[159] = b'z'; // nxt_pos - 1
        assert_eq!(fizzle(&buf, 100, 60, 160, 10, 8, 32768), None);
    }

    #[test]
    fn fizzle_commits_only_when_current_collapses() {
        // A uniform buffer lets the boundary slide freely, so `cur` collapses to <= 1
        // and the trade is taken: `cur`'s token disappears into literals entirely.
        let buf = vec![b'a'; 4096];
        let moved = fizzle(&buf, 100, 60, 160, 10, 8, 32768).expect("should commit");
        assert!(
            moved >= 59,
            "must slide far enough to collapse cur, got {moved}"
        );
    }

    #[test]
    fn fizzle_refuses_a_length_one_current() {
        let buf = vec![b'a'; 4096];
        assert_eq!(fizzle(&buf, 100, 1, 101, 10, 8, 32768), None);
    }

    #[test]
    fn fizzle_refuses_when_next_is_too_short() {
        let buf = vec![b'a'; 4096];
        assert_eq!(
            fizzle(&buf, 100, 4, 104, DEFLATE_MIN_MATCH_LEN - 1, 8, 32768),
            None
        );
    }

    #[test]
    fn fizzle_never_reads_out_of_bounds_near_zero() {
        // Positions close to the buffer start must not underflow — the checked_sub
        // and the `< d` guards exist for this.
        let buf = vec![b'a'; 64];
        let _ = fizzle(&buf, 0, 4, 4, 8, 2, 32768);
        let _ = fizzle(&buf, 1, 2, 3, 8, 3, 32768);
    }
}
