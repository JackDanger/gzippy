//! Lazy / lazy2 parser (levels 5-9).
//!
//! Port of libdeflate `deflate_compress_lazy_generic`
//! (`vendor/libdeflate/lib/deflate_compress.c` ~:2604-2808): before committing a
//! match, look one position ahead (lazy) or two (lazy2) for a better one, using
//! the `bsr32` offset-cost tie-break `4*(next_len-cur_len) + (bsr(cur_offset) -
//! bsr(next_offset)) > 2` (lazy) / `> 6` (lazy2). Levels 5-7 use lazy, 8-9 lazy2.

use super::super::bitstream::BitWriter;
use super::super::huffman::{CodeScratch, HeaderScratch};
use super::super::level::LevelParams;
use super::super::matchfinder::hc::HcMatchfinder;
use super::super::tables::{DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN};
use super::{
    adjust_max_and_nice_len, bsr32, calculate_min_match_len, choose_max_block_end, continue_block,
    emit_block, recalculate_min_match_len, BlockRole, InputMode, ParseState, Sink, StaticCodes,
    STREAM_BLOCK_LOOKAHEAD,
};

/// The offset-cost tie-break test shared by lazy and lazy2 (threshold differs).
#[inline]
fn better_match(
    cur_len: u32,
    cur_offset: u32,
    next_len: u32,
    next_offset: u32,
    threshold: i32,
) -> bool {
    next_len >= cur_len
        && 4 * (next_len as i32 - cur_len as i32)
            + (bsr32(cur_offset) as i32 - bsr32(next_offset) as i32)
            > threshold
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    lazy2: bool,
    is_last: bool,
) {
    // Task C (bucket-split-oracle): time the per-`run()` allocation itself
    // (see `anatomy_wall.rs`'s `mf_new_ns` doc comment). Per-invocation, not
    // per-position -- zero cost when `anatomy-wall` is off.
    let mut state = ParseState::new();
    if data_start > 0 {
        let ParseState {
            mf,
            in_base,
            next_hashes,
        } = &mut state;
        mf.skip_bytes(buf, in_base, 0, in_end, data_start, next_hashes);
    }
    run_resumable(
        buf,
        &mut state,
        data_start,
        in_end,
        params,
        statics,
        bw,
        lazy2,
        if is_last {
            BlockRole::Final
        } else {
            BlockRole::Interior
        },
        InputMode::Drain,
    );
}

/// [`run`] with the matchfinder state supplied by the caller, so it can span
/// several calls over a sliding buffer. See `greedy::run_resumable` for why
/// `consume_all` and `is_last` are independent, and why the
/// [`STREAM_BLOCK_LOOKAHEAD`] margin makes the chunk seam invisible in the
/// emitted bytes.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_resumable(
    buf: &[u8],
    state: &mut ParseState,
    from: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    lazy2: bool,
    role: BlockRole,
    input_mode: InputMode,
) -> usize {
    let mut sink = Sink::acquire();
    // See `greedy.rs`'s sibling declaration: one scratch per call, reused
    // across every internal block.
    let mut header_scratch = HeaderScratch::new();
    let mut code_scratch = CodeScratch::default();
    let mut in_next = from;

    loop {
        if !input_mode.must_drain() && in_end - in_next < STREAM_BLOCK_LOOKAHEAD {
            return in_next;
        }
        // Start a new DEFLATE block.
        let block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end);
        sink.begin();

        // `anatomy-wall` region: `parse_match` — see `greedy.rs`'s sibling
        // call site for the fused-bucket rationale (identical here: the
        // lazy/lazy2 per-block token loop has no call boundary between
        // "probing" and "emission" without per-position timing). Zero cost
        // when `anatomy-wall` is off.
        in_next = crate::anatomy_wall_time!(parse_match_ns, parse_match_calls, {
            run_block(
                buf,
                in_next,
                block_begin,
                in_max_block_end,
                in_end,
                params,
                lazy2,
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

/// The lazy/lazy2 inner token loop for ONE block. Factored out of [`run`]
/// (pure code motion — `run`'s per-call behavior is unchanged) so
/// [`super::gated::run`] can dispatch to this SAME logic per-block, composed
/// with [`super::greedy::run_block`] under a content detector (see
/// `gated.rs`'s module doc comment for the l3-tune DETECTOR-GATED LAZY-L3
/// composition this exists for).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_block(
    buf: &[u8],
    mut in_next: usize,
    block_begin: usize,
    in_max_block_end: usize,
    in_end: usize,
    params: &LevelParams,
    lazy2: bool,
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

    loop {
        // Refresh the min match length periodically from real literal usage.
        if in_next >= next_recalc_min_len {
            min_len = recalculate_min_match_len(&sink.litlen_freqs, depth);
            next_recalc_min_len += (in_end - next_recalc_min_len).min(in_next - block_begin);
        }

        adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
        let (mut cur_len, mut cur_offset) = mf.longest_match(
            buf,
            in_base,
            in_next,
            min_len - 1,
            max_len,
            nice_len,
            depth,
            next_hashes,
        );

        // FALSIFY 2026-07-31 (FALSIFIED) — do NOT extend the too-far rejection to
        // length-4. Adding `|| (cur_len == MIN+1 && cur_offset > 16384)` was measured at
        // L6, vanilla build, and it LOSES badly:
        //     dickens    4,539,505 -> 4,554,296   (+14,791)
        //     aozora.txt 4,072,294 ->  4,080,816   (+8,522)
        //     photo.jpg  6,472,062 ->  6,473,073   (+1,011)
        //     movie.mp4 12,890,404 -> 12,891,403     (+999)
        //     data.csv   3,372,612 ->  3,372,536      (-76)  <- the only gain
        //
        // WHERE THE IDEA CAME FROM, AND WHY IT WAS WRONG. `fulcrum anatomy ratio map` on
        // movie.mp4 shows the optimal-parse frontier beating BOTH us and libdeflate by
        // 10,172 bits, and the winning regions look like:
        //     ours     (4,18850)@2482408  (3,5796)@2482412       53 bits
        //     frontier lit x2@2482408     (5,5796)@2482410       41 bits
        // i.e. the frontier takes two LITERALS over a length-4 match at distance 18,850.
        // I generalised a blanket distance threshold from that. The frontier was not
        // applying a threshold — it chose literals BECAUSE IT KNEW a better match sat two
        // positions later. That is LOOKAHEAD, and no static distance cutoff approximates
        // it: length-4 matches at moderate distance are highly profitable on text, which
        // is why dickens loses 14.7 KB.
        //
        // GENERAL: a single region from an optimal-parse diff shows you WHAT the optimum
        // did, never WHY. Reading a rule out of one region and applying it unconditionally
        // is the same error as generalising a measurement across levels. The frontier's
        // headroom here is real and is worth 10,172 bits on this file — but it is
        // reachable only by cost-based lookahead, not by another threshold.
        if cur_len < min_len || (cur_len == DEFLATE_MIN_MATCH_LEN && cur_offset > 8192) {
            // No (usable) match — emit a literal.
            sink.push_literal(buf[in_next]);
            in_next += 1;
        } else {
            in_next += 1;
            'have_cur_match: loop {
                // A very long match is taken immediately.
                if cur_len >= nice_len {
                    sink.push_match(cur_len, cur_offset);
                    mf.skip_bytes(
                        buf,
                        in_base,
                        in_next,
                        in_end,
                        (cur_len - 1) as usize,
                        next_hashes,
                    );
                    in_next += (cur_len - 1) as usize;
                    break 'have_cur_match;
                }

                // Look one position ahead (half the search depth).
                adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                let (next_len, next_offset) = mf.longest_match(
                    buf,
                    in_base,
                    in_next,
                    cur_len - 1,
                    max_len,
                    nice_len,
                    depth >> 1,
                    next_hashes,
                );
                in_next += 1;
                if better_match(cur_len, cur_offset, next_len, next_offset, 2) {
                    // Better match one ahead: output a literal, promote it.
                    sink.push_literal(buf[in_next - 2]);
                    cur_len = next_len;
                    cur_offset = next_offset;
                    continue 'have_cur_match;
                }

                if lazy2 {
                    // Look a second position ahead (quarter the search depth).
                    adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                    let (next_len, next_offset) = mf.longest_match(
                        buf,
                        in_base,
                        in_next,
                        cur_len - 1,
                        max_len,
                        nice_len,
                        depth >> 2,
                        next_hashes,
                    );
                    in_next += 1;
                    if better_match(cur_len, cur_offset, next_len, next_offset, 6) {
                        sink.push_literal(buf[in_next - 3]);
                        sink.push_literal(buf[in_next - 2]);
                        cur_len = next_len;
                        cur_offset = next_offset;
                        continue 'have_cur_match;
                    }
                    sink.push_match(cur_len, cur_offset);
                    if cur_len > 3 {
                        mf.skip_bytes(
                            buf,
                            in_base,
                            in_next,
                            in_end,
                            (cur_len - 3) as usize,
                            next_hashes,
                        );
                        in_next += (cur_len - 3) as usize;
                    }
                    break 'have_cur_match;
                } else {
                    sink.push_match(cur_len, cur_offset);
                    mf.skip_bytes(
                        buf,
                        in_base,
                        in_next,
                        in_end,
                        (cur_len - 2) as usize,
                        next_hashes,
                    );
                    in_next += (cur_len - 2) as usize;
                    break 'have_cur_match;
                }
            }
        }

        if !continue_block(sink, in_next, block_begin, in_max_block_end, in_end) {
            break;
        }
    }

    in_next
}
