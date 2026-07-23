//! Lazy / lazy2 parser (levels 5-9).
//!
//! Port of libdeflate `deflate_compress_lazy_generic`
//! (`vendor/libdeflate/lib/deflate_compress.c` ~:2604-2808): before committing a
//! match, look one position ahead (lazy) or two (lazy2) for a better one, using
//! the `bsr32` offset-cost tie-break `4*(next_len-cur_len) + (bsr(cur_offset) -
//! bsr(next_offset)) > 2` (lazy) / `> 6` (lazy2). Levels 5-7 use lazy, 8-9 lazy2.

use super::super::bitstream::BitWriter;
use super::super::level::LevelParams;
use super::super::matchfinder::hc::HcMatchfinder;
use super::super::tables::{DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN};
use super::{
    adjust_max_and_nice_len, bsr32, calculate_min_match_len, choose_max_block_end, continue_block,
    emit_block, recalculate_min_match_len, Sink, StaticCodes,
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
    let mut mf = HcMatchfinder::new();
    let mut in_base = 0usize;
    let mut next_hashes = [0u32; 2];
    let mut sink = Sink::new();

    if data_start > 0 {
        mf.skip_bytes(buf, &mut in_base, 0, in_end, data_start, &mut next_hashes);
    }

    let mut in_next = data_start;

    loop {
        // Start a new DEFLATE block.
        let block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end);
        sink.begin();

        in_next = run_block(
            buf,
            in_next,
            block_begin,
            in_max_block_end,
            in_end,
            params,
            lazy2,
            &mut mf,
            &mut in_base,
            &mut next_hashes,
            &mut sink,
        );

        emit_block(
            bw,
            buf,
            block_begin,
            &sink,
            statics,
            is_last && in_next == in_end,
        );
        if in_next == in_end {
            break;
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
