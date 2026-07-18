//! Greedy parser (levels 2-4).
//!
//! Port of libdeflate `deflate_compress_greedy`
//! (`vendor/libdeflate/lib/deflate_compress.c` ~:2528-2602): at each position
//! take the longest match; accept it when it clears the min-match-length
//! heuristic (and the short-match offset guard), otherwise emit a literal.

use super::super::bitstream::BitWriter;
use super::super::level::LevelParams;
use super::super::matchfinder::hc::HcMatchfinder;
use super::super::tables::{DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN};
use super::{
    adjust_max_and_nice_len, calculate_min_match_len, choose_max_block_end, continue_block,
    emit_block, Sink, StaticCodes,
};

pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
) {
    let mut mf = HcMatchfinder::new();
    let mut in_base = 0usize;
    let mut next_hashes = [0u32; 2];
    let mut sink = Sink::new();

    // Seed a preset dictionary into the matchfinder (positions before data_start
    // may be referenced by matches but are not coded).
    if data_start > 0 {
        mf.skip_bytes(buf, &mut in_base, 0, in_end, data_start, &mut next_hashes);
    }

    let mut in_next = data_start;
    let mut max_len = DEFLATE_MAX_MATCH_LEN;
    let mut nice_len = params.nice_match_length.min(max_len);

    loop {
        // Start a new DEFLATE block.
        let block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end);
        sink.begin();
        let min_len = calculate_min_match_len(&buf[in_next..in_end], params.max_search_depth);

        loop {
            adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
            let (length, offset) = mf.longest_match(
                buf,
                &mut in_base,
                in_next,
                min_len - 1,
                max_len,
                nice_len,
                params.max_search_depth,
                &mut next_hashes,
            );

            if length >= min_len && (length > DEFLATE_MIN_MATCH_LEN || offset <= 4096) {
                sink.push_match(length, offset);
                mf.skip_bytes(
                    buf,
                    &mut in_base,
                    in_next + 1,
                    in_end,
                    (length - 1) as usize,
                    &mut next_hashes,
                );
                in_next += length as usize;
            } else {
                sink.push_literal(buf[in_next]);
                in_next += 1;
            }

            if !continue_block(&mut sink, in_next, block_begin, in_max_block_end, in_end) {
                break;
            }
        }

        emit_block(bw, buf, block_begin, &sink, statics, in_next == in_end);
        if in_next == in_end {
            break;
        }
    }
}
