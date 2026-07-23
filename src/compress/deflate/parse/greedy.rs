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
    is_last: bool,
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

/// The greedy inner token loop for ONE block (`block_begin` ..
/// `in_max_block_end`, though `continue_block`'s entropy/seq-count checks may
/// end it earlier, or a straddling match may run it slightly past). Factored
/// out of [`run`] (pure code motion — `run`'s per-call behavior is
/// unchanged) so [`super::gated::run`] can dispatch to this SAME logic
/// per-block, composed with [`super::lazy::run_block`] under a content
/// detector (see `gated.rs`'s module doc comment for the l3-tune
/// DETECTOR-GATED LAZY-L3 composition this exists for).
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
    let mut max_len = DEFLATE_MAX_MATCH_LEN;
    let mut nice_len = params.nice_match_length.min(max_len);
    let min_len = calculate_min_match_len(&buf[in_next..in_end], params.max_search_depth);

    loop {
        adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
        let (length, offset) = mf.longest_match(
            buf,
            in_base,
            in_next,
            min_len - 1,
            max_len,
            nice_len,
            params.max_search_depth,
            next_hashes,
        );

        if length >= min_len && (length > DEFLATE_MIN_MATCH_LEN || offset <= 4096) {
            sink.push_match(length, offset);
            mf.skip_bytes(
                buf,
                in_base,
                in_next + 1,
                in_end,
                (length - 1) as usize,
                next_hashes,
            );
            in_next += length as usize;
        } else {
            sink.push_literal(buf[in_next]);
            in_next += 1;
        }

        if !continue_block(sink, in_next, block_begin, in_max_block_end, in_end) {
            break;
        }
    }

    in_next
}
