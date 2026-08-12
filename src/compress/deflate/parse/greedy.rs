//! Greedy parser (levels 2-4).
//!
//! Port of libdeflate `deflate_compress_greedy`
//! (`vendor/libdeflate/lib/deflate_compress.c` ~:2528-2602): at each position
//! take the longest match; accept it when it clears the min-match-length
//! heuristic (and the short-match offset guard), otherwise emit a literal.

use super::super::bitstream::BitWriter;
use super::super::encode_types::HeaderBudget;
use super::super::huffman::{CodeScratch, HeaderScratch};
use super::super::level::LevelParams;
use super::super::matchfinder::hc::HcMatchfinder;
use super::super::tables::{DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN};
use super::{
    adjust_max_and_nice_len, calculate_min_match_len, choose_max_block_end, continue_block,
    emit_block, far_len3, BlockRole, FarLen3Gate, InputMode, ParseState, Sink, StaticCodes,
    STREAM_BLOCK_LOOKAHEAD,
};

pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
    budget: HeaderBudget,
) {
    let mut state = ParseState::new();
    // Seed a preset dictionary into the matchfinder (positions before data_start
    // may be referenced by matches but are not coded).
    if data_start > 0 {
        let ParseState {
            mf,
            in_base,
            next_hashes,
            ..
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
        if is_last {
            BlockRole::Final
        } else {
            BlockRole::Interior
        },
        InputMode::Drain,
        budget,
    );
}

/// [`run`] with the matchfinder state supplied by the caller, so it can span
/// several calls over a sliding buffer.
///
/// Returns the position after the last COMPLETE block emitted.
///
/// `consume_all` and `is_last` are INDEPENDENT and must not be conflated —
/// doing so was a real bug caught by the concatenation tests. `is_last` marks
/// BFINAL on the closing block; it is false for every non-final chunk of a
/// CONCATENATED stream (the T>1 path), which nonetheless has to consume all
/// the input it was handed. `consume_all` is what the single-pass streaming
/// encoder sets to false: it means "this buffer will be refilled, so stop at
/// the last block boundary that still had [`STREAM_BLOCK_LOOKAHEAD`] bytes of
/// input behind it and let me carry the tail forward". That margin is exactly
/// what makes every block-boundary decision identical to a whole-buffer
/// encode, so the chunk seam leaves no trace in the emitted bytes.
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
    budget: HeaderBudget,
) -> usize {
    let mut sink = Sink::acquire();
    // One dynamic-header scratch buffer for the WHOLE call, reused across
    // every internal block (see `HeaderScratch`'s doc comment) instead of
    // `build_dynamic_header` allocating a fresh `Vec` per block.
    let mut header_scratch = HeaderScratch::new();
    let mut code_scratch = CodeScratch {
        budget,
        ..Default::default()
    };
    let mut in_next = from;

    loop {
        if !input_mode.must_drain() && in_end - in_next < STREAM_BLOCK_LOOKAHEAD {
            return in_next;
        }
        // Start a new DEFLATE block.
        let block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end);
        sink.begin();

        // `anatomy-wall` region: `parse_match` — one timer per INTERNAL
        // BLOCK, matching `fast.rs`'s L0/L1 convention (see
        // `anatomy_wall` module docs). For the greedy parser (L2/L4) this
        // wraps the WHOLE per-block token loop (`run_block`): match probe
        // + accept/literal decision + `Sink::push_{literal,match}` emission
        // are ALSO fused here, same rationale as the fast parser's fused
        // bucket — there is no call boundary inside `run_block` to split
        // "probing" from "emission" without going to per-position
        // granularity (disallowed). Zero cost when `anatomy-wall` is off.
        in_next = crate::anatomy_wall_time!(parse_match_ns, parse_match_calls, {
            run_block(
                buf,
                in_next,
                block_begin,
                in_max_block_end,
                in_end,
                params,
                state,
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
            params.try_exact_huffman,
        );
        if in_next == in_end {
            return in_next;
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
    // ONE state root, not three. `ParseState` already owns the matchfinder, `in_base`
    // and `next_hashes`; the caller used to split it back into three separate `&mut`
    // borrows here, so after `#[inline(always)]` fusion LLVM saw three independent
    // pointer roots it had to keep live across the whole token loop. libdeflate's
    // equivalent is one `struct libdeflate_compressor *c` at a fixed displacement.
    // Output is IDENTICAL by construction — this is pure code motion.
    state: &mut ParseState,
    sink: &mut Sink,
) -> usize {
    let ParseState {
        mf,
        in_base,
        next_hashes,
        ..
    } = state;
    let mf: &mut HcMatchfinder = mf;
    let in_base: &mut usize = in_base;
    let next_hashes: &mut [u32; 2] = next_hashes;
    let mut max_len = DEFLATE_MAX_MATCH_LEN;
    let mut nice_len = params.nice_match_length.min(max_len);
    let min_len = calculate_min_match_len(&buf[in_next..in_end], params.max_search_depth);
    // Far-len-3 cost gate: recomputed from the block's running frequencies at
    // the same cadence the lazy parser refreshes `min_len` (10 KB, then
    // doubling). Starts inert (= the shipped fixed guard) — it only opens
    // once this block's own statistics prove a far len-3 beats the three
    // literals it replaces.
    let mut far_len3 = FarLen3Gate::INERT;
    let mut next_recalc = in_next + (in_end - in_next).min(10000);

    loop {
        if in_next >= next_recalc {
            if params.far_len3_gate {
                far_len3 = FarLen3Gate::recalc(
                    &sink.litlen_freqs,
                    &sink.offset_freqs,
                    far_len3::GREEDY_MARGIN_EIGHTH_BITS,
                );
            }
            next_recalc += (in_end - next_recalc).min(in_next - block_begin);
        }

        adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
        let (length, offset) = mf.longest_match(
            buf,
            in_base,
            in_next,
            min_len - 1,
            max_len,
            nice_len,
            params.max_search_depth,
            params.good_match,
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
        } else if length >= min_len
            && far_len3.allows(offset, buf[in_next], buf[in_next + 1], buf[in_next + 2])
        {
            // Cost-gated far len-3 (see `far_len3.rs`). Greedy has no
            // lookahead, so a chance len-3 here would SHADOW a longer match
            // starting one position later — the mechanism that made the
            // ungated accept bleed on executables while the lazy levels
            // stayed near-clean. Probe position +1 first (the probe also
            // performs that position's matchfinder insert, so the skip below
            // starts one position later than the normal accept arm's).
            adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - (in_next + 1));
            let (next_len, next_offset) = mf.longest_match(
                buf,
                in_base,
                in_next + 1,
                DEFLATE_MIN_MATCH_LEN, // only a STRICTLY longer match shadows
                max_len,
                nice_len,
                params.max_search_depth,
                params.good_match,
                next_hashes,
            );
            if next_len > DEFLATE_MIN_MATCH_LEN {
                // The len-3 was a shadow: literal here, longer match at +1.
                sink.push_literal(buf[in_next]);
                sink.push_match(next_len, next_offset);
                mf.skip_bytes(
                    buf,
                    in_base,
                    in_next + 2,
                    in_end,
                    (next_len - 1) as usize,
                    next_hashes,
                );
                in_next += 1 + next_len as usize;
            } else {
                // Nothing longer behind it: take the far len-3. Position +1
                // is already inserted by the probe; only +2 remains.
                sink.push_match(DEFLATE_MIN_MATCH_LEN, offset);
                mf.skip_bytes(buf, in_base, in_next + 2, in_end, 1, next_hashes);
                in_next += DEFLATE_MIN_MATCH_LEN as usize;
            }
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
