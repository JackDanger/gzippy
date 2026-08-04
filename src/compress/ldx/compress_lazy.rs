//! C: `vendor/libdeflate/lib/deflate_compress.c:2605-2835` —
//! `deflate_compress_lazy_generic`, plus its two instantiations
//! `deflate_compress_lazy` (`lazy2 = false`) and `deflate_compress_lazy2`
//! (`lazy2 = true`).
//!
//! Before choosing a match, the lazy compressor checks whether there is a better match
//! at the next position. If yes, it outputs a literal and continues; if no, it outputs
//! the match. `lazy2` looks ahead two positions rather than one — slightly slower,
//! slightly smaller.
//!
//! **This gates L5-L9**: lazy at (16,30), (35,65), (100,130) and lazy2 at (300,258),
//! (600,258).

use super::bitstream::DeflateOutputBitstream;
use super::compress_fastest::choose_max_block_end;
use super::compress_greedy::GreedyState;
use super::flush::Compressor;
use super::hc_matchfinder::{hc_matchfinder_longest_match, hc_matchfinder_skip_bytes};
use super::min_match::{calculate_min_match_len, recalculate_min_match_len};
use super::sequences::{
    adjust_max_and_nice_len, deflate_begin_sequences, deflate_choose_literal, deflate_choose_match,
    deflate_finish_block,
};
use super::split::{init_block_split_stats, should_end_block};
use super::{
    DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN, SEQ_STORE_LENGTH, SOFT_MAX_BLOCK_LENGTH,
};

/// C: `bsr32(u32 v)` — index of the highest set bit.
///
/// Used only to compare the *magnitudes* of two offsets, so what matters is that it is
/// `floor(log2(v))`. Undefined for 0 in the C; both call sites pass a match offset,
/// which is always >= 1.
#[inline(always)]
fn bsr32(v: u32) -> i32 {
    debug_assert!(v != 0, "bsr32(0) is undefined in the C");
    (31 - v.leading_zeros()) as i32
}

/// C: `deflate_compress_lazy_generic(..., bool lazy2)` (:2605)
///
/// # The lazy decision rule, and why it is not just "is the next match longer"
///
/// ```c
/// next_len >= cur_len &&
/// 4 * (int)(next_len - cur_len) + ((int)bsr32(cur_offset) - (int)bsr32(next_offset)) > 2
/// ```
///
/// Each extra byte of match length is worth 4 points; each halving of the offset is
/// worth 1. So a match one byte longer at a much worse offset can LOSE, and a
/// same-length match at a much nearer offset can WIN. The threshold is **2** for the
/// one-ahead check and **6** for lazy2's two-ahead check, because deferring by two
/// positions costs two literals rather than one.
///
/// `next_len >= cur_len` is evaluated FIRST and short-circuits, which is what makes
/// `(int)(next_len - cur_len)` safe — the subtraction is unsigned in the C.
///
/// # The lookahead uses a SHALLOWER search, deliberately
///
/// The C: *"since we already have a match at the current position, we use only half the
/// `max_search_depth` when checking the next position. This is a useful trade-off
/// because it's more worthwhile to use a greater search depth on the initial match."*
/// One-ahead gets `depth >> 1`; lazy2's two-ahead gets `depth >> 2`. Using the full
/// depth is a different, slower, differently-compressing program.
///
/// # `min_len` is recalculated DURING the block, on a widening cadence
///
/// The first recalculation happens 10,000 bytes in; after that the interval becomes the
/// block's current length, so checks get rarer as the block grows and its literal
/// statistics stabilise. Greedy never does this — only the lazy parsers do.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deflate_compress_lazy_generic(
    c: &mut Compressor,
    p: &mut GreedyState,
    r#in: &[u8],
    in_nbytes: usize,
    os: &mut DeflateOutputBitstream<'_>,
    max_search_depth: u32,
    nice_match_length: u32,
    lazy2: bool,
) {
    let mut in_next: usize = 0;
    let in_end: usize = in_nbytes;
    let mut in_cur_base: usize = 0;
    let mut max_len: u32 = DEFLATE_MAX_MATCH_LEN;
    let mut nice_len: u32 = core::cmp::min(nice_match_length, max_len);
    let mut next_hashes: [u32; 2] = [0, 0];

    p.hc_mf.init();

    loop {
        // Starting a new DEFLATE block.
        let in_block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end, SOFT_MAX_BLOCK_LENGTH);
        let mut next_recalc_min_len = in_next + core::cmp::min(in_end - in_next, 10000);
        let mut seq_idx: usize = 0;

        init_block_split_stats(&mut c.split_stats);
        deflate_begin_sequences(c, &mut p.sequences[0]);
        let mut min_len = calculate_min_match_len(
            &r#in[in_next..],
            in_max_block_end - in_next,
            max_search_depth,
        );

        loop {
            // Recalculate the minimum match length if it hasn't been done recently.
            if in_next >= next_recalc_min_len {
                min_len = recalculate_min_match_len(&c.freqs, max_search_depth);
                next_recalc_min_len +=
                    core::cmp::min(in_end - next_recalc_min_len, in_next - in_block_begin);
            }

            // Find the longest match at the current position.
            adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
            let mut cur_offset: u32 = 0;
            let mut cur_len = hc_matchfinder_longest_match(
                &mut p.hc_mf,
                r#in,
                &mut in_cur_base,
                in_next,
                min_len - 1,
                max_len,
                nice_len,
                max_search_depth,
                &mut next_hashes,
                &mut cur_offset,
            );

            // Note the threshold is 8192 here, where greedy uses 4096: the lazy parser
            // has already paid for a lookahead, so it is stricter about cheap matches.
            if cur_len < min_len || (cur_len == DEFLATE_MIN_MATCH_LEN && cur_offset > 8192) {
                // No match found. Choose a literal.
                let lit = r#in[in_next] as usize;
                in_next += 1;
                deflate_choose_literal(c, lit, true, &mut p.sequences[seq_idx]);
            } else {
                in_next += 1;

                // C: the `have_cur_match:` label. The C `goto`s back here after
                // promoting a lookahead match to the current one; a labelled loop is
                // the same control flow without a `goto`.
                'have_cur_match: loop {
                    // We have a match at the current position. If it's very long,
                    // choose it immediately.
                    if cur_len >= nice_len {
                        deflate_choose_match(
                            c,
                            cur_len,
                            cur_offset,
                            true,
                            &mut p.sequences,
                            &mut seq_idx,
                        );
                        hc_matchfinder_skip_bytes(
                            &mut p.hc_mf,
                            r#in,
                            &mut in_cur_base,
                            in_next,
                            in_end,
                            cur_len - 1,
                            &mut next_hashes,
                        );
                        in_next += cur_len as usize - 1;
                        break 'have_cur_match;
                    }

                    // Try to find a better match at the next position, at half depth.
                    adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                    let mut next_offset: u32 = 0;
                    let mut next_len = hc_matchfinder_longest_match(
                        &mut p.hc_mf,
                        r#in,
                        &mut in_cur_base,
                        in_next,
                        cur_len - 1,
                        max_len,
                        nice_len,
                        max_search_depth >> 1,
                        &mut next_hashes,
                        &mut next_offset,
                    );
                    in_next += 1;

                    if next_len >= cur_len
                        && 4 * (next_len - cur_len) as i32
                            + (bsr32(cur_offset) - bsr32(next_offset))
                            > 2
                    {
                        // Found a better match at the next position. Output a literal.
                        // Then the next match becomes the current match.
                        let lit = r#in[in_next - 2] as usize;
                        deflate_choose_literal(c, lit, true, &mut p.sequences[seq_idx]);
                        cur_len = next_len;
                        cur_offset = next_offset;
                        continue 'have_cur_match;
                    }

                    if lazy2 {
                        // In lazy2 mode, look ahead another position, at quarter depth.
                        adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                        next_len = hc_matchfinder_longest_match(
                            &mut p.hc_mf,
                            r#in,
                            &mut in_cur_base,
                            in_next,
                            cur_len - 1,
                            max_len,
                            nice_len,
                            max_search_depth >> 2,
                            &mut next_hashes,
                            &mut next_offset,
                        );
                        in_next += 1;

                        if next_len >= cur_len
                            && 4 * (next_len - cur_len) as i32
                                + (bsr32(cur_offset) - bsr32(next_offset))
                                > 6
                        {
                            // There's a much better match two positions ahead, so use
                            // two literals.
                            let l3 = r#in[in_next - 3] as usize;
                            deflate_choose_literal(c, l3, true, &mut p.sequences[seq_idx]);
                            let l2 = r#in[in_next - 2] as usize;
                            deflate_choose_literal(c, l2, true, &mut p.sequences[seq_idx]);
                            cur_len = next_len;
                            cur_offset = next_offset;
                            continue 'have_cur_match;
                        }
                        // No better match at either of the next 2 positions. Output the
                        // current match.
                        deflate_choose_match(
                            c,
                            cur_len,
                            cur_offset,
                            true,
                            &mut p.sequences,
                            &mut seq_idx,
                        );
                        if cur_len > 3 {
                            hc_matchfinder_skip_bytes(
                                &mut p.hc_mf,
                                r#in,
                                &mut in_cur_base,
                                in_next,
                                in_end,
                                cur_len - 3,
                                &mut next_hashes,
                            );
                            in_next += cur_len as usize - 3;
                        }
                    } else {
                        // No better match at the next position. Output the current
                        // match.
                        deflate_choose_match(
                            c,
                            cur_len,
                            cur_offset,
                            true,
                            &mut p.sequences,
                            &mut seq_idx,
                        );
                        hc_matchfinder_skip_bytes(
                            &mut p.hc_mf,
                            r#in,
                            &mut in_cur_base,
                            in_next,
                            in_end,
                            cur_len - 2,
                            &mut next_hashes,
                        );
                        in_next += cur_len as usize - 2;
                    }
                    break 'have_cur_match;
                }
            }

            // Check if it's time to output another block.
            if !(in_next < in_max_block_end
                && seq_idx < SEQ_STORE_LENGTH
                && !should_end_block(&mut c.split_stats, in_block_begin, in_next, in_end))
            {
                break;
            }
        }

        deflate_finish_block(
            c,
            os,
            &r#in[in_block_begin..],
            (in_next - in_block_begin) as u32,
            &p.sequences,
            in_next == in_end,
        );

        if in_next == in_end || os.overflow {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::ldx::compress::LdxCompressor;
    use std::io::Read;

    fn inflate(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::DeflateDecoder::new(bytes)
            .read_to_end(&mut out)
            .expect("emitted stream must inflate");
        out
    }

    fn compress_at(level: u32, data: &[u8]) -> Vec<u8> {
        let mut c = LdxCompressor::new(level).expect("level must be ported");
        let mut out = vec![0u8; data.len() * 2 + 65536];
        let n = c.compress(data, data.len(), &mut out);
        assert!(n > 0, "level {level}: output buffer reported too small");
        out.truncate(n);
        out
    }

    /// `bsr32` must be `floor(log2(v))` — the lazy score treats a difference of 1 as
    /// "one halving of the offset", so an off-by-one here silently reweights every
    /// lazy decision at L5-L9.
    #[test]
    fn bsr32_is_floor_log2() {
        for v in 1..=4096u32 {
            let want = (32 - v.leading_zeros() - 1) as i32;
            assert_eq!(bsr32(v), want, "v={v}");
        }
        assert_eq!(bsr32(1), 0);
        assert_eq!(bsr32(2), 1);
        assert_eq!(bsr32(3), 1);
        assert_eq!(bsr32(4), 2);
        assert_eq!(bsr32(32768), 15, "the largest legal DEFLATE offset");
    }

    /// Levels 5-9 end to end through an independent decoder, across shapes that
    /// exercise the lookahead, the block splitter and the min-match recalculation.
    #[test]
    fn levels_5_to_9_round_trip_end_to_end() {
        let mut cases: Vec<Vec<u8>> = Vec::new();

        // Across each level's passthrough boundary (55 - 4*level = 35..19).
        for n in 15..200usize {
            cases.push((0..n).map(|i| b"the quick brown fox "[i % 20]).collect());
        }

        let unit = b"the rain in spain falls mainly on the plain. ";
        let mut rep = Vec::new();
        while rep.len() < 400_000 {
            rep.extend_from_slice(unit);
        }
        cases.push(rep);

        let mut state: u32 = 0xDEAD_10CC;
        cases.push(
            (0..120_000)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (state >> 16) as u8
                })
                .collect(),
        );

        // Text turning into binary: forces a split, and moves the literal alphabet so
        // the mid-block `recalculate_min_match_len` actually changes `min_len`.
        let mut mixed: Vec<u8> = Vec::new();
        for i in 0..250_000 {
            mixed.push(if i < 120_000 {
                b"abcdefgh "[i % 9]
            } else {
                ((i * 7) % 256) as u8
            });
        }
        cases.push(mixed);

        // Four-symbol data — the case the min-match heuristic exists for.
        cases.push((0..200_000).map(|i| b"ACGT"[i % 4]).collect());

        for level in 5..=9u32 {
            for data in &cases {
                let bytes = compress_at(level, data);
                assert_eq!(
                    &inflate(&bytes),
                    data,
                    "L{level} round-trip failed for {} bytes",
                    data.len()
                );
            }
        }
    }

    /// Several 32 KiB window slides at every lazy level.
    #[test]
    fn round_trips_across_several_window_slides() {
        let mut data = Vec::new();
        let mut state: u32 = 0xC0DE_D00D;
        while data.len() < 250_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) & 0xFF;
            if r < 150 && data.len() > 50_000 {
                let back = 25_000 + (r as usize * 37) % 6_000;
                let start = data.len() - back;
                let n = 6 + (r as usize % 80);
                for k in 0..n {
                    let b = data[start + k];
                    data.push(b);
                }
            } else {
                data.push((r % 96) as u8);
            }
        }
        for level in 5..=9u32 {
            assert_eq!(inflate(&compress_at(level, &data)), data, "L{level}");
        }
    }

    /// Effort must rise across the WHOLE ladder now that L0-L9 exist. This is the
    /// invariant CLAUDE.md clause 5 asks for — it pins that every level's knobs are
    /// wired through without pinning what they are.
    ///
    /// Note this is asserted on `ldx`, which is libdeflate's ladder. Our SHIPPING
    /// ladder sags at L4 (`-4` is larger than `-3` on 10/11 TUNE files) — a defect
    /// recorded at `level.rs:267-278`, and one this port does not inherit.
    #[test]
    fn effort_rises_monotonically_across_levels_1_to_9() {
        let unit = b"the rain in spain falls mainly on the plain, and elsewhere besides. ";
        let mut data = Vec::new();
        while data.len() < 400_000 {
            data.extend_from_slice(unit);
        }

        let sizes: Vec<usize> = (1..=9u32).map(|l| compress_at(l, &data).len()).collect();
        for w in sizes.windows(2) {
            assert!(w[1] <= w[0], "the ladder sags: {sizes:?} (levels 1..=9)");
        }
        assert!(
            sizes[8] < sizes[0],
            "L9 ({}) is not smaller than L1 ({})",
            sizes[8],
            sizes[0]
        );
    }

    /// lazy2 (L8/L9) must not be WORSE than lazy (L7) at the same task, and the deeper
    /// search must actually be doing something.
    #[test]
    fn lazy2_is_at_least_as_good_as_lazy() {
        let mut data = Vec::new();
        let mut state: u32 = 0x1234_ABCD;
        while data.len() < 300_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) & 0xFF;
            if r < 120 && data.len() > 5_000 {
                let back = 100 + (r as usize * 29) % 4_000;
                let start = data.len() - back;
                let n = 4 + (r as usize % 30);
                for k in 0..n {
                    let b = data[start + k];
                    data.push(b);
                }
            } else {
                data.push((r % 50) as u8);
            }
        }
        let l7 = compress_at(7, &data).len();
        let l8 = compress_at(8, &data).len();
        let l9 = compress_at(9, &data).len();
        assert!(l8 <= l7, "lazy2 L8 ({l8}) is worse than lazy L7 ({l7})");
        assert!(l9 <= l8, "L9 ({l9}) is worse than L8 ({l8})");
    }
}
