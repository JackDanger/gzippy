//! C: `vendor/libdeflate/lib/deflate_compress.c:2524-2603` —
//! `deflate_compress_greedy`, the "greedy" DEFLATE compressor. It always chooses the
//! longest match.
//!
//! **This gates levels 2, 3 AND 4** — libdeflate uses greedy for all three (`:3931`,
//! `:3936`, `:3941`), with `(max_search_depth, nice_match_length)` of (6,10), (12,14)
//! and (16,30).
//!
//! # L3 is where we currently DIVERGE, and we win there
//!
//! Our shipping L3 is LAZY(12,14) against libdeflate's GREEDY(12,14) — same knobs,
//! different parser — and ours is smaller on 20 of 22 files, median ~44 KB. So the
//! differential at L3 is expected to match THEIR choice, which is phase-1 parity
//! succeeding, not a regression. Our lazy L3 is a phase-2 win to re-layer once every
//! cell is a tie. Do not "fix" the port to keep our L3.

use super::bitstream::DeflateOutputBitstream;
use super::compress_fastest::choose_max_block_end;
use super::flush::{Compressor, DeflateSequence};
use super::hc_matchfinder::{
    hc_matchfinder_longest_match, hc_matchfinder_skip_bytes, HcMatchfinder,
};
use super::min_match::calculate_min_match_len;
use super::sequences::{
    adjust_max_and_nice_len, deflate_begin_sequences, deflate_choose_literal, deflate_choose_match,
    deflate_finish_block,
};
use super::split::{init_block_split_stats, should_end_block};
use super::{
    DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN, SEQ_STORE_LENGTH, SOFT_MAX_BLOCK_LENGTH,
};

/// The parser state `deflate_compress_greedy` (and later the lazy parsers) owns.
/// C: the `p.g` arm of the compressor's parser union (:520).
pub(crate) struct GreedyState {
    pub(crate) hc_mf: HcMatchfinder,
    /// C: `struct deflate_sequence sequences[SEQ_STORE_LENGTH + 1]`
    pub(crate) sequences: Vec<DeflateSequence>,
}

impl GreedyState {
    pub(crate) fn new() -> Self {
        Self {
            hc_mf: HcMatchfinder::new(),
            sequences: vec![DeflateSequence::default(); SEQ_STORE_LENGTH + 1],
        }
    }
}

/// C: `deflate_compress_greedy(...)` (:2528)
///
/// # The match-acceptance test has TWO clauses, and the second one is easy to miss
///
/// ```c
/// if (length >= min_len && (length > DEFLATE_MIN_MATCH_LEN || offset <= 4096))
/// ```
///
/// A length-3 match is only taken if its offset is 4096 or less. That is a cost
/// judgement, not a safety check: a length-3 match at a long offset needs a wide offset
/// code, which typically costs more than the three literals it replaces. Dropping the
/// clause produces valid, larger output on exactly the binaries where offsets run long.
///
/// # `min_len` is computed ONCE PER BLOCK, from the block's own first 4 KiB
///
/// `calculate_min_match_len` is called inside the outer (per-block) loop with
/// `in_max_block_end - in_next`, so each block re-approximates from its own data. The C
/// also has `recalculate_min_match_len`, but greedy does not use it — only the lazy
/// parsers do. Hoisting this out of the loop would be a different program.
#[inline(never)]
pub(crate) fn deflate_compress_greedy(
    c: &mut Compressor,
    p: &mut GreedyState,
    r#in: &[u8],
    in_nbytes: usize,
    os: &mut DeflateOutputBitstream<'_>,
    max_search_depth: u32,
    nice_match_length: u32,
    good_match: u32,
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
        let mut seq_idx: usize = 0;

        init_block_split_stats(&mut c.split_stats);
        deflate_begin_sequences(c, unsafe { p.sequences.get_unchecked_mut(0) });
        let min_len = calculate_min_match_len(
            unsafe { r#in.get_unchecked(in_next..) },
            in_max_block_end - in_next,
            max_search_depth,
        );

        loop {
            adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);

            let mut offset: u32 = 0;
            let length = hc_matchfinder_longest_match(
                &mut p.hc_mf,
                r#in,
                &mut in_cur_base,
                in_next,
                min_len - 1,
                max_len,
                nice_len,
                max_search_depth,
                good_match,
                &mut next_hashes,
                &mut offset,
            );

            if length >= min_len && (length > DEFLATE_MIN_MATCH_LEN || offset <= 4096) {
                // Match found.
                deflate_choose_match(c, length, offset, true, &mut p.sequences, &mut seq_idx);
                hc_matchfinder_skip_bytes(
                    &mut p.hc_mf,
                    r#in,
                    &mut in_cur_base,
                    in_next + 1,
                    in_end,
                    length - 1,
                    &mut next_hashes,
                );
                in_next += length as usize;
            } else {
                // No match found.
                debug_assert!(in_next < r#in.len());
                let lit = unsafe { *r#in.get_unchecked(in_next) } as usize;
                in_next += 1;
                deflate_choose_literal(c, lit, true, unsafe {
                    p.sequences.get_unchecked_mut(seq_idx)
                });
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
            unsafe { r#in.get_unchecked(in_block_begin..) },
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

    /// Levels 2, 3 and 4 end to end through an independent decoder, across shapes that
    /// exercise the block splitter, the sequence-store cap, long matches and the
    /// literal-only path.
    #[test]
    fn levels_2_3_4_round_trip_end_to_end() {
        let mut cases: Vec<Vec<u8>> = Vec::new();

        // Across the level-2/3/4 passthrough boundaries (55 - 4*level = 47/43/39).
        for n in 30..200usize {
            cases.push((0..n).map(|i| b"the quick brown fox "[i % 20]).collect());
        }

        let unit = b"the rain in spain falls mainly on the plain. ";
        let mut rep = Vec::new();
        while rep.len() < 400_000 {
            rep.extend_from_slice(unit);
        }
        cases.push(rep);

        // Incompressible.
        let mut state: u32 = 0x600D_F00D;
        cases.push(
            (0..120_000)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (state >> 16) as u8
                })
                .collect(),
        );

        // Text that turns into binary — forces a block split.
        let mut mixed: Vec<u8> = Vec::new();
        for i in 0..200_000 {
            mixed.push(if i < 100_000 {
                b"abcdefgh "[i % 9]
            } else {
                ((i * 7) % 256) as u8
            });
        }
        cases.push(mixed);

        // Low-variety data, which is what the min-match heuristic exists for.
        cases.push((0..150_000).map(|i| b"ACGT"[i % 4]).collect());

        for level in [2u32, 3, 4] {
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

    /// Data spanning several 32 KiB window slides, at every greedy level.
    #[test]
    fn round_trips_across_several_window_slides() {
        let mut data = Vec::new();
        let mut state: u32 = 0xFACE_B00C;
        while data.len() < 200_000 {
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
        for level in [2u32, 3, 4] {
            assert_eq!(inflate(&compress_at(level, &data)), data, "L{level}");
        }
    }

    /// Effort must rise with level: L4 (depth 16, nice 30) must not produce a LARGER
    /// output than L2 (depth 6, nice 10) on compressible data. This is CLAUDE.md
    /// clause 5's "test the INVARIANT, not the VALUE" — it pins that the knobs are
    /// wired through, without pinning what they are.
    #[test]
    fn effort_rises_with_level() {
        let unit = b"the rain in spain falls mainly on the plain, and elsewhere too. ";
        let mut data = Vec::new();
        while data.len() < 300_000 {
            data.extend_from_slice(unit);
        }
        let l2 = compress_at(2, &data).len();
        let l3 = compress_at(3, &data).len();
        let l4 = compress_at(4, &data).len();
        assert!(l3 <= l2, "L3 ({l3}) is larger than L2 ({l2})");
        assert!(l4 <= l3, "L4 ({l4}) is larger than L3 ({l3})");
    }

    /// It must actually compress — a round-trip passes just as happily if every block
    /// is stored, which would prove nothing about the matchfinder.
    #[test]
    fn repetitive_input_is_actually_compressed() {
        let unit = b"the rain in spain falls mainly on the plain. ";
        let mut data = Vec::new();
        while data.len() < 300_000 {
            data.extend_from_slice(unit);
        }
        for level in [2u32, 3, 4] {
            let ratio = compress_at(level, &data).len() as f64 / data.len() as f64;
            assert!(
                ratio < 0.01,
                "L{level} compressed to {ratio:.4} — finder is idle"
            );
        }
    }
}
