//! C: `vendor/libdeflate/lib/deflate_compress.c:2042-2051` and `:2224-2270` — closing
//! a block, and the three calls a parser makes while filling the sequence store.
//!
//! # The sequence store is written as a rolling "current" entry
//!
//! `deflate_choose_literal` only increments a counter; the literal bytes themselves
//! are never copied, because `deflate_flush_block` reads them straight out of the
//! input. `deflate_choose_match` closes the current entry by writing the match into
//! its high bits and then opens the next one. So a block's store always ends with a
//! half-open entry whose `length` field is 0, and THAT is the terminator
//! `deflate_flush_block` breaks on — there is no count.

use super::codes::deflate_make_huffman_codes;
use super::flush::{deflate_flush_block, Compressor, DeflateSequence, SEQ_LENGTH_SHIFT};
use super::split::{observe_literal, observe_match};
use super::tables::{deflate_get_offset_slot, DEFLATE_LENGTH_SLOT};
use super::{DEFLATE_END_OF_BLOCK, DEFLATE_FIRST_LEN_SYM, MAX_BLOCK_LENGTH};
use crate::compress::ldx::bitstream::DeflateOutputBitstream;
use crate::compress::ldx::flush::SEQ_LITRUNLEN_MASK;

/// C: `deflate_finish_block(...)` (:2042)
///
/// Count the end-of-block symbol, build the block's Huffman codes from the
/// accumulated frequencies, and flush.
///
/// The end-of-block frequency is incremented HERE and not by the parser, so a parser
/// that never emits a symbol still produces a well-formed block. It is also why
/// `deflate_flush_block` can assume `codes.lens.litlen[256] != 0`.
#[inline(always)]
pub(crate) fn deflate_finish_block(
    c: &mut Compressor,
    os: &mut DeflateOutputBitstream<'_>,
    block_begin: &[u8],
    block_length: u32,
    sequences: &[DeflateSequence],
    is_final_block: bool,
) {
    c.freqs.litlen[DEFLATE_END_OF_BLOCK as usize] += 1;
    // `freqs` and `codes` are disjoint fields, so a destructuring borrow gives the C's
    // `(&c->freqs, &c->codes)` pair directly. Copying the 1,280-byte frequency table
    // in and out to satisfy the borrow checker would be real work on a per-block path.
    let Compressor { freqs, codes, .. } = &mut *c;
    deflate_make_huffman_codes(freqs, codes);
    deflate_flush_block(c, os, block_begin, block_length, sequences, is_final_block);
}

/// C: `deflate_begin_sequences(struct libdeflate_compressor *c,
/// struct deflate_sequence *first_seq)` (:2224)
pub(crate) fn deflate_begin_sequences(c: &mut Compressor, first_seq: &mut DeflateSequence) {
    // C: deflate_reset_symbol_frequencies(c)
    c.freqs.reset();
    first_seq.litrunlen_and_length = 0;
}

/// C: `deflate_choose_literal(...)` (:2232)
///
/// `gather_split_stats` is a compile-time `bool` in the C, passed as a literal so the
/// `forceinline` specialises both ways: the near-optimal parser's second pass has
/// already chosen its boundaries and must not perturb the statistics.
#[inline(always)]
pub(crate) fn deflate_choose_literal(
    c: &mut Compressor,
    literal: usize,
    gather_split_stats: bool,
    seq: &mut DeflateSequence,
) {
    // `literal` is a byte read from the input; `freqs.litlen` has
    // DEFLATE_NUM_LITLEN_SYMS (288) entries. The C indexes with a `u8` widened in
    // place and emits no check.
    crate::anatomy_count!(literals_emitted);
    crate::anatomy_count!(histogram_updates);
    debug_assert!(literal < c.freqs.litlen.len());
    unsafe { *c.freqs.litlen.get_unchecked_mut(literal) += 1 };

    if gather_split_stats {
        observe_literal(&mut c.split_stats, literal as u8);
    }

    // The literal run counter shares a u32 with the match length, so the whole run
    // must fit in SEQ_LITRUNLEN_MASK.
    const _: () = assert!(MAX_BLOCK_LENGTH as u32 <= SEQ_LITRUNLEN_MASK);
    seq.litrunlen_and_length += 1;
}

/// C: `deflate_choose_match(...)` (:2246)
///
/// Closes the current sequence with this match and opens the next one. The C takes
/// `struct deflate_sequence **seq_p` and advances the caller's pointer; we take the
/// store plus an index and advance the index, which is the same bookkeeping without
/// pointer arithmetic.
///
/// Note the `|=` on `litrunlen_and_length`: the literal run count is already in the
/// low bits, and the length is OR-ed into the high bits. An `=` would silently
/// discard the run.
#[inline(always)]
pub(crate) fn deflate_choose_match(
    c: &mut Compressor,
    length: u32,
    offset: u32,
    gather_split_stats: bool,
    sequences: &mut [DeflateSequence],
    seq_idx: &mut usize,
) {
    // `length <= DEFLATE_MAX_MATCH_LEN`, which is the last index of
    // DEFLATE_LENGTH_SLOT; `length_slot < 29` so the litlen index stays under 288;
    // `deflate_get_offset_slot` returns < DEFLATE_NUM_OFFSET_SYMS by construction.
    crate::anatomy_count!(matches_emitted);
    crate::anatomy_count!(match_length_bytes_total, length as u64);
    // A match updates TWO histograms (litlen and offset); a literal updates one.
    // That is the legacy accounting: 60,363 literals + 2 x 112,068 matches = 284,499.
    crate::anatomy_count!(histogram_updates, 2u64);
    debug_assert!((length as usize) < DEFLATE_LENGTH_SLOT.len());
    let length_slot = unsafe { *DEFLATE_LENGTH_SLOT.get_unchecked(length as usize) } as usize;
    let offset_slot = deflate_get_offset_slot(offset) as usize;

    debug_assert!(DEFLATE_FIRST_LEN_SYM as usize + length_slot < c.freqs.litlen.len());
    debug_assert!(offset_slot < c.freqs.offset.len());
    unsafe {
        *c.freqs
            .litlen
            .get_unchecked_mut(DEFLATE_FIRST_LEN_SYM as usize + length_slot) += 1;
        *c.freqs.offset.get_unchecked_mut(offset_slot) += 1;
    }
    if gather_split_stats {
        observe_match(&mut c.split_stats, length);
    }

    // Every caller guards `seq_idx < SEQ_STORE_LENGTH` while `sequences` is
    // SEQ_STORE_LENGTH + 1 long precisely so this advance and the following reset
    // are both in range — that +1 IS the C's guarantee.
    debug_assert!(*seq_idx + 1 < sequences.len());
    let seq = unsafe { sequences.get_unchecked_mut(*seq_idx) };
    seq.litrunlen_and_length |= length << SEQ_LENGTH_SHIFT;
    seq.offset = offset as u16;
    seq.offset_slot = offset_slot as u16;

    *seq_idx += 1;
    unsafe { sequences.get_unchecked_mut(*seq_idx).litrunlen_and_length = 0 };
}

/// C: `adjust_max_and_nice_len(u32 *max_len, u32 *nice_len, size_t remaining)` (:2266)
///
/// Decrease the maximum and nice match lengths if we're approaching the end of the
/// input buffer.
#[inline(always)]
pub(crate) fn adjust_max_and_nice_len(max_len: &mut u32, nice_len: &mut u32, remaining: usize) {
    if remaining < super::DEFLATE_MAX_MATCH_LEN as usize {
        *max_len = remaining as u32;
        *nice_len = core::cmp::min(*nice_len, *max_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::ldx::flush::SEQ_LITRUNLEN_MASK;

    /// A literal run followed by a match must pack both into one `u32` without
    /// either clobbering the other. `|=` vs `=` is the whole bug class here.
    #[test]
    fn a_literal_run_and_a_match_share_one_word() {
        let mut c = Compressor::new();
        let mut seqs = [DeflateSequence::default(); 4];
        let mut idx = 0usize;

        deflate_begin_sequences(&mut c, &mut seqs[0]);
        for &b in b"abcde" {
            deflate_choose_literal(&mut c, b as usize, true, &mut seqs[idx]);
        }
        deflate_choose_match(&mut c, 42, 300, true, &mut seqs, &mut idx);

        assert_eq!(idx, 1, "the store must have advanced");
        assert_eq!(
            seqs[0].litrunlen_and_length & SEQ_LITRUNLEN_MASK,
            5,
            "the literal run must survive the match write"
        );
        assert_eq!(seqs[0].litrunlen_and_length >> SEQ_LENGTH_SHIFT, 42);
        assert_eq!(seqs[0].offset, 300);
        assert_eq!(
            seqs[0].offset_slot,
            deflate_get_offset_slot(300) as u16,
            "offset slot must come from the ported map, not be recomputed"
        );
        assert_eq!(
            seqs[1].litrunlen_and_length, 0,
            "the next entry must be opened empty — it is the terminator"
        );
    }

    /// Frequencies must be counted for exactly the symbols that will be emitted:
    /// one per literal, one for the length SLOT (not the length), one for the offset
    /// slot. A miscount produces a valid but suboptimal — or invalid — code.
    #[test]
    fn frequencies_track_the_symbols_that_will_be_emitted() {
        let mut c = Compressor::new();
        let mut seqs = [DeflateSequence::default(); 4];
        let mut idx = 0usize;

        deflate_begin_sequences(&mut c, &mut seqs[0]);
        deflate_choose_literal(&mut c, b'x' as usize, true, &mut seqs[idx]);
        deflate_choose_literal(&mut c, b'x' as usize, true, &mut seqs[idx]);
        deflate_choose_match(&mut c, 10, 5, true, &mut seqs, &mut idx);

        assert_eq!(c.freqs.litlen[b'x' as usize], 2);

        let len_slot = DEFLATE_LENGTH_SLOT[10] as usize;
        assert_eq!(c.freqs.litlen[DEFLATE_FIRST_LEN_SYM as usize + len_slot], 1);
        assert_eq!(c.freqs.offset[deflate_get_offset_slot(5) as usize], 1);

        // Nothing else was touched.
        let litlen_total: u32 = c.freqs.litlen.iter().sum();
        let offset_total: u32 = c.freqs.offset.iter().sum();
        assert_eq!(litlen_total, 3);
        assert_eq!(offset_total, 1);
    }

    /// `gather_split_stats = false` must leave the statistics completely untouched.
    /// The near-optimal parser's second pass depends on it — perturbing the stats
    /// there would move a boundary that has already been chosen.
    #[test]
    fn split_stats_are_only_gathered_when_asked() {
        let mut c = Compressor::new();
        let mut seqs = [DeflateSequence::default(); 4];
        let mut idx = 0usize;

        deflate_begin_sequences(&mut c, &mut seqs[0]);
        for _ in 0..100 {
            deflate_choose_literal(&mut c, b'q' as usize, false, &mut seqs[idx]);
        }
        deflate_choose_match(&mut c, 20, 7, false, &mut seqs, &mut idx);

        assert_eq!(c.split_stats.num_new_observations, 0);
        assert_eq!(c.split_stats.num_observations, 0);
        assert!(c.split_stats.new_observations.iter().all(|&o| o == 0));
    }

    /// `deflate_begin_sequences` must clear the frequencies from the previous block.
    #[test]
    fn begin_sequences_clears_the_previous_blocks_frequencies() {
        let mut c = Compressor::new();
        let mut seqs = [DeflateSequence::default(); 4];

        c.freqs.litlen[b'z' as usize] = 999;
        c.freqs.offset[3] = 7;
        seqs[0].litrunlen_and_length = 12345;

        deflate_begin_sequences(&mut c, &mut seqs[0]);

        assert!(c.freqs.litlen.iter().all(|&f| f == 0));
        assert!(c.freqs.offset.iter().all(|&f| f == 0));
        assert_eq!(seqs[0].litrunlen_and_length, 0);
    }

    /// `adjust_max_and_nice_len` only fires near the end of the input, and must never
    /// raise `nice_len` above `max_len`.
    #[test]
    fn adjust_max_and_nice_len_only_clamps_near_the_end() {
        let (mut max_len, mut nice_len) = (258u32, 32u32);
        adjust_max_and_nice_len(&mut max_len, &mut nice_len, 100_000);
        assert_eq!(
            (max_len, nice_len),
            (258, 32),
            "far from the end: unchanged"
        );

        let (mut max_len, mut nice_len) = (258u32, 32u32);
        adjust_max_and_nice_len(&mut max_len, &mut nice_len, 100);
        assert_eq!((max_len, nice_len), (100, 32));

        let (mut max_len, mut nice_len) = (258u32, 32u32);
        adjust_max_and_nice_len(&mut max_len, &mut nice_len, 10);
        assert_eq!(
            (max_len, nice_len),
            (10, 10),
            "nice_len must not exceed max"
        );

        // Exactly DEFLATE_MAX_MATCH_LEN remaining is NOT "approaching the end".
        let (mut max_len, mut nice_len) = (258u32, 32u32);
        adjust_max_and_nice_len(&mut max_len, &mut nice_len, 258);
        assert_eq!((max_len, nice_len), (258, 32));
    }

    /// End to end at this rung: hand-drive the sequence store the way a parser will,
    /// close the block with `deflate_finish_block`, and inflate the result with an
    /// independent decoder. This is the first test where the store, the frequency
    /// counting, the code construction and the emitter all run together.
    #[test]
    fn a_hand_driven_block_round_trips() {
        use std::io::Read;

        let data = b"the rain in spain the rain in spain the rain in spain".to_vec();

        let mut c = Compressor::new();
        let mut seqs = [DeflateSequence::default(); 8];
        let mut idx = 0usize;
        deflate_begin_sequences(&mut c, &mut seqs[0]);

        // "the rain in spain " literally (18 bytes), then two matches back 18.
        let prefix = 18usize;
        for &b in &data[..prefix] {
            deflate_choose_literal(&mut c, b as usize, true, &mut seqs[idx]);
        }
        deflate_choose_match(&mut c, 18, 18, true, &mut seqs, &mut idx);
        deflate_choose_match(
            &mut c,
            (data.len() - 36) as u32,
            18,
            true,
            &mut seqs,
            &mut idx,
        );

        let mut buf = vec![0u8; 4096];
        let mut os = DeflateOutputBitstream::new(&mut buf);
        deflate_finish_block(&mut c, &mut os, &data, data.len() as u32, &seqs, true);
        assert!(!os.overflow);

        let mut n = os.next;
        if os.bitcount > 0 {
            os.buf[n] = os.bitbuf as u8;
            n += 1;
        }
        buf.truncate(n);

        let mut out = Vec::new();
        flate2::read::DeflateDecoder::new(&buf[..])
            .read_to_end(&mut out)
            .expect("must inflate");
        assert_eq!(out, data);
    }
}
