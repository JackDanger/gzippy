//! C: `vendor/libdeflate/lib/deflate_compress.c:2380-2521` — `choose_max_block_end`
//! and `deflate_compress_fastest`, the level-1 compressor.
//!
//! This is a faster variant of `deflate_compress_greedy`. It uses the
//! `ht_matchfinder` rather than the `hc_matchfinder`. It also skips the block
//! splitting algorithm and just uses fixed length blocks. `c->max_search_depth` has no
//! effect with this algorithm, as it is hardcoded in `ht_matchfinder.h`.
//!
//! # Why this function specifically
//!
//! Of the 17 T1 cells where we LOSE to libdeflate, **15 are L1** — and L1 is the one
//! level where our encoder is not libdeflate's at all: we ship igzip's chainless
//! finder (`deflate/parse/fast.rs`, 3,380 lines) where libdeflate runs this. Every
//! other level is already a byte tie. So this pair of functions is very nearly the
//! whole phase-1 parity gap.
//!
//! **This does not mean routing L1 here will close those cells.** The binding FALSIFY
//! at `src/compress/deflate/parse/mod.rs:540` records two prior attempts, one dead on
//! size and one dead on the T1 wall at 1.2662x on a cell `main` already ties. What
//! this module changes is that a third attempt can be measured against the REAL
//! libdeflate algorithm instead of against a derivative of it.

use super::bitstream::DeflateOutputBitstream;
use super::flush::{Compressor, DeflateSequence};
use super::ht_matchfinder::{
    ht_matchfinder_longest_match, ht_matchfinder_skip_bytes, HtMatchfinder,
    HT_MATCHFINDER_REQUIRED_NBYTES,
};
use super::sequences::{
    deflate_begin_sequences, deflate_choose_literal, deflate_choose_match, deflate_finish_block,
};
use super::{
    DEFLATE_MAX_MATCH_LEN, FAST_SEQ_STORE_LENGTH, FAST_SOFT_MAX_BLOCK_LENGTH, MIN_BLOCK_LENGTH,
};

/// C: `choose_max_block_end(const u8 *in_block_begin, const u8 *in_end,
/// size_t soft_max_len)` (:2381)
///
/// If the remaining input is less than `soft_max_len + MIN_BLOCK_LENGTH`, take it all:
/// splitting there would leave a final block below `MIN_BLOCK_LENGTH`, whose Huffman
/// header would cost more than it saves.
#[inline(always)]
pub(crate) fn choose_max_block_end(
    in_block_begin: usize,
    in_end: usize,
    soft_max_len: usize,
) -> usize {
    if in_end - in_block_begin < soft_max_len + MIN_BLOCK_LENGTH {
        return in_end;
    }
    in_block_begin + soft_max_len
}

/// The parser state `deflate_compress_fastest` owns. C: the `p.f` arm of the
/// compressor's parser union (:530).
pub(crate) struct FastestState {
    pub(crate) ht_mf: HtMatchfinder,
    /// C: `struct deflate_sequence sequences[FAST_SEQ_STORE_LENGTH + 1]`
    pub(crate) sequences: Vec<DeflateSequence>,
}

impl FastestState {
    pub(crate) fn new() -> Self {
        Self {
            ht_mf: HtMatchfinder::new(),
            sequences: vec![DeflateSequence::default(); FAST_SEQ_STORE_LENGTH + 1],
        }
    }
}

/// C: `deflate_compress_fastest(struct libdeflate_compressor *c, const u8 *in,
/// size_t in_nbytes, struct deflate_output_bitstream *os)` (:2452)
///
/// # The short-tail path
///
/// When fewer than `HT_MATCHFINDER_REQUIRED_NBYTES` bytes remain, the matchfinder
/// cannot be called at all (it reads 5 bytes to compute the next hash), so the C
/// emits every remaining byte as a literal and `break`s out of the block loop. That
/// is why the last four bytes of any stream are always literals, regardless of how
/// well they would have matched.
///
/// # `nice_len` and `max_len` are clamped ONCE and stay clamped
///
/// Both are function-scope locals in the C, not per-iteration values: once the tail
/// shortens them they stay short for the rest of the stream. Since they only shrink at
/// the very end that is harmless, but recomputing them per iteration would be a
/// different program.
pub(crate) fn deflate_compress_fastest(
    c: &mut Compressor,
    p: &mut FastestState,
    r#in: &[u8],
    in_nbytes: usize,
    os: &mut DeflateOutputBitstream<'_>,
    nice_match_length: u32,
) {
    let mut in_next: usize = 0;
    let in_end: usize = in_nbytes;
    let mut in_cur_base: usize = 0;
    let mut max_len: u32 = DEFLATE_MAX_MATCH_LEN;
    let mut nice_len: u32 = core::cmp::min(nice_match_length, max_len);
    let mut next_hash: u32 = 0;

    p.ht_mf.init();

    loop {
        // Starting a new DEFLATE block.
        let in_block_begin = in_next;
        let in_max_block_end = choose_max_block_end(in_next, in_end, FAST_SOFT_MAX_BLOCK_LENGTH);
        let mut seq_idx: usize = 0;

        deflate_begin_sequences(c, &mut p.sequences[0]);

        loop {
            let remaining = in_end - in_next;

            if remaining < DEFLATE_MAX_MATCH_LEN as usize {
                max_len = remaining as u32;
                if max_len < HT_MATCHFINDER_REQUIRED_NBYTES {
                    // C: `do { deflate_choose_literal(...); } while (--max_len);`
                    while max_len != 0 {
                        let lit = r#in[in_next] as usize;
                        in_next += 1;
                        deflate_choose_literal(c, lit, false, &mut p.sequences[seq_idx]);
                        max_len -= 1;
                    }
                    break;
                }
                nice_len = core::cmp::min(nice_len, max_len);
            }

            let mut offset: u32 = 0;
            let length = ht_matchfinder_longest_match(
                &mut p.ht_mf,
                r#in,
                &mut in_cur_base,
                in_next,
                max_len,
                nice_len,
                &mut next_hash,
                &mut offset,
            );

            if length != 0 {
                // Match found.
                deflate_choose_match(c, length, offset, false, &mut p.sequences, &mut seq_idx);
                ht_matchfinder_skip_bytes(
                    &mut p.ht_mf,
                    r#in,
                    &mut in_cur_base,
                    in_next + 1,
                    in_end,
                    length - 1,
                    &mut next_hash,
                );
                in_next += length as usize;
            } else {
                // No match found.
                let lit = r#in[in_next] as usize;
                in_next += 1;
                deflate_choose_literal(c, lit, false, &mut p.sequences[seq_idx]);
            }

            // Check if it's time to output another block.
            if !(in_next < in_max_block_end && seq_idx < FAST_SEQ_STORE_LENGTH) {
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
    use std::io::Read;

    /// libdeflate's level 1: `nice_match_length = 32`, `max_search_depth` unused
    /// (hardcoded in the matchfinder as the bucket size). C: `:3926`.
    const L1_NICE_MATCH_LENGTH: u32 = 32;

    fn compress(data: &[u8]) -> Vec<u8> {
        let mut c = Compressor::new();
        let mut p = FastestState::new();
        // Generous, so a short buffer never masks a real result.
        let mut buf = vec![0u8; data.len() * 2 + 4096];
        let mut os = DeflateOutputBitstream::new(&mut buf);

        deflate_compress_fastest(
            &mut c,
            &mut p,
            data,
            data.len(),
            &mut os,
            L1_NICE_MATCH_LENGTH,
        );
        assert!(!os.overflow, "output buffer overflowed");

        let mut n = os.next;
        if os.bitcount > 0 {
            os.buf[n] = os.bitbuf as u8;
            n += 1;
        }
        buf.truncate(n);
        buf
    }

    fn inflate(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        flate2::read::DeflateDecoder::new(bytes)
            .read_to_end(&mut out)
            .expect("emitted stream must inflate");
        out
    }

    /// The gate that matters: whole-stream round-trip through an INDEPENDENT decoder.
    /// Anything that corrupts the matchfinder, the sequence store, the codes or the
    /// emitter shows up here.
    #[test]
    fn round_trips_across_shapes() {
        let mut cases: Vec<Vec<u8>> = vec![
            b"the quick brown fox jumps over the lazy dog, twice over now".to_vec(),
            vec![b'a'; 60],
        ];

        // Highly repetitive: exercises long matches and nice_len early-outs.
        let unit = b"the rain in spain falls mainly on the plain. ";
        let mut rep = Vec::new();
        while rep.len() < 300_000 {
            rep.extend_from_slice(unit);
        }
        cases.push(rep);

        // Incompressible: exercises the literal path and stored-block selection.
        let mut state: u32 = 0x1234_5678;
        cases.push(
            (0..120_000)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (state >> 16) as u8
                })
                .collect(),
        );

        // Mixed: text that turns into binary halfway.
        let mut mixed: Vec<u8> = Vec::new();
        for i in 0..80_000 {
            mixed.push(if i < 40_000 {
                b"abcdefgh "[i % 9]
            } else {
                ((i * 7) % 256) as u8
            });
        }
        cases.push(mixed);

        for data in &cases {
            let bytes = compress(data);
            assert_eq!(
                &inflate(&bytes),
                data,
                "round-trip failed for {} bytes",
                data.len()
            );
        }
    }

    /// Every length from 52 to 400 bytes. This is where the short-tail path, the
    /// `HT_MATCHFINDER_REQUIRED_NBYTES` cutover and the `max_len` clamp interact, and
    /// where an off-by-one produces a stream that inflates to the wrong length rather
    /// than failing outright.
    ///
    /// **52 is not an arbitrary floor.** `deflate_flush_block` ASSERTs
    /// `block_length >= MIN_BLOCK_LENGTH || (is_final_block && block_length > 0)`, and
    /// `libdeflate_deflate_compress` guarantees it by routing inputs of
    /// `max_passthrough_size` bytes or fewer — 51 at level 1 — to
    /// `deflate_compress_none` instead. Calling this function with a shorter input is
    /// something the C never does; a first draft of this test started at 0 and
    /// panicked in the cost model, which is how the passthrough got ported.
    /// `compress::tests::level_1_passthrough_boundary_is_51_bytes` covers 0..=51.
    #[test]
    fn every_short_length_round_trips() {
        for n in 52..=400usize {
            let data: Vec<u8> = (0..n).map(|i| b"abcabcabd"[i % 9]).collect();
            let bytes = compress(&data);
            assert_eq!(inflate(&bytes), data, "length {n} did not round-trip");
        }
    }

    /// Input larger than the 32 KiB match window, so the matchfinder slides at least
    /// three times mid-stream while the compressor is running for real.
    #[test]
    fn round_trips_across_several_window_slides() {
        let mut data = Vec::new();
        let mut state: u32 = 0xFEED_FACE;
        while data.len() < 150_000 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let r = (state >> 16) & 0xFF;
            if r < 140 && data.len() > 40_000 {
                // Reference something well outside the current window.
                let back = 20_000 + (r as usize * 31) % 12_000;
                let start = data.len() - back;
                let n = 8 + (r as usize % 60);
                for k in 0..n {
                    let b = data[start + k];
                    data.push(b);
                }
            } else {
                data.push((r % 90) as u8);
            }
        }
        let bytes = compress(&data);
        assert_eq!(inflate(&bytes), data);
    }

    /// It must actually COMPRESS — a round-trip test passes just as happily if every
    /// block is stored, which would prove nothing about the matchfinder.
    #[test]
    fn repetitive_input_is_actually_compressed() {
        let unit = b"the rain in spain falls mainly on the plain. ";
        let mut data = Vec::new();
        while data.len() < 200_000 {
            data.extend_from_slice(unit);
        }
        let bytes = compress(&data);
        let ratio = bytes.len() as f64 / data.len() as f64;
        assert!(
            ratio < 0.02,
            "highly repetitive input compressed to {ratio:.4} of its size — \
             the matchfinder is finding nothing"
        );
    }

    /// `choose_max_block_end` must never leave a final block shorter than
    /// `MIN_BLOCK_LENGTH`: either it takes everything, or it leaves at least that much.
    #[test]
    fn choose_max_block_end_never_strands_a_tiny_final_block() {
        let soft = FAST_SOFT_MAX_BLOCK_LENGTH;
        for total in [
            0usize,
            1,
            100,
            soft - 1,
            soft,
            soft + 1,
            soft + MIN_BLOCK_LENGTH - 1,
            soft + MIN_BLOCK_LENGTH,
            soft * 3,
        ] {
            let end = choose_max_block_end(0, total, soft);
            assert!(end <= total);
            if end != total {
                assert!(
                    total - end >= MIN_BLOCK_LENGTH,
                    "total={total}: split at {end} strands {} bytes",
                    total - end
                );
            }
        }
    }
}
