//! Lempel-Ziv parsers (greedy / lazy / lazy2) + block emission.
//!
//! These transliterate libdeflate's `deflate_compress_greedy` and
//! `deflate_compress_lazy_generic` (`vendor/libdeflate/lib/deflate_compress.c`
//! ~:2528-2808) on top of Increment 1's substrate: the [`super::matchfinder`]
//! primitives (now with the hash-chains finder in [`super::matchfinder::hc`]),
//! the length-limited [`super::huffman`] code builder + dynamic header, the
//! [`super::block_split`] statistic, and the word-oriented [`super::bitstream`].
//!
//! Unlike the vendor, which streams symbols into a fixed sequence buffer and
//! flushes with a bit-exact cost model, we accumulate a small [`Token`] stream
//! per block and choose the cheaper of a dynamic-Huffman, static-Huffman, or
//! stored block from computed bit costs. Any valid parse + valid Huffman coding
//! roundtrips; the exact block-type decision only affects ratio, not
//! correctness. Match finding, the min-match-length heuristics, the lazy
//! offset-cost tie-break, and the block-split boundaries are faithful ports so
//! the ratio tracks libdeflate.

use super::bitstream::BitWriter;
use super::block_split::{BlockSplitStats, MIN_BLOCK_LENGTH};
use super::huffman::{build_dynamic_header, make_huffman_code, HuffmanCode};
use super::level::{LevelParams, Strategy};
use super::tables::{
    length_slot, offset_slot, static_litlen_freqs, static_offset_freqs,
    DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, DEFLATE_BLOCKTYPE_STATIC_HUFFMAN, DEFLATE_END_OF_BLOCK,
    DEFLATE_FIRST_LEN_SYM, DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN, DEFLATE_NUM_LITLEN_SYMS,
    DEFLATE_NUM_OFFSET_SYMS, LENGTH_EXTRA_BITS, LENGTH_SLOT_BASE, MAX_LITLEN_CODEWORD_LEN,
    MAX_OFFSET_CODEWORD_LEN, OFFSET_EXTRA_BITS, OFFSET_SLOT_BASE,
};

mod greedy;
mod lazy;
mod near_optimal;

/// Number of trailing pad bytes appended to the matchfinder's working buffer so
/// its speculative 4-byte / 8-byte loads never read out of bounds.
pub(super) const BUF_PAD: usize = 16;

/// `SOFT_MAX_BLOCK_LENGTH` — soft cap on the bytes covered by one block.
const SOFT_MAX_BLOCK_LENGTH: usize = 300_000;
/// `SEQ_STORE_LENGTH` — cap on the number of match "sequences" per block.
const SEQ_STORE_LENGTH: usize = 50_000;

/// Number of DEFLATE literal symbols (0..=255).
const NUM_LITERALS: usize = 256;

/// One parsed token: a literal byte or a back-reference.
#[derive(Clone, Copy)]
enum Token {
    Literal(u8),
    /// `length` in 3..=258, `offset` in 1..=32768.
    Match {
        length: u16,
        offset: u16,
    },
}

/// The precomputed RFC 1951 fixed (static) Huffman codes, built once per parse.
struct StaticCodes {
    litcode: HuffmanCode,
    offcode: HuffmanCode,
}

impl StaticCodes {
    fn build() -> Self {
        StaticCodes {
            litcode: make_huffman_code(
                DEFLATE_NUM_LITLEN_SYMS,
                MAX_LITLEN_CODEWORD_LEN,
                &static_litlen_freqs(),
            ),
            offcode: make_huffman_code(
                DEFLATE_NUM_OFFSET_SYMS,
                MAX_OFFSET_CODEWORD_LEN,
                &static_offset_freqs(),
            ),
        }
    }
}

/// Per-block accumulator: tokens, symbol frequencies, split stats.
struct Sink {
    tokens: Vec<Token>,
    litlen_freqs: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: [u32; DEFLATE_NUM_OFFSET_SYMS],
    /// Number of match sequences emitted (the `SEQ_STORE_LENGTH` counter).
    num_seqs: usize,
    /// Input bytes covered by the current block so far.
    block_length: usize,
    stats: BlockSplitStats,
}

impl Sink {
    fn new() -> Self {
        Sink {
            tokens: Vec::new(),
            litlen_freqs: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset_freqs: [0; DEFLATE_NUM_OFFSET_SYMS],
            num_seqs: 0,
            block_length: 0,
            stats: BlockSplitStats::new(),
        }
    }

    /// `deflate_begin_sequences` + `init_block_split_stats`.
    fn begin(&mut self) {
        self.tokens.clear();
        self.litlen_freqs = [0; DEFLATE_NUM_LITLEN_SYMS];
        self.offset_freqs = [0; DEFLATE_NUM_OFFSET_SYMS];
        self.num_seqs = 0;
        self.block_length = 0;
        self.stats.reset();
    }

    /// `deflate_choose_literal` (with split-stat gathering always on, as greedy
    /// and lazy do).
    #[inline]
    fn push_literal(&mut self, lit: u8) {
        self.litlen_freqs[lit as usize] += 1;
        self.stats.observe_literal(lit);
        self.tokens.push(Token::Literal(lit));
        self.block_length += 1;
    }

    /// `deflate_choose_match`.
    #[inline]
    fn push_match(&mut self, length: u32, offset: u32) {
        debug_assert!((DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN).contains(&length));
        debug_assert!((1..=32768).contains(&offset));
        let ls = length_slot(length) as usize;
        let os = offset_slot(offset) as usize;
        self.litlen_freqs[DEFLATE_FIRST_LEN_SYM + ls] += 1;
        self.offset_freqs[os] += 1;
        self.stats.observe_match(length);
        self.tokens.push(Token::Match {
            length: length as u16,
            offset: offset as u16,
        });
        self.num_seqs += 1;
        self.block_length += length as usize;
    }
}

/// Compress `buf[data_start..in_end]` into DEFLATE blocks appended to `bw`.
///
/// `buf` MUST have at least [`BUF_PAD`] trailing bytes beyond `in_end`. Bytes in
/// `buf[..data_start]` (a preset dictionary) are seeded into the matchfinder but
/// not coded; matches may reference them. Only greedy/lazy/lazy2 strategies are
/// handled here — the caller routes `Strategy::Stored` elsewhere.
pub(super) fn compress(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    bw: &mut BitWriter,
) {
    let statics = StaticCodes::build();
    match params.strategy {
        Strategy::Greedy => greedy::run(buf, data_start, in_end, params, &statics, bw),
        Strategy::Lazy => lazy::run(buf, data_start, in_end, params, &statics, bw, false),
        Strategy::Lazy2 => lazy::run(buf, data_start, in_end, params, &statics, bw, true),
        Strategy::NearOptimal => near_optimal::run(buf, data_start, in_end, params, &statics, bw),
        Strategy::Stored => unreachable!("stored strategy is handled by the caller"),
    }
}

// ---- block-boundary helpers ----

/// `choose_max_block_end`: the soft byte limit for a block starting at
/// `block_begin`.
#[inline]
fn choose_max_block_end(block_begin: usize, in_end: usize) -> usize {
    if in_end - block_begin < SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH {
        in_end
    } else {
        block_begin + SOFT_MAX_BLOCK_LENGTH
    }
}

/// `adjust_max_and_nice_len`: clamp match lengths near the end of input.
#[inline]
fn adjust_max_and_nice_len(max_len: &mut u32, nice_len: &mut u32, remaining: usize) {
    if remaining < DEFLATE_MAX_MATCH_LEN as usize {
        *max_len = remaining as u32;
        *nice_len = (*nice_len).min(*max_len);
    }
}

/// Whether the block loop should continue after emitting a token.
#[inline]
fn continue_block(
    sink: &mut Sink,
    in_next: usize,
    block_begin: usize,
    in_max_block_end: usize,
    in_end: usize,
) -> bool {
    in_next < in_max_block_end
        && sink.num_seqs < SEQ_STORE_LENGTH
        && !sink
            .stats
            .should_end_block(in_next - block_begin, in_end - in_next)
}

// ---- minimum-match-length heuristics ----

/// `choose_min_match_len`.
pub(crate) fn choose_min_match_len(num_used_literals: u32, max_search_depth: u32) -> u32 {
    // map from num_used_literals to min_len (`min_lens[]`, the rest is 3).
    const MIN_LENS: [u8; 80] = [
        9, 9, 9, 9, 9, 9, 8, 8, 7, 7, 6, 6, 6, 6, 6, 6, //
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, //
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, //
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, //
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ];
    if num_used_literals as usize >= MIN_LENS.len() {
        return 3;
    }
    let mut min_len = MIN_LENS[num_used_literals as usize] as u32;
    if max_search_depth < 16 {
        if max_search_depth < 5 {
            min_len = min_len.min(4);
        } else if max_search_depth < 10 {
            min_len = min_len.min(5);
        } else {
            min_len = min_len.min(7);
        }
    }
    min_len
}

/// `calculate_min_match_len`: initial approximation from the first 4 KiB.
fn calculate_min_match_len(data: &[u8], max_search_depth: u32) -> u32 {
    if data.len() < 512 {
        return DEFLATE_MIN_MATCH_LEN;
    }
    let scan = &data[..data.len().min(4096)];
    let mut used = [false; 256];
    for &b in scan {
        used[b as usize] = true;
    }
    let num_used_literals = used.iter().filter(|&&u| u).count() as u32;
    choose_min_match_len(num_used_literals, max_search_depth)
}

/// `recalculate_min_match_len`: refine from the block's actual literal usage.
fn recalculate_min_match_len(litlen_freqs: &[u32], max_search_depth: u32) -> u32 {
    let literal_freq: u32 = litlen_freqs[..NUM_LITERALS].iter().sum();
    let cutoff = literal_freq >> 10;
    let num_used_literals = litlen_freqs[..NUM_LITERALS]
        .iter()
        .filter(|&&f| f > cutoff)
        .count() as u32;
    choose_min_match_len(num_used_literals, max_search_depth)
}

/// `bsr32`: index of the most-significant set bit (`x` must be nonzero).
#[inline]
fn bsr32(x: u32) -> u32 {
    debug_assert!(x != 0);
    31 - x.leading_zeros()
}

// ---- block emission ----

/// Emit the accumulated block, choosing the cheapest of stored / static-Huffman
/// / dynamic-Huffman. `block_start` is the absolute offset of the block's first
/// byte in `buf`.
fn emit_block(
    bw: &mut BitWriter,
    buf: &[u8],
    block_start: usize,
    sink: &Sink,
    statics: &StaticCodes,
    is_final: bool,
) {
    // Add the end-of-block symbol to the litlen frequencies (as the vendor does
    // in deflate_flush_block).
    let mut litlen_freqs = sink.litlen_freqs;
    litlen_freqs[DEFLATE_END_OF_BLOCK] += 1;

    let litcode = make_huffman_code(
        DEFLATE_NUM_LITLEN_SYMS,
        MAX_LITLEN_CODEWORD_LEN,
        &litlen_freqs,
    );
    let offcode = make_huffman_code(
        DEFLATE_NUM_OFFSET_SYMS,
        MAX_OFFSET_CODEWORD_LEN,
        &sink.offset_freqs,
    );
    let header = build_dynamic_header(&litcode.lens, &offcode.lens);

    let dynamic_bits = 3 + header.header_bits() + data_bits(&sink.tokens, &litcode, &offcode);
    let static_bits = 3 + data_bits(&sink.tokens, &statics.litcode, &statics.offcode);
    let stored_bits = stored_block_bits(sink.block_length);

    if stored_bits <= dynamic_bits && stored_bits <= static_bits {
        super::emit_stored_block(
            bw,
            &buf[block_start..block_start + sink.block_length],
            is_final,
        );
    } else if static_bits <= dynamic_bits {
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_STATIC_HUFFMAN as u64, 2);
        emit_tokens(bw, &sink.tokens, &statics.litcode, &statics.offcode);
    } else {
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN as u64, 2);
        header.emit(bw);
        emit_tokens(bw, &sink.tokens, &litcode, &offcode);
    }
}

/// Exact coded-data bit cost of the token stream (including the EOB symbol) for
/// a given litlen/offset code.
fn data_bits(tokens: &[Token], litcode: &HuffmanCode, offcode: &HuffmanCode) -> u64 {
    let mut bits = 0u64;
    for &t in tokens {
        match t {
            Token::Literal(b) => bits += litcode.lens[b as usize] as u64,
            Token::Match { length, offset } => {
                let ls = length_slot(length as u32) as usize;
                bits +=
                    litcode.lens[DEFLATE_FIRST_LEN_SYM + ls] as u64 + LENGTH_EXTRA_BITS[ls] as u64;
                let os = offset_slot(offset as u32) as usize;
                bits += offcode.lens[os] as u64 + OFFSET_EXTRA_BITS[os] as u64;
            }
        }
    }
    bits += litcode.lens[DEFLATE_END_OF_BLOCK] as u64;
    bits
}

/// Emit the token stream (and the trailing EOB codeword) with the given codes.
fn emit_tokens(bw: &mut BitWriter, tokens: &[Token], litcode: &HuffmanCode, offcode: &HuffmanCode) {
    for &t in tokens {
        match t {
            Token::Literal(b) => {
                bw.add_bits(
                    litcode.codewords[b as usize] as u64,
                    litcode.lens[b as usize] as u32,
                );
            }
            Token::Match { length, offset } => {
                let length = length as u32;
                let offset = offset as u32;
                let ls = length_slot(length) as usize;
                bw.add_bits(
                    litcode.codewords[DEFLATE_FIRST_LEN_SYM + ls] as u64,
                    litcode.lens[DEFLATE_FIRST_LEN_SYM + ls] as u32,
                );
                bw.add_bits(
                    (length - LENGTH_SLOT_BASE[ls]) as u64,
                    LENGTH_EXTRA_BITS[ls] as u32,
                );
                let os = offset_slot(offset) as usize;
                bw.add_bits(offcode.codewords[os] as u64, offcode.lens[os] as u32);
                bw.add_bits(
                    (offset - OFFSET_SLOT_BASE[os]) as u64,
                    OFFSET_EXTRA_BITS[os] as u32,
                );
            }
        }
    }
    bw.add_bits(
        litcode.codewords[DEFLATE_END_OF_BLOCK] as u64,
        litcode.lens[DEFLATE_END_OF_BLOCK] as u32,
    );
}

/// Approximate bit cost of storing `len` bytes as stored (BTYPE=00) sub-blocks.
/// Mirrors the estimate in [`super`], used only for the block-type decision.
fn stored_block_bits(len: usize) -> u64 {
    let subblocks = (len / 65535) + 1;
    (8 * (len + 5 * subblocks)) as u64
}
