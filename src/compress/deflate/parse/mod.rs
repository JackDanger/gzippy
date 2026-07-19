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

mod fast;
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

/// One parsed token, packed into a `u32` (was a 6-byte `enum { Literal(u8),
/// Match { length: u16, offset: u16 } }`). The tighter representation shrinks the
/// per-block token buffer by a third and makes the literal/match discriminant a
/// single-bit test in the emit hot loop.
///
/// Layout (bit 31 is the literal tag; a match's value never reaches bit 31
/// because `offset <= 32768` puts its highest set bit at 24):
///   * Literal `b`  →  `TOK_LITERAL_FLAG | b`            (b in bits 0..8)
///   * Match l/o    →  `l | (o << TOK_OFF_SHIFT)`         (l in bits 0..9,
///                                                          o in bits 9..25)
type Token = u32;

/// Bit 31: set iff the token is a literal.
const TOK_LITERAL_FLAG: u32 = 0x8000_0000;
/// Low 9 bits hold a match length (3..=258 fits in 9 bits).
const TOK_LEN_MASK: u32 = 0x1FF;
/// A match offset (1..=32768, 16 bits) starts at bit 9.
const TOK_OFF_SHIFT: u32 = 9;

#[inline]
fn pack_literal(b: u8) -> Token {
    TOK_LITERAL_FLAG | b as u32
}

#[inline]
fn pack_match(length: u32, offset: u32) -> Token {
    debug_assert!(length <= TOK_LEN_MASK);
    debug_assert!(offset <= 0xFFFF);
    length | (offset << TOK_OFF_SHIFT)
}

#[inline]
fn tok_is_literal(t: Token) -> bool {
    t & TOK_LITERAL_FLAG != 0
}

#[inline]
fn tok_literal_byte(t: Token) -> u8 {
    t as u8
}

#[inline]
fn tok_match_len(t: Token) -> u32 {
    t & TOK_LEN_MASK
}

#[inline]
fn tok_match_off(t: Token) -> u32 {
    (t >> TOK_OFF_SHIFT) & 0xFFFF
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
        // SAFETY: `lit` is a u8 (0..=255) and `litlen_freqs` has
        // DEFLATE_NUM_LITLEN_SYMS (288) entries, so `lit as usize` is in bounds.
        unsafe {
            *self.litlen_freqs.get_unchecked_mut(lit as usize) += 1;
        }
        self.stats.observe_literal(lit);
        self.tokens.push(pack_literal(lit));
        self.block_length += 1;
    }

    /// `deflate_choose_match`.
    #[inline]
    fn push_match(&mut self, length: u32, offset: u32) {
        debug_assert!((DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN).contains(&length));
        debug_assert!((1..=32768).contains(&offset));
        let ls = length_slot(length) as usize;
        let os = offset_slot(offset) as usize;
        // SAFETY: `length_slot` returns 0..=28 so `DEFLATE_FIRST_LEN_SYM + ls`
        // (257..=285) is < DEFLATE_NUM_LITLEN_SYMS (288); `offset_slot` returns
        // 0..=29 so `os` is < DEFLATE_NUM_OFFSET_SYMS (32). Both are in bounds.
        unsafe {
            *self
                .litlen_freqs
                .get_unchecked_mut(DEFLATE_FIRST_LEN_SYM + ls) += 1;
            *self.offset_freqs.get_unchecked_mut(os) += 1;
        }
        self.stats.observe_match(length);
        self.tokens.push(pack_match(length, offset));
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
    is_last: bool,
    bw: &mut BitWriter,
) {
    let statics = StaticCodes::build();
    match params.strategy {
        Strategy::Fast => fast::run(buf, data_start, in_end, &statics, bw, is_last),
        Strategy::Greedy => greedy::run(buf, data_start, in_end, params, &statics, bw, is_last),
        Strategy::Lazy => lazy::run(
            buf, data_start, in_end, params, &statics, bw, false, is_last,
        ),
        Strategy::Lazy2 => lazy::run(buf, data_start, in_end, params, &statics, bw, true, is_last),
        Strategy::NearOptimal => {
            near_optimal::run(buf, data_start, in_end, params, &statics, bw, is_last)
        }
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

    let dynamic_bits = 3
        + header.header_bits()
        + cost_from_freqs(&litlen_freqs, &sink.offset_freqs, &litcode, &offcode);
    let static_bits = 3 + cost_from_freqs(
        &litlen_freqs,
        &sink.offset_freqs,
        &statics.litcode,
        &statics.offcode,
    );
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

/// Exact coded-data bit cost (including the EOB symbol) of a token stream whose
/// per-symbol histogram is `litlen_freqs` / `offset_freqs`, coded with the given
/// litlen/offset code. `litlen_freqs[DEFLATE_END_OF_BLOCK]` must already include
/// the one EOB symbol.
///
/// Port of the cost half of `deflate_compute_true_cost`
/// (`deflate_compress.c:2889-2921`) — the frequency-array × code-length sum. This
/// replaces walking every token twice (once per candidate code) with two passes
/// over the fixed-size frequency arrays. Because the frequencies ARE the token
/// histogram (Sink bumps them inline as tokens are pushed), the sum is
/// bit-for-bit identical to the old per-token walk (`data_bits` in the tests),
/// so the dyn/static/stored decision — and thus the emitted bytes — is unchanged.
fn cost_from_freqs(
    litlen_freqs: &[u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: &[u32; DEFLATE_NUM_OFFSET_SYMS],
    litcode: &HuffmanCode,
    offcode: &HuffmanCode,
) -> u64 {
    // SAFETY (whole body): `litcode.lens` has DEFLATE_NUM_LITLEN_SYMS entries and
    // `offcode.lens` has DEFLATE_NUM_OFFSET_SYMS entries (make_huffman_code
    // asserts `freqs.len() == num_syms`). The litlen loops index 0..286 (< 288)
    // and the offset loop indexes 0..30 (< 32), all in bounds; the `litlen_freqs`
    // / `offset_freqs` array refs match those loop bounds by type.
    let mut bits = 0u64;
    // Literals 0..=255 and the EOB symbol (256) — plain codeword lengths.
    for sym in 0..DEFLATE_FIRST_LEN_SYM {
        bits += unsafe {
            *litlen_freqs.get_unchecked(sym) as u64 * *litcode.lens.get_unchecked(sym) as u64
        };
    }
    // Length symbols: codeword length + extra length bits for the slot.
    for slot in 0..LENGTH_EXTRA_BITS.len() {
        let sym = DEFLATE_FIRST_LEN_SYM + slot;
        bits += unsafe {
            *litlen_freqs.get_unchecked(sym) as u64
                * (*litcode.lens.get_unchecked(sym) as u64
                    + *LENGTH_EXTRA_BITS.get_unchecked(slot) as u64)
        };
    }
    // Offset symbols: codeword length + extra offset bits for the slot.
    for slot in 0..OFFSET_EXTRA_BITS.len() {
        bits += unsafe {
            *offset_freqs.get_unchecked(slot) as u64
                * (*offcode.lens.get_unchecked(slot) as u64
                    + *OFFSET_EXTRA_BITS.get_unchecked(slot) as u64)
        };
    }
    bits
}

/// Precomputed "full" match-length codewords: the litlen codeword concatenated
/// with the extra-length bits, one packed value + one length per match length.
///
/// Port of `deflate_compute_full_len_codewords` (C:1638-1658). Building this
/// once per block lets a match's length field emit with ONE `add_bits` instead
/// of a symbol-then-extra pair.
struct FullLenCodewords {
    /// `codewords[len] = litlen_cw | (extra_bits << litlen_len)`, len 3..=258.
    codewords: [u32; DEFLATE_MAX_MATCH_LEN as usize + 1],
    /// `lens[len] = litlen_len + extra_length_bits[slot]`.
    lens: [u8; DEFLATE_MAX_MATCH_LEN as usize + 1],
}

impl FullLenCodewords {
    #[inline]
    fn build(litcode: &HuffmanCode) -> Self {
        // MAX_LITLEN_CODEWORD_LEN (14) + max extra length bits (5) <= 32, so the
        // concatenation fits in a u32 (C's STATIC_ASSERT at :1642).
        let mut codewords = [0u32; DEFLATE_MAX_MATCH_LEN as usize + 1];
        let mut lens = [0u8; DEFLATE_MAX_MATCH_LEN as usize + 1];
        for len in DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN {
            let slot = length_slot(len) as usize;
            let sym = DEFLATE_FIRST_LEN_SYM + slot;
            let extra_bits = len - LENGTH_SLOT_BASE[slot];
            let litlen_len = litcode.lens[sym] as u32;
            codewords[len as usize] = litcode.codewords[sym] | (extra_bits << litlen_len);
            lens[len as usize] = litcode.lens[sym] + LENGTH_EXTRA_BITS[slot];
        }
        FullLenCodewords { codewords, lens }
    }
}

/// Compile-time proof that the 64-bit accumulator can buffer a full match
/// (length codeword + offset symbol + offset extra) after a single flush, and a
/// run of 4 literals, without any intermediate flush. When these hold, the
/// `CAN_BUFFER`-gated flushes inside `WRITE_MATCH` (C:1660-1694) are elided,
/// leaving exactly one flush per match / per 4 literals.
const _: () = {
    use super::bitstream::can_buffer;
    let match_bits = MAX_LITLEN_CODEWORD_LEN + 5 /* DEFLATE_MAX_EXTRA_LENGTH_BITS */
        + MAX_OFFSET_CODEWORD_LEN + 13 /* DEFLATE_MAX_EXTRA_OFFSET_BITS */;
    assert!(can_buffer(match_bits), "match cannot buffer in one word");
    assert!(
        can_buffer(4 * MAX_LITLEN_CODEWORD_LEN),
        "4-literal run cannot buffer"
    );
};

/// Emit the token stream (and the trailing EOB codeword) with the given codes.
///
/// Port of the literals/matches output loop in `deflate_flush_block`
/// (C:1938-2024): a precomputed full-length-codeword LUT (mechanism 1), a
/// 4-literals-per-flush packed run (mechanism 2), whole-word branchless flushes
/// (mechanism 3), `CAN_BUFFER`-elided match flushes (mechanism 4), and pure
/// accumulate `add_bits_raw` (mechanism 5).
fn emit_tokens(bw: &mut BitWriter, tokens: &[Token], litcode: &HuffmanCode, offcode: &HuffmanCode) {
    let full_len = FullLenCodewords::build(litcode);

    // Normalize the accumulator to <= 7 buffered bits before the raw
    // `add_bits_raw`/`flush_word_unchecked` batch. libdeflate reaches this loop
    // with `bitcount <= 7` because its header emission ends in FLUSH_BITS
    // (C:2021 asserts `bitcount <= 7`); gzippy's block-type prefix + dynamic
    // header use the auto-flushing `add_bits`, which can leave up to 63 bits
    // buffered. Draining full bytes here (NOT byte-aligning — that would inject
    // zero pad bits) restores the invariant the raw path relies on, without
    // changing the emitted bit sequence.
    bw.flush_bits();

    // Ensure every whole-word flush in this batch has 8 spare bytes: a token
    // codes to at most 47 bits (< 6 bytes) and the EOB to <= 2 bytes.
    bw.reserve(tokens.len() * 6 + 16);

    let n = tokens.len();
    let mut i = 0usize;
    while i < n {
        // Gather the run of consecutive literals starting at `i`, then emit it
        // in groups of 4 (one flush per group), then the 1-3-literal tail.
        let run_start = i;
        // SAFETY: `i < n` is checked first, so the token read is in bounds.
        while i < n && tok_is_literal(unsafe { *tokens.get_unchecked(i) }) {
            i += 1;
        }
        let mut p = run_start;
        let mut litrunlen = i - run_start;
        while litrunlen >= 4 {
            for _ in 0..4 {
                // SAFETY: `p` walks `run_start..i`, positions the scan above
                // proved are all literals, so `p < n` (in-bounds token). `b` is a
                // u8, so `b as usize` is < 256 <= the litcode arrays' length
                // (DEFLATE_NUM_LITLEN_SYMS).
                let b = tok_literal_byte(unsafe { *tokens.get_unchecked(p) });
                unsafe {
                    bw.add_bits_raw(
                        *litcode.codewords.get_unchecked(b as usize) as u64,
                        *litcode.lens.get_unchecked(b as usize) as u32,
                    );
                }
                p += 1;
            }
            // SAFETY: reserve() above guarantees 8 spare bytes for every flush.
            unsafe { bw.flush_word_unchecked() };
            litrunlen -= 4;
        }
        if litrunlen != 0 {
            for _ in 0..litrunlen {
                // SAFETY: as above — `p < i <= n` and the token is a literal.
                let b = tok_literal_byte(unsafe { *tokens.get_unchecked(p) });
                unsafe {
                    bw.add_bits_raw(
                        *litcode.codewords.get_unchecked(b as usize) as u64,
                        *litcode.lens.get_unchecked(b as usize) as u32,
                    );
                }
                p += 1;
            }
            // SAFETY: see above.
            unsafe { bw.flush_word_unchecked() };
        }

        if i >= n {
            break;
        }

        // The run was terminated by a match (the end-of-stream case broke above).
        // SAFETY: the outer `while i < n` guard and the `i >= n` break above
        // guarantee `i < n`, so `tokens[i]` is in bounds.
        {
            let t = unsafe { *tokens.get_unchecked(i) };
            let length = tok_match_len(t);
            let offset = tok_match_off(t);
            let os = offset_slot(offset) as usize;
            // SAFETY: `length` is 3..=258 so it indexes the `full_len` arrays
            // (length DEFLATE_MAX_MATCH_LEN+1 = 259); `offset_slot` returns
            // 0..=29 so `os` < 32 = the offcode arrays' length and < 30 = the
            // OFFSET_* tables' length.
            unsafe {
                // Litlen symbol + extra length bits as ONE add (mechanism 1).
                bw.add_bits_raw(
                    *full_len.codewords.get_unchecked(length as usize) as u64,
                    *full_len.lens.get_unchecked(length as usize) as u32,
                );
                // Offset symbol, then extra offset bits. The intermediate
                // `CAN_BUFFER` flushes are elided (see the const assertion
                // above), so the whole match costs one flush.
                bw.add_bits_raw(
                    *offcode.codewords.get_unchecked(os) as u64,
                    *offcode.lens.get_unchecked(os) as u32,
                );
                bw.add_bits_raw(
                    (offset - *OFFSET_SLOT_BASE.get_unchecked(os)) as u64,
                    *OFFSET_EXTRA_BITS.get_unchecked(os) as u32,
                );
                // SAFETY: reserve() guarantees 8 spare bytes for every flush.
                bw.flush_word_unchecked();
            }
        }
        i += 1;
    }

    // End-of-block symbol.
    bw.add_bits_raw(
        litcode.codewords[DEFLATE_END_OF_BLOCK] as u64,
        litcode.lens[DEFLATE_END_OF_BLOCK] as u32,
    );
    // SAFETY: reserve() left 16 slack bytes beyond the token bytes for the EOB.
    unsafe { bw.flush_word_unchecked() };
}

/// Approximate bit cost of storing `len` bytes as stored (BTYPE=00) sub-blocks.
/// Mirrors the estimate in [`super`], used only for the block-type decision.
fn stored_block_bits(len: usize) -> u64 {
    let subblocks = (len / 65535) + 1;
    (8 * (len + 5 * subblocks)) as u64
}

#[cfg(test)]
mod emit_tests {
    use super::*;

    /// The PRE-lever emit: symbol-then-extra, one `add_bits` per field, per-call
    /// auto-flush. Kept verbatim as the byte-for-byte reference the fast path
    /// must match.
    fn emit_tokens_reference(
        bw: &mut BitWriter,
        tokens: &[Token],
        litcode: &HuffmanCode,
        offcode: &HuffmanCode,
    ) {
        for &t in tokens {
            if tok_is_literal(t) {
                let b = tok_literal_byte(t);
                bw.add_bits(
                    litcode.codewords[b as usize] as u64,
                    litcode.lens[b as usize] as u32,
                );
            } else {
                let length = tok_match_len(t);
                let offset = tok_match_off(t);
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
        bw.add_bits(
            litcode.codewords[DEFLATE_END_OF_BLOCK] as u64,
            litcode.lens[DEFLATE_END_OF_BLOCK] as u32,
        );
    }

    /// The PRE-lever cost model: an exact per-token walk of the coded-data bit
    /// cost (including the EOB codeword). Kept verbatim as the byte-for-byte
    /// reference `cost_from_freqs` (the frequency-array sum) must equal.
    fn data_bits(tokens: &[Token], litcode: &HuffmanCode, offcode: &HuffmanCode) -> u64 {
        let mut bits = 0u64;
        for &t in tokens {
            if tok_is_literal(t) {
                bits += litcode.lens[tok_literal_byte(t) as usize] as u64;
            } else {
                let ls = length_slot(tok_match_len(t)) as usize;
                bits +=
                    litcode.lens[DEFLATE_FIRST_LEN_SYM + ls] as u64 + LENGTH_EXTRA_BITS[ls] as u64;
                let os = offset_slot(tok_match_off(t)) as usize;
                bits += offcode.lens[os] as u64 + OFFSET_EXTRA_BITS[os] as u64;
            }
        }
        bits += litcode.lens[DEFLATE_END_OF_BLOCK] as u64;
        bits
    }

    /// Build the litlen (with EOB) + offset histograms for a token stream, the
    /// way [`Sink`] does inline, so the two cost models can be compared.
    fn histograms_with_eob(
        tokens: &[Token],
    ) -> (
        [u32; DEFLATE_NUM_LITLEN_SYMS],
        [u32; DEFLATE_NUM_OFFSET_SYMS],
    ) {
        let mut litlen = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        let mut offset = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        for &t in tokens {
            if tok_is_literal(t) {
                litlen[tok_literal_byte(t) as usize] += 1;
            } else {
                litlen[DEFLATE_FIRST_LEN_SYM + length_slot(tok_match_len(t)) as usize] += 1;
                offset[offset_slot(tok_match_off(t)) as usize] += 1;
            }
        }
        litlen[DEFLATE_END_OF_BLOCK] += 1;
        (litlen, offset)
    }

    #[test]
    fn cost_from_freqs_equals_token_walk() {
        // Static codes and a skewed dynamic code, over the mixed fixture.
        let statics = StaticCodes::build();
        let tokens = fixture_tokens();
        let (litlen_freqs, offset_freqs) = histograms_with_eob(&tokens);

        assert_eq!(
            cost_from_freqs(
                &litlen_freqs,
                &offset_freqs,
                &statics.litcode,
                &statics.offcode
            ),
            data_bits(&tokens, &statics.litcode, &statics.offcode),
            "cost_from_freqs != data_bits (static code)"
        );

        let litcode = make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &litlen_freqs,
        );
        let offcode = make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &offset_freqs,
        );
        assert_eq!(
            cost_from_freqs(&litlen_freqs, &offset_freqs, &litcode, &offcode),
            data_bits(&tokens, &litcode, &offcode),
            "cost_from_freqs != data_bits (dynamic code)"
        );
    }

    fn lit_run(v: &mut Vec<Token>, bytes: &[u8]) {
        for &b in bytes {
            v.push(pack_literal(b));
        }
    }

    /// A token list exercising: literal runs of every remainder class (0/1/2/3
    /// past a multiple of 4), matches spanning the extreme and interior
    /// length/offset slots, and back-to-back matches with no literals between.
    fn fixture_tokens() -> Vec<Token> {
        let mut t = Vec::new();
        // 10 literals => two groups of 4 + a 2-literal tail.
        lit_run(&mut t, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // Min-length / min-offset match.
        t.push(pack_match(3, 1));
        // Exactly 4 literals => one group, empty tail.
        lit_run(&mut t, &[10, 20, 30, 40]);
        // Max-length / max-offset match (largest length + offset slots).
        t.push(pack_match(258, 32768));
        // Back-to-back matches, no literals between, interior slots.
        t.push(pack_match(24, 100));
        t.push(pack_match(130, 5000));
        // 1-literal tail.
        lit_run(&mut t, &[200]);
        t.push(pack_match(11, 300));
        // 3-literal tail.
        lit_run(&mut t, &[201, 202, 203]);
        t.push(pack_match(66, 12000));
        // A long literal run to exercise many packed groups + a 2-tail.
        let long: Vec<u8> = (0..50).map(|i| (i * 5) as u8).collect();
        lit_run(&mut t, &long);
        t
    }

    #[test]
    fn fast_emit_is_byte_identical_to_reference() {
        let statics = StaticCodes::build();
        let tokens = fixture_tokens();

        // Try several starting bitcounts, including > 7, since real blocks reach
        // emit_tokens with up to 63 bits still buffered from the (auto-flushing)
        // header emission — the raw fast path must normalize that first.
        for seed_bits in [0u32, 3, 7, 8, 20, 40, 63] {
            let seed_val = if seed_bits == 0 {
                0
            } else {
                0x5A5A_5A5A_5A5A_5A5Au64 & ((1u64 << seed_bits) - 1)
            };

            // Fast (lever) path.
            let mut fast = BitWriter::new();
            if seed_bits != 0 {
                fast.add_bits(seed_val, seed_bits);
            }
            emit_tokens(&mut fast, &tokens, &statics.litcode, &statics.offcode);
            let fast_bytes = fast.finish();

            // Reference path with an identical seed.
            let mut refr = BitWriter::new();
            if seed_bits != 0 {
                refr.add_bits(seed_val, seed_bits);
            }
            emit_tokens_reference(&mut refr, &tokens, &statics.litcode, &statics.offcode);
            let ref_bytes = refr.finish();

            assert_eq!(
                fast_bytes, ref_bytes,
                "fast emit diverged from the reference emit (seed_bits={seed_bits})"
            );
        }
    }

    #[test]
    fn fast_emit_matches_reference_on_all_literals() {
        let statics = StaticCodes::build();
        let bytes: Vec<u8> = (0..=255u16).map(|b| b as u8).collect();
        let tokens: Vec<Token> = bytes.iter().map(|&b| pack_literal(b)).collect();

        let mut fast = BitWriter::new();
        emit_tokens(&mut fast, &tokens, &statics.litcode, &statics.offcode);
        let fast_bytes = fast.finish();

        let mut refr = BitWriter::new();
        emit_tokens_reference(&mut refr, &tokens, &statics.litcode, &statics.offcode);
        let ref_bytes = refr.finish();

        assert_eq!(fast_bytes, ref_bytes);
    }

    #[test]
    fn fast_emit_matches_reference_with_dynamic_codes() {
        // Build non-static codes from a skewed frequency distribution so the
        // codeword lengths differ from the fixed code, then check equality.
        let mut litfreqs = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        for (i, f) in litfreqs.iter_mut().enumerate() {
            *f = ((i * 7 + 1) % 13 + 1) as u32;
        }
        litfreqs[DEFLATE_END_OF_BLOCK] += 1;
        let mut offfreqs = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        for (i, f) in offfreqs.iter_mut().enumerate() {
            *f = ((i * 3 + 2) % 11 + 1) as u32;
        }
        let litcode =
            make_huffman_code(DEFLATE_NUM_LITLEN_SYMS, MAX_LITLEN_CODEWORD_LEN, &litfreqs);
        let offcode =
            make_huffman_code(DEFLATE_NUM_OFFSET_SYMS, MAX_OFFSET_CODEWORD_LEN, &offfreqs);
        let tokens = fixture_tokens();

        let mut fast = BitWriter::new();
        emit_tokens(&mut fast, &tokens, &litcode, &offcode);
        let fast_bytes = fast.finish();

        let mut refr = BitWriter::new();
        emit_tokens_reference(&mut refr, &tokens, &litcode, &offcode);
        let ref_bytes = refr.finish();

        assert_eq!(fast_bytes, ref_bytes);
    }
}
