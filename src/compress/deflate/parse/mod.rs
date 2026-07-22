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
/// The crown engine (zopfli port + LzFind/squeeze/recursive-splitter Pareto
/// tier). Reached only via `-F`/`-I`/`-J`. See
/// `docs/compressor-architecture.md` for the full module map.
pub mod ultra;

/// Number of trailing pad bytes appended to the matchfinder's working buffer so
/// its speculative 4-byte / 8-byte loads never read out of bounds.
pub(super) const BUF_PAD: usize = 16;

/// `SOFT_MAX_BLOCK_LENGTH` — soft cap on the bytes covered by one block.
const SOFT_MAX_BLOCK_LENGTH: usize = 300_000;
/// `SEQ_STORE_LENGTH` — cap on the number of match "sequences" per block.
const SEQ_STORE_LENGTH: usize = 50_000;

/// Number of DEFLATE literal symbols (0..=255).
const NUM_LITERALS: usize = 256;

/// One literal-run + match "sequence" — the port of libdeflate's
/// `struct deflate_sequence` (`deflate_compress.c:242-262`). A block's body is
/// `seqs[0] .. seqs[n-1]` followed by `trailing_lits` literals: each `Seq` says
/// "emit `litrunlen` literals (read straight from the input buffer), then this
/// match". Storing runs instead of per-token records means the parse does NOT
/// push anything per literal, and the emit does NOT scan a token stream to
/// re-discover runs — the two costs cachegrind named as the emit-path excess vs
/// igzip (the run-scan line alone was ~25.7M Ir on `bin` L1). The offset slot is
/// precomputed at push time (the parser already computes it for the frequency
/// bump), eliminating the emit-side `offset_slot()` recompute (~8.9M Ir on
/// `text` L1).
#[derive(Clone, Copy)]
struct Seq {
    /// Number of literals preceding this match.
    litrunlen: u32,
    /// Match offset (1..=32768).
    offset: u16,
    /// Match length (3..=258) in bits 0..9 | offset slot (0..=29) << 9.
    length_and_slot: u16,
}

/// Bits 0..9 of [`Seq::length_and_slot`] hold the match length.
const SEQ_LEN_MASK: u16 = 0x1FF;
/// The offset slot starts at bit 9 of [`Seq::length_and_slot`].
const SEQ_SLOT_SHIFT: u16 = 9;

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

/// Per-block accumulator: sequences, symbol frequencies, split stats.
///
/// The literal bytes themselves are NOT stored — the emit reads them from the
/// input buffer (as libdeflate's `deflate_flush_block` does), so a literal push
/// is just a frequency bump + run-counter increment.
struct Sink {
    seqs: Vec<Seq>,
    /// Literals accumulated since the last match (the pending litrun; becomes
    /// the next `Seq::litrunlen`, or the block's trailing literals at flush).
    litrun: u32,
    litlen_freqs: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: [u32; DEFLATE_NUM_OFFSET_SYMS],
    /// Input bytes covered by the current block so far.
    block_length: usize,
    stats: BlockSplitStats,
}

impl Sink {
    fn new() -> Self {
        Sink {
            seqs: Vec::new(),
            litrun: 0,
            litlen_freqs: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset_freqs: [0; DEFLATE_NUM_OFFSET_SYMS],
            block_length: 0,
            stats: BlockSplitStats::new(),
        }
    }

    /// `deflate_begin_sequences` + `init_block_split_stats`.
    fn begin(&mut self) {
        self.seqs.clear();
        self.litrun = 0;
        self.litlen_freqs = [0; DEFLATE_NUM_LITLEN_SYMS];
        self.offset_freqs = [0; DEFLATE_NUM_OFFSET_SYMS];
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
        self.litrun += 1;
        self.block_length += 1;
    }

    /// Fast-path literal push: frequency bump + run counter only.
    ///
    /// The fast (L1) parser never calls `should_end_block`, so the block-split
    /// stats `push_literal` gathers are DEAD there (cachegrind: ~34M Ir/6MiB of
    /// `block_split.rs` attributed to the L1 bin run, none of it consulted), and
    /// the fast parser derives `block_length` once at flush (`pos - block_begin`)
    /// instead of a per-push `+= 1`. Emitted bytes are identical: `emit_block`
    /// consumes only the freqs/seqs/litrun/`block_length`.
    #[inline]
    fn push_literal_fast(&mut self, lit: u8) {
        // SAFETY: `lit` is a u8 (0..=255) and `litlen_freqs` has
        // DEFLATE_NUM_LITLEN_SYMS (288) entries, so `lit as usize` is in bounds.
        unsafe {
            *self.litlen_freqs.get_unchecked_mut(lit as usize) += 1;
        }
        self.litrun += 1;
    }

    /// Push the pending literal run + this match as one [`Seq`].
    #[inline]
    fn push_seq(&mut self, length: u32, offset: u32, os: usize) {
        self.seqs.push(Seq {
            litrunlen: self.litrun,
            offset: offset as u16,
            length_and_slot: (length as u16) | ((os as u16) << SEQ_SLOT_SHIFT),
        });
        self.litrun = 0;
    }

    /// Fast-path match push: frequencies + sequence only (see [`Self::push_literal_fast`]).
    #[inline]
    fn push_match_fast(&mut self, length: u32, offset: u32) {
        debug_assert!((DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN).contains(&length));
        debug_assert!((1..=32768).contains(&offset));
        let ls = length_slot(length) as usize;
        let os = offset_slot(offset) as usize;
        // SAFETY: as in `push_match` — `length_slot` returns 0..=28 and
        // `offset_slot` returns 0..=29, so both indices are in bounds.
        unsafe {
            *self
                .litlen_freqs
                .get_unchecked_mut(DEFLATE_FIRST_LEN_SYM + ls) += 1;
            *self.offset_freqs.get_unchecked_mut(os) += 1;
        }
        self.push_seq(length, offset, os);
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
        self.push_seq(length, offset, os);
        self.block_length += length as usize;
    }
}

/// Compress `buf[data_start..in_end]` into DEFLATE blocks appended to `bw`.
///
/// `buf` MUST have at least [`BUF_PAD`] trailing bytes beyond `in_end`. Bytes in
/// `buf[..data_start]` (a preset dictionary) are seeded into the matchfinder but
/// not coded; matches may reference them. Dispatches on `params.strategy` to
/// the matching parser (all levels 0-12 route through here now).
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
        // ACCEL is a const generic (see fast::run's doc comment): `::<true>`
        // (L0) monomorphizes with the scan-step ramp; `::<false>` (L1)
        // monomorphizes with that code compiled away entirely, not merely
        // runtime-disabled.
        Strategy::Fast0 => fast::run::<true>(
            buf,
            data_start,
            in_end,
            &statics,
            bw,
            is_last,
            fast::FAST0_BLOCK_LENGTH,
            true,
            fast::LIMIT_HASH_UPDATE_INSERTS_L0,
        ),
        Strategy::Fast => fast::run::<false>(
            buf,
            data_start,
            in_end,
            &statics,
            bw,
            is_last,
            fast::FAST_BLOCK_LENGTH,
            true,
            fast::LIMIT_HASH_UPDATE_INSERTS_L1,
        ),
        Strategy::Greedy => greedy::run(buf, data_start, in_end, params, &statics, bw, is_last),
        Strategy::Lazy => lazy::run(
            buf, data_start, in_end, params, &statics, bw, false, is_last,
        ),
        Strategy::Lazy2 => lazy::run(buf, data_start, in_end, params, &statics, bw, true, is_last),
        Strategy::NearOptimal => {
            near_optimal::run(buf, data_start, in_end, params, &statics, bw, is_last)
        }
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
        && sink.seqs.len() < SEQ_STORE_LENGTH
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
        emit_sequences(
            bw,
            buf,
            block_start,
            sink,
            &statics.litcode,
            &statics.offcode,
        );
    } else {
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN as u64, 2);
        header.emit(bw);
        emit_sequences(bw, buf, block_start, sink, &litcode, &offcode);
    }
}

/// Emit the accumulated block, choosing the cheaper of stored / static-Huffman
/// ONLY — the dynamic-Huffman candidate is never built. This is the L0
/// ("`Strategy::Fast0`") block emitter: skipping `make_huffman_code` (a
/// length-limited canonical-code build, effectively a package-merge pass) and
/// `build_dynamic_header` for both the litlen and offset alphabets is the
/// per-block cost [`emit_block`] pays that this function does not, which is
/// what makes L0 cheaper than L1 while sharing the identical chainless
/// single-probe matchfinder (`fast::run`). Ratio is a bit worse than L1's
/// (no per-block adaptive code), which is an intentional L0/L1 trade — L0's
/// bar is beating igzip -0 (which sometimes EXPANDS incompressible input),
/// not matching L1.
fn emit_block_static_or_stored(
    bw: &mut BitWriter,
    buf: &[u8],
    block_start: usize,
    sink: &Sink,
    statics: &StaticCodes,
    is_final: bool,
) {
    let mut litlen_freqs = sink.litlen_freqs;
    litlen_freqs[DEFLATE_END_OF_BLOCK] += 1;

    let static_bits = 3 + cost_from_freqs(
        &litlen_freqs,
        &sink.offset_freqs,
        &statics.litcode,
        &statics.offcode,
    );
    let stored_bits = stored_block_bits(sink.block_length);

    if stored_bits <= static_bits {
        super::emit_stored_block(
            bw,
            &buf[block_start..block_start + sink.block_length],
            is_final,
        );
    } else {
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_STATIC_HUFFMAN as u64, 2);
        emit_sequences(
            bw,
            buf,
            block_start,
            sink,
            &statics.litcode,
            &statics.offcode,
        );
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

/// Per-block emit tables with the codeword and its bit count MERGED into one
/// `u32` entry (`codeword | nbits << 24`), so each symbol costs ONE table load
/// in the hot loop instead of a codewords[] + lens[] pair (the igzip layout;
/// libdeflate keeps separate arrays and pays two loads).
///
/// `full_len` is the port of `deflate_compute_full_len_codewords` (C:1638-1658):
/// the litlen codeword concatenated with the extra-length bits, so a match's
/// length field emits with ONE `add_bits`. Entry format:
/// `(litlen_cw | extra_bits << litlen_len) | total_nbits << 24` — the packed
/// value uses at most 14 + 5 = 19 bits, comfortably below bit 24.
struct EmitTables {
    /// `lit[b] = codeword | nbits << 24` (codeword <= 14 bits).
    lit: [u32; NUM_LITERALS],
    /// `full_len[len]`, len 3..=258 (see above).
    full_len: [u32; DEFLATE_MAX_MATCH_LEN as usize + 1],
    /// `off[slot] = codeword | cwlen << 16 | (cwlen + extra_offset_bits) << 24`
    /// (codeword <= 15 bits fits in the low 16). The emit concatenates the
    /// offset's extra bits above the codeword with one shift, so a match's
    /// offset field is also ONE `add_bits`.
    off: [u32; DEFLATE_NUM_OFFSET_SYMS],
    /// End-of-block symbol: `codeword | nbits << 24`.
    eob: u32,
}

impl EmitTables {
    fn build(litcode: &HuffmanCode, offcode: &HuffmanCode) -> Self {
        let mut lit = [0u32; NUM_LITERALS];
        for (b, e) in lit.iter_mut().enumerate() {
            *e = litcode.codewords[b] | ((litcode.lens[b] as u32) << 24);
        }
        // MAX_LITLEN_CODEWORD_LEN (14) + max extra length bits (5) <= 24, so the
        // concatenation stays below the nbits byte (C's STATIC_ASSERT at :1642).
        let mut full_len = [0u32; DEFLATE_MAX_MATCH_LEN as usize + 1];
        for len in DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN {
            let slot = length_slot(len) as usize;
            let sym = DEFLATE_FIRST_LEN_SYM + slot;
            let extra_bits = len - LENGTH_SLOT_BASE[slot];
            let litlen_len = litcode.lens[sym] as u32;
            full_len[len as usize] = (litcode.codewords[sym] | (extra_bits << litlen_len))
                | (((litcode.lens[sym] + LENGTH_EXTRA_BITS[slot]) as u32) << 24);
        }
        let mut off = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        for (slot, e) in off.iter_mut().enumerate().take(OFFSET_EXTRA_BITS.len()) {
            let cwlen = offcode.lens[slot] as u32;
            *e = offcode.codewords[slot]
                | (cwlen << 16)
                | ((cwlen + OFFSET_EXTRA_BITS[slot] as u32) << 24);
        }
        let eob = litcode.codewords[DEFLATE_END_OF_BLOCK]
            | ((litcode.lens[DEFLATE_END_OF_BLOCK] as u32) << 24);
        EmitTables {
            lit,
            full_len,
            off,
            eob,
        }
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

/// Emit one merged-table entry (`codeword | nbits << 24`).
///
/// # Safety
/// Caller upholds `add_bits_raw`'s contract (accumulator has room).
#[inline(always)]
unsafe fn add_entry(bw: &mut BitWriter, e: u32) {
    bw.add_bits_raw((e & 0x00FF_FFFF) as u64, e >> 24);
}

/// Emit a run of `litrunlen` literals starting at `buf[p]` in groups of 4 (one
/// whole-word flush per group), then the 1-3-literal tail. Returns the position
/// one past the run.
///
/// # Safety
/// `p + litrunlen <= buf.len()`, and the output buffer must have been
/// `reserve`d so every `flush_word_unchecked` has 8 spare bytes.
#[inline(always)]
unsafe fn emit_literal_run(
    bw: &mut BitWriter,
    buf: &[u8],
    mut p: usize,
    mut litrunlen: usize,
    lit: &[u32; NUM_LITERALS],
) -> usize {
    while litrunlen >= 4 {
        // SAFETY: the run [p, p+litrunlen) is in bounds per the contract; a u8
        // always indexes the 256-entry `lit` table in bounds.
        for k in 0..4 {
            let b = *buf.get_unchecked(p + k);
            add_entry(bw, *lit.get_unchecked(b as usize));
        }
        // SAFETY: reserve() guarantees 8 spare bytes for every flush.
        bw.flush_word_unchecked();
        p += 4;
        litrunlen -= 4;
    }
    if litrunlen != 0 {
        // SAFETY: as above.
        for k in 0..litrunlen {
            let b = *buf.get_unchecked(p + k);
            add_entry(bw, *lit.get_unchecked(b as usize));
        }
        // SAFETY: see above.
        bw.flush_word_unchecked();
        p += litrunlen;
    }
    p
}

/// Emit the block body (literal runs + matches + trailing EOB codeword) with the
/// given codes, reading literal bytes straight from `buf`.
///
/// Port of the sequences output loop in `deflate_flush_block` (C:1938-2024):
/// literals come from the input via the per-match litrunlen (no token stream to
/// scan), a precomputed full-length-codeword LUT (mechanism 1), a 4-literals-
/// per-flush packed run (mechanism 2), whole-word branchless flushes (mechanism
/// 3), `CAN_BUFFER`-elided match flushes (mechanism 4), pure accumulate
/// `add_bits_raw` (mechanism 5), plus merged codeword|nbits entries and a
/// stored offset slot (one load per symbol, no emit-side `offset_slot()`).
///
/// FALSIFIED (2026-07-22): replacing this loop's four sequential
/// `add_bits_raw` calls (per literal-run group-of-4) and the match's two
/// sequential `add_bits_raw` calls with a scalar parallel-prefix-sum +
/// variable-shift + OR-reduce merge (igzip `encode_df_0{4,6}.asm`'s
/// vpgatherdd/vpsllvq technique, done as plain independent-operand scalar
/// ops instead of AVX2 intrinsics, on the theory that it collapses N
/// dependent read-modify-writes of `bw`'s accumulator into one) was byte-
/// identical (L0-12 x p1/p4 x {dd79_text6,dd79_bin6,dickens,data.parquet},
/// both roundtrip- and sha-verified) but was a measured NET REGRESSION, not
/// a win: `perf stat -r 15` on solvency (AMD Zen2, `-C target-cpu=native`,
/// /root/gzippy-locate5) showed whole-program instructions UP 0.1-2.0% (the
/// merge math costs more total instructions than it saves) and wall UP in 3
/// of 4 {corpus x level} cells (up to +3.5%, exceeding the <1% run-to-run
/// spread) with the 4th cell an unreplicated tie; M1/aarch64 (same scalar
/// code, no cfg split) showed a flat-to-slightly-worse wall and instructions
/// up 0.1-0.55% too. Root cause (HYPOTHESIS, unvalidated): the OOO schedulers
/// on both arches already hide the original chain's latency across loop
/// iterations, so this trades real extra instructions for latency headroom
/// that was never the bottleneck. Reopen trigger: an explicit-width SIMD
/// version (real `vpgatherdd`/`vpsllvq`, one instruction per 4-8 lanes
/// instead of one scalar instruction per lane) could still pay where the
/// scalar analog didn't, since it changes the instruction-COUNT term the
/// scalar attempt made worse, not just the dependency-depth term — untested,
/// not this session's finding. The exact falsified diff (both increments,
/// independently splittable) is reproduced in full in the commit message of
/// the commit that introduced this note (`git log --grep=FALSIFIED -p` on
/// this file) so a future session can rebuild and re-measure it without
/// re-deriving the technique from scratch.
fn emit_sequences(
    bw: &mut BitWriter,
    buf: &[u8],
    block_start: usize,
    sink: &Sink,
    litcode: &HuffmanCode,
    offcode: &HuffmanCode,
) {
    let tabs = EmitTables::build(litcode, offcode);

    // Normalize the accumulator to <= 7 buffered bits before the raw
    // `add_bits_raw`/`flush_word_unchecked` batch. libdeflate reaches this loop
    // with `bitcount <= 7` because its header emission ends in FLUSH_BITS
    // (C:2021 asserts `bitcount <= 7`); gzippy's block-type prefix + dynamic
    // header use the auto-flushing `add_bits`, which can leave up to 63 bits
    // buffered. Draining full bytes here (NOT byte-aligning — that would inject
    // zero pad bits) restores the invariant the raw path relies on, without
    // changing the emitted bit sequence.
    bw.flush_bits();

    // Ensure every whole-word flush in this batch has 8 spare bytes: a literal
    // codes to <= 14 bits (< 2 bytes) and a match to <= 47 bits (< 6 bytes)
    // while covering >= 3 input bytes, so 2 output bytes per covered input byte
    // bounds both; + slack for the EOB and the flushes' 8-byte headroom.
    bw.reserve(sink.block_length * 2 + 16);

    // `p` walks the input: each Seq's literals are exactly the input bytes
    // between the previous match's end and this match's start.
    let mut p = block_start;
    for seq in &sink.seqs {
        // SAFETY: every Seq was pushed with its literals + match inside the
        // block, so [p, p + litrunlen + length) stays within
        // `block_start + sink.block_length <= buf.len()`; reserve() above
        // covers every flush in the run and the match flush below.
        unsafe {
            p = emit_literal_run(bw, buf, p, seq.litrunlen as usize, &tabs.lit);

            let length = (seq.length_and_slot & SEQ_LEN_MASK) as usize;
            let os = (seq.length_and_slot >> SEQ_SLOT_SHIFT) as usize;
            // Litlen symbol + extra length bits as ONE add (mechanism 1).
            // SAFETY: `length` is 3..=258, indexing the 259-entry table.
            add_entry(bw, *tabs.full_len.get_unchecked(length));
            // Offset codeword + extra offset bits, concatenated into ONE add:
            // the stored slot makes cwlen/base/extra table lookups, and the
            // intermediate `CAN_BUFFER` flushes are elided (const assertion
            // above), so the whole match costs one flush.
            // SAFETY: `os` is 0..=29 — in bounds for the 32-entry `off` table
            // and the 30-entry OFFSET_* tables.
            let e = *tabs.off.get_unchecked(os);
            let cwlen = (e >> 16) & 0xFF;
            let extra = (seq.offset as u32 - *OFFSET_SLOT_BASE.get_unchecked(os)) as u64;
            bw.add_bits_raw(((e & 0xFFFF) as u64) | (extra << cwlen), e >> 24);
            // SAFETY: reserve() guarantees 8 spare bytes for every flush.
            bw.flush_word_unchecked();

            p += length;
        }
    }

    // Trailing literals after the last match, then the end-of-block symbol.
    // SAFETY: the trailing run is the block's final `litrun` input bytes, ending
    // exactly at `block_start + sink.block_length <= buf.len()`; reserve()'s 16
    // slack bytes cover the EOB flush.
    unsafe {
        emit_literal_run(bw, buf, p, sink.litrun as usize, &tabs.lit);
        add_entry(bw, tabs.eob);
        bw.flush_word_unchecked();
    }
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

    /// Test-side token: the logical (pre-Seq) representation of a block body.
    #[derive(Clone, Copy)]
    enum Tok {
        Lit(u8),
        Match { length: u32, offset: u32 },
    }

    /// The PRE-lever emit: symbol-then-extra, one `add_bits` per field, per-call
    /// auto-flush, walking a per-token stream. Kept verbatim as the byte-for-byte
    /// reference the sequence-based fast path must match.
    fn emit_tokens_reference(
        bw: &mut BitWriter,
        tokens: &[Tok],
        litcode: &HuffmanCode,
        offcode: &HuffmanCode,
    ) {
        for &t in tokens {
            match t {
                Tok::Lit(b) => {
                    bw.add_bits(
                        litcode.codewords[b as usize] as u64,
                        litcode.lens[b as usize] as u32,
                    );
                }
                Tok::Match { length, offset } => {
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

    /// The PRE-lever cost model: an exact per-token walk of the coded-data bit
    /// cost (including the EOB codeword). Kept verbatim as the byte-for-byte
    /// reference `cost_from_freqs` (the frequency-array sum) must equal.
    fn data_bits(tokens: &[Tok], litcode: &HuffmanCode, offcode: &HuffmanCode) -> u64 {
        let mut bits = 0u64;
        for &t in tokens {
            match t {
                Tok::Lit(b) => bits += litcode.lens[b as usize] as u64,
                Tok::Match { length, offset } => {
                    let ls = length_slot(length) as usize;
                    bits += litcode.lens[DEFLATE_FIRST_LEN_SYM + ls] as u64
                        + LENGTH_EXTRA_BITS[ls] as u64;
                    let os = offset_slot(offset) as usize;
                    bits += offcode.lens[os] as u64 + OFFSET_EXTRA_BITS[os] as u64;
                }
            }
        }
        bits += litcode.lens[DEFLATE_END_OF_BLOCK] as u64;
        bits
    }

    /// Build the litlen (with EOB) + offset histograms for a token stream, the
    /// way [`Sink`] does inline, so the two cost models can be compared.
    fn histograms_with_eob(
        tokens: &[Tok],
    ) -> (
        [u32; DEFLATE_NUM_LITLEN_SYMS],
        [u32; DEFLATE_NUM_OFFSET_SYMS],
    ) {
        let mut litlen = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        let mut offset = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        for &t in tokens {
            match t {
                Tok::Lit(b) => litlen[b as usize] += 1,
                Tok::Match { length, offset: o } => {
                    litlen[DEFLATE_FIRST_LEN_SYM + length_slot(length) as usize] += 1;
                    offset[offset_slot(o) as usize] += 1;
                }
            }
        }
        litlen[DEFLATE_END_OF_BLOCK] += 1;
        (litlen, offset)
    }

    /// Drive the PRODUCTION push path: feed the token stream through
    /// [`Sink::push_literal`]/[`Sink::push_match`] and synthesize the input
    /// buffer whose literal positions hold the literal bytes (match spans get
    /// filler — the emit never reads them, only their length).
    fn sink_and_buf(tokens: &[Tok]) -> (Sink, Vec<u8>) {
        let mut sink = Sink::new();
        sink.begin();
        let mut buf = Vec::new();
        for &t in tokens {
            match t {
                Tok::Lit(b) => {
                    sink.push_literal(b);
                    buf.push(b);
                }
                Tok::Match { length, offset } => {
                    sink.push_match(length, offset);
                    buf.resize(buf.len() + length as usize, 0xEE);
                }
            }
        }
        (sink, buf)
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

    fn lit_run(v: &mut Vec<Tok>, bytes: &[u8]) {
        for &b in bytes {
            v.push(Tok::Lit(b));
        }
    }

    fn mat(length: u32, offset: u32) -> Tok {
        Tok::Match { length, offset }
    }

    /// A token list exercising: literal runs of every remainder class (0/1/2/3
    /// past a multiple of 4), matches spanning the extreme and interior
    /// length/offset slots, and back-to-back matches with no literals between.
    fn fixture_tokens() -> Vec<Tok> {
        let mut t = Vec::new();
        // 10 literals => two groups of 4 + a 2-literal tail.
        lit_run(&mut t, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // Min-length / min-offset match.
        t.push(mat(3, 1));
        // Exactly 4 literals => one group, empty tail.
        lit_run(&mut t, &[10, 20, 30, 40]);
        // Max-length / max-offset match (largest length + offset slots).
        t.push(mat(258, 32768));
        // Back-to-back matches, no literals between, interior slots.
        t.push(mat(24, 100));
        t.push(mat(130, 5000));
        // 1-literal tail.
        lit_run(&mut t, &[200]);
        t.push(mat(11, 300));
        // 3-literal tail.
        lit_run(&mut t, &[201, 202, 203]);
        t.push(mat(66, 12000));
        // A long literal run to exercise many packed groups + a 2-tail
        // (trailing literals with no following match — the `sink.litrun` path).
        let long: Vec<u8> = (0..50).map(|i| (i * 5) as u8).collect();
        lit_run(&mut t, &long);
        t
    }

    #[test]
    fn fast_emit_is_byte_identical_to_reference() {
        let statics = StaticCodes::build();
        let tokens = fixture_tokens();
        let (sink, buf) = sink_and_buf(&tokens);

        // Try several starting bitcounts, including > 7, since real blocks reach
        // emit_sequences with up to 63 bits still buffered from the (auto-
        // flushing) header emission — the raw fast path must normalize that
        // first.
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
            emit_sequences(
                &mut fast,
                &buf,
                0,
                &sink,
                &statics.litcode,
                &statics.offcode,
            );
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
        let tokens: Vec<Tok> = (0..=255u16).map(|b| Tok::Lit(b as u8)).collect();
        let (sink, buf) = sink_and_buf(&tokens);

        let mut fast = BitWriter::new();
        emit_sequences(
            &mut fast,
            &buf,
            0,
            &sink,
            &statics.litcode,
            &statics.offcode,
        );
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
        let (sink, buf) = sink_and_buf(&tokens);

        let mut fast = BitWriter::new();
        emit_sequences(&mut fast, &buf, 0, &sink, &litcode, &offcode);
        let fast_bytes = fast.finish();

        let mut refr = BitWriter::new();
        emit_tokens_reference(&mut refr, &tokens, &litcode, &offcode);
        let ref_bytes = refr.finish();

        assert_eq!(fast_bytes, ref_bytes);
    }

    #[test]
    fn emit_reads_literals_from_arbitrary_block_start() {
        // The same logical stream emitted from a nonzero block_start (literals
        // prefixed by unrelated bytes) must produce identical output.
        let statics = StaticCodes::build();
        let tokens = fixture_tokens();
        let (sink, buf) = sink_and_buf(&tokens);

        let mut shifted = vec![0xAAu8; 37];
        shifted.extend_from_slice(&buf);

        let mut a = BitWriter::new();
        emit_sequences(&mut a, &buf, 0, &sink, &statics.litcode, &statics.offcode);
        let mut b = BitWriter::new();
        emit_sequences(
            &mut b,
            &shifted,
            37,
            &sink,
            &statics.litcode,
            &statics.offcode,
        );
        assert_eq!(a.finish(), b.finish());
    }
}
