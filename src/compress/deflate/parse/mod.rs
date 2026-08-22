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
use super::encode_types::{BlockRole, HeaderBudget, InputMode};
use super::huffman::{
    build_dynamic_header, make_huffman_code, make_huffman_code_exact_into, make_huffman_code_into,
    CodeScratch, DynamicHeader, HeaderScratch, HuffmanCode,
};
use super::level::{LevelParams, Strategy};
use super::matchfinder::hc::WINDOW_SIZE;
use super::tables::{
    length_slot, offset_slot, static_litlen_freqs, static_offset_freqs,
    DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, DEFLATE_BLOCKTYPE_STATIC_HUFFMAN, DEFLATE_END_OF_BLOCK,
    DEFLATE_FIRST_LEN_SYM, DEFLATE_MAX_MATCH_LEN, DEFLATE_MIN_MATCH_LEN, DEFLATE_NUM_LITLEN_SYMS,
    DEFLATE_NUM_OFFSET_SYMS, LENGTH_EXTRA_BITS, LENGTH_SLOT_BASE, MAX_LITLEN_CODEWORD_LEN,
    MAX_OFFSET_CODEWORD_LEN, OFFSET_EXTRA_BITS, OFFSET_SLOT_BASE,
};

mod fast;
// L1-band ratio-close-out config-space search (2026-07-22 campaign, `l1-tune`
// Cargo feature, OFF by default): re-export `fast::tune` publicly ONLY under
// the feature so `examples/l1_search.rs` (a separate binary crate depending
// on this lib through its public API) can call `tune::set`/`get` to sweep
// configs within one process. `fast` itself stays private in the default
// build — this re-export is the sole surface the search tool needs.
#[cfg(feature = "l1-tune")]
pub use fast::tune;
// DELETED 2026-07-27 by user order: `gated.rs`, the detector-gated lazy-L3 parser.
// A nine-line doc comment for it survived here until 2026-07-30, dangling on `mod
// greedy;` and still advertising `GZIPPY_L3TUNE_GATE_*` env vars plus "a `--tune`-style
// channel + `fulcrum l3search`" as "a real, named, un-taken next step". Both are
// forbidden: `CLAUDE.md` non-negotiable #3 bans env-var knobs and content detectors
// choosing a parser, and `fulcrum l1search` was deleted as constitutionally banned.
//
// Recorded because a retraction that does not reach every statement of the thing gets
// re-inherited by the next session that reads the file. A stale comment proposing
// forbidden work is not inert documentation; it is an instruction.
mod far_len3;
mod greedy;
use far_len3::FarLen3Gate;
/// Level-1 parser over the 2-way hash-table matchfinder — libdeflate's
/// `deflate_compress_fastest`. See its module doc for the vendor diff and the
/// REOPEN it rests on.
#[allow(dead_code)]
mod ht_fast;
mod lazy;
mod near_optimal;
/// Post-parse block splitter (zopfli `blocksplitter.c`). See its module doc.
pub(super) mod postsplit;
/// The crown engine (zopfli port + LzFind/squeeze/recursive-splitter Pareto
/// tier). Reached only via `-F`/`-I`/`-J`. See
/// `docs/compressor-architecture.md` for the full module map.
pub mod ultra;

/// Number of trailing pad bytes appended to the matchfinder's working buffer so
/// its speculative 4-byte / 8-byte loads never read out of bounds.
pub(super) const BUF_PAD: usize = 16;

/// `SOFT_MAX_BLOCK_LENGTH` — soft cap on the bytes covered by one block.
pub(crate) const SOFT_MAX_BLOCK_LENGTH: usize = 300_000;
/// `SEQ_STORE_LENGTH` — cap on the number of match "sequences" per block.
///
/// This is a POLICY cap, enforced by [`continue_block`], and only the greedy and
/// lazy parsers consult it. It is NOT the size of the backing store; see
/// [`SEQ_STORE_CAPACITY`].
const SEQ_STORE_LENGTH: usize = 50_000;

/// Allocated length of [`Sink::seqs`] — the worst-case number of sequences any
/// parser can put in one block.
const SEQ_STORE_CAPACITY: usize = {
    let widest_block = if fast::FAST0_BLOCK_LENGTH > SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH {
        fast::FAST0_BLOCK_LENGTH
    } else {
        SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH
    };
    (widest_block + DEFLATE_MAX_MATCH_LEN as usize) / DEFLATE_MIN_MATCH_LEN as usize + 1
};

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
#[derive(Clone, Copy, Default)]
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

/// The precomputed RFC 1951 fixed (static) Huffman codes.
///
/// PROCESS-WIDE, built exactly once — see [`StaticCodes::get`]. These are the FIXED
/// codes of RFC 1951 section 3.2.6, derived from two compile-time constant frequency
/// tables, so they are the same bytes on every call, forever.
struct StaticCodes {
    litcode: HuffmanCode,
    offcode: HuffmanCode,
}

static STATIC_CODES: std::sync::OnceLock<StaticCodes> = std::sync::OnceLock::new();

impl StaticCodes {
    /// The process-wide RFC 1951 fixed codes.
    ///
    /// STRUCTURAL. `build()` was called on EVERY parser entry — `parse::compress` and
    /// `parse::compress_resumable` — which is once per streaming pass at T=1 and ONCE PER
    /// CHUNK at T>1. It runs two full `make_huffman_code` passes (symbol sort, package-merge
    /// tree build, length-limiting, codeword generation) plus their allocations, to
    /// reproduce a table that RFC 1951 fixes as a CONSTANT. Same constant frequency tables
    /// in, same bytes out, every time.
    ///
    /// `OnceLock` rather than `thread_local!` (which is what `Sink` and `HcMatchfinder` use)
    /// because this value is immutable and shared: every parser takes `&StaticCodes` and none
    /// mutates it, so one process-wide copy is correct and costs one relaxed load per entry
    /// instead of a per-thread rebuild.
    fn get() -> &'static StaticCodes {
        STATIC_CODES.get_or_init(StaticCodes::build)
    }

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
pub(crate) struct Sink {
    /// Backing store for `nseqs` sequences. Capacity is fixed at
    /// `SEQ_STORE_LENGTH` and `len()` is deliberately left at 0 — elements are
    /// written through the spare capacity, so the buffer is ALLOCATED but never
    /// initialised. A `vec![Seq::default(); SEQ_STORE_LENGTH]` zeroes 400 KiB on
    /// every `Sink::new()`, which measurably cost the deep levels (L5/L9
    /// regressed up to 3.3% while L2/L4 improved) — that memory competes with
    /// the 32 KiB window and the hash tables for cache.
    seqs: Vec<Seq>,
    /// Literals accumulated since the last match (the pending litrun; becomes
    /// the next `Seq::litrunlen`, or the block's trailing literals at flush).
    litrun: u32,
    litlen_freqs: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: [u32; DEFLATE_NUM_OFFSET_SYMS],
    /// Input bytes covered by the current block so far.
    block_length: usize,
    stats: BlockSplitStats,
    /// Sequences written so far. Paired with the preallocated `seqs` store:
    /// this replaces `Vec::len`, which `continue_block` had to LOAD on every
    /// token where libdeflate compares a pointer already in a register.
    nseqs: usize,
    /// L3-only adaptive split gate (`lazy_sparse_len3_guard_mul`; 0 = off).
    sparse_split_guard_mul: u32,
    /// When true, non-ultra-sparse blocks need [`L3_NON_ULTRA_SPLIT_MIN_BYTES`]
    /// before an entropy split (high block-rate inputs only).
    sparse_split_hold: bool,
}

/// Everything [`emit_block`] needs to render ONE DEFLATE block: a contiguous
/// run of sequences, the trailing literal count, the matching symbol
/// histograms, and the input byte span they cover.
///
/// A [`Sink`] is exactly one of these ([`Sink::view`]), which is how every
/// parser reaches the emitter. It is a separate type so the post-parse block
/// splitter ([`postsplit`]) can hand the SAME emitter a SUB-RANGE of a longer
/// token buffer without copying the sequences.
pub(crate) struct BlockView<'a> {
    seqs: &'a [Seq],
    litrun: u32,
    litlen_freqs: &'a [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: &'a [u32; DEFLATE_NUM_OFFSET_SYMS],
    block_length: usize,
}

thread_local! {
    /// One recycled [`Sink`] per thread — the exact shape already used for
    /// `HcMatchfinder` (`matchfinder/hc.rs`'s `HC_POOL`), for the same reason.
    ///
    /// STRUCTURAL. `Sink::new()` allocates `SEQ_STORE_CAPACITY` sequences =
    /// **2,796,896 bytes**, and it was built fresh on EVERY parser invocation:
    /// once per streaming pass at T=1, and ONCE PER CHUNK at T>1. The 262,144-byte
    /// matchfinder beside it was pooled; the object ten times larger was not. That
    /// asymmetry is most of why the T>1 path allocated 83,909,568 bytes to compress
    /// 6,000,000 (T=1: 13,647,061) while libdeflate allocates 6,674,327 once.
    ///
    /// `thread_local!` rather than a shared pool, for the reason `HC_POOL` gives:
    /// each `infra::scheduler` worker owns its slot, so there is no cross-thread
    /// mutable state and no synchronisation.
    static SINK_POOL: std::cell::RefCell<Option<Sink>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII handle from [`Sink::acquire`]. `Deref`/`DerefMut` to [`Sink`] so every
/// existing `&mut sink` / `&sink` call site is unchanged. On drop the `Sink` goes
/// back to this thread's [`SINK_POOL`] instead of being freed, so the next
/// `acquire()` on the same thread reuses the 2.8 MB `seqs` allocation.
pub(crate) struct PooledSink(Option<Sink>);

impl std::ops::Deref for PooledSink {
    type Target = Sink;
    #[inline]
    fn deref(&self) -> &Sink {
        // SAFETY/invariant: `Some` for the whole lifetime outside `Drop::drop`.
        self.0.as_ref().expect("PooledSink used after drop")
    }
}

impl std::ops::DerefMut for PooledSink {
    #[inline]
    fn deref_mut(&mut self) -> &mut Sink {
        self.0.as_mut().expect("PooledSink used after drop")
    }
}

impl Drop for PooledSink {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            SINK_POOL.with(|cell| {
                *cell.borrow_mut() = Some(s);
            });
        }
    }
}

impl Sink {
    /// Take this thread's recycled [`Sink`], or build one on first use.
    ///
    /// The returned sink is `begin()`-reset, so it is indistinguishable from a
    /// fresh `Sink::new()` to every caller — `begin()` already zeroes the
    /// frequencies and resets `nseqs`/`litrun`/`block_length`, and `seqs` is
    /// written through `nseqs` and never read past it. Emitted bytes are
    /// therefore identical whether the sink is fresh or recycled.
    pub(crate) fn acquire() -> PooledSink {
        let mut s = SINK_POOL
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_else(Sink::new);
        s.begin();
        PooledSink(Some(s))
    }
}

impl Sink {
    fn new() -> Self {
        Sink {
            // Capacity for the worst-case block ANY parser can produce, never
            // grown and never initialised. `Vec::push` cost a capacity check on
            // EVERY match — 22.6M Ir of alloc/vec/mod.rs inlined into the L2 hot
            // loop — against libdeflate's `seq->offset = ...; seq++` into a
            // fixed array. The bound is not a hope; see `SEQ_STORE_CAPACITY`.
            seqs: Vec::with_capacity(SEQ_STORE_CAPACITY),
            litrun: 0,
            litlen_freqs: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset_freqs: [0; DEFLATE_NUM_OFFSET_SYMS],
            block_length: 0,
            stats: BlockSplitStats::new(),
            nseqs: 0,
            sparse_split_guard_mul: 0,
            sparse_split_hold: false,
        }
    }

    /// The sequences written so far.
    ///
    /// SAFETY-carrying accessor: `seqs` deliberately keeps `Vec::len == 0` and
    /// is written through spare capacity by [`Sink::push_seq`], so the slice
    /// must be built by hand. Exactly `nseqs` slots are initialised.
    #[inline]
    fn seq_slice(&self) -> &[Seq] {
        // SAFETY: `push_seq` has initialised exactly `nseqs` slots inside the
        // reserved capacity (see its own SAFETY note for the bound).
        unsafe { std::slice::from_raw_parts(self.seqs.as_ptr(), self.nseqs) }
    }

    /// This sink as one whole block, for [`emit_block`].
    #[inline]
    fn view(&self) -> BlockView<'_> {
        BlockView {
            seqs: self.seq_slice(),
            litrun: self.litrun,
            litlen_freqs: &self.litlen_freqs,
            offset_freqs: &self.offset_freqs,
            block_length: self.block_length,
        }
    }

    /// `deflate_begin_sequences` + `init_block_split_stats`.
    fn begin(&mut self) {
        self.nseqs = 0;
        self.litrun = 0;
        self.litlen_freqs = [0; DEFLATE_NUM_LITLEN_SYMS];
        self.offset_freqs = [0; DEFLATE_NUM_OFFSET_SYMS];
        self.block_length = 0;
        self.stats.reset();
        self.sparse_split_guard_mul = 0;
        self.sparse_split_hold = false;
    }

    /// `deflate_choose_literal` (with split-stat gathering always on, as greedy
    /// and lazy do).
    #[inline]
    fn push_literal(&mut self, lit: u8) {
        crate::anatomy_count!(literals_emitted);
        crate::anatomy_count!(histogram_updates);
        // `bucket-oracle-no-histogram` (Cargo.toml doc comment): skip the
        // freq/stats histogram work, keep the real emission bookkeeping.
        #[cfg(not(feature = "bucket-oracle-no-histogram"))]
        {
            // SAFETY: `lit` is a u8 (0..=255) and `litlen_freqs` has
            // DEFLATE_NUM_LITLEN_SYMS (288) entries, so `lit as usize` is in bounds.
            unsafe {
                *self.litlen_freqs.get_unchecked_mut(lit as usize) += 1;
            }
            self.stats.observe_literal(lit);
        }
        // `bucket-oracle-no-emission` (Cargo.toml doc comment): skip the
        // emission bookkeeping that feeds the eventual token, keep the real
        // histogram work above.
        #[cfg(not(feature = "bucket-oracle-no-emission"))]
        {
            self.litrun += 1;
            self.block_length += 1;
        }
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
        crate::anatomy_count!(literals_emitted);
        crate::anatomy_count!(literals_emitted_fast);
        crate::anatomy_count!(histogram_updates);
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
        // SAFETY: `nseqs < SEQ_STORE_CAPACITY` on entry, for EVERY parser, and
        // `begin()` resets `nseqs` to 0 for each block:
        //   * greedy/lazy check `continue_block` after each push and stop at
        //     `SEQ_STORE_LENGTH`, well below capacity;
        //   * `fast` bounds its block at `FAST_BLOCK_LENGTH` (65536) bytes;
        //   * `near_optimal` bounds its block at `choose_max_block_end`.
        // In all three the block span is within `SEQ_STORE_CAPACITY * 3` bytes
        // and each sequence eats at least 3 of them. The capacity test
        // `Vec::push` performed here was therefore provably dead.
        debug_assert!(self.nseqs < SEQ_STORE_CAPACITY);
        debug_assert!(self.seqs.capacity() >= SEQ_STORE_CAPACITY);
        // SAFETY: writing into reserved-but-uninitialised capacity. `nseqs` is
        // below `SEQ_STORE_LENGTH` (see above) and capacity is at least that,
        // so the slot is inside the allocation. `Seq` is `Copy` with no `Drop`,
        // so overwriting a never-initialised slot is sound and leaks nothing.
        unsafe {
            self.seqs.as_mut_ptr().add(self.nseqs).write(Seq {
                litrunlen: self.litrun,
                offset: offset as u16,
                length_and_slot: (length as u16) | ((os as u16) << SEQ_SLOT_SHIFT),
            });
        }
        self.nseqs += 1;
        self.litrun = 0;
    }

    /// Fast-path match push: frequencies + sequence only (see [`Self::push_literal_fast`]).
    #[inline]
    fn push_match_fast(&mut self, length: u32, offset: u32) {
        crate::anatomy_count!(matches_emitted);
        crate::anatomy_count!(matches_emitted_fast);
        crate::anatomy_count!(histogram_updates, 2u64);
        crate::anatomy_count!(match_length_bytes_total, length);
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
        crate::anatomy_count!(matches_emitted);
        crate::anatomy_count!(histogram_updates, 2u64);
        crate::anatomy_count!(match_length_bytes_total, length);
        debug_assert!((DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN).contains(&length));
        debug_assert!((1..=32768).contains(&offset));
        let ls = length_slot(length) as usize;
        let os = offset_slot(offset) as usize;
        // `bucket-oracle-no-histogram` (Cargo.toml doc comment): skip the
        // freq/stats histogram work, keep the real emission bookkeeping.
        #[cfg(not(feature = "bucket-oracle-no-histogram"))]
        {
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
        }
        // `bucket-oracle-no-emission` (Cargo.toml doc comment): skip the
        // token write + emission bookkeeping, keep the real histogram work
        // above.
        #[cfg(not(feature = "bucket-oracle-no-emission"))]
        {
            self.push_seq(length, offset, os);
            self.block_length += length as usize;
        }
    }
}

/// L1 [`fast::run`] dispatch: `GZIP_HASH` is a const generic, so the mmap
/// pick-min gzip arm is selected via a runtime branch over two monomorphizations.
#[allow(clippy::too_many_arguments)]
fn fast_run_dispatch<const REACH: bool, const INTERLEAVED: bool>(
    gzip_primary: bool,
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    input_total_len: usize,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
    block_length: usize,
    use_dynamic: bool,
    limit_hash_update_inserts: usize,
    bucket2: fast::Bucket2Cfg,
    cost_gate: fast::LazyPeekCostGateCfg,
    budget: HeaderBudget,
) {
    if gzip_primary {
        fast::run::<false, REACH, INTERLEAVED, true>(
            buf,
            data_start,
            in_end,
            input_total_len,
            statics,
            bw,
            is_last,
            block_length,
            use_dynamic,
            limit_hash_update_inserts,
            bucket2,
            cost_gate,
            budget,
        );
    } else {
        fast::run::<false, REACH, INTERLEAVED, false>(
            buf,
            data_start,
            in_end,
            input_total_len,
            statics,
            bw,
            is_last,
            block_length,
            use_dynamic,
            limit_hash_update_inserts,
            bucket2,
            cost_gate,
            budget,
        );
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
    input_total_len: usize,
    params: &LevelParams,
    is_last: bool,
    budget: HeaderBudget,
    bw: &mut BitWriter,
) {
    let statics = StaticCodes::get();
    match params.strategy {
        // ACCEL is a const generic (see fast::run's doc comment): `::<true>`
        // (L0) monomorphizes with the scan-step ramp; `::<false>` (L1)
        // monomorphizes with that code compiled away entirely, not merely
        // runtime-disabled.
        // L0: neither REACH nor INTERLEAVED — `fastloop_l0` has no 2-way
        // bucket and no interior-insert shift at all.
        Strategy::Fast0 => fast::run::<true, false, false, false>(
            buf,
            data_start,
            in_end,
            input_total_len,
            statics,
            bw,
            is_last,
            fast::FAST0_BLOCK_LENGTH,
            true,
            fast::LIMIT_HASH_UPDATE_INSERTS_L0,
            fast::Bucket2Cfg::DISABLED,
            fast::LazyPeekCostGateCfg::DISABLED,
            budget,
        ),
        // `l1-tune` (2026-07-22 L1-band search campaign, OFF by default):
        // block length and insert-depth are already plain runtime params to
        // `fast::run`, so overriding them for the search is just swapping
        // the two consts below for the env-var-backed tune values here — no
        // change to `fast::run`'s signature needed. Byte-identical to the
        // `not(feature)` arm when no `GZIPPY_L1TUNE_*` env var is set.
        #[cfg(not(feature = "l1-tune"))]
        Strategy::Fast => {
            let bucket2 = fast::Bucket2Cfg {
                enabled: params.fast_bucket2,
                gate_max_len: params.fast_bucket2_gate_max_len,
                probe_on_miss: params.fast_bucket2_probe_on_miss,
            };
            let cost_gate = fast::LazyPeekCostGateCfg {
                enabled: params.fast_lazy_peek_cost_gate,
                margin_bits: params.fast_lazy_peek_cost_margin_bits,
                lit_threshold_pct: 98,
                sparse_guard_mul: params.fast_lazy_peek_sparse_guard_mul,
                sparse_margin_bits: params.fast_lazy_peek_sparse_margin_bits,
            };
            // REACH / INTERLEAVED dispatch — const generics, one branch per
            // whole-buffer call. T>1 (`fast_interleaved_bucket`) and T1 REACH
            // (`fast_dense_interior_insert`) are mutually exclusive by construction.
            if params.fast_interleaved_bucket {
                fast_run_dispatch::<false, true>(
                    params.fast_gzip_primary,
                    buf,
                    data_start,
                    in_end,
                    input_total_len,
                    statics,
                    bw,
                    is_last,
                    fast::FAST_BLOCK_LENGTH,
                    true,
                    params.fast_hash_update_inserts,
                    bucket2,
                    cost_gate,
                    budget,
                )
            } else if params.fast_dense_interior_insert {
                fast_run_dispatch::<true, false>(
                    params.fast_gzip_primary,
                    buf,
                    data_start,
                    in_end,
                    input_total_len,
                    statics,
                    bw,
                    is_last,
                    fast::FAST_BLOCK_LENGTH,
                    true,
                    params.fast_hash_update_inserts,
                    bucket2,
                    cost_gate,
                    budget,
                )
            } else {
                fast_run_dispatch::<false, false>(
                    params.fast_gzip_primary,
                    buf,
                    data_start,
                    in_end,
                    input_total_len,
                    statics,
                    bw,
                    is_last,
                    fast::FAST_BLOCK_LENGTH,
                    true,
                    params.fast_hash_update_inserts,
                    bucket2,
                    cost_gate,
                    budget,
                )
            }
        }
        #[cfg(feature = "l1-tune")]
        Strategy::Fast => {
            let t = fast::tune::get();
            let bucket2 = fast::Bucket2Cfg {
                enabled: params.fast_bucket2,
                gate_max_len: params.fast_bucket2_gate_max_len,
                probe_on_miss: params.fast_bucket2_probe_on_miss,
            };
            let cost_gate = fast::LazyPeekCostGateCfg {
                enabled: params.fast_lazy_peek_cost_gate,
                margin_bits: params.fast_lazy_peek_cost_margin_bits,
                lit_threshold_pct: 98,
                sparse_guard_mul: params.fast_lazy_peek_sparse_guard_mul,
                sparse_margin_bits: params.fast_lazy_peek_sparse_margin_bits,
            };
            // Same REACH / INTERLEAVED dispatch as the default-build arm above.
            if params.fast_interleaved_bucket {
                fast_run_dispatch::<false, true>(
                    params.fast_gzip_primary,
                    buf,
                    data_start,
                    in_end,
                    input_total_len,
                    statics,
                    bw,
                    is_last,
                    t.block_length,
                    true,
                    params.fast_hash_update_inserts,
                    bucket2,
                    cost_gate,
                    budget,
                )
            } else if params.fast_dense_interior_insert {
                fast_run_dispatch::<true, false>(
                    params.fast_gzip_primary,
                    buf,
                    data_start,
                    in_end,
                    input_total_len,
                    statics,
                    bw,
                    is_last,
                    t.block_length,
                    true,
                    params.fast_hash_update_inserts,
                    bucket2,
                    cost_gate,
                    budget,
                )
            } else {
                fast_run_dispatch::<false, false>(
                    params.fast_gzip_primary,
                    buf,
                    data_start,
                    in_end,
                    input_total_len,
                    statics,
                    bw,
                    is_last,
                    t.block_length,
                    true,
                    params.fast_hash_update_inserts,
                    bucket2,
                    cost_gate,
                    budget,
                )
            }
        }
        Strategy::Greedy => greedy::run(
            buf, data_start, in_end, params, statics, bw, is_last, budget,
        ),
        Strategy::Lazy => lazy::run(
            buf, data_start, in_end, params, statics, bw, false, is_last, budget,
        ),
        Strategy::Lazy2 => lazy::run(
            buf, data_start, in_end, params, statics, bw, true, is_last, budget,
        ),
        // DETECTOR-GATED LAZY-L3 (`l3-tune` feature): see `gated.rs`'s module
        // doc comment. `level.rs`'s L3 arm is the only producer of this
        // strategy; not reachable from a default (non-`l3-tune`) build.
        Strategy::NearOptimal => near_optimal::run(
            buf, data_start, in_end, params, statics, bw, is_last, budget,
        ),
    }
}

// ---- resumable (streaming) parse state ----

/// Everything a parser must carry from one streaming chunk to the next.
///
/// The whole-buffer entry points build this fresh per call, so their behaviour
/// is unchanged. The streaming encoder keeps ONE across the whole file, which
/// is what makes streamed output byte-identical to whole-buffer output: match
/// choices depend on the matchfinder's accumulated chains, so a matchfinder
/// rebuilt per chunk would make different choices at every seam even though
/// every candidate it could legally reference is within the 32 KiB window.
pub(super) struct ParseState {
    pub mf: crate::compress::deflate::matchfinder::hc::PooledHc,
    /// Base offset the matchfinder's stored positions are relative to.
    /// ALWAYS a multiple of [`WINDOW_SIZE`], with `in_next - in_base` in
    /// `0..WINDOW_SIZE` — see `HcMatchfinder`'s slide condition. The FAST
    /// (L1) strategy stores absolute positions instead and only maintains
    /// this field for the caller's slide arithmetic — see
    /// `fast::run_resumable`'s closing comment.
    pub in_base: usize,
    pub next_hashes: [u32; 2],
    /// The FAST (L1) strategy's carried state (head tables + block gates),
    /// created lazily by `fast::run_resumable` on its first call. `None` for
    /// every other strategy — they carry their state in `mf` above.
    pub fast: Option<fast::FastResume>,
    /// Uncompressed input length when known. L1 ultra-sparse mid-block tier
    /// arms only once this reaches [`fast::L1_SPARSE_LARGE_INPUT_MIN_BYTES`].
    pub input_total_len: usize,
}

impl ParseState {
    pub fn new() -> Self {
        Self {
            mf: crate::anatomy_wall_time!(mf_new_ns, mf_new_calls, {
                crate::compress::deflate::matchfinder::hc::HcMatchfinder::acquire()
            }),
            in_base: 0,
            next_hashes: [0u32; 2],
            fast: None,
            input_total_len: 0,
        }
    }

    // Used by the sliding-buffer streaming loop, which lands next; the
    // resumable parsers above are the half of that change that could be
    // verified independently (whole-buffer output byte-identical, full suite
    // green), so it is committed separately rather than as one large diff.
    #[allow(dead_code)]
    /// Largest amount the caller may shift buffer contents down by while
    /// keeping at least one full window of history behind `in_base`.
    ///
    /// Returns a multiple of [`WINDOW_SIZE`], because the matchfinder stores
    /// each position as `pos - in_base` and slides in exact window steps: a
    /// shift that is not a whole number of windows would leave `in_next -
    /// in_base` outside `0..WINDOW_SIZE` and corrupt every chain index.
    pub fn max_shift(&self) -> usize {
        self.in_base.saturating_sub(WINDOW_SIZE)
    }

    #[allow(dead_code)]
    /// Tell the state the caller moved buffer contents down by `shift` bytes.
    ///
    /// O(1) and lossless: stored nodes are offsets from `in_base`, so
    /// decrementing `in_base` by the same amount leaves every
    /// `in_base + node` pointing at the same BYTE in the moved buffer. No
    /// table rebase, no chain rebuild. `shift` must come from
    /// [`max_shift`](Self::max_shift).
    pub fn shift_down(&mut self, shift: usize) {
        debug_assert_eq!(shift % WINDOW_SIZE, 0, "shift must be whole windows");
        debug_assert!(shift <= self.in_base, "shift would push in_base negative");
        self.in_base -= shift;
        // The FAST tables store absolute positions (not `pos - in_base`), so
        // they need a real rebase sweep — once per multi-megabyte slide.
        if let Some(f) = self.fast.as_mut() {
            f.rebase(shift);
        }
    }
}

/// Bytes that must be available past a block's start before the streaming
/// encoder may begin that block.
///
/// [`choose_max_block_end`] consults `in_end` ONLY when fewer than
/// `SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH` bytes remain; above that it
/// returns `block_begin + SOFT_MAX_BLOCK_LENGTH` regardless. So a streaming
/// chunk that always has at least this much input in hand makes exactly the
/// same block-boundary decisions as a whole-buffer encode — the seam becomes
/// invisible instead of forcing a short block. The same margin also keeps
/// `adjust_max_and_nice_len` from clamping match lengths near the buffer end,
/// since it exceeds `DEFLATE_MAX_MATCH_LEN` by three orders of magnitude.
pub(super) const STREAM_BLOCK_LOOKAHEAD: usize = SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH;

/// Whether `level`'s strategy has a resumable runner, so the streaming encoder
/// can carry one matchfinder across chunks and emit byte-identical output.
///
/// "Byte-identical" here means identical to OUR OWN whole-buffer output at the
/// same level — the T>1 == T1 invariant. It does NOT mean identical to
/// libdeflate's, and must never be read that way.
pub(crate) fn level_has_resumable_parser(level: u32) -> bool {
    // A level that runs WHOLE-BUFFER PICK-MIN cannot stream. The streaming
    // parse is ONE arm; the whole-buffer path runs two and keeps the smaller,
    // so streaming such a level does not merely reorder blocks — it emits a
    // DIFFERENT, LARGER stream than the same input passed as a file. That
    // violates this module's own rule ("a level streams only when its output
    // is PROVABLY unaffected by streaming") and the contract it cites
    // ("being at-least-as-small at the level the user typed").
    //
    // This is the SAME reasoning that already excludes L1 below; it was simply
    // never generalised to the other pick-min levels when they landed.
    //
    // MEASURED on main e888ac9f, 23-file corpus x L4-L7, T1, size only:
    // piping cost 5,429,807 B = 0.6377% larger, up to 2.535% at L4
    // (minjs.min.js). Per-level on engine.wasm: L4 +9,528 B, L5 +1,085,
    // L6 +754, L7 +70; L3/L8/L9 byte-identical (they run no pick-min).
    if super::level_uses_t1_zlib_pick_min(level) || super::level_uses_t1_mmap_pick_min(level) {
        return false;
    }
    // A level that RE-CUTS its blocks over a 1,000,000-byte span cannot stream
    // either, for the same reason and with a sharper failure mode. The span
    // holds tokens whose literals are read back out of the input buffer at
    // emit time, so it must be flushed before either (a) a resumable call
    // returns, which cuts the span at a seam the whole-buffer encoder does not
    // have, or (b) the streaming buffer slides, which would drop the literal
    // bytes the span still needs.
    //
    // (a) is not hypothetical: the mmap two-pass route
    // (`encode_gzip_unpadded_slice_to_writer`) calls the resumable parser
    // twice, and with the span flushed at the pass-1 return
    // `unpadded_slice_is_byte_identical_to_whole_buffer` FAILED at L9,
    // len=305,017 — one forced boundary the whole-buffer encoder never emits.
    //
    // Excluding these levels routes both the pipe and the mmap paths to the
    // whole-buffer encoder, which is byte-identical by construction. It costs
    // the in-place mmap parse and the streaming RSS bound at L8/L9; see this
    // branch's verdict commit, which prices that.
    if super::level::level_uses_postsplit(level) {
        return false;
    }
    let strategy = super::level::params(level).strategy;
    if matches!(
        strategy,
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2
    ) {
        return true;
    }
    // `Strategy::Fast` (L1) streams via `fast::run_resumable` in tune builds
    // only. Shipped L1 runs mmap pick-min (`deflate_one_shot_t1_l1_pick_min`),
    // which is whole-buffer only — the resumable fast path would diverge from
    // `encode_gzip_slack_padded_to_vec` and mmap pick-min
    // (`tests/streaming_identity.rs`).
    #[cfg(not(feature = "l1-tune"))]
    if matches!(strategy, Strategy::Fast) {
        return false;
    }
    false
}

/// Greedy/Lazy/Lazy2 only — the T>1 stateful path uses `parse_resumable` with
/// `params_parallel`, which must not be applied to L1's parallel fast knobs.
pub(crate) fn level_uses_stateful_t4(level: u32) -> bool {
    matches!(
        super::level::params(level).strategy,
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2
    )
}

/// Resume a parse over `buf[from..in_end]` using caller-owned `state`.
///
/// See `greedy::run_resumable` for the `consume_all` / `is_last` distinction
/// and why the lookahead margin keeps block boundaries identical to a
/// whole-buffer encode. Returns the position after the last complete block.
/// Callers must check [`level_has_resumable_parser`] first; other strategies
/// panic rather than silently emitting a differently-shaped stream.
#[allow(clippy::too_many_arguments)]
pub(super) fn parse_resumable(
    buf: &[u8],
    state: &mut ParseState,
    from: usize,
    in_end: usize,
    params: &LevelParams,
    role: BlockRole,
    input_mode: InputMode,
    budget: super::encode_types::HeaderBudget,
    bw: &mut BitWriter,
) -> usize {
    let statics = StaticCodes::get();
    match params.strategy {
        // Same const arguments as `parse()`'s whole-buffer `Strategy::Fast`
        // arm (block length / dynamic emitter / insert depth); `params`
        // carries nothing the fast parser reads. Default builds only — see
        // `level_has_resumable_parser` for why `l1-tune` never routes here.
        // REACH DISPATCH, same as `compress`'s whole-buffer arm — one branch
        // per resumable call selecting a whole monomorphization of the L1
        // parser, not a per-position runtime test. This is the arm T1's
        // STREAMING path takes, so it is where the two record-file cells
        // (`libdeflate:{access.log,ecoli.fastq}:L1:T1:size`) are actually won.
        #[cfg(not(feature = "l1-tune"))]
        Strategy::Fast => {
            if params.fast_dense_interior_insert {
                if params.fast_gzip_primary {
                    fast::run_resumable::<true, true>(
                        buf,
                        state,
                        from,
                        in_end,
                        params,
                        statics,
                        bw,
                        role,
                        input_mode,
                        fast::FAST_BLOCK_LENGTH,
                        true,
                        budget,
                    )
                } else {
                    fast::run_resumable::<true, false>(
                        buf,
                        state,
                        from,
                        in_end,
                        params,
                        statics,
                        bw,
                        role,
                        input_mode,
                        fast::FAST_BLOCK_LENGTH,
                        true,
                        budget,
                    )
                }
            } else if params.fast_gzip_primary {
                fast::run_resumable::<false, true>(
                    buf,
                    state,
                    from,
                    in_end,
                    params,
                    statics,
                    bw,
                    role,
                    input_mode,
                    fast::FAST_BLOCK_LENGTH,
                    true,
                    budget,
                )
            } else {
                fast::run_resumable::<false, false>(
                    buf,
                    state,
                    from,
                    in_end,
                    params,
                    statics,
                    bw,
                    role,
                    input_mode,
                    fast::FAST_BLOCK_LENGTH,
                    true,
                    budget,
                )
            }
        }
        Strategy::Greedy => greedy::run_resumable(
            buf, state, from, in_end, params, statics, bw, role, input_mode, budget,
        ),
        Strategy::Lazy | Strategy::Lazy2 => lazy::run_resumable(
            buf,
            state,
            from,
            in_end,
            params,
            statics,
            bw,
            matches!(params.strategy, Strategy::Lazy2),
            role,
            input_mode,
            budget,
        ),
        other => unreachable!("parse_resumable called for non-resumable strategy {other:?}"),
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

/// Ultra-sparse block test (shared with L3 lazy len-3 guard): `nseqs*64 ≤ bytes`
/// AND `nseqs*M < bytes`. `guard_mul == 0` means disabled.
#[inline]
fn block_ultra_sparse(sink: &Sink, bytes_in_block: usize, guard_mul: u32) -> bool {
    guard_mul > 0
        && sink.nseqs.saturating_mul(64) <= bytes_in_block
        && sink.nseqs.saturating_mul(guard_mul as usize) < bytes_in_block
}

/// Minimum in-block bytes before adaptive split on semi-sparse L3 blocks.
const L3_NON_ULTRA_SPLIT_MIN_BYTES: usize = 50_000;

/// Latch the over-split hold after this many completed blocks.
pub(super) const L3_OVER_SPLIT_LATCH_BLOCKS: u32 = 8;

/// Arm when mean block size is in this band (ecoli FASTQ over-split signature).
pub(super) const L3_OVER_SPLIT_AVG_BLOCK_MIN_BYTES: usize = 50_000;
pub(super) const L3_OVER_SPLIT_AVG_BLOCK_MAX_BYTES: usize = 65_000;

/// Arm when block rate is at or below this (blocks per MiB).
pub(super) const L3_OVER_SPLIT_MAX_BLOCKS_PER_MIB: u32 = 20;

/// Whether the L3 over-split hold is active for the current block.
#[inline]
pub(super) fn l3_sparse_split_hold(guard_mul: u32, latched: bool) -> bool {
    guard_mul > 0 && latched
}

/// Decide whether to arm the split hold for the rest of the file.
#[inline]
pub(super) fn l3_sparse_split_latch(
    guard_mul: u32,
    blocks_completed: u32,
    file_start: usize,
    block_begin: usize,
) -> bool {
    if guard_mul == 0 || blocks_completed != L3_OVER_SPLIT_LATCH_BLOCKS {
        return false;
    }
    let bytes = block_begin - file_start;
    if bytes == 0 {
        return false;
    }
    let prior_avg = bytes / blocks_completed as usize;
    let blocks_per_mib = blocks_completed.saturating_mul(1_048_576) / bytes as u32;
    blocks_per_mib <= L3_OVER_SPLIT_MAX_BLOCKS_PER_MIB
        && prior_avg > L3_OVER_SPLIT_AVG_BLOCK_MIN_BYTES
        && prior_avg < L3_OVER_SPLIT_AVG_BLOCK_MAX_BYTES
}

/// Whether adaptive block-split may run at this position (L3 gate).
///
/// Non-ultra-sparse blocks over-split on FASTQ; hold them to
/// [`L3_NON_ULTRA_SPLIT_MIN_BYTES`] before entropy split. Ultra-sparse blocks
/// keep libdeflate's adaptive cadence.
#[inline]
fn sparse_split_active(sink: &Sink, bytes_in_block: usize, sparse_split_guard_mul: u32) -> bool {
    if sparse_split_guard_mul == 0 || !sink.sparse_split_hold {
        return true;
    }
    block_ultra_sparse(sink, bytes_in_block, sparse_split_guard_mul)
        || bytes_in_block >= L3_NON_ULTRA_SPLIT_MIN_BYTES
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
    let bytes_in_block = in_next - block_begin;
    let guard_mul = sink.sparse_split_guard_mul;
    let end_block = if guard_mul > 0 {
        sparse_split_active(sink, bytes_in_block, guard_mul)
            && sink
                .stats
                .should_end_block(bytes_in_block, in_end - in_next)
    } else {
        sink.stats
            .should_end_block(bytes_in_block, in_end - in_next)
    };
    in_next < in_max_block_end && sink.nseqs < SEQ_STORE_LENGTH && !end_block
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

/// Coalesces CONSECUTIVE parse blocks that each chose STORED into one deferred
/// span, so the physical BTYPE=00 sub-block grid over that span is maximal
/// (all 65,535-byte sub-blocks, the last takes the remainder).
///
/// THE DEFECT THIS REMOVES (issue #266): the fast path parses in
/// [`fast::FAST_BLOCK_LENGTH`] = 65,536-byte blocks, and `emit_stored_block`
/// splits each block's payload independently — `ceil(65536/65535)` = 2
/// sub-blocks of 65,535 + 1 bytes. On incompressible input every parse block
/// goes stored, so a 1 MiB file emitted a 65535/1 ALTERNATING grid: 32 stored
/// blocks where libdeflate (whose fast path parses in 65,535-byte blocks)
/// emits 17. Each extra sub-block is 5 bytes of pure framing (3-bit header +
/// byte-align pad + LEN/NLEN).
///
/// MONOTONE BY CONSTRUCTION: the per-block stored/static/dynamic DECISION is
/// unchanged (same costs, same inputs); only already-chosen-stored payloads
/// are repacked, and `ceil((a+b)/65535) <= ceil(a/65535) + ceil(b/65535)`,
/// so the framing count can only shrink. A Huffman block or end-of-call
/// flushes the pending span first, preserving byte order.
///
/// Callers that pass `None` to [`emit_block`] /
/// [`emit_block_static_or_stored`] keep today's emit-immediately behavior
/// byte-for-byte (greedy/lazy/near_optimal/ht_fast are tie-locked to
/// libdeflate's block grid; only the fast path opts in).
///
/// EMISSION IS EAGER, NOT LAZY: `push` writes every full 65,535-byte
/// sub-block the moment the span covers it, holding back only a 1..=65,535
/// byte remainder (never zero while a span is open, so the eventual `flush`
/// always has a sub-block to carry BFINAL). Two properties depend on this:
///
/// * **Identical bytes to a whole-span lazy flush** — the sub-block grid over
///   a span is `start + k*65535` either way; eager just writes each piece as
///   soon as its bytes cannot change.
/// * **Bounded carry** — the pending remainder is <= 65,535 bytes, so the
///   streaming path ([`fast::run_resumable`]) can carry it across input
///   refills in `FastResume` (keeping streamed L1 byte-identical to the
///   whole-buffer encoder, the invariant `tests/streaming_identity.rs` pins)
///   while the slide only has to retain a bounded window of unemitted input.
#[derive(Default)]
pub(super) struct StoredCoalescer {
    /// Absolute offset in `buf` of the pending remainder's first byte.
    start: usize,
    /// Pending remainder length, always <= 65,535 after `push`; 0 means empty.
    len: usize,
}

impl StoredCoalescer {
    /// Append a block that chose STORED to the pending span, eagerly writing
    /// every completed maximal sub-block (see the struct doc). Blocks arrive
    /// in input order and are contiguous, so the span only grows rightward.
    fn push(&mut self, bw: &mut BitWriter, buf: &[u8], block_start: usize, block_len: usize) {
        debug_assert!(block_len > 0, "empty blocks never price stored cheapest");
        if self.len == 0 {
            self.start = block_start;
        }
        debug_assert_eq!(
            self.start + self.len,
            block_start,
            "non-contiguous stored blocks"
        );
        self.len += block_len;
        // `>` not `>=`: keep at least one byte pending so BFINAL always has
        // a carrier sub-block at the final `flush`.
        while self.len > super::MAX_STORED_SUBBLOCK {
            super::emit_stored_block(
                bw,
                &buf[self.start..self.start + super::MAX_STORED_SUBBLOCK],
                false,
            );
            self.start += super::MAX_STORED_SUBBLOCK;
            self.len -= super::MAX_STORED_SUBBLOCK;
        }
    }

    /// Write the pending remainder (if any) as one stored sub-block and clear
    /// it. No-op when empty. `is_final` marks it BFINAL.
    pub(super) fn flush(&mut self, bw: &mut BitWriter, buf: &[u8], is_final: bool) {
        if self.len == 0 {
            return;
        }
        debug_assert!(self.len <= super::MAX_STORED_SUBBLOCK);
        super::emit_stored_block(bw, &buf[self.start..self.start + self.len], is_final);
        self.len = 0;
    }

    /// Whether a remainder is pending (streaming: it must survive the slide).
    pub(super) fn is_pending(&self) -> bool {
        self.len > 0
    }

    /// Absolute buffer offset of the pending remainder (meaningless when
    /// [`Self::is_pending`] is false).
    pub(super) fn pending_start(&self) -> usize {
        self.start
    }

    /// The caller slid `buf` down by `shift` bytes; keep the pending
    /// remainder pointing at the same bytes. The slide contract
    /// ([`fast::run_resumable`]'s `in_base` accounting) guarantees the
    /// remainder is retained, so `start` cannot underflow.
    pub(super) fn rebase(&mut self, shift: usize) {
        if self.len > 0 {
            debug_assert!(shift <= self.start, "slide discarded a pending stored span");
            self.start -= shift;
        }
    }
}

/// Emit the accumulated block, choosing the cheapest of stored / static-Huffman
/// / dynamic-Huffman. `block_start` is the absolute offset of the block's first
/// byte in `buf`.
/// PARKED 2026-08-01 — RLE-shaped histogram, gated to the T>1 path. A STRICT SIZE
/// WIN that dies on clause 5, not on the encoder. Branch `lever/rle-shape-t4`
/// (`b9cf59ef`), adjudicated NO-SHIP; artifact
/// `~/www/gzippy-bench/campaign/lever-lever-rle-shape-t4/try.json`.
///
/// THE MECHANISM. Beside the true histogram, cost a second one shaped the way zopfli's
/// `TryOptimizeHuffmanForRle` shapes it, and keep it only when the header comes out
/// STRICTLY smaller. A `HeaderBudget` enum threaded to this function makes it
/// `Generous` on the T>1 path and `Lean` everywhere else, so T1 never pays.
///
/// WHAT IT BOUGHT (deterministic, both censuses over the same 792 comparable cells):
///     23 size cells CLOSED, 0 opened — all `libdeflate_T4`, spread over 7 levels
///     T1 output byte-identical to main: 198/198 (22 corpus files x levels 1-9)
///     T4 strictly smaller on every file checked (-151 .. -208 B on armexe.elf)
/// It cannot make a block bigger: the shaped code is adopted only when it is cheaper.
///
/// WHAT IT COST. `make lever ARGS="--threads 1,4"`, NO-SHIP on clause 5 with 25 cells
/// over the 0.005 erosion budget: 18 at T4, 12 of them against pigz. `Generous` runs a
/// second symbol-sort + package-merge per block, which is real work on the exact path
/// it is gated to. That is the FOURTH size lever to die in this clause.
///
/// ⚠ THE EROSION MAGNITUDES ARE NOT TRUSTWORTHY AND MUST NOT BE QUOTED. 7 of the 25
/// cells were T1, where the output is provably byte-identical, so those readings are
/// impossible and the run was contaminated: ~12 `fulcrum ab paired` jobs were run on
/// the same laptop during rows 176-264 of it, and `scripts/campaign/lever.sh` invokes
/// no box-freeze guard. The VERDICT stands (18 T4 cells, real mechanism, 4-8x over
/// budget); the NUMBERS need an idle re-run before any successor prices against them.
///
/// WHAT WOULD REVIVE IT. Not a re-run, and not tuning the shaping. Only a way to
/// obtain the shaped candidate for materially less than a second package-merge —
/// or a clause-5 budget argument made at a coordinate where T4 wall slack is real
/// rather than assumed. Compose it with a T4 wall win before re-adjudicating; on its
/// own the size case is already proven and already insufficient.
///
/// Do not re-derive the size win. It is measured, it is 23 cells, and it is not the
/// part that fails.
#[allow(clippy::too_many_arguments)]
fn emit_block(
    bw: &mut BitWriter,
    buf: &[u8],
    block_start: usize,
    blk: &BlockView,
    statics: &StaticCodes,
    is_final: bool,
    header_scratch: &mut HeaderScratch,
    code_scratch: &mut CodeScratch,
    try_exact: bool,
    pending_stored: Option<&mut StoredCoalescer>,
) {
    // `anatomy-wall` region: `huffman_table` — the code-BUILDING phase for
    // this block, before any bit is written: both candidate Huffman codes,
    // the dynamic header, and the three-way stored/static/dynamic cost
    // comparison. Zero cost when `anatomy-wall` is off.
    // The two codes are built INTO `code_scratch`, which the caller owns and
    // reuses for every block. See `huffman::CodeScratch`: building them fresh here
    // was the per-block allocation that made our allocation count grow with input
    // (733 allocs on 6 MB where libdeflate uses 3 and gzip 0).
    let CodeScratch {
        litcode,
        offcode,
        alt_litcode,
        alt_offcode,
        alt_header,
        shape,
        budget,
    } = code_scratch;
    let budget = *budget;
    let (header, dynamic_bits, static_bits, stored_bits) =
        crate::anatomy_wall_time!(huffman_table_ns, huffman_table_calls, {
            // Add the end-of-block symbol to the litlen frequencies (as the
            // vendor does in deflate_flush_block).
            let mut litlen_freqs = *blk.litlen_freqs;
            litlen_freqs[DEFLATE_END_OF_BLOCK] += 1;

            let shaped = if budget.may_shape() {
                shaped_freqs_if_smaller(&litlen_freqs, blk.offset_freqs, shape)
            } else {
                None
            };
            let (build_lit, build_off): (&[u32], &[u32]) = match shaped {
                Some((ref l, ref o)) => (l, o),
                None => (&litlen_freqs, blk.offset_freqs),
            };

            make_huffman_code_into(
                litcode,
                DEFLATE_NUM_LITLEN_SYMS,
                MAX_LITLEN_CODEWORD_LEN,
                build_lit,
            );
            make_huffman_code_into(
                offcode,
                DEFLATE_NUM_OFFSET_SYMS,
                MAX_OFFSET_CODEWORD_LEN,
                build_off,
            );

            // SECOND DYNAMIC CANDIDATE: the same two codes under the EXACT
            // (Katajainen package-merge) length assignment.
            //
            // Package-merge minimises the CODED-DATA bits, but a dynamic block also
            // transmits its length vector in an RLE-coded header, and the two builders
            // produce different vectors — so "exact" is exact on data, NOT on total.
            // Measured: swapping unconditionally is a wash (data.json L6 -737 B but
            // winexe.exe L6 +385 B, which flips that cell from 178 B smaller than
            // libdeflate to 207 B larger). Costing BOTH and taking the cheaper is
            // non-worse than the heuristic BY CONSTRUCTION, so it cannot open a size
            // cell, and it is paid ONCE PER BLOCK rather than once per position —
            // the only cost shape that can fund the ~0.01% of margin the 109
            // zero-headroom T4 cells need. See docs/target-encoder-and-gap-analysis.md
            // G15 for why the parse side cannot.
            if try_exact {
                make_huffman_code_exact_into(
                    alt_litcode,
                    DEFLATE_NUM_LITLEN_SYMS,
                    MAX_LITLEN_CODEWORD_LEN,
                    &litlen_freqs,
                );
                make_huffman_code_exact_into(
                    alt_offcode,
                    DEFLATE_NUM_OFFSET_SYMS,
                    MAX_OFFSET_CODEWORD_LEN,
                    blk.offset_freqs,
                );
            }

            let heur_header = build_dynamic_header(&litcode.lens, &offcode.lens, header_scratch);
            let heur_bits = 3
                + heur_header.header_bits()
                + cost_from_freqs(&litlen_freqs, blk.offset_freqs, litcode, offcode);

            // The exact candidate is costed ONLY when enabled. Building its header
            // unconditionally would both pay for work T1 must not pay for and read
            // `alt_*code` scratch that was never filled. Strict `<` keeps the heuristic on
            // ties, so a tie emits today's bytes.
            //
            // At T>1 with RLE shaping enabled, also try package-merge on the SAME shaped
            // histogram the heuristic used (`build_lit`/`build_off`). True-frequency
            // exact and shaped-frequency exact are both costed with true token counts;
            // take the cheaper. T1 never shapes, so only the true-frequency arm runs.
            let (header, dynamic_bits) = if try_exact {
                let exact_header_true =
                    build_dynamic_header(&alt_litcode.lens, &alt_offcode.lens, alt_header);
                let exact_bits_true = 3
                    + exact_header_true.header_bits()
                    + cost_from_freqs(&litlen_freqs, blk.offset_freqs, alt_litcode, alt_offcode);

                enum Pick<'a> {
                    Heuristic(DynamicHeader<'a>, u64),
                    ExactTrue(DynamicHeader<'a>, u64),
                    ExactShaped(DynamicHeader<'a>, u64),
                }

                let mut pick = if exact_bits_true < heur_bits {
                    Pick::ExactTrue(exact_header_true, exact_bits_true)
                } else {
                    drop(exact_header_true);
                    Pick::Heuristic(heur_header, heur_bits)
                };

                if shaped.is_some() {
                    make_huffman_code_exact_into(
                        &mut shape.cand_litcode,
                        DEFLATE_NUM_LITLEN_SYMS,
                        MAX_LITLEN_CODEWORD_LEN,
                        build_lit,
                    );
                    make_huffman_code_exact_into(
                        &mut shape.cand_offcode,
                        DEFLATE_NUM_OFFSET_SYMS,
                        MAX_OFFSET_CODEWORD_LEN,
                        build_off,
                    );
                    let exact_header_shaped = build_dynamic_header(
                        &shape.cand_litcode.lens,
                        &shape.cand_offcode.lens,
                        &mut shape.raw_header,
                    );
                    let exact_bits_shaped = 3
                        + exact_header_shaped.header_bits()
                        + cost_from_freqs(
                            &litlen_freqs,
                            blk.offset_freqs,
                            &shape.cand_litcode,
                            &shape.cand_offcode,
                        );
                    let better = match &pick {
                        Pick::Heuristic(_, b) | Pick::ExactTrue(_, b) | Pick::ExactShaped(_, b) => {
                            exact_bits_shaped < *b
                        }
                    };
                    if better {
                        if let Pick::ExactTrue(h, _) | Pick::ExactShaped(h, _) = pick {
                            drop(h);
                        }
                        pick = Pick::ExactShaped(exact_header_shaped, exact_bits_shaped);
                    } else {
                        drop(exact_header_shaped);
                    }
                }

                match pick {
                    Pick::Heuristic(h, b) => (h, b),
                    Pick::ExactTrue(h, b) => {
                        crate::anatomy_count!(huffman_exact_code_chosen);
                        std::mem::swap(litcode, alt_litcode);
                        std::mem::swap(offcode, alt_offcode);
                        (h, b)
                    }
                    Pick::ExactShaped(h, b) => {
                        crate::anatomy_count!(huffman_exact_code_chosen);
                        std::mem::swap(litcode, &mut shape.cand_litcode);
                        std::mem::swap(offcode, &mut shape.cand_offcode);
                        (h, b)
                    }
                }
            } else {
                (heur_header, heur_bits)
            };
            let static_bits = 3 + cost_from_freqs(
                &litlen_freqs,
                blk.offset_freqs,
                &statics.litcode,
                &statics.offcode,
            );
            let stored_bits = stored_block_bits(blk.block_length);
            (header, dynamic_bits, static_bits, stored_bits)
        });
    let (litcode, offcode) = (&*litcode, &*offcode);

    crate::block_cost_probe!(
        sink.block_length,
        stored_bits,
        static_bits,
        dynamic_bits as i64,
        if stored_bits <= dynamic_bits && stored_bits <= static_bits {
            's'
        } else if static_bits <= dynamic_bits {
            'f'
        } else {
            'd'
        },
        'e'
    );

    if stored_bits <= dynamic_bits && stored_bits <= static_bits {
        // blocks_emitted_stored is counted in `write_stored_subblock`
        // (deflate/mod.rs) — the single physical-BTYPE=00-block emission
        // site, shared with the T>1 pipelined sync-flush path — not here,
        // to avoid double-counting (see that function's doc comment). Note
        // for `anatomy-wall`: a stored block never enters `huffman_encode`
        // at all (no Huffman machinery involved) — its byte-copy cost
        // lands in RESIDUAL, which is correct: it genuinely isn't Huffman
        // encoding time.
        match pending_stored {
            // Coalescing caller: defer, so an adjacent stored block can share
            // a maximal sub-block grid (see `StoredCoalescer`).
            Some(pending) => {
                pending.push(bw, buf, block_start, blk.block_length);
                if is_final {
                    pending.flush(bw, buf, true);
                }
            }
            None => super::emit_stored_block(
                bw,
                &buf[block_start..block_start + blk.block_length],
                is_final,
            ),
        }
    } else if static_bits <= dynamic_bits {
        // A pending stored span (coalescing callers only) precedes this
        // block in input order — write it before the first Huffman bit.
        if let Some(pending) = pending_stored {
            pending.flush(bw, buf, false);
        }
        crate::anatomy_count!(blocks_emitted_fixed);
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_STATIC_HUFFMAN as u64, 2);
        // `anatomy-wall` region: `huffman_encode` — walks this block's
        // tokens and writes codeword bits via `BitWriter`. Bitstream
        // flush/serialization (`BitWriter::add_bits`'s internal
        // `flush_word_unchecked`) is FUSED into this region, not
        // separately timed — see `anatomy_wall` module docs for why
        // (fires roughly once per ~56 bits, thousands of times per block).
        crate::anatomy_wall_time!(huffman_encode_ns, huffman_encode_calls, {
            emit_sequences(
                bw,
                buf,
                block_start,
                blk,
                &statics.litcode,
                &statics.offcode,
            );
        });
    } else {
        // Same ordering rule as the static arm above.
        if let Some(pending) = pending_stored {
            pending.flush(bw, buf, false);
        }
        crate::anatomy_count!(blocks_emitted_dynamic);
        crate::anatomy_count!(dynamic_header_bits_total, header.header_bits());
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN as u64, 2);
        header.emit(bw);
        crate::anatomy_wall_time!(huffman_encode_ns, huffman_encode_calls, {
            emit_sequences(bw, buf, block_start, blk, litcode, offcode);
        });
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
    blk: &BlockView,
    statics: &StaticCodes,
    is_final: bool,
    pending_stored: Option<&mut StoredCoalescer>,
) {
    let (static_bits, stored_bits) =
        crate::anatomy_wall_time!(huffman_table_ns, huffman_table_calls, {
            let mut litlen_freqs = *blk.litlen_freqs;
            litlen_freqs[DEFLATE_END_OF_BLOCK] += 1;

            let static_bits = 3 + cost_from_freqs(
                &litlen_freqs,
                blk.offset_freqs,
                &statics.litcode,
                &statics.offcode,
            );
            let stored_bits = stored_block_bits(blk.block_length);
            (static_bits, stored_bits)
        });

    crate::block_cost_probe!(
        sink.block_length,
        stored_bits,
        static_bits,
        -1i64,
        if stored_bits <= static_bits { 's' } else { 'f' },
        's'
    );

    if stored_bits <= static_bits {
        // See the sibling `emit_block`'s comment: counted in
        // `write_stored_subblock`, not here.
        match pending_stored {
            // Coalescing caller: defer for a maximal sub-block grid (see
            // `StoredCoalescer`).
            Some(pending) => {
                pending.push(bw, buf, block_start, blk.block_length);
                if is_final {
                    pending.flush(bw, buf, true);
                }
            }
            None => super::emit_stored_block(
                bw,
                &buf[block_start..block_start + blk.block_length],
                is_final,
            ),
        }
    } else {
        // A pending stored span (coalescing callers only) precedes this
        // block in input order — write it before the first Huffman bit.
        if let Some(pending) = pending_stored {
            pending.flush(bw, buf, false);
        }
        crate::anatomy_count!(blocks_emitted_fixed);
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_STATIC_HUFFMAN as u64, 2);
        crate::anatomy_wall_time!(huffman_encode_ns, huffman_encode_calls, {
            emit_sequences(
                bw,
                buf,
                block_start,
                blk,
                &statics.litcode,
                &statics.offcode,
            );
        });
    }
}

/// Exact coded-data bit cost (including the EOB symbol) of a token stream whose
/// per-symbol histogram is `litlen_freqs` / `offset_freqs`, coded with the given
/// litlen/offset code. `litlen_freqs[DEFLATE_END_OF_BLOCK]` must already include
/// the one EOB symbol.
///
/// Returns shaped histograms to build the block's codes from when strictly cheaper.
fn shaped_freqs_if_smaller(
    litlen_freqs: &[u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: &[u32; DEFLATE_NUM_OFFSET_SYMS],
    shape: &mut super::huffman::ShapeScratch,
) -> Option<(
    [u32; DEFLATE_NUM_LITLEN_SYMS],
    [u32; DEFLATE_NUM_OFFSET_SYMS],
)> {
    use super::huffman::optimal::optimize_huffman_for_rle_into;

    let mut s_lit = [0usize; DEFLATE_NUM_LITLEN_SYMS];
    for (d, &s) in s_lit.iter_mut().zip(litlen_freqs.iter()) {
        *d = s as usize;
    }
    let mut s_off = [0usize; DEFLATE_NUM_OFFSET_SYMS];
    for (d, &s) in s_off.iter_mut().zip(offset_freqs.iter()) {
        *d = s as usize;
    }
    optimize_huffman_for_rle_into(&mut s_lit, &mut shape.rle_flags);
    optimize_huffman_for_rle_into(&mut s_off, &mut shape.rle_flags);

    let mut shaped_lit = [0u32; DEFLATE_NUM_LITLEN_SYMS];
    for (d, &s) in shaped_lit.iter_mut().zip(s_lit.iter()) {
        *d = s as u32;
    }
    let mut shaped_off = [0u32; DEFLATE_NUM_OFFSET_SYMS];
    for (d, &s) in shaped_off.iter_mut().zip(s_off.iter()) {
        *d = s as u32;
    }

    if shaped_lit == *litlen_freqs && shaped_off == *offset_freqs {
        return None;
    }

    make_huffman_code_into(
        &mut shape.cand_litcode,
        DEFLATE_NUM_LITLEN_SYMS,
        MAX_LITLEN_CODEWORD_LEN,
        &shaped_lit,
    );
    make_huffman_code_into(
        &mut shape.cand_offcode,
        DEFLATE_NUM_OFFSET_SYMS,
        MAX_OFFSET_CODEWORD_LEN,
        &shaped_off,
    );
    let cand_bits = {
        let h = build_dynamic_header(
            &shape.cand_litcode.lens,
            &shape.cand_offcode.lens,
            &mut shape.cand_header,
        );
        h.header_bits()
    } + cost_from_freqs(
        litlen_freqs,
        offset_freqs,
        &shape.cand_litcode,
        &shape.cand_offcode,
    );

    make_huffman_code_into(
        &mut shape.cand_litcode,
        DEFLATE_NUM_LITLEN_SYMS,
        MAX_LITLEN_CODEWORD_LEN,
        litlen_freqs,
    );
    make_huffman_code_into(
        &mut shape.cand_offcode,
        DEFLATE_NUM_OFFSET_SYMS,
        MAX_OFFSET_CODEWORD_LEN,
        offset_freqs,
    );
    let raw_bits = {
        let h = build_dynamic_header(
            &shape.cand_litcode.lens,
            &shape.cand_offcode.lens,
            &mut shape.raw_header,
        );
        h.header_bits()
    } + cost_from_freqs(
        litlen_freqs,
        offset_freqs,
        &shape.cand_litcode,
        &shape.cand_offcode,
    );

    if cand_bits < raw_bits {
        Some((shaped_lit, shaped_off))
    } else {
        None
    }
}

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
        // Hoisted length proofs (measured as 6 live per-iteration panic
        // guards in the release binary's `emit_sequences`, executed once per
        // symbol-loop iteration of every per-block table build): `lens`/
        // `codewords` are `Vec`s, so the compiler re-checked their runtime
        // length inside each loop. One fixed-size-array view per input
        // replaces all of them — every subsequent index is against a
        // compile-time-length array with a provably in-range index
        // (`b < 256`, `sym = 257 + slot <= 285 < 288` via `length_slot`'s
        // <= 28 postcondition, `slot < 30 <= 32`, `DEFLATE_END_OF_BLOCK ==
        // 256 < 288`). Safe code: `try_into` is ONE check per build, and
        // every `HuffmanCode` reaching here is built by `make_huffman_code*`
        // with exactly DEFLATE_NUM_LITLEN_SYMS / DEFLATE_NUM_OFFSET_SYMS
        // entries, so the expect never fires in practice.
        let lit_lens: &[u8; DEFLATE_NUM_LITLEN_SYMS] = litcode.lens[..DEFLATE_NUM_LITLEN_SYMS]
            .try_into()
            .expect("litcode.lens has DEFLATE_NUM_LITLEN_SYMS entries");
        let lit_cw: &[u32; DEFLATE_NUM_LITLEN_SYMS] = litcode.codewords[..DEFLATE_NUM_LITLEN_SYMS]
            .try_into()
            .expect("litcode.codewords has DEFLATE_NUM_LITLEN_SYMS entries");
        let off_lens: &[u8; DEFLATE_NUM_OFFSET_SYMS] = offcode.lens[..DEFLATE_NUM_OFFSET_SYMS]
            .try_into()
            .expect("offcode.lens has DEFLATE_NUM_OFFSET_SYMS entries");
        let off_cw: &[u32; DEFLATE_NUM_OFFSET_SYMS] = offcode.codewords[..DEFLATE_NUM_OFFSET_SYMS]
            .try_into()
            .expect("offcode.codewords has DEFLATE_NUM_OFFSET_SYMS entries");

        let mut lit = [0u32; NUM_LITERALS];
        for (b, e) in lit.iter_mut().enumerate() {
            *e = lit_cw[b] | ((lit_lens[b] as u32) << 24);
        }
        // MAX_LITLEN_CODEWORD_LEN (14) + max extra length bits (5) <= 24, so the
        // concatenation stays below the nbits byte (C's STATIC_ASSERT at :1642).
        let mut full_len = [0u32; DEFLATE_MAX_MATCH_LEN as usize + 1];
        for len in DEFLATE_MIN_MATCH_LEN..=DEFLATE_MAX_MATCH_LEN {
            let slot = length_slot(len) as usize;
            let sym = DEFLATE_FIRST_LEN_SYM + slot;
            let extra_bits = len - LENGTH_SLOT_BASE[slot];
            let litlen_len = lit_lens[sym] as u32;
            full_len[len as usize] = (lit_cw[sym] | (extra_bits << litlen_len))
                | (((lit_lens[sym] + LENGTH_EXTRA_BITS[slot]) as u32) << 24);
        }
        let mut off = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        for (slot, e) in off.iter_mut().enumerate().take(OFFSET_EXTRA_BITS.len()) {
            let cwlen = off_lens[slot] as u32;
            *e = off_cw[slot] | (cwlen << 16) | ((cwlen + OFFSET_EXTRA_BITS[slot] as u32) << 24);
        }
        let eob = lit_cw[DEFLATE_END_OF_BLOCK] | ((lit_lens[DEFLATE_END_OF_BLOCK] as u32) << 24);
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
    crate::anatomy_count!(emit_body_bits, (e >> 24) as u64);
    bw.add_bits_raw((e & 0x00FF_FFFF) as u64, e >> 24);
}

/// Emit a run of `litrunlen` literals starting at `buf[p]` in groups of 4 (one
/// whole-word flush per group), then the 1-3-literal tail. Returns the position
/// one past the run.
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
fn emit_sequences(
    bw: &mut BitWriter,
    buf: &[u8],
    block_start: usize,
    blk: &BlockView,
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
    bw.reserve(blk.block_length * 2 + 16);

    // `p` walks the input: each Seq's literals are exactly the input bytes
    // between the previous match's end and this match's start.
    let mut p = block_start;
    let written = blk.seqs;
    for seq in written {
        // SAFETY: every Seq was pushed with its literals + match inside the
        // block, so [p, p + litrunlen + length) stays within
        // `block_start + blk.block_length <= buf.len()`; reserve() above
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
            crate::anatomy_count!(emit_body_bits, (e >> 24) as u64);
            bw.add_bits_raw(((e & 0xFFFF) as u64) | (extra << cwlen), e >> 24);
            // SAFETY: reserve() guarantees 8 spare bytes for every flush.
            bw.flush_word_unchecked();

            p += length;
        }
    }

    // Trailing literals after the last match, then the end-of-block symbol.
    // SAFETY: the trailing run is the block's final `litrun` input bytes, ending
    // exactly at `block_start + blk.block_length <= buf.len()`; reserve()'s 16
    // slack bytes cover the EOB flush.
    unsafe {
        emit_literal_run(bw, buf, p, blk.litrun as usize, &tabs.lit);
        add_entry(bw, tabs.eob);
        bw.flush_word_unchecked();
    }
}

/// Approximate bit cost of storing `len` bytes as stored (BTYPE=00) sub-blocks.
/// Mirrors the estimate in [`super`], used only for the block-type decision.
fn stored_block_bits(len: usize) -> u64 {
    // `div_ceil`, NOT `(len / 65535) + 1`. The old form over-counted by one whole
    // sub-block whenever `len` is an exact multiple of 65,535 — charging 5 bytes that
    // `emit_stored_block` never writes, so the three-way stored/static/dynamic compare
    // could reject a STORED block that was in fact the cheapest. On incompressible
    // input, where stored is the right answer, that is a direct loss.
    //
    // The codebase already had the correct formula for the same quantity
    // (`estimate_output_cap` in `deflate/mod.rs` uses
    // `len.div_ceil(MAX_STORED_SUBBLOCK).max(1) * 5`) and this site disagreed with it.
    // Two formulas for one physical fact is how a cost model drifts from the emitter it
    // prices; `emit_stored_block` writes exactly `ceil(len / MAX_STORED_SUBBLOCK)`
    // sub-blocks, and this now says so.
    let subblocks = len.div_ceil(65535).max(1);
    (8 * (len + 5 * subblocks)) as u64
}

#[cfg(test)]
mod seq_store_bound_tests {
    use super::*;

    /// `push_seq` writes through reserved capacity with NO growth check, so
    /// `SEQ_STORE_CAPACITY` is a memory-safety bound, not a tuning constant.
    /// Pin the two facts it rests on: every block span a parser can choose is
    /// covered, and every sequence eats at least `DEFLATE_MIN_MATCH_LEN` bytes.
    /// Raising a block length without raising the store is a heap overflow, and
    /// the failure would be silent in release — so it fails here instead.
    #[test]
    fn capacity_covers_every_parser_block_span() {
        for (name, span) in [
            (
                "greedy/lazy/near_optimal",
                SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH,
            ),
            ("fast L1", fast::FAST_BLOCK_LENGTH),
            ("fast L0", fast::FAST0_BLOCK_LENGTH),
        ] {
            // Worst case: the block runs its full span, one final match
            // straddles the limit, and every match is the 3-byte minimum.
            let worst =
                (span + DEFLATE_MAX_MATCH_LEN as usize).div_ceil(DEFLATE_MIN_MATCH_LEN as usize);
            assert!(
                worst <= SEQ_STORE_CAPACITY,
                "{name}: a {span}-byte block can hold {worst} sequences, \
                 above SEQ_STORE_CAPACITY {SEQ_STORE_CAPACITY}",
            );
        }
    }

    /// The policy cap greedy and lazy stop at must sit inside the allocation.
    /// Both are consts, so this holds at compile time.
    const _: () = assert!(SEQ_STORE_LENGTH <= SEQ_STORE_CAPACITY);

    /// `Sink::new` must reserve the full bound up front — `push_seq` never grows.
    #[test]
    fn sink_reserves_the_full_bound() {
        let sink = Sink::new();
        assert!(sink.seqs.capacity() >= SEQ_STORE_CAPACITY);
        assert_eq!(
            sink.seqs.len(),
            0,
            "len stays 0; slots are written via spare capacity"
        );
    }
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
                &sink.view(),
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
            &sink.view(),
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
        emit_sequences(&mut fast, &buf, 0, &sink.view(), &litcode, &offcode);
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
        emit_sequences(&mut a, &buf, 0, &sink.view(), &statics.litcode, &statics.offcode);
        let mut b = BitWriter::new();
        emit_sequences(
            &mut b,
            &shifted,
            37,
            &sink.view(),
            &statics.litcode,
            &statics.offcode,
        );
        assert_eq!(a.finish(), b.finish());
    }
}

#[cfg(test)]
mod l1_bakeoff {
    //! A ONE-BLOCK, SUB-SECOND optimisation loop for the L1 matchfinder.
    //!
    //! WHY THIS EXISTS. The L1 deficit is the largest remaining SIZE class on the
    //! board (16 of 37 residual cells after #227, and the largest level in the
    //! 22-file census at 35 cells). `fulcrum why` established that it is
    //! ALGORITHMIC, not implementation — at `data.csv:L01:T01` we emit
    //! **741,183 literals against libdeflate's 256,099 (2.89x)** and find 4.4%
    //! fewer matches, while header bits are near-identical. Every other level
    //! reports POSITION COUNTS MATCH at delta 0.00%; L1 is the one that parses
    //! differently.
    //!
    //! Iterating on that through the board is hopeless: a census is hours and a
    //! wall run needs a quiet box. But match quality is DETERMINISTIC — it does
    //! not care about load, arch, or thread count — so one block through both
    //! matchfinders answers "did that change find more matches?" in under a
    //! second, on any machine, with no instrument between the code and the number.
    //!
    //! ⚠ WHAT THIS RATCHET GUARDS. `ht_fast` is **NOT ROUTED IN PRODUCTION** —
    //! its only call site in the whole tree is this test (`grep -rn 'ht_fast::run'
    //! src/`). So the `ht_fast` column is what we COULD ship, not what we do, and
    //! nothing measured here can change the shipped binary. The SHIPPED L1 is the
    //! `Fast` column, and it is +1.634% vs libdeflate — that gap IS the L1 class.
    //! The ratchet exists to keep the candidate honest until the routing lands
    //! (blocked on #227's `params_parallel`, since it must be gated T>1).
    //!
    //! `Strategy::Fast` (shipped) is igzip-class chainless SINGLE-PROBE.
    //! `ht_fast` is the libdeflate-class 2-ENTRY-BUCKET synthesis that also keeps
    //! our length-3 table. Both are already in the tree and both are
    //! `fulcrum verify`-clean; only the ROUTING was reverted (see the FALSIFY at
    //! the `Strategy::Fast` dispatch arm).
    //!
    //! HOW TO USE IT:
    //!     cargo test --release l1_bakeoff -- --nocapture
    //! Smaller `ht_fast` bytes = the 2-way bucket is finding matches the single
    //! probe misses. This is a SIZE proxy only — it says nothing about the wall,
    //! and the routing decision needs a frozen paired wall run either way.

    use super::*;

    /// Real corpus, never synthetic: a synthetic input once said "+1 byte" where
    /// the real corpus said "+2.02%".
    fn corpus_roots() -> Vec<std::path::PathBuf> {
        let mut v = vec![];
        if let Ok(h) = std::env::var("HOME") {
            v.push(std::path::PathBuf::from(&h).join("www/gzippy-bench/corpus"));
        }
        v.push(std::path::PathBuf::from("/root/gzippy-bench/corpus"));
        v
    }

    /// Compress ONE block with the shipped L1 path and with `ht_fast`, and
    /// return (fast_bytes, ht_bytes). Both get the identical padded buffer.
    fn bakeoff_one(block: &[u8]) -> (usize, usize) {
        let mut buf = Vec::with_capacity(block.len() + BUF_PAD);
        buf.extend_from_slice(block);
        buf.resize(block.len() + BUF_PAD, 0);
        let in_end = block.len();
        let statics = StaticCodes::get();

        let params = crate::compress::deflate::level::params(1);
        let mut a = BitWriter::new();
        compress(
            &buf,
            0,
            in_end,
            in_end,
            &params,
            true,
            HeaderBudget::Lean,
            &mut a,
        );
        let fast_bytes = a.finish().len();

        let mut b = BitWriter::new();
        ht_fast::run(
            &buf,
            0,
            in_end,
            &params,
            statics,
            &mut b,
            true,
            HeaderBudget::Lean,
        );
        let ht_bytes = b.finish().len();

        (fast_bytes, ht_bytes)
    }

    /// The FOUR rivals the board actually grades against, at L1.
    /// `-c` to stdout; thread-capable ones pinned to 1 so this is a T1 statement.
    const RIVALS: &[(&str, &[&str])] = &[
        ("libdeflate", &["-1", "-c"]),
        ("gzip", &["-1", "-c"]),
        ("pigz", &["-1", "-p", "1", "-c"]),
        ("igzip", &["-1", "-T", "1", "-c"]),
    ];

    /// Resolve a rival binary. igzip is usually NOT on PATH but IS built in the
    /// vendor tree — `scripts/campaign/lib.sh` checks exactly there first, and a
    /// test that silently omits a rival has not evaluated the goal (the shipped
    /// size census once measured three rivals and said nothing about the fourth).
    fn rival_bin(name: &str) -> String {
        if name == "igzip" {
            for c in [
                "vendor/isa-l/build/igzip",
                "/root/gzippy/vendor/isa-l/build/igzip",
            ] {
                if std::path::Path::new(c).is_file() {
                    return c.to_string();
                }
            }
        }
        match name {
            "libdeflate" => "libdeflate-gzip".into(),
            "gzip" => "gzip".into(),
            "pigz" => "pigz".into(),
            _ => "igzip".into(),
        }
    }

    /// One rival's L1 output on the same bytes — the DEFLATE payload (gzip frame
    /// minus its 10-byte header and 8-byte trailer) so it is comparable to the
    /// raw-block sizes. `None` if that rival is not on this box; the file is then
    /// reported without it rather than inventing a reference.
    fn rival_l1(name: &str, args: &[&str], block: &[u8]) -> Option<usize> {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let mut c = Command::new(rival_bin(name))
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        // TAKE the handle: dropping it closes the pipe and delivers EOF.
        // Borrowing hangs forever — that bug cost two stalled runs elsewhere.
        {
            let mut si = c.stdin.take()?;
            si.write_all(block).ok()?;
        }
        let out = c.wait_with_output().ok()?;
        if !out.status.success() || out.stdout.len() < 18 {
            return None;
        }
        Some(out.stdout.len() - 18)
    }

    /// THE RATCHET. One number per file: the worst `ht_fast - libdeflate` delta
    /// we are willing to accept, in bytes, on the first `BLOCK` bytes.
    /// **Negative means we are SMALLER than libdeflate and must stay that way.**
    ///
    /// This encodes the GOAL — non-negotiable #2, "at level N we beat their
    /// level N", per file, never a curve or a total — and nothing about the
    /// implementation. Any matchfinder, any hash, any bucket geometry is fair
    /// game; the test only asks that no file gets worse. When a change improves
    /// a file the test SAYS SO and prints the new number to paste in. Tighten
    /// freely; loosening one needs a reason in the commit message.
    ///
    /// Seeded 2026-08-01 from `ht_fast` as it stands. `Strategy::Fast` (shipped)
    /// is far worse on 7 of these 8 — that gap is the L1 class, and it is why
    /// this file exists.
    /// Re-seeded 2026-08-01 from the WORST-OF-FOUR-RIVALS measurement. The first
    /// seed used libdeflate only and was therefore wrong on `armexe.elf`, whose
    /// binding rival is **pigz** (-3863) not libdeflate (-4013) — the ratchet
    /// fired REGRESSION on its own seed data, which is the guard working.
    ///
    /// libdeflate is the binding rival on 7 of 8; gzip/pigz/igzip trail by
    /// 2,600-11,600 B at L1. But "usually libdeflate" is not "always", and the
    /// one exception is exactly the file the FALSIFY at `parse/mod.rs` warned
    /// about losing.
    const RATCHET: &[(&str, i64)] = &[
        ("armexe.elf", -3863),
        ("data.parquet", -984),
        ("data.csv", -210),
        ("engine.wasm", 61),
        ("minjs.min.js", 313),
        ("data.json", 449),
        ("aozora.txt", 550),
        ("dickens", 644),
    ];

    #[test]
    fn l1_bakeoff() {
        const BLOCK: usize = 256 * 1024;
        let Some(root) = corpus_roots().into_iter().find(|p| p.is_dir()) else {
            eprintln!("l1_bakeoff: no corpus dir — SKIPPED (this is a real-corpus test)");
            return;
        };

        println!();
        println!("L1 RATCHET — first {BLOCK} bytes, ALL FOUR RIVALS, deterministic");
        println!("  corpus: {}", root.display());
        println!();
        println!(
            "  {:<16} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}",
            "file", "ht_fast", "libdefl", "gzip", "pigz", "igzip", "WORST", "allowed"
        );

        let mut regressions: Vec<String> = vec![];
        let mut improvements: Vec<String> = vec![];
        let mut measured = 0usize;
        let (mut wins, mut ties, mut losses) = (0usize, 0usize, 0usize);

        for (f, allowed) in RATCHET {
            let Ok(data) = std::fs::read(root.join(f)) else {
                continue;
            };
            if data.len() < BLOCK {
                continue;
            }
            let block = &data[..BLOCK];
            let (_fa, ht) = bakeoff_one(block);

            let mut cells: Vec<String> = vec![];
            let mut worst: Option<i64> = None;
            let mut worst_name = "";
            for (name, args) in RIVALS {
                match rival_l1(name, args, block) {
                    Some(r) => {
                        let d = ht as i64 - r as i64;
                        cells.push(format!("{d:>+9}"));
                        if worst.is_none_or(|w| d > w) {
                            worst = Some(d);
                            worst_name = name;
                        }
                    }
                    None => cells.push(format!("{:>9}", "-")),
                }
            }
            let Some(w) = worst else { continue };
            measured += 1;
            match w.cmp(&0) {
                std::cmp::Ordering::Less => wins += 1,
                std::cmp::Ordering::Equal => ties += 1,
                std::cmp::Ordering::Greater => losses += 1,
            }
            let mark = if w > *allowed {
                regressions.push(format!(
                    "{f}: worst {w:+} (vs {worst_name}) exceeds allowed {allowed:+}"
                ));
                " <- REGRESSION"
            } else if w < *allowed {
                improvements.push(format!("(\"{f}\", {w}),"));
                " <- improved"
            } else {
                ""
            };
            println!(
                "  {f:<16} {ht:>9} {} {w:>+9} {allowed:>+8}{mark}",
                cells.join(" ")
            );
            println!("L1SWEEP file={f} ht={ht} worst={w} worst_rival={worst_name}");
        }

        if measured == 0 {
            eprintln!("l1_bakeoff: nothing measurable — SKIPPED");
            return;
        }
        println!();
        println!("  ACROSS ALL FOUR RIVALS: {wins} win, {ties} tie, {losses} LOSE");
        println!("L1SWEEP TOTAL win={wins} tie={ties} lose={losses}");
        if !improvements.is_empty() {
            println!();
            println!("  RATCHET DOWN — paste into RATCHET:");
            for i in &improvements {
                println!("      {i}");
            }
        }
        println!();

        assert!(
            regressions.is_empty(),
            "L1 ratchet regressed on {} file(s):\n  {}\n\
             The bar is the WORST rival per file — losing to ANY of the four fails \
             the cell. If a regression is intended, say why in the commit message.",
            regressions.len(),
            regressions.join("\n  ")
        );
    }
}
