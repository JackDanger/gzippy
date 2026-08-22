//! Block-PLACEMENT denominator probe — OFF by default (`--features split-probe`).
//!
//! ONE blocked question: how many bytes are available from better DEFLATE block
//! PLACEMENT alone, with the LZ77 parse held fixed? Every splitter lever
//! (cost-model split, cadence, latch) is blocked on that denominator, and no
//! instrument in the tree could produce it: `blockcensus`/`blockspans` see
//! boundaries but not tokens, `fingerprint_tool blocks` sees token COUNTS but
//! not the tokens themselves, so none of them can re-price a DIFFERENT partition
//! of the SAME token stream.
//!
//! This module records, per emitted block, the exact token list the shipped
//! parser handed to `emit_block` plus the exact bit cost of the candidate
//! `emit_block` picked. `examples/split_headroom.rs` then re-partitions that
//! token stream with zopfli's `ZopfliBlockSplitLZ77` search and prices both
//! partitions with [`cost_span`], which is a line-for-line re-statement of
//! `emit_block`'s three-way stored/static/dynamic decision.
//!
//! ZERO effect on the shipped binary: the record call sites are `#[cfg(feature =
//! "split-probe")]` and the feature is not in `default`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use super::super::huffman::{
    build_dynamic_header, make_huffman_code, make_huffman_code_exact_into, HeaderScratch,
    HuffmanCode,
};
use super::super::tables::{
    DEFLATE_END_OF_BLOCK, DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS,
    MAX_LITLEN_CODEWORD_LEN, MAX_OFFSET_CODEWORD_LEN,
};
use super::{cost_from_freqs, stored_block_bits, Sink, StaticCodes};

/// One `Seq` as recorded: literal-run length, then a match.
#[derive(Clone, Copy, Debug)]
pub struct ProbeSeq {
    pub litrunlen: u32,
    pub offset: u16,
    /// length | slot << 9, exactly as `Seq::length_and_slot`.
    pub length_and_slot: u16,
}

impl ProbeSeq {
    #[inline]
    pub fn length(&self) -> u32 {
        (self.length_and_slot & super::SEQ_LEN_MASK) as u32
    }
    #[inline]
    pub fn offset_slot(&self) -> usize {
        (self.length_and_slot >> super::SEQ_SLOT_SHIFT) as usize
    }
}

/// One block as the shipped splitter placed it.
#[derive(Clone, Debug)]
pub struct ProbeBlock {
    /// Which parser invocation produced it (pick-min runs several arms, and at
    /// T>1 several chunks, possibly on different threads).
    pub run: u64,
    /// Absolute offset of the block's first byte in the parser's input buffer.
    pub start: usize,
    /// Input bytes covered.
    pub len: usize,
    /// 0 = stored, 1 = static, 2 = dynamic — the candidate `emit_block` chose.
    pub btype: u8,
    /// Exact bits of the chosen candidate, including its 3-bit block header.
    pub bits: u64,
    pub is_final: bool,
    /// `params.try_exact_huffman` for this block (needed to re-cost identically).
    pub try_exact: bool,
    pub seqs: Vec<ProbeSeq>,
    /// Literals after the last `Seq` (the block's trailing literal run).
    pub trailing_lits: u32,
    /// The histogram `emit_block` actually costed (WITHOUT the end-of-block bump).
    /// Recorded rather than re-derived so a divergence between "the tokens" and
    /// "the frequencies the encoder priced" is visible instead of silently folded
    /// into a headroom number.
    pub litlen_freqs: [u32; DEFLATE_NUM_LITLEN_SYMS],
    pub offset_freqs: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

static ON: AtomicBool = AtomicBool::new(false);
static NEXT_RUN: AtomicU64 = AtomicU64::new(0);
static REC: Mutex<Vec<ProbeBlock>> = Mutex::new(Vec::new());

thread_local! {
    /// Run id for the block sequence this thread is currently emitting. A new id
    /// is taken on the first block after an `is_final` block (or on first use),
    /// so one id == one whole-stream parse pass on one thread.
    static RUN: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Start recording. Idempotent.
pub fn enable() {
    ON.store(true, Ordering::SeqCst);
}

/// Stop recording and drain everything captured so far.
pub fn take() -> Vec<ProbeBlock> {
    ON.store(false, Ordering::SeqCst);
    let mut g = REC.lock().unwrap();
    std::mem::take(&mut *g)
}

/// Called from `emit_block` with the three candidate costs already computed.
pub(super) fn record(
    block_start: usize,
    sink: &Sink,
    is_final: bool,
    try_exact: bool,
    dynamic_bits: u64,
    static_bits: u64,
    stored_bits: u64,
) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let (btype, bits) = if stored_bits <= dynamic_bits && stored_bits <= static_bits {
        (0u8, stored_bits)
    } else if static_bits <= dynamic_bits {
        (1u8, static_bits)
    } else {
        (2u8, dynamic_bits)
    };
    let run = RUN.with(|c| match c.get() {
        Some(r) => r,
        None => {
            let r = NEXT_RUN.fetch_add(1, Ordering::SeqCst);
            c.set(Some(r));
            r
        }
    });
    if is_final {
        RUN.with(|c| c.set(None));
    }
    // SAFETY: `seqs[..nseqs]` was written by `push_seq` before this block flushed
    // (see `Sink::seqs`: the store is written through spare capacity, so `len()`
    // stays 0 and `nseqs` is the true count).
    let seqs: Vec<ProbeSeq> = unsafe {
        std::slice::from_raw_parts(sink.seqs.as_ptr(), sink.nseqs)
            .iter()
            .map(|s| ProbeSeq {
                litrunlen: s.litrunlen,
                offset: s.offset,
                length_and_slot: s.length_and_slot,
            })
            .collect()
    };
    REC.lock().unwrap().push(ProbeBlock {
        run,
        start: block_start,
        len: sink.block_length,
        btype,
        bits,
        is_final,
        try_exact,
        seqs,
        trailing_lits: sink.litrun,
        litlen_freqs: sink.litlen_freqs,
        offset_freqs: sink.offset_freqs,
    });
}

/// The blocks of the run that tiles `[0, input_len)` for the fewest bits — the arm
/// whose bytes the encoder actually shipped when pick-min ran several.
///
/// **This is the instrument's fail-closed edge, and it fails closed on purpose.**
/// MEASURED (`examples/split_headroom --levels 1,2,4,5,6,7,8 engine.wasm`,
/// 2026-08-22, `cad4ae7e` + this branch): L1, L2 and L4 record TWO runs and NEITHER
/// tiles the input — `PROBE_DUMP=1` on `text` L1 shows one arm emitting
/// `start = 0, 65538, 131075 …` and another emitting `start = 32784, 98321,
/// 163857, 229395(final)`, i.e. at least one L1/L2/L4 arm reports `block_start` in
/// a base that is NOT an offset into the whole input. Those are exactly the three
/// mmap pick-min levels. Rather than let a wrong base silently corrupt a headroom
/// number, both callers refuse the cell: this returns `None` and the caller prints
/// `NO COVERING RUN RECORDED`. L3 and L8 record one run; L5-L7 and L9 record a
/// covering one. Anything measured at L1/L2/L4 needs the base resolved first.
pub fn covering_run(rec: Vec<ProbeBlock>, input_len: usize) -> Option<Vec<ProbeBlock>> {
    let mut runs: std::collections::BTreeMap<u64, Vec<ProbeBlock>> = Default::default();
    for b in rec {
        runs.entry(b.run).or_default().push(b);
    }
    let mut best: Option<(u64, Vec<ProbeBlock>)> = None;
    for (_, mut blocks) in runs {
        blocks.sort_by_key(|b| b.start);
        let mut expect = 0usize;
        let tiles = blocks.iter().all(|b| {
            let ok = b.start == expect;
            expect = b.start + b.len;
            ok
        }) && expect == input_len;
        if !tiles {
            continue;
        }
        let bits: u64 = blocks.iter().map(|b| b.bits).sum();
        if best.as_ref().map(|(t, _)| bits < *t).unwrap_or(true) {
            best = Some((bits, blocks));
        }
    }
    best.map(|(_, b)| b)
}

/// The three-way stored/static/dynamic decision of `emit_block`, for an
/// ARBITRARY span's frequencies. Returns `(bits, btype)`.
///
/// This is the whole instrument's validity: it must be the SAME arithmetic
/// `emit_block` runs, or a "headroom" number compares two different cost models.
/// It is checked two ways — `probe_cost_span_reprices_emit_block` in `parse/mod.rs`
/// re-prices real emitted blocks and asserts bits and btype match what `record`
/// captured, and `examples/split_headroom.rs` checks the summed bits against the
/// real gzip output size.
///
/// `litlen_freqs` must NOT include the end-of-block symbol; this adds it, as
/// `emit_block` does. Mirrors the T1 (`HeaderBudget::Lean`) path only: RLE
/// shaping is a T>1-only candidate and this probe runs at T1.
#[allow(dead_code)] // called from examples/tests; unused in the binary
pub fn cost_span(
    litlen_freqs: &[u32; DEFLATE_NUM_LITLEN_SYMS],
    offset_freqs: &[u32; DEFLATE_NUM_OFFSET_SYMS],
    block_len: usize,
    try_exact: bool,
) -> (u64, u8) {
    let mut litlen_freqs = *litlen_freqs;
    litlen_freqs[DEFLATE_END_OF_BLOCK] += 1;

    let litcode = make_huffman_code(
        DEFLATE_NUM_LITLEN_SYMS,
        MAX_LITLEN_CODEWORD_LEN,
        &litlen_freqs,
    );
    let offcode = make_huffman_code(
        DEFLATE_NUM_OFFSET_SYMS,
        MAX_OFFSET_CODEWORD_LEN,
        offset_freqs,
    );
    let mut scratch = HeaderScratch::new();
    let heur_bits = {
        let heur_header = build_dynamic_header(&litcode.lens, &offcode.lens, &mut scratch);
        3 + heur_header.header_bits()
            + cost_from_freqs(&litlen_freqs, offset_freqs, &litcode, &offcode)
    };

    let mut dynamic_bits = heur_bits;
    if try_exact {
        let mut alt_lit = HuffmanCode::default();
        let mut alt_off = HuffmanCode::default();
        make_huffman_code_exact_into(
            &mut alt_lit,
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &litlen_freqs,
        );
        make_huffman_code_exact_into(
            &mut alt_off,
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            offset_freqs,
        );
        let mut alt_scratch = HeaderScratch::new();
        let exact_bits = {
            let exact_header = build_dynamic_header(&alt_lit.lens, &alt_off.lens, &mut alt_scratch);
            3 + exact_header.header_bits()
                + cost_from_freqs(&litlen_freqs, offset_freqs, &alt_lit, &alt_off)
        };
        // Strict `<`: ties keep the heuristic, exactly as `emit_block` does.
        if exact_bits < dynamic_bits {
            dynamic_bits = exact_bits;
        }
    }

    let statics = StaticCodes::get();
    let static_bits = 3 + cost_from_freqs(
        &litlen_freqs,
        offset_freqs,
        &statics.litcode,
        &statics.offcode,
    );
    let stored_bits = stored_block_bits(block_len);

    if stored_bits <= dynamic_bits && stored_bits <= static_bits {
        (stored_bits, 0)
    } else if static_bits <= dynamic_bits {
        (static_bits, 1)
    } else {
        (dynamic_bits, 2)
    }
}
