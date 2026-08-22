//! PROPOSER RECALL of the block-split heuristic — a measurement instrument.
//!
//! ## The blocked question
//!
//! `block_split.rs` is a port of libdeflate's `should_end_block`. It summarises
//! every token into one of TEN observation buckets (8 literal classes from
//! `(lit >> 5) & 0x6 | (lit & 1)`, plus short-match `<9` and long-match `>=9`)
//! and ends a block when a sum of absolute differences between the running and
//! recent bucket distributions crosses a `200/512` cutoff. **It never computes a
//! bit.** The bucketing is blind by construction to which literal inside a
//! class, to match LENGTH beyond short/long, and to match DISTANCE entirely.
//!
//! PR #342 (`lever/cost-model-block-split`, commits `fc721c28` + `9c4cccb6`)
//! demoted that check from DECIDER to PROPOSER and adjudicated with exact bits;
//! it closed the online-retrospective form of that design and left one named,
//! unmeasured fact behind (`9c4cccb6`, point 3):
//!
//! > zopfli has NO proposer — it searches for the minimum. A confirm-only design
//! > is ceilinged by the proposer's recall, and the 10-bucket proxy is blind to
//! > match-length and distance shifts, which is what changes in binary data.
//!
//! **This module measures that ceiling.** It changes no shipped behaviour: it is
//! behind the OFF-by-default `split-recall` Cargo feature, and with the feature
//! off every hook in `parse/mod.rs` compiles to nothing.
//!
//! ## What it records (feature ON) and what it computes
//!
//! Two hooks in `parse/mod.rs`, both on the greedy/lazy/lazy2 path:
//!
//! * [`record_check`] — called from `continue_block` at EVERY position where the
//!   proposer is actually consulted (`ready_to_check_block` true and, at L3, the
//!   sparse-split gate open). It records the sink's token cursor and the
//!   proposer's verdict. It calls nothing on `BlockSplitStats`, so it cannot
//!   perturb the split decision.
//! * [`record_block`] — called from `emit_block` with the finished block's
//!   sequences. Literal BYTES are not in the sink (the emit reads them from the
//!   input), so the analysis re-reads them from the same buffer.
//!
//! Costing is EXACT, using the same primitives the shipped block-TYPE decision
//! uses: `make_huffman_code_into` + `build_dynamic_header().header_bits()` +
//! `cost_from_freqs` for dynamic, the RFC 1951 fixed codes for static, and
//! `stored_block_bits` for stored — i.e. zopfli's `EstimateCost` ==
//! `ZopfliCalculateBlockSizeAutoType`, which includes the stored/fixed/dynamic
//! choice (`vendor/zopfli/src/zopfli/blocksplitter.c`).
//!
//! ## The profitability criterion, and its limits
//!
//! zopfli prices a cut AT the cut and against real neighbours:
//! `SplitCost(i) = EstimateCost(start, i) + EstimateCost(i, end)`
//! (`blocksplitter.c:125`). Both perturbations here are one-step moves around
//! the SHIPPED boundary set:
//!
//! * a checkpoint `i` INSIDE a shipped block `[L, R)` where the proposer said NO
//!   is PROFITABLE iff `Auto(L,i) + Auto(i,R) < Auto(L,R)`. Profitable and not
//!   proposed = **FALSE NEGATIVE**.
//! * a shipped boundary `R` the proposer FIRED at, between blocks `[L,R)` and
//!   `[R,R2)`, is PROFITABLE iff `Auto(L,R) + Auto(R,R2) < Auto(L,R2)`, i.e. the
//!   cut beats merging the two blocks back together. Profitable = **TRUE
//!   POSITIVE**, otherwise **FALSE POSITIVE**.
//!
//! LIMITS, stated because a retrospective boundary price has already produced one
//! self-contradictory model on this campaign:
//!
//! 1. The TOKEN STREAM IS FIXED — it is the one the shipped splitter produced.
//!    Moving a boundary changes `recalculate_min_match_len` (per-block literal
//!    census) and therefore the parse itself. Costs here are on the observed
//!    tokens, not on a re-parse.
//! 2. The CONFUSION MATRIX prices each cut ONE AT A TIME with neighbours held
//!    fixed, so its `fn_bits_sum` column is an over-estimate — two missed cuts in
//!    one block are not additive. The MAGNITUDE column that matters is
//!    `fn_bits_greedy`: zopfli's own recursive minimum-split
//!    (`ZopfliBlockSplit`/`FindMinimumSplit`) re-run over the rejected positions,
//!    which takes the interactions into account. Measured, it is ~8x smaller than
//!    the independent sum and ~1.2x larger than one-cut-per-block.
//! 3. The candidate set for recall is the proposer's own decision domain — the
//!    positions where `should_end_block` is consulted. `--fine N` additionally
//!    scores every Nth token to price the 512-observation CADENCE itself, which
//!    is a property of `ready_to_check_block`, not of the SAD test.
//! 4. `Auto` omits the RLE-shaped-frequency dynamic candidate
//!    (`shaped_freqs_if_smaller`, T>1 only — `HeaderBudget::Lean` at T1 never
//!    shapes) but DOES include the package-merge exact candidate whenever the
//!    level enables it, so it matches what T1 actually costs.
//! 5. Merging two blocks (the TP/FP test) can exceed `SOFT_MAX_BLOCK_LENGTH` or
//!    the sequence-store cap, in which case the merge is not a move the parser
//!    could make. Those boundaries are counted separately as `forced`.
//!
//! ## Driver
//!
//! [`analyze_file`] runs ONE `parse::compress` pass over the whole padded buffer
//! with `level::params(level)` and `HeaderBudget::Lean` — deliberately NOT
//! `encode_deflate_bytes_to_vec`, whose T1 pick-min arms (L1/L2/L4 mmap, L5-L7
//! zlib) would interleave several parses' recordings. The resulting stream is
//! round-tripped through the decoder before any number is reported.
//!
//! ROUTE ASSERTION (measured 2026-08-22, `scratchpad/route2.sh`): at L3 and L9 —
//! neither of which is a pick-min level — this is EXACTLY the stream the shipped
//! CLI writes at `-p1`, from a file and from a pipe alike, byte for byte on
//! engine.wasm / dickens / data.json / dd79_bin6 / symbols.dwarf (delta 0 on all
//! 10 pairs, after subtracting the 18-byte gzip frame). The DEFAULT
//! multi-threaded CLI route is a different, stronger parse (`params_parallel` +
//! `HeaderBudget::Generous`) and is NOT what these numbers describe.
//!
//! MODEL SELF-CHECK: the sum of this module's price for the blocks as shipped is
//! within one byte per file of the bytes the encoder actually wrote (23 files,
//! L3: 215,420,157 predicted vs 215,420,167 actual over 6,079 blocks). The
//! residual is the final byte-align, and it is the receipt that the cost model
//! is the encoder's, not a proxy for it.

use super::*;
use std::cell::RefCell;

/// One position where the proposer was actually consulted.
#[derive(Clone, Copy, Debug)]
struct Check {
    /// Sequences pushed into the sink at this point.
    nseqs: usize,
    /// Literals pending since the last sequence.
    litrun: u32,
    /// Input bytes covered by the block so far (`sink.block_length`).
    block_length: usize,
    /// The proposer's verdict: true == "end the block here".
    fired: bool,
}

/// One emitted DEFLATE block, with everything needed to re-cost any sub-range.
struct Block {
    /// Absolute offset of the block's first byte in the working buffer.
    begin: usize,
    /// Input bytes in the block.
    length: usize,
    /// `(litrunlen, match_len, offset)` per sequence.
    seqs: Vec<(u32, u16, u16)>,
    /// Literals after the last sequence.
    final_litrun: u32,
    /// Every position the proposer was consulted at, in order.
    checks: Vec<Check>,
}

#[derive(Default)]
struct Recorder {
    on: bool,
    pending: Vec<Check>,
    blocks: Vec<Block>,
}

thread_local! {
    static REC: RefCell<Recorder> = RefCell::new(Recorder::default());
}

/// Hook: the proposer was consulted at this position. Called from
/// `continue_block` ONLY when the check actually ran.
#[inline]
pub(super) fn record_check(sink: &Sink, fired: bool) {
    REC.with(|r| {
        let mut r = r.borrow_mut();
        if !r.on {
            return;
        }
        r.pending.push(Check {
            nseqs: sink.nseqs,
            litrun: sink.litrun,
            block_length: sink.block_length,
            fired,
        });
    });
}

/// Hook: a block is about to be emitted. Called from `emit_block`.
#[inline]
pub(super) fn record_block(block_start: usize, sink: &Sink) {
    REC.with(|r| {
        let mut r = r.borrow_mut();
        if !r.on {
            return;
        }
        // SAFETY: identical to `emit_sequences` — `push_seq` has initialised
        // exactly `nseqs` slots in the reserved capacity of `seqs`.
        let written = unsafe { std::slice::from_raw_parts(sink.seqs.as_ptr(), sink.nseqs) };
        let seqs = written
            .iter()
            .map(|s| (s.litrunlen, s.length_and_slot & SEQ_LEN_MASK, s.offset))
            .collect();
        let checks = std::mem::take(&mut r.pending);
        r.blocks.push(Block {
            begin: block_start,
            length: sink.block_length,
            seqs,
            final_litrun: sink.litrun,
            checks,
        });
    });
}

// ---------------------------------------------------------------------------
// Exact cost of an arbitrary token range
// ---------------------------------------------------------------------------

/// Symbol frequencies for one candidate block.
#[derive(Clone)]
struct Freqs {
    lit: [u32; DEFLATE_NUM_LITLEN_SYMS],
    off: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

impl Freqs {
    fn zero() -> Self {
        Freqs {
            lit: [0; DEFLATE_NUM_LITLEN_SYMS],
            off: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }

    /// `self - other`, component-wise (prefix-difference over a token range).
    fn sub(&self, other: &Freqs) -> Freqs {
        let mut out = Freqs::zero();
        for i in 0..DEFLATE_NUM_LITLEN_SYMS {
            out.lit[i] = self.lit[i] - other.lit[i];
        }
        for i in 0..DEFLATE_NUM_OFFSET_SYMS {
            out.off[i] = self.off[i] - other.off[i];
        }
        out
    }

    fn add(&self, other: &Freqs) -> Freqs {
        let mut out = Freqs::zero();
        for i in 0..DEFLATE_NUM_LITLEN_SYMS {
            out.lit[i] = self.lit[i] + other.lit[i];
        }
        for i in 0..DEFLATE_NUM_OFFSET_SYMS {
            out.off[i] = self.off[i] + other.off[i];
        }
        out
    }
}

/// Reusable scratch so a per-candidate cost does not allocate.
struct CostScratch {
    litcode: HuffmanCode,
    offcode: HuffmanCode,
    alt_litcode: HuffmanCode,
    alt_offcode: HuffmanCode,
    header: HeaderScratch,
    alt_header: HeaderScratch,
    statics: &'static StaticCodes,
}

impl CostScratch {
    fn new() -> Self {
        let empty = || HuffmanCode {
            lens: Vec::new(),
            codewords: Vec::new(),
        };
        CostScratch {
            litcode: empty(),
            offcode: empty(),
            alt_litcode: empty(),
            alt_offcode: empty(),
            header: HeaderScratch::new(),
            alt_header: HeaderScratch::new(),
            statics: StaticCodes::get(),
        }
    }
}

/// EXACT bit cost of coding `f` over `nbytes` input bytes as ONE DEFLATE block,
/// taking the cheapest of stored / static / dynamic — the same three-way choice
/// `emit_block` makes, built from the same primitives.
fn auto_cost_bits(f: &Freqs, nbytes: usize, try_exact: bool, s: &mut CostScratch) -> u64 {
    // `emit_block` adds the end-of-block symbol before costing; so does this.
    let mut lit = f.lit;
    lit[DEFLATE_END_OF_BLOCK] += 1;

    make_huffman_code_into(
        &mut s.litcode,
        DEFLATE_NUM_LITLEN_SYMS,
        MAX_LITLEN_CODEWORD_LEN,
        &lit,
    );
    make_huffman_code_into(
        &mut s.offcode,
        DEFLATE_NUM_OFFSET_SYMS,
        MAX_OFFSET_CODEWORD_LEN,
        &f.off,
    );
    let mut dynamic_bits = 3
        + build_dynamic_header(&s.litcode.lens, &s.offcode.lens, &mut s.header).header_bits()
        + cost_from_freqs(&lit, &f.off, &s.litcode, &s.offcode);

    if try_exact {
        make_huffman_code_exact_into(
            &mut s.alt_litcode,
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &lit,
        );
        make_huffman_code_exact_into(
            &mut s.alt_offcode,
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &f.off,
        );
        let exact = 3
            + build_dynamic_header(&s.alt_litcode.lens, &s.alt_offcode.lens, &mut s.alt_header)
                .header_bits()
            + cost_from_freqs(&lit, &f.off, &s.alt_litcode, &s.alt_offcode);
        dynamic_bits = dynamic_bits.min(exact);
    }

    let static_bits = 3 + cost_from_freqs(&lit, &f.off, &s.statics.litcode, &s.statics.offcode);
    let stored_bits = stored_block_bits(nbytes);
    dynamic_bits.min(static_bits).min(stored_bits)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// The confusion matrix over candidate boundaries, plus its magnitude in bits.
#[derive(Default, Clone, Debug)]
pub struct Recall {
    pub file: String,
    pub level: u32,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub blocks: usize,
    /// Boundaries the proposer FIRED at that beat merging (true positives).
    pub tp: usize,
    /// Boundaries the proposer FIRED at that merging would beat (false positives).
    pub fp: usize,
    /// Fired boundaries where the merge is structurally impossible (block-length
    /// or sequence-store cap) — excluded from tp/fp.
    pub forced: usize,
    /// Consulted positions the proposer REJECTED that a cut would have paid at.
    pub fneg: usize,
    /// Consulted positions the proposer rejected that a cut would NOT pay at.
    pub tn: usize,
    /// Bits saved by the cuts the proposer took, vs merging them away.
    pub tp_bits: i64,
    /// Bits wasted by the cuts the proposer took that merging would beat.
    pub fp_bits: i64,
    /// Bits available at rejected-but-profitable positions, summed independently.
    pub fn_bits_sum: i64,
    /// Bits available taking only the single best missed cut per block.
    pub fn_bits_best: i64,
    /// Bits available REALISABLY: zopfli's recursive minimum-split run over the
    /// rejected positions only. This is the honest magnitude — `fn_bits_sum`
    /// over-states it (the cuts interact) and `fn_bits_best` under-states it
    /// (one cut per block).
    pub fn_bits_greedy: i64,
    /// Largest single missed saving, in bits.
    pub fn_bits_max: i64,
    /// Sum of this cost model's price for the blocks AS SHIPPED. Its agreement
    /// with `output_bytes * 8` is the model's end-to-end self-check: the same
    /// primitives, the same three-way type choice, so the only slack is the
    /// per-block byte alignment of the final flush.
    pub predicted_bits: i64,
    /// Optional finer grid (`--fine N`): candidates at every Nth sequence that
    /// the proposer was NEVER consulted at.
    pub fine_stride: usize,
    pub fine_candidates: usize,
    pub fine_profitable: usize,
    pub fine_bits_best: i64,
}

impl Recall {
    /// FALSE-NEGATIVE RATE over the proposer's own decision domain: of the
    /// consulted positions where a cut pays, the fraction the proposer rejected.
    pub fn fn_rate(&self) -> f64 {
        let profitable = self.tp + self.fneg;
        if profitable == 0 {
            0.0
        } else {
            self.fneg as f64 / profitable as f64
        }
    }

    /// Fraction of the total available saving that the proposer left behind:
    /// realisable-missed / (captured + realisable-missed). "Captured" is
    /// `tp_bits` — what the cuts it DID take are worth against merging them
    /// away — and "realisable-missed" is the recursive re-split of the rejected
    /// positions.
    pub fn missed_saving_fraction(&self) -> f64 {
        let total = self.tp_bits.max(0) + self.fn_bits_greedy;
        if total <= 0 {
            0.0
        } else {
            self.fn_bits_greedy as f64 / total as f64
        }
    }

    /// The same fraction with the INDEPENDENT-SUM numerator — an over-estimate,
    /// kept because it bounds the other side (see limit 2).
    pub fn missed_saving_fraction_indep(&self) -> f64 {
        let total = self.tp_bits.max(0) + self.fn_bits_sum;
        if total <= 0 {
            0.0
        } else {
            self.fn_bits_sum as f64 / total as f64
        }
    }

    /// The realisable missed saving (recursive re-split of the rejected
    /// positions) as a percentage of output bytes.
    pub fn missed_greedy_pct_of_output(&self) -> f64 {
        if self.output_bytes == 0 {
            0.0
        } else {
            100.0 * (self.fn_bits_greedy as f64 / 8.0) / self.output_bytes as f64
        }
    }

    pub fn header() -> String {
        format!(
            "{:<22} {:>2} {:>6} {:>5} {:>5} {:>6} {:>6} {:>6} {:>7} {:>10} {:>11} {:>10} {:>7} {:>9}",
            "file",
            "L",
            "blocks",
            "TP",
            "FP",
            "forced",
            "FN",
            "TN",
            "FN-rate",
            "TP bits",
            "FN bits sum",
            "FN greedy",
            "miss%",
            "greedy/out"
        )
    }

    pub fn row(&self) -> String {
        format!(
            "{:<22} {:>2} {:>6} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6.1}% {:>10} {:>11} {:>10} {:>6.2}% {:>8.3}%",
            self.file,
            self.level,
            self.blocks,
            self.tp,
            self.fp,
            self.forced,
            self.fneg,
            self.tn,
            100.0 * self.fn_rate(),
            self.tp_bits,
            self.fn_bits_sum,
            self.fn_bits_greedy,
            100.0 * self.missed_saving_fraction(),
            self.missed_greedy_pct_of_output(),
        )
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Compress `data` at `level` with ONE `parse::compress` pass and return the
/// proposer-recall confusion matrix. `fine` (0 = off) additionally scores every
/// `fine`-th sequence boundary, to price the 512-observation cadence itself.
///
/// Panics if the produced stream does not round-trip — a cost report taken from
/// a stream that does not decode is worthless (`sizecensus`'s rule).
pub fn analyze_file(name: &str, data: &[u8], level: u32, fine: usize) -> Recall {
    let params = crate::compress::deflate::level::params(level);
    assert!(
        !matches!(params.strategy, Strategy::Fast | Strategy::Fast0),
        "level {level} uses the fast parser, which never calls should_end_block"
    );
    assert!(
        !matches!(params.strategy, Strategy::NearOptimal),
        "level {level} uses near_optimal, whose split call site is not hooked"
    );

    let mut buf = Vec::with_capacity(data.len() + BUF_PAD);
    buf.extend_from_slice(data);
    buf.resize(data.len() + BUF_PAD, 0);

    REC.with(|r| {
        let mut r = r.borrow_mut();
        r.on = true;
        r.pending.clear();
        r.blocks.clear();
    });

    let mut bw = BitWriter::new();
    compress(
        &buf,
        0,
        data.len(),
        data.len(),
        &params,
        true,
        HeaderBudget::Lean,
        &mut bw,
    );
    let out = bw.finish();

    let blocks = REC.with(|r| {
        let mut r = r.borrow_mut();
        r.on = false;
        r.pending.clear();
        std::mem::take(&mut r.blocks)
    });

    // Round-trip gate: never quote a cost from a stream that does not decode.
    let round = crate::decompress::decompress_raw_bytes(&out).expect("raw DEFLATE round-trip");
    assert!(round == data, "round-trip mismatch on {name} L{level}");

    score(
        name,
        level,
        data.len(),
        out.len(),
        &buf,
        &blocks,
        &params,
        fine,
    )
}

/// Prefix frequencies at a set of cut cursors within one block.
struct Prefixes {
    /// `at[k]` = frequencies of everything strictly before cut `k`.
    at: Vec<Freqs>,
    /// Input bytes before cut `k`.
    bytes: Vec<usize>,
    total: Freqs,
}

/// Walk one block's sequences once, snapshotting prefix frequencies at each
/// `(nseqs, litrun)` cursor in `cuts` (which must be sorted).
fn prefixes(buf: &[u8], b: &Block, cuts: &[(usize, u32)]) -> Prefixes {
    let mut f = Freqs::zero();
    let mut pos = b.begin;
    let mut at: Vec<Freqs> = Vec::with_capacity(cuts.len());
    let mut bytes: Vec<usize> = Vec::with_capacity(cuts.len());
    let mut ci = 0usize;

    for i in 0..=b.seqs.len() {
        while ci < cuts.len() && cuts[ci].0 == i {
            let extra = cuts[ci].1 as usize;
            let mut g = f.clone();
            for &byte in &buf[pos..pos + extra] {
                g.lit[byte as usize] += 1;
            }
            at.push(g);
            bytes.push(pos + extra - b.begin);
            ci += 1;
        }
        if i == b.seqs.len() {
            break;
        }
        let (litrunlen, mlen, moff) = b.seqs[i];
        for &byte in &buf[pos..pos + litrunlen as usize] {
            f.lit[byte as usize] += 1;
        }
        pos += litrunlen as usize;
        f.lit[DEFLATE_FIRST_LEN_SYM + length_slot(mlen as u32) as usize] += 1;
        f.off[offset_slot(moff as u32) as usize] += 1;
        pos += mlen as usize;
    }
    assert_eq!(ci, cuts.len(), "unconsumed cut cursors");

    let mut total = f;
    for &byte in &buf[pos..pos + b.final_litrun as usize] {
        total.lit[byte as usize] += 1;
    }
    assert_eq!(
        pos + b.final_litrun as usize - b.begin,
        b.length,
        "sequence walk did not cover the block"
    );
    Prefixes { at, bytes, total }
}

/// What a scored candidate cut is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The proposer was consulted here and said NO. Carries `sink.block_length`
    /// as recorded AT the check, which the sequence walk must independently
    /// reproduce — that is what proves the cut cursor is aligned with the
    /// position the proposer actually judged.
    Rejected(usize),
    /// Grid-only: the proposer was never consulted here.
    Grid,
}

#[allow(clippy::too_many_arguments)]
fn score(
    name: &str,
    level: u32,
    input_bytes: usize,
    output_bytes: usize,
    buf: &[u8],
    blocks: &[Block],
    params: &LevelParams,
    fine: usize,
) -> Recall {
    let try_exact = params.try_exact_huffman;
    let mut s = CostScratch::new();
    let mut out = Recall {
        file: name.to_string(),
        level,
        input_bytes,
        output_bytes,
        blocks: blocks.len(),
        fine_stride: fine,
        ..Default::default()
    };

    let mut block_totals: Vec<Freqs> = Vec::with_capacity(blocks.len());
    let mut block_cost: Vec<u64> = Vec::with_capacity(blocks.len());

    for b in blocks {
        // Candidate set: every consulted position strictly inside the block
        // (the block-ENDING check is the boundary itself, not a candidate),
        // plus, when `fine > 0`, every `fine`-th sequence boundary.
        let mut cand: Vec<((usize, u32), Kind)> = b
            .checks
            .iter()
            .filter(|c| c.block_length > 0 && c.block_length < b.length)
            .map(|c| ((c.nseqs, c.litrun), Kind::Rejected(c.block_length)))
            .collect();
        if fine > 0 {
            // A grid point sharing a sequence cursor with a consulted position
            // is the SAME boundary (they differ only by a pending literal run),
            // so it is not evidence about the cadence. Drop those.
            let consulted: std::collections::HashSet<usize> =
                b.checks.iter().map(|c| c.nseqs).collect();
            let mut k = fine;
            while k < b.seqs.len() {
                if !consulted.contains(&k) {
                    cand.push(((k, 0), Kind::Grid));
                }
                k += fine;
            }
            // Stable sort by cursor keeps the `Rejected` entries (inserted
            // first) ahead of any `Grid` duplicate, and `dedup_by_key` keeps the
            // first of each run — so a consulted position is never re-labelled
            // as grid-only.
            cand.sort_by_key(|(c, _)| *c);
            cand.dedup_by_key(|(c, _)| *c);
        }
        let cuts: Vec<(usize, u32)> = cand.iter().map(|(c, _)| *c).collect();

        let p = prefixes(buf, b, &cuts);
        let whole = auto_cost_bits(&p.total, b.length, try_exact, &mut s);
        let mut best_missed = 0i64;
        let mut best_fine = 0i64;

        for (i, (_, kind)) in cand.iter().enumerate() {
            if p.bytes[i] == 0 || p.bytes[i] >= b.length {
                continue;
            }
            let right = p.total.sub(&p.at[i]);
            let lc = auto_cost_bits(&p.at[i], p.bytes[i], try_exact, &mut s);
            let rc = auto_cost_bits(&right, b.length - p.bytes[i], try_exact, &mut s);
            let saving = whole as i64 - (lc as i64 + rc as i64);
            match kind {
                Kind::Grid => {
                    out.fine_candidates += 1;
                    if saving > 0 {
                        out.fine_profitable += 1;
                        best_fine = best_fine.max(saving);
                    }
                }
                Kind::Rejected(recorded) => {
                    // The walk's byte count is derived from the sequences alone;
                    // `recorded` came from `sink.block_length` at the check. If
                    // they disagree, the candidate is not the position the
                    // proposer judged and every number below is meaningless.
                    assert_eq!(
                        p.bytes[i], *recorded,
                        "cut cursor misaligned with the recorded check position"
                    );
                    if saving > 0 {
                        out.fneg += 1;
                        out.fn_bits_sum += saving;
                        out.fn_bits_max = out.fn_bits_max.max(saving);
                        best_missed = best_missed.max(saving);
                    } else {
                        out.tn += 1;
                    }
                }
            }
        }
        out.fn_bits_best += best_missed;
        out.fine_bits_best += best_fine;

        // REALISABLE total: zopfli's own algorithm (`ZopfliBlockSplit` ->
        // `FindMinimumSplit` -> recurse, `blocksplitter.c:186-267`) run over the
        // positions the proposer REJECTED. Take the cheapest cut in the range,
        // apply it if it pays, recurse into both halves. This is the number
        // `fn_bits_sum` over-states and `fn_bits_best` under-states.
        let rejected: Vec<usize> = cand
            .iter()
            .enumerate()
            .filter(|(i, (_, k))| {
                matches!(k, Kind::Rejected(_)) && p.bytes[*i] > 0 && p.bytes[*i] < b.length
            })
            .map(|(i, _)| i)
            .collect();
        if !rejected.is_empty() {
            // Cut-point array: 0 = block start, 1..=m = rejected candidates,
            // m+1 = block end. Prefix frequencies and byte counts at each.
            let zero = Freqs::zero();
            let mut pf: Vec<&Freqs> = Vec::with_capacity(rejected.len() + 2);
            let mut pb: Vec<usize> = Vec::with_capacity(rejected.len() + 2);
            pf.push(&zero);
            pb.push(0);
            for &i in &rejected {
                pf.push(&p.at[i]);
                pb.push(p.bytes[i]);
            }
            pf.push(&p.total);
            pb.push(b.length);
            let split_total = greedy_split(&pf, &pb, 0, pf.len() - 1, try_exact, &mut s);
            out.fn_bits_greedy += (whole as i64 - split_total as i64).max(0);
        }

        block_totals.push(p.total);
        block_cost.push(whole);
    }

    out.predicted_bits = block_cost.iter().sum::<u64>() as i64;

    // Shipped boundaries: did the cut the proposer TOOK beat merging back?
    for i in 0..blocks.len().saturating_sub(1) {
        let (a, b) = (&blocks[i], &blocks[i + 1]);
        let splitter_chose = a
            .checks
            .last()
            .map(|c| c.fired && c.block_length == a.length)
            .unwrap_or(false);
        if !splitter_chose {
            continue;
        }
        // `continue_block`'s other two stop conditions make the merge
        // structurally unavailable, so it is not a decision the proposer made.
        if a.length + b.length > SOFT_MAX_BLOCK_LENGTH
            || a.seqs.len() + b.seqs.len() >= SEQ_STORE_LENGTH
        {
            out.forced += 1;
            continue;
        }
        let merged = block_totals[i].add(&block_totals[i + 1]);
        let mc = auto_cost_bits(&merged, a.length + b.length, try_exact, &mut s);
        let split = block_cost[i] as i64 + block_cost[i + 1] as i64;
        let saving = mc as i64 - split;
        if saving > 0 {
            out.tp += 1;
            out.tp_bits += saving;
        } else {
            out.fp += 1;
            out.fp_bits += -saving;
        }
    }

    out
}

/// zopfli's block splitter, restricted to the cut points the SAD proposer
/// REJECTED. Port of the shape of `ZopfliBlockSplit` +
/// `FindMinimumSplit`/`SplitCost` (`vendor/zopfli/src/zopfli/blocksplitter.c`):
/// price every candidate cut in `[lo, hi]` as `Cost(lo, i) + Cost(i, hi)`, take
/// the cheapest, keep it only if it beats `Cost(lo, hi)`, then recurse into both
/// halves. Returns the total bits for the range.
///
/// zopfli scans a coarse grid and then a local minimum; this scans EVERY
/// candidate in the range, which can only find a cut at least as good.
fn greedy_split(
    pf: &[&Freqs],
    pb: &[usize],
    lo: usize,
    hi: usize,
    try_exact: bool,
    s: &mut CostScratch,
) -> u64 {
    let cost = |a: usize, b: usize, s: &mut CostScratch| -> u64 {
        auto_cost_bits(&pf[b].sub(pf[a]), pb[b] - pb[a], try_exact, s)
    };
    let base = cost(lo, hi, s);
    if hi - lo < 2 {
        return base;
    }
    let mut best = base;
    let mut best_i = None;
    for i in (lo + 1)..hi {
        let c = cost(lo, i, s) + cost(i, hi, s);
        if c < best {
            best = c;
            best_i = Some(i);
        }
    }
    match best_i {
        None => base,
        Some(i) => {
            greedy_split(pf, pb, lo, i, try_exact, s) + greedy_split(pf, pb, i, hi, try_exact, s)
        }
    }
}
