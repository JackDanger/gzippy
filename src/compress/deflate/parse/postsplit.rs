//! Post-parse block splitting — a port of zopfli's `blocksplitter.c` onto
//! gzippy's `Seq` token buffer.
//!
//! ## Why this exists, and what it is NOT
//!
//! The block BOUNDARY decision on every non-L1 level is libdeflate's
//! `do_end_block_check` ([`super::super::block_split`]): a SAD-of-probabilities
//! test over 10 coarse observation buckets against a 200/512 cutoff. It never
//! computes a bit, and it is a PROPOSER — it looks at the tail of the block it
//! is already inside and asks "does the recent data look unlike the block's
//! average?".
//!
//! Two attempts to adjudicate that proposer with an exact cost have now failed
//! (branch `lever/cost-model-block-split`, PR #342: a 10-bucket entropy model
//! at +487,154 B on L3, then exact bits at +10,675 B, and +38,616 B once made
//! coherent). Reading zopfli's splitter names why: **it prices the cut against
//! HISTORY**, i.e. it can only ever ask "would splitting back THERE have been
//! cheaper?", and a retrospective test can only fire when the tail is large,
//! which is exactly when the point priced and the point taken diverge.
//!
//! zopfli has none of those properties (`vendor/zopfli/src/zopfli/blocksplitter.c`):
//!
//! 1. `SplitCost(i) = EstimateCost(start, i) + EstimateCost(i, end)` (:125) —
//!    the cut is priced AT the position it is taken.
//! 2. `EstimateCost` is `ZopfliCalculateBlockSizeAutoType` (:110) — the split
//!    cost INCLUDES the stored/fixed/dynamic decision for each side, so a split
//!    that turns one side STORED is visible to the search. (`blockcensus` says
//!    we emit zero stored blocks on every corpus file; gzip emits one on sil40.)
//! 3. There is NO proposer. `FindMinimum` (:43) SEARCHES a range for the
//!    cheapest cut, `ZopfliBlockSplitLZ77` (:215) recurses into the largest
//!    remaining block, and `done[]` retires a block when splitting it stops
//!    paying.
//!
//! ## The structural obstacle, and the span that resolves it
//!
//! zopfli splits POST-PARSE, over a whole token stream. Our parser is online:
//! [`super::Sink`] is emitted and reset at every `SOFT_MAX_BLOCK_LENGTH`
//! (300,000 bytes) / `SEQ_STORE_LENGTH` (50,000 sequences) / drift-check
//! boundary. So the tokens for a span have to be retained before its cuts can
//! be decided.
//!
//! [`SPAN_BYTES`] is **1,000,000 — zopfli's own `ZOPFLI_MASTER_BLOCK_SIZE`**
//! (`vendor/zopfli/src/zopfli/util.h:60`), the unit zopfli itself LZ77s and
//! splits as one piece. It is also, independently, the largest span this can
//! buffer for free: `SEQ_STORE_CAPACITY` is already sized for
//! `fast::FAST0_BLOCK_LENGTH` = 1 MiB of tokens, so a 1,000,000-byte span needs
//! no larger sequence store than the one every parser already allocates.
//!
//! ## What is deliberately NOT changed
//!
//! The PARSE. Blocks are still parsed exactly as `main` parses them — same
//! `SOFT_MAX_BLOCK_LENGTH` cap, same sequence cap, same drift proposer, same
//! `recalculate_min_match_len` cadence anchored to the same block starts, same
//! per-block histogram resets. [`SpanSplitter::append`] concatenates the
//! resulting token streams (block *k*'s trailing literal run merges into block
//! *k+1*'s first sequence's literal run — the two runs are contiguous in the
//! input, so this is exact), and the splitter then re-decides the boundaries
//! over the whole span, ignoring the proposer's cuts entirely.
//!
//! That isolation is the point: any size delta this module produces is
//! attributable to the BOUNDARY decision alone, because the token stream fed to
//! the emitter is bit-identical to `main`'s.
//!
//! ## Deliberate divergences from the C
//!
//! * `splitcost >= origcost` retires the block, where zopfli uses `>`. An
//!   equal-cost split is the same number of bits with one more block; taking it
//!   can only cost wall.
//! * `FindLargestSplittableBlock` (:195) computes the last segment's end as
//!   `lz77size - 1`, which leaves the final token out of every split search.
//!   We use the true token count.

use super::super::bitstream::BitWriter;
use super::super::huffman::{
    build_dynamic_header, make_huffman_code_into, CodeScratch, HeaderScratch, HuffmanCode,
};
use super::super::tables::{
    length_slot, DEFLATE_END_OF_BLOCK, DEFLATE_FIRST_LEN_SYM, DEFLATE_NUM_LITLEN_SYMS,
    DEFLATE_NUM_OFFSET_SYMS, MAX_LITLEN_CODEWORD_LEN, MAX_OFFSET_CODEWORD_LEN,
};
use super::{
    cost_from_freqs, emit_block, stored_block_bits, BlockView, Seq, Sink, StaticCodes,
    SEQ_LEN_MASK, SEQ_SLOT_SHIFT,
};

/// `ZOPFLI_MASTER_BLOCK_SIZE` (`vendor/zopfli/src/zopfli/util.h:60`).
pub(crate) const SPAN_BYTES: usize = 1_000_000;

/// `maxblocks` (`ZopfliBlockSplitLZ77`, `blocksplitter.c:236`). Zopfli's own
/// UNLIMITED mode — its `options->blocksplittingmax` default of 15
/// (`util.c:34`) is a COMPUTE bound on a splitter that re-LZ77s and re-squeezes
/// every block 15 times, not a quality rule; the quality rule is the
/// `splitcost >= origcost` stop. MEASURED: at 15 the cap binds on
/// `data.parquet` (722 blocks on `main`, 331 under the cap = 15/span x 21
/// spans) and that cell is the ONLY size regression of the whole lever
/// (L9 +634 B, breaking a byte-exact tie with libdeflate).
const MAX_BLOCKS_PER_SPAN: usize = 0;

/// `NUM` in `FindMinimum` (`blocksplitter.c:60`, "Good value: 9").
const FIND_MINIMUM_NUM: usize = 9;

/// `FindMinimum` brute-forces ranges narrower than this (`blocksplitter.c:45`).
const FIND_MINIMUM_BRUTE_FORCE_MAX: usize = 1024;

/// `if (lz77->size < 10) return;` (`blocksplitter.c:225`) and the
/// `if (lend - lstart < 10) break;` stop (:263).
const MIN_SPLITTABLE_TOKENS: usize = 10;

/// Tokens between cumulative-histogram checkpoints.
///
/// `EstimateCost` is called ~4,000 times per span, and each call needs the
/// symbol histogram of an arbitrary token range. Re-walking the range is
/// O(bytes) and made the whole encode 3.8x slower than `main` at L9 T1
/// (dickens 12 MB: 0.334 s -> 1.244 s). Checkpointed cumulative histograms turn
/// a query into 320 subtractions plus at most `2 * CHECKPOINT_TOKENS` symbols
/// of edge walking. Cost is unchanged BY CONSTRUCTION — the same counts, summed
/// in a different order.
const CHECKPOINT_TOKENS: usize = 1024;

/// Symbols in one checkpoint row.
const CP_STRIDE: usize = DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS;

/// A cut point, resolved against the ORIGINAL (unmutated) sequence buffer:
/// `seq` sequences are wholly before the cut, plus `lits` literals taken from
/// the front of sequence `seq`'s literal run. `tok` is the same position as a
/// flat token index.
#[derive(Clone, Copy)]
struct Cut {
    seq: usize,
    lits: u32,
    tok: usize,
}

/// Accumulates the token streams of several parsed blocks and re-cuts them.
pub(super) struct SpanSplitter {
    /// Concatenated sequences of every block appended into this span.
    seqs: Vec<Seq>,
    /// Literals after the last sequence (merged into the next appended
    /// sequence's literal run, or the span's trailing run at flush).
    carry_lits: u32,
    /// Absolute offset in `buf` of the span's first byte.
    start: usize,
    /// Input bytes covered by the span so far.
    bytes: usize,

    // ---- prefix indexes, rebuilt per flush ----
    /// `tok_at[i]` = number of tokens before sequence `i`'s match symbol.
    /// `tok_at[nseqs]` = total token count.
    tok_at: Vec<u32>,
    /// `byte_at[i]` = input bytes before sequence `i`'s literal run.
    byte_at: Vec<u32>,
    /// Cumulative symbol histograms, one [`CP_STRIDE`]-wide row every
    /// [`CHECKPOINT_TOKENS`] tokens. Row `j` covers tokens `[0, j*C)`.
    cp: Vec<u32>,
    /// The cut each checkpoint row sits at.
    cp_cut: Vec<Cut>,

    // ---- cost-model scratch ----
    litcode: HuffmanCode,
    offcode: HuffmanCode,
    hscratch: HeaderScratch,
    lf: [u32; DEFLATE_NUM_LITLEN_SYMS],
    of: [u32; DEFLATE_NUM_OFFSET_SYMS],

    // ---- splitter scratch ----
    bounds: Vec<usize>,
    done: Vec<bool>,
    cuts: Vec<Cut>,
}

impl SpanSplitter {
    pub(super) fn new() -> Self {
        SpanSplitter {
            seqs: Vec::with_capacity(super::SEQ_STORE_CAPACITY),
            carry_lits: 0,
            start: 0,
            bytes: 0,
            tok_at: Vec::new(),
            byte_at: Vec::new(),
            cp: Vec::new(),
            cp_cut: Vec::new(),
            litcode: HuffmanCode::default(),
            offcode: HuffmanCode::default(),
            hscratch: HeaderScratch::new(),
            lf: [0; DEFLATE_NUM_LITLEN_SYMS],
            of: [0; DEFLATE_NUM_OFFSET_SYMS],
            bounds: Vec::new(),
            done: Vec::new(),
            cuts: Vec::new(),
        }
    }

    /// True when appending a block of `add` bytes would take the span past
    /// [`SPAN_BYTES`], i.e. the span must be flushed first.
    #[inline]
    pub(super) fn is_full_for(&self, add: usize) -> bool {
        self.bytes > 0 && self.bytes + add > SPAN_BYTES
    }

    /// Concatenate one parsed block's token stream onto the span.
    pub(super) fn append(&mut self, sink: &Sink, block_begin: usize) {
        if self.bytes == 0 {
            self.start = block_begin;
            self.carry_lits = 0;
            self.seqs.clear();
        }
        debug_assert_eq!(self.start + self.bytes, block_begin);
        let s = sink.seq_slice();
        if !s.is_empty() {
            let base = self.seqs.len();
            self.seqs.extend_from_slice(s);
            // The previous block's trailing literals immediately precede this
            // block's first literal run in the input, so the two runs are one.
            self.seqs[base].litrunlen += self.carry_lits;
            self.carry_lits = 0;
        }
        self.carry_lits += sink.litrun;
        self.bytes += sink.block_length;
    }

    /// Number of literals in sequence `i`'s run AS PARSED (`nseqs` = the
    /// trailing run).
    ///
    /// Derived from `byte_at`, NOT from `seqs[i].litrunlen`, because
    /// [`Self::flush`] shortens the `litrunlen` of a sequence whose literal run
    /// a cut lands inside — that mutation is what tells the emitter how many of
    /// the run's literals the NEXT block still owns, and reading it back here
    /// would double-count the ones the previous block already took. (It did,
    /// and the u32 underflow showed up as a 4 GB slice index.)
    #[inline]
    fn litrun_at(&self, i: usize) -> u32 {
        if i == self.seqs.len() {
            self.carry_lits
        } else {
            self.byte_at[i + 1]
                - self.byte_at[i]
                - (self.seqs[i].length_and_slot & SEQ_LEN_MASK) as u32
        }
    }

    fn build_prefix(&mut self) {
        self.tok_at.clear();
        self.byte_at.clear();
        self.tok_at.reserve(self.seqs.len() + 1);
        self.byte_at.reserve(self.seqs.len() + 1);
        let mut tok: u32 = 0;
        let mut byte: u32 = 0;
        for s in &self.seqs {
            self.byte_at.push(byte);
            tok += s.litrunlen;
            self.tok_at.push(tok);
            tok += 1;
            byte += s.litrunlen + (s.length_and_slot & SEQ_LEN_MASK) as u32;
        }
        self.byte_at.push(byte);
        self.tok_at.push(tok + self.carry_lits);
        debug_assert_eq!(byte as usize + self.carry_lits as usize, self.bytes);
    }

    /// Resolve a token index to the cut it names.
    fn locate(&self, t: usize) -> Cut {
        // Smallest `i` with `tok_at[i] >= t`.
        let i = self.tok_at.partition_point(|&x| (x as usize) < t);
        let run_start = self.tok_at[i] as usize - self.litrun_at(i) as usize;
        debug_assert!(t >= run_start);
        Cut {
            seq: i,
            lits: (t - run_start) as u32,
            tok: t,
        }
    }

    /// One pass over the span's tokens, snapshotting the running cumulative
    /// histogram every [`CHECKPOINT_TOKENS`] tokens.
    fn build_checkpoints(&mut self, buf: &[u8]) {
        self.cp.clear();
        self.cp_cut.clear();
        let n_tokens = *self.tok_at.last().unwrap() as usize;
        let nseq = self.seqs.len();
        let mut lf = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        let mut of = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        let mut t = 0usize;
        let mut p = self.start;
        let mut next = 0usize;

        for i in 0..=nseq {
            let run = self.litrun_at(i) as usize;
            for k in 0..run {
                if t == next {
                    self.cp.extend_from_slice(&lf);
                    self.cp.extend_from_slice(&of);
                    self.cp_cut.push(Cut {
                        seq: i,
                        lits: k as u32,
                        tok: t,
                    });
                    next += CHECKPOINT_TOKENS;
                }
                lf[buf[p] as usize] += 1;
                p += 1;
                t += 1;
            }
            if i == nseq {
                break;
            }
            if t == next {
                self.cp.extend_from_slice(&lf);
                self.cp.extend_from_slice(&of);
                self.cp_cut.push(Cut {
                    seq: i,
                    lits: run as u32,
                    tok: t,
                });
                next += CHECKPOINT_TOKENS;
            }
            let s = self.seqs[i];
            let length = (s.length_and_slot & SEQ_LEN_MASK) as u32;
            lf[DEFLATE_FIRST_LEN_SYM + length_slot(length) as usize] += 1;
            of[(s.length_and_slot >> SEQ_SLOT_SHIFT) as usize] += 1;
            p += length as usize;
            t += 1;
        }
        debug_assert_eq!(t, n_tokens);
        // A range's upper checkpoint index is `b.tok / C`, which reaches
        // `n_tokens / C` exactly when the span's token count is a multiple of
        // the stride. Close that row so every in-range lookup is present.
        if next == n_tokens {
            self.cp.extend_from_slice(&lf);
            self.cp.extend_from_slice(&of);
            self.cp_cut.push(Cut {
                seq: nseq,
                lits: self.carry_lits,
                tok: n_tokens,
            });
        }
    }

    #[inline]
    fn byte_of(&self, c: Cut) -> usize {
        self.byte_at[c.seq] as usize + c.lits as usize
    }

    /// Symbol histogram of the token range `[a, b)`, into `self.lf` / `self.of`.
    ///
    /// Whole checkpoint rows are differenced; only the two partial ends are
    /// walked. Identical counts to a full walk — see [`CHECKPOINT_TOKENS`].
    fn histogram(&mut self, buf: &[u8], a: Cut, b: Cut) {
        let j0 = a.tok.div_ceil(CHECKPOINT_TOKENS);
        let j1 = b.tok / CHECKPOINT_TOKENS;
        if j1 > j0 {
            let (o0, o1) = (j0 * CP_STRIDE, j1 * CP_STRIDE);
            for s in 0..DEFLATE_NUM_LITLEN_SYMS {
                self.lf[s] = self.cp[o1 + s] - self.cp[o0 + s];
            }
            for s in 0..DEFLATE_NUM_OFFSET_SYMS {
                self.of[s] = self.cp[o1 + DEFLATE_NUM_LITLEN_SYMS + s]
                    - self.cp[o0 + DEFLATE_NUM_LITLEN_SYMS + s];
            }
            let (c0, c1) = (self.cp_cut[j0], self.cp_cut[j1]);
            self.walk_add(buf, a, c0);
            self.walk_add(buf, c1, b);
        } else {
            self.lf.fill(0);
            self.of.fill(0);
            self.walk_add(buf, a, b);
        }
    }

    /// Add the symbols of the token range `[a, b)` to `self.lf` / `self.of`.
    fn walk_add(&mut self, buf: &[u8], a: Cut, b: Cut) {
        if a.tok == b.tok {
            return;
        }
        let mut p = self.start + self.byte_of(a);

        let count_lits = |lf: &mut [u32; DEFLATE_NUM_LITLEN_SYMS], from: usize, n: usize| {
            for &byte in &buf[from..from + n] {
                lf[byte as usize] += 1;
            }
        };

        if a.seq == b.seq {
            count_lits(&mut self.lf, p, (b.lits - a.lits) as usize);
            return;
        }

        // Tail of sequence `a.seq`'s literal run, then its match.
        let first_run = (self.litrun_at(a.seq) - a.lits) as usize;
        count_lits(&mut self.lf, p, first_run);
        p += first_run;
        for i in a.seq..b.seq {
            let s = self.seqs[i];
            if i != a.seq {
                let run = self.litrun_at(i) as usize;
                count_lits(&mut self.lf, p, run);
                p += run;
            }
            let length = (s.length_and_slot & SEQ_LEN_MASK) as u32;
            let os = (s.length_and_slot >> SEQ_SLOT_SHIFT) as usize;
            self.lf[DEFLATE_FIRST_LEN_SYM + length_slot(length) as usize] += 1;
            self.of[os] += 1;
            p += length as usize;
        }
        // Head of sequence `b.seq`'s literal run.
        count_lits(&mut self.lf, p, b.lits as usize);
    }

    /// `EstimateCost` = `ZopfliCalculateBlockSizeAutoType`: the EXACT emitted
    /// size in bits of `[a, b)` as one block, under the same
    /// stored / static / dynamic decision [`emit_block`] makes.
    fn cost(&mut self, buf: &[u8], statics: &StaticCodes, a: Cut, b: Cut) -> u64 {
        self.histogram(buf, a, b);
        self.lf[DEFLATE_END_OF_BLOCK] += 1;

        let SpanSplitter {
            litcode,
            offcode,
            hscratch,
            lf,
            of,
            ..
        } = self;

        make_huffman_code_into(
            litcode,
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            lf,
        );
        make_huffman_code_into(
            offcode,
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            of,
        );
        let dynamic_bits = {
            let header = build_dynamic_header(&litcode.lens, &offcode.lens, hscratch);
            3 + header.header_bits() + cost_from_freqs(lf, of, litcode, offcode)
        };
        let static_bits = 3 + cost_from_freqs(lf, of, &statics.litcode, &statics.offcode);
        let stored_bits = stored_block_bits(self.byte_of(b) - self.byte_of(a));
        dynamic_bits.min(static_bits).min(stored_bits)
    }

    /// `SplitCost` (`blocksplitter.c:125`).
    fn split_cost(
        &mut self,
        buf: &[u8],
        statics: &StaticCodes,
        i: usize,
        start: usize,
        end: usize,
    ) -> u64 {
        let (a, m, b) = (self.locate(start), self.locate(i), self.locate(end));
        self.cost(buf, statics, a, m) + self.cost(buf, statics, m, b)
    }

    /// `FindMinimum` (`blocksplitter.c:43`) over `[start, end)`.
    fn find_minimum(
        &mut self,
        buf: &[u8],
        statics: &StaticCodes,
        start: usize,
        end: usize,
        block: (usize, usize),
    ) -> (usize, u64) {
        if end - start < FIND_MINIMUM_BRUTE_FORCE_MAX {
            let mut best = u64::MAX;
            let mut result = start;
            for i in start..end {
                let v = self.split_cost(buf, statics, i, block.0, block.1);
                if v < best {
                    best = v;
                    result = i;
                }
            }
            return (result, best);
        }

        // Recursive 9-way narrowing.
        let (mut start, mut end) = (start, end);
        let mut p = [0usize; FIND_MINIMUM_NUM];
        let mut vp = [0u64; FIND_MINIMUM_NUM];
        let mut pos = start;
        let mut lastbest = u64::MAX;
        loop {
            if end - start <= FIND_MINIMUM_NUM {
                break;
            }
            for i in 0..FIND_MINIMUM_NUM {
                p[i] = start + (i + 1) * ((end - start) / (FIND_MINIMUM_NUM + 1));
                vp[i] = self.split_cost(buf, statics, p[i], block.0, block.1);
            }
            let mut besti = 0;
            let mut best = vp[0];
            for i in 1..FIND_MINIMUM_NUM {
                if vp[i] < best {
                    best = vp[i];
                    besti = i;
                }
            }
            if best > lastbest {
                break;
            }
            start = if besti == 0 { start } else { p[besti - 1] };
            end = if besti == FIND_MINIMUM_NUM - 1 {
                end
            } else {
                p[besti + 1]
            };
            pos = p[besti];
            lastbest = best;
        }
        (pos, lastbest)
    }

    /// `ZopfliBlockSplitLZ77` (`blocksplitter.c:215`). Fills `self.cuts` with
    /// the chosen boundaries (exclusive of the span's own two ends).
    fn split(&mut self, buf: &[u8], statics: &StaticCodes) {
        self.cuts.clear();
        let n_tokens = *self.tok_at.last().unwrap() as usize;
        if n_tokens < MIN_SPLITTABLE_TOKENS {
            return;
        }
        self.bounds.clear();
        self.bounds.push(0);
        self.bounds.push(n_tokens);
        self.done.clear();
        self.done.push(false);
        let mut numblocks = 1usize;

        loop {
            if MAX_BLOCKS_PER_SPAN > 0 && numblocks >= MAX_BLOCKS_PER_SPAN {
                break;
            }
            // `FindLargestSplittableBlock` (:195).
            let mut bi = usize::MAX;
            let mut longest = 0usize;
            for j in 0..self.done.len() {
                let span = self.bounds[j + 1] - self.bounds[j];
                if !self.done[j] && span > longest {
                    longest = span;
                    bi = j;
                }
            }
            if bi == usize::MAX || longest < MIN_SPLITTABLE_TOKENS {
                break;
            }
            let (lstart, lend) = (self.bounds[bi], self.bounds[bi + 1]);

            let (llpos, splitcost) =
                self.find_minimum(buf, statics, lstart + 1, lend, (lstart, lend));
            let (a, b) = (self.locate(lstart), self.locate(lend));
            let origcost = self.cost(buf, statics, a, b);

            if splitcost >= origcost || llpos == lstart + 1 || llpos >= lend {
                self.done[bi] = true;
            } else {
                self.bounds.insert(bi + 1, llpos);
                self.done.insert(bi + 1, false);
                numblocks += 1;
            }
        }

        for j in 1..self.bounds.len() - 1 {
            let c = self.locate(self.bounds[j]);
            self.cuts.push(c);
        }
    }

    /// Split the accumulated span and emit its blocks, then reset.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn flush(
        &mut self,
        bw: &mut BitWriter,
        buf: &[u8],
        statics: &StaticCodes,
        is_final: bool,
        header_scratch: &mut HeaderScratch,
        code_scratch: &mut CodeScratch,
        try_exact: bool,
    ) {
        if self.bytes == 0 {
            return;
        }
        self.build_prefix();
        self.build_checkpoints(buf);
        self.split(buf, statics);

        let end = Cut {
            seq: self.seqs.len(),
            lits: self.carry_lits,
            tok: *self.tok_at.last().unwrap() as usize,
        };
        let nblocks = self.cuts.len() + 1;

        // Emit left to right. Each block is a contiguous `seqs` slice plus a
        // trailing literal count; when a cut lands INSIDE a literal run the
        // sequence owning that run has its `litrunlen` reduced by the literals
        // the earlier block took, which is why `self.cuts` was resolved against
        // the prefix indexes BEFORE any mutation.
        let mut cur = Cut {
            seq: 0,
            lits: 0,
            tok: 0,
        };
        for k in 0..nblocks {
            let next = if k + 1 == nblocks { end } else { self.cuts[k] };
            let block_start = self.start + self.byte_of(cur);
            let block_length = self.byte_of(next) - self.byte_of(cur);
            let trailing = if next.seq == cur.seq {
                next.lits - cur.lits
            } else {
                next.lits
            };

            self.histogram(buf, cur, next);
            let seqs = &self.seqs[cur.seq..next.seq];
            let blk = BlockView {
                seqs,
                litrun: trailing,
                litlen_freqs: &self.lf,
                offset_freqs: &self.of,
                block_length,
            };
            emit_block(
                bw,
                buf,
                block_start,
                &blk,
                statics,
                is_final && k + 1 == nblocks,
                header_scratch,
                code_scratch,
                try_exact,
                None,
            );
            if next.seq < self.seqs.len() {
                self.seqs[next.seq].litrunlen -= trailing;
            }
            cur = next;
        }

        self.seqs.clear();
        self.carry_lits = 0;
        self.bytes = 0;
        self.start = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::deflate::tables::offset_slot;

    /// Deterministic xorshift, so a failure is reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// A span with many short literal runs, so cuts land INSIDE runs often.
    fn synthetic_span(nseq: usize) -> (SpanSplitter, Vec<u8>) {
        let mut sp = SpanSplitter::new();
        let mut buf = Vec::new();
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..nseq {
            let nlit = (rng.next() % 7) as u32;
            for _ in 0..nlit {
                buf.push((rng.next() % 251) as u8);
            }
            let length = 3 + (rng.next() % 60) as u32;
            let offset = 1 + (rng.next() % 32768) as u32;
            for _ in 0..length {
                buf.push((rng.next() % 251) as u8);
            }
            sp.seqs.push(Seq {
                litrunlen: nlit,
                offset: offset as u16,
                length_and_slot: (length as u16) | ((offset_slot(offset) as u16) << SEQ_SLOT_SHIFT),
            });
        }
        let tail = (rng.next() % 5000) as u32;
        for _ in 0..tail {
            buf.push((rng.next() % 251) as u8);
        }
        sp.carry_lits = tail;
        sp.bytes = buf.len();
        sp.start = 0;
        (sp, buf)
    }

    /// Independent full enumeration of a token range's symbol histogram.
    fn reference_histogram(
        sp: &SpanSplitter,
        buf: &[u8],
        a: Cut,
        b: Cut,
    ) -> (
        [u32; DEFLATE_NUM_LITLEN_SYMS],
        [u32; DEFLATE_NUM_OFFSET_SYMS],
    ) {
        let mut lf = [0u32; DEFLATE_NUM_LITLEN_SYMS];
        let mut of = [0u32; DEFLATE_NUM_OFFSET_SYMS];
        // Walk EVERY token of the span from scratch, counting only those whose
        // flat index falls in `[a.tok, b.tok)`.
        let mut t = 0usize;
        let mut p = 0usize;
        for i in 0..=sp.seqs.len() {
            let run = sp.litrun_at(i) as usize;
            for _ in 0..run {
                if t >= a.tok && t < b.tok {
                    lf[buf[p] as usize] += 1;
                }
                p += 1;
                t += 1;
            }
            if i == sp.seqs.len() {
                break;
            }
            let s = sp.seqs[i];
            let length = (s.length_and_slot & SEQ_LEN_MASK) as u32;
            if t >= a.tok && t < b.tok {
                lf[DEFLATE_FIRST_LEN_SYM + length_slot(length) as usize] += 1;
                of[(s.length_and_slot >> SEQ_SLOT_SHIFT) as usize] += 1;
            }
            p += length as usize;
            t += 1;
        }
        (lf, of)
    }

    /// The checkpointed histogram must agree with a full enumeration on EVERY
    /// range — including ranges shorter than one checkpoint stride, ranges that
    /// start and end inside the same literal run, and ranges whose ends fall
    /// exactly on a checkpoint.
    ///
    /// This is the invariant that broke first: `histogram` originally read
    /// `seqs[i].litrunlen`, which `flush` mutates as it emits, so the second
    /// block of every span counted the first block's literals again and the u32
    /// underflow surfaced as a 4 GB slice index.
    #[test]
    fn checkpointed_histogram_equals_full_enumeration() {
        let (mut sp, buf) = synthetic_span(6000);
        sp.build_prefix();
        sp.build_checkpoints(&buf);
        let n = *sp.tok_at.last().unwrap() as usize;
        assert!(n > 8 * CHECKPOINT_TOKENS, "span too small to exercise rows");

        let mut rng = Rng(0xdead_beef_0bad_f00d);
        let mut probes: Vec<(usize, usize)> = vec![
            (0, n),
            (0, CHECKPOINT_TOKENS),
            (CHECKPOINT_TOKENS, 2 * CHECKPOINT_TOKENS),
            (1, n - 1),
            (n - 3, n),
            (CHECKPOINT_TOKENS - 1, CHECKPOINT_TOKENS + 1),
        ];
        for _ in 0..300 {
            let x = (rng.next() as usize) % (n + 1);
            let y = (rng.next() as usize) % (n + 1);
            probes.push((x.min(y), x.max(y)));
        }

        for (lo, hi) in probes {
            let (a, b) = (sp.locate(lo), sp.locate(hi));
            let (rlf, rof) = reference_histogram(&sp, &buf, a, b);
            sp.histogram(&buf, a, b);
            assert_eq!(sp.lf, rlf, "litlen histogram differs on [{lo}, {hi})");
            assert_eq!(sp.of, rof, "offset histogram differs on [{lo}, {hi})");
        }
    }

    /// `locate` must round-trip every token index, and the byte offsets it
    /// yields must be monotone and cover the span exactly.
    #[test]
    fn cuts_tile_the_span() {
        let (mut sp, buf) = synthetic_span(2000);
        sp.build_prefix();
        let n = *sp.tok_at.last().unwrap() as usize;
        let mut prev_byte = 0usize;
        for t in 0..=n {
            let c = sp.locate(t);
            assert_eq!(c.tok, t);
            let byte = sp.byte_of(c);
            assert!(byte >= prev_byte, "byte offset went backwards at token {t}");
            prev_byte = byte;
        }
        assert_eq!(prev_byte, buf.len());
        assert_eq!(sp.byte_of(sp.locate(0)), 0);
    }
}
