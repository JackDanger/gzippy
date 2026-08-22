//! `split_headroom` — the block-PLACEMENT DENOMINATOR.
//!
//! ONE question, and it is a denominator, not a lever: **with the LZ77 parse held
//! fixed, how many bytes are available from placing the DEFLATE block boundaries
//! better?** Three named blocked classes depend on the answer and none of them can
//! be adjudicated without it:
//!   * `project_block_boundary_never_computes_a_bit` — the boundary decision uses a
//!     10-bucket drift proxy while `cost_from_freqs` prices a block exactly.
//!   * `project_cost_model_split_needs_real_histograms` — an entropy splitter was
//!     built and came out +487 KB WORSE; nobody knew the ceiling it was aiming at.
//!   * `project_block_cadence_is_a_symptom` — forcing gzip's block COUNT widened the
//!     gap, which says nothing about what the best PLACEMENT is worth.
//! If placement is worth 0.02% the whole class is dead; if it is worth 0.4% it is
//! the campaign. This tool produces that number and nothing else.
//!
//! METHOD
//!   1. Run the SHIPPED encoder (`encode_gzip_bytes_to_vec`) with `--features
//!      split-probe`, which records every emitted block's token list and the exact
//!      bit cost of the candidate `emit_block` chose. No shipped decision changes.
//!   2. Rebuild the flat LZ77 token stream from those records and re-price OUR
//!      partition with `parse::probe::cost_span` (the same three-way
//!      stored/static/dynamic arithmetic `emit_block` runs).
//!   3. Search a better partition of the SAME token stream with zopfli's
//!      `ZopfliBlockSplitLZ77` (`vendor/zopfli/src/zopfli/blocksplitter.c`:
//!      `FindMinimum` 9-way narrowing, `SplitCost(i) = EstimateCost(start,i) +
//!      EstimateCost(i,end)`, recurse on the largest splittable interval, stop when
//!      splitting stops paying). `EstimateCost` there is
//!      `ZopfliCalculateBlockSizeAutoType`, so best-of-three matches step 2.
//!      Cuts may only fall on token boundaries, so every searched partition is
//!      REACHABLE by the shipped emitter with the shipped tokens.
//!   4. Also run the same search SEEDED from our own cut set (add-cuts inside our
//!      blocks, then a remove-cut pass), so the reported best is <= ours by
//!      construction and cannot be beaten by our own splitter.
//!
//! VALIDATION (hard gate; the tool prints VALID/INVALID per cell and the caller must
//! not quote a number from an INVALID row): the summed bits of OUR partition must
//! reproduce the real gzip output size to within the byte-alignment slack. If it
//! does not, the cost model is not the encoder's and every derived number is void.
//!
//! WHAT THIS IS NOT: the true optimum. Greedy recursive bisection is a LOWER BOUND
//! on the achievable gain. `--dp` runs an exact O(C^2) dynamic program over a coarse
//! candidate grid on one file to bound how far the greedy search sits from optimal.
//! MEASURED, engine.wasm, grid refined 512 -> 256 -> 128 -> 64 tokens
//! (`--levels 3,9 --dp {512,256,128,64}`), gain over our partition in bytes:
//!
//!     grid   512    256    128     64   | greedy (free cut positions)
//!     L3    1649   1813   1965   2121   | 2069
//!     L9    1613   1807   2005   2161   | 2114
//!
//! The DP is still RISING at grid 64 and has already passed greedy by 51 B (L3) and
//! 46 B (L9) — so on this file the greedy number under-reports the optimum by at
//! least 2.5% / 2.2%, and the true free-position optimum is above even the grid-64
//! DP. Read every headroom figure this tool prints as a floor, not a ceiling.
//!
//!   cargo run --release --features split-probe --example split_headroom -- \
//!       --levels 3,9 CORPUS/dickens CORPUS/engine.wasm ...
//!   cargo run --release --features split-probe --example split_headroom -- \
//!       --levels 9 --dp 1024 CORPUS/engine.wasm

use gzippy::compress::deflate::parse::probe::{self, ProbeBlock};
use gzippy::compress::deflate::tables::{
    length_slot, offset_slot, DEFLATE_FIRST_LEN_SYM, DEFLATE_NUM_LITLEN_SYMS,
    DEFLATE_NUM_OFFSET_SYMS,
};

/// One LZ77 token. `len == 0` means "the literal byte at `pos`"; otherwise a match
/// of `len` bytes at distance `dist` starting at `pos`.
#[derive(Clone, Copy)]
struct Tok {
    pos: u32,
    len: u16,
    dist: u16,
}

/// Tokens per frequency checkpoint. Frequencies for an arbitrary span are
/// `cum(b) - cum(a)`, and `cum(x)` is a checkpoint plus a walk of at most this many
/// tokens — which is what makes ~1e5 span costings affordable. Pure accounting: the
/// frequencies are identical to a full walk (asserted by `check_freqs`).
const CKPT: usize = 4096;

struct Freqs {
    lit: [u32; DEFLATE_NUM_LITLEN_SYMS],
    off: [u32; DEFLATE_NUM_OFFSET_SYMS],
    bytes: u64,
}

impl Freqs {
    fn zero() -> Self {
        Freqs {
            lit: [0; DEFLATE_NUM_LITLEN_SYMS],
            off: [0; DEFLATE_NUM_OFFSET_SYMS],
            bytes: 0,
        }
    }
}

/// Checkpointed cumulative token histograms over one file's token stream.
struct Hist<'a> {
    toks: &'a [Tok],
    input: &'a [u8],
    ck: Vec<Freqs>,
}

impl<'a> Hist<'a> {
    fn build(toks: &'a [Tok], input: &'a [u8]) -> Self {
        let mut ck = Vec::with_capacity(toks.len() / CKPT + 2);
        let mut cur = Freqs::zero();
        ck.push(Freqs {
            lit: cur.lit,
            off: cur.off,
            bytes: cur.bytes,
        });
        for (i, t) in toks.iter().enumerate() {
            bump(&mut cur, t, input);
            if (i + 1) % CKPT == 0 {
                ck.push(Freqs {
                    lit: cur.lit,
                    off: cur.off,
                    bytes: cur.bytes,
                });
            }
        }
        Hist { toks, input, ck }
    }

    /// Cumulative frequencies over `toks[..x]`, accumulated into `out`.
    fn cum_into(&self, x: usize, out: &mut Freqs) {
        let c = x / CKPT;
        out.lit = self.ck[c].lit;
        out.off = self.ck[c].off;
        out.bytes = self.ck[c].bytes;
        for t in &self.toks[c * CKPT..x] {
            bump(out, t, self.input);
        }
    }

    /// Frequencies of `toks[a..b)` into `out` (differencing two cumulatives).
    fn span_into(&self, a: usize, b: usize, out: &mut Freqs, scratch: &mut Freqs) {
        self.cum_into(b, out);
        self.cum_into(a, scratch);
        for i in 0..DEFLATE_NUM_LITLEN_SYMS {
            out.lit[i] -= scratch.lit[i];
        }
        for i in 0..DEFLATE_NUM_OFFSET_SYMS {
            out.off[i] -= scratch.off[i];
        }
        out.bytes -= scratch.bytes;
    }
}

#[inline]
fn bump(f: &mut Freqs, t: &Tok, input: &[u8]) {
    if t.len == 0 {
        f.lit[input[t.pos as usize] as usize] += 1;
        f.bytes += 1;
    } else {
        f.lit[DEFLATE_FIRST_LEN_SYM + length_slot(t.len as u32) as usize] += 1;
        f.off[offset_slot(t.dist as u32) as usize] += 1;
        f.bytes += t.len as u64;
    }
}

/// Exact cost in bits of emitting `toks[a..b)` as ONE block, using the encoder's own
/// stored/static/dynamic decision.
struct Coster<'a> {
    hist: &'a Hist<'a>,
    try_exact: bool,
    evals: std::cell::Cell<u64>,
    a: std::cell::RefCell<(Freqs, Freqs)>,
}

impl<'a> Coster<'a> {
    fn new(hist: &'a Hist<'a>, try_exact: bool) -> Self {
        Coster {
            hist,
            try_exact,
            evals: std::cell::Cell::new(0),
            a: std::cell::RefCell::new((Freqs::zero(), Freqs::zero())),
        }
    }
    fn cost(&self, a: usize, b: usize) -> u64 {
        self.evals.set(self.evals.get() + 1);
        let mut cell = self.a.borrow_mut();
        let (out, scratch) = &mut *cell;
        self.hist.span_into(a, b, out, scratch);
        probe::cost_span(&out.lit, &out.off, out.bytes as usize, self.try_exact).0
    }
    fn total(&self, cuts: &[usize], n: usize) -> u64 {
        let mut t = 0u64;
        let mut prev = 0usize;
        for &c in cuts {
            t += self.cost(prev, c);
            prev = c;
        }
        t + self.cost(prev, n)
    }
}

// ── zopfli blocksplitter.c, on OUR tokens and OUR cost ───────────────────────

const FIND_MINIMUM_NUM: usize = 9;
const LARGE: u64 = u64::MAX / 4;

fn find_minimum<F: Fn(usize) -> u64>(f: F, start: usize, end: usize) -> (usize, u64) {
    if end - start < 1024 {
        let mut best = LARGE;
        let mut result = start;
        for i in start..end {
            let v = f(i);
            if v < best {
                best = v;
                result = i;
            }
        }
        return (result, best);
    }
    let (mut start, mut end) = (start, end);
    let mut p = [0usize; FIND_MINIMUM_NUM];
    let mut vp = [0u64; FIND_MINIMUM_NUM];
    let mut pos = start;
    let mut lastbest = LARGE;
    loop {
        if end - start <= FIND_MINIMUM_NUM {
            break;
        }
        for (i, slot) in p.iter_mut().enumerate() {
            *slot = start + (i + 1) * ((end - start) / (FIND_MINIMUM_NUM + 1));
        }
        for i in 0..FIND_MINIMUM_NUM {
            vp[i] = f(p[i]);
        }
        let mut besti = 0;
        let mut best = vp[0];
        for (i, &v) in vp.iter().enumerate().skip(1) {
            if v < best {
                best = v;
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

fn add_sorted(value: usize, out: &mut Vec<usize>) {
    let idx = out.partition_point(|&x| x < value);
    out.insert(idx, value);
}

/// `FindLargestSplittableBlock`: longest not-yet-`done` interval bounded by existing
/// split points.
fn find_largest_splittable(n: usize, done: &[u8], splits: &[usize]) -> Option<(usize, usize)> {
    let mut longest = 0usize;
    let mut found: Option<(usize, usize)> = None;
    let mut last = 0usize;
    for &s in splits.iter().chain(std::iter::once(&n)) {
        if done[last] == 0 && s - last > longest {
            longest = s - last;
            found = Some((last, s));
        }
        last = s;
    }
    found
}

/// `ZopfliBlockSplitLZ77` over `[lo, hi)` of the token stream, seeded with no cuts.
/// Returns interior cut points, sorted, in absolute token indices.
fn block_split(coster: &Coster, lo: usize, hi: usize, min_span: usize) -> Vec<usize> {
    let n = hi - lo;
    let mut splits: Vec<usize> = Vec::new();
    if n < 10 {
        return splits;
    }
    let mut done = vec![0u8; n + 1];
    let (mut lstart, mut lend) = (0usize, n);
    loop {
        let (llpos, splitcost) = find_minimum(
            |i| coster.cost(lo + lstart, lo + i) + coster.cost(lo + i, lo + lend),
            lstart + 1,
            lend,
        );
        let origcost = coster.cost(lo + lstart, lo + lend);
        if splitcost > origcost || llpos == lstart + 1 || llpos == lend {
            done[lstart] = 1;
        } else {
            add_sorted(llpos, &mut splits);
        }
        match find_largest_splittable(n, &done, &splits) {
            None => break,
            Some((s, e)) => {
                lstart = s;
                lend = e;
            }
        }
        if lend - lstart < min_span {
            break;
        }
    }
    splits.iter().map(|&s| s + lo).collect()
}

/// Remove any cut whose removal strictly reduces total cost (merge pass), repeated
/// to a fixed point. Cannot increase cost.
fn merge_pass(coster: &Coster, cuts: &mut Vec<usize>, n: usize) {
    loop {
        let mut removed = false;
        let mut i = 0;
        while i < cuts.len() {
            let lo = if i == 0 { 0 } else { cuts[i - 1] };
            let hi = if i + 1 < cuts.len() { cuts[i + 1] } else { n };
            let split = coster.cost(lo, cuts[i]) + coster.cost(cuts[i], hi);
            let merged = coster.cost(lo, hi);
            if merged < split {
                cuts.remove(i);
                removed = true;
            } else {
                i += 1;
            }
        }
        if !removed {
            break;
        }
    }
}

/// Exact DP over a coarse candidate grid: `cost[j] = min_{i<j} cost[i] + block(i,j)`.
/// Optimal ON THAT GRID; a bound on how much the greedy search leaves behind.
fn dp_optimal(coster: &Coster, cands: &[usize]) -> (u64, usize) {
    let c = cands.len();
    let mut best = vec![LARGE; c];
    let mut prev = vec![0usize; c];
    best[0] = 0;
    for j in 1..c {
        for i in 0..j {
            if best[i] >= LARGE {
                continue;
            }
            let v = best[i] + coster.cost(cands[i], cands[j]);
            if v < best[j] {
                best[j] = v;
                prev[j] = i;
            }
        }
    }
    let mut blocks = 0usize;
    let mut j = c - 1;
    while j != 0 {
        blocks += 1;
        j = prev[j];
    }
    (best[c - 1], blocks)
}

// ── driver ──────────────────────────────────────────────────────────────────

/// Reassemble the winning parser run's blocks into a flat token stream plus our
/// own cut points (token indices).
fn rebuild(run: &[ProbeBlock], input: &[u8]) -> Option<(Vec<Tok>, Vec<usize>)> {
    let mut toks: Vec<Tok> = Vec::new();
    let mut cuts: Vec<usize> = Vec::new();
    let mut expect = 0usize;
    for b in run {
        if b.start != expect {
            return None;
        }
        let mut p = b.start;
        for s in &b.seqs {
            for _ in 0..s.litrunlen {
                toks.push(Tok {
                    pos: p as u32,
                    len: 0,
                    dist: 0,
                });
                p += 1;
            }
            let l = s.length();
            toks.push(Tok {
                pos: p as u32,
                len: l as u16,
                dist: s.offset,
            });
            p += l as usize;
        }
        let end = b.start + b.len;
        assert_eq!(
            (end - p) as u32,
            b.trailing_lits,
            "trailing literal run disagrees with block length"
        );
        while p < end {
            toks.push(Tok {
                pos: p as u32,
                len: 0,
                dist: 0,
            });
            p += 1;
        }
        expect = end;
        cuts.push(toks.len());
    }
    if expect != input.len() {
        return None;
    }
    cuts.pop();
    Some((toks, cuts))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut levels: Vec<u32> = vec![3, 9];
    let mut files: Vec<String> = Vec::new();
    let mut dp_grid: usize = 0;
    let mut cap: usize = 32 << 20;
    let mut min_span: usize = 100;
    let mut emit_dir: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--emit-dir" => emit_dir = Some(args.next().unwrap()),
            "--levels" => {
                levels = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--dp" => dp_grid = args.next().unwrap().parse().unwrap(),
            "--cap" => cap = args.next().unwrap().parse::<usize>().unwrap() << 20,
            "--min-span" => min_span = args.next().unwrap().parse().unwrap(),
            other => files.push(other.to_string()),
        }
    }
    if files.is_empty() {
        eprintln!("usage: split_headroom [--levels 3,9] [--cap MiB] [--dp GRID] FILE ...");
        std::process::exit(2);
    }

    println!(
        "{:<22} {:>2} {:>11} {:>7} {:>14} {:>14} {:>7} {:>7} {:>12} {:>8} {:>8} {:>7} {:>7}",
        "file",
        "L",
        "in_bytes",
        "runs",
        "ours_bits",
        "best_bits",
        "ours_bl",
        "best_bl",
        "gain_bytes",
        "gain_%",
        "arm",
        "valid",
        "evals"
    );

    let mut tot_in = 0u64;
    let mut tot_gain = [0i64; 10];
    let mut tot_in_lvl = [0u64; 10];
    let mut tot_ours = [0u64; 10];

    for path in &files {
        let mut data = std::fs::read(path).expect("read input");
        let truncated = data.len() > cap;
        if truncated {
            data.truncate(cap);
        }
        let name = std::path::Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let name = if truncated {
            format!("{name}~{}M", cap >> 20)
        } else {
            name
        };
        tot_in += data.len() as u64;

        for &level in &levels {
            probe::enable();
            let out = gzippy::compress::deflate::encode_gzip_bytes_to_vec(&data, level);
            let rec = probe::take();

            // Roundtrip the real output before anything is derived from it. With
            // `--emit-dir` the same bytes are written out so the driver can repeat
            // the roundtrip through the REAL `gzip -dc` (an independent decoder) —
            // an in-process check alone shares our own bugs.
            let mut back: Vec<u8> = Vec::new();
            gzippy::decompress::decompress_bytes(&out, &mut back, 1).expect("roundtrip");
            assert_eq!(back, data, "{name} L{level}: roundtrip mismatch");
            if let Some(dir) = &emit_dir {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(format!("{dir}/{name}.L{level}.gz"), &out).unwrap();
                std::fs::write(format!("{dir}/{name}.L{level}.raw"), &data).unwrap();
            }

            // Pick-min runs several arms; the shipped bytes are the cheapest run that
            // TILES the input. `probe::covering_run` owns that rule (and documents
            // why L1/L2/L4 have none) so the example and the pinned test cannot drift
            // apart on it.
            let nruns = rec
                .iter()
                .map(|b| b.run)
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let Some(blocks) = probe::covering_run(rec, data.len()) else {
                println!("{name:<22} {level:>2}  NO COVERING RUN RECORDED (runs={nruns})");
                continue;
            };
            let ours_bits_recorded: u64 = blocks.iter().map(|b| b.bits).sum();

            // VALIDATION: our summed per-block bits must reproduce the real deflate
            // payload. gzip framing is 10 header + 8 trailer bytes; the payload is
            // padded to a byte boundary, so 0..7 bits of slack are expected.
            let deflate_bits = (out.len() as i64 - 18) * 8;
            let slack = deflate_bits - ours_bits_recorded as i64;
            let valid = (0..8).contains(&slack);

            let Some((toks, our_cuts)) = rebuild(&blocks, &data) else {
                println!("{name:<22} {level:>2}  REBUILD FAILED (non-contiguous blocks)");
                continue;
            };
            let try_exact = blocks[0].try_exact;
            let hist = Hist::build(&toks, &data);
            let coster = Coster::new(&hist, try_exact);
            let n = toks.len();

            // Instrument self-check: re-pricing our own partition with cost_span must
            // reproduce the bits emit_block recorded, block for block.
            let ours_bits = coster.total(&our_cuts, n);
            let self_ok = ours_bits == ours_bits_recorded;

            // (a) zopfli greedy from scratch.
            let greedy = block_split(&coster, 0, n, min_span);
            let greedy_bits = coster.total(&greedy, n);

            // (b) seeded from OUR cuts: split inside each of our blocks, then merge.
            let mut seeded: Vec<usize> = our_cuts.clone();
            {
                let bounds: Vec<usize> = std::iter::once(0)
                    .chain(our_cuts.iter().copied())
                    .chain(std::iter::once(n))
                    .collect();
                for w in bounds.windows(2) {
                    for c in block_split(&coster, w[0], w[1], min_span) {
                        seeded.push(c);
                    }
                }
                seeded.sort_unstable();
                seeded.dedup();
                merge_pass(&coster, &mut seeded, n);
            }
            let seeded_bits = coster.total(&seeded, n);

            // WHICH arm won matters more than the margin: "scratch" means throwing our
            // cut set away and re-searching beats keeping it, i.e. our boundaries are
            // not merely incomplete but actively in the wrong places.
            let (best_bits, best_blocks, arm) = if greedy_bits <= seeded_bits {
                (greedy_bits, greedy.len() + 1, "scratch")
            } else {
                (seeded_bits, seeded.len() + 1, "seeded")
            };
            let gain_bytes = (ours_bits as i64 - best_bits as i64) / 8;
            let gain_pct = (ours_bits as f64 - best_bits as f64) / ours_bits as f64 * 100.0;

            println!(
                "{:<22} {:>2} {:>11} {:>7} {:>14} {:>14} {:>7} {:>7} {:>12} {:>7.4}% {:>8} {:>7} {:>7}",
                name,
                level,
                data.len(),
                nruns,
                ours_bits,
                best_bits,
                our_cuts.len() + 1,
                best_blocks,
                gain_bytes,
                gain_pct,
                arm,
                if valid && self_ok {
                    "VALID".to_string()
                } else {
                    format!("BAD s{slack}/{self_ok}")
                },
                coster.evals.get()
            );

            if valid && self_ok {
                let li = (level as usize).min(9);
                tot_gain[li] += gain_bytes;
                tot_in_lvl[li] += data.len() as u64;
                tot_ours[li] += ours_bits / 8;
            }

            if dp_grid > 0 {
                let mut cands: Vec<usize> = (0..n).step_by(dp_grid).collect();
                if *cands.last().unwrap() != n {
                    cands.push(n);
                }
                let (dp_bits, dp_blocks) = dp_optimal(&coster, &cands);
                // Same grid, greedy-restricted: how much of the DP's win is grid, and
                // how much is the search?
                println!(
                    "    DP(grid={dp_grid}, cands={}) = {dp_bits} bits in {dp_blocks} blocks; \
                     vs ours {ours_bits} ({:+} B), vs greedy-best {best_bits} ({:+} B)",
                    cands.len(),
                    (ours_bits as i64 - dp_bits as i64) / 8,
                    (best_bits as i64 - dp_bits as i64) / 8,
                );
            }
        }
    }

    println!();
    for l in 0..10 {
        if tot_in_lvl[l] > 0 {
            println!(
                "L{l}: placement headroom {} B over {} input B ({:.4}% of input, {:.4}% of our output)",
                tot_gain[l],
                tot_in_lvl[l],
                tot_gain[l] as f64 / tot_in_lvl[l] as f64 * 100.0,
                tot_gain[l] as f64 / tot_ours[l] as f64 * 100.0,
            );
        }
    }
    let _ = tot_in;
}
