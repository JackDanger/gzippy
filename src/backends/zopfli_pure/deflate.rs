//! DEFLATE encoder. Built across plan Steps 14 (`BitWriter`) and
//! 15 (tree emission, block emission, deflate driver).
//!
//! Port of the bit-emitting half of Google Zopfli deflate.c.

use super::blocksplitter::block_split;
use super::deflate_size::{
    build_rle, calculate_block_size, calculate_block_size_auto_type, get_dynamic_lengths_final,
    get_fixed_tree, CL_ORDER_PUB,
};
use super::lz77::{BlockState, LZ77Store};
use super::squeeze::{lz77_optimal, lz77_optimal_fixed};
use super::symbols::{
    dist_extra_bits, dist_extra_bits_value, dist_symbol, length_extra_bits,
    length_extra_bits_value, length_symbol, ZOPFLI_MASTER_BLOCK_SIZE, ZOPFLI_NUM_D, ZOPFLI_NUM_LL,
};
use super::tree::lengths_to_symbols;
use super::ZopfliOptions;

/// Bit-level writer that appends to a byte buffer. The C code uses a
/// `(out, outsize, bp)` triple where `bp` is the bit pointer in `[0, 7]`;
/// when `bp == 0` a fresh zero byte is appended before any bits land.
///
/// Note: a Step-27 prototype that wrote whole byte-fragments per
/// iteration showed zero measurable wall-clock improvement on the
/// `--ultra` workloads (the squeeze loop dominates, not bit emission),
/// so we kept the literal one-bit-per-iteration C port.
pub struct BitWriter<'a> {
    pub out: &'a mut Vec<u8>,
    pub bp: u8,
}

impl<'a> BitWriter<'a> {
    /// Creates a new writer wrapping `out`. `bp` is the bit pointer for
    /// the *current* tail byte: 0 means the next bit will allocate a new
    /// byte; 1..=7 means there is a half-filled tail byte already.
    pub fn new(out: &'a mut Vec<u8>, bp: u8) -> Self {
        debug_assert!(bp < 8);
        Self { out, bp }
    }

    /// Append a single bit (0 or 1).
    pub fn add_bit(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        if self.bp == 0 {
            self.out.push(0);
        }
        let n = self.out.len();
        self.out[n - 1] |= bit << self.bp;
        self.bp = (self.bp + 1) & 7;
    }

    /// Append `length` bits of `symbol`, LSB-first. Used for everything
    /// except canonical Huffman codes.
    pub fn add_bits(&mut self, symbol: u32, length: u32) {
        for i in 0..length {
            let bit = ((symbol >> i) & 1) as u8;
            if self.bp == 0 {
                self.out.push(0);
            }
            let n = self.out.len();
            self.out[n - 1] |= bit << self.bp;
            self.bp = (self.bp + 1) & 7;
        }
    }

    /// Append `length` bits of `symbol`, MSB-first. The DEFLATE spec
    /// requires this orientation for canonical Huffman codes.
    pub fn add_huffman_bits(&mut self, symbol: u32, length: u32) {
        for i in 0..length {
            let bit = ((symbol >> (length - i - 1)) & 1) as u8;
            if self.bp == 0 {
                self.out.push(0);
            }
            let n = self.out.len();
            self.out[n - 1] |= bit << self.bp;
            self.bp = (self.bp + 1) & 7;
        }
    }
}

// ── Step 15: tree emission ───────────────────────────────────────────────────

/// Emits the dynamic-tree encoding for a single (use_16, use_17, use_18)
/// flag triple. Mirrors the bit-emitting half of `EncodeTree`.
fn encode_tree_emit(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    use_16: bool,
    use_17: bool,
    use_18: bool,
    w: &mut BitWriter<'_>,
) {
    let mut rle: Vec<u32> = Vec::new();
    let mut rle_bits: Vec<u32> = Vec::new();
    let (clcounts, hclen, hlit, hdist) = build_rle(
        ll_lengths,
        d_lengths,
        use_16,
        use_17,
        use_18,
        Some(&mut rle),
        Some(&mut rle_bits),
    );

    use super::tree::calculate_bit_lengths;
    let mut clcl = [0u32; 19];
    calculate_bit_lengths(&clcounts, 7, &mut clcl);
    let mut clsymbols = [0u32; 19];
    lengths_to_symbols(&clcl, 7, &mut clsymbols);

    w.add_bits(hlit as u32, 5);
    w.add_bits(hdist as u32, 5);
    w.add_bits(hclen as u32, 4);

    for i in 0..hclen + 4 {
        w.add_bits(clcl[CL_ORDER_PUB[i]], 3);
    }

    for i in 0..rle.len() {
        let sym = rle[i] as usize;
        w.add_huffman_bits(clsymbols[sym], clcl[sym]);
        match sym {
            16 => w.add_bits(rle_bits[i], 2),
            17 => w.add_bits(rle_bits[i], 3),
            18 => w.add_bits(rle_bits[i], 7),
            _ => {}
        }
    }
}

/// Picks the smallest of the 8 (use_16, use_17, use_18) tree encodings and
/// emits it. Mirrors `AddDynamicTree`. Re-walks the 8 combinations because
/// `deflate_size::calculate_tree_size` returns only the size, not the
/// winning flag triple.
fn add_dynamic_tree(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    w: &mut BitWriter<'_>,
) {
    let mut best = 0u32;
    let mut best_size: usize = 0;
    for i in 0..8u32 {
        let s = encode_tree_size_local(
            ll_lengths,
            d_lengths,
            (i & 1) != 0,
            (i & 2) != 0,
            (i & 4) != 0,
        );
        if best_size == 0 || s < best_size {
            best_size = s;
            best = i;
        }
    }
    encode_tree_emit(
        ll_lengths,
        d_lengths,
        (best & 1) != 0,
        (best & 2) != 0,
        (best & 4) != 0,
        w,
    );
}

/// Mirrors `deflate_size`'s private `encode_tree_size`, computed via the
/// shared `build_rle` so the size-only and emit paths walk the same loop.
fn encode_tree_size_local(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    use_16: bool,
    use_17: bool,
    use_18: bool,
) -> usize {
    use super::tree::calculate_bit_lengths;
    let (clcounts, hclen, _hlit, _hdist) =
        build_rle(ll_lengths, d_lengths, use_16, use_17, use_18, None, None);

    let mut clcl = [0u32; 19];
    calculate_bit_lengths(&clcounts, 7, &mut clcl);

    let mut result: usize = 14;
    result += (hclen + 4) * 3;
    for i in 0..19 {
        result += clcl[i] as usize * clcounts[i];
    }
    result += clcounts[16] * 2;
    result += clcounts[17] * 3;
    result += clcounts[18] * 7;
    result
}

// ── Step 15: data emission ───────────────────────────────────────────────────

/// Emits the LZ77 stream as Huffman codes + extra bits. Does NOT emit the
/// end symbol (caller handles that). Mirrors `AddLZ77Data`.
#[allow(clippy::too_many_arguments)]
fn add_lz77_data(
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
    expected_data_size: usize,
    ll_symbols: &[u32; ZOPFLI_NUM_LL],
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_symbols: &[u32; ZOPFLI_NUM_D],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    w: &mut BitWriter<'_>,
) {
    let mut testlength: usize = 0;
    for i in lstart..lend {
        let dist = lz77.dists[i];
        let litlen = lz77.litlens[i];
        if dist == 0 {
            debug_assert!(litlen < 256);
            debug_assert!(ll_lengths[litlen as usize] > 0);
            w.add_huffman_bits(ll_symbols[litlen as usize], ll_lengths[litlen as usize]);
            testlength += 1;
        } else {
            let lls = length_symbol(litlen as i32) as usize;
            let ds = dist_symbol(dist as i32) as usize;
            debug_assert!((3..=288).contains(&litlen));
            debug_assert!(ll_lengths[lls] > 0);
            debug_assert!(d_lengths[ds] > 0);
            w.add_huffman_bits(ll_symbols[lls], ll_lengths[lls]);
            w.add_bits(
                length_extra_bits_value(litlen as i32) as u32,
                length_extra_bits(litlen as i32) as u32,
            );
            w.add_huffman_bits(d_symbols[ds], d_lengths[ds]);
            w.add_bits(
                dist_extra_bits_value(dist as i32) as u32,
                dist_extra_bits(dist as i32) as u32,
            );
            testlength += litlen as usize;
        }
    }
    debug_assert!(expected_data_size == 0 || testlength == expected_data_size);
}

/// Stored-block emitter. Splits into ≤65535-byte chunks, writes each with
/// `BTYPE = 00`, sets only the final block's "final" bit. Mirrors
/// `AddNonCompressedBlock`.
fn add_non_compressed_block(
    _options: &ZopfliOptions,
    final_: bool,
    in_: &[u8],
    instart: usize,
    inend: usize,
    w: &mut BitWriter<'_>,
) {
    let mut pos = instart;
    loop {
        let mut blocksize: u16 = 65535;
        if pos + blocksize as usize > inend {
            blocksize = (inend - pos) as u16;
        }
        let currentfinal = pos + blocksize as usize >= inend;
        let nlen: u16 = !blocksize;

        w.add_bit(if final_ && currentfinal { 1 } else { 0 });
        // BTYPE 00 (LSB-first: two zero bits).
        w.add_bit(0);
        w.add_bit(0);

        // Align to next byte boundary; any leftover bits in the current
        // byte are zero-padded (they were already zero-initialised).
        w.bp = 0;

        w.out.push((blocksize & 0xFF) as u8);
        w.out.push(((blocksize >> 8) & 0xFF) as u8);
        w.out.push((nlen & 0xFF) as u8);
        w.out.push(((nlen >> 8) & 0xFF) as u8);

        for i in 0..blocksize as usize {
            w.out.push(in_[pos + i]);
        }

        if currentfinal {
            break;
        }
        pos += blocksize as usize;
    }
}

/// Writes a single deflate block of the given `btype` (must be 0, 1, or 2).
/// Mirrors `AddLZ77Block`. For `btype == 0` falls through to
/// `add_non_compressed_block`.
#[allow(clippy::too_many_arguments)]
fn add_lz77_block(
    options: &ZopfliOptions,
    btype: i32,
    final_: bool,
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
    expected_data_size: usize,
    w: &mut BitWriter<'_>,
) {
    if btype == 0 {
        let length = lz77.byte_range(lstart, lend);
        let pos = if lstart == lend { 0 } else { lz77.pos[lstart] };
        let end = pos + length;
        add_non_compressed_block(options, final_, lz77.data, pos, end, w);
        return;
    }

    w.add_bit(if final_ { 1 } else { 0 });
    w.add_bit((btype & 1) as u8);
    w.add_bit(((btype & 2) >> 1) as u8);

    let mut ll_lengths = [0u32; ZOPFLI_NUM_LL];
    let mut d_lengths = [0u32; ZOPFLI_NUM_D];

    if btype == 1 {
        get_fixed_tree(&mut ll_lengths, &mut d_lengths);
    } else {
        debug_assert_eq!(btype, 2);
        let detect_tree_size = w.out.len();
        // FourWay table-shaping: block boundaries are final by this point
        // (see `get_dynamic_lengths_final`'s doc), so the extra smoothing
        // candidates can only shrink this block's emitted size.
        get_dynamic_lengths_final(lz77, lstart, lend, &mut ll_lengths, &mut d_lengths);
        add_dynamic_tree(&ll_lengths, &d_lengths, w);
        if options.verbose != 0 {
            eprintln!("treesize: {}", w.out.len() - detect_tree_size);
        }
    }

    let mut ll_symbols = [0u32; ZOPFLI_NUM_LL];
    let mut d_symbols = [0u32; ZOPFLI_NUM_D];
    lengths_to_symbols(&ll_lengths, 15, &mut ll_symbols);
    lengths_to_symbols(&d_lengths, 15, &mut d_symbols);

    let detect_block_size = w.out.len();
    add_lz77_data(
        lz77,
        lstart,
        lend,
        expected_data_size,
        &ll_symbols,
        &ll_lengths,
        &d_symbols,
        &d_lengths,
        w,
    );
    // End symbol.
    w.add_huffman_bits(ll_symbols[256], ll_lengths[256]);

    if options.verbose != 0 {
        let mut uncompressed_size = 0usize;
        for i in lstart..lend {
            uncompressed_size += if lz77.dists[i] == 0 {
                1
            } else {
                lz77.litlens[i] as usize
            };
        }
        let compressed_size = w.out.len() - detect_block_size;
        eprintln!(
            "compressed block size: {} ({}k) (unc: {})",
            compressed_size,
            compressed_size / 1024,
            uncompressed_size
        );
    }
}

/// Picks the cheapest btype (0/1/2) for the given block. For non-tiny inputs
/// or when fixed cost is close to dynamic, it also tries running
/// `lz77_optimal_fixed` over the byte range to see if a smaller fixed-tree
/// block is achievable. Mirrors `AddLZ77BlockAutoType`.
fn add_lz77_block_auto_type(
    options: &ZopfliOptions,
    final_: bool,
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
    expected_data_size: usize,
    w: &mut BitWriter<'_>,
) {
    let uncompressedcost = calculate_block_size(lz77, lstart, lend, 0);
    let mut fixedcost = calculate_block_size(lz77, lstart, lend, 1);
    let dyncost = calculate_block_size(lz77, lstart, lend, 2);

    let expensivefixed = lz77.size() < 1000 || fixedcost <= dyncost * 1.1;

    if lstart == lend {
        // Smallest empty block: fixed tree, end-symbol code 0000000.
        w.add_bits(if final_ { 1 } else { 0 }, 1);
        w.add_bits(1, 2); // btype 01
        w.add_bits(0, 7); // end symbol = 0000000
        return;
    }

    let mut fixedstore = LZ77Store::new(lz77.data);
    if expensivefixed {
        let instart = lz77.pos[lstart];
        let inend = instart + lz77.byte_range(lstart, lend);
        lz77_optimal_fixed(lz77.data, instart, inend, &mut fixedstore);
        fixedcost = calculate_block_size(&fixedstore, 0, fixedstore.size(), 1);
    }

    if uncompressedcost < fixedcost && uncompressedcost < dyncost {
        add_lz77_block(
            options,
            0,
            final_,
            lz77,
            lstart,
            lend,
            expected_data_size,
            w,
        );
    } else if fixedcost < dyncost {
        if expensivefixed {
            add_lz77_block(
                options,
                1,
                final_,
                &fixedstore,
                0,
                fixedstore.size(),
                expected_data_size,
                w,
            );
        } else {
            add_lz77_block(
                options,
                1,
                final_,
                lz77,
                lstart,
                lend,
                expected_data_size,
                w,
            );
        }
    } else {
        add_lz77_block(
            options,
            2,
            final_,
            lz77,
            lstart,
            lend,
            expected_data_size,
            w,
        );
    }
}

/// Re-squeeze one final block's byte range under BLOCK-LOCAL prices.
///
/// The per-greedy-chunk squeeze in `deflate_part` prices each chunk over a
/// coarse, often heterogeneous region; the recursive exact-cost re-split then
/// moves block boundaries to places that squeeze never priced against. Running
/// the full multi-seed `lz77_optimal` over the FINAL block's exact byte range
/// recovers block-local length/distance statistics, which is what the
/// optimal-parse frontier (ECT) achieves and what lets it keep medium near
/// matches (len 17-32 / dist 1-64) that the coarser prices under-took — the
/// measured parse residual. Returns the re-squeezed store only when it is
/// strictly cheaper by true auto-type cost, so a block can never regress.
fn resqueeze_block<'a>(
    options: &ZopfliOptions,
    in_: &'a [u8],
    lz77: &LZ77Store<'_>,
    start: usize,
    end: usize,
) -> Option<LZ77Store<'a>> {
    if start == end {
        return None;
    }
    let instart = lz77.pos[start];
    let inend = instart + lz77.byte_range(start, end);
    let mut s = BlockState::new(options, instart, inend, true);
    let mut cand = LZ77Store::new(in_);
    lz77_optimal(
        &mut s,
        in_,
        instart,
        inend,
        options.numiterations,
        &mut cand,
    );
    let cand_cost = calculate_block_size_auto_type(&cand, 0, cand.size());
    let orig_cost = calculate_block_size_auto_type(lz77, start, end);
    if cand_cost < orig_cost {
        Some(cand)
    } else {
        None
    }
}

/// True auto-type cost of a full block partition (sum of best-of-3 per-block
/// costs). Used to compare candidate (lz77, splitpoints) trajectories.
fn partition_true_cost(lz77: &LZ77Store<'_>, splitpoints: &[usize]) -> f64 {
    let n = splitpoints.len();
    let mut c = 0.0f64;
    for i in 0..=n {
        let start = if i == 0 { 0 } else { splitpoints[i - 1] };
        let end = if i == n { lz77.size() } else { splitpoints[i] };
        c += calculate_block_size_auto_type(lz77, start, end);
    }
    c
}

/// Re-split + re-squeeze one FIXED-STRATEGY trajectory for `rounds` rounds,
/// mirroring ECT's `twice` mode (ect-10009 = twice=1: the optimal parse is
/// fed back through block-splitting + squeeze one extra time). Each round is:
///   (a) recursive exact-cost re-split (optimal-parse frontier / ECT port),
///       adopted only when strictly cheaper by the true auto-type cost;
///   (b) per-final-block re-squeeze with block-local prices, keeping the
///       cheaper of (original tokens, re-squeezed) per block.
/// Both sub-steps only ever lower total cost, so a round can never regress
/// the trajectory it started from; a round that changes nothing ends the
/// loop early (converged).
///
/// `recursive_auto_type` selects the recursive splitter's per-candidate cost
/// model (see `blocksplitter::RecursiveSplitter`): `false` is the original
/// dynamic-only-cost search (ECT's `SplitCost` shape); `true` prices every
/// candidate with the TRUE best-of-{stored,fixed,dynamic} cost, matching the
/// reference optimal-parse frontier's `Splitter` (fulcrum
/// `src/ratio/squeeze.rs`, always calls `block_cost_exact`/`plan_block` —
/// never dynamic-only). `true` sees under-split cases `false` is blind to
/// (isolating a near-random sub-region into its own STORED block is far
/// cheaper than forcing it through a shared dynamic table) and over-split
/// cases where a split `false` judges marginally cheaper has header overhead
/// a true best-of-3 read recognizes as not worth it — but its narrowing
/// walks a DIFFERENT cost surface and can converge to a different (measured:
/// occasionally worse) local optimum than `false`'s. Callers run BOTH
/// strategies as independent whole trajectories from the same starting
/// point and keep whichever FINAL result is cheaper by true cost
/// (`partition_true_cost`) — never a regression vs either strategy alone.
#[allow(clippy::too_many_arguments)]
fn refine_split_and_squeeze<'a>(
    options: &ZopfliOptions,
    in_: &'a [u8],
    mut lz77: LZ77Store<'a>,
    mut splitpoints: Vec<usize>,
    recursive_auto_type: bool,
    rounds: usize,
) -> (LZ77Store<'a>, Vec<usize>) {
    if options.blocksplitting == 0 {
        return (lz77, splitpoints);
    }
    let recursive_maxblocks = match options.blocksplittingmax {
        0 => 0,
        m => (m as usize).max(64),
    };
    for _round in 0..rounds {
        let mut changed = false;

        // (a) recursive exact-cost re-split. Always runs (even for a single
        // squeezed chunk) so boundary placement is decided by exact cost,
        // not carried over from the coarse greedy chunk split. The per-master
        // cap matches the frontier's 64 blocks (honoring a larger override).
        let splitpoints2 = super::blocksplitter::block_split_lz77_recursive(
            &lz77,
            recursive_maxblocks,
            recursive_auto_type,
        );

        if splitpoints2 != splitpoints
            && partition_true_cost(&lz77, &splitpoints2) < partition_true_cost(&lz77, &splitpoints)
        {
            splitpoints = splitpoints2;
            changed = true;
        }

        // (b) per-final-block re-squeeze with block-local prices. See
        // `resqueeze_block`: the recursive re-split lands boundaries the
        // coarse per-chunk squeeze never priced against, so a final block's
        // tokens can be under-matched relative to its own local statistics.
        // Re-squeeze every final block over its exact byte range and keep
        // the cheaper of (original tokens, re-squeezed) — no block regresses.
        let np = splitpoints.len();
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(np + 1);
        for i in 0..=np {
            let start = if i == 0 { 0 } else { splitpoints[i - 1] };
            let end = if i == np { lz77.size() } else { splitpoints[i] };
            ranges.push((start, end));
        }
        let winners: Vec<Option<LZ77Store<'_>>> = if options.thread_budget == 1 {
            ranges
                .iter()
                .map(|&(s, e)| resqueeze_block(options, in_, &lz77, s, e))
                .collect()
        } else {
            let lz = &lz77;
            std::thread::scope(|scope| {
                let handles: Vec<_> = ranges
                    .iter()
                    .map(|&(s, e)| scope.spawn(move || resqueeze_block(options, in_, lz, s, e)))
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        if winners.iter().any(|w| w.is_some()) {
            let mut refined = LZ77Store::new(in_);
            let mut new_splits: Vec<usize> = Vec::new();
            for (i, &(start, end)) in ranges.iter().enumerate() {
                match &winners[i] {
                    Some(cand) => refined.append_from(cand),
                    None => {
                        for k in start..end {
                            refined.store_lit_len_dist(lz77.litlens[k], lz77.dists[k], lz77.pos[k]);
                        }
                    }
                }
                if i < np {
                    new_splits.push(refined.size());
                }
            }
            lz77 = refined;
            splitpoints = new_splits;
            changed = true;
        }

        if !changed {
            break;
        }
    }
    (lz77, splitpoints)
}

// ── Step 15: deflate driver ──────────────────────────────────────────────────

/// Compress `in_[instart..inend]` into one deflate sub-stream. `bp` is the
/// initial bit pointer; the new bit pointer is returned. For `btype == 2`
/// this can produce multiple deflate blocks via the splitter.
#[allow(clippy::too_many_arguments)]
pub fn deflate_part(
    options: &ZopfliOptions,
    btype: i32,
    final_: bool,
    in_: &[u8],
    instart: usize,
    inend: usize,
    bp: u8,
    out: &mut Vec<u8>,
) -> u8 {
    let mut w = BitWriter::new(out, bp);

    if btype == 0 {
        add_non_compressed_block(options, final_, in_, instart, inend, &mut w);
        return w.bp;
    }

    if btype == 1 {
        let mut store = LZ77Store::new(in_);
        lz77_optimal_fixed(in_, instart, inend, &mut store);
        let size = store.size();
        add_lz77_block(options, btype, final_, &store, 0, size, 0, &mut w);
        return w.bp;
    }

    // btype == 2
    let mut splitpoints_uncompressed: Vec<usize> = Vec::new();
    let mut splitpoints: Vec<usize> = Vec::new();
    let mut lz77 = LZ77Store::new(in_);

    if options.blocksplitting != 0 {
        splitpoints_uncompressed = block_split(
            options,
            in_,
            instart,
            inend,
            options.blocksplittingmax as usize,
        );
    }

    let npoints = splitpoints_uncompressed.len();

    // Step 29: parallel block evaluation, gated on `thread_budget` so it
    // doesn't oversubscribe gzippy's outer pool (see plan.md note). Each
    // lz77_optimal call on [start..end] is independent (own BlockState,
    // LongestMatchCache, LZ77Store; in_ is read-only), so the parallel
    // path is bit-identical to the serial one.
    //
    //   thread_budget == 1 → serial in-thread (right thing when an outer
    //                        pool is already running one deflate_part per
    //                        CPU, e.g. ZopfliGzEncoder::compress_parallel).
    //   thread_budget != 1 → spawn one thread per chunk via
    //                        std::thread::scope (right thing when the
    //                        caller is single-threaded, e.g.
    //                        ZopfliGzEncoder::compress_single).
    let mut chunks: Vec<(usize, usize)> = Vec::with_capacity(npoints + 1);
    for i in 0..=npoints {
        let start = if i == 0 {
            instart
        } else {
            splitpoints_uncompressed[i - 1]
        };
        let end = if i == npoints {
            inend
        } else {
            splitpoints_uncompressed[i]
        };
        chunks.push((start, end));
    }

    let stores: Vec<LZ77Store<'_>> = if options.thread_budget == 1 {
        chunks
            .iter()
            .map(|&(start, end)| {
                let mut s = BlockState::new(options, start, end, true);
                let mut store = LZ77Store::new(in_);
                lz77_optimal(&mut s, in_, start, end, options.numiterations, &mut store);
                store
            })
            .collect()
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunks
                .iter()
                .map(|&(start, end)| {
                    scope.spawn(move || {
                        let mut s = BlockState::new(options, start, end, true);
                        let mut store = LZ77Store::new(in_);
                        lz77_optimal(&mut s, in_, start, end, options.numiterations, &mut store);
                        store
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };

    for (i, store) in stores.iter().enumerate() {
        lz77.append_from(store);
        if i < npoints {
            splitpoints.push(lz77.size());
        }
    }

    // Re-split + re-squeeze the squeezed LZ77 stream: run TWO independent
    // whole trajectories (dynamic-only recursive-split search, and true
    // best-of-3 recursive-split search — see `refine_split_and_squeeze` doc)
    // from the SAME starting (lz77, splitpoints), each for two rounds
    // (mirrors ECT's `twice` mode), then keep whichever FINAL trajectory has
    // the lower true auto-type total cost. A per-round greedy mix of the two
    // search strategies was tried and measured to regress tool.bin net (the
    // strategies' resqueeze side-effects diverge after round 1, so a
    // locally-cheaper choice mid-trajectory is not globally cheaper) — running
    // both as committed, independent trajectories and comparing only the
    // final result is the version that cannot regress vs either strategy
    // alone, because "dynamic-only, unchanged from before this campaign" is
    // literally one of the two candidates.
    if options.blocksplitting != 0 {
        let (lz77_dyn, splitpoints_dyn) =
            refine_split_and_squeeze(options, in_, lz77.clone(), splitpoints.clone(), false, 2);
        let (lz77_auto, splitpoints_auto) =
            refine_split_and_squeeze(options, in_, lz77.clone(), splitpoints.clone(), true, 2);

        let cost_dyn = partition_true_cost(&lz77_dyn, &splitpoints_dyn);
        let cost_auto = partition_true_cost(&lz77_auto, &splitpoints_auto);
        if cost_auto < cost_dyn {
            lz77 = lz77_auto;
            splitpoints = splitpoints_auto;
        } else {
            lz77 = lz77_dyn;
            splitpoints = splitpoints_dyn;
        }
    }

    let np = splitpoints.len();
    for i in 0..=np {
        let start = if i == 0 { 0 } else { splitpoints[i - 1] };
        let end = if i == np { lz77.size() } else { splitpoints[i] };
        add_lz77_block_auto_type(options, i == np && final_, &lz77, start, end, 0, &mut w);
    }

    w.bp
}

/// Top-level deflate. Slices `in_` into ≤1 MB master blocks (per
/// `ZOPFLI_MASTER_BLOCK_SIZE`) and chains `deflate_part` calls. The returned
/// bit pointer is appended at the very end.
pub fn deflate(
    options: &ZopfliOptions,
    btype: i32,
    final_: bool,
    in_: &[u8],
    bp: u8,
    out: &mut Vec<u8>,
) -> u8 {
    let offset = out.len();
    let insize = in_.len();
    let mut bp = bp;
    let mut i: usize = 0;
    if ZOPFLI_MASTER_BLOCK_SIZE == 0 {
        bp = deflate_part(options, btype, final_, in_, 0, insize, bp, out);
    } else {
        loop {
            let masterfinal = i + ZOPFLI_MASTER_BLOCK_SIZE >= insize;
            let final2 = final_ && masterfinal;
            let size = if masterfinal {
                insize - i
            } else {
                ZOPFLI_MASTER_BLOCK_SIZE
            };
            bp = deflate_part(options, btype, final2, in_, i, i + size, bp, out);
            i += size;
            if i >= insize {
                break;
            }
        }
    }
    if options.verbose != 0 {
        let written = out.len() - offset;
        let saved = if insize > 0 {
            100.0 * (insize as f64 - written as f64) / insize as f64
        } else {
            0.0
        };
        eprintln!(
            "Original Size: {}, Deflate: {}, Compression: {}% Removed",
            insize, written, saved
        );
    }
    bp
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a writer, run a closure, return the resulting (bytes, bp).
    fn run<F: FnOnce(&mut BitWriter<'_>)>(f: F) -> (Vec<u8>, u8) {
        let mut out = Vec::new();
        let mut w = BitWriter::new(&mut out, 0);
        f(&mut w);
        let bp = w.bp;
        (out, bp)
    }

    #[test]
    fn add_bit_packs_lsb_first() {
        let (bytes, bp) = run(|w| {
            // Write 1, 0, 1, 1 → byte = 0b0000_1101 = 0x0D, bp = 4.
            w.add_bit(1);
            w.add_bit(0);
            w.add_bit(1);
            w.add_bit(1);
        });
        assert_eq!(bytes, vec![0x0D]);
        assert_eq!(bp, 4);
    }

    #[test]
    fn add_bits_lsb_first_eight_bits_makes_one_byte() {
        let (bytes, bp) = run(|w| {
            // 0xA5 = 0b1010_0101 — written LSB-first as bits 1,0,1,0,0,1,0,1.
            w.add_bits(0xA5, 8);
        });
        assert_eq!(bytes, vec![0xA5]);
        // 8 bits modulo 8 == 0 → bp wraps back to 0.
        assert_eq!(bp, 0);
    }

    #[test]
    fn add_bits_spans_two_bytes() {
        let (bytes, bp) = run(|w| {
            // 12 bits of 0xABC = 0b1010_1011_1100, LSB-first.
            // First 8 bits land in byte 0: bits 0..8 of 0xABC = 0xBC.
            // Next 4 bits land in byte 1, low nibble: 0xA.
            w.add_bits(0xABC, 12);
        });
        assert_eq!(bytes, vec![0xBC, 0x0A]);
        assert_eq!(bp, 4);
    }

    #[test]
    fn add_huffman_bits_msb_first() {
        let (bytes, bp) = run(|w| {
            // Symbol 0b101, length 3. MSB-first → emit 1, 0, 1.
            // LSB-stuffed into byte 0 → bits 0..3 = 1,0,1 → 0b0000_0101 = 0x05.
            w.add_huffman_bits(0b101, 3);
        });
        assert_eq!(bytes, vec![0x05]);
        assert_eq!(bp, 3);
    }

    #[test]
    fn matches_c_addbits_addhuffmanbits_for_random_sequence() {
        // Mirror the C routines bit-for-bit using a literal port; build a
        // reference byte stream and assert our writer matches.
        fn ref_add_bits(out: &mut Vec<u8>, bp: &mut u8, symbol: u32, length: u32) {
            for i in 0..length {
                let bit = ((symbol >> i) & 1) as u8;
                if *bp == 0 {
                    out.push(0);
                }
                let n = out.len();
                out[n - 1] |= bit << *bp;
                *bp = (*bp + 1) & 7;
            }
        }
        fn ref_add_huffman(out: &mut Vec<u8>, bp: &mut u8, symbol: u32, length: u32) {
            for i in 0..length {
                let bit = ((symbol >> (length - i - 1)) & 1) as u8;
                if *bp == 0 {
                    out.push(0);
                }
                let n = out.len();
                out[n - 1] |= bit << *bp;
                *bp = (*bp + 1) & 7;
            }
        }

        // Deterministic LCG so the test is reproducible.
        let mut s: u32 = 0xDEADBEEF;
        let mut next = || -> u32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s
        };

        let mut ref_out = Vec::new();
        let mut ref_bp: u8 = 0;
        let mut got_out = Vec::new();
        let got_bp;
        {
            let mut w = BitWriter::new(&mut got_out, 0);
            for _ in 0..200 {
                let r = next();
                let length = (r % 16) + 1; // 1..=16 bits
                let symbol = next() & ((1u32 << length) - 1);
                if r & 0x10 == 0 {
                    ref_add_bits(&mut ref_out, &mut ref_bp, symbol, length);
                    w.add_bits(symbol, length);
                } else {
                    ref_add_huffman(&mut ref_out, &mut ref_bp, symbol, length);
                    w.add_huffman_bits(symbol, length);
                }
            }
            got_bp = w.bp;
        }
        assert_eq!(got_out, ref_out);
        assert_eq!(got_bp, ref_bp);
    }
}
