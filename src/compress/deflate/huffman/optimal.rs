//! EXACT length-limited Huffman code construction + RLE-aware count shaping —
//! the crown engine's (`parse::ultra`) huffman builder. Distinct from
//! [`super::fast`]'s APPROXIMATE libdeflate-style builder; kept as a sibling
//! module (Stage A moved it here structurally; Stage C, this module, adds the
//! RLE-shaping functions that were reachable only from `parse::ultra` before).
//!
//! Two sources are ported here:
//!
//! - Bounded package-merge length-limiting (`length_limited_code_lengths` +
//!   helpers) and the canonical-code conversion helpers
//!   (`calculate_bit_lengths`, `lengths_to_symbols`): port of Google Zopfli
//!   `katajainen.c` and the non-entropy half of `tree.c`. `calculate_entropy`
//!   (`tree.c`'s OTHER half) is cost-model machinery, not Huffman
//!   construction — it stayed with (moved into) `parse::ultra::squeeze`, its
//!   sole consumer; see the note there.
//! - `optimize_huffman_for_rle` / `try_optimize_huffman_for_rle` /
//!   `ShapeDepth`: RLE-aware count shaping, MOVED from
//!   `parse::ultra::deflate_size` (Stage C) because they shape Huffman
//!   symbol COUNTS before length-limiting, not block-SIZE math — the
//!   block-size estimation they're called from stays in `deflate_size`.
//!   `deflate_size::get_dynamic_lengths[_final]` call back into this module;
//!   `deflate_size::calculate_tree_size` /
//!   `deflate_size::calculate_block_symbol_size_given_counts` stay put and
//!   are called FROM here — a two-way edge between this module and
//!   `parse::ultra::deflate_size` that is inherent to the port (ultra keeps
//!   calling the moved functions via this new path), not a layering mistake.

#[derive(Clone, Copy)]
struct Node {
    weight: usize,
    count: i32,
    tail: i32, // arena index, or -1 for None
}

/// Computes length-limited Huffman code lengths.
///
/// Returns Ok(()) on success, Err(()) if (1<<maxbits) < numsymbols or weight overflow.
// Stage A structural move (docs/compressor-architecture.md) newly exposes
// this fn under the crate's already-public `compress::` tree (it was
// previously nested under the private `mod backends;`), which activates
// clippy's public-API-shape lint for the first time. This is a faithful
// port of Zopfli's `ZopfliLengthLimitedCodeLengths` — a bool-shaped
// Result<(), ()> — kept as-is rather than "improved" per the no-behavior-
// change rule for a structural move.
#[allow(clippy::result_unit_err)]
pub fn length_limited_code_lengths(
    frequencies: &[usize],
    maxbits: i32,
    bitlengths: &mut [u32],
) -> Result<(), ()> {
    let n = frequencies.len();

    for b in bitlengths.iter_mut() {
        *b = 0;
    }

    let mut leaves: Vec<Node> = Vec::with_capacity(n);
    for (i, &f) in frequencies.iter().enumerate() {
        if f > 0 {
            leaves.push(Node {
                weight: f,
                count: i as i32,
                tail: -1,
            });
        }
    }
    let numsymbols = leaves.len() as i32;

    if (1 << maxbits) < numsymbols {
        return Err(());
    }
    if numsymbols == 0 {
        return Ok(());
    }
    if numsymbols == 1 {
        bitlengths[leaves[0].count as usize] = 1;
        return Ok(());
    }
    if numsymbols == 2 {
        bitlengths[leaves[0].count as usize] += 1;
        bitlengths[leaves[1].count as usize] += 1;
        return Ok(());
    }

    for leaf in &leaves {
        if leaf.weight >= (1usize << (usize::BITS - 9)) {
            return Err(());
        }
    }

    leaves.sort_by(|a, b| a.weight.cmp(&b.weight).then(a.count.cmp(&b.count)));

    let maxbits = if numsymbols - 1 < maxbits {
        numsymbols - 1
    } else {
        maxbits
    } as usize;

    let arena_size = maxbits * 2 * numsymbols as usize;
    let mut arena: Vec<Node> = vec![
        Node {
            weight: 0,
            count: 0,
            tail: -1
        };
        arena_size
    ];
    let mut pool_next: usize = 0;

    let mut lists: Vec<[i32; 2]> = vec![[0i32; 2]; maxbits];

    let node0 = pool_next as i32;
    pool_next += 1;
    arena[node0 as usize] = Node {
        weight: leaves[0].weight,
        count: 1,
        tail: -1,
    };

    let node1 = pool_next as i32;
    pool_next += 1;
    arena[node1 as usize] = Node {
        weight: leaves[1].weight,
        count: 2,
        tail: -1,
    };

    lists.fill([node0, node1]);

    let num_runs = 2 * numsymbols as usize - 4;
    for _ in 0..num_runs - 1 {
        boundary_pm(
            &mut lists,
            &leaves,
            numsymbols,
            &mut arena,
            &mut pool_next,
            maxbits - 1,
        );
    }
    boundary_pm_final(
        &mut lists,
        &leaves,
        numsymbols,
        &mut arena,
        &mut pool_next,
        maxbits - 1,
    );

    extract_bit_lengths(lists[maxbits - 1][1], &arena, &leaves, bitlengths);

    Ok(())
}

fn init_node(weight: usize, count: i32, tail: i32, arena: &mut [Node], idx: usize) {
    arena[idx] = Node {
        weight,
        count,
        tail,
    };
}

fn boundary_pm(
    lists: &mut [[i32; 2]],
    leaves: &[Node],
    numsymbols: i32,
    arena: &mut [Node],
    pool_next: &mut usize,
    index: usize,
) {
    let lastcount = arena[lists[index][1] as usize].count;

    if index == 0 && lastcount >= numsymbols {
        return;
    }

    let newchain = *pool_next as i32;
    *pool_next += 1;
    let oldchain_idx = lists[index][1];

    lists[index][0] = oldchain_idx;
    lists[index][1] = newchain;

    if index == 0 {
        let lw = leaves[lastcount as usize].weight;
        init_node(lw, lastcount + 1, -1, arena, newchain as usize);
    } else {
        let sum =
            arena[lists[index - 1][0] as usize].weight + arena[lists[index - 1][1] as usize].weight;
        if lastcount < numsymbols && sum > leaves[lastcount as usize].weight {
            let lw = leaves[lastcount as usize].weight;
            let old_tail = arena[oldchain_idx as usize].tail;
            init_node(lw, lastcount + 1, old_tail, arena, newchain as usize);
        } else {
            let prev1 = lists[index - 1][1];
            init_node(sum, lastcount, prev1, arena, newchain as usize);
            boundary_pm(lists, leaves, numsymbols, arena, pool_next, index - 1);
            boundary_pm(lists, leaves, numsymbols, arena, pool_next, index - 1);
        }
    }
}

fn boundary_pm_final(
    lists: &mut [[i32; 2]],
    leaves: &[Node],
    numsymbols: i32,
    arena: &mut [Node],
    pool_next: &mut usize,
    index: usize,
) {
    let lastcount = arena[lists[index][1] as usize].count;
    let sum =
        arena[lists[index - 1][0] as usize].weight + arena[lists[index - 1][1] as usize].weight;

    if lastcount < numsymbols && sum > leaves[lastcount as usize].weight {
        let newchain_idx = *pool_next as i32;
        let oldchain_tail = arena[lists[index][1] as usize].tail;
        arena[newchain_idx as usize] = Node {
            weight: 0,
            count: lastcount + 1,
            tail: oldchain_tail,
        };
        lists[index][1] = newchain_idx;
    } else {
        let prev1 = lists[index - 1][1];
        arena[lists[index][1] as usize].tail = prev1;
    }
}

fn extract_bit_lengths(chain: i32, arena: &[Node], leaves: &[Node], bitlengths: &mut [u32]) {
    let mut counts = [0i32; 16];
    let mut end: usize = 16;
    let mut ptr: usize = 15;
    let mut value: u32 = 1;

    let mut node = chain;
    while node != -1 {
        end -= 1;
        counts[end] = arena[node as usize].count;
        node = arena[node as usize].tail;
    }

    let mut val = counts[15];
    while ptr >= end {
        let prev_count = if ptr > 0 { counts[ptr - 1] } else { 0 };
        while val > prev_count {
            bitlengths[leaves[(val - 1) as usize].count as usize] = value;
            val -= 1;
        }
        ptr = ptr.wrapping_sub(1);
        if ptr == usize::MAX {
            break;
        }
        value += 1;
    }
}

// ---- calculate_bit_lengths / lengths_to_symbols (port of zopfli tree.c) ----

/// Friendlier-signature wrapper over [`length_limited_code_lengths`]. Port of
/// Google Zopfli `tree.c`'s `ZopfliCalculateBitLengths`.
pub fn calculate_bit_lengths(count: &[usize], maxbits: i32, bitlengths: &mut [u32]) {
    length_limited_code_lengths(count, maxbits, bitlengths)
        .expect("ZopfliLengthLimitedCodeLengths failed");
}

/// Assign canonical (forward, non-bit-reversed) codewords from codeword
/// lengths. Port of Google Zopfli `tree.c`'s `ZopfliLengthsToSymbols`.
pub fn lengths_to_symbols(lengths: &[u32], maxbits: u32, symbols: &mut [u32]) {
    let n = lengths.len();
    for s in symbols.iter_mut() {
        *s = 0;
    }

    let mut bl_count = vec![0usize; maxbits as usize + 1];
    let mut next_code = vec![0usize; maxbits as usize + 1];

    for &len in lengths.iter() {
        bl_count[len as usize] += 1;
    }

    let mut code: usize = 0;
    bl_count[0] = 0;
    for bits in 1..=maxbits as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    for i in 0..n {
        let len = lengths[i] as usize;
        if len != 0 {
            symbols[i] = next_code[len] as u32;
            next_code[len] += 1;
        }
    }
}

// ---- RLE-aware count shaping (moved from parse::ultra::deflate_size, Stage C) ----

/// Table-shaping search depth for the RLE-smoothing candidate set. `TwoWay`
/// is the original zopfli behavior ({raw,raw} vs {smoothed,smoothed} — ll and
/// d counts smoothed only together); `FourWay` additionally tries smoothing
/// ll/d independently ({smoothed,raw}, {raw,smoothed}), mirroring the
/// reference optimal-parse frontier's `plan_dynamic` (fulcrum
/// `src/ratio/encode.rs`), which found those two combos matter (never a
/// regression AT A FIXED BLOCK: strictly more candidates, keep-min).
///
/// Kept SEPARATE from block-PLACEMENT decisions on purpose: measured, letting
/// `FourWay` costing leak into the recursive/greedy block-splitter's search
/// perturbs the split-position search (a non-exhaustive narrowing walk) onto
/// a different, occasionally WORSE, local optimum for the chosen boundaries
/// — even though each individual per-block table choice only ever improves.
/// So every SEARCH/cost-estimation call site (`calculate_block_size`,
/// `calculate_block_size_auto_type`, and therefore the block splitters) uses
/// `TwoWay`, bit-for-bit reproducing the pre-table-shaping search/placement
/// trajectory; only the literal FINAL emission call
/// (`get_dynamic_lengths_final`, `deflate.rs`'s `add_lz77_block`) uses
/// `FourWay`, so table-shaping can only ever shrink whatever block placement
/// search already chose — never move the boundaries.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeDepth {
    TwoWay,
    FourWay,
}

/// In-place histogram smoothing that biases the upcoming RLE encoding toward
/// reusable runs. Mirrors the C `OptimizeHuffmanForRle` exactly.
pub fn optimize_huffman_for_rle(counts: &mut [usize]) {
    let mut length = counts.len() as isize;
    // Trim trailing zeros — we may not modify them (would lengthen the code).
    loop {
        if length == 0 {
            return;
        }
        if counts[(length - 1) as usize] != 0 {
            break;
        }
        length -= 1;
    }
    let length = length as usize;

    let mut good_for_rle = vec![false; length];

    // Mark existing reasonable runs so they don't get smoothed away.
    let mut symbol = counts[0];
    let mut stride: usize = 0;
    for i in 0..=length {
        if i == length || counts[i] != symbol {
            if (symbol == 0 && stride >= 5) || (symbol != 0 && stride >= 7) {
                for k in 0..stride {
                    good_for_rle[i - k - 1] = true;
                }
            }
            stride = 1;
            if i != length {
                symbol = counts[i];
            }
        } else {
            stride += 1;
        }
    }

    // Replace counts that promise more rle codes if collapsed.
    let mut stride: usize = 0;
    let mut limit: usize = counts[0];
    let mut sum: usize = 0;
    for i in 0..=length {
        if i == length || good_for_rle[i] || counts[i].abs_diff(limit) >= 4 {
            if stride >= 4 || (stride >= 3 && sum == 0) {
                let mut count = (sum + stride / 2) / stride;
                if count < 1 {
                    count = 1;
                }
                if sum == 0 {
                    count = 0;
                }
                for k in 0..stride {
                    counts[i - k - 1] = count;
                }
            }
            stride = 0;
            sum = 0;
            limit = if i + 3 < length {
                (counts[i] + counts[i + 1] + counts[i + 2] + counts[i + 3] + 2) / 4
            } else if i < length {
                counts[i]
            } else {
                0
            };
        }
        stride += 1;
        if i != length {
            sum += counts[i];
        }
    }
}

/// Tries the RLE-smoothed litlen/offset count candidates (per `depth`,
/// [`ShapeDepth`]) against the as-given lengths and keeps whichever produces
/// the smallest true tree+data bit cost. Mirrors the C `TryOptimizeHuffmanForRle`.
///
/// Calls back into `parse::ultra::deflate_size` for the true-cost helpers
/// (`calculate_tree_size`, `calculate_block_symbol_size_given_counts`) —
/// those stay block-SIZE math, not Huffman-count shaping, so they were not
/// moved here; this function was, because what it DOES is reshape Huffman
/// symbol counts before length-limiting.
pub(crate) fn try_optimize_huffman_for_rle(
    lz77: &crate::compress::deflate::parse::ultra::lz77::LZ77Store<'_>,
    lstart: usize,
    lend: usize,
    ll_counts: &[usize; crate::compress::deflate::parse::ultra::symbols::ZOPFLI_NUM_LL],
    d_counts: &[usize; crate::compress::deflate::parse::ultra::symbols::ZOPFLI_NUM_D],
    ll_lengths: &mut [u32; crate::compress::deflate::parse::ultra::symbols::ZOPFLI_NUM_LL],
    d_lengths: &mut [u32; crate::compress::deflate::parse::ultra::symbols::ZOPFLI_NUM_D],
    depth: ShapeDepth,
) -> f64 {
    use crate::compress::deflate::parse::ultra::deflate_size::{
        calculate_block_symbol_size_given_counts, calculate_tree_size,
    };
    use crate::compress::deflate::parse::ultra::symbols::{ZOPFLI_NUM_D, ZOPFLI_NUM_LL};

    let treesize = calculate_tree_size(ll_lengths, d_lengths) as f64;
    let datasize = calculate_block_symbol_size_given_counts(
        ll_counts, d_counts, ll_lengths, d_lengths, lz77, lstart, lend,
    ) as f64;
    let mut best_total = treesize + datasize;
    let mut best_ll = *ll_lengths;
    let mut best_d = *d_lengths;

    let mut sll_counts = *ll_counts;
    optimize_huffman_for_rle(&mut sll_counts);
    let mut sd_counts = *d_counts;
    optimize_huffman_for_rle(&mut sd_counts);

    let mut sll_lengths = [0u32; ZOPFLI_NUM_LL];
    calculate_bit_lengths(&sll_counts, 15, &mut sll_lengths);
    let mut sd_lengths = [0u32; ZOPFLI_NUM_D];
    calculate_bit_lengths(&sd_counts, 15, &mut sd_lengths);
    crate::compress::deflate::parse::ultra::deflate_size::patch_distance_codes_for_buggy_decoders(
        &mut sd_lengths,
    );

    // TwoWay: only {smoothed, smoothed} (the original zopfli candidate).
    // FourWay: additionally {smoothed, raw} and {raw, smoothed}.
    let mut candidates: Vec<(&[u32; ZOPFLI_NUM_LL], &[u32; ZOPFLI_NUM_D])> = Vec::with_capacity(3);
    if depth == ShapeDepth::FourWay {
        candidates.push((&sll_lengths, d_lengths));
        candidates.push((ll_lengths, &sd_lengths));
    }
    candidates.push((&sll_lengths, &sd_lengths));

    for (cand_ll, cand_d) in candidates {
        let treesize_c = calculate_tree_size(cand_ll, cand_d) as f64;
        let datasize_c = calculate_block_symbol_size_given_counts(
            ll_counts, d_counts, cand_ll, cand_d, lz77, lstart, lend,
        ) as f64;
        let total_c = treesize_c + datasize_c;
        if total_c < best_total {
            best_total = total_c;
            best_ll = *cand_ll;
            best_d = *cand_d;
        }
    }

    *ll_lengths = best_ll;
    *d_lengths = best_d;
    best_total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_huffman() {
        let freq = vec![1usize, 2, 4, 8];
        let mut bl = vec![0u32; 4];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        for &b in &bl {
            assert!(b > 0);
        }
        let kraft: f64 = bl.iter().map(|&b| 2f64.powi(-(b as i32))).sum();
        assert!(kraft <= 1.0 + 1e-9, "Kraft inequality violated: {}", kraft);
    }

    #[test]
    fn single_symbol() {
        let freq = vec![0usize, 5, 0];
        let mut bl = vec![0u32; 3];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        assert_eq!(bl[1], 1);
        assert_eq!(bl[0], 0);
        assert_eq!(bl[2], 0);
    }

    #[test]
    fn two_symbols() {
        let freq = vec![3usize, 0, 7];
        let mut bl = vec![0u32; 3];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        assert_eq!(bl[0], 1);
        assert_eq!(bl[2], 1);
    }

    #[test]
    fn all_zero_frequencies() {
        let freq = vec![0usize; 10];
        let mut bl = vec![0u32; 10];
        length_limited_code_lengths(&freq, 15, &mut bl).unwrap();
        assert!(bl.iter().all(|&b| b == 0));
    }

    #[test]
    fn respects_maxbits() {
        let freq: Vec<usize> = (1..=32).collect();
        let mut bl = vec![0u32; 32];
        length_limited_code_lengths(&freq, 7, &mut bl).unwrap();
        for &b in &bl {
            assert!(b <= 7, "bitlength {} exceeds maxbits 7", b);
        }
    }

    #[test]
    fn lengths_to_symbols_basic() {
        // 4-symbol code with lengths [2, 2, 3, 3] → symbols [0, 1, 4, 5]
        let lengths = [2u32, 2, 3, 3];
        let mut symbols = [0u32; 4];
        lengths_to_symbols(&lengths, 3, &mut symbols);
        // Canonical Huffman: 00, 01, 100, 101
        assert_eq!(symbols[0], 0b00); // 0
        assert_eq!(symbols[1], 0b01); // 1
        assert_eq!(symbols[2], 0b100); // 4
        assert_eq!(symbols[3], 0b101); // 5
    }
}
