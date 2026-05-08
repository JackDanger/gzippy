//! Block-size estimation, dynamic-tree size, RLE optimizer.
//! Size-only half of vendor/zopfli/src/zopfli/deflate.c (lines 74-621).
//! The bit-emitting half lives in `deflate.rs` (Step 14-15).

#![allow(dead_code)]

use super::lz77::LZ77Store;
use super::symbols::{
    dist_symbol, dist_symbol_extra_bits, length_symbol, length_symbol_extra_bits, ZOPFLI_NUM_D,
    ZOPFLI_NUM_LL,
};
use super::tree::calculate_bit_lengths;

/// Ensures there are at least 2 distance codes (some decoders require it).
pub fn patch_distance_codes_for_buggy_decoders(d_lengths: &mut [u32; ZOPFLI_NUM_D]) {
    let mut num_dist_codes = 0;
    for &len in d_lengths.iter().take(30) {
        if len != 0 {
            num_dist_codes += 1;
        }
        if num_dist_codes >= 2 {
            return;
        }
    }
    if num_dist_codes == 0 {
        d_lengths[0] = 1;
        d_lengths[1] = 1;
    } else if num_dist_codes == 1 {
        let i = if d_lengths[0] != 0 { 1 } else { 0 };
        d_lengths[i] = 1;
    }
}

/// Fixed Huffman code lengths for btype=01 blocks (DEFLATE §3.2.6).
pub fn get_fixed_tree(ll_lengths: &mut [u32; ZOPFLI_NUM_LL], d_lengths: &mut [u32; ZOPFLI_NUM_D]) {
    for x in ll_lengths.iter_mut().take(144) {
        *x = 8;
    }
    for x in ll_lengths.iter_mut().take(256).skip(144) {
        *x = 9;
    }
    for x in ll_lengths.iter_mut().take(280).skip(256) {
        *x = 7;
    }
    for x in ll_lengths.iter_mut().take(288).skip(280) {
        *x = 8;
    }
    for d in d_lengths.iter_mut() {
        *d = 5;
    }
}

/// Order in which the 19 code-length code lengths are emitted.
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Walks the concatenated `(ll_lengths || d_lengths)` table once, optionally
/// recording the RLE-encoded `(symbol, extra_bits)` stream into the provided
/// vectors, and returning the `(clcounts, hclen, hlit, hdist)` summary.
///
/// Pass `None` for `rle` and `rle_bits` to run in size-only mode (no
/// allocations beyond the four scalar return values). The size-only path
/// (Step 9 dynamic-tree cost loop) calls this with `None`; the emit path
/// (Step 15 `add_dynamic_tree`) calls it with `Some` so the rle stream is
/// available for `add_huffman_bits`.
pub(super) fn build_rle(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    use_16: bool,
    use_17: bool,
    use_18: bool,
    mut rle: Option<&mut Vec<u32>>,
    mut rle_bits: Option<&mut Vec<u32>>,
) -> ([usize; 19], usize, usize, usize) {
    let mut clcounts = [0usize; 19];

    let mut hlit: usize = 29;
    let mut hdist: usize = 29;
    while hlit > 0 && ll_lengths[257 + hlit - 1] == 0 {
        hlit -= 1;
    }
    while hdist > 0 && d_lengths[1 + hdist - 1] == 0 {
        hdist -= 1;
    }
    let hlit2 = hlit + 257;
    let lld_total = hlit2 + hdist + 1;

    let symbol_at = |i: usize| -> u32 {
        if i < hlit2 {
            ll_lengths[i]
        } else {
            d_lengths[i - hlit2]
        }
    };

    let emit = |sym: u32,
                bits: u32,
                rle: &mut Option<&mut Vec<u32>>,
                rle_bits: &mut Option<&mut Vec<u32>>| {
        if let Some(v) = rle.as_deref_mut() {
            v.push(sym);
        }
        if let Some(v) = rle_bits.as_deref_mut() {
            v.push(bits);
        }
    };

    let mut i = 0usize;
    while i < lld_total {
        let symbol = symbol_at(i) as u8;
        let mut count: usize = 1;
        if use_16 || (symbol == 0 && (use_17 || use_18)) {
            let mut j = i + 1;
            while j < lld_total && symbol_at(j) as u8 == symbol {
                count += 1;
                j += 1;
            }
        }
        i += count - 1;

        if symbol == 0 && count >= 3 {
            if use_18 {
                while count >= 11 {
                    let count2 = count.min(138);
                    emit(18, (count2 - 11) as u32, &mut rle, &mut rle_bits);
                    clcounts[18] += 1;
                    count -= count2;
                }
            }
            if use_17 {
                while count >= 3 {
                    let count2 = count.min(10);
                    emit(17, (count2 - 3) as u32, &mut rle, &mut rle_bits);
                    clcounts[17] += 1;
                    count -= count2;
                }
            }
        }

        if use_16 && count >= 4 {
            count -= 1;
            emit(symbol as u32, 0, &mut rle, &mut rle_bits);
            clcounts[symbol as usize] += 1;
            while count >= 3 {
                let count2 = count.min(6);
                emit(16, (count2 - 3) as u32, &mut rle, &mut rle_bits);
                clcounts[16] += 1;
                count -= count2;
            }
        }

        clcounts[symbol as usize] += count;
        for _ in 0..count {
            emit(symbol as u32, 0, &mut rle, &mut rle_bits);
        }

        i += 1;
    }

    let mut hclen: usize = 15;
    while hclen > 0 && clcounts[CL_ORDER[hclen + 4 - 1]] == 0 {
        hclen -= 1;
    }

    (clcounts, hclen, hlit, hdist)
}

/// CL_ORDER constant is also used by the emit side.
pub(super) const CL_ORDER_PUB: [usize; 19] = CL_ORDER;

/// Size in bits of the dynamic-tree encoding for one (use_16, use_17, use_18)
/// triple. Sums the rle-symbol bits, extra bits, and the fixed 14 + (hclen+4)*3
/// header bits.
fn encode_tree_size(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    use_16: bool,
    use_17: bool,
    use_18: bool,
) -> usize {
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

/// Smallest dynamic-tree encoding bits across the 8 (use_16, use_17, use_18)
/// flag combinations.
pub fn calculate_tree_size(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
) -> usize {
    let mut best: usize = 0;
    for i in 0..8 {
        let s = encode_tree_size(
            ll_lengths,
            d_lengths,
            (i & 1) != 0,
            (i & 2) != 0,
            (i & 4) != 0,
        );
        if best == 0 || s < best {
            best = s;
        }
    }
    best
}

fn calculate_block_symbol_size_small(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
) -> usize {
    let mut result = 0usize;
    for i in lstart..lend {
        let ll = lz77.litlens[i];
        let dist = lz77.dists[i];
        if dist == 0 {
            result += ll_lengths[ll as usize] as usize;
        } else {
            let ls = length_symbol(ll as i32);
            let ds = dist_symbol(dist as i32);
            result += ll_lengths[ls as usize] as usize;
            result += d_lengths[ds as usize] as usize;
            result += length_symbol_extra_bits(ls) as usize;
            result += dist_symbol_extra_bits(ds) as usize;
        }
    }
    result += ll_lengths[256] as usize;
    result
}

fn calculate_block_symbol_size_given_counts(
    ll_counts: &[usize; ZOPFLI_NUM_LL],
    d_counts: &[usize; ZOPFLI_NUM_D],
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
) -> usize {
    if lstart + ZOPFLI_NUM_LL * 3 > lend {
        return calculate_block_symbol_size_small(ll_lengths, d_lengths, lz77, lstart, lend);
    }
    let mut result = 0usize;
    for i in 0..256 {
        result += ll_lengths[i] as usize * ll_counts[i];
    }
    for i in 257..286 {
        result += ll_lengths[i] as usize * ll_counts[i];
        result += length_symbol_extra_bits(i as i32) as usize * ll_counts[i];
    }
    for i in 0..30 {
        result += d_lengths[i] as usize * d_counts[i];
        result += dist_symbol_extra_bits(i as i32) as usize * d_counts[i];
    }
    result += ll_lengths[256] as usize;
    result
}

fn calculate_block_symbol_size(
    ll_lengths: &[u32; ZOPFLI_NUM_LL],
    d_lengths: &[u32; ZOPFLI_NUM_D],
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
) -> usize {
    if lstart + ZOPFLI_NUM_LL * 3 > lend {
        return calculate_block_symbol_size_small(ll_lengths, d_lengths, lz77, lstart, lend);
    }
    let mut ll_counts = [0usize; ZOPFLI_NUM_LL];
    let mut d_counts = [0usize; ZOPFLI_NUM_D];
    lz77.get_histogram(lstart, lend, &mut ll_counts, &mut d_counts);
    calculate_block_symbol_size_given_counts(
        &ll_counts, &d_counts, ll_lengths, d_lengths, lz77, lstart, lend,
    )
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

fn try_optimize_huffman_for_rle(
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
    ll_counts: &[usize; ZOPFLI_NUM_LL],
    d_counts: &[usize; ZOPFLI_NUM_D],
    ll_lengths: &mut [u32; ZOPFLI_NUM_LL],
    d_lengths: &mut [u32; ZOPFLI_NUM_D],
) -> f64 {
    let treesize = calculate_tree_size(ll_lengths, d_lengths) as f64;
    let datasize = calculate_block_symbol_size_given_counts(
        ll_counts, d_counts, ll_lengths, d_lengths, lz77, lstart, lend,
    ) as f64;

    let mut ll_counts2 = *ll_counts;
    let mut d_counts2 = *d_counts;
    optimize_huffman_for_rle(&mut ll_counts2);
    optimize_huffman_for_rle(&mut d_counts2);

    let mut ll_lengths2 = [0u32; ZOPFLI_NUM_LL];
    let mut d_lengths2 = [0u32; ZOPFLI_NUM_D];
    calculate_bit_lengths(&ll_counts2, 15, &mut ll_lengths2);
    calculate_bit_lengths(&d_counts2, 15, &mut d_lengths2);
    patch_distance_codes_for_buggy_decoders(&mut d_lengths2);

    let treesize2 = calculate_tree_size(&ll_lengths2, &d_lengths2) as f64;
    let datasize2 = calculate_block_symbol_size_given_counts(
        ll_counts,
        d_counts,
        &ll_lengths2,
        &d_lengths2,
        lz77,
        lstart,
        lend,
    ) as f64;

    if treesize2 + datasize2 < treesize + datasize {
        *ll_lengths = ll_lengths2;
        *d_lengths = d_lengths2;
        return treesize2 + datasize2;
    }
    treesize + datasize
}

/// Computes (ll_lengths, d_lengths) for the dynamic block and returns the
/// total bit cost of tree+data (excluding the 3-bit block header).
pub fn get_dynamic_lengths(
    lz77: &LZ77Store<'_>,
    lstart: usize,
    lend: usize,
    ll_lengths: &mut [u32; ZOPFLI_NUM_LL],
    d_lengths: &mut [u32; ZOPFLI_NUM_D],
) -> f64 {
    let mut ll_counts = [0usize; ZOPFLI_NUM_LL];
    let mut d_counts = [0usize; ZOPFLI_NUM_D];
    lz77.get_histogram(lstart, lend, &mut ll_counts, &mut d_counts);
    ll_counts[256] = 1;
    calculate_bit_lengths(&ll_counts, 15, ll_lengths);
    calculate_bit_lengths(&d_counts, 15, d_lengths);
    patch_distance_codes_for_buggy_decoders(d_lengths);
    try_optimize_huffman_for_rle(
        lz77, lstart, lend, &ll_counts, &d_counts, ll_lengths, d_lengths,
    )
}

/// Bit cost of a single LZ77 block in `lz77[lstart..lend]` for the chosen
/// `btype` (0 = stored, 1 = fixed, 2 = dynamic).
pub fn calculate_block_size(lz77: &LZ77Store<'_>, lstart: usize, lend: usize, btype: i32) -> f64 {
    let mut result: f64 = 3.0; // bfinal + btype bits

    if btype == 0 {
        let length = lz77.byte_range(lstart, lend);
        let rem = length % 65535;
        let blocks = length / 65535 + usize::from(rem != 0);
        return (blocks * 5 * 8 + length * 8) as f64;
    }
    if btype == 1 {
        let mut ll_lengths = [0u32; ZOPFLI_NUM_LL];
        let mut d_lengths = [0u32; ZOPFLI_NUM_D];
        get_fixed_tree(&mut ll_lengths, &mut d_lengths);
        result += calculate_block_symbol_size(&ll_lengths, &d_lengths, lz77, lstart, lend) as f64;
    } else {
        let mut ll_lengths = [0u32; ZOPFLI_NUM_LL];
        let mut d_lengths = [0u32; ZOPFLI_NUM_D];
        result += get_dynamic_lengths(lz77, lstart, lend, &mut ll_lengths, &mut d_lengths);
    }
    result
}

/// Best-of-three across (stored, fixed, dynamic). Skips the fixed-cost
/// computation for stores larger than 1000 symbols since fixed almost never
/// wins there (matches C heuristic).
pub fn calculate_block_size_auto_type(lz77: &LZ77Store<'_>, lstart: usize, lend: usize) -> f64 {
    let uncompressedcost = calculate_block_size(lz77, lstart, lend, 0);
    let fixedcost = if lz77.size() > 1000 {
        uncompressedcost
    } else {
        calculate_block_size(lz77, lstart, lend, 1)
    };
    let dyncost = calculate_block_size(lz77, lstart, lend, 2);
    if uncompressedcost < fixedcost && uncompressedcost < dyncost {
        uncompressedcost
    } else if fixedcost < dyncost {
        fixedcost
    } else {
        dyncost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_tree_lengths() {
        let mut ll = [0u32; ZOPFLI_NUM_LL];
        let mut d = [0u32; ZOPFLI_NUM_D];
        get_fixed_tree(&mut ll, &mut d);
        assert_eq!(ll[0], 8);
        assert_eq!(ll[143], 8);
        assert_eq!(ll[144], 9);
        assert_eq!(ll[255], 9);
        assert_eq!(ll[256], 7);
        assert_eq!(ll[279], 7);
        assert_eq!(ll[280], 8);
        assert_eq!(ll[287], 8);
        for &x in d.iter() {
            assert_eq!(x, 5);
        }
    }

    #[test]
    fn patch_dist_zero() {
        let mut d = [0u32; ZOPFLI_NUM_D];
        patch_distance_codes_for_buggy_decoders(&mut d);
        assert_eq!(d[0], 1);
        assert_eq!(d[1], 1);
    }

    #[test]
    fn patch_dist_one() {
        let mut d = [0u32; ZOPFLI_NUM_D];
        d[5] = 4;
        patch_distance_codes_for_buggy_decoders(&mut d);
        assert_eq!(d[0], 1);
        assert_eq!(d[5], 4);
    }

    #[test]
    fn patch_dist_two_no_op() {
        let mut d = [0u32; ZOPFLI_NUM_D];
        d[3] = 4;
        d[7] = 5;
        let before = d;
        patch_distance_codes_for_buggy_decoders(&mut d);
        assert_eq!(d, before);
    }
}
