//! C: `vendor/libdeflate/lib/deflate_compress.c:816-1320` — Huffman code
//! construction: symbol sorting, tree building, length counting, codeword
//! generation.
//!
//! Ported function-by-function. See `super`'s module docs for the porting rules.

use super::heap::heap_sort;
use super::DEFLATE_MAX_NUM_SYMS;

// C: :816-819
//
// Symbols are packed into a u32 as `(freq << NUM_SYMBOL_BITS) | symbol` so that a
// single integer sort orders primarily by frequency and secondarily by symbol
// value. This packing is why `heap_sort` operates on `u32` and why its tie
// behaviour does not matter: exact duplicates are impossible, because the low bits
// always disambiguate.
pub(crate) const NUM_SYMBOL_BITS: u32 = 10;
pub(crate) const NUM_FREQ_BITS: u32 = 32 - NUM_SYMBOL_BITS;
pub(crate) const SYMBOL_MASK: u32 = (1 << NUM_SYMBOL_BITS) - 1;
pub(crate) const FREQ_MASK: u32 = !SYMBOL_MASK;

/// C: `GET_NUM_COUNTERS(num_syms)` (:821)
///
/// The C keeps this as a macro so the counter count can be tuned independently of
/// the alphabet size; it currently resolves to the identity. Kept as a function so
/// the call sites read the same as the C's.
#[inline]
const fn get_num_counters(num_syms: usize) -> usize {
    num_syms
}

/// C: `sort_symbols(unsigned num_syms, const u32 freqs[], u8 lens[], u32 symout[])`
/// (:848)
///
/// Sort the symbols primarily by frequency and secondarily by symbol value.
/// Discard symbols with zero frequency and fill in an array with the remaining
/// symbols, along with their frequencies. The low `NUM_SYMBOL_BITS` bits of each
/// array entry will contain the symbol value, and the remaining bits will contain
/// the frequency.
///
/// * `num_syms` — number of symbols in the alphabet, at most `1 << NUM_SYMBOL_BITS`.
/// * `freqs[num_syms]` — frequency of each symbol, summing to at most
///   `(1 << NUM_FREQ_BITS) - 1`.
/// * `lens[num_syms]` — an array that eventually will hold the length of each
///   codeword. This function only fills in the codeword lengths for symbols that
///   have zero frequency, which are not well defined per se but will be set to 0.
/// * `symout[num_syms]` — the output array, described above.
///
/// Returns the number of entries in `symout` that were filled. This is the number of
/// symbols that have nonzero frequency.
///
/// # Why this is a counting sort with a heapsort tail
///
/// Frequencies are bucketed by `min(freq, num_counters - 1)`. Every symbol whose
/// frequency is below that saturation point lands in a bucket that is already in
/// exact order, so no comparison sort is needed for it. Only the final, saturated
/// bucket — symbols with `freq >= num_counters - 1`, which all collide — gets
/// `heap_sort`. That is the whole trick, and it is why the sort is O(n) in the
/// common case.
pub(crate) fn sort_symbols(
    num_syms: usize,
    freqs: &[u32],
    lens: &mut [u8],
    symout: &mut [u32],
) -> usize {
    // unsigned counters[GET_NUM_COUNTERS(DEFLATE_MAX_NUM_SYMS)];
    let mut counters = [0usize; DEFLATE_MAX_NUM_SYMS];

    let num_counters = get_num_counters(num_syms);

    // memset(counters, 0, num_counters * sizeof(counters[0]));
    // (already zero-initialised above; the slice below is the live prefix)
    let counters = &mut counters[..num_counters];

    // for (sym = 0; sym < num_syms; sym++)
    //         counters[MIN(freqs[sym], num_counters - 1)]++;
    for sym in 0..num_syms {
        let idx = core::cmp::min(freqs[sym] as usize, num_counters - 1);
        counters[idx] += 1;
    }

    // Sum the counts to transform them into offsets.
    //
    // NOTE: the loop starts at 1, deliberately. counters[0] counts the zero-frequency
    // symbols, which are DISCARDED — they never enter symout — so its count must not
    // contribute to any offset.
    let mut num_used_syms = 0usize;
    for i in 1..num_counters {
        let count = counters[i];
        counters[i] = num_used_syms;
        num_used_syms += count;
    }

    // Sort the symbols into symout, in order of increasing frequency.
    for sym in 0..num_syms {
        let freq = freqs[sym];

        if freq != 0 {
            let idx = core::cmp::min(freq as usize, num_counters - 1);
            symout[counters[idx]] = (sym as u32) | (freq << NUM_SYMBOL_BITS);
            counters[idx] += 1;
        } else {
            lens[sym] = 0;
        }
    }

    // Sort the symbols counted in the last counter. The counting sort above placed
    // every other symbol in exact order already; this bucket holds all the symbols
    // whose frequency saturated at `num_counters - 1`, so they need a real sort.
    //
    // After the fill loop each counters[i] points just past the end of bucket i,
    // so counters[num_counters - 2] is the START of the last bucket and
    // counters[num_counters - 1] is its END.
    let start = counters[num_counters - 2];
    let end = counters[num_counters - 1];
    heap_sort(&mut symout[start..], end - start);

    num_used_syms
}

/// C: `build_tree(u32 A[], unsigned sym_count)` (:941)
///
/// Build a Huffman tree.
///
/// This is an implementation of Algorithm FGK from "Van Leeuwen, J. (1976). On the
/// construction of Huffman trees" — the two-queue method. It takes the symbols
/// already sorted by frequency (as `sort_symbols` leaves them) and merges in O(n)
/// without any heap, because a sorted leaf queue plus a naturally-sorted internal
/// queue means the two smallest items are always at one of two heads.
///
/// # Input
///
/// `A[0..sym_count]` holds the symbols sorted primarily by frequency and
/// secondarily by symbol value, packed as `(freq << NUM_SYMBOL_BITS) | symbol`.
///
/// # Output
///
/// `A[0..sym_count - 1]` becomes the tree's internal nodes. Node `A[e]` has its
/// frequency in the high bits; a node's PARENT index is written into the high bits
/// of its children. The last node, `A[sym_count - 2]`, is the root.
///
/// # The two queues
///
/// * `i` — head of the LEAF queue: symbols not yet merged, already sorted.
/// * `b` — head of the INTERNAL queue: nodes created by earlier merges. These come
///   out in nondecreasing frequency automatically, which is the whole reason no
///   heap is needed.
/// * `e` — the node currently being written.
///
/// Each iteration picks the two smallest available items from the two queue heads.
/// The three-way branch is exactly that choice: two leaves, two internals, or one
/// of each. **The comparison operators are load-bearing** — the first branch uses
/// `<=` and the second `<`, which is what breaks ties toward taking leaves. Swap
/// either and the tree shape changes, which changes codeword lengths, which changes
/// the emitted bytes. Do not "normalise" them.
///
/// # Precondition
///
/// `sym_count >= 2`. The C's callers guarantee this — `deflate_make_huffman_code`
/// handles the 0-symbol and 1-symbol cases separately, because a do-while with
/// `last_idx == 0` would merge a node with itself.
// `i + 1 <= last_idx` and `b + 2 <= e` are clippy::int_plus_one hits. They are
// EXACTLY the C's expressions (deflate_compress.c:947, :955) and are kept
// character-for-character on purpose: this file's contract is that a reader can
// diff it against the C line by line. The rewrite clippy wants (`i < last_idx`) is
// arithmetically identical here, which is precisely why taking it costs a real
// review property and buys nothing.
#[allow(clippy::int_plus_one)]
pub(crate) fn build_tree(a: &mut [u32], sym_count: usize) {
    debug_assert!(
        sym_count >= 2,
        "build_tree requires >= 2 symbols; callers special-case 0 and 1"
    );

    let last_idx = sym_count - 1;

    // Index of the next parentless node in the leaf queue.
    let mut i: usize = 0;
    // Index of the next parentless node in the internal (branch) queue.
    let mut b: usize = 0;
    // Index of the next node to be created.
    let mut e: usize = 0;

    // C: do { ... } while (++e < last_idx);
    loop {
        let new_freq: u32;

        if i + 1 <= last_idx && (b == e || (a[i + 1] & FREQ_MASK) <= (a[b] & FREQ_MASK)) {
            // Two leaves are the cheapest pair.
            new_freq = (a[i] & FREQ_MASK) + (a[i + 1] & FREQ_MASK);
            i += 2;
        } else if b + 2 <= e && (i > last_idx || (a[b + 1] & FREQ_MASK) < (a[i] & FREQ_MASK)) {
            // Two internal nodes are the cheapest pair. Record `e` as their parent.
            new_freq = (a[b] & FREQ_MASK) + (a[b + 1] & FREQ_MASK);
            a[b] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b] & SYMBOL_MASK);
            a[b + 1] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b + 1] & SYMBOL_MASK);
            b += 2;
        } else {
            // One leaf and one internal node. Only the internal node needs its
            // parent recorded here; the leaf's parent is recorded when the leaf
            // slot is later overwritten as a node (see the C's comment).
            new_freq = (a[i] & FREQ_MASK) + (a[b] & FREQ_MASK);
            a[b] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b] & SYMBOL_MASK);
            i += 1;
            b += 1;
        }
        a[e] = new_freq | (a[e] & SYMBOL_MASK);

        e += 1;
        if e >= last_idx {
            break;
        }
    }
}

/// C: `compute_length_counts(u32 A[], unsigned root_idx, unsigned len_counts[],
/// unsigned max_codeword_len)` (:1024)
///
/// Given the stripped-down Huffman tree produced by `build_tree`, determine the
/// number of codewords that should be assigned each possible length, taking into
/// account the length-limited constraint.
///
/// # Inputs
///
/// * `A` — the array produced by `build_tree`, containing parent index information
///   for the non-leaf nodes of the tree. Each entry in this array is a node; a
///   node's parent always has a GREATER index than that of the node itself. This
///   function will overwrite the parent index information in this array, so
///   essentially it will destroy the tree. However, the data it needs will still be
///   valid at the time it is used.
/// * `root_idx` — the 0-based index of the root node in `A`, and consequently one
///   less than the number of tree node entries.
/// * `len_counts` — an array of length `max_codeword_len + 1` that will be filled in
///   with the number of codewords having each length `<= max_codeword_len`.
/// * `max_codeword_len` — the maximum permissible codeword length.
///
/// # How the length limit is enforced — and why it is NOT exact
///
/// The tree is walked from the root down (indices descending, which works because a
/// node's parent always has a greater index). Each node's depth is one more than its
/// parent's, and that depth is written back over the parent index — this is the
/// "destroys the tree" part, and it is safe only because parents are always visited
/// before their children.
///
/// When a node's depth would exceed `max_codeword_len`, it is clamped and the
/// deficit is paid for by moving a codeword from some shorter length: the `do {
/// depth--; } while (len_counts[depth] == 0)` scan walks DOWN to the nearest
/// non-empty length and steals from it. That is a HEURISTIC rebalance, not the
/// optimal length-limited code (package-merge would be optimal).
///
/// This matters for the campaign and is worth stating precisely: a binding
/// falsification already exists at `src/compress/deflate/huffman/fast.rs:432`
/// recording that libdeflate's heuristic limiter is within ~0.001% of the exact
/// package-merge optimum, that building it both ways is a wash which OPENS cells,
/// and that the costed dual-candidate variant holds size flat at ~0.001% while
/// costing 10-14% wall. So: this heuristic is the thing to COPY, not to improve.
/// Replacing it with an exact limiter is a known-dead lever, and doing so would also
/// break byte-identity, which is the entire point of this module.
pub(crate) fn compute_length_counts(
    a: &mut [u32],
    root_idx: usize,
    len_counts: &mut [u32],
    max_codeword_len: usize,
) {
    // for (len = 0; len <= max_codeword_len; len++) len_counts[len] = 0;
    for len in 0..=max_codeword_len {
        len_counts[len] = 0;
    }

    // The root node counts as 2 codewords of length 1: it has two children, and
    // every codeword descends from one of them.
    len_counts[1] = 2;

    // Set the root node's depth to 0. (The high bits held its parent index, which
    // is meaningless for the root.)
    a[root_idx] &= SYMBOL_MASK;

    // Walk from the root downward. `node` descends, and because a node's parent
    // always has a GREATER index, every parent's depth is already computed by the
    // time its children are visited.
    //
    // NOTE: the C uses a signed `int node` counting down to 0 inclusive, so the loop
    // ends when node goes negative. A usize would wrap, so this is written as a
    // descending range — the visit order is identical.
    for node in (0..root_idx).rev() {
        let parent = (a[node] >> NUM_SYMBOL_BITS) as usize;
        let parent_depth = a[parent] >> NUM_SYMBOL_BITS;
        let mut depth = parent_depth + 1;

        // Overwrite the parent index with this node's depth, in place.
        a[node] = (a[node] & SYMBOL_MASK) | (depth << NUM_SYMBOL_BITS);

        // If needed, decrease the length to meet the length-limited constraint,
        // paying for it by lengthening a codeword at some shorter, non-empty length.
        if depth as usize >= max_codeword_len {
            depth = max_codeword_len as u32;
            // do { depth--; } while (len_counts[depth] == 0);
            loop {
                depth -= 1;
                if len_counts[depth as usize] != 0 {
                    break;
                }
            }
        }

        len_counts[depth as usize] -= 1;
        len_counts[depth as usize + 1] += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the C's contract, restated — nonzero-frequency symbols only,
    /// ordered by (freq, symbol) ascending, packed as `(freq << 10) | sym`.
    fn reference_sorted(num_syms: usize, freqs: &[u32]) -> Vec<u32> {
        let mut v: Vec<u32> = (0..num_syms)
            .filter(|&s| freqs[s] != 0)
            .map(|s| (s as u32) | (freqs[s] << NUM_SYMBOL_BITS))
            .collect();
        v.sort_unstable();
        v
    }

    fn check(num_syms: usize, freqs: &[u32]) {
        let mut lens = vec![0xAAu8; num_syms];
        let mut symout = vec![0u32; num_syms];
        let n = sort_symbols(num_syms, freqs, &mut lens, &mut symout);

        let want = reference_sorted(num_syms, freqs);
        assert_eq!(n, want.len(), "num_used_syms");
        assert_eq!(&symout[..n], &want[..], "ordering (freqs={freqs:?})");

        // Zero-frequency symbols must have had their length cleared; nonzero ones
        // must NOT have been touched by this function.
        for s in 0..num_syms {
            if freqs[s] == 0 {
                assert_eq!(lens[s], 0, "sym {s} zero-freq len");
            } else {
                assert_eq!(lens[s], 0xAA, "sym {s} nonzero-freq len must be untouched");
            }
        }
    }

    #[test]
    fn sort_symbols_matches_reference_on_small_alphabets() {
        check(4, &[0, 0, 0, 0]);
        check(4, &[1, 0, 0, 0]);
        check(4, &[1, 1, 1, 1]);
        check(4, &[3, 1, 2, 0]);
        check(8, &[5, 5, 5, 5, 1, 1, 1, 1]);
    }

    /// The saturating bucket is the only part that reaches `heap_sort`. Frequencies
    /// at and above `num_counters - 1` all collide there, so this is the case that
    /// would break if the heapsort or the start/end arithmetic were wrong.
    #[test]
    fn sort_symbols_saturated_bucket_is_sorted() {
        let num_syms = 16;
        // Every frequency >= num_syms - 1 = 15 saturates into the last bucket.
        let freqs: Vec<u32> = vec![100, 3, 99, 1, 15, 2, 40, 0, 16, 7, 15, 0, 250, 1, 60, 4];
        check(num_syms, &freqs);
    }

    /// `build_tree`'s strongest checkable invariant without the rest of the chain
    /// ported: the ROOT's frequency must equal the sum of every input frequency,
    /// because every leaf is merged in exactly once. A wrong branch condition, a
    /// double-consumed queue head, or an off-by-one in `last_idx` all break this.
    fn build_tree_root_freq_equals_total(freqs: &[u32]) {
        let num_syms = freqs.len();
        let mut lens = vec![0u8; num_syms];
        let mut a = vec![0u32; num_syms];
        let n = sort_symbols(num_syms, freqs, &mut lens, &mut a);
        if n < 2 {
            return; // build_tree's precondition; callers special-case these.
        }

        let total: u32 = freqs.iter().sum();
        build_tree(&mut a, n);

        // Root is A[sym_count - 2]; its frequency lives in the high bits.
        let root_freq = (a[n - 2] & FREQ_MASK) >> NUM_SYMBOL_BITS;
        assert_eq!(root_freq, total, "root freq != total (freqs={freqs:?})");
    }

    #[test]
    fn build_tree_conserves_frequency() {
        build_tree_root_freq_equals_total(&[1, 1]);
        build_tree_root_freq_equals_total(&[1, 2, 3]);
        build_tree_root_freq_equals_total(&[1, 1, 1, 1]);
        build_tree_root_freq_equals_total(&[5, 0, 3, 0, 1, 9]);
        build_tree_root_freq_equals_total(&[100, 1, 1, 1, 1, 1, 1, 1]);
        // Powers of two: the shape that makes the internal queue overtake the leaf
        // queue earliest, exercising the two-internals branch hardest.
        build_tree_root_freq_equals_total(&[1, 2, 4, 8, 16, 32, 64, 128]);
        // Flat: every merge is a tie, so this is where the `<=` vs `<` operators
        // decide the tree shape.
        build_tree_root_freq_equals_total(&[7u32; 32]);
    }

    #[test]
    fn build_tree_conserves_frequency_on_random_spreads() {
        let mut state: u32 = 0xBEEF_0042;
        for trial in 0..64 {
            let num_syms = 2 + (trial % 60);
            let freqs: Vec<u32> = (0..num_syms)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (state >> 20) % 1000
                })
                .collect();
            build_tree_root_freq_equals_total(&freqs);
        }
    }

    /// `compute_length_counts` must produce a COMPLETE code: Kraft equality,
    /// `sum(len_counts[l] * 2^-l) == 1`, checked in integers as
    /// `sum(len_counts[l] << (max - l)) == 1 << max`.
    ///
    /// This is the strongest available check on the length-limiter. If the
    /// `do { depth--; } while (len_counts[depth] == 0)` steal-scan took from the
    /// wrong length, or the `len_counts[depth+1] += 2` accounting were off, the
    /// codespace would over- or under-fill and this would catch it. It also pins
    /// that the limit is actually RESPECTED: no count above `max_codeword_len`.
    fn check_kraft(freqs: &[u32], max_codeword_len: usize) {
        let num_syms = freqs.len();
        let mut lens = vec![0u8; num_syms];
        let mut a = vec![0u32; num_syms];
        let n = sort_symbols(num_syms, freqs, &mut lens, &mut a);
        if n < 2 {
            return; // build_tree's precondition
        }
        build_tree(&mut a, n);

        let mut len_counts = vec![0u32; max_codeword_len + 2];
        compute_length_counts(&mut a, n - 2, &mut len_counts, max_codeword_len);

        let mut space: u64 = 0;
        for l in 1..=max_codeword_len {
            space += (len_counts[l] as u64) << (max_codeword_len - l);
        }
        assert_eq!(
            space,
            1u64 << max_codeword_len,
            "Kraft inequality not tight (n={n}, max={max_codeword_len}, counts={:?})",
            &len_counts[..=max_codeword_len]
        );

        let total: u32 = len_counts[1..=max_codeword_len].iter().sum();
        assert_eq!(total as usize, n, "codeword count != symbol count");
    }

    #[test]
    fn compute_length_counts_produces_a_complete_code() {
        for &max in &[7usize, 15] {
            check_kraft(&[1, 1], max);
            check_kraft(&[1, 2, 3], max);
            check_kraft(&[1, 1, 1, 1], max);
            check_kraft(&[5, 0, 3, 0, 1, 9], max);
            check_kraft(&[7; 32], max);
            // Fibonacci frequencies force the DEEPEST possible tree, which is the
            // only shape that actually exercises the length-limiting steal-scan.
            let mut fib = vec![1u32, 1];
            while fib.len() < 30 {
                let n = fib[fib.len() - 1] + fib[fib.len() - 2];
                fib.push(n);
            }
            check_kraft(&fib, max);
        }
    }

    #[test]
    fn compute_length_counts_random_spreads_stay_complete() {
        let mut state: u32 = 0x5EED_1234;
        for trial in 0..48 {
            let num_syms = 2 + (trial % 40);
            let freqs: Vec<u32> = (0..num_syms)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    1 + (state >> 18) % 4000
                })
                .collect();
            check_kraft(&freqs, 15);
        }
    }

    /// Full litlen alphabet with a deterministic spread, including many zeros and a
    /// long saturated tail — the shape a real block produces.
    #[test]
    fn sort_symbols_full_litlen_alphabet() {
        let num_syms = DEFLATE_MAX_NUM_SYMS;
        let mut state: u32 = 0xC0FF_EE01;
        let mut freqs = vec![0u32; num_syms];
        for f in freqs.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            // Bias hard toward zero and toward saturation, like real symbol data.
            *f = match (state >> 16) % 4 {
                0 => 0,
                1 => (state >> 8) % 3,
                2 => (state >> 8) % (num_syms as u32),
                _ => (state >> 4) % 5000,
            };
        }
        check(num_syms, &freqs);
    }
}
