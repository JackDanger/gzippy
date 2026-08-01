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
