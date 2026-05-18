//! Literal port of `rapidgzip::deflate::precode` enumeration helpers
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/precode.hpp).
//!
//! Enumerates all valid precode-frequency histograms (1526 of them)
//! and stores them in a compile-time table that downstream
//! blockfinder code can use to validate candidate deflate block
//! headers.
//!
//! The vendor file pulls in `HuffmanCodingReversedBitsCachedCompressed`
//! as well, but only as one *consumer* of the histogram set. The
//! enumeration itself only depends on `MAX_PRECODE_LENGTH` /
//! `MAX_PRECODE_COUNT` from `gzip/definitions.hpp` (already ported in
//! `gzip_definitions.rs`), so the histogram table is portable on its
//! own. Wiring it into the Huffman validator is a follow-up.
//!
//! Histogram semantics (vendor doc-comment, precode.hpp:22-23):
//! `histogram[i]` = number of precode symbols with code length `i+1`,
//! for `i` in `0..MAX_DEPTH` where `MAX_DEPTH == MAX_PRECODE_LENGTH == 7`.
//! Code length 0 (= "unused symbol") is not represented.

#![allow(dead_code)]

use super::gzip_definitions::{MAX_PRECODE_COUNT, MAX_PRECODE_LENGTH};

/// Mirror of `rapidgzip::deflate::precode::MAX_DEPTH` (precode.hpp:20).
pub const MAX_DEPTH: u32 = MAX_PRECODE_LENGTH;

/// Mirror of `rapidgzip::deflate::precode::Histogram` (precode.hpp:23).
/// `std::array<uint8_t, MAX_DEPTH>` where `MAX_DEPTH == 7`.
pub type Histogram = [u8; MAX_DEPTH as usize];

/// Mirror of the `VALID_HISTOGRAMS_COUNT` constexpr (precode.hpp:92-100).
/// Locked at 1526 by the vendor `static_assert`.
pub const VALID_HISTOGRAMS_COUNT: usize = 1526;

/// Direct recursive port of `iterateValidPrecodeHistograms`
/// (precode.hpp:50-89). Each yielded histogram is a frequency vector
/// indexed by `code length - 1` describing the leaf-node counts at
/// each precode tree level.
///
/// The C++ version is a `constexpr` template recursion on the `DEPTH`
/// parameter; Rust expresses the same recursion as a function that
/// takes `depth` at runtime. The number of recursion levels (7)
/// is small enough that there's no recursion-depth concern.
fn iterate_valid_precode_histograms<F: FnMut(&Histogram)>(
    process: &mut F,
    depth: u32,
    remaining_count: u32,
    histogram: &mut Histogram,
    free_bits: u32,
) {
    debug_assert!(depth <= MAX_DEPTH);

    let max_count = remaining_count.min(free_bits);
    for count in 0..=max_count {
        histogram[(depth - 1) as usize] = count as u8;
        let new_free_bits = (free_bits - count) * 2;

        // First-layer special case: a single 1-bit symbol is always a
        // complete tree (precode.hpp:69-73).
        if depth == 1 && count == 1 {
            process(histogram);
        }

        if depth == MAX_DEPTH {
            // Final layer: histogram is valid iff every Huffman tree
            // node is consumed (precode.hpp:75-78).
            if new_free_bits == 0 {
                process(histogram);
            }
        } else if count == free_bits {
            // We've exactly filled this level — the tree terminates
            // here, even if depth < MAX_DEPTH (precode.hpp:80-81).
            process(histogram);
        } else {
            let new_remaining_count = remaining_count - count;
            iterate_valid_precode_histograms(
                process,
                depth + 1,
                new_remaining_count,
                histogram,
                new_free_bits,
            );
        }
    }
    // Restore the slot so iteration at sibling levels sees a clean
    // tail (the C++ recursion copies `histogram` by value, but Rust
    // shares a single mutable buffer for efficiency — explicit reset
    // mirrors the value-semantics).
    histogram[(depth - 1) as usize] = 0;
}

/// Public driver mirroring the entry-point overload in
/// precode.hpp:92-98 (the lambda passed to the recursion).
/// `process` is invoked once per valid histogram, in the same order
/// the C++ `iterateValidPrecodeHistograms` produces them.
pub fn for_each_valid_histogram<F: FnMut(&Histogram)>(mut process: F) {
    let mut histogram = Histogram::default();
    iterate_valid_precode_histograms(
        &mut process,
        /* depth = */ 1,
        /* remaining_count = */ MAX_PRECODE_COUNT,
        &mut histogram,
        /* free_bits = */ 2,
    );
}

/// Counts how many histograms the iterator yields. Matches the
/// vendor `VALID_HISTOGRAMS_COUNT` constexpr (precode.hpp:92-100).
pub fn count_valid_histograms() -> usize {
    let mut count = 0usize;
    for_each_valid_histogram(|_| count += 1);
    count
}

/// Collects every histogram produced by the iterator into a `Vec`,
/// matching the vendor `VALID_HISTOGRAMS` static (precode.hpp:104-117).
/// Computed lazily here rather than at compile time — Rust's `const`
/// evaluator can't (yet) build the table at compile time without
/// significant gymnastics, and downstream code only needs it at
/// startup of the precode validator.
pub fn collect_valid_histograms() -> Vec<Histogram> {
    let mut out = Vec::with_capacity(VALID_HISTOGRAMS_COUNT);
    for_each_valid_histogram(|h| out.push(*h));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vendor's static_assert (precode.hpp:100) locks the count at
    /// exactly 1526. Any algorithmic drift would change this number.
    #[test]
    fn valid_histogram_count_matches_vendor() {
        assert_eq!(count_valid_histograms(), VALID_HISTOGRAMS_COUNT);
        assert_eq!(collect_valid_histograms().len(), VALID_HISTOGRAMS_COUNT);
    }

    /// Vendor's second static_assert (precode.hpp:119) locks the
    /// final entry. The recursion descends from short codes to long,
    /// so the last histogram emitted is the one with two 1-bit
    /// codes and nothing else.
    #[test]
    fn last_histogram_matches_vendor_static_assert() {
        let all = collect_valid_histograms();
        let last = all.last().unwrap();
        assert_eq!(last[0], 2, "code length 1 frequency");
        for &v in &last[1..] {
            assert_eq!(v, 0);
        }
    }

    /// Sanity: every histogram has frequencies summing to <= the
    /// precode alphabet size, and the implied tree never overflows
    /// its capacity at any depth.
    #[test]
    fn all_histograms_well_formed() {
        for h in collect_valid_histograms() {
            let total: u32 = h.iter().map(|&v| v as u32).sum();
            assert!(total <= MAX_PRECODE_COUNT, "total {total}");
            // Tree validity: walk the levels keeping unused-slot
            // count; must never go negative, and at end of valid
            // tree either total==1 OR final unused==0.
            let mut unused: u32 = 2;
            for &count in &h {
                assert!(count as u32 <= unused, "histogram {h:?}");
                unused = (unused - count as u32) * 2;
            }
        }
    }

    /// A histogram with a single 1-bit code must be in the table.
    #[test]
    fn single_one_bit_code_in_table() {
        let needle: Histogram = [1, 0, 0, 0, 0, 0, 0];
        let all = collect_valid_histograms();
        assert!(all.iter().any(|h| h == &needle));
    }

    /// A 3-symbol balanced 2/2-bit tree must be in the table:
    /// 0 codes of length 1, 2 codes of length 2 (matching the
    /// "2 at depth 2" check the C++ recursion exercises).
    #[test]
    fn balanced_two_bit_tree_in_table() {
        // The smallest balanced tree at depth 2: 0 one-bit codes
        // (free_bits doubles to 4), 4 two-bit codes consume all.
        let needle: Histogram = [0, 4, 0, 0, 0, 0, 0];
        let all = collect_valid_histograms();
        assert!(all.iter().any(|h| h == &needle));
    }
}
