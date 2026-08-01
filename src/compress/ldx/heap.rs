//! C: `vendor/libdeflate/lib/deflate_compress.c:761-814` — the heapsort used to
//! order symbols by frequency before Huffman tree construction.
//!
//! # Why a hand-rolled heapsort and not `slice::sort_unstable`
//!
//! This is NOT a place to substitute Rust's sort. The C's comment for
//! `sort_symbols` (line 941 region) is explicit that the algorithm's *tie
//! ordering* is part of the contract: entries pack `(freq << 10) | symbol`, so
//! sorting the packed `u32` orders primarily by frequency and secondarily by
//! symbol value. Any sort that produces the same total order would agree here —
//! but the packed representation makes the comparison exact, and reproducing the
//! C's exact algorithm keeps that guarantee independent of what a library sort
//! does with equal keys in future. Byte-identity is the gate; a "faster sort" that
//! reorders equal elements differently would change codeword assignment and flip
//! 154 tied cells.
//!
//! # 1-based indexing
//!
//! The C does `A--` and then indexes `A[1] ..= A[length]`, so that the children of
//! `A[i]` are exactly `A[2i]` and `A[2i + 1]`. That arithmetic is why the array is
//! 1-based; rewriting it 0-based changes every child/parent expression and is
//! precisely the kind of "idiomatic cleanup" this port forbids. We keep the 1-based
//! logic and translate at the single point of access: logical `A[i]` is `a[i - 1]`.

/// C: `heapify_subtree(u32 A[], unsigned length, unsigned subtree_idx)` (:761)
///
/// Given the binary tree node `A[subtree_idx]` whose children already satisfy the
/// maxheap property, swap the node with its greater child until it is greater than
/// or equal to both of its children, so that the maxheap property is satisfied in
/// the subtree rooted at `A[subtree_idx]`. `A` uses 1-based indices.
fn heapify_subtree(a: &mut [u32], length: usize, subtree_idx: usize) {
    debug_assert!(subtree_idx >= 1);

    // v = A[subtree_idx];
    let v = a[subtree_idx - 1];
    let mut parent_idx = subtree_idx;
    let mut child_idx;

    // while ((child_idx = parent_idx * 2) <= length) {
    loop {
        child_idx = parent_idx * 2;
        if child_idx > length {
            break;
        }
        // if (child_idx < length && A[child_idx + 1] > A[child_idx]) child_idx++;
        if child_idx < length && a[child_idx] > a[child_idx - 1] {
            child_idx += 1;
        }
        // if (v >= A[child_idx]) break;
        if v >= a[child_idx - 1] {
            break;
        }
        // A[parent_idx] = A[child_idx];
        a[parent_idx - 1] = a[child_idx - 1];
        parent_idx = child_idx;
    }
    // A[parent_idx] = v;
    a[parent_idx - 1] = v;
}

/// C: `heapify_array(u32 A[], unsigned length)` (:785)
///
/// Rearrange the array `A` so that it satisfies the maxheap property. `A` uses
/// 1-based indices, so the children of `A[i]` are `A[i*2]` and `A[i*2 + 1]`.
fn heapify_array(a: &mut [u32], length: usize) {
    // for (subtree_idx = length / 2; subtree_idx >= 1; subtree_idx--)
    //
    // NOTE: `subtree_idx` is `unsigned` in the C, so when `length / 2 == 0` the
    // condition `0 >= 1` is false and the loop body never runs — there is no
    // underflow to reproduce. A `for i in (1..=length/2).rev()` has exactly that
    // behaviour (empty range when length < 2).
    for subtree_idx in (1..=(length / 2)).rev() {
        heapify_subtree(a, length, subtree_idx);
    }
}

/// C: `heap_sort(u32 A[], unsigned length)` (:800)
///
/// Sort the array `A`, which contains `length` unsigned 32-bit integers, ascending.
///
/// (The C notes it is named `heap_sort` rather than `heapsort` to avoid colliding
/// with `heapsort()` from `stdlib.h` on BSD-derived systems. Kept for grep parity.)
pub(crate) fn heap_sort(a: &mut [u32], length: usize) {
    debug_assert!(length <= a.len());

    // A--;  /* Use 1-based indices */   -- handled by the (i - 1) translation.
    heapify_array(a, length);

    // while (length >= 2) { swap A[length], A[1]; length--; heapify_subtree(A, length, 1); }
    let mut length = length;
    while length >= 2 {
        a.swap(length - 1, 0);
        length -= 1;
        heapify_subtree(a, length, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The port must sort ascending, matching the C. Checked against a reference
    /// sort on a deterministic spread of inputs — including the degenerate lengths
    /// where the 1-based arithmetic is easiest to get wrong.
    #[test]
    fn heap_sort_matches_reference() {
        // Deterministic LCG — no rand dependency, and reproducible across runs so a
        // failure is always the same failure.
        let mut state: u32 = 0x1234_5678;
        let mut next = move || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            state >> 8
        };

        for len in 0..=64usize {
            let mut a: Vec<u32> = (0..len).map(|_| next()).collect();
            let mut want = a.clone();
            want.sort_unstable();
            heap_sort(&mut a, len);
            assert_eq!(a, want, "len={len}");
        }
    }

    /// Equal keys: the packed `(freq << 10) | symbol` representation means exact
    /// duplicates cannot occur in real use (symbol disambiguates), but the sort
    /// must still be correct when they do.
    #[test]
    fn heap_sort_handles_duplicates_and_extremes() {
        let mut a = vec![7u32; 17];
        heap_sort(&mut a, 17);
        assert_eq!(a, vec![7u32; 17]);

        let mut a = vec![u32::MAX, 0, u32::MAX, 0, 1];
        heap_sort(&mut a, 5);
        assert_eq!(a, vec![0, 0, 1, u32::MAX, u32::MAX]);

        // Already sorted, and reverse sorted — the two shapes that exercise the
        // sift-down early-break (`v >= A[child_idx]`) most and least.
        let mut a: Vec<u32> = (0..32).collect();
        heap_sort(&mut a, 32);
        assert_eq!(a, (0..32).collect::<Vec<u32>>());

        let mut a: Vec<u32> = (0..32).rev().collect();
        heap_sort(&mut a, 32);
        assert_eq!(a, (0..32).collect::<Vec<u32>>());
    }

    /// `length` may be shorter than the slice; the tail must be untouched. The C
    /// relies on this — `sort_symbols` sorts only the populated prefix.
    #[test]
    fn heap_sort_respects_length_shorter_than_slice() {
        let mut a = vec![5, 3, 4, 1, 2, 99, 98];
        heap_sort(&mut a, 5);
        assert_eq!(&a[..5], &[1, 2, 3, 4, 5]);
        assert_eq!(&a[5..], &[99, 98], "tail past `length` must not be touched");
    }
}
