#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Literal port of `rapidgzip::GzipBlockFinder`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/GzipBlockFinder.hpp:34-307).
//!
//! Partitioner that hands out compressed-stream offsets — either
//! KNOWN-CONFIRMED ones (inserted via `insert(actual_end)` as workers
//! complete) or GUESSED ones at spacing-aligned positions for not-yet-
//! seen indexes. Mirror of rapidgzip's class with the FileReader / BGZF
//! / FileType branches stripped because gzippy detects the format at
//! routing layer (`decompress/mod.rs::classify_gzip`) and this module
//! is only ever called for the single-member parallel path.
//!
//! # Why this exists
//!
//! Rapidgzip's `BlockFetcher::get(blockIndex)` consults this
//! partitioner; when a worker completes, `appendSubchunksToIndexes`
//! calls `m_blockFinder->insert(actualEnd)` so the next-block index
//! promotes from a guess to a confirmed offset. Pre-port, gzippy's
//! `chunk_fetcher.rs::drive` used a STATIC partition by
//! `TARGET_COMPRESSED_CHUNK_BYTES * 8` with no feedback when chunk N
//! actually ends at bit X.
//!
//! `chunk_fetcher::consumer_loop` queries `get(idx)` for partition
//! seeds and calls `insert(actual_end)` per subchunk — mirror of
//! vendor's `m_blockFinder->get(...)` + `insert(...)` cascade in
//! `GzipChunkFetcher::processNextChunk`
//! (vendor/.../GzipChunkFetcher.hpp:318 + 374).

use std::sync::Mutex;

/// Return code for `get` — mirror of rapidgzip's `GetReturnCode`
/// (vendor/.../core/BlockFinderInterface.hpp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GetReturnCode {
    /// The returned offset is valid (either confirmed or a within-file
    /// guess).
    Success,
    /// The block index is past the file end; the returned offset is
    /// `file_size_in_bits` (so the caller can use it as an `until`
    /// boundary without waiting for that index to "become available").
    /// Mirror of rapidgzip's `GetReturnCode::FAILURE` at
    /// GzipBlockFinder.hpp:144-157.
    Failure,
}

/// Slim port of rapidgzip's `GzipBlockFinder` for the single-member
/// parallel path. Holds a sorted vec of confirmed deflate block
/// offsets in BITS and a spacing-in-bits constant for guesses.
///
/// **Threading**: all methods take `&self` and lock an internal
/// mutex — mirror of rapidgzip's `mutable std::mutex m_mutex` at
/// GzipBlockFinder.hpp:290.
pub struct GzipBlockFinder {
    inner: Mutex<Inner>,
    /// Spacing between guessed partition offsets, in BITS. Mirror of
    /// rapidgzip's `m_spacingInBits` (GzipBlockFinder.hpp:295).
    spacing_in_bits: usize,
    /// Optional file size in BITS. When set, `get` returns
    /// `(file_size, FAILURE)` for indexes past EOF; when None, all
    /// guesses succeed unconditionally.
    file_size_in_bits: Option<usize>,
}

struct Inner {
    /// Confirmed block offsets in bits, monotonically increasing.
    /// Mirror of rapidgzip's `std::deque<size_t> m_blockOffsets`
    /// (GzipBlockFinder.hpp:301).
    block_offsets: Vec<usize>,
    /// Once finalized, no further `insert` calls accepted; mirror of
    /// rapidgzip's `m_finalized` flag (GzipBlockFinder.hpp:294).
    finalized: bool,
}

impl GzipBlockFinder {
    /// Construct a new finder with the first confirmed offset
    /// (typically the byte position of the first deflate block in the
    /// input — `gzip_header_size * 8`). Mirror of rapidgzip's
    /// constructor (GzipBlockFinder.hpp:41-69), minus the file IO and
    /// the format-detection branch (gzippy detects at routing layer).
    ///
    /// # Panics
    /// Panics if `spacing_in_bytes * 8 < 32 KiB`, mirroring
    /// rapidgzip's `throw std::invalid_argument` at GzipBlockFinder.hpp:50-56.
    pub fn new(
        first_block_offset_in_bits: usize,
        spacing_in_bytes: usize,
        file_size_in_bits: Option<usize>,
    ) -> Self {
        let spacing_in_bits = spacing_in_bytes * 8;
        // Rapidgzip enforces "spacing smaller than window size makes no
        // sense" at GzipBlockFinder.hpp:50-56. The window is 32 KiB.
        const MIN_SPACING_BITS: usize = 32 * 1024 * 8;
        assert!(
            spacing_in_bits >= MIN_SPACING_BITS,
            "spacing {spacing_in_bytes} bytes too small (must be >= 32 KiB)"
        );
        Self {
            inner: Mutex::new(Inner {
                block_offsets: vec![first_block_offset_in_bits],
                finalized: false,
            }),
            spacing_in_bits,
            file_size_in_bits,
        }
    }

    /// Number of confirmed block offsets currently held. Mirror of
    /// rapidgzip's `size()` (GzipBlockFinder.hpp:74-79).
    pub fn size(&self) -> usize {
        self.inner.lock().unwrap().block_offsets.len()
    }

    /// Mark the finder as finalized — no further `insert` calls will be
    /// accepted. Mirror of rapidgzip's `finalize()`
    /// (GzipBlockFinder.hpp:81-86).
    pub fn finalize(&self) {
        self.inner.lock().unwrap().finalized = true;
    }

    /// Whether the finder has been finalized. Mirror of
    /// rapidgzip's `finalized()` (GzipBlockFinder.hpp:88-93).
    pub fn finalized(&self) -> bool {
        self.inner.lock().unwrap().finalized
    }

    /// Spacing-in-bits between guessed partition offsets. Mirror of
    /// `spacingInBits()` (GzipBlockFinder.hpp:202-206).
    pub fn spacing_in_bits(&self) -> usize {
        self.spacing_in_bits
    }

    /// Insert a known-exact block offset. Inserts are typically in
    /// sequence (workers finalize chunks in order), but out-of-order is
    /// supported via a sorted insert. Mirror of rapidgzip's `insert()`
    /// (GzipBlockFinder.hpp:101-110) which forwards to `insertUnsafe()`
    /// (GzipBlockFinder.hpp:226-243).
    ///
    /// Returns `false` if the offset is past EOF (and so couldn't be
    /// inserted); `true` if newly added OR already present. Mirror of
    /// rapidgzip's `insertUnsafe` bool return.
    ///
    /// # Panics
    /// Panics if `block_offset` is novel AND the finder has already
    /// been finalized — mirror of rapidgzip's `throw std::invalid_argument`
    /// at GzipBlockFinder.hpp:235-237.
    pub fn insert(&self, block_offset_in_bits: usize) -> bool {
        if let Some(size) = self.file_size_in_bits {
            if block_offset_in_bits >= size {
                return false;
            }
        }
        let mut g = self.inner.lock().unwrap();
        match g.block_offsets.binary_search(&block_offset_in_bits) {
            Ok(_) => true, // already present; mirror C++ returning true
            Err(pos) => {
                if g.finalized {
                    panic!(
                        "GzipBlockFinder: already finalized, may not insert further block offsets"
                    );
                }
                g.block_offsets.insert(pos, block_offset_in_bits);
                debug_assert!(g.block_offsets.windows(2).all(|w| w[0] < w[1]));
                true
            }
        }
    }

    /// Look up the offset for a given block index. If the index is
    /// within the confirmed-offsets vec, return that exact offset.
    /// Otherwise, return a GUESSED offset at the first spacing-aligned
    /// position past the last confirmed offset. Mirror of rapidgzip's
    /// `get(blockIndex, timeoutInSeconds)`
    /// (GzipBlockFinder.hpp:120-158); the timeout parameter is dropped
    /// (gzippy's get is non-blocking).
    pub fn get(&self, block_index: usize) -> (Option<usize>, GetReturnCode) {
        let g = self.inner.lock().unwrap();
        let block_offsets = &g.block_offsets;

        if block_index < block_offsets.len() {
            return (Some(block_offsets[block_index]), GetReturnCode::Success);
        }

        // Guess: take the first partition index whose offset is past
        // the last confirmed offset, then add (blockIndex - knownCount).
        // Mirror of GzipBlockFinder.hpp:134-157.
        debug_assert!(!block_offsets.is_empty());
        let block_index_outside = block_index - block_offsets.len();
        // firstPartitionIndex() (GzipBlockFinder.hpp:279-287):
        // last_confirmed / spacing + 1. Integer division rounds down,
        // so +1 gives the next strictly-greater multiple of spacing.
        let first_partition_index = block_offsets.last().unwrap() / self.spacing_in_bits + 1;
        let partition_index = first_partition_index + block_index_outside;
        let block_offset = partition_index.saturating_mul(self.spacing_in_bits);

        if let Some(file_size) = self.file_size_in_bits {
            if block_offset >= file_size {
                // GzipBlockFinder.hpp:148-154 — return file_size as the
                // failure-offset so the caller can use it as `until`.
                if partition_index > 0 {
                    return (Some(file_size), GetReturnCode::Failure);
                }
                return (Some(0), GetReturnCode::Failure);
            }
        }
        (Some(block_offset), GetReturnCode::Success)
    }

    /// Return the block index for a given encoded offset, or None if
    /// not found. Mirror of rapidgzip's `find()`
    /// (GzipBlockFinder.hpp:163-186) which `throw std::out_of_range`;
    /// we return Option for the more idiomatic Rust caller contract.
    pub fn find(&self, encoded_block_offset_in_bits: usize) -> Option<usize> {
        let g = self.inner.lock().unwrap();
        let block_offsets = &g.block_offsets;
        match block_offsets.binary_search(&encoded_block_offset_in_bits) {
            Ok(idx) => Some(idx),
            Err(_) => {
                // Past-end + spacing-aligned guess support. Mirror of
                // GzipBlockFinder.hpp:174-182.
                if let Some(&last) = block_offsets.last() {
                    if encoded_block_offset_in_bits > last
                        && encoded_block_offset_in_bits.is_multiple_of(self.spacing_in_bits)
                    {
                        let first_partition_index = last / self.spacing_in_bits + 1;
                        let blocks_past_known = encoded_block_offset_in_bits / self.spacing_in_bits
                            - first_partition_index;
                        return Some(block_offsets.len() + blocks_past_known);
                    }
                }
                None
            }
        }
    }

    /// Round `block_offset` down to the spacing grid. Mirror of
    /// `partitionOffsetContainingOffset` (GzipBlockFinder.hpp:195-200).
    pub fn partition_offset_containing_offset(&self, block_offset_in_bits: usize) -> usize {
        (block_offset_in_bits / self.spacing_in_bits) * self.spacing_in_bits
    }

    /// Snapshot of all confirmed offsets (for diagnostics / tests).
    pub fn confirmed_offsets(&self) -> Vec<usize> {
        self.inner.lock().unwrap().block_offsets.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPACING_BYTES: usize = 4 * 1024 * 1024; // 4 MiB — rapidgzip's default

    #[test]
    fn new_holds_first_offset() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        assert_eq!(f.size(), 1);
        assert_eq!(f.confirmed_offsets(), vec![0]);
    }

    #[test]
    #[should_panic(expected = "spacing")]
    fn new_rejects_subwindow_spacing() {
        let _ = GzipBlockFinder::new(0, 16 * 1024, None); // < 32 KiB
    }

    #[test]
    fn insert_returns_true_for_novel_and_duplicate() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        assert!(f.insert(SPACING_BYTES * 8));
        assert!(f.insert(SPACING_BYTES * 8)); // duplicate
        assert_eq!(f.size(), 2);
    }

    #[test]
    fn insert_returns_false_past_eof() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, Some(1000));
        assert!(!f.insert(1000));
        assert!(!f.insert(2000));
        assert!(f.insert(500));
    }

    #[test]
    #[should_panic(expected = "finalized")]
    fn insert_after_finalize_panics_on_novel() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        f.finalize();
        f.insert(SPACING_BYTES * 8);
    }

    #[test]
    fn insert_after_finalize_ok_on_duplicate() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        f.finalize();
        // Already present — should NOT panic.
        assert!(f.insert(0));
    }

    #[test]
    fn get_returns_confirmed_offset_for_known_index() {
        let f = GzipBlockFinder::new(8, SPACING_BYTES, None);
        f.insert(SPACING_BYTES * 8);
        assert_eq!(f.get(0), (Some(8), GetReturnCode::Success));
        assert_eq!(f.get(1), (Some(SPACING_BYTES * 8), GetReturnCode::Success));
    }

    #[test]
    fn get_returns_guessed_offset_past_known() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        // Only confirmed: [0]. get(1) guesses spacing*1.
        // firstPartitionIndex = 0 / spacing + 1 = 1.
        // partition_index = 1 + (1 - 1) = 1.
        // block_offset = 1 * spacing_bits.
        assert_eq!(f.get(1), (Some(SPACING_BYTES * 8), GetReturnCode::Success));
        assert_eq!(
            f.get(2),
            (Some(2 * SPACING_BYTES * 8), GetReturnCode::Success)
        );
    }

    #[test]
    fn get_returns_failure_past_eof() {
        let file_size_bits = 5 * SPACING_BYTES * 8;
        let f = GzipBlockFinder::new(0, SPACING_BYTES, Some(file_size_bits));
        // Indexes 1..4 are valid guesses; index 5 = 5 * spacing = file_size.
        assert_eq!(f.get(5), (Some(file_size_bits), GetReturnCode::Failure));
    }

    #[test]
    fn find_locates_confirmed_offset() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        f.insert(SPACING_BYTES * 8);
        f.insert(2 * SPACING_BYTES * 8);
        assert_eq!(f.find(0), Some(0));
        assert_eq!(f.find(SPACING_BYTES * 8), Some(1));
        assert_eq!(f.find(2 * SPACING_BYTES * 8), Some(2));
    }

    #[test]
    fn find_returns_none_for_unaligned_unknown() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        // 12345 is neither confirmed nor spacing-aligned.
        assert_eq!(f.find(12345), None);
    }

    #[test]
    fn find_returns_index_for_spacing_aligned_guess() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        // Known: [0]. Spacing-aligned guess: spacing*3.
        // firstPartitionIndex = 1. blocks_past_known = 3 - 1 = 2.
        // Index = 1 + 2 = 3.
        assert_eq!(f.find(3 * SPACING_BYTES * 8), Some(3));
    }

    #[test]
    fn partition_offset_containing_offset_rounds_down() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        let spacing = SPACING_BYTES * 8;
        assert_eq!(f.partition_offset_containing_offset(0), 0);
        assert_eq!(f.partition_offset_containing_offset(spacing - 1), 0);
        assert_eq!(f.partition_offset_containing_offset(spacing), spacing);
        assert_eq!(
            f.partition_offset_containing_offset(spacing + 12345),
            spacing
        );
    }

    #[test]
    fn insert_keeps_offsets_sorted() {
        let f = GzipBlockFinder::new(0, SPACING_BYTES, None);
        f.insert(3 * SPACING_BYTES * 8);
        f.insert(SPACING_BYTES * 8);
        f.insert(2 * SPACING_BYTES * 8);
        let offsets = f.confirmed_offsets();
        let mut sorted = offsets.clone();
        sorted.sort();
        assert_eq!(offsets, sorted);
    }
}
