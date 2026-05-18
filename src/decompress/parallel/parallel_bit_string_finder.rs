//! Literal port of `rapidgzip::ParallelBitStringFinder`
//! (vendor/rapidgzip/librapidarchive/src/core/ParallelBitStringFinder.hpp:34-266).
//!
//! Parallel version of the in-memory [`bit_string_finder::BitStringFinder`]
//! (commit a609079). Splits the input buffer into sub-chunks of size
//! `max(8 * bit_string_size, 4096)` bytes (or `ceilDiv(buffer.len(),
//! threads)`, whichever is larger), scans each sub-chunk with
//! [`bit_string_finder::find_bit_strings`] in parallel, and merges the
//! global-bit-offset results in chunk order — exactly mirroring the
//! "Idea" block at ParallelBitStringFinder.hpp:148-159.
//!
//! Difference from the vendor: the vendor streams from an arbitrary
//! [`FileReader`], refilling buffers with `BitStringFinder::refillBuffer`
//! (ParallelBitStringFinder.hpp:214). gzippy's port operates on an
//! already-resident in-memory buffer because that matches the only call
//! site we need (block-finder pre-scan on a memory-mapped gzip stream).
//! The cross-chunk overlap (`m_movingBitsToKeep`,
//! ParallelBitStringFinder.hpp:227-235) collapses to a constant
//! `bit_string_size - 1` bits straddling each sub-chunk boundary.
//!
//! Wiring: this lands standalone with unit tests. Threading the
//! production block-finder onto it is a follow-up.

#![allow(dead_code)]

use std::thread;

use crate::decompress::parallel::bit_string_finder::find_bit_strings;

/// Mirror of `std::numeric_limits<size_t>::max()` returned by
/// `BitStringFinder::find()` at EOF
/// (ParallelBitStringFinder.hpp:215-216).
pub const NO_MORE_MATCHES: usize = usize::MAX;

/// In-memory parallel finder. Mirror of
/// `ParallelBitStringFinder<bitStringSize>` (ParallelBitStringFinder.hpp:34-145)
/// in its test-overload constructor (ParallelBitStringFinder.hpp:56-64).
pub struct ParallelBitStringFinder {
    /// `m_buffer` (inherited from BitStringFinder.hpp:138).
    buffer: Vec<u8>,
    /// `m_bitStringToFind` (BitStringFinder.hpp:123). Already masked to
    /// the lowest `bit_string_size` bits.
    bit_string_to_find: u64,
    /// `bitStringSize` template parameter
    /// (ParallelBitStringFinder.hpp:34). Must be a multiple of 8 because
    /// the underlying [`find_bit_strings`] only accepts byte-aligned
    /// bit-string sizes.
    bit_string_size: u8,
    /// `m_threadPool` capacity equivalent
    /// (ParallelBitStringFinder.hpp:53). Number of parallel workers used
    /// when `find_all` is invoked.
    parallelization: usize,
}

impl ParallelBitStringFinder {
    /// Mirror of the test/in-memory constructor at
    /// ParallelBitStringFinder.hpp:56-64.
    ///
    /// # Panics
    ///
    /// Panics if `bit_string_size < 8` or `bit_string_size % 8 != 0`
    /// (mirror of `find_bit_strings`'s precondition; the underlying
    /// vendor `findBitStrings` carries the same static_assert at
    /// BitStringFinder.hpp:163-164).
    pub fn new_in_memory(
        buffer: Vec<u8>,
        bit_string_to_find: u64,
        bit_string_size: u8,
        parallelization: usize,
    ) -> Self {
        assert!(
            bit_string_size >= 8 && bit_string_size.is_multiple_of(8),
            "ParallelBitStringFinder requires bit_string_size in {{8,16,...,64}}; \
             mirrors BitStringFinder.hpp:163-164 static_assert"
        );
        assert!(parallelization >= 1, "parallelization must be >= 1");
        Self {
            buffer,
            bit_string_to_find: bit_string_to_find & low_bits_mask(bit_string_size),
            bit_string_size,
            parallelization,
        }
    }

    /// Mirror of the `chunkSize` helper (ParallelBitStringFinder.hpp:90-103).
    /// Used to decide both the minimum per-thread sub-chunk size and the
    /// rolling overlap.
    fn min_sub_chunk_size_bytes(&self) -> usize {
        // ParallelBitStringFinder.hpp:221:
        //   minSubChunkSizeInBytes = max(8 * bitStringSize, 4096)
        // Vendor uses 8*bitStringSize but `bit_string_size` here is BITS,
        // and the multiplication `8UL * bitStringSize` in C++ is bits-to-
        // bytes-of-context (the constant 8 is `CHAR_BIT`, not a magic
        // factor on the size). We match the literal expression.
        std::cmp::max(8usize.saturating_mul(self.bit_string_size as usize), 4096)
    }

    fn sub_chunk_stride_bytes(&self) -> usize {
        // ParallelBitStringFinder.hpp:222-223:
        //   subChunkStrideInBytes = max(minSubChunkSizeInBytes,
        //                               ceilDiv(buffer.size(), threadPool.capacity()))
        let ceil_div = self.buffer.len().div_ceil(self.parallelization);
        std::cmp::max(self.min_sub_chunk_size_bytes(), ceil_div)
    }

    /// One-shot variant: scan the entire buffer in parallel and return
    /// **all** bit-offsets at which `bit_string_to_find` (of width
    /// `bit_string_size` bits) appears, in ascending order.
    ///
    /// Mirror of repeated `find()` calls until
    /// `std::numeric_limits<size_t>::max()` is returned
    /// (ParallelBitStringFinder.hpp:160-266). We collect into a `Vec`
    /// because gzippy's call sites (`block_finder` pre-scan) all
    /// materialize the offsets anyway.
    pub fn find_all(&self) -> Vec<usize> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        if self.parallelization <= 1 {
            // ParallelBitStringFinder.hpp:220-221 short-circuits to a
            // single-threaded scan when sub-chunks would be tiny. We
            // mirror the structural fallback. The vendor's `workerMain`
            // sorts before pushing (ParallelBitStringFinder.hpp:125-127),
            // so we sort + dedup here for the same observable output.
            let mut out =
                find_bit_strings(&self.buffer, self.bit_string_to_find, self.bit_string_size);
            out.sort_unstable();
            out.dedup();
            return out;
        }

        let stride = self.sub_chunk_stride_bytes();
        // Overlap: `m_movingBitsToKeep == bit_string_size - 1` bits,
        // i.e. `ceil_div(bit_string_size - 1, 8)` bytes; for a size
        // multiple of 8 we keep `bit_string_size / 8 - 1` bytes
        // (ParallelBitStringFinder.hpp:227-235).
        let overlap_bytes = (self.bit_string_size as usize).div_ceil(8) - 1;

        // Build chunk descriptors (start_byte, length).
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut cursor = 0usize;
        while cursor < self.buffer.len() {
            let start = if cursor == 0 {
                0
            } else {
                cursor.saturating_sub(overlap_bytes)
            };
            let end = (cursor + stride).min(self.buffer.len());
            let len = end - start;
            chunks.push((start, len));
            cursor = end;
        }

        // Mirror of the worker fan-out
        // (ParallelBitStringFinder.hpp:248-262). One scoped thread per
        // sub-chunk; results merged in chunk order.
        let bit_string_to_find = self.bit_string_to_find;
        let bit_string_size = self.bit_string_size;

        let collected = thread::scope(|s| {
            let mut handles = Vec::with_capacity(chunks.len());
            for &(start, len) in &chunks {
                let slice = &self.buffer[start..start + len];
                let handle = s.spawn(move || {
                    let mut local = find_bit_strings(slice, bit_string_to_find, bit_string_size);
                    // Worker sorts before pushing — mirror of
                    // workerMain at ParallelBitStringFinder.hpp:125-127.
                    local.sort_unstable();
                    (start, local)
                });
                handles.push(handle);
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("worker panicked"))
                .collect::<Vec<_>>()
        });

        // Merge per-chunk results into the global bit-offset space and
        // dedup the overlap. Each chunk straddles `overlap_bytes` of the
        // previous chunk's tail (so cross-boundary matches are found),
        // which means each cross-boundary offset shows up in both
        // chunks' result lists. A sort + dedup is the simplest faithful
        // implementation of the vendor's `firstBitsToIgnore` filter
        // (ParallelBitStringFinder.hpp:128-133); the vendor manages the
        // de-dup via the `firstBitsToIgnore` parameter, but here we
        // collect-then-dedup because all sub-chunks are scanned in one
        // batch rather than streamed.
        let mut merged: Vec<usize> = Vec::new();
        for (start_byte, mut local) in collected {
            for off in local.iter_mut() {
                *off += start_byte * 8;
            }
            merged.append(&mut local);
        }
        merged.sort_unstable();
        merged.dedup();
        merged
    }
}

/// Mirror of `lowBitsMask` used throughout BitStringFinder.hpp.
fn low_bits_mask(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::bit_string_finder::find_bit_strings;

    /// Parallel scan must agree with serial scan on a 1 MiB random buffer.
    #[test]
    fn matches_serial_on_random() {
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut state: u32 = 0xCAFEBABE;
        for b in buffer.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (state >> 16) as u8;
        }
        // Look for a 16-bit pattern likely to appear several times.
        let pattern: u64 = 0x1F8B;
        let mut serial = find_bit_strings(&buffer, pattern, 16);
        // find_bit_strings emits matches in the order of an outer loop
        // over byte-shift; sort to compare with the parallel finder
        // which globally sorts results.
        serial.sort_unstable();
        serial.dedup();
        let pf = ParallelBitStringFinder::new_in_memory(buffer.clone(), pattern, 16, 4);
        let parallel = pf.find_all();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn empty_buffer_no_matches() {
        let pf = ParallelBitStringFinder::new_in_memory(Vec::new(), 0x1F8B, 16, 4);
        assert!(pf.find_all().is_empty());
    }

    /// 24-bit gzip magic across a chunk boundary must still be found.
    #[test]
    fn finds_match_across_chunk_boundary() {
        // Pad with random; embed 24-bit pattern 0x1F8B08 at byte 4095 so
        // it straddles a likely sub-chunk boundary.
        let mut buffer = vec![0xCDu8; 8192];
        buffer[4095] = 0x1F;
        buffer[4096] = 0x8B;
        buffer[4097] = 0x08;
        let pattern: u64 = 0x1F8B08;
        // Force smallest possible sub-chunks by maxing parallelization,
        // but bit_string_size = 24 means min_sub_chunk = max(192, 4096)
        // = 4096; with stride 4096, the second chunk overlaps 2 bytes
        // (= 24/8 - 1) before its nominal start, covering 4094..8192.
        let pf = ParallelBitStringFinder::new_in_memory(buffer.clone(), pattern, 24, 2);
        let parallel = pf.find_all();
        let mut serial = find_bit_strings(&buffer, pattern, 24);
        serial.sort_unstable();
        serial.dedup();
        assert_eq!(parallel, serial);
        assert!(
            !parallel.is_empty(),
            "expected to find the embedded pattern"
        );
    }

    #[test]
    fn parallelization_one_equals_serial() {
        let mut buffer = vec![0u8; 4096];
        for (i, b) in buffer.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let pattern: u64 = 0xAB;
        let mut serial = find_bit_strings(&buffer, pattern, 8);
        serial.sort_unstable();
        serial.dedup();
        let pf = ParallelBitStringFinder::new_in_memory(buffer.clone(), pattern, 8, 1);
        assert_eq!(serial, pf.find_all());
    }
}
