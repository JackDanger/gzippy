//! Literal port of `rapidgzip::BitStringFinder`
//! (vendor/rapidgzip/librapidarchive/src/core/BitStringFinder.hpp:33-321).
//!
//! From the vendor header doc (BitStringFinder.hpp:27-32):
//! > No matter the input, the data is read from an input buffer.
//! > If a file is given, then that input buffer will be refilled when the
//! > input buffer empties.
//! > It is less a file object and acts more like an iterator.
//! > It offers a `find` method returning the next match or
//! > `std::numeric_limits<size_t>::max()` if the end was reached.
//!
//! In rapidgzip this is the engine that locates bit-aligned magic patterns
//! across compressed streams — used by `BgzfBlockFinder` (gzip magic), by
//! `ParallelBitStringFinder` (bzip2 magic across chunk boundaries), and by
//! the multi-member gzip pre-scan. Porting it unblocks the Bzip2Chunk port
//! (which sits on `ParallelBitStringFinder<48>` for the bzip2 block magic
//! 0x314159265359) and gives gzippy a faithful reference implementation for
//! the gzip-magic case too.
//!
//! Mapping rapidgzip -> Rust
//! -------------------------
//! - `template<uint8_t bitStringSize> class BitStringFinder`
//!   (BitStringFinder.hpp:33-34) -> runtime `bit_string_size: u8` field. C++
//!   forces a compile-time bit width because the template specializes the
//!   shift LUT layout; Rust const generics could mirror that, but every
//!   call site we will port in this cutover (bgzf 32-bit, bzip2 48-bit) is
//!   known at construction time and a runtime parameter keeps the tests
//!   trivial. Switching to const generics later is a no-op refactor.
//! - `m_bitStringToFind & nLowestBitsSet<uint64_t>(bitStringSize)`
//!   (BitStringFinder.hpp:53) -> `bit_string & low_bits_mask(bit_string_size)`.
//! - `m_movingBitsToKeep = bitStringSize > 0 ? bitStringSize - 1 : 0`
//!   (BitStringFinder.hpp:54) -> same.
//! - `m_movingBytesToKeep = ceilDiv(m_movingBitsToKeep, CHAR_BIT)`
//!   (BitStringFinder.hpp:55) -> `ceil_div`.
//! - `UniqueFileReader m_fileReader` (BitStringFinder.hpp:146) -> deferred
//!   to the FileReader port. This commit lands only the in-memory
//!   constructor (BitStringFinder.hpp:71-77), which the vendor explicitly
//!   documents as `/** @note This overload is used for the tests but can
//!   also be useful for other things. */`.
//! - `findBitStrings(string_view, uint64_t)` static helper
//!   (BitStringFinder.hpp:118-238) -> [`find_bit_strings`] free function.
//!   This is the core algorithm — every shift in 0..8, locate the
//!   byte-aligned middle chunk via memmem, then validate the leading and
//!   trailing partial bytes.
//! - `find()` iterator (BitStringFinder.hpp:101-285) -> [`BitStringFinder::find`]
//!   on the in-memory variant; pops from a descending-sorted offset stack
//!   so successive calls walk the file in ascending bit-offset order.
//!
//! Restriction on bit_string_size
//! -----------------------------
//! The vendor's `findBitStrings` carries a static_assert
//! (BitStringFinder.hpp:163-164):
//! > static_assert(
//! >     (bitStringSize >= CHAR_BIT) && (bitStringSize % CHAR_BIT == 0),
//! >     "This is a highly optimized bit string finder for bzip2 magic bytes."
//! > );
//!
//! We mirror that as a runtime requirement: `find_bit_strings` requires
//! `bit_string_size >= 8 && bit_string_size % 8 == 0`. The `BitStringFinder`
//! constructor enforces the same. Both bgzf (gzip magic, 32 bits) and bzip2
//! (block magic, 48 bits) satisfy this.

#![allow(dead_code)]

/// Sentinel returned by `find` when the end of the buffer is reached.
/// Mirror of `std::numeric_limits<size_t>::max()` (BitStringFinder.hpp:101 /
/// BitStringFinder.hpp:284).
pub const NOT_FOUND: usize = usize::MAX;

/// `ceilDiv(dividend, divisor)` (common.hpp:81-85).
#[inline]
fn ceil_div(dividend: usize, divisor: usize) -> usize {
    dividend.div_ceil(divisor)
}

/// `nLowestBitsSet<uint64_t>(nBitsSet)` (BitManipulation.hpp:60-73).
#[inline]
fn low_bits_mask(n_bits: u8) -> u64 {
    if n_bits == 0 {
        0
    } else if n_bits as u32 >= u64::BITS {
        u64::MAX
    } else {
        (1u64 << n_bits) - 1
    }
}

/// Locate all occurrences of `bit_string` (lowest `bit_string_size` bits) in
/// `buffer`, returning bit-offsets (LSB-first relative to the start of
/// `buffer`).
///
/// Mirror of `BitStringFinder<bitStringSize>::findBitStrings`
/// (BitStringFinder.hpp:159-238).
///
/// # Algorithm (mirroring lines 165-236)
///
/// For each `shift in 0..8`, take the byte-aligned middle slice of length
/// `bit_string_size / 8 - 1` from `bit_string >> shift`, find every
/// occurrence of that middle in `buffer` (vendor uses
/// `std::string_view::find`; we use [`memchr::memmem`]'s `find_iter`), then
/// validate the partial byte before (`headMatches`) and after
/// (`tailMatches`) the match.
///
/// # Panics
///
/// Panics if `bit_string_size < 8 || bit_string_size % 8 != 0`. Mirror of
/// the `static_assert` at BitStringFinder.hpp:163-164.
pub fn find_bit_strings(buffer: &[u8], bit_string: u64, bit_string_size: u8) -> Vec<usize> {
    assert!(
        bit_string_size >= 8 && bit_string_size.is_multiple_of(8),
        "find_bit_strings requires bit_string_size in {{8, 16, 24, ..., 64}}; \
         mirrors BitStringFinder.hpp:163-164 static_assert"
    );

    let mut block_offsets: Vec<usize> = Vec::new();

    // The vendor closure msbToString (BitStringFinder.hpp:180-191) renders
    // the top `(bitStringSize - CHAR_BIT)` bits of `bitString >> shift` to
    // a big-endian byte string. The middle slice length is
    // `bit_string_size/8 - 1` bytes.
    let middle_bytes = (bit_string_size / 8 - 1) as usize;

    for shift in 0u32..8 {
        // `bitString >> shift` with the top `bit_string_size - 8` bits taken
        // MSB-first.
        let shifted = bit_string >> shift;
        let middle_size_bits = bit_string_size - 8;

        let mut needle = vec![0u8; middle_bytes];
        let mut remaining = middle_size_bits;
        for byte in needle.iter_mut() {
            remaining -= 8;
            *byte = ((shifted >> remaining) & 0xFF) as u8;
        }

        // `findStrings` (BitStringFinder.hpp:166-178) — every occurrence of
        // the middle byte string. memchr::memmem matches std::string_view::find
        // semantics: overlapping matches enumerated one-past-start at a time.
        let n_bits_after = shift; // BitStringFinder.hpp:226
        let n_bits_before = 8 - shift as u8; // BitStringFinder.hpp:227

        let head_matches = |offset: usize| -> bool {
            // BitStringFinder.hpp:194-205.
            if n_bits_before == 0 {
                return true;
            }
            if offset == 0 || offset > buffer.len() {
                return false;
            }
            let mask = low_bits_mask(n_bits_before);
            let lhs = u64::from(buffer[offset - 1]) & mask;
            let rhs = (bit_string >> (bit_string_size - n_bits_before)) & mask;
            lhs == rhs
        };

        let tail_matches = |offset: usize| -> bool {
            // BitStringFinder.hpp:208-219.
            if n_bits_after == 0 {
                return true;
            }
            if offset >= buffer.len() {
                return false;
            }
            let mask = low_bits_mask(n_bits_after as u8);
            let lhs = u64::from(buffer[offset]) >> (8 - n_bits_after);
            let rhs = bit_string & mask;
            lhs == rhs
        };

        // Iterate every match position. The vendor uses
        // `data.find(stringToFind, position + 1U)` — that gives
        // non-overlapping positions advanced by one byte at a time.
        // memmem::find_iter is the matching primitive.
        if middle_bytes == 0 {
            // bit_string_size == 8: there is no middle, every byte position
            // is a candidate. The vendor still iterates positions 0..size().
            for pos in 0..buffer.len() {
                if head_matches(pos) && tail_matches(pos + middle_bytes) {
                    // bit_offset = pos*8 - n_bits_before; underflow at
                    // pos==0 would have been rejected by head_matches.
                    let offset = pos * 8 - n_bits_before as usize;
                    block_offsets.push(offset);
                }
            }
        } else {
            for pos in memchr::memmem::find_iter(buffer, &needle) {
                if head_matches(pos) && tail_matches(pos + middle_bytes) {
                    let offset = pos * 8 - n_bits_before as usize;
                    block_offsets.push(offset);
                }
            }
        }
    }

    block_offsets
}

/// In-memory iterator over bit-aligned magic-pattern matches.
///
/// Mirror of `BitStringFinder` (BitStringFinder.hpp:33-155), restricted to
/// the in-memory constructor at lines 71-77. The file-backed constructor
/// (lines 50-66) requires a `FileReader` trait we have not yet ported.
pub struct BitStringFinder {
    /// `m_bitStringToFind` (BitStringFinder.hpp:123). Already masked to the
    /// lowest `bit_string_size` bits at construction time.
    bit_string_to_find: u64,
    /// `bitStringSize` template parameter (BitStringFinder.hpp:34).
    bit_string_size: u8,
    /// `m_buffer` (BitStringFinder.hpp:138). We do not refill from a file
    /// in this port; the in-memory variant is fixed-size.
    buffer: Vec<u8>,
    /// `m_offsetsInBuffer` (BitStringFinder.hpp:139). Sorted descending so
    /// that `find()` can pop from the back in ascending bit-offset order
    /// (BitStringFinder.hpp:273-274).
    offsets_in_buffer: Vec<usize>,
    /// `m_bufferBitsRead` (BitStringFinder.hpp:144). Once we have scanned
    /// the buffer, this is set to `buffer.len() * 8` so that subsequent
    /// `find()` calls do not rescan (BitStringFinder.hpp:276). The
    /// in-memory variant cannot grow.
    buffer_bits_read: usize,
    /// `m_nTotalBytesRead` (BitStringFinder.hpp:154). Always 0 for the
    /// in-memory variant since the buffer is never refilled. Kept here
    /// only to make the structural correspondence to the vendor obvious.
    n_total_bytes_read: usize,
    /// First-call flag: `find()` must invoke `findBitStrings` once on the
    /// in-memory buffer before consuming matches.
    scanned: bool,
}

impl BitStringFinder {
    /// In-memory constructor. Mirror of the test-overload at
    /// BitStringFinder.hpp:71-77:
    /// ```c++
    /// BitStringFinder(const char* buffer, size_t size, uint64_t bitStringToFind)
    ///     : BitStringFinder(UniqueFileReader(), bitStringToFind)
    /// { m_buffer.assign(buffer, buffer + size); }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `bit_string_size < 8 || bit_string_size % 8 != 0` (the
    /// vendor static_assert at BitStringFinder.hpp:163-164 plus the
    /// implicit precondition that `find()` calls `findBitStrings`).
    pub fn new_in_memory(buffer: Vec<u8>, bit_string_to_find: u64, bit_string_size: u8) -> Self {
        assert!(
            bit_string_size >= 8 && bit_string_size.is_multiple_of(8),
            "BitStringFinder requires bit_string_size in {{8,16,...,64}}; mirrors BitStringFinder.hpp:163-164"
        );
        let masked = bit_string_to_find & low_bits_mask(bit_string_size);
        Self {
            bit_string_to_find: masked,
            bit_string_size,
            buffer,
            offsets_in_buffer: Vec::new(),
            buffer_bits_read: 0,
            n_total_bytes_read: 0,
            scanned: false,
        }
    }

    /// `bool seekable() const` (BitStringFinder.hpp:82-87). For the
    /// in-memory variant this is always true (vendor comment: "If
    /// m_fileReader is not set, then we are working on an in-memory
    /// buffer, which can be seeked.").
    pub fn seekable(&self) -> bool {
        true
    }

    /// `bool eof() const` (BitStringFinder.hpp:89-96). In-memory branch:
    /// "return m_buffer.empty();" — but a more useful semantics is
    /// "everything scanned and no pending offsets", which is what the
    /// vendor's `find()` actually consults via the `while (!eof())` loop
    /// after `findBitStrings` has been called.
    pub fn eof(&self) -> bool {
        self.buffer.is_empty() || (self.scanned && self.offsets_in_buffer.is_empty())
    }

    /// `size_t find()` (BitStringFinder.hpp:101-102, 242-285).
    ///
    /// Returns the next bit-offset of a match, or [`NOT_FOUND`] when no
    /// further matches remain.
    pub fn find(&mut self) -> usize {
        // BitStringFinder.hpp:245-247: bitStringSize == 0 means "never
        // matches"; our static check forbids the case but mirror the
        // vendor early-exit shape.
        if self.bit_string_size == 0 {
            return NOT_FOUND;
        }

        // BitStringFinder.hpp:249-253: if we have already-scanned offsets,
        // pop one and return.
        if let Some(off) = self.offsets_in_buffer.pop() {
            return self.n_total_bytes_read * 8 + off;
        }

        // The vendor's outer `while (!eof())` would refill on `bufferEof`;
        // the in-memory variant has nothing to refill. Scan once.
        if !self.scanned && !self.buffer.is_empty() {
            self.scanned = true;
            let mut offsets =
                find_bit_strings(&self.buffer, self.bit_string_to_find, self.bit_string_size);

            // BitStringFinder.hpp:266-271: discard offsets that fall
            // inside the prefix already consumed by an earlier `find()`
            // run. For the in-memory variant `m_bufferBitsRead` starts at
            // 0, so this is a no-op the first time. Mirror it anyway so
            // refactors that add `seek` semantics stay correct.
            let first_bits_to_ignore = self.buffer_bits_read % 8;
            offsets.retain(|&off| off >= first_bits_to_ignore);

            // BitStringFinder.hpp:274: sort descending so we can pop from
            // the back to deliver ascending offsets.
            offsets.sort_unstable_by(|a, b| b.cmp(a));

            self.offsets_in_buffer = offsets;
            self.buffer_bits_read = self.buffer.len() * 8;

            if let Some(off) = self.offsets_in_buffer.pop() {
                return self.n_total_bytes_read * 8 + off;
            }
        }

        NOT_FOUND
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a buffer with `magic` packed at the given bit offset,
    /// LSB-first inside each byte (matching the vendor convention).
    fn pack_magic_at(bit_offset: usize, magic: u64, magic_size: u8, total_size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; total_size];
        for i in 0..(magic_size as usize) {
            // The vendor stores bit_string MSB-first within bytes (see
            // findBitStrings msbToString lambda BitStringFinder.hpp:180-191:
            // it shifts the top bits off first).
            let bit_value = ((magic >> (magic_size as usize - 1 - i)) & 1) as u8;
            let abs_bit = bit_offset + i;
            let byte_idx = abs_bit / 8;
            let bit_idx_in_byte = 7 - (abs_bit % 8);
            buf[byte_idx] |= bit_value << bit_idx_in_byte;
        }
        buf
    }

    #[test]
    fn finds_byte_aligned_gzip_magic() {
        // Gzip magic: 0x1f 0x8b — the most common 16-bit pattern this
        // primitive is used for in the broader rapidgzip code. Place it
        // at byte 0 and again at byte 100.
        let mut buf = vec![0u8; 256];
        buf[0] = 0x1f;
        buf[1] = 0x8b;
        buf[100] = 0x1f;
        buf[101] = 0x8b;

        let mut finder = BitStringFinder::new_in_memory(buf, 0x1f8b, 16);
        let mut hits = Vec::new();
        loop {
            let off = finder.find();
            if off == NOT_FOUND {
                break;
            }
            hits.push(off);
        }
        // Bit-offset 0 (byte 0) and bit-offset 800 (byte 100). Ascending
        // order is the vendor guarantee.
        assert_eq!(hits, vec![0, 800]);
    }

    #[test]
    fn finds_bit_aligned_bzip2_magic_at_known_offset() {
        // The bzip2 block magic is 48 bits: 0x314159265359 (BCD of pi).
        // Place it at bit-offset 3 inside a 16-byte buffer.
        let magic: u64 = 0x314159265359;
        let buf = pack_magic_at(3, magic, 48, 16);

        let mut finder = BitStringFinder::new_in_memory(buf, magic, 48);
        let mut hits = Vec::new();
        loop {
            let off = finder.find();
            if off == NOT_FOUND {
                break;
            }
            hits.push(off);
        }
        assert!(
            hits.contains(&3),
            "expected bzip2 magic at bit-offset 3; got {hits:?}"
        );
    }

    #[test]
    fn finds_multiple_bit_aligned_occurrences_in_ascending_order() {
        let magic: u64 = 0xFACECAFE;
        // Place the 32-bit magic at three different bit-offsets.
        let mut buf = pack_magic_at(0, magic, 32, 64);
        // OR in two more copies — pack_magic_at on a zero buffer wouldn't
        // hurt them, but to keep them independent we use distinct slots.
        let copy_b = pack_magic_at(35, magic, 32, 64);
        let copy_c = pack_magic_at(120, magic, 32, 64);
        for (i, b) in buf.iter_mut().enumerate() {
            *b |= copy_b[i] | copy_c[i];
        }

        let mut finder = BitStringFinder::new_in_memory(buf, magic, 32);
        let mut hits = Vec::new();
        loop {
            let off = finder.find();
            if off == NOT_FOUND {
                break;
            }
            hits.push(off);
        }
        assert!(hits.contains(&0));
        assert!(hits.contains(&35));
        assert!(hits.contains(&120));
        // Ascending order, no duplicates (sort_unstable_by descending then
        // pop guarantees ascending output).
        let mut sorted = hits.clone();
        sorted.sort();
        assert_eq!(hits, sorted);
    }

    #[test]
    fn returns_not_found_on_empty_buffer() {
        let mut finder = BitStringFinder::new_in_memory(Vec::new(), 0x1f8b, 16);
        assert_eq!(finder.find(), NOT_FOUND);
    }

    #[test]
    fn returns_not_found_when_magic_absent() {
        let buf = vec![0u8; 1024];
        let mut finder = BitStringFinder::new_in_memory(buf, 0x1f8b, 16);
        assert_eq!(finder.find(), NOT_FOUND);
    }

    #[test]
    fn find_bit_strings_static_helper_matches_class_method() {
        // Cross-check: the static helper and the class iterator must agree
        // on the offset set for any given input. This mirrors the vendor
        // contract where `find()` is built on top of `findBitStrings`.
        let magic: u64 = 0x1f8b08;
        let mut buf = pack_magic_at(7, magic, 24, 64);
        let copy_b = pack_magic_at(120, magic, 24, 64);
        for (i, b) in buf.iter_mut().enumerate() {
            *b |= copy_b[i];
        }

        let static_hits = find_bit_strings(&buf, magic, 24);
        let mut sorted_static = static_hits.clone();
        sorted_static.sort();

        let mut finder = BitStringFinder::new_in_memory(buf.clone(), magic, 24);
        let mut iter_hits = Vec::new();
        loop {
            let off = finder.find();
            if off == NOT_FOUND {
                break;
            }
            iter_hits.push(off);
        }
        assert_eq!(iter_hits, sorted_static);
    }

    #[test]
    #[should_panic(expected = "bit_string_size")]
    fn rejects_non_byte_aligned_bit_string_size() {
        // Vendor static_assert: bitStringSize >= 8 && bitStringSize % 8 == 0.
        let _ = find_bit_strings(&[0u8; 8], 0x55, 7);
    }
}
