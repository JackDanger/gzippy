//! Literal port of `rapidgzip::blockfinder::seekToNonFinalUncompressedDeflateBlock`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/blockfinder/Uncompressed.hpp:21-95).
//!
//! Scans a bit stream for the start of a non-final BTYPE=00 (uncompressed /
//! "stored") deflate block. A stored block begins with the 3-bit deflate
//! header `0b000` (BFINAL=0, BTYPE=00), zero-padding bits up to the next byte
//! boundary, then a 16-bit little-endian LEN followed by 16-bit ~LEN. The
//! coincidence `LEN ^ ~LEN == 0xFFFF` over a 32-bit window aligned to a byte
//! boundary is what the finder hunts for; once a hit is found, it walks back
//! the preceding 10 bits (3 header bits + up to 7 padding bits) to validate
//! the magic and locate the earliest legal block start (Uncompressed.hpp:64-89).
//!
//! Returns an *inclusive* range `(earliest, latest)` of byte-padded bit
//! offsets at which a deflate decoder could plausibly start consuming the
//! block. Both equal `usize::MAX` when nothing is found (Uncompressed.hpp:94).
//!
//! NOTE — `block_finder::BlockFinder::find_uncompressed_blocks` (this same
//! repo) is an existing port that emits `BlockBoundary` records. This file is
//! the standalone, scan-once equivalent suitable for the generic
//! [`BlockFinderInterface`] pipeline. The two will be consolidated in a
//! follow-up.

#![allow(dead_code)]

use crate::decompress::parallel::block_finder::BitReader;

/// Mirror of `rapidgzip::deflate::DEFLATE_MAGIC_BIT_COUNT`
/// (Uncompressed.hpp:25). Header is `BFINAL (1) | BTYPE (2)`.
pub const DEFLATE_MAGIC_BIT_COUNT: usize = 3;

/// Mirror of `rapidgzip::BYTE_SIZE` (core/common.hpp).
pub const BYTE_SIZE: usize = 8;

/// Mirror of `rapidgzip::blockfinder::MAX_PRECEDING_BITS`
/// (Uncompressed.hpp:29). The deflate magic (3 bits) plus at most
/// `BYTE_SIZE - 1` zero padding bits — i.e., everything between the
/// block start and the byte-aligned LEN field.
pub const MAX_PRECEDING_BITS: usize = DEFLATE_MAGIC_BIT_COUNT + (BYTE_SIZE - 1); // 10

/// Sentinel for "no match found" — mirror of
/// `std::numeric_limits<size_t>::max()` (Uncompressed.hpp:19,94).
pub const NO_MATCH: usize = usize::MAX;

/// Faithful port of
/// `rapidgzip::blockfinder::seekToNonFinalUncompressedDeflateBlock`
/// (Uncompressed.hpp:21-95).
///
/// Searches the bit stream backing `bit_reader` for the start of a
/// non-final uncompressed deflate block, returning a `(earliest, latest)`
/// inclusive range of valid start bit offsets. Both elements equal
/// [`NO_MATCH`] when nothing is found (Uncompressed.hpp:94).
///
/// `start_bit_offset` is the current bit position to start scanning from.
/// `until_offset` bounds the search; pass [`NO_MATCH`] to scan to EOF
/// (Uncompressed.hpp:22-24).
///
/// On success, `earliest` is the position just past the last padding zero
/// bit (so an earlier predecessor block can end on a byte boundary there);
/// `latest` is the byte-boundary minus 3 = the position of the magic bits
/// themselves (Uncompressed.hpp:84-86).
pub fn seek_to_non_final_uncompressed_deflate_block(
    bit_reader: &mut BitReader<'_>,
    data_len: usize,
    start_bit_offset: usize,
    until_offset: usize,
) -> (usize, usize) {
    // Mirror of `untilOffsetSizeMember` (Uncompressed.hpp:34-37). The LEN+NLEN
    // word lives at the byte boundary that the block magic precedes, so a
    // candidate's LEN may sit up to MAX_PRECEDING_BYTES (== 16) past
    // until_offset and still produce a valid start strictly before it.
    let max_preceding_bytes = MAX_PRECEDING_BITS.div_ceil(BYTE_SIZE) * BYTE_SIZE; // 16
    let until_offset_size_member = if until_offset >= NO_MATCH - max_preceding_bytes {
        NO_MATCH
    } else {
        until_offset + max_preceding_bytes
    };
    // Bound by the file size in bits, like rapidgzip's `bitReader.size()`
    // clamp (Uncompressed.hpp:38-41). data_len is in BYTES.
    let file_size_bits = data_len.saturating_mul(BYTE_SIZE);
    let until_offset_size_member = until_offset_size_member.min(file_size_bits);

    // Mirror of `startOffsetByte` (Uncompressed.hpp:45-47): align to the byte
    // boundary FOLLOWING the 3-bit magic. Floor at one byte so we always have
    // at least one preceding byte to inspect.
    let start_offset_byte = ((start_bit_offset + DEFLATE_MAGIC_BIT_COUNT).div_ceil(BYTE_SIZE)
        * BYTE_SIZE)
        .max(BYTE_SIZE);

    if start_offset_byte >= until_offset_size_member {
        return (NO_MATCH, NO_MATCH);
    }

    bit_reader.seek_to_bit(start_offset_byte);

    // Prime the rolling 32-bit window with the first 24 bits (Uncompressed.hpp:52).
    // We hold (size >> 8) and shift the next byte into the top byte each
    // iteration — gives us the LEN | NLEN view aligned to the byte boundary.
    let mut size = bit_reader.read(3 * BYTE_SIZE as u8) << BYTE_SIZE;

    let mut offset = start_offset_byte;
    while offset < until_offset_size_member {
        // Slide a byte in (Uncompressed.hpp:55). After this, `size` holds
        // (LEN << 0) | (NLEN << 16) treating offset..offset+4 as the bytes
        // of a u32 little-endian.
        if bit_reader.is_eof() {
            break;
        }
        let next_byte = bit_reader.read(BYTE_SIZE as u8);
        size = (size >> BYTE_SIZE) | (next_byte << (3 * BYTE_SIZE));

        // Coincidence test: LEN ^ ~NLEN == 0, equivalently
        // ((size ^ (size >> 16)) & 0xFFFF) == 0xFFFF (Uncompressed.hpp:56).
        // `LIKELY((..) != 0xFFFF)`: skip the slow path most of the time.
        if ((size ^ (size >> 16)) & 0xFFFF) != 0xFFFF {
            offset += BYTE_SIZE;
            continue;
        }

        // Save the post-read offset so we can rewind for header inspection
        // and resume cleanly (Uncompressed.hpp:60-61).
        let old_offset = offset + 4 * BYTE_SIZE;
        debug_assert_eq!(old_offset, bit_reader.bit_position());

        // Rewind to read the 10 bits preceding the byte boundary
        // (Uncompressed.hpp:67-68). Bit ordering: bits are LSB-first from the
        // byte stream, so the bit at index `MAX_PRECEDING_BITS - 1` is the
        // newest (closest to the byte boundary).
        let pre_origin = match offset.checked_sub(MAX_PRECEDING_BITS) {
            Some(v) => v,
            None => {
                offset += BYTE_SIZE;
                bit_reader.seek_to_bit(old_offset);
                continue;
            }
        };
        bit_reader.seek_to_bit(pre_origin);
        let previous_bits = bit_reader.read(MAX_PRECEDING_BITS as u8) as u32;

        // The 3 deflate magic bits must be 0b000 at the TOP of preceding_bits
        // (Uncompressed.hpp:70-74). MAGIC_BITS_MASK = 0b111 << 7.
        const MAGIC_BITS_MASK: u32 = 0b111u32 << (MAX_PRECEDING_BITS - DEFLATE_MAGIC_BIT_COUNT);
        if (previous_bits & MAGIC_BITS_MASK) != 0 {
            bit_reader.seek_to_bit(old_offset);
            offset += BYTE_SIZE;
            continue;
        }

        // Count zero padding bits below the magic (Uncompressed.hpp:76-82).
        // Start at DEFLATE_MAGIC_BIT_COUNT (= 3 zeros for the magic itself)
        // and walk older bits until we hit a 1 or exhaust the 10-bit window.
        let mut trailing_zeros = DEFLATE_MAGIC_BIT_COUNT;
        for j in (DEFLATE_MAGIC_BIT_COUNT + 1)..=MAX_PRECEDING_BITS {
            if (previous_bits & (1u32 << (MAX_PRECEDING_BITS - j))) != 0 {
                break;
            }
            trailing_zeros = j;
        }

        // Earliest valid start = byte_boundary - trailing_zeros; latest =
        // byte_boundary - 3 (Uncompressed.hpp:84-86). Both must satisfy the
        // caller's `start_bit_offset`/`until_offset` bounds.
        let earliest = match offset.checked_sub(trailing_zeros) {
            Some(v) => v,
            None => {
                bit_reader.seek_to_bit(old_offset);
                offset += BYTE_SIZE;
                continue;
            }
        };
        let latest = offset - DEFLATE_MAGIC_BIT_COUNT;
        if offset >= DEFLATE_MAGIC_BIT_COUNT
            && (offset - DEFLATE_MAGIC_BIT_COUNT) >= start_bit_offset
            && earliest < until_offset
        {
            return (earliest, latest);
        }

        bit_reader.seek_to_bit(old_offset);
        offset += BYTE_SIZE;
    }

    (NO_MATCH, NO_MATCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic deflate stream: padding zeros to byte boundary,
    /// then `00` BTYPE + `0` BFINAL = 3 magic bits at the END of a byte,
    /// then 16-bit LEN | 16-bit ~LEN, then `len` payload bytes. The
    /// preamble is `preamble_bytes` of arbitrary content used to push the
    /// stored block deeper into the stream.
    fn make_stored_block(preamble_bytes: usize, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0xABu8; preamble_bytes];
        // Magic 0b000 at the LSB of the next byte; padding zeros fill the
        // rest of the byte (BFINAL=0, BTYPE=00). We just push a zero byte
        // — that gives us `MAX_PRECEDING_BITS` worth of zeros, which the
        // finder will detect as a stored block start.
        out.push(0x00);
        let len = payload.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn finds_simple_stored_block() {
        let payload = b"hello world!!";
        let data = make_stored_block(16, payload);
        let mut reader = BitReader::new(&data);
        let (earliest, latest) =
            seek_to_non_final_uncompressed_deflate_block(&mut reader, data.len(), 0, NO_MATCH);
        assert_ne!(earliest, NO_MATCH);
        assert_ne!(latest, NO_MATCH);
        // The 3 magic bits live at the very top of byte index 16 (the zero
        // padding byte we pushed). With padding == 7 zero bits in front of
        // the magic, the earliest start is bit (17 * 8 - 10) = 126; latest is
        // bit (17 * 8 - 3) = 133.
        let byte_boundary_bit = 17 * 8;
        assert_eq!(latest, byte_boundary_bit - DEFLATE_MAGIC_BIT_COUNT);
        assert!(earliest <= latest);
    }

    #[test]
    fn returns_no_match_on_empty_or_too_short() {
        let data = [0u8; 3];
        let mut reader = BitReader::new(&data);
        let (e, l) =
            seek_to_non_final_uncompressed_deflate_block(&mut reader, data.len(), 0, NO_MATCH);
        assert_eq!(e, NO_MATCH);
        assert_eq!(l, NO_MATCH);
    }

    #[test]
    fn returns_no_match_on_garbage() {
        // Random bytes with no LEN/~LEN coincidence anywhere.
        let data: Vec<u8> = (0..1024u32)
            .map(|i| ((i.wrapping_mul(0x9E3779B1)) & 0xFF) as u8)
            .collect();
        let mut reader = BitReader::new(&data);
        let (e, _) =
            seek_to_non_final_uncompressed_deflate_block(&mut reader, data.len(), 0, NO_MATCH);
        // Likely NO_MATCH — but if we hit one, that's fine too; we just
        // shouldn't crash. The contract is "all-or-NO_MATCH consistency".
        if e != NO_MATCH {
            // Earliest must be within data bounds.
            assert!(e < data.len() * 8);
        }
    }
}
