//! High-Performance Deflate Block Finder
//!
//! Implements rapidgzip's approach to finding valid deflate block boundaries:
//! 1. 15-bit LUT for quick invalid position skipping
//! 2. Precode validation via leaf counting
//! 3. Full Huffman table validation
//! 4. Symbol 256 (END_OF_BLOCK) must have non-zero length

#![allow(clippy::unusual_byte_groupings)]
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Constants
// ============================================================================

/// Maximum precode length (7 bits)
const MAX_PRECODE_LENGTH: u8 = 7;

/// Number of precode symbols
const MAX_PRECODE_COUNT: usize = 19;

/// Bits per precode length
const PRECODE_BITS: u8 = 3;

/// Precode alphabet order (RFC 1951)
const PRECODE_ALPHABET: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// END_OF_BLOCK symbol
const END_OF_BLOCK_SYMBOL: usize = 256;

/// LUT size in bits (use 13 bits to keep compile time reasonable)
const LUT_BITS: usize = 13;
const LUT_SIZE: usize = 1 << LUT_BITS;

// ============================================================================
// 13-bit LUT for block candidate detection
// ============================================================================

/// Check if bits could be a valid deflate block start (stored or dynamic).
/// Accepts bfinal=0 or bfinal=1. BTYPE=01 (fixed Huffman) is excluded
/// because the 3-bit header alone matches ~25% of random positions —
/// emitting that many candidates pollutes the result list and makes
/// downstream consumers slow.
///
/// BTYPE=01 boundary detection lives instead in
/// `single_member::search_boundary_forward` as a separate tier that
/// runs only when the other tiers find nothing. That keeps BlockFinder
/// fast for the common path (where boundaries are BTYPE=00 / BTYPE=10)
/// and avoids the candidate explosion observed in
/// `test_find_blocks_parallel_matches_sequential`.
#[inline]
fn is_valid_candidate_13(bits: u32) -> bool {
    // Literal port of rapidgzip's `isDeflateCandidate<13>` at
    // vendor/rapidgzip/.../blockfinder/DynamicHuffman.hpp:39-79.
    //
    // Filters:
    //   bit 0       = bfinal, must be 0 (skip final blocks)
    //   bits 1-2    = btype, must be 0b10 (dynamic Huffman ONLY).
    //                 Stored blocks (BTYPE=00) are NOT emitted here —
    //                 rapidgzip uses a separate seekToNonFinalUncompressedDeflateBlock
    //                 for those. Mixing dynamic + stored candidates lets
    //                 stored phantoms beat dynamic naturals at find_first_candidate.
    //   bits 3-7    = HLIT (5 bits), value+257 must be ≤ 286 → HLIT ≤ 29
    //   bits 8-12   = HDIST (5 bits), value+1 must be ≤ 30 → HDIST ≤ 29
    //
    // The full lit/dist Huffman validation happens later in
    // validate_huffman_codes (called by find_blocks for accepted
    // candidates), matching rapidgzip's seekToNonFinalDynamicDeflateBlock
    // post-LUT validation.
    let bfinal = bits & 1;
    if bfinal == 1 {
        return false;
    }
    let btype = (bits >> 1) & 3;
    if btype != 2 {
        return false;
    }
    let hlit = (bits >> 3) & 31;
    let hdist = (bits >> 8) & 31;
    hlit <= 29 && hdist <= 29
}

/// Generate the LUT at runtime (called once via lazy_static pattern)
fn generate_deflate_lut() -> Vec<i8> {
    let mut lut = vec![0i8; LUT_SIZE];

    for i in 0..LUT_SIZE {
        // Simple approach: check if valid, skip 1 if not
        if is_valid_candidate_13(i as u32) {
            lut[i] = 0; // Valid candidate
        } else {
            // Skip forward until we find a potentially valid position
            let mut skip = 1i8;
            for s in 1..13 {
                if is_valid_candidate_13((i >> s) as u32) {
                    skip = s as i8;
                    break;
                }
                skip = (s + 1) as i8;
            }
            lut[i] = skip.min(13);
        }
    }

    lut
}

use std::sync::OnceLock;
static DEFLATE_LUT: OnceLock<Vec<i8>> = OnceLock::new();

fn get_lut() -> &'static [i8] {
    DEFLATE_LUT.get_or_init(generate_deflate_lut)
}

// ============================================================================
// Precode Leaf Count LUT
// ============================================================================

/// Compute virtual leaf count for a precode length
/// A length of L uses 2^(7-L) leaves in a depth-7 tree
const fn precode_to_leaves(length: u8) -> u16 {
    if length == 0 || length > MAX_PRECODE_LENGTH {
        0
    } else {
        1 << (MAX_PRECODE_LENGTH - length)
    }
}

/// LUT for 4 precodes at once (12 bits -> leaf count)
const fn generate_precode_lut() -> [u16; 1 << 12] {
    let mut lut = [0u16; 1 << 12];
    let mut i = 0u32;
    while i < (1 << 12) {
        let p0 = (i & 7) as u8;
        let p1 = ((i >> 3) & 7) as u8;
        let p2 = ((i >> 6) & 7) as u8;
        let p3 = ((i >> 9) & 7) as u8;
        lut[i as usize] = precode_to_leaves(p0)
            + precode_to_leaves(p1)
            + precode_to_leaves(p2)
            + precode_to_leaves(p3);
        i += 1;
    }
    lut
}

static PRECODE_LEAF_LUT: [u16; 1 << 12] = generate_precode_lut();

/// Validate precode by counting allocated leaves (rapidgzip's approach)
/// A valid Huffman tree has exactly 2^max_length leaves (128 for 7-bit)
/// Exception: single symbol with length 1 uses 64 leaves
///
/// This is the key fail-fast check: 99.9% of random bit patterns fail here
#[inline]
fn validate_precode(hclen: usize, precode_bits: u64) -> bool {
    // Note: Any precode value 0-7 is valid, so we skip bit-pattern checking

    // Count leaves using LUT (4 precodes at a time) - Duff's device unrolled
    let mut leaf_count: u16 = 0;

    // Only count the precodes we actually have (hclen + 4)
    let precode_count = hclen.min(19);
    let total_bits = precode_count * 3;
    let active_mask = (1u64 << total_bits) - 1;
    let active_bits = precode_bits & active_mask;

    // Chunk 0: bits 0-11 (precodes 0-3)
    leaf_count += PRECODE_LEAF_LUT[(active_bits & 0xFFF) as usize];

    if precode_count > 4 {
        // Chunk 1: bits 12-23 (precodes 4-7)
        leaf_count += PRECODE_LEAF_LUT[((active_bits >> 12) & 0xFFF) as usize];
    }

    if precode_count > 8 {
        // Chunk 2: bits 24-35 (precodes 8-11)
        leaf_count += PRECODE_LEAF_LUT[((active_bits >> 24) & 0xFFF) as usize];
    }

    if precode_count > 12 {
        // Chunk 3: bits 36-47 (precodes 12-15)
        leaf_count += PRECODE_LEAF_LUT[((active_bits >> 36) & 0xFFF) as usize];
    }

    if precode_count > 16 {
        // Chunk 4: bits 48-56 (precodes 16-18)
        let chunk4 = (active_bits >> 48) & 0x1FF;
        let p16 = (chunk4 & 7) as u8;
        let p17 = ((chunk4 >> 3) & 7) as u8;
        let p18 = ((chunk4 >> 6) & 7) as u8;

        // Only count what's valid
        if precode_count > 16 {
            leaf_count += precode_to_leaves(p16);
        }
        if precode_count > 17 {
            leaf_count += precode_to_leaves(p17);
        }
        if precode_count > 18 {
            leaf_count += precode_to_leaves(p18);
        }
    }

    // FAIL FAST: Exact leaf count check
    // Valid: exactly 128 (full tree) or 64 (single symbol with length 1)
    leaf_count == 128 || leaf_count == 64
}

// ============================================================================
// Fast Bit Reader
// ============================================================================

pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u64,
    bits_available: u8,
}

impl<'a> BitReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bits_available: 0,
        };
        reader.refill();
        reader
    }

    #[inline]
    pub fn seek_to_bit(&mut self, bit_offset: usize) {
        self.byte_pos = bit_offset / 8;
        self.bit_buf = 0;
        self.bits_available = 0;
        self.refill();
        let skip = (bit_offset % 8) as u8;
        if skip > 0 {
            self.bit_buf >>= skip;
            self.bits_available = self.bits_available.saturating_sub(skip);
        }
    }

    #[inline]
    fn refill(&mut self) {
        while self.bits_available <= 56 && self.byte_pos < self.data.len() {
            self.bit_buf |= (self.data[self.byte_pos] as u64) << self.bits_available;
            self.bits_available += 8;
            self.byte_pos += 1;
        }
    }

    #[inline]
    pub fn peek(&self, n: u8) -> u64 {
        self.bit_buf & ((1u64 << n) - 1)
    }

    #[inline]
    pub fn skip(&mut self, n: u8) {
        self.bit_buf >>= n;
        self.bits_available = self.bits_available.saturating_sub(n);
        if self.bits_available < 32 {
            self.refill();
        }
    }

    #[inline]
    pub fn read(&mut self, n: u8) -> u64 {
        let val = self.peek(n);
        self.skip(n);
        val
    }

    #[inline]
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 - self.bits_available as usize
    }

    #[inline]
    pub fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len() && self.bits_available == 0
    }
}

// ============================================================================
// Block Boundary
// ============================================================================

#[derive(Clone, Debug)]
pub struct BlockBoundary {
    /// Bit offset in the compressed stream
    pub bit_offset: usize,
    /// Whether this is a valid block start
    pub valid: bool,
    /// HLIT value (literal code count - 257)
    pub hlit: u8,
    /// HDIST value (distance code count - 1)
    pub hdist: u8,
    /// HCLEN value (precode count - 4)
    pub hclen: u8,
}

// ============================================================================
// Fixed-Huffman prefilter (BTYPE=01)
// ============================================================================
//
// BlockFinder originally excluded BTYPE=01 candidates entirely on the
// theory that "fixed Huffman has no header to validate." That left a
// class of real boundaries invisible to the heuristic — on Silesia at
// least one chunk's search region had only fixed-Huffman boundaries,
// and `decompress_parallel` silently fell back to sequential libdeflate
// (CI bench reported 0.62× rapidgzip; the deletion-trap killer test on
// the routing test fixture didn't catch this because that fixture has
// BTYPE=10 boundaries).
//
// We can validate BTYPE=01 candidates cheaply by decoding a handful of
// fixed-Huffman symbols and checking each is well-formed:
//
//   - All four symbols decode without falling off the alphabet (no
//     "code length 0" entry hit).
//   - No symbol is 286 or 287 (unused per RFC 1951 §3.2.6).
//   - The first symbol is *not* `END_OF_BLOCK` (256) — a real fixed-
//     Huffman block produces at least one byte before EOB.
//   - For length codes (symbols 257..=285), the following distance
//     code is in 0..=29 (codes 30/31 are reserved).
//
// Per-probe cost: 4× 9-bit table lookup + a couple of range checks =
// well under 100 ns. The table is a single `[u16; 512]` built once.

/// Entry layout: low 9 bits = symbol, top 4 bits = code length (7/8/9).
/// 0 = invalid (no canonical code matches this 9-bit pattern's prefix).
type FixedLitlenEntry = u16;

/// 9-bit reverse-bits LUT → (symbol, code_len) for the deflate fixed
/// Huffman litlen table. Indexed by `peek(9) & 0x1FF` — the LSB-first
/// bits as they come out of the bit buffer; we encode the bit-reversal
/// into the table itself.
fn fixed_litlen_lut() -> &'static [FixedLitlenEntry; 512] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[FixedLitlenEntry; 512]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0u16; 512];
        // Per RFC 1951 §3.2.6:
        //   codes  0..=23  (7 bits) -> symbols 256..=279
        //   codes 48..=191 (8 bits) -> symbols   0..=143
        //   codes 192..=199 (8 bits) -> symbols 280..=287
        //   codes 400..=511 (9 bits) -> symbols 144..=255
        let mut emit = |code: u32, code_len: u8, sym: u16| {
            // The stream is LSB-first. Canonical Huffman codes are read
            // MSB-first into the code value. So we reverse the
            // `code_len` low bits of `code` to get the bit pattern that
            // appears in the bit buffer.
            let reversed = reverse_low_bits(code, code_len);
            let entry = (sym & 0x1FF) | ((code_len as u16) << 12);
            let stride = 1u32 << code_len;
            let mut idx = reversed;
            while idx < 512 {
                table[idx as usize] = entry;
                idx += stride;
            }
        };
        for code in 0..=23 {
            emit(code, 7, (256 + code) as u16);
        }
        for code in 48..=191 {
            emit(code, 8, (code - 48) as u16);
        }
        for code in 192..=199 {
            emit(code, 8, (280 + (code - 192)) as u16);
        }
        for code in 400..=511 {
            emit(code, 9, (144 + (code - 400)) as u16);
        }
        table
    })
}

/// 5-bit reverse-bits LUT → distance symbol. Codes 0..=29 are valid;
/// 30 and 31 are reserved (entry remains 0xFFFF as "invalid").
fn fixed_dist_lut() -> &'static [u16; 32] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[u16; 32]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0xFFFFu16; 32];
        for code in 0u32..=29 {
            let reversed = reverse_low_bits(code, 5);
            table[reversed as usize] = code as u16;
        }
        table
    })
}

#[inline(always)]
fn reverse_low_bits(mut v: u32, n: u8) -> u32 {
    let mut r = 0u32;
    for _ in 0..n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// Decode the first two fixed-Huffman symbols at `bit_offset` and accept
/// the position only if both are well-formed (valid litlen alphabet
/// entries, not the unused 286/287 codes, first symbol not EOB).
///
/// This is the cheap prefilter; the strict check is `validate_boundary`
/// in `fast_marker_inflate`, run by `try_decode_at` on every accepted
/// candidate. The job here is to *eliminate most false positives* —
/// catching all of them is `validate_boundary`'s job. Two symbols is the
/// sweet spot: enough to reject ~95% of random positions (each random
/// 9-bit pattern has ~256/512 chance of mapping to a valid symbol; two
/// in a row is ~25%, and the EOB/unused-symbol checks knock that down
/// further), without paying for distance-code validation.
///
/// Earlier versions decoded 4 symbols and validated distance codes too.
/// On a 2 MiB random fixture that ran for 3+ minutes — `find_blocks`
/// invokes the prefilter on every BTYPE=01 candidate (about 1/4 of all
/// bit positions), so per-probe cost dominates. The 2-symbol prefilter
/// here runs ~50 ns per call.
pub(crate) fn validate_fixed_block_prefix(data: &[u8], bit_offset: usize) -> bool {
    let litlen_lut = fixed_litlen_lut();
    let mut bit = bit_offset;

    // First symbol: must not be EOB (real fixed-Huffman blocks rarely
    // emit EOB as their first symbol; rejecting it loses a tiny fraction
    // of real boundaries and rejects many false positives).
    let Some(window) = peek_bits_at(data, bit, 9) else {
        return false;
    };
    let entry = litlen_lut[(window & 0x1FF) as usize];
    let sym = entry & 0x1FF;
    let code_len = (entry >> 12) & 0xF;
    if code_len == 0 || sym == 256 || sym >= 286 {
        return false;
    }
    bit += code_len as usize;
    // For length codes, skip past the bits we know come next so the
    // second symbol's peek lands on the right bit. We don't validate
    // those extras / distance codes — that's `validate_boundary`'s job.
    if sym >= 257 {
        let lidx = (sym - 257) as usize;
        const LENGTH_EXTRA_BITS: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];
        const DIST_EXTRA_BITS: [u8; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
            12, 13, 13,
        ];
        if lidx >= LENGTH_EXTRA_BITS.len() {
            return false;
        }
        bit += LENGTH_EXTRA_BITS[lidx] as usize;
        // 5-bit distance code; check valid (codes 30/31 are reserved).
        let Some(dist_window) = peek_bits_at(data, bit, 5) else {
            return false;
        };
        let dist_sym = fixed_dist_lut()[(dist_window & 0x1F) as usize];
        if dist_sym == 0xFFFF {
            return false;
        }
        bit += 5 + DIST_EXTRA_BITS[dist_sym as usize] as usize;
    }

    // Second symbol.
    let Some(window) = peek_bits_at(data, bit, 9) else {
        return false;
    };
    let entry = litlen_lut[(window & 0x1FF) as usize];
    let sym = entry & 0x1FF;
    let code_len = (entry >> 12) & 0xF;
    code_len != 0 && sym < 286
}

/// Peek `n` bits starting at absolute bit offset `bit` in `data`
/// (LSB-first within bytes). Returns `None` if the read would overrun
/// `data`. `n` must be ≤ 16.
#[inline]
fn peek_bits_at(data: &[u8], bit: usize, n: u8) -> Option<u32> {
    debug_assert!(n <= 16);
    let byte_idx = bit / 8;
    let bit_in_byte = (bit % 8) as u32;
    // Need to read up to 3 bytes to get 16+7 = 23 bits' worth.
    if byte_idx + 2 >= data.len() {
        // Tail of stream — be conservative and bail.
        return None;
    }
    let combined = (data[byte_idx] as u32)
        | ((data[byte_idx + 1] as u32) << 8)
        | ((data[byte_idx + 2] as u32) << 16);
    Some((combined >> bit_in_byte) & ((1u32 << n) - 1))
}

// ============================================================================
// Block Finder
// ============================================================================

pub struct BlockFinder<'a> {
    data: &'a [u8],
}

impl<'a> BlockFinder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Find the first plausible block-header candidate at or after
    /// `from_bit`, scanning up to `scan_radius_bits` ahead. Returns the
    /// raw bit offset of the candidate (no `validate_boundary` filter;
    /// caller is expected to trial-decode and trust ISA-L's verdict).
    /// Mirrors what rapidgzip does in `decodeChunkWithRapidgzip` — it
    /// trusts a single candidate and lets the deflate decoder reject
    /// false positives via its own error path.
    pub fn find_first_candidate(&self, from_bit: usize, scan_radius_bits: usize) -> Option<usize> {
        let end_bit = from_bit
            .saturating_add(scan_radius_bits)
            .min(self.data.len() * 8);
        let candidates = self.find_blocks(from_bit, end_bit);
        candidates
            .into_iter()
            .find(|b| b.bit_offset >= from_bit)
            .map(|b| b.bit_offset)
    }

    /// Literal port of rapidgzip's `seekToNonFinalUncompressedDeflateBlock`
    /// (vendor/rapidgzip/.../blockfinder/Uncompressed.hpp:21-95).
    /// Scans byte boundaries for the LEN/~LEN coincidence pattern, then
    /// validates the 3-bit deflate header preceding the byte boundary.
    /// Returns all detected stored-block candidates (one per match), each
    /// emitted at the EARLIEST valid bit offset (rapidgzip's offset.first,
    /// matching the next-block boundary the predecessor would land on).
    ///
    /// `start_bit` and `end_bit` are bit positions into `self.data`.
    fn find_uncompressed_blocks(&self, start_bit: usize, end_bit: usize) -> Vec<BlockBoundary> {
        const DEFLATE_MAGIC_BIT_COUNT: usize = 3;
        const BYTE_SIZE: usize = 8;
        const MAX_PRECEDING_BITS: usize = DEFLATE_MAGIC_BIT_COUNT + (BYTE_SIZE - 1); // 10
        let data = self.data;
        let n = data.len();
        let mut out = Vec::new();

        // Mirror of rapidgzip's `untilOffsetSizeMember`: where the LEN word
        // can live. We have at most 4 bytes of LEN + NLEN.
        let until_byte_for_size = end_bit.div_ceil(BYTE_SIZE).min(n.saturating_sub(4));

        // Mirror of `startOffsetByte`: align to next byte AFTER the 3-bit
        // header (so the LEN field is byte-aligned). Floor at byte 1 so
        // we have at least one byte of preceding bits to inspect.
        let start_byte = ((start_bit + DEFLATE_MAGIC_BIT_COUNT).div_ceil(BYTE_SIZE)).max(1);
        if start_byte >= until_byte_for_size {
            return out;
        }

        // Walk byte boundaries, reading LEN(2) | NLEN(2) and checking
        // len == ~nlen. Equivalent to rapidgzip's sliding 32-bit window;
        // we just re-read each iteration (cheap on x86_64).
        for byte_off in start_byte..until_byte_for_size {
            // LEN ^ ~NLEN == 0  ⇔  (size ^ (size >> 16)) & 0xFFFF == 0xFFFF
            let size = u32::from_le_bytes([
                data[byte_off],
                data[byte_off + 1],
                data[byte_off + 2],
                data[byte_off + 3],
            ]);
            if ((size ^ (size >> 16)) & 0xFFFF) != 0xFFFF {
                continue;
            }

            let byte_bit = byte_off * BYTE_SIZE;
            // Read MAX_PRECEDING_BITS BEFORE the byte boundary. Rapidgzip
            // seeks back and peeks 10 bits; we just construct from bytes.
            // Bit ordering: LSB-first across bytes, so the bit at position
            // (byte_bit - 1) is the MSB of byte (byte_off - 1)'s LSB-first
            // numbering — i.e., bit 7 of data[byte_off - 1].
            //
            // Following rapidgzip's convention: bit indices in preBits
            // are numbered such that index 0 is the OLDEST bit
            // (byte_bit - MAX_PRECEDING_BITS) and index MAX_PRECEDING_BITS-1
            // is the NEWEST (byte_bit - 1). Build by reading the bytes
            // covering [byte_bit - MAX_PRECEDING_BITS, byte_bit).
            let first_pre_bit = match byte_bit.checked_sub(MAX_PRECEDING_BITS) {
                Some(v) => v,
                None => continue,
            };
            let first_pre_byte = first_pre_bit / BYTE_SIZE;
            let shift = first_pre_bit % BYTE_SIZE;
            // Need 2 bytes (16 bits) to extract up to 10 bits starting at
            // first_pre_bit. previous_bits is laid out so that bit 0 is
            // the oldest, bit (MAX_PRECEDING_BITS-1) is the newest.
            let raw = (data[first_pre_byte] as u32) | ((data[first_pre_byte + 1] as u32) << 8);
            let preceding_bits = (raw >> shift) & ((1u32 << MAX_PRECEDING_BITS) - 1);

            // The 3 magic bits (bfinal=0, btype=00) must be at the TOP of
            // preceding_bits (the newest bits, closest to byte_bit).
            // MAGIC_BITS_MASK = 0b111 << (MAX_PRECEDING_BITS - 3) = 0b111 << 7.
            const MAGIC_BITS_MASK: u32 = 0b111u32 << (MAX_PRECEDING_BITS - DEFLATE_MAGIC_BIT_COUNT);
            if (preceding_bits & MAGIC_BITS_MASK) != 0 {
                continue;
            }

            // Count trailing zeros in the padding region (bits 0..7 of
            // preceding_bits, going from bit MAX_PRECEDING_BITS-4 downward).
            // trailingZeros starts at DEFLATE_MAGIC_BIT_COUNT and grows
            // for each preceding zero padding bit. Mirror of rapidgzip's
            // for-loop at Uncompressed.hpp:77-82.
            let mut trailing_zeros = DEFLATE_MAGIC_BIT_COUNT;
            for j in (trailing_zeros + 1)..=MAX_PRECEDING_BITS {
                if (preceding_bits & (1u32 << (MAX_PRECEDING_BITS - j))) != 0 {
                    break;
                }
                trailing_zeros = j;
            }

            // The earliest valid start: byte_bit - trailing_zeros.
            // Must be ≥ start_bit and < end_bit.
            let earliest = match byte_bit.checked_sub(trailing_zeros) {
                Some(v) => v,
                None => continue,
            };
            let latest = byte_bit - DEFLATE_MAGIC_BIT_COUNT;
            if earliest >= start_bit && latest < end_bit {
                // Validate data fits: aligned_byte + 4 + len ≤ n.
                let len = (size & 0xFFFF) as usize;
                if len == 0 || byte_off + 4 + len > n {
                    continue;
                }
                out.push(BlockBoundary {
                    bit_offset: earliest,
                    valid: true,
                    hlit: 0,
                    hdist: 0,
                    hclen: 0,
                });
            }
        }
        out
    }

    /// Find all valid deflate block starts in a range.
    ///
    /// Searches dynamic (via LUT + Huffman validation) AND uncompressed
    /// (via byte-boundary LEN/~LEN scan) candidates, merging results in
    /// ascending bit-offset order. Mirrors rapidgzip's alternating
    /// dynamic + uncompressed search in `GzipChunk::tryToDecode`
    /// (vendor/rapidgzip/.../chunkdecoding/GzipChunk.hpp:803-846).
    pub fn find_blocks(&self, start_bit: usize, end_bit: usize) -> Vec<BlockBoundary> {
        let mut dynamic = self.find_dynamic_blocks(start_bit, end_bit);
        let uncompressed = self.find_uncompressed_blocks(start_bit, end_bit);
        dynamic.extend(uncompressed);
        dynamic.sort_by_key(|b| b.bit_offset);
        dynamic
    }

    /// Dynamic-Huffman block finder. Direct port of rapidgzip's
    /// `seekToNonFinalDynamicDeflateBlock` (DynamicHuffman.hpp:168+).
    /// Was inlined into find_blocks; extracted so find_blocks can also
    /// run the uncompressed pass and merge results.
    fn find_dynamic_blocks(&self, start_bit: usize, end_bit: usize) -> Vec<BlockBoundary> {
        let mut blocks = Vec::new();
        let mut reader = BitReader::new(self.data);
        reader.seek_to_bit(start_bit);

        let lut = get_lut();
        let mut bit_offset = start_bit;

        while bit_offset < end_bit && !reader.is_eof() {
            // LEVEL 1: LUT check (fastest rejection)
            let lut_bits = reader.peek(LUT_BITS as u8) as usize;
            if lut_bits >= lut.len() {
                reader.skip(1);
                bit_offset += 1;
                continue;
            }

            let skip = lut[lut_bits];

            if skip > 0 {
                reader.skip(skip as u8);
                bit_offset += skip as usize;
                continue;
            }

            let header = reader.peek(17);
            let btype = ((header >> 1) & 3) as u8;

            match btype {
                0 => {
                    // Stored block: skip to byte boundary, read len and ~len
                    if self.validate_stored_block(bit_offset) {
                        blocks.push(BlockBoundary {
                            bit_offset,
                            valid: true,
                            hlit: 0,
                            hdist: 0,
                            hclen: 0,
                        });
                    }
                    reader.seek_to_bit(bit_offset + 1);
                    bit_offset += 1;
                }
                1 => {
                    // Fixed Huffman — not emitted here. BTYPE=01 boundary
                    // detection lives in `single_member::search_boundary_forward`
                    // as a separate tier with its own cheap prefilter.
                    reader.skip(1);
                    bit_offset += 1;
                }
                2 => {
                    // Dynamic Huffman: full validation
                    let hlit = ((header >> 3) & 31) as u8;
                    let hdist = ((header >> 8) & 31) as u8;
                    let hclen = ((header >> 13) & 15) as u8;

                    if hlit > 29 || hdist > 29 {
                        reader.skip(1);
                        bit_offset += 1;
                        continue;
                    }

                    let precode_count = (hclen + 4) as usize;
                    let precode_bit_count = precode_count * 3;

                    reader.skip(17);
                    if reader.is_eof() {
                        break;
                    }

                    let precode_bits = reader.read(precode_bit_count as u8);

                    if !validate_precode(precode_count, precode_bits) {
                        reader.seek_to_bit(bit_offset + 1);
                        bit_offset += 1;
                        continue;
                    }

                    if let Some(precode_lengths) = self.parse_precode(precode_count, precode_bits) {
                        let hlit_count = 257 + hlit as usize;
                        let lit_dist_count = hlit_count + (1 + hdist as usize);

                        if self.validate_huffman_codes(
                            &mut reader,
                            &precode_lengths,
                            lit_dist_count,
                            hlit_count,
                        ) {
                            blocks.push(BlockBoundary {
                                bit_offset,
                                valid: true,
                                hlit,
                                hdist,
                                hclen,
                            });
                        }
                    }

                    reader.seek_to_bit(bit_offset + 1);
                    bit_offset += 1;
                }
                _ => {
                    reader.skip(1);
                    bit_offset += 1;
                }
            }
        }

        blocks
    }

    /// Validate a stored block at the given bit position.
    /// After BFINAL + BTYPE (3 bits), skip to next byte boundary,
    /// then read LEN (2 bytes) and NLEN (2 bytes). LEN == ~NLEN.
    /// Also checks that the block data fits within the stream.
    fn validate_stored_block(&self, bit_offset: usize) -> bool {
        let after_header_bit = bit_offset + 3;
        let aligned_byte = after_header_bit.div_ceil(8);
        if aligned_byte + 4 > self.data.len() {
            return false;
        }
        let len = u16::from_le_bytes([self.data[aligned_byte], self.data[aligned_byte + 1]]);
        let nlen = u16::from_le_bytes([self.data[aligned_byte + 2], self.data[aligned_byte + 3]]);
        if len != !nlen {
            return false;
        }
        // Reject trivial (len=0) blocks and check data fits
        let data_start = aligned_byte + 4;
        len > 0 && data_start + len as usize <= self.data.len()
    }

    /// Parse precode lengths from bits
    fn parse_precode(&self, count: usize, bits: u64) -> Option<[u8; 19]> {
        let mut lengths = [0u8; 19];

        for i in 0..count {
            let len = ((bits >> (i * 3)) & 7) as u8;
            if len > MAX_PRECODE_LENGTH {
                return None;
            }
            lengths[PRECODE_ALPHABET[i]] = len;
        }

        // Validate it forms a valid Huffman code
        if !self.is_valid_huffman_lengths(&lengths) {
            return None;
        }

        Some(lengths)
    }

    /// Check if lengths form a valid Huffman code
    fn is_valid_huffman_lengths(&self, lengths: &[u8]) -> bool {
        let mut bl_count = [0u32; 16];

        for &len in lengths {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
            }
        }

        // Kraft inequality check
        let mut code = 0u32;
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            if code > (1 << bits) {
                return false;
            }
        }

        true
    }

    /// Validate that we can build valid literal/distance Huffman codes.
    /// `hlit_count` = 257 + HLIT (number of lit/len codes).
    fn validate_huffman_codes(
        &self,
        reader: &mut BitReader,
        precode_lengths: &[u8; 19],
        total_codes: usize,
        hlit_count: usize,
    ) -> bool {
        // Build precode Huffman table
        let precode_table = match build_huffman_table(precode_lengths, 7) {
            Some(t) => t,
            None => return false,
        };

        // Read literal/distance code lengths
        let mut lengths = vec![0u8; total_codes];
        let mut i = 0;

        while i < total_codes {
            if reader.is_eof() {
                return false;
            }

            let symbol = decode_huffman(reader, &precode_table, 7);
            if symbol.is_none() {
                return false;
            }
            let symbol = symbol.unwrap();

            match symbol {
                0..=15 => {
                    lengths[i] = symbol as u8;
                    i += 1;
                }
                16 => {
                    // Copy previous 3-6 times
                    if i == 0 {
                        return false;
                    }
                    let repeat = reader.read(2) as usize + 3;
                    let prev = lengths[i - 1];
                    for _ in 0..repeat {
                        if i >= total_codes {
                            return false;
                        }
                        lengths[i] = prev;
                        i += 1;
                    }
                }
                17 => {
                    // Repeat 0, 3-10 times
                    let repeat = reader.read(3) as usize + 3;
                    for _ in 0..repeat {
                        if i >= total_codes {
                            return false;
                        }
                        lengths[i] = 0;
                        i += 1;
                    }
                }
                18 => {
                    // Repeat 0, 11-138 times
                    let repeat = reader.read(7) as usize + 11;
                    for _ in 0..repeat {
                        if i >= total_codes {
                            return false;
                        }
                        lengths[i] = 0;
                        i += 1;
                    }
                }
                _ => return false,
            }
        }

        // Check END_OF_BLOCK (symbol 256) has non-zero length
        if lengths.len() > END_OF_BLOCK_SYMBOL && lengths[END_OF_BLOCK_SYMBOL] == 0 {
            return false;
        }

        let lit_lengths = &lengths[..hlit_count.min(lengths.len())];
        let dist_lengths = if lengths.len() > hlit_count {
            &lengths[hlit_count..]
        } else {
            &[]
        };

        self.is_valid_huffman_lengths(lit_lengths) && self.is_valid_huffman_lengths(dist_lengths)
    }
}

// ============================================================================
// Huffman Table Helpers
// ============================================================================

/// Simple Huffman table: code -> (symbol, length)
fn build_huffman_table(lengths: &[u8], max_bits: u8) -> Option<Vec<(u16, u8)>> {
    let table_size = 1 << max_bits;
    let mut table = vec![(0u16, 0u8); table_size];

    // Count codes per length
    let mut bl_count = [0u32; 16];
    for &len in lengths {
        if len > 0 && len <= 15 {
            bl_count[len as usize] += 1;
        }
    }

    // Generate next_code
    let mut code = 0u32;
    let mut next_code = [0u32; 16];
    for bits in 1..=max_bits as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes and fill table
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 && len <= max_bits {
            let c = next_code[len as usize];
            next_code[len as usize] += 1;

            // Reverse bits
            let reversed = reverse_bits(c, len);

            // Fill table entries
            let fill = 1 << (max_bits - len);
            for i in 0..fill {
                let idx = (reversed | (i << len)) as usize;
                if idx < table.len() {
                    table[idx] = (sym as u16, len);
                }
            }
        }
    }

    Some(table)
}

fn reverse_bits(value: u32, bits: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

fn decode_huffman(reader: &mut BitReader, table: &[(u16, u8)], max_bits: u8) -> Option<u16> {
    let bits = reader.peek(max_bits) as usize;
    if bits >= table.len() {
        return None;
    }

    let (symbol, len) = table[bits];
    if len == 0 {
        return None;
    }

    reader.skip(len);
    Some(symbol)
}

// ============================================================================
// Parallel Block Finding
// ============================================================================

/// Find blocks in parallel across the data
pub fn find_blocks_parallel(data: &[u8], num_threads: usize) -> Vec<BlockBoundary> {
    let data_bits = data.len() * 8;
    let chunk_bits = data_bits / num_threads;

    if chunk_bits < 1024 || num_threads <= 1 {
        return BlockFinder::new(data).find_blocks(0, data_bits);
    }

    let results: Vec<std::sync::Mutex<Vec<BlockBoundary>>> = (0..num_threads)
        .map(|_| std::sync::Mutex::new(Vec::new()))
        .collect();
    let next_chunk = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..num_threads {
            let results_ref = &results;
            let next_ref = &next_chunk;

            scope.spawn(move || {
                let finder = BlockFinder::new(data);

                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_threads {
                        break;
                    }

                    let start = idx * chunk_bits;
                    let end = if idx == num_threads - 1 {
                        data_bits
                    } else {
                        (idx + 1) * chunk_bits + 1024 // Overlap for boundary blocks
                    };

                    let blocks = finder.find_blocks(start, end);
                    *results_ref[idx].lock().unwrap() = blocks;
                }
            });
        }
    });

    // Merge results
    let mut all_blocks: Vec<BlockBoundary> = results
        .into_iter()
        .flat_map(|m| m.into_inner().unwrap())
        .collect();

    // Sort by bit offset and deduplicate
    all_blocks.sort_by_key(|b| b.bit_offset);
    all_blocks.dedup_by_key(|b| b.bit_offset);

    all_blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_generation() {
        let lut = get_lut();

        // Invalid: BTYPE=11 (reserved), BFINAL=0
        assert!(lut[0b110] > 0);
        // Invalid: BTYPE=11, BFINAL=1
        assert!(lut[0b111] > 0);

        // Invalid: BTYPE=00 (stored) — not emitted by this BlockFinder.
        // Rapidgzip's seekToNonFinalDynamicDeflateBlock is dynamic-only;
        // a separate seekToNonFinalUncompressedDeflateBlock handles stored.
        assert!(lut[0b000] > 0);
        assert!(lut[0b001] > 0);

        // Invalid: BTYPE=01 (fixed) — excluded from candidate search at
        // the BlockFinder level. Detection lives in
        // `single_member::search_boundary_forward` as a dedicated tier
        // that runs only when the other tiers find nothing.
        assert!(lut[0b010] > 0);
        assert!(lut[0b011] > 0);

        // Valid: BTYPE=10 (dynamic), BFINAL=0, HLIT=0, HDIST=0
        let valid = 0b00000_00000_10_0u32;
        assert_eq!(lut[valid as usize], 0);

        // Invalid: BTYPE=10 (dynamic), BFINAL=1 — bfinal filter rejects.
        let final_dyn = 0b00000_00000_10_1u32;
        assert!(lut[final_dyn as usize] > 0);
    }

    #[test]
    fn test_block_finder_at_oracle_positions() {
        let mut data = Vec::with_capacity(8 * 1024 * 1024);
        let mut rng: u64 = 0xdeadbeef;
        while data.len() < 8 * 1024 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(8 * 1024 * 1024 - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(8 * 1024 * 1024);

        let mut gz = Vec::new();
        {
            use std::io::Write;
            let mut enc = flate2::write::GzEncoder::new(&mut gz, flate2::Compression::default());
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }

        let header_size = crate::decompress::parallel::single_member::skip_gzip_header(&gz)
            .expect("valid header");
        let deflate = &gz[header_size..gz.len() - 8];

        let scan = crate::decompress::scan_inflate::scan_deflate_fast(deflate, 512 * 1024, 0)
            .expect("scan");

        let lut = get_lut();
        let finder = BlockFinder::new(deflate);
        let mut found = 0;

        for cp in &scan.checkpoints {
            let real_bitsleft = (cp.bitsleft as u8) as usize;
            let bit_pos = cp.input_byte_pos * 8 - real_bitsleft;

            let mut reader = BitReader::new(deflate);
            reader.seek_to_bit(bit_pos);

            let lut_bits = reader.peek(LUT_BITS as u8) as usize;
            let lut_pass = lut_bits < lut.len() && lut[lut_bits] == 0;

            let header = reader.peek(17);
            let btype = (header >> 1) & 3;
            let hlit = ((header >> 3) & 31) as u8;
            let hdist = ((header >> 8) & 31) as u8;
            let hclen = ((header >> 13) & 15) as u8;

            let mut precode_pass = false;
            let mut huffman_pass = false;

            if lut_pass && btype == 2 {
                let precode_count = (hclen + 4) as usize;
                let precode_bit_count = precode_count * 3;
                reader.skip(17);

                if !reader.is_eof() {
                    let precode_bits = reader.read(precode_bit_count as u8);
                    precode_pass = validate_precode(precode_count, precode_bits);

                    if precode_pass {
                        if let Some(precode_lengths) =
                            finder.parse_precode(precode_count, precode_bits)
                        {
                            let hlit_count = 257 + hlit as usize;
                            let lit_dist_count = hlit_count + (1 + hdist as usize);
                            reader.seek_to_bit(bit_pos + 17 + precode_bit_count as usize);
                            huffman_pass = finder.validate_huffman_codes(
                                &mut reader,
                                &precode_lengths,
                                lit_dist_count,
                                hlit_count,
                            );
                        }
                    }
                }
            }

            if huffman_pass {
                found += 1;
            } else {
                eprintln!(
                    "MISS bit={}: btype={} lut={} precode={} huffman={} hlit={} hdist={} hclen={}",
                    bit_pos, btype, lut_pass, precode_pass, huffman_pass, hlit, hdist, hclen
                );
            }
        }

        let total = scan.checkpoints.len();
        let recall_pct = if total > 0 {
            found as f64 / total as f64 * 100.0
        } else {
            100.0
        };
        eprintln!(
            "validation: {}/{} oracle boundaries pass all levels ({:.0}%)",
            found, total, recall_pct
        );
        assert!(
            recall_pct >= 80.0,
            "block finder recall {:.0}% ({}/{}) below 80% threshold",
            recall_pct,
            found,
            total
        );
    }

    #[test]
    fn test_precode_leaf_lut() {
        // Length 7 -> 1 leaf
        assert_eq!(precode_to_leaves(7), 1);
        // Length 1 -> 64 leaves
        assert_eq!(precode_to_leaves(1), 64);
        // Length 0 -> 0 leaves
        assert_eq!(precode_to_leaves(0), 0);
    }

    #[test]
    fn test_find_blocks_parallel_matches_sequential() {
        let mut data = Vec::with_capacity(2 * 1024 * 1024);
        let mut rng: u64 = 0x12345678;
        while data.len() < 2 * 1024 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(2 * 1024 * 1024 - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(2 * 1024 * 1024);

        let mut gz = Vec::new();
        {
            use std::io::Write;
            let mut enc = flate2::write::GzEncoder::new(&mut gz, flate2::Compression::default());
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }

        let header_size = crate::decompress::parallel::single_member::skip_gzip_header(&gz)
            .expect("valid header");
        let deflate = &gz[header_size..gz.len() - 8];

        let sequential = BlockFinder::new(deflate).find_blocks(0, deflate.len() * 8);
        let parallel = find_blocks_parallel(deflate, 4);

        let seq_offsets: Vec<usize> = sequential.iter().map(|b| b.bit_offset).collect();
        let par_offsets: Vec<usize> = parallel.iter().map(|b| b.bit_offset).collect();

        // Parallel must find every block that sequential finds (superset due to overlap)
        for offset in &seq_offsets {
            assert!(
                par_offsets.contains(offset),
                "parallel missing sequential boundary at bit {}",
                offset
            );
        }

        // Both should be sorted
        for w in seq_offsets.windows(2) {
            assert!(w[0] < w[1], "sequential not sorted: {} >= {}", w[0], w[1]);
        }
        for w in par_offsets.windows(2) {
            assert!(w[0] < w[1], "parallel not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_find_blocks_parallel_small_data_fallback() {
        // Very small data: parallel should fall back to sequential
        let data = vec![0u8; 64];
        let sequential = BlockFinder::new(&data).find_blocks(0, data.len() * 8);
        let parallel = find_blocks_parallel(&data, 4);
        assert_eq!(
            sequential.len(),
            parallel.len(),
            "small data: parallel must match sequential count"
        );
    }

    #[test]
    fn test_find_blocks_parallel_single_thread() {
        let data = vec![0u8; 4096];
        let sequential = BlockFinder::new(&data).find_blocks(0, data.len() * 8);
        let parallel = find_blocks_parallel(&data, 1);
        assert_eq!(
            sequential.len(),
            parallel.len(),
            "T1: parallel must match sequential"
        );
    }
}
