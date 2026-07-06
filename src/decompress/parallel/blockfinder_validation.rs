#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Deflate block-boundary VALIDATION primitives (the candidate finder/validator).
//!
//! Port of rapidgzip's `blockfinder/` directory — the per-position deflate-header
//! checkers — NOT vendor's `core/BlockFinder.hpp` (that async coordinator is
//! [`super::async_block_finder`]) and NOT `GzipBlockFinder.hpp` (the offset
//! partitioner is [`super::gzip_block_finder`]). Concretely this mirrors:
//!   - `blockfinder/DynamicHuffman.hpp` (15-bit next-candidate LUT + precode/
//!     Huffman validation)
//!   - `blockfinder/Uncompressed.hpp` (stored-block boundary check)
//!
//! The shared [`BitReader`] now lives in [`super::bit_reader`] (rg
//! `core/BitReader.hpp`).
//!
//! Validates valid deflate block boundaries via:
//! 1. 15-bit LUT for quick invalid position skipping
//! 2. Precode validation via leaf counting
//! 3. Full Huffman table validation
//! 4. Symbol 256 (END_OF_BLOCK) must have non-zero length

#![allow(clippy::unusual_byte_groupings)]
#![allow(clippy::needless_range_loop)]

use std::sync::atomic::{AtomicUsize, Ordering};

use super::bit_reader::BitReader;

// ============================================================================
// Constants
// ============================================================================

/// Maximum precode length (7 bits)
const MAX_PRECODE_LENGTH: u8 = 7;

/// Precode alphabet order (RFC 1951)
const PRECODE_ALPHABET: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// END_OF_BLOCK symbol
const END_OF_BLOCK_SYMBOL: usize = 256;

/// LUT size in bits. Matches rapidgzip's OPTIMAL_NEXT_DEFLATE_LUT_SIZE
/// = 15 (vendor/rapidgzip/.../blockfinder/DynamicHuffman.hpp:145). 15
/// bits lets the LUT fully verify bfinal(1)+btype(2)+hlit(5)+hdist(5)
/// = 13 bits, far fewer false-positive candidates than a 13-bit LUT.
const LUT_BITS: usize = 15;
const LUT_SIZE: usize = 1 << LUT_BITS;

// ============================================================================
// 15-bit LUT for block candidate detection
// ============================================================================

/// Literal port of rapidgzip's `isDeflateCandidate<bitCount>`
/// (vendor/rapidgzip/.../blockfinder/DynamicHuffman.hpp:39-79).
///
/// Conservative: only checks the bits that are KNOWN (i.e. the lowest
/// `bit_count` bits of the input). At smaller `bit_count` fewer fields
/// are inspected — never rejects based on bits the caller doesn't know.
///
/// Filters applied (each gated on having enough bits):
///   bit 0       = bfinal, must be 0
///   bits 1-2    = btype, must be 0b10 (dynamic Huffman only)
///   bits 3-7    = HLIT, ≤ 29
///   bits 8-12   = HDIST, ≤ 29
#[inline]
fn is_deflate_candidate_n(bits: u32, bit_count: u8) -> bool {
    if bit_count == 0 {
        return false;
    }
    let is_last_block = (bits & 1) != 0;
    let mut matches = !is_last_block;
    if bit_count <= 1 {
        return matches;
    }
    let compression_type = (bits >> 1) & 0b11;
    matches &= (compression_type & 1) == 0;
    if bit_count <= 2 {
        return matches;
    }
    matches &= compression_type == 0b10;
    if bit_count < 1 + 2 + 5 {
        return matches;
    }
    let code_count = (bits >> 3) & 0b11111;
    matches &= code_count <= 29;
    if bit_count < 1 + 2 + 5 + 5 {
        return matches;
    }
    let distance_code_count = (bits >> 8) & 0b11111;
    matches &= distance_code_count <= 29;
    matches
}

/// Literal port of rapidgzip's `nextDeflateCandidate<bitCount>`
/// (DynamicHuffman.hpp:82-98). Recursively shifts right by 1, decreasing
/// the trusted-bit-count, until isDeflateCandidate returns true or
/// bit_count hits 0.
#[inline]
fn next_deflate_candidate(bits: u32, bit_count: u8) -> u8 {
    if is_deflate_candidate_n(bits, bit_count) {
        return 0;
    }
    if bit_count == 0 {
        return 0;
    }
    1 + next_deflate_candidate(bits >> 1, bit_count - 1)
}

/// Generate the 15-bit NEXT_DYNAMIC_DEFLATE_CANDIDATE_LUT.
/// Literal port of `NEXT_DYNAMIC_DEFLATE_CANDIDATE_LUT` (DynamicHuffman.hpp:113-124),
/// INCLUDING the negative-encoding for candidate positions (`lut[i] == 0`).
///
/// A candidate (a position that passes `isDeflateCandidate` on the known
/// bits) is stored as `-(1 + nextDeflateCandidate<CACHED-1>(i >> 1))` — a
/// NEGATIVE value whose sign flags "this is a candidate, run the precode
/// check" and whose MAGNITUDE is the unconditional advance to the next
/// candidate after this one. This lets the scan loop
/// ([`DeflateBlockValidator::find_dynamic_blocks`]) advance branchlessly:
/// `next <= 0` selects the validation arm, `next.abs()` is the slide amount,
/// with no separate `if skip > 0` test and no redundant 17-bit re-peek.
///
/// Non-candidate positions keep the positive skip, so the legacy scalar
/// scan ([`DeflateBlockValidator::find_dynamic_blocks_scalar`]) still works
/// unchanged (`skip > 0` ⇒ non-candidate; `skip <= 0` ⇒ candidate, formerly
/// `skip == 0`).
fn generate_deflate_lut() -> Vec<i8> {
    let mut lut = vec![0i8; LUT_SIZE];
    for i in 0..LUT_SIZE {
        let positive = next_deflate_candidate(i as u32, LUT_BITS as u8);
        lut[i] = if positive == 0 {
            // Candidate at this position: encode the distance to the NEXT
            // candidate (checked on the bits shifted by 1) as a negative.
            -((1 + next_deflate_candidate((i as u32) >> 1, (LUT_BITS - 1) as u8)) as i8)
        } else {
            positive as i8
        };
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

    // Only count the precodes we actually have (hclen + 4)
    let precode_count = hclen.min(19);
    let total_bits = precode_count * 3; // 12..=57, always < 64
    let active_mask = (1u64 << total_bits) - 1;
    let active_bits = precode_bits & active_mask;

    // BRANCHLESS leaf count — direct port of rapidgzip's
    // `PrecodeCheck::CountAllocatedLeaves::checkPrecode`
    // (blockfinder/precodecheck/CountAllocatedLeaves.hpp:119-131). `active_mask`
    // already zeroed every precode slot past the declared count, so the masked-out
    // 12-bit chunks index PRECODE_LEAF_LUT[0] == 0. That lets ALL FIVE lookups run
    // UNCONDITIONALLY — no per-chunk `if precode_count > N` data-dependent branch.
    //
    // Those chunk-gate branches were the hot branch-mispredict on incompressible
    // input: the finder reaches this validator at ~1-in-9 random bit positions
    // (bfinal=0 · btype=10 · hlit≤29 · hdist≤29 pass rate) with a uniformly random
    // hclen, so `precode_count > {4,8,12,16,17,18}` are all near-coin-flips the
    // predictor cannot learn. Byte-identical to the branched form: the extra
    // chunks contribute exactly LUT[0] == 0 leaves.
    //
    // Chunk 4 covers precode slots 16-18 (bits 48-56, at most 9 bits => index
    // <= 0x1FF) and reuses the SAME 12-bit LUT (its 4th slot, bits 9-11, is 0),
    // matching vendor's `LUT[precodeBits >> (4 * CACHED_BITS)]`.
    let leaf_count: u16 = PRECODE_LEAF_LUT[(active_bits & 0xFFF) as usize]
        + PRECODE_LEAF_LUT[((active_bits >> 12) & 0xFFF) as usize]
        + PRECODE_LEAF_LUT[((active_bits >> 24) & 0xFFF) as usize]
        + PRECODE_LEAF_LUT[((active_bits >> 36) & 0xFFF) as usize]
        + PRECODE_LEAF_LUT[(active_bits >> 48) as usize];

    // FAIL FAST: Exact leaf count check
    // Valid: exactly 128 (full tree) or 64 (single symbol with length 1)
    leaf_count == 128 || leaf_count == 64
}

// ============================================================================
// Block Boundary
// ============================================================================

#[derive(Clone, Debug)]
pub struct BlockBoundary {
    /// Earliest valid bit offset for this boundary (`offset.first` in vendor).
    pub bit_offset: usize,
    /// Actual bit to seek/decode from (`offset.second` in vendor).
    ///
    /// Dynamic/fixed boundaries decode at `bit_offset`. Stored blocks may
    /// admit an earlier logical boundary due to zero padding, so `seek_bit`
    /// is the byte-aligned header position (`byte_bit - 3`) while
    /// `bit_offset` stays at the earliest valid start.
    pub seek_bit: usize,
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
// Dynamic-only block finder — no BTYPE=01 (fixed-Huffman) candidates
// ============================================================================
//
// The BlockFinder is DYNAMIC-ONLY, faithful to vendor
// blockfinder/DynamicHuffman.hpp (which has NO fixed-Huffman finder).
// Fixed-Huffman regions produce no candidates here; they degrade to
// confirmed-chain decode exactly as vendor: when the speculative start
// mismatches the predecessor's confirmed boundary the consumer re-decodes
// from that boundary, traversing whatever blocks (including BTYPE=01)
// follow without special finder support.
//
// DELETED 2026-06-09: the BTYPE=01 prefilter (FixedLitlenEntry,
// fixed_litlen_lut, fixed_dist_lut, reverse_low_bits,
// validate_fixed_block_prefix) — a deliberate divergence from vendor that
// admitted fixed-Huffman candidates from the block finder. On bignasa
// (820 MB, 10.4× web log, T8) random bytes parse as BFINAL=1/BTYPE=01
// at the 4 MiB grid-base positions (1-in-8 per position), pass the cheap
// 2-symbol prefilter, decode a 23–495-byte phantom final-block stream,
// then are discarded at consume time when the predecessor's confirmed
// boundary lands 19 288–85 273 bits later — each costing ~150 ms
// head-of-line confirmed re-decode. 3–9 phantoms/run ≈ 39 % of T8 wall.
// rapidgzip cannot make this error: its deflate finder is dynamic-only by
// construction and scans forward to the true dynamic boundary instead.

/// Cheap structural prefilter before a full speculative trial decode.
/// Used for the EOF tail byte walk in `speculative_decode_find_boundary`
/// (thousands of offsets); block-finder candidates skip this.
///
/// BTYPE=01 (fixed-Huffman) returns false — dynamic-only, faithful to
/// vendor blockfinder/DynamicHuffman.hpp; fixed-block regions degrade to
/// confirmed-chain decode exactly as vendor.
pub(crate) fn plausible_trial_decode_offset(data: &[u8], bit_offset: usize) -> bool {
    let header = match peek_bits_at(data, bit_offset, 3) {
        Some(h) => h,
        None => return false,
    };
    let btype = (header >> 1) & 3;
    match btype {
        0 => {
            let aligned_bit = (bit_offset + 3).div_ceil(8) * 8;
            aligned_bit + 32 <= data.len() * 8
        }
        1 => false, // BTYPE=01 fixed-Huffman: dynamic-only finder, no candidate (vendor parity)
        2 => {
            let h = match peek_bits_at(data, bit_offset, 17) {
                Some(v) => v,
                None => return false,
            };
            let hlit = ((h >> 3) & 31) as u8;
            let hdist = ((h >> 8) & 31) as u8;
            let hclen = ((h >> 13) & 15) as u8;
            hlit <= 29 && hdist <= 29 && hclen <= 15
        }
        _ => false,
    }
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

pub struct DeflateBlockValidator<'a> {
    data: &'a [u8],
}

impl<'a> DeflateBlockValidator<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Find the first plausible block-header candidate at or after
    /// `from_bit`, scanning up to `scan_radius_bits` ahead. Returns the
    /// raw bit offset of the candidate (no full trial-decode here;
    /// caller trial-decodes via `decode_chunk_isal`).
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
            .map(|b| b.seek_bit)
    }

    /// Literal port of rapidgzip's `seekToNonFinalUncompressedDeflateBlock`
    /// (vendor/rapidgzip/.../blockfinder/Uncompressed.hpp:21-95).
    /// Scans byte boundaries for the LEN/~LEN coincidence pattern, then
    /// validates the 3-bit deflate header preceding the byte boundary.
    /// Returns all detected stored-block candidates (one per match), each
    /// emitted as a vendor-style offset pair: `bit_offset = offset.first`
    /// (the EARLIEST valid bit offset) and `seek_bit = offset.second`
    /// (`byte_bit - 3`, the actual decode seek position).
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
                    seek_bit: latest,
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
        dynamic.sort_by_key(|b| (b.bit_offset, b.seek_bit));
        dynamic
    }

    pub(crate) fn find_next_dynamic_block(
        &self,
        start_bit: usize,
        end_bit: usize,
    ) -> Option<BlockBoundary> {
        self.find_dynamic_blocks(start_bit, end_bit)
            .into_iter()
            .next()
    }

    pub(crate) fn find_next_uncompressed_block(
        &self,
        start_bit: usize,
        end_bit: usize,
    ) -> Option<BlockBoundary> {
        self.find_uncompressed_blocks(start_bit, end_bit)
            .into_iter()
            .next()
    }

    /// Count validated DYNAMIC deflate block headers in `[start_bit, end_bit)`,
    /// stopping as soon as `need` are found. DIAGNOSTIC-ONLY (the ignored
    /// `marker_density_measure` test in `single_member.rs`) — not on any
    /// production path.
    ///
    /// Kept as the instrument that FALSIFIED the "block-boundary density
    /// separates marker-heavy from marker-dormant" hypothesis (2026-07-05):
    /// measured mean dynamic-block gap logs=5 KiB / software=6 KiB /
    /// silesia=25 KiB — the high-ratio "marker-heavy" corpora have DENSER
    /// boundaries than silesia, so no density predicate can serialize them.
    /// Counts DYNAMIC headers only (each candidate has passed
    /// `validate_precode` + full canonical-Huffman validation); the stored
    /// `LEN/~LEN` finder is excluded (16-bit coincidence, high false-positive
    /// rate). Steps in bounded sub-windows; never materializes the whole
    /// range's candidate set.
    #[cfg(test)]
    pub(crate) fn count_dynamic_boundaries_capped(
        &self,
        start_bit: usize,
        end_bit: usize,
        need: usize,
    ) -> usize {
        const SUBWINDOW_BITS: usize = 32 * 1024 * 8;
        if need == 0 || start_bit >= end_bit {
            return 0;
        }
        let mut count = 0usize;
        let mut pos = start_bit;
        while pos < end_bit {
            let win_end = pos.saturating_add(SUBWINDOW_BITS).min(end_bit);
            count += self.find_dynamic_blocks(pos, win_end).len();
            if count >= need {
                return count;
            }
            pos = win_end;
        }
        count
    }

    /// Dynamic-Huffman block finder. Direct port of rapidgzip's
    /// `seekToNonFinalDynamicDeflateBlock` (DynamicHuffman.hpp:168-298) —
    /// the BRANCHLESS dual-sliding-bit-buffer scan.
    ///
    /// Two in-register bit buffers slide forward together (vendor keeps two
    /// buffers so the 74 bits needed for LUT+precode span more than a u64):
    ///   - `bit_buffer_for_lut`  : the 15 LUT bits at `offset`
    ///   - `bit_buffer_precode`  : the 61 bits at `offset + 13`
    ///                             (HCLEN[4] + 57 precode bits)
    /// Each position indexes the SIGNED LUT once. `next <= 0` ⇒ candidate:
    /// validate precode/Huffman straight from `bit_buffer_precode` — NO
    /// re-peek of the 17-bit header. The advance is `next.abs()`,
    /// UNCONDITIONAL (no `if skip > 0` branch), refilling the LUT buffer
    /// from the precode buffer's overlap and the precode buffer from the
    /// bit reader. This replaces the legacy per-position
    /// `peek(15)`/`if skip > 0`/`peek(17)` scan
    /// ([`Self::find_dynamic_blocks_scalar`], still used byte-identically
    /// for the &lt;74-bit tail where the two buffers cannot be fully loaded).
    ///
    /// BYTE-TRANSPARENT: emits the exact same `BlockBoundary` set as the
    /// scalar scan (locked by `branchless_scan_matches_scalar`). The LUT's
    /// candidate set is unchanged (`isDeflateCandidate` inspects only 13
    /// bits, so the 14- vs 15-bit prefix used by the skip encoding lands on
    /// the identical next candidate); every candidate runs the identical
    /// `validate_precode`/`parse_precode`/`validate_huffman_codes` gates.
    fn find_dynamic_blocks(&self, start_bit: usize, end_bit: usize) -> Vec<BlockBoundary> {
        const DUPLICATED_BITS: u32 = (LUT_BITS - 13) as u32; // 2
        const PRECODE_COUNT_BITS: u32 = 4;
        const ALL_PRECODE_BITS: u32 = 61;
        const PRECODE_57_MASK: u64 = (1u64 << 57) - 1;

        let data_bits = self.data.len() * 8;
        // The fast loop needs 13 + 61 = 74 bits of lookahead to fully load
        // both sliding buffers. Past that watermark, fall back to the
        // byte-identical scalar tail scan.
        let fast_end = end_bit.min(data_bits.saturating_sub(74));
        if start_bit >= fast_end {
            return self.find_dynamic_blocks_scalar(start_bit, end_bit);
        }

        let lut = get_lut();
        let mut blocks = Vec::new();
        let mut reader = BitReader::new(self.data);

        // Seed the two sliding bit buffers (rg DynamicHuffman.hpp:184-191).
        reader.seek_to_bit(start_bit);
        let mut bit_buffer_for_lut = reader.peek_refilled(LUT_BITS as u8);
        // The 61-bit precode buffer covers bits [start_bit+13, start_bit+74).
        // A single 61-bit read cannot be served right after a mid-byte seek
        // (the refill watermark leaves only 59 bits), so load it as 32 + 29.
        reader.seek_to_bit(start_bit + 13);
        let precode_lo = reader.read(32);
        let precode_hi = reader.read(ALL_PRECODE_BITS as u8 - 32);
        let mut bit_buffer_precode = precode_lo | (precode_hi << 32);
        // reader is now at the high watermark start_bit + 74.

        let mut offset = start_bit;
        while offset < fast_end {
            let next_position = lut[bit_buffer_for_lut as usize];
            let bits_to_load = next_position.unsigned_abs() as u32; // ∈ [1, 15]

            if next_position <= 0 {
                // Candidate — validate straight from the precode buffer.
                let next4 = (bit_buffer_precode & ((1 << PRECODE_COUNT_BITS) - 1)) as usize;
                let next57 = (bit_buffer_precode >> PRECODE_COUNT_BITS) & PRECODE_57_MASK;
                let precode_count = 4 + next4;
                if validate_precode(precode_count, next57) {
                    if let Some(precode_lengths) = self.parse_precode(precode_count, next57) {
                        let hlit = ((bit_buffer_for_lut >> 3) & 0x1F) as u8;
                        let hdist = ((bit_buffer_for_lut >> 8) & 0x1F) as u8;
                        let hlit_count = 257 + hlit as usize;
                        let lit_dist_count = hlit_count + (1 + hdist as usize);
                        // Validate the lit/dist code lengths from a fresh
                        // reader positioned just past the precode bits;
                        // keeps the hot-loop reader/buffers untouched.
                        let mut vreader = BitReader::new(self.data);
                        vreader.seek_to_bit(offset + 17 + precode_count * 3);
                        if self.validate_huffman_codes(
                            &mut vreader,
                            &precode_lengths,
                            lit_dist_count,
                            hlit_count,
                        ) {
                            blocks.push(BlockBoundary {
                                bit_offset: offset,
                                seek_bit: offset,
                                valid: true,
                                hlit,
                                hdist,
                                hclen: next4 as u8,
                            });
                        }
                    }
                }
            }

            // Unconditional branchless slide (rg DynamicHuffman.hpp:274-291).
            bit_buffer_for_lut >>= bits_to_load;
            bit_buffer_for_lut |= ((bit_buffer_precode >> DUPLICATED_BITS)
                & ((1u64 << bits_to_load) - 1))
                << (LUT_BITS as u32 - bits_to_load);
            bit_buffer_precode >>= bits_to_load;
            bit_buffer_precode |=
                reader.read(bits_to_load as u8) << (ALL_PRECODE_BITS - bits_to_load);
            offset += bits_to_load as usize;
        }

        // Byte-identical tail for the last <74 bits where the buffers
        // cannot be safely loaded.
        if offset < end_bit {
            blocks.extend(self.find_dynamic_blocks_scalar(offset, end_bit));
        }
        blocks
    }

    /// Legacy per-position scalar scan — the reference implementation the
    /// branchless [`Self::find_dynamic_blocks`] must match byte-for-byte,
    /// and the tail handler for the final <74 bits. Kept verbatim so the
    /// `branchless_scan_matches_scalar` differential has an independent
    /// oracle.
    fn find_dynamic_blocks_scalar(&self, start_bit: usize, end_bit: usize) -> Vec<BlockBoundary> {
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
                            seek_bit: bit_offset,
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
                    // BTYPE=01 (fixed-Huffman): dynamic-only finder — no
                    // candidate emitted. Faithful to vendor
                    // blockfinder/DynamicHuffman.hpp (no fixed finder);
                    // fixed-block regions degrade to confirmed-chain decode
                    // exactly as vendor. The BTYPE=01 prefilter was deleted
                    // 2026-06-09: it produced phantom candidates at 4 MiB
                    // grid-base positions on bignasa (BFINAL=1 random bytes
                    // passing the 2-symbol prefilter → 23–495-byte phantom
                    // final-block streams → ~150 ms confirmed re-decode per
                    // phantom; 3–9/run ≈ 39 % of T8 wall).
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
                                seek_bit: bit_offset,
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

    /// Check if lengths form a valid (canonical) Huffman code.
    ///
    /// Literal port of vendor's `checkHuffmanCodeLengths<MAX_CODE_LENGTH>`
    /// at `src/huffman/HuffmanCodingBase.hpp:215-236`. Tests Kraft
    /// **equality** (full canonical tree, sum-of-leaves == 2^MAX_LEN),
    /// with the standard edge case allowing a single-symbol tree
    /// (`sum == 2^(MAX_LEN - 1)` AND no code length > 1).
    ///
    /// The prior implementation only tested Kraft **inequality**
    /// (`code <= 2^bits` at every step), which accepts non-canonical
    /// trees with missing codes — admitting false-positive block
    /// candidates that fail later in `try_speculative_decode_candidate`.
    /// On silesia (May 2026), the loose check produced ~58 false
    /// positives per decode; vendor reports 0. Each FP triggers a full
    /// `bootstrap_with_marker_inflate` candidate trial before erroring
    /// out, accounting for ~0.55s of useless worker time.
    fn is_valid_huffman_lengths(&self, lengths: &[u8]) -> bool {
        // MAX_CODE_LENGTH for deflate literal/distance alphabets is 15
        // (RFC 1951 §3.2.7). Vendor uses the same constant via the
        // `MAX_CODE_LENGTH` template parameter.
        const MAX_CODE_LENGTH: u32 = 15;
        let mut virtual_leaf_count: u32 = 0;
        let mut greater_than_one: u32 = 0;
        for &len in lengths {
            if len == 0 || (len as u32) > MAX_CODE_LENGTH {
                if (len as u32) > MAX_CODE_LENGTH {
                    return false;
                }
                continue;
            }
            virtual_leaf_count += 1u32 << (MAX_CODE_LENGTH - len as u32);
            if len > 1 {
                greater_than_one += 1;
            }
        }
        // Single-symbol tree edge case: sum == 2^(MAX_LEN - 1) AND no
        // code length > 1 (vendor HuffmanCodingBase.hpp:226-234).
        if virtual_leaf_count == (1u32 << (MAX_CODE_LENGTH - 1)) {
            return greater_than_one == 0;
        }
        virtual_leaf_count == (1u32 << MAX_CODE_LENGTH)
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
        return DeflateBlockValidator::new(data).find_blocks(0, data_bits);
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
                let finder = DeflateBlockValidator::new(data);

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
    all_blocks.sort_by_key(|b| (b.bit_offset, b.seek_bit));
    all_blocks.dedup_by_key(|b| (b.bit_offset, b.seek_bit));

    all_blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plausible_trial_rejects_reserved_btype() {
        let data = [0b111u8];
        assert!(!plausible_trial_decode_offset(&data, 0));
    }

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

        // BTYPE=01 (fixed) — dynamic-only finder, no candidate emitted
        // (vendor DynamicHuffman.hpp parity; prefilter deleted 2026-06-09).
        assert!(lut[0b010] > 0);
        assert!(lut[0b011] > 0);

        // Valid: BTYPE=10 (dynamic), BFINAL=0, HLIT=0, HDIST=0.
        // Candidates are now encoded as NEGATIVE (sign = "is candidate",
        // magnitude = advance to next candidate) — formerly stored as 0.
        let valid = 0b00000_00000_10_0u32;
        assert!(lut[valid as usize] <= 0);

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
        let finder = DeflateBlockValidator::new(deflate);
        let mut found = 0;

        for cp in &scan.checkpoints {
            let real_bitsleft = (cp.bitsleft as u8) as usize;
            let bit_pos = cp.input_byte_pos * 8 - real_bitsleft;

            let mut reader = BitReader::new(deflate);
            reader.seek_to_bit(bit_pos);

            let lut_bits = reader.peek(LUT_BITS as u8) as usize;
            // Candidate positions are LUT <= 0 (negative-encoded).
            let lut_pass = lut_bits < lut.len() && lut[lut_bits] <= 0;

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

    /// BYTE-TRANSPARENCY GATE for the branchless scan port: on real
    /// deflate data the branchless `find_dynamic_blocks` must emit the
    /// EXACT same block-boundary set (offsets + hlit/hdist/hclen fields, in
    /// the same order) as the scalar reference `find_dynamic_blocks_scalar`.
    /// Exercised over several corpora shapes and several sub-ranges so the
    /// fast-region/tail hand-off is covered.
    #[test]
    fn branchless_scan_matches_scalar() {
        use std::io::Write;

        // A few different data shapes: mixed literal+match (default gz),
        // near-incompressible (random), and highly repetitive.
        let mut shapes: Vec<Vec<u8>> = Vec::new();
        {
            // Mixed / RLE-ish (same generator as the recall test).
            let mut d = Vec::with_capacity(3 * 1024 * 1024);
            let mut rng: u64 = 0xc0ffee;
            while d.len() < 3 * 1024 * 1024 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 32) % 5 < 3 {
                    d.push((rng >> 16) as u8);
                } else {
                    let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                    let repeat = ((rng >> 40) % 8 + 2) as usize;
                    for _ in 0..repeat.min(3 * 1024 * 1024 - d.len()) {
                        d.push(byte);
                    }
                }
            }
            shapes.push(d);
        }
        {
            // Near-incompressible: every position is a scan miss/candidate.
            let mut d = vec![0u8; 2 * 1024 * 1024];
            let mut rng: u64 = 0x1badb002;
            for b in &mut d {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (rng >> 24) as u8;
            }
            shapes.push(d);
        }

        for (si, data) in shapes.iter().enumerate() {
            let mut gz = Vec::new();
            {
                let mut enc =
                    flate2::write::GzEncoder::new(&mut gz, flate2::Compression::default());
                enc.write_all(data).unwrap();
                enc.finish().unwrap();
            }
            let header_size =
                crate::decompress::parallel::single_member::skip_gzip_header(&gz).unwrap();
            let deflate = &gz[header_size..gz.len() - 8];
            let total_bits = deflate.len() * 8;
            let finder = DeflateBlockValidator::new(deflate);

            // Full range plus a few offset sub-ranges (exercise fast/tail
            // hand-off and mid-stream starts, mirroring the parallel finder).
            let ranges = [
                (0usize, total_bits),
                (0, total_bits / 2),
                (total_bits / 3, total_bits),
                (total_bits.saturating_sub(200), total_bits),
            ];
            for &(s, e) in &ranges {
                let fast = finder.find_dynamic_blocks(s, e);
                let scalar = finder.find_dynamic_blocks_scalar(s, e);
                assert_eq!(
                    fast.len(),
                    scalar.len(),
                    "shape {si} range {s}..{e}: count {} != scalar {}",
                    fast.len(),
                    scalar.len()
                );
                for (a, b) in fast.iter().zip(scalar.iter()) {
                    assert_eq!(
                        a.bit_offset, b.bit_offset,
                        "shape {si} range {s}..{e}: offset"
                    );
                    assert_eq!(a.seek_bit, b.seek_bit, "shape {si} range {s}..{e}: seek");
                    assert_eq!(a.hlit, b.hlit, "shape {si} range {s}..{e}: hlit");
                    assert_eq!(a.hdist, b.hdist, "shape {si} range {s}..{e}: hdist");
                    assert_eq!(a.hclen, b.hclen, "shape {si} range {s}..{e}: hclen");
                }
            }
            // Guard against a vacuous match: the mixed/compressible shape
            // must yield dynamic boundaries. (The near-incompressible shape
            // legitimately compresses to stored blocks — zero dynamic
            // boundaries — but the branchless scan still runs over every
            // byte, and the count/field equality above proves transparency.)
            if si == 0 {
                assert!(
                    !finder.find_dynamic_blocks(0, total_bits).is_empty(),
                    "shape {si}: expected at least one dynamic boundary"
                );
            }
        }
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

        let sequential = DeflateBlockValidator::new(deflate).find_blocks(0, deflate.len() * 8);
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
        let mut rng_data = Vec::with_capacity(64 * 1024);
        let mut rng: u64 = 0xabc123;
        while rng_data.len() < 64 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_data.push((rng >> 16) as u8);
        }
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        use std::io::Write;
        enc.write_all(&rng_data).unwrap();
        let data = enc.finish().unwrap();
        let sequential = DeflateBlockValidator::new(&data).find_blocks(0, data.len() * 8);
        let parallel = find_blocks_parallel(&data, 4);
        assert_eq!(
            sequential.len(),
            parallel.len(),
            "small span: parallel must match sequential count"
        );
        assert!(
            !sequential.is_empty(),
            "random deflate must yield at least one dynamic boundary"
        );
    }

    #[test]
    fn test_find_blocks_parallel_single_thread() {
        let mut rng_data = vec![0u8; 256 * 1024];
        let mut rng: u64 = 0xfeedface;
        for b in &mut rng_data {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 16) as u8;
        }
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        use std::io::Write;
        enc.write_all(&rng_data).unwrap();
        let data = enc.finish().unwrap();
        let sequential = DeflateBlockValidator::new(&data).find_blocks(0, data.len() * 8);
        let parallel = find_blocks_parallel(&data, 1);
        assert_eq!(
            sequential.len(),
            parallel.len(),
            "T1: parallel must match sequential"
        );
        assert!(
            !sequential.is_empty(),
            "random deflate must yield at least one dynamic boundary"
        );
    }

    /// Mirrors benchmarks.yml random-data: 10 MiB incompressible input,
    /// L9 pipelined compress. The tail partition (from 10 MiB bit offset)
    /// often contains only fixed-Huffman blocks.
    ///
    /// NEW EXPECTATION (2026-06-09, vendor DynamicHuffman.hpp parity):
    /// The dynamic-only block finder produces NO fixed-Huffman candidates —
    /// this is the correct vendor behavior. Fixed-block tail regions are
    /// handled by the confirmed-chain re-decode path in the consumer.
    /// The test verifies: (a) the finder does not panic, and (b) all
    /// candidates returned are dynamic-Huffman (BTYPE=10) or stored
    /// (BTYPE=00), never fixed-Huffman (BTYPE=01).
    #[test]
    fn test_find_blocks_finds_fixed_huffman_in_tail_partition() {
        use crate::compress::pipelined::PipelinedGzEncoder;
        use std::io::Cursor;

        let mut original = vec![0u8; 10 * 1024 * 1024];
        let mut state = 0x12345678u64;
        for b in &mut original {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (state >> 32) as u8;
        }

        let encoder = PipelinedGzEncoder::new(9, 4);
        let mut compressed = Vec::new();
        encoder
            .compress(Cursor::new(&original), &mut compressed)
            .unwrap();
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "fixture must exceed 10 MiB parallel gate (got {} bytes)",
            compressed.len()
        );

        let header_size =
            crate::decompress::format::parse_gzip_header_size(&compressed).unwrap_or(10);
        let deflate = &compressed[header_size..compressed.len() - 8];
        let partition_bit = 10 * 1024 * 1024 * 8;
        let finder = DeflateBlockValidator::new(deflate);
        // Dynamic-only finder (vendor DynamicHuffman.hpp parity): no
        // fixed-Huffman candidates are emitted. Verify no BTYPE=01 slips
        // through — each candidate's 3-bit header must be BTYPE=00 or
        // BTYPE=10 (the dynamic LUT already filters BTYPE=01 out, but
        // belt-and-suspenders).
        let candidates = finder.find_blocks(partition_bit, deflate.len() * 8);
        for c in &candidates {
            // Peek the 3-bit header at the candidate's seek_bit.
            let byte = c.seek_bit / 8;
            let shift = c.seek_bit % 8;
            if byte + 1 < deflate.len() {
                let raw = (deflate[byte] as u32) | ((deflate[byte + 1] as u32) << 8);
                let header = (raw >> shift) & 0b111;
                let btype = (header >> 1) & 0b11;
                assert_ne!(
                    btype, 1,
                    "fixed-Huffman candidate at bit {}: vendor parity requires dynamic-only finder",
                    c.seek_bit
                );
            }
        }
    }
}
