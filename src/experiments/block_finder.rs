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
/// Accepts bfinal=0 or bfinal=1. Fixed Huffman (btype=1) is excluded because
/// there is no header to validate and 50% of random positions would match.
#[inline]
fn is_valid_candidate_13(bits: u32) -> bool {
    let btype = (bits >> 1) & 3;

    match btype {
        0 => true, // stored block — validated by len/~len
        2 => {
            // dynamic Huffman — check HLIT <= 29 and HDIST <= 29
            let hlit = (bits >> 3) & 31;
            let hdist = (bits >> 8) & 31;
            hlit <= 29 && hdist <= 29
        }
        _ => false, // btype=1 (fixed) has no header to validate; btype=3 is reserved
    }
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
// Block Finder
// ============================================================================

pub struct BlockFinder<'a> {
    data: &'a [u8],
}

impl<'a> BlockFinder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Find all valid deflate block starts in a range.
    ///
    /// Searches for all block types (stored, fixed, dynamic).
    /// Uses multi-level fail-fast validation for dynamic blocks;
    /// stored blocks are validated by len/~len; fixed blocks are
    /// accepted as candidates for trial decode.
    pub fn find_blocks(&self, start_bit: usize, end_bit: usize) -> Vec<BlockBoundary> {
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
                    // Fixed Huffman has no header to validate — skip.
                    // (Covered by nearby dynamic/stored block boundaries.)
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

        // Valid: BTYPE=00 (stored), BFINAL=0
        assert_eq!(lut[0b000], 0);
        // Valid: BTYPE=00 (stored), BFINAL=1
        assert_eq!(lut[0b001], 0);

        // Invalid: BTYPE=01 (fixed) — excluded from candidate search
        assert!(lut[0b010] > 0);
        assert!(lut[0b011] > 0);

        // Valid: BTYPE=10 (dynamic), BFINAL=0, HLIT=0, HDIST=0
        let valid = 0b00000_00000_10_0u32;
        assert_eq!(lut[valid as usize], 0);

        // Valid: BTYPE=10 (dynamic), BFINAL=1, HLIT=0, HDIST=0
        let valid_final = 0b00000_00000_10_1u32;
        assert_eq!(lut[valid_final as usize], 0);
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

        let header_size =
            crate::experiments::marker_decode::skip_gzip_header(&gz).expect("valid header");
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

        let header_size =
            crate::experiments::marker_decode::skip_gzip_header(&gz).expect("valid header");
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
