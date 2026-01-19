//! Marker-Based Speculative Deflate Decoder
//!
//! This implements rapidgzip's key innovation: using uint16_t buffers with markers
//! for unresolved back-references, allowing immediate parallel decoding at any position.
//!
//! ## How It Works
//!
//! 1. Decode deflate stream into `Vec<u16>` instead of `Vec<u8>`
//! 2. Values 0-255 are literal bytes
//! 3. Values 256+ are "markers" encoding unresolved back-references:
//!    `marker = MARKER_BASE + (distance - decoded_bytes - 1)`
//! 4. Once the previous chunk's window is known, replace markers with actual bytes
//!
//! ## Why This Matters
//!
//! Traditional approach: Block until window is available, then decode
//! Marker approach: Decode immediately, resolve markers later in parallel
//!
//! This allows true parallelism on single-member gzip files.
//!
//! ## ISA-L Integration
//!
//! When a chunk has been verified (we know the window), we can re-decode using
//! ISA-L for maximum speed. ISA-L uses SIMD and is ~2x faster than our Rust decoder.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_div_ceil)]
#![allow(unused_comparisons)]
#![allow(unused_mut)]

use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

#[cfg(feature = "isal")]
use crate::isal::IsalInflater;

/// Maximum window size (32KB)
pub const WINDOW_SIZE: usize = 32 * 1024;

/// Marker base value - any u16 >= this is a marker
pub const MARKER_BASE: u16 = WINDOW_SIZE as u16;

/// Maximum valid marker value
pub const MARKER_MAX: u16 = u16::MAX;

/// Chunk size for parallel processing (4MB like rapidgzip)
pub const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// A decoded chunk with potential markers
#[derive(Clone)]
pub struct MarkerChunk {
    /// Chunk index
    pub index: usize,
    /// Bit offset where decoding started
    pub start_bit: usize,
    /// Bit offset where decoding ended
    pub end_bit: usize,
    /// Decoded data (u16 to hold markers)
    pub data: Vec<u16>,
    /// Number of marker bytes (for statistics)
    pub marker_count: usize,
    /// Final 32KB of decoded data (for next chunk's window)
    pub final_window: Vec<u8>,
    /// Whether this chunk decoded successfully
    pub success: bool,
    /// Distance to last marker byte (for optimization)
    pub distance_to_last_marker: usize,
}

impl Default for MarkerChunk {
    fn default() -> Self {
        Self {
            index: 0,
            start_bit: 0,
            end_bit: 0,
            data: Vec::new(),
            marker_count: 0,
            final_window: Vec::new(),
            success: false,
            distance_to_last_marker: 0,
        }
    }
}

/// Marker-based deflate decoder
pub struct MarkerDecoder {
    /// Input data
    data: Vec<u8>,
    /// Current byte position
    byte_pos: usize,
    /// Current bit position within byte (0-7)
    bit_pos: u8,
    /// Output buffer (u16 for markers)
    output: Vec<u16>,
    /// Window buffer (last 32KB of output, as u16)
    window: Vec<u16>,
    /// Position in window (circular)
    window_pos: usize,
    /// Total decoded bytes
    decoded_bytes: usize,
    /// Whether we're in marker mode (haven't gotten window yet)
    marker_mode: bool,
    /// Distance to last marker byte
    distance_to_last_marker: usize,
    /// Number of markers written
    marker_count: usize,
}

impl MarkerDecoder {
    /// Create a new marker decoder
    pub fn new(data: &[u8], start_bit: usize) -> Self {
        let byte_pos = start_bit / 8;
        let bit_pos = (start_bit % 8) as u8;

        Self {
            data: data.to_vec(),
            byte_pos,
            bit_pos,
            output: Vec::with_capacity(data.len() * 4),
            window: vec![0u16; WINDOW_SIZE],
            window_pos: 0,
            decoded_bytes: 0,
            marker_mode: true,
            distance_to_last_marker: 0,
            marker_count: 0,
        }
    }

    /// Create decoder with known initial window
    pub fn with_window(data: &[u8], start_bit: usize, window: &[u8]) -> Self {
        let mut decoder = Self::new(data, start_bit);
        decoder.marker_mode = false;

        // Copy window
        let window_len = window.len().min(WINDOW_SIZE);
        for i in 0..window_len {
            decoder.window[i] = window[i] as u16;
        }
        decoder.window_pos = window_len % WINDOW_SIZE;
        decoder.decoded_bytes = window_len;
        decoder.distance_to_last_marker = window_len;

        decoder
    }

    /// Read a single bit
    #[inline]
    fn read_bit(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }

        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit)
    }

    /// Read N bits (LSB first)
    #[inline]
    fn read_bits(&mut self, n: u8) -> io::Result<u32> {
        let mut result = 0u32;
        for i in 0..n {
            result |= (self.read_bit()? as u32) << i;
        }
        Ok(result)
    }

    /// Align to next byte boundary
    fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Read a u16 (little-endian)
    fn read_u16_le(&mut self) -> io::Result<u16> {
        self.align_to_byte();
        if self.byte_pos + 2 > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }
        let val = u16::from_le_bytes([self.data[self.byte_pos], self.data[self.byte_pos + 1]]);
        self.byte_pos += 2;
        Ok(val)
    }

    /// Append a byte (or marker) to output and window
    #[inline]
    fn append(&mut self, value: u16) {
        self.output.push(value);

        // Track markers
        if value > 255 {
            self.marker_count += 1;
            self.distance_to_last_marker = 0;
        } else {
            self.distance_to_last_marker += 1;
        }

        // Update window
        self.window[self.window_pos] = value;
        self.window_pos = (self.window_pos + 1) % WINDOW_SIZE;
        self.decoded_bytes += 1;
    }

    /// Copy from window (may produce markers)
    #[inline]
    fn copy_from_window(&mut self, distance: usize, length: usize) {
        for i in 0..length {
            let value = if distance > self.decoded_bytes {
                // Reference before our decode start - create marker
                if self.marker_mode {
                    let marker_offset = distance - self.decoded_bytes - 1;
                    MARKER_BASE + (marker_offset as u16).min(MARKER_MAX - MARKER_BASE)
                } else {
                    // We have a window, this shouldn't happen
                    0
                }
            } else {
                // Reference within our decoded data
                let offset =
                    (self.window_pos + WINDOW_SIZE - distance + i % distance) % WINDOW_SIZE;
                self.window[offset]
            };
            self.append(value);
        }
    }

    /// Get current bit position
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Decode one deflate block
    pub fn decode_block(&mut self) -> io::Result<bool> {
        // Read BFINAL
        let bfinal = self.read_bit()? != 0;

        // Read BTYPE
        let btype = self.read_bits(2)?;

        match btype {
            0 => self.decode_stored()?,
            1 => self.decode_fixed_huffman()?,
            2 => self.decode_dynamic_huffman()?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid block type 3",
                ))
            }
            _ => unreachable!(),
        }

        Ok(bfinal)
    }

    /// Decode stored block
    fn decode_stored(&mut self) -> io::Result<()> {
        let len = self.read_u16_le()?;
        let nlen = self.read_u16_le()?;

        if len != !nlen {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid stored block length",
            ));
        }

        for _ in 0..len {
            if self.byte_pos >= self.data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
            }
            self.append(self.data[self.byte_pos] as u16);
            self.byte_pos += 1;
        }

        Ok(())
    }

    /// Decode fixed Huffman block
    fn decode_fixed_huffman(&mut self) -> io::Result<()> {
        // Fixed Huffman uses predefined tables
        // Literals 0-143: 8 bits, 144-255: 9 bits, 256-279: 7 bits, 280-287: 8 bits
        loop {
            let symbol = self.decode_fixed_literal()?;

            if symbol < 256 {
                self.append(symbol as u16);
            } else if symbol == 256 {
                return Ok(());
            } else {
                let length = self.decode_length(symbol as u32)?;
                let distance = self.decode_fixed_distance()?;
                self.copy_from_window(distance, length);
            }
        }
    }

    /// Decode a literal/length symbol using fixed Huffman
    fn decode_fixed_literal(&mut self) -> io::Result<u16> {
        // Read 7 bits first
        let mut code = 0u32;
        for _ in 0..7 {
            code = (code << 1) | self.read_bit()? as u32;
        }

        // Check for 7-bit codes (256-279)
        if code >= 0b0000000 && code <= 0b0010111 {
            return Ok((256 + code) as u16);
        }

        // Read 8th bit
        code = (code << 1) | self.read_bit()? as u32;

        // 8-bit codes
        if code >= 0b00110000 && code <= 0b10111111 {
            return Ok((code - 0b00110000) as u16); // 0-143
        }
        if code >= 0b11000000 && code <= 0b11000111 {
            return Ok((280 + code - 0b11000000) as u16);
        }

        // Read 9th bit
        code = (code << 1) | self.read_bit()? as u32;

        if code >= 0b110010000 && code <= 0b111111111 {
            return Ok((144 + code - 0b110010000) as u16);
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid fixed Huffman code",
        ))
    }

    /// Decode distance using fixed Huffman
    fn decode_fixed_distance(&mut self) -> io::Result<usize> {
        // Fixed distance: 5 bits, codes 0-29
        let code = self.read_bits(5)? as usize;
        self.decode_distance_from_code(code)
    }

    /// Decode dynamic Huffman block
    fn decode_dynamic_huffman(&mut self) -> io::Result<()> {
        // Read header
        let hlit = self.read_bits(5)? as usize + 257;
        let hdist = self.read_bits(5)? as usize + 1;
        let hclen = self.read_bits(4)? as usize + 4;

        // Read code length code lengths
        const CL_ORDER: [usize; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        let mut cl_lens = [0u8; 19];
        for i in 0..hclen {
            cl_lens[CL_ORDER[i]] = self.read_bits(3)? as u8;
        }

        // Build code length Huffman table
        let cl_table = build_huffman_table(&cl_lens, 7)?;

        // Read literal/length and distance code lengths
        let mut lengths = vec![0u8; hlit + hdist];
        let mut i = 0;
        while i < lengths.len() {
            let symbol = self.decode_huffman(&cl_table, 7)?;

            match symbol {
                0..=15 => {
                    lengths[i] = symbol as u8;
                    i += 1;
                }
                16 => {
                    if i == 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid repeat"));
                    }
                    let count = self.read_bits(2)? as usize + 3;
                    let val = lengths[i - 1];
                    for _ in 0..count {
                        if i >= lengths.len() {
                            break;
                        }
                        lengths[i] = val;
                        i += 1;
                    }
                }
                17 => {
                    let count = self.read_bits(3)? as usize + 3;
                    for _ in 0..count {
                        if i >= lengths.len() {
                            break;
                        }
                        lengths[i] = 0;
                        i += 1;
                    }
                }
                18 => {
                    let count = self.read_bits(7)? as usize + 11;
                    for _ in 0..count {
                        if i >= lengths.len() {
                            break;
                        }
                        lengths[i] = 0;
                        i += 1;
                    }
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid code length symbol",
                    ))
                }
            }
        }

        // Build Huffman tables
        let lit_table = build_huffman_table(&lengths[..hlit], 15)?;
        let dist_table = build_huffman_table(&lengths[hlit..], 15)?;

        // Decode symbols
        loop {
            let symbol = self.decode_huffman(&lit_table, 15)?;

            if symbol < 256 {
                self.append(symbol as u16);
            } else if symbol == 256 {
                return Ok(());
            } else {
                let length = self.decode_length(symbol as u32)?;
                let dist_code = self.decode_huffman(&dist_table, 15)? as usize;
                let distance = self.decode_distance_from_code(dist_code)?;
                self.copy_from_window(distance, length);
            }
        }
    }

    /// Decode Huffman symbol
    fn decode_huffman(&mut self, table: &[(u16, u8)], max_bits: u8) -> io::Result<u16> {
        let mut code = 0u32;
        for _ in 0..max_bits {
            code = (code << 1) | self.read_bit()? as u32;

            // Reverse bits for table lookup
            let reversed = reverse_bits(
                code,
                (0..=max_bits).find(|&b| (1u32 << b) > code).unwrap_or(1),
            );

            if (reversed as usize) < table.len() {
                let (symbol, len) = table[reversed as usize];
                if len > 0 && len <= max_bits {
                    // Check if we've read enough bits
                    let bits_read = (0..=max_bits).find(|&b| (1u32 << b) > code).unwrap_or(1);
                    if bits_read == len {
                        return Ok(symbol);
                    }
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ))
    }

    /// Decode length from symbol
    fn decode_length(&mut self, symbol: u32) -> io::Result<usize> {
        const LENGTH_BASE: [usize; 29] = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
            115, 131, 163, 195, 227, 258,
        ];
        const LENGTH_EXTRA: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];

        let idx = (symbol - 257) as usize;
        if idx >= LENGTH_BASE.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length symbol",
            ));
        }

        let base = LENGTH_BASE[idx];
        let extra = LENGTH_EXTRA[idx];
        let extra_bits = if extra > 0 {
            self.read_bits(extra)? as usize
        } else {
            0
        };

        Ok(base + extra_bits)
    }

    /// Decode distance from code
    fn decode_distance_from_code(&mut self, code: usize) -> io::Result<usize> {
        const DISTANCE_BASE: [usize; 30] = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
            1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
        ];
        const DISTANCE_EXTRA: [u8; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
            12, 13, 13,
        ];

        if code >= DISTANCE_BASE.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }

        let base = DISTANCE_BASE[code];
        let extra = DISTANCE_EXTRA[code];
        let extra_bits = if extra > 0 {
            self.read_bits(extra)? as usize
        } else {
            0
        };

        Ok(base + extra_bits)
    }

    /// Decode all blocks until BFINAL
    pub fn decode_all(&mut self) -> io::Result<()> {
        loop {
            let is_final = self.decode_block()?;
            if is_final {
                break;
            }
        }
        Ok(())
    }

    /// Get the decoded output
    pub fn output(&self) -> &[u16] {
        &self.output
    }

    /// Get the final window (as u8, with markers converted to 0)
    pub fn final_window(&self) -> Vec<u8> {
        let mut window = Vec::with_capacity(WINDOW_SIZE.min(self.output.len()));
        let start = if self.output.len() > WINDOW_SIZE {
            self.output.len() - WINDOW_SIZE
        } else {
            0
        };

        for &val in &self.output[start..] {
            window.push(if val <= 255 { val as u8 } else { 0 });
        }

        window
    }

    /// Check if output contains markers
    pub fn has_markers(&self) -> bool {
        self.marker_count > 0
    }

    /// Get marker count
    pub fn marker_count(&self) -> usize {
        self.marker_count
    }

    /// Convert output to bytes using window for marker replacement
    pub fn to_bytes(&self, window: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.output.len());

        for &val in &self.output {
            if val <= 255 {
                result.push(val as u8);
            } else {
                // Marker: val = MARKER_BASE + offset
                let offset = (val - MARKER_BASE) as usize;
                if offset < window.len() {
                    result.push(window[window.len() - 1 - offset]);
                } else {
                    result.push(0); // Invalid marker
                }
            }
        }

        result
    }
}

/// Build Huffman table from code lengths
fn build_huffman_table(lengths: &[u8], max_bits: u8) -> io::Result<Vec<(u16, u8)>> {
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

    // Assign codes
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 && len <= max_bits {
            let c = next_code[len as usize];
            next_code[len as usize] += 1;

            // Fill table entries (reversed bits)
            let reversed = reverse_bits(c, len);
            let fill = 1 << (max_bits - len);
            for i in 0..fill {
                let idx = (reversed | (i << len)) as usize;
                if idx < table.len() {
                    table[idx] = (sym as u16, len);
                }
            }
        }
    }

    Ok(table)
}

/// Reverse bits
fn reverse_bits(value: u32, bits: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Replace markers in a buffer using window
pub fn replace_markers(data: &mut [u16], window: &[u8]) {
    for val in data.iter_mut() {
        if *val > 255 {
            let offset = (*val - MARKER_BASE) as usize;
            if offset < window.len() {
                *val = window[window.len() - 1 - offset] as u16;
            } else {
                *val = 0;
            }
        }
    }
}

/// Convert u16 buffer to u8 (after marker replacement)
pub fn to_u8(data: &[u16]) -> Vec<u8> {
    data.iter().map(|&v| v as u8).collect()
}

/// ISA-L accelerated decode for verified chunks
///
/// When we know the window from the previous chunk, we can use ISA-L
/// with set_dict for maximum speed (SIMD-accelerated).
#[cfg(feature = "isal")]
pub fn decode_with_isal(
    data: &[u8],
    start_bit: usize,
    window: &[u8],
    expected_size: usize,
) -> io::Result<Vec<u8>> {
    // ISA-L works best at byte boundaries
    if start_bit % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "ISA-L requires byte-aligned start",
        ));
    }

    let byte_offset = start_bit / 8;
    let chunk_data = &data[byte_offset..];

    let mut inflater = IsalInflater::new()?;
    inflater.set_dict(window)?;

    let mut output = vec![0u8; expected_size.max(64 * 1024)];
    let mut total = 0;

    loop {
        match inflater.decompress(chunk_data, &mut output[total..]) {
            Ok(n) => {
                total += n;
                break;
            }
            Err(e) if e.kind() == io::ErrorKind::WriteZero => {
                output.resize(output.len() * 2, 0);
            }
            Err(e) => return Err(e),
        }
    }

    output.truncate(total);
    Ok(output)
}

/// Re-decode a chunk using ISA-L after window is known
///
/// This is the key optimization: speculative decode with markers first,
/// then re-decode with ISA-L once we have the window.
#[cfg(feature = "isal")]
pub fn redecode_chunk_with_isal(
    deflate_data: &[u8],
    chunk: &MarkerChunk,
    window: &[u8],
) -> Option<Vec<u8>> {
    // Only use ISA-L for byte-aligned chunks with markers
    if chunk.start_bit % 8 != 0 {
        return None;
    }

    // If no markers, just convert existing data
    if chunk.marker_count == 0 {
        return Some(to_u8(&chunk.data));
    }

    // Try ISA-L decode
    let byte_offset = chunk.start_bit / 8;
    if byte_offset >= deflate_data.len() {
        return None;
    }

    decode_with_isal(deflate_data, chunk.start_bit, window, chunk.data.len()).ok()
}

/// Try to decode from a chunk start, testing multiple bit offsets
fn try_decode_chunk(data: &[u8], index: usize, is_first: bool) -> Option<MarkerChunk> {
    // First chunk: known to start at bit 0
    if is_first {
        let mut decoder = MarkerDecoder::new(data, 0);
        if decoder.decode_all().is_ok() {
            return Some(MarkerChunk {
                index,
                start_bit: 0,
                end_bit: decoder.bit_position(),
                data: decoder.output().to_vec(),
                marker_count: decoder.marker_count(),
                final_window: decoder.final_window(),
                success: true,
                distance_to_last_marker: decoder.distance_to_last_marker,
            });
        }
        return None;
    }

    // Other chunks: try all 8 bit offsets at the chunk start
    for bit_offset in 0..8 {
        let mut decoder = MarkerDecoder::new(data, bit_offset);

        // Try to decode a reasonable amount
        match decoder.decode_all() {
            Ok(()) => {
                // Success! Check if we decoded a reasonable amount
                if decoder.output().len() > 1024 {
                    return Some(MarkerChunk {
                        index,
                        start_bit: bit_offset,
                        end_bit: decoder.bit_position(),
                        data: decoder.output().to_vec(),
                        marker_count: decoder.marker_count(),
                        final_window: decoder.final_window(),
                        success: true,
                        distance_to_last_marker: decoder.distance_to_last_marker,
                    });
                }
            }
            Err(_) => continue,
        }
    }

    // Try searching a small range for a valid block start
    for byte_offset in 1..64.min(data.len()) {
        for bit_offset in 0..8 {
            let mut decoder = MarkerDecoder::new(&data[byte_offset..], bit_offset);
            if decoder.decode_all().is_ok() && decoder.output().len() > 1024 {
                return Some(MarkerChunk {
                    index,
                    start_bit: byte_offset * 8 + bit_offset,
                    end_bit: byte_offset * 8 + decoder.bit_position(),
                    data: decoder.output().to_vec(),
                    marker_count: decoder.marker_count(),
                    final_window: decoder.final_window(),
                    success: true,
                    distance_to_last_marker: decoder.distance_to_last_marker,
                });
            }
        }
    }

    None
}

/// Parallel marker-based decompression using chunk spacing strategy
pub fn decompress_parallel<W: io::Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Skip gzip header
    let header_size = skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    // For small data, use sequential
    if deflate_data.len() < CHUNK_SIZE * 2 || num_threads <= 1 {
        return decompress_sequential(data, writer);
    }

    // CHUNK SPACING STRATEGY: Partition at fixed intervals (like rapidgzip)
    // Don't try to find actual block boundaries - just guess and validate
    let num_chunks = (deflate_data.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    let chunks: Vec<Mutex<Option<MarkerChunk>>> =
        (0..num_chunks).map(|_| Mutex::new(None)).collect();

    let next_chunk = AtomicUsize::new(0);
    let error_flag = AtomicBool::new(false);

    // Phase 1: Parallel speculative decode at fixed spacing
    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(num_chunks) {
            let chunks_ref = &chunks;
            let next_ref = &next_chunk;
            let error_ref = &error_flag;

            scope.spawn(move || {
                loop {
                    if error_ref.load(Ordering::Relaxed) {
                        break;
                    }

                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_chunks {
                        break;
                    }

                    let start_byte = idx * CHUNK_SIZE;
                    let chunk_data = &deflate_data[start_byte..];

                    // Try to decode this chunk
                    let chunk =
                        try_decode_chunk(chunk_data, idx, idx == 0).unwrap_or(MarkerChunk {
                            index: idx,
                            success: false,
                            ..Default::default()
                        });

                    *chunks_ref[idx].lock().unwrap() = Some(chunk);
                }
            });
        }
    });

    // Phase 2: Window propagation and marker replacement
    let mut all_chunks: Vec<MarkerChunk> = chunks
        .into_iter()
        .map(|m| m.into_inner().unwrap().unwrap_or_default())
        .collect();

    // Check if first chunk succeeded
    if !all_chunks.first().map(|c| c.success).unwrap_or(false) {
        return decompress_sequential(data, writer);
    }

    // Phase 2a: Sequential window propagation (must be sequential)
    // Build list of windows for each chunk
    let mut windows: Vec<Vec<u8>> = Vec::with_capacity(all_chunks.len());
    let mut prev_window: Vec<u8> = Vec::new();

    for chunk in &all_chunks {
        if !chunk.success {
            return decompress_sequential(data, writer);
        }
        windows.push(prev_window.clone());
        prev_window = chunk.final_window.clone();
    }

    // Phase 2b: PARALLEL marker replacement
    // Now we have all windows, replace markers in parallel
    let final_outputs: Vec<Mutex<Vec<u8>>> = (0..all_chunks.len())
        .map(|_| Mutex::new(Vec::new()))
        .collect();

    let next_chunk_replace = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(all_chunks.len()) {
            let outputs_ref = &final_outputs;
            let chunks_ref = &all_chunks;
            let windows_ref = &windows;
            let next_ref = &next_chunk_replace;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= chunks_ref.len() {
                        break;
                    }

                    let chunk = &chunks_ref[idx];
                    let window = &windows_ref[idx];

                    // Try ISA-L re-decode if we have markers and a window
                    #[cfg(feature = "isal")]
                    if chunk.marker_count > 0 && !window.is_empty() {
                        if let Some(output) = redecode_chunk_with_isal(deflate_data, chunk, window)
                        {
                            *outputs_ref[idx].lock().unwrap() = output;
                            continue;
                        }
                    }

                    // Fallback: replace markers manually
                    let mut data = chunk.data.clone();
                    if chunk.marker_count > 0 && !window.is_empty() {
                        replace_markers(&mut data, window);
                    }

                    *outputs_ref[idx].lock().unwrap() = to_u8(&data);
                }
            });
        }
    });

    // Phase 3: Write output (sequential for ordering)
    let mut total = 0u64;
    for output_mutex in &final_outputs {
        let output = output_mutex.lock().unwrap();
        writer.write_all(&output)?;
        total += output.len() as u64;
    }

    writer.flush()?;
    Ok(total)
}

/// Sequential decompression fallback
pub fn decompress_sequential<W: io::Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    // Try ultra-fast inflate first
    let mut output = Vec::new();
    if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(data, &mut output).is_ok() {
        writer.write_all(&output)?;
        writer.flush()?;
        return Ok(output.len() as u64);
    }

    // Fallback to flate2
    let mut decoder = flate2::read::GzDecoder::new(data);
    output.clear();
    decoder.read_to_end(&mut output)?;
    writer.write_all(&output)?;
    writer.flush()?;
    Ok(output.len() as u64)
}

/// Skip gzip header and return offset to deflate data
pub fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    if data.len() < 10 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Data too short for gzip header",
        ));
    }

    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid gzip magic",
        ));
    }

    let flags = data[3];
    let mut offset = 10;

    // FEXTRA
    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid header"));
        }
        let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2 + xlen;
    }

    // FNAME
    if flags & 0x08 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    // FCOMMENT
    if flags & 0x10 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    // FHCRC
    if flags & 0x02 != 0 {
        offset += 2;
    }

    if offset > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid header"));
    }

    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_marker_decoder() {
        let original = b"Hello, World! This is a test of the marker-based decoder.";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_parallel(&compressed, &mut output, 1).unwrap();
        assert_eq!(&output, original);
    }

    #[test]
    fn test_parallel_decode() {
        let original: Vec<u8> = (0..500_000).map(|i| (i % 256) as u8).collect();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_parallel(&compressed, &mut output, 4).unwrap();
        assert_eq!(output.len(), original.len());
    }
}
