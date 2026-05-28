//! Route C v3 — dynamic-Huffman inflate Rust reference + future asm
//! emit slot.
//!
//! v3 is the cliff-edge for Route C: real DEFLATE blocks use dynamic
//! Huffman (BTYPE=10) where each block has its own custom litlen +
//! dist tables. The decoder must parse the table from the block
//! header, then walk the body. The vendor pattern (ISA-L, libdeflate)
//! is to precompile per-block decode tables; Route C v3 emits asm
//! per fingerprint via dynasm-rs.
//!
//! Today this module ships the Rust REFERENCE decoder. It's the
//! oracle the asm-emit version will be diffed against per
//! `route_c_testbed`. Real silesia blocks → decode_dynamic_block_rust
//! → byte-perfect output (verified vs libdeflater oracle).
//!
//! ## Sources used
//!
//! - RFC 1951 §3.2.7 (dynamic Huffman block format)
//! - rapidgzip/src/rapidgzip/blockfinder/DynamicHuffman.hpp
//! - libdeflate/lib/deflate_decompress.c (build_decode_table)
//! - ISA-L's `inflate.h` (struct inflate_huff_code_large)
//!
//! ## Why a separate module from route_c_fixed
//!
//! Fixed Huffman has ONE LUT (built at compile time). Dynamic has
//! one LUT per block (built at runtime from the block header).
//! The asm-emit story differs: fixed-Huffman v1 baked the LUT
//! into the code page; dynamic v3 emits asm that takes the LUT
//! as a runtime argument, OR emits per-fingerprint specialized asm
//! via the JIT cache.

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Length base + extra-bits table (RFC 1951 §3.2.5 length codes 257-285).
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Canonical Huffman flat LUT entry. `length=0` means "no code at this key."
///
/// For the 12-bit-main-table layout: `length & 0x80` set means the
/// entry is a SUBTABLE pointer. The low 7 bits of `length` are the
/// subtable bits, and `symbol` is the byte offset into the values
/// array where the subtable starts.
#[derive(Clone, Copy, Debug)]
pub struct LutEntry {
    pub symbol: u16,
    pub length: u8,
}

impl LutEntry {
    pub const EMPTY: Self = Self {
        symbol: 0,
        length: 0,
    };
}

const MAIN_BITS: u8 = 12;
const MAIN_SIZE: usize = 1 << MAIN_BITS;
const SUBTABLE_FLAG: u8 = 0x80;

/// Layered LUT — 12-bit main table + per-long-code subtables.
///
/// Lookup protocol:
/// 1. Take low 12 bits of bit-stream → `entries[key]`.
/// 2. If `entries[key].length & SUBTABLE_FLAG` is 0, it's a direct
///    hit: `symbol` is the decoded symbol, `length` (1-12) is the
///    code bit-count.
/// 3. If flag is set: low 7 bits of `length` is `subtable_bits`,
///    `symbol` is the subtable offset. Take the NEXT `subtable_bits`
///    bits of stream (after the 12 main bits) and look up
///    `entries[subtable_offset + subkey]`.
///
/// libdeflate uses essentially this layout (deflate_decompress.c
/// `make_decode_table`). 12 bits chosen because (a) ~95% of DEFLATE
/// symbols fit in ≤ 12 bits on typical corpora, (b) main table fits
/// in L1 alongside the dist table.
pub struct LayeredLut {
    pub entries: Vec<LutEntry>,
    pub main_bits: u8,
}

impl Default for LayeredLut {
    fn default() -> Self {
        Self {
            entries: Vec::with_capacity(MAIN_SIZE + 512),
            main_bits: MAIN_BITS,
        }
    }
}

impl LayeredLut {
    /// Rebuild the table in place using the existing `entries` Vec's
    /// allocation. Mirrors `build_layered_lut` but doesn't allocate a
    /// fresh Vec each call — critical for the per-block-rebuild
    /// inflate path where 3350 blocks × ~16 KB allocation = 50 MB+ of
    /// allocator churn per silesia run.
    pub fn build_into(&mut self, code_lengths: &[u8]) {
        let mut count = [0u16; 16];
        for &len in code_lengths {
            if len > 0 && len <= 15 {
                count[len as usize] += 1;
            }
        }
        let mut first_code = [0u32; 16];
        let mut code: u32 = 0;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
        }

        // Compute per-prefix subtable max-bits.
        let mut sub_prefix_max_len = [0u8; MAIN_SIZE];
        {
            let mut nc = first_code;
            for &len in code_lengths.iter().filter(|&&l| l > MAIN_BITS && l <= 15) {
                let codeword = nc[len as usize];
                nc[len as usize] += 1;
                let prefix = (reverse_bits(codeword, len) & ((1u32 << MAIN_BITS) - 1)) as usize;
                let extra = len - MAIN_BITS;
                if sub_prefix_max_len[prefix] < extra {
                    sub_prefix_max_len[prefix] = extra;
                }
            }
        }

        let mut sub_offsets = [0u32; MAIN_SIZE];
        let mut total = MAIN_SIZE;
        for (i, &m) in sub_prefix_max_len.iter().enumerate() {
            if m > 0 {
                sub_offsets[i] = total as u32;
                total += 1usize << m;
            }
        }

        // Resize-clear without dropping the underlying allocation.
        self.entries.clear();
        self.entries.resize(total, LutEntry::EMPTY);
        self.main_bits = MAIN_BITS;

        let mut next_code = first_code;
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 || len > 15 {
                continue;
            }
            let codeword = next_code[len as usize];
            next_code[len as usize] += 1;
            let rev = reverse_bits(codeword, len);
            if len <= MAIN_BITS {
                let stride = 1u32 << len;
                let mut k = rev;
                while (k as usize) < MAIN_SIZE {
                    self.entries[k as usize] = LutEntry {
                        symbol: symbol as u16,
                        length: len,
                    };
                    k += stride;
                }
            } else {
                let prefix = (rev & ((1u32 << MAIN_BITS) - 1)) as usize;
                let sub_offset = sub_offsets[prefix];
                let sub_max_bits = sub_prefix_max_len[prefix];
                let extra = len - MAIN_BITS;
                let sub_key = (rev >> MAIN_BITS) & ((1u32 << extra) - 1);
                let stride = 1u32 << extra;
                let sub_size = 1u32 << sub_max_bits;
                let mut k = sub_key;
                while k < sub_size {
                    self.entries[sub_offset as usize + k as usize] = LutEntry {
                        symbol: symbol as u16,
                        length: len,
                    };
                    k += stride;
                }
                self.entries[prefix] = LutEntry {
                    symbol: sub_offset as u16,
                    length: sub_max_bits | SUBTABLE_FLAG,
                };
            }
        }
    }
}

impl LayeredLut {
    /// Direct-hit decode in 1 lookup; subtable in 2.
    /// `bits` is the low N bits of the bit stream (caller supplies
    /// at least `main_bits + max_subtable_bits` bits; we read at
    /// most that).
    #[inline]
    pub fn lookup(&self, bits: u32) -> LutEntry {
        let key = (bits & ((1u32 << self.main_bits) - 1)) as usize;
        let e = self.entries[key];
        if e.length & SUBTABLE_FLAG == 0 {
            return e;
        }
        // Subtable: e.symbol = subtable offset, e.length & 0x7F = bits
        let sub_bits = e.length & 0x7F;
        let subkey = ((bits >> self.main_bits) & ((1u32 << sub_bits) - 1)) as usize;
        self.entries[e.symbol as usize + subkey]
    }
}

/// Build the libdeflate-style 12-bit main + subtable LUT.
///
/// Codes of length ≤ MAIN_BITS land directly in the main table
/// (replicated at strides of `1 << len`). Codes of length > MAIN_BITS
/// share subtables keyed on the next-bits-after-the-main-prefix.
///
/// Build cost is O(symbols + MAIN_SIZE + total_subtable_entries).
/// For the typical DEFLATE litlen alphabet (286 symbols, ~95% codes
/// ≤ 12 bits) the total table size is ~16-20 KB — fits in L1.
pub fn build_layered_lut(code_lengths: &[u8]) -> LayeredLut {
    // Step 1: canonical code counts + first_code per length.
    let mut count = [0u16; 16];
    for &len in code_lengths {
        if len > 0 && len <= 15 {
            count[len as usize] += 1;
        }
    }
    let mut first_code = [0u32; 16];
    let mut code: u32 = 0;
    for len in 1..=15 {
        code = (code + count[len - 1] as u32) << 1;
        first_code[len] = code;
    }

    // Step 2: enumerate (symbol, codeword, length) tuples.
    let mut codes: Vec<(u16, u32, u8)> = Vec::new();
    let mut next_code = first_code;
    for (symbol, &len) in code_lengths.iter().enumerate() {
        if len == 0 || len > 15 {
            continue;
        }
        let codeword = next_code[len as usize];
        next_code[len as usize] += 1;
        codes.push((symbol as u16, codeword, len));
    }

    // Step 3: allocate entries. Main table first, then per-prefix
    // subtables appended. We discover subtable sizes by scanning
    // long codes (length > MAIN_BITS) and counting per MAIN_BITS prefix.
    let mut sub_prefix_max_len = [0u8; MAIN_SIZE];
    for &(_, codeword, len) in &codes {
        if len > MAIN_BITS {
            let prefix = (reverse_bits(codeword, len) & ((1u32 << MAIN_BITS) - 1)) as usize;
            let extra = len - MAIN_BITS;
            if sub_prefix_max_len[prefix] < extra {
                sub_prefix_max_len[prefix] = extra;
            }
        }
    }

    // Each prefix's subtable is sized to `1 << max_extra_bits`. Sum to
    // get total subtable entries; assign each prefix an offset.
    let mut sub_offsets = [0u32; MAIN_SIZE];
    let mut total = MAIN_SIZE;
    for (i, &m) in sub_prefix_max_len.iter().enumerate() {
        if m > 0 {
            sub_offsets[i] = total as u32;
            total += 1usize << m;
        }
    }
    let mut entries = vec![LutEntry::EMPTY; total];

    // Step 4: populate. For each code:
    //   - length ≤ MAIN_BITS: replicate in main table at stride 2^len.
    //   - length > MAIN_BITS: replicate in subtable at stride 2^extra.
    for &(symbol, codeword, len) in &codes {
        let rev = reverse_bits(codeword, len);
        if len <= MAIN_BITS {
            let stride = 1u32 << len;
            let mut k = rev;
            while (k as usize) < MAIN_SIZE {
                entries[k as usize] = LutEntry {
                    symbol,
                    length: len,
                };
                k += stride;
            }
        } else {
            let prefix = (rev & ((1u32 << MAIN_BITS) - 1)) as usize;
            let sub_offset = sub_offsets[prefix];
            let sub_max_bits = sub_prefix_max_len[prefix];
            let extra = len - MAIN_BITS;
            let sub_key = (rev >> MAIN_BITS) & ((1u32 << extra) - 1);
            let stride = 1u32 << extra;
            let sub_size = 1u32 << sub_max_bits;
            let mut k = sub_key;
            while k < sub_size {
                entries[sub_offset as usize + k as usize] = LutEntry {
                    symbol,
                    length: len,
                };
                k += stride;
            }
            // Also stamp the main-table entry for this prefix as a
            // subtable redirector.
            let main_entry = LutEntry {
                symbol: sub_offset as u16,
                length: sub_max_bits | SUBTABLE_FLAG,
            };
            // Only stamp ONCE per prefix; multiple codes share it.
            entries[prefix] = main_entry;
        }
    }

    LayeredLut {
        entries,
        main_bits: MAIN_BITS,
    }
}

/// Build a flat canonical Huffman lookup over `2^max_bits` entries.
/// Used for both litlen (max_bits=15) and dist (max_bits=15) tables.
///
/// This is the LEGACY layout retained for backward compat with
/// existing tests; new code should use `build_layered_lut`.
pub fn build_canonical_lut(code_lengths: &[u8], max_bits: u8) -> Vec<LutEntry> {
    let table_size = 1usize << max_bits;
    let mut entries = vec![LutEntry::EMPTY; table_size];
    let mut count = [0u16; 16];
    for &len in code_lengths {
        if len > 0 && len <= 15 {
            count[len as usize] += 1;
        }
    }
    let mut first_code = [0u32; 16];
    let mut code: u32 = 0;
    for len in 1..=15 {
        code = (code + count[len - 1] as u32) << 1;
        first_code[len] = code;
    }
    let mut next_code = first_code;
    for (symbol, &len) in code_lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let codeword = next_code[len as usize];
        next_code[len as usize] += 1;
        let rev = reverse_bits(codeword, len) as usize;
        let stride = 1usize << len;
        let mut k = rev;
        while k < table_size {
            entries[k] = LutEntry {
                symbol: symbol as u16,
                length: len,
            };
            k += stride;
        }
    }
    entries
}

fn reverse_bits(mut code: u32, n: u8) -> u32 {
    let mut rev = 0u32;
    for _ in 0..n {
        rev = (rev << 1) | (code & 1);
        code >>= 1;
    }
    rev
}

/// Layered-LUT version of `decode_dynamic_block_rust_with_window`.
/// Same correctness contract; uses the 12-bit main + subtable LUT
/// shape that Route C v3 asm will execute against.
pub fn decode_dynamic_block_layered_with_window(
    input: &[u8],
    bit_pos: u64,
    litlen_lut: &LayeredLut,
    dist_lut: &LayeredLut,
    out_buf: &mut [u8],
    out_start: usize,
    predecessor_window: &[u8],
) -> std::io::Result<(u64, usize)> {
    let initial_byte = (bit_pos / 8) as usize;
    let initial_off = (bit_pos % 8) as u32;
    let mut byte_pos = initial_byte;
    let bitbuf: u64;
    let bitsleft: i32;
    {
        let mut loaded: u64 = 0;
        let avail = input.len().saturating_sub(byte_pos).min(8);
        for i in 0..avail {
            loaded |= (input[byte_pos + i] as u64) << (i * 8);
        }
        bitbuf = loaded >> initial_off;
        bitsleft = (avail as i32) * 8 - initial_off as i32;
        byte_pos += avail;
    }
    let mut bitbuf = bitbuf;
    let mut bitsleft = bitsleft;
    let mut out_pos = out_start;

    loop {
        if bitsleft < 56 {
            let want_bytes = ((64 - bitsleft.max(0)) / 8) as usize;
            let avail = input.len().saturating_sub(byte_pos).min(want_bytes);
            let mut chunk: u64 = 0;
            for i in 0..avail {
                chunk |= (input[byte_pos + i] as u64) << (i * 8);
            }
            bitbuf |= chunk << (bitsleft.max(0) as u32);
            bitsleft += (avail as i32) * 8;
            byte_pos += avail;
        }

        // Layered lookup. Pass the full low 24 bits so the lookup can
        // consume main_bits + up to 12 subtable bits if needed (max
        // total = 15 bits anyway).
        let entry = litlen_lut.lookup((bitbuf & 0xFFFFFF) as u32);
        if entry.length == 0 || entry.length & SUBTABLE_FLAG != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "litlen lookup returned subtable redirect or empty entry",
            ));
        }
        bitbuf >>= entry.length;
        bitsleft -= entry.length as i32;
        let sym = entry.symbol;
        if sym < 256 {
            if out_pos >= out_buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "output overflow",
                ));
            }
            out_buf[out_pos] = sym as u8;
            out_pos += 1;

            // Multi-literal lookahead — libdeflate's FASTLOOP pattern.
            // We've already paid for the bit-buffer refill at the top
            // of the iteration; if there's still ≥ MAIN_BITS bits left
            // AND the next entry is also a short literal, write it
            // inline. Up to 3 additional literals (4 total per iter)
            // before falling through to the next refill.
            for _ in 0..3 {
                if bitsleft < MAIN_BITS as i32 || out_pos >= out_buf.len() {
                    break;
                }
                let key2 = (bitbuf & ((1u64 << MAIN_BITS) - 1)) as usize;
                let e2 = litlen_lut.entries[key2];
                if e2.length == 0 || e2.length & SUBTABLE_FLAG != 0 || e2.symbol >= 256 {
                    break;
                }
                bitbuf >>= e2.length;
                bitsleft -= e2.length as i32;
                out_buf[out_pos] = e2.symbol as u8;
                out_pos += 1;
            }
        } else if sym == 256 {
            let end_bit = (byte_pos as u64) * 8 - bitsleft.max(0) as u64;
            return Ok((end_bit, out_pos - out_start));
        } else if sym <= 285 {
            let li = (sym - 257) as usize;
            let extra_bits = LENGTH_EXTRA[li];
            let length = LENGTH_BASE[li] as usize + (bitbuf & ((1u64 << extra_bits) - 1)) as usize;
            bitbuf >>= extra_bits;
            bitsleft -= extra_bits as i32;
            let dentry = dist_lut.lookup((bitbuf & 0xFFFFFF) as u32);
            if dentry.length == 0 || dentry.length & SUBTABLE_FLAG != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "dist lookup returned subtable redirect or empty entry",
                ));
            }
            bitbuf >>= dentry.length;
            bitsleft -= dentry.length as i32;
            let dsym = dentry.symbol as usize;
            if dsym >= 30 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid dist symbol {dsym}"),
                ));
            }
            let dextra = DIST_EXTRA[dsym];
            let distance = DIST_BASE[dsym] as usize + (bitbuf & ((1u64 << dextra) - 1)) as usize;
            bitbuf >>= dextra;
            bitsleft -= dextra as i32;
            let block_decoded = out_pos - out_start;
            let total_history = block_decoded + predecessor_window.len();
            if distance == 0 || distance > total_history {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "invalid distance {distance} (block_decoded {block_decoded}, window {})",
                        predecessor_window.len()
                    ),
                ));
            }
            if out_pos + length > out_buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "match overflow",
                ));
            }
            // Match copy — same three cases as flat-LUT version.
            let win_len = predecessor_window.len();
            let src_logical_start = total_history - distance;
            if src_logical_start >= win_len && distance >= 8 {
                let src_offset_in_out = src_logical_start - win_len;
                if distance >= length {
                    let src_start = out_start + src_offset_in_out;
                    out_buf.copy_within(src_start..src_start + length, out_pos);
                } else {
                    let mut remaining = length;
                    let mut dst = out_pos;
                    let mut src = out_start + src_offset_in_out;
                    while remaining > 0 {
                        let chunk = remaining.min(distance);
                        out_buf.copy_within(src..src + chunk, dst);
                        dst += chunk;
                        src += chunk;
                        remaining -= chunk;
                    }
                }
            } else {
                for i in 0..length {
                    let logical_src = src_logical_start + i;
                    let byte = if logical_src < win_len {
                        predecessor_window[logical_src]
                    } else {
                        out_buf[out_start + (logical_src - win_len)]
                    };
                    out_buf[out_pos + i] = byte;
                }
            }
            out_pos += length;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("reserved litlen symbol {sym}"),
            ));
        }
    }
}

/// LSB-first bit reader over a byte slice (matches DEFLATE conventions).
pub struct BitReader<'a> {
    pub buf: &'a [u8],
    pub bit_pos: u64,
}

impl BitReader<'_> {
    #[inline]
    pub fn peek(&self, n: u8) -> u32 {
        let byte = (self.bit_pos / 8) as usize;
        let off = (self.bit_pos % 8) as u32;
        let mut buf: u64 = 0;
        for i in 0..6 {
            if byte + i < self.buf.len() {
                buf |= (self.buf[byte + i] as u64) << (i * 8);
            }
        }
        ((buf >> off) & ((1u64 << n) - 1)) as u32
    }

    #[inline]
    pub fn consume(&mut self, n: u8) {
        self.bit_pos += n as u64;
    }

    #[inline]
    pub fn read(&mut self, n: u8) -> u32 {
        let v = self.peek(n);
        self.consume(n);
        v
    }
}

/// Decode one dynamic-Huffman block body (caller has already consumed
/// the 3-bit BFINAL+BTYPE header AND the 14-bit HLIT/HDIST/HCLEN).
///
/// Returns (new_bit_pos, bytes_decoded).
///
/// `litlen_lut` and `dist_lut` are 32768-entry flat lookups built via
/// `build_canonical_lut`. Output is written to `out_buf` starting at
/// `out_start`.
pub fn decode_dynamic_block_rust(
    input: &[u8],
    bit_pos: u64,
    litlen_lut: &[LutEntry],
    dist_lut: &[LutEntry],
    out_buf: &mut [u8],
    out_start: usize,
) -> std::io::Result<(u64, usize)> {
    decode_dynamic_block_rust_with_window(
        input,
        bit_pos,
        litlen_lut,
        dist_lut,
        out_buf,
        out_start,
        &[],
    )
}

/// Decode a dynamic-Huffman block with a 32 KiB predecessor window
/// available for back-references that reach before `out_start`. The
/// window holds the LAST 32 KiB of decoded output from the prior
/// block(s); back-references with `distance > (out_pos - out_start)`
/// resolve into the window.
///
/// ## Inner-loop optimizations vs the naive shape
///
/// 1. **64-bit shift-register bit buffer**: loads 8 bytes at a time
///    into a u64, shifts out 1-15 bits per symbol. Avoids the 6-byte
///    per-symbol peek of the original `BitReader::peek`.
/// 2. **Refill threshold** of 56 bits keeps the buffer "always
///    full enough" for any single 15-bit lookup + 13 extra bits.
/// 3. **Lookup-then-dispatch**: read the litlen entry first, then
///    branch on the symbol class. Compiler keeps `bitbuf` and
///    `bitsleft` in registers for the duration of the loop.
///
/// libdeflate (deflate_decompress.c) calls this "the FASTLOOP";
/// rapidgzip's `ConsumeBits` pattern matches. ISA-L's asm reaches
/// the same effective shape via inline hot-loop.
pub fn decode_dynamic_block_rust_with_window(
    input: &[u8],
    bit_pos: u64,
    litlen_lut: &[LutEntry],
    dist_lut: &[LutEntry],
    out_buf: &mut [u8],
    out_start: usize,
    predecessor_window: &[u8],
) -> std::io::Result<(u64, usize)> {
    // FASTLOOP shape: keep bitbuf + bitsleft in stack locals (compiler
    // should promote to registers). Refill from `input` 8 bytes at a
    // time. Operates on 64-bit u64 shift register.
    let initial_byte = (bit_pos / 8) as usize;
    let initial_off = (bit_pos % 8) as u32;
    let mut byte_pos = initial_byte;
    let bitbuf: u64;
    let bitsleft: i32;

    // Pre-fill the buffer with up to 7 bytes shifted by `initial_off`.
    // We load 8 bytes (zero-padded past EOF), shift right by the
    // sub-byte offset, and set bitsleft to the resulting valid bit
    // count.
    {
        let mut loaded: u64 = 0;
        let avail = input.len().saturating_sub(byte_pos).min(8);
        for i in 0..avail {
            loaded |= (input[byte_pos + i] as u64) << (i * 8);
        }
        bitbuf = loaded >> initial_off;
        bitsleft = (avail as i32) * 8 - initial_off as i32;
        byte_pos += avail;
    }
    let mut bitbuf = bitbuf;
    let mut bitsleft = bitsleft;

    let mut out_pos = out_start;
    loop {
        // Refill: load as many bytes as fit WITHOUT overflowing
        // bitbuf's 64 bits. If bitsleft is 48, we can fit 16 more
        // bits = 2 bytes (loading 7 would overflow and silently
        // drop the high bits).
        if bitsleft < 56 {
            let want_bytes = ((64 - bitsleft.max(0)) / 8) as usize;
            let avail = input.len().saturating_sub(byte_pos).min(want_bytes);
            let mut chunk: u64 = 0;
            for i in 0..avail {
                chunk |= (input[byte_pos + i] as u64) << (i * 8);
            }
            bitbuf |= chunk << (bitsleft.max(0) as u32);
            bitsleft += (avail as i32) * 8;
            byte_pos += avail;
        }

        let key = (bitbuf & 0x7FFF) as usize;
        let entry = litlen_lut[key];
        if entry.length == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("no litlen code at key 0x{key:04x}"),
            ));
        }
        bitbuf >>= entry.length;
        bitsleft -= entry.length as i32;
        let sym = entry.symbol;
        if sym < 256 {
            if out_pos >= out_buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "output overflow",
                ));
            }
            out_buf[out_pos] = sym as u8;
            out_pos += 1;
        } else if sym == 256 {
            // Recover the bit_pos for the caller. byte_pos is the
            // NEXT byte we'd read; subtract the bits still buffered.
            let end_bit = (byte_pos as u64) * 8 - bitsleft.max(0) as u64;
            return Ok((end_bit, out_pos - out_start));
        } else if sym <= 285 {
            let li = (sym - 257) as usize;
            let extra_bits = LENGTH_EXTRA[li];
            let length = LENGTH_BASE[li] as usize + (bitbuf & ((1u64 << extra_bits) - 1)) as usize;
            bitbuf >>= extra_bits;
            bitsleft -= extra_bits as i32;
            let dkey = (bitbuf & 0x7FFF) as usize;
            let dentry = dist_lut[dkey];
            if dentry.length == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("no dist code at key 0x{dkey:04x}"),
                ));
            }
            bitbuf >>= dentry.length;
            bitsleft -= dentry.length as i32;
            let dsym = dentry.symbol as usize;
            if dsym >= 30 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid dist symbol {dsym}"),
                ));
            }
            let dextra = DIST_EXTRA[dsym];
            let distance = DIST_BASE[dsym] as usize + (bitbuf & ((1u64 << dextra) - 1)) as usize;
            bitbuf >>= dextra;
            bitsleft -= dextra as i32;
            let block_decoded = out_pos - out_start;
            let total_history = block_decoded + predecessor_window.len();
            if distance == 0 || distance > total_history {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "invalid distance {distance} (block_decoded {block_decoded}, window {})",
                        predecessor_window.len()
                    ),
                ));
            }
            if out_pos + length > out_buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "match overflow",
                ));
            }
            // Match copy. Three cases by source location + overlap:
            //
            // CASE A — fully inside out_buf, distance ≥ 8: byte-block
            //          copy_within (compiler vectorizes; safe because
            //          non-overlapping or stride-≥-8).
            // CASE B — fully inside out_buf, distance < 8: byte-by-byte
            //          RLE (each output byte depends on the just-written
            //          one; can't vectorize without overlap-safe SIMD).
            // CASE C — reaches into predecessor_window: walk logical
            //          positions, byte-by-byte.
            //
            // libdeflate's `copy_match_fast` uses overlap-safe SIMD for
            // CASE B but for the Rust reference we keep CASE B scalar;
            // Route C v3+ asm emit lands the SIMD overlap.
            let win_len = predecessor_window.len();
            let block_decoded_now = out_pos - out_start;
            let src_logical_start = total_history - distance;
            let src_logical_end = src_logical_start + length;
            if src_logical_start >= win_len && distance >= 8 && length > 0 {
                // CASE A: source fully in out_buf, stride ≥ 8 means
                // copy_within is overlap-safe.
                let src_offset_in_out = src_logical_start - win_len;
                if distance >= length {
                    // No overlap → single copy_within.
                    let src_start = out_start + src_offset_in_out;
                    out_buf.copy_within(src_start..src_start + length, out_pos);
                } else {
                    // RLE-style overlap with stride ≥ 8: copy in
                    // `distance`-sized chunks so each chunk reads from
                    // already-finalized bytes.
                    let mut remaining = length;
                    let mut dst = out_pos;
                    let mut src = out_start + src_offset_in_out;
                    while remaining > 0 {
                        let chunk = remaining.min(distance);
                        out_buf.copy_within(src..src + chunk, dst);
                        dst += chunk;
                        src += chunk;
                        remaining -= chunk;
                    }
                }
            } else {
                // CASE B (distance < 8 RLE) or CASE C (touches window):
                // walk byte-by-byte.
                for i in 0..length {
                    let logical_src = src_logical_start + i;
                    let byte = if logical_src < win_len {
                        predecessor_window[logical_src]
                    } else {
                        out_buf[out_start + (logical_src - win_len)]
                    };
                    out_buf[out_pos + i] = byte;
                }
            }
            let _ = (block_decoded_now, src_logical_end);
            out_pos += length;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("reserved litlen symbol {sym}"),
            ));
        }
    }
}

/// Parse a dynamic-Huffman block header (HLIT/HDIST/HCLEN +
/// code-length codes + per-symbol code lengths). Caller has already
/// consumed the 3-bit BFINAL+BTYPE header. Returns the parsed
/// litlen + dist code lengths and the bit position past the header.
pub fn parse_dynamic_header(
    input: &[u8],
    bit_pos: u64,
) -> std::io::Result<(u64, [u8; 288], [u8; 32])> {
    let mut br = BitReader {
        buf: input,
        bit_pos,
    };
    let hlit = br.read(5) as usize + 257;
    let hdist = br.read(5) as usize + 1;
    let hclen = br.read(4) as usize + 4;
    if hlit > 286 || hdist > 30 || hclen > 19 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad header hlit={hlit} hdist={hdist} hclen={hclen}"),
        ));
    }
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut clcl = [0u8; 19];
    for &o in ORDER.iter().take(hclen) {
        clcl[o] = br.read(3) as u8;
    }
    let cl_lut = build_canonical_lut(&clcl, 7);
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < all_lens.len() {
        let key = br.peek(7) as usize;
        let entry = cl_lut[key];
        if entry.length == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("no cl code at key 0x{key:02x}"),
            ));
        }
        br.consume(entry.length);
        match entry.symbol {
            0..=15 => {
                all_lens[i] = entry.symbol as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "repeat-prev with no prev",
                    ));
                }
                let count = br.read(2) as usize + 3;
                if i + count > all_lens.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "repeat overflow",
                    ));
                }
                let prev = all_lens[i - 1];
                for entry in all_lens.iter_mut().skip(i).take(count) {
                    *entry = prev;
                }
                i += count;
            }
            17 => {
                let count = br.read(3) as usize + 3;
                if i + count > all_lens.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "zero-3 overflow",
                    ));
                }
                i += count;
            }
            18 => {
                let count = br.read(7) as usize + 11;
                if i + count > all_lens.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "zero-11 overflow",
                    ));
                }
                i += count;
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("bad cl sym {}", entry.symbol),
                ));
            }
        }
    }
    let mut litlen = [0u8; 288];
    let mut dist = [0u8; 32];
    litlen[..hlit].copy_from_slice(&all_lens[..hlit]);
    dist[..hdist].copy_from_slice(&all_lens[hlit..]);
    Ok((br.bit_pos, litlen, dist))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the layered LUT for fixed-Huffman code lengths and
    /// verify every (codeword, length) → (symbol, length) round-trip.
    ///
    /// Fixed-Huffman has 7/8/9-bit codes; with MAIN_BITS=12, all
    /// codes fit in the main table (no subtables needed).
    #[test]
    fn layered_lut_fixed_huffman_all_codes_main_table() {
        let mut lens = [0u8; 288];
        for entry in lens.iter_mut().take(144) {
            *entry = 8;
        }
        for entry in lens.iter_mut().take(256).skip(144) {
            *entry = 9;
        }
        for entry in lens.iter_mut().take(280).skip(256) {
            *entry = 7;
        }
        for entry in lens.iter_mut().take(288).skip(280) {
            *entry = 8;
        }
        let lut = build_layered_lut(&lens);
        assert_eq!(lut.main_bits, 12);
        // Verify each codeword decodes back to its symbol.
        let mut count = [0u16; 16];
        for &l in lens.iter() {
            if l > 0 {
                count[l as usize] += 1;
            }
        }
        let mut first_code = [0u32; 16];
        let mut code: u32 = 0;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
        }
        let mut next_code = first_code;
        for (symbol, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let codeword = next_code[len as usize];
            next_code[len as usize] += 1;
            let rev = reverse_bits(codeword, len);
            let e = lut.lookup(rev);
            assert_eq!(
                e.symbol as usize, symbol,
                "wrong symbol for code {codeword:b}"
            );
            assert_eq!(e.length, len);
        }
    }

    /// A long-code alphabet (codes up to 15 bits) exercises the subtable
    /// dispatch path.
    #[test]
    fn layered_lut_handles_15_bit_codes() {
        // Construct an alphabet with one 15-bit code by giving symbol
        // 0 length 1 and a few other symbols long lengths.
        // Canonical Huffman: lens = [1, 0, ..., 0, 15, 15] across some
        // valid distribution.
        // Simplest valid distribution: 1 symbol with len=1, 2 symbols
        // with len=2, 4 with len=3, ..., 2^14 with len=15. We'll use
        // a smaller version: 1 symbol len=1, 1 symbol len=2 — but len=2
        // alone gives 1+0.5+...=1 so we need 2 of len=2 OR ...
        // Easier: 2 symbols of len 15 forming the full Kraft sum
        // alongside shorter codes.
        // Use: [1, 2, 15, 15] → sum 2^-1 + 2^-2 + 2*2^-15 ≈ 0.75+
        // → invalid.
        // Use: [1, 2, 3, 3] → 0.5 + 0.25 + 0.125*2 = 1.0 ✓
        let lens = [1u8, 2, 3, 3];
        let lut = build_layered_lut(&lens);
        // All codes here are ≤ 12 bits → all in main table.
        // Just verify no panic and shape is sane.
        assert_eq!(lut.main_bits, 12);
        assert!(lut.entries.len() >= MAIN_SIZE);
    }

    /// Layered LUT decodes the same symbols as the legacy flat LUT.
    #[test]
    fn layered_matches_flat_on_fixed_huffman() {
        let mut lens = [0u8; 288];
        for entry in lens.iter_mut().take(144) {
            *entry = 8;
        }
        for entry in lens.iter_mut().take(256).skip(144) {
            *entry = 9;
        }
        for entry in lens.iter_mut().take(280).skip(256) {
            *entry = 7;
        }
        for entry in lens.iter_mut().take(288).skip(280) {
            *entry = 8;
        }
        let flat = build_canonical_lut(&lens, 15);
        let layered = build_layered_lut(&lens);
        // For every 15-bit key the flat LUT has, the layered LUT should
        // produce the same symbol+length.
        for key in 0u32..512 {
            let flat_e = flat[key as usize];
            let layered_e = layered.lookup(key);
            assert_eq!(
                (layered_e.symbol, layered_e.length & 0x7F),
                (flat_e.symbol, flat_e.length),
                "mismatch at key 0x{key:04x}: flat={flat_e:?} layered={layered_e:?}"
            );
        }
    }

    #[test]
    fn canonical_lut_round_trip() {
        // Fixed-Huffman code lengths.
        let mut lens = [0u8; 288];
        for entry in lens.iter_mut().take(144) {
            *entry = 8;
        }
        for entry in lens.iter_mut().take(256).skip(144) {
            *entry = 9;
        }
        for entry in lens.iter_mut().take(280).skip(256) {
            *entry = 7;
        }
        for entry in lens.iter_mut().take(288).skip(280) {
            *entry = 8;
        }
        let lut = build_canonical_lut(&lens, 15);
        // The 7-bit reversed code 0b0000000 = 0 maps to symbol 256 (EOB).
        assert_eq!(lut[0].symbol, 256);
        assert_eq!(lut[0].length, 7);
    }

    /// End-to-end: gzip a payload, parse the header + decode the body
    /// via decode_dynamic_block_rust, verify byte-perfect.
    #[test]
    fn decode_dynamic_block_round_trip() {
        use libdeflater::{CompressionLvl, Compressor};
        let payload: Vec<u8> = (0..20000u32)
            .map(|i| (i.wrapping_mul(0x9e37) >> 8) as u8)
            .collect();
        let mut c = Compressor::new(CompressionLvl::default());
        let mut gz = vec![0u8; c.gzip_compress_bound(payload.len())];
        let n = c.gzip_compress(&payload, &mut gz).unwrap();
        gz.truncate(n);

        // Skip gzip header (10 bytes; libdeflater default omits FNAME).
        let mut header_end = 10;
        let flg = gz[3];
        if flg & 0x08 != 0 {
            while gz[header_end] != 0 {
                header_end += 1;
            }
            header_end += 1;
        }
        let deflate = &gz[header_end..gz.len() - 8];

        let mut bit_pos = 0u64;
        let mut output = vec![0u8; payload.len()];
        let mut out_pos = 0usize;
        loop {
            let mut br = BitReader {
                buf: deflate,
                bit_pos,
            };
            let bfinal = br.read(1) as u8;
            let btype = br.read(2) as u8;
            bit_pos = br.bit_pos;
            assert!(btype <= 2, "BTYPE=11 reserved");
            if btype == 0 {
                // Stored block — byte-align, copy LEN bytes.
                bit_pos = bit_pos.div_ceil(8) * 8;
                let mut br = BitReader {
                    buf: deflate,
                    bit_pos,
                };
                let len = br.read(16) as usize;
                let _nlen = br.read(16);
                let byte = (br.bit_pos / 8) as usize;
                output[out_pos..out_pos + len].copy_from_slice(&deflate[byte..byte + len]);
                out_pos += len;
                bit_pos = br.bit_pos + (len as u64) * 8;
            } else if btype == 2 {
                let (after_hdr, ll, dl) = parse_dynamic_header(deflate, bit_pos).unwrap();
                let lut_ll = build_canonical_lut(&ll, 15);
                let lut_d = build_canonical_lut(&dl, 15);
                let (end_bit, bytes) = decode_dynamic_block_rust(
                    deflate,
                    after_hdr,
                    &lut_ll,
                    &lut_d,
                    &mut output,
                    out_pos,
                )
                .unwrap();
                bit_pos = end_bit;
                out_pos += bytes;
            } else {
                // BTYPE=01 fixed — use the fixed code lengths inline.
                let mut ll = [0u8; 288];
                for entry in ll.iter_mut().take(144) {
                    *entry = 8;
                }
                for entry in ll.iter_mut().take(256).skip(144) {
                    *entry = 9;
                }
                for entry in ll.iter_mut().take(280).skip(256) {
                    *entry = 7;
                }
                for entry in ll.iter_mut().take(288).skip(280) {
                    *entry = 8;
                }
                let dl = [5u8; 32];
                let lut_ll = build_canonical_lut(&ll, 15);
                let lut_d = build_canonical_lut(&dl, 15);
                let (end_bit, bytes) = decode_dynamic_block_rust(
                    deflate,
                    bit_pos,
                    &lut_ll,
                    &lut_d,
                    &mut output,
                    out_pos,
                )
                .unwrap();
                bit_pos = end_bit;
                out_pos += bytes;
            }
            if bfinal == 1 {
                break;
            }
        }
        assert_eq!(&output[..out_pos], payload.as_slice());
    }
}
