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

/// Build a flat canonical Huffman lookup over `2^max_bits` entries.
/// Used for both litlen (max_bits=15) and dist (max_bits=15) tables.
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
