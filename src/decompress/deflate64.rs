/// Deflate64 (Enhanced Deflate, ZIP method 9) decompressor.
///
/// Differences from standard DEFLATE (RFC 1951):
/// - 64 KB (65536-byte) sliding window instead of 32 KB
/// - Length code 285: 16 extra bits, base = 3  (max length 65538)
/// - Distance codes 30-31: 14 extra bits each  (max distance 65536)
use crate::error::{GzippyError, GzippyResult};
use std::io::Write;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WINDOW: usize = 65536;

// Length base values for codes 257..=285. Code 257 → index 0.
// Code 285 gets base=3 with 16 extra bits (Deflate64 extension).
#[rustfmt::skip]
const LENGTH_BASE: [u32; 29] = [
     3,  4,  5,  6,  7,  8,  9, 10,
    11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115,
    131, 163, 195, 227,
    3, // code 285 — Deflate64: base 3 + 16 extra bits
];

// Extra bits for length codes 257..=285.
#[rustfmt::skip]
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 5,
    16, // code 285 — Deflate64 extension
];

// Distance base values for codes 0..=31.
// Codes 0-29 identical to DEFLATE; codes 30-31 are Deflate64 additions.
#[rustfmt::skip]
const DIST_BASE: [u32; 32] = [
       1,    2,    3,    4,    5,    7,    9,   13,
      17,   25,   33,   49,   65,   97,  129,  193,
     257,  385,  513,  769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193,12289,16385,24577,
    32769, 49153, // Deflate64: codes 30-31
];

// Extra bits for distance codes 0..=31.
#[rustfmt::skip]
const DIST_EXTRA: [u8; 32] = [
     0, 0, 0, 0, 1, 1, 2, 2,
     3, 3, 4, 4, 5, 5, 6, 6,
     7, 7, 8, 8, 9, 9,10,10,
    11,11,12,12,13,13,
    14, 14, // Deflate64: codes 30-31
];

// Code-length alphabet ordering (same as DEFLATE).
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

// Fixed Huffman bit-lengths per the DEFLATE spec (RFC 1951 §3.2.6).
fn fixed_lit_lengths() -> [u8; 288] {
    let mut lens = [0u8; 288];
    lens[..=143].fill(8);
    lens[144..=255].fill(9);
    lens[256..=279].fill(7);
    lens[280..=287].fill(8);
    lens
}

fn fixed_dist_lengths() -> [u8; 32] {
    [5u8; 32]
}

// ---------------------------------------------------------------------------
// Huffman decoder — canonical codes, max 15-bit code lengths.
// ---------------------------------------------------------------------------

struct HuffTable {
    counts: [u16; 16], // how many codes have each bit-length
    symbols: Vec<u16>, // symbols sorted by (length, code value)
}

impl HuffTable {
    fn build(lengths: &[u8]) -> GzippyResult<Self> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err(GzippyError::decompression("deflate64: code length > 15"));
            }
            counts[l as usize] += 1;
        }
        // Symbols sorted: first by length, then by symbol value.
        let mut symbols: Vec<u16> = (0..lengths.len() as u16)
            .filter(|&s| lengths[s as usize] != 0)
            .collect();
        symbols.sort_by_key(|&s| (lengths[s as usize], s));
        Ok(Self { counts, symbols })
    }

    /// Decode one symbol from the bit-reader (DEFLATE LSB-first order).
    ///
    /// This is the puff.c canonical decode algorithm.  `code`, `first`, and
    /// `index` must be signed so that `code - count` can go negative when
    /// code < count — the same arithmetic puff.c relies on with C `int`.
    #[inline(always)]
    fn decode(&self, br: &mut BitReader) -> GzippyResult<u16> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;

        for bits in 1u8..=15 {
            code |= br.read_bit()? as i32;
            let count = self.counts[bits as usize] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(GzippyError::decompression("deflate64: bad Huffman code"))
    }
}

// ---------------------------------------------------------------------------
// Bit reader — LSB-first, as required by DEFLATE.
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    src: &'a [u8],
    pos: usize,
    buf: u64,
    avail: u8,
}

impl<'a> BitReader<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            buf: 0,
            avail: 0,
        }
    }

    #[inline(always)]
    fn read_bit(&mut self) -> GzippyResult<u8> {
        if self.avail == 0 {
            self.refill()?;
        }
        let b = (self.buf & 1) as u8;
        self.buf >>= 1;
        self.avail -= 1;
        Ok(b)
    }

    #[inline(always)]
    fn read_bits(&mut self, n: u8) -> GzippyResult<u32> {
        while self.avail < n {
            self.refill()?;
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.avail -= n;
        Ok(v)
    }

    /// Align to the next byte boundary and read `n` bytes as a little-endian u16 pair.
    fn read_u16_pair(&mut self) -> GzippyResult<(u16, u16)> {
        // Discard any partial-byte bits.
        let partial = self.avail % 8;
        if partial != 0 {
            self.buf >>= partial;
            self.avail -= partial;
        }
        // Drain buffered bytes back out (they were read into buf LSB-first).
        // Easiest: just read directly from src, adjusting pos for how many
        // full bytes are still in buf.
        let buffered_bytes = (self.avail / 8) as usize;
        // Rewind pos by the bytes still buffered so we re-read from the right place.
        self.pos -= buffered_bytes;
        self.buf = 0;
        self.avail = 0;

        if self.pos + 4 > self.src.len() {
            return Err(GzippyError::decompression(
                "deflate64: truncated stored block header",
            ));
        }
        let len = u16::from_le_bytes([self.src[self.pos], self.src[self.pos + 1]]);
        let nlen = u16::from_le_bytes([self.src[self.pos + 2], self.src[self.pos + 3]]);
        self.pos += 4;
        Ok((len, nlen))
    }

    fn read_bytes(&mut self, n: usize) -> GzippyResult<&'a [u8]> {
        if self.pos + n > self.src.len() {
            return Err(GzippyError::decompression(
                "deflate64: truncated stored block",
            ));
        }
        let slice = &self.src[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    #[inline]
    fn refill(&mut self) -> GzippyResult<()> {
        if self.pos >= self.src.len() {
            return Err(GzippyError::decompression(
                "deflate64: unexpected end of input",
            ));
        }
        self.buf |= (self.src[self.pos] as u64) << self.avail;
        self.pos += 1;
        self.avail += 8;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Main decoder
// ---------------------------------------------------------------------------

/// Decompress a raw Deflate64 stream into `out`.
/// Returns the number of decompressed bytes written.
#[allow(dead_code)] // called from lib.rs; unused in the binary
pub fn decompress_deflate64_to_writer<W: Write>(data: &[u8], out: &mut W) -> GzippyResult<u64> {
    let mut br = BitReader::new(data);
    // Ring buffer acting as sliding window.
    let mut window = vec![0u8; WINDOW];
    let mut wpos: usize = 0;
    let mut total: u64 = 0;
    // Scratch buffer for flushing window spans without allocating per back-ref.
    let mut flush_buf = Vec::with_capacity(65538);

    loop {
        let bfinal = br.read_bit()?;
        let btype = br.read_bits(2)?;

        match btype {
            0b00 => {
                // Stored block
                let (len, nlen) = br.read_u16_pair()?;
                if len != !nlen {
                    return Err(GzippyError::decompression(
                        "deflate64: stored block LEN/NLEN mismatch",
                    ));
                }
                let bytes = br.read_bytes(len as usize)?;
                out.write_all(bytes)?;
                // Copy into window.
                for &b in bytes {
                    window[wpos] = b;
                    wpos = (wpos + 1) % WINDOW;
                }
                total += len as u64;
            }
            0b01 => {
                // Fixed Huffman
                let lit_table = HuffTable::build(&fixed_lit_lengths())?;
                let dist_table = HuffTable::build(&fixed_dist_lengths())?;
                total += decode_block(
                    &lit_table,
                    &dist_table,
                    &mut br,
                    &mut window,
                    &mut wpos,
                    out,
                    &mut flush_buf,
                )?;
            }
            0b10 => {
                // Dynamic Huffman
                let (lit_table, dist_table) = read_dynamic_tables(&mut br)?;
                total += decode_block(
                    &lit_table,
                    &dist_table,
                    &mut br,
                    &mut window,
                    &mut wpos,
                    out,
                    &mut flush_buf,
                )?;
            }
            _ => {
                return Err(GzippyError::decompression("deflate64: reserved block type"));
            }
        }

        if bfinal == 1 {
            break;
        }
    }

    out.flush()?;
    Ok(total)
}

/// Decompress a raw Deflate64 stream, returning an owned `Vec<u8>`.
#[allow(dead_code)] // called from lib.rs; unused in the binary
pub fn decompress_deflate64(data: &[u8]) -> GzippyResult<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len().saturating_mul(4).max(4096));
    decompress_deflate64_to_writer(data, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Block decoder (shared by fixed and dynamic paths)
// ---------------------------------------------------------------------------

fn decode_block<W: Write>(
    lit: &HuffTable,
    dist: &HuffTable,
    br: &mut BitReader<'_>,
    window: &mut [u8],
    wpos: &mut usize,
    out: &mut W,
    flush_buf: &mut Vec<u8>,
) -> GzippyResult<u64> {
    let mut total = 0u64;

    loop {
        let sym = lit.decode(br)?;

        if sym < 256 {
            // Literal byte
            let b = sym as u8;
            out.write_all(std::slice::from_ref(&b))?;
            window[*wpos] = b;
            *wpos = (*wpos + 1) % WINDOW;
            total += 1;
        } else if sym == 256 {
            // End of block
            break;
        } else {
            // Back-reference
            let idx = (sym - 257) as usize;
            if idx >= 29 {
                return Err(GzippyError::decompression("deflate64: invalid length code"));
            }
            let extra_len = LENGTH_EXTRA[idx];
            let length = LENGTH_BASE[idx]
                + if extra_len > 0 {
                    br.read_bits(extra_len)?
                } else {
                    0
                };

            let dist_code = dist.decode(br)? as usize;
            if dist_code >= 32 {
                return Err(GzippyError::decompression(
                    "deflate64: invalid distance code",
                ));
            }
            let extra_dist = DIST_EXTRA[dist_code];
            let distance = DIST_BASE[dist_code]
                + if extra_dist > 0 {
                    br.read_bits(extra_dist)?
                } else {
                    0
                };

            if distance as usize > WINDOW {
                return Err(GzippyError::decompression(
                    "deflate64: distance exceeds window",
                ));
            }

            // Copy `length` bytes from window at `distance` back, byte-by-byte
            // so overlapping copies (run-length encoding) work correctly.
            flush_buf.clear();
            let mut copy_src = (*wpos + WINDOW - distance as usize) % WINDOW;
            for _ in 0..length {
                let b = window[copy_src];
                flush_buf.push(b);
                window[*wpos] = b;
                *wpos = (*wpos + 1) % WINDOW;
                copy_src = (copy_src + 1) % WINDOW;
            }
            out.write_all(flush_buf)?;
            total += length as u64;
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Dynamic Huffman table reader
// ---------------------------------------------------------------------------

fn read_dynamic_tables(br: &mut BitReader<'_>) -> GzippyResult<(HuffTable, HuffTable)> {
    let hlit = br.read_bits(5)? as usize + 257;
    let hdist = br.read_bits(5)? as usize + 1;
    let hclen = br.read_bits(4)? as usize + 4;

    // Read code-length alphabet code lengths.
    let mut cl_lens = [0u8; 19];
    for i in 0..hclen {
        cl_lens[CL_ORDER[i]] = br.read_bits(3)? as u8;
    }
    let cl_table = HuffTable::build(&cl_lens)?;

    // Decode literal/length + distance code lengths together.
    let total = hlit + hdist;
    let mut lengths = vec![0u8; total];
    let mut i = 0;
    while i < total {
        let sym = cl_table.decode(br)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                // Repeat previous length 3-6 times.
                if i == 0 {
                    return Err(GzippyError::decompression(
                        "deflate64: repeat with no prior code",
                    ));
                }
                let count = br.read_bits(2)? as usize + 3;
                let prev = lengths[i - 1];
                for _ in 0..count {
                    if i >= total {
                        return Err(GzippyError::decompression(
                            "deflate64: code length overflow",
                        ));
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                // Repeat zero 3-10 times.
                let count = br.read_bits(3)? as usize + 3;
                i += count;
                if i > total {
                    return Err(GzippyError::decompression(
                        "deflate64: code length overflow",
                    ));
                }
            }
            18 => {
                // Repeat zero 11-138 times.
                let count = br.read_bits(7)? as usize + 11;
                i += count;
                if i > total {
                    return Err(GzippyError::decompression(
                        "deflate64: code length overflow",
                    ));
                }
            }
            _ => {
                return Err(GzippyError::decompression(
                    "deflate64: invalid code-length symbol",
                ))
            }
        }
    }

    let lit_table = HuffTable::build(&lengths[..hlit])?;
    let dist_table = HuffTable::build(&lengths[hlit..])?;
    Ok((lit_table, dist_table))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Produces a trivial Deflate64 stream using a stored block (BTYPE=00).
    // Format: BFINAL=1, BTYPE=00, then 0-pad to byte, then LEN NLEN data.
    fn make_stored_stream(data: &[u8]) -> Vec<u8> {
        let len = data.len() as u16;
        let nlen = !len;
        // First byte: BFINAL=1 (bit 0), BTYPE=00 (bits 1-2), rest 0 → 0x01
        let mut out = vec![0x01u8];
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn test_stored_block_roundtrip() {
        let input = b"hello, deflate64!";
        let stream = make_stored_stream(input);
        let got = decompress_deflate64(&stream).unwrap();
        assert_eq!(got, input);
    }

    #[test]
    fn test_empty_stored_block() {
        let stream = make_stored_stream(b"");
        let got = decompress_deflate64(&stream).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn test_stored_block_large() {
        let input: Vec<u8> = (0..60_000).map(|i| (i % 251) as u8).collect();
        let stream = make_stored_stream(&input);
        let got = decompress_deflate64(&stream).unwrap();
        assert_eq!(got, input);
    }

    // Minimal fixed-Huffman block that encodes a single literal 'A' (65) then EOB.
    // Encoded with RFC 1951 §3.2.6 fixed codes:
    //   BFINAL=1, BTYPE=01 → header byte 0x03 (bits: 1 01 00000)
    //   Literal 65 ('A') → 9-bit code for symbols 144-255: 1_0001_0001 = 0x111, reversed LSB = 0b10001000 1
    //   EOB (256) → 7-bit code: 0b0000000 in the fixed table
    // Rather than hand-encoding, use zlib's fixed-Huffman output directly.
    // We test with the zlib-ng / flate2 deflate output and cross-check.
    #[test]
    fn test_fixed_huffman_literals() {
        // flate2 in deflate mode can produce standard DEFLATE which is a subset
        // of Deflate64 for lengths ≤258. Verify our decoder handles it.
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        let input = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // 52 'A's
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(input).unwrap();
        let compressed = enc.finish().unwrap();

        let got = decompress_deflate64(&compressed).unwrap();
        assert_eq!(got.as_slice(), input.as_ref());
    }

    // For back-reference and multi-block tests we avoid length code 285.
    // Standard DEFLATE emits code 285 only for match length exactly 258.
    // Deflate64 uses code 285 differently (16 extra bits, base 3), so
    // decoding flate2 output that contains code 285 would corrupt the stream.
    //
    // Rule: keep total input ≤ 258 bytes  →  max match < 258  →  no code 285.
    // For multi-block coverage use random data: no repeats, no back-refs,
    // so flate2 just emits literals (still tests multi-DEFLATE-block paths).

    #[test]
    fn test_dynamic_huffman_short_input() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // 200 bytes total, period-50 → max match 150 < 258. Dynamic Huffman
        // is triggered by the non-uniform literal distribution.
        let input: Vec<u8> = (0u8..50).cycle().take(200).collect();
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&input).unwrap();
        let compressed = enc.finish().unwrap();

        let got = decompress_deflate64(&compressed).unwrap();
        assert_eq!(got, input);
    }

    #[test]
    fn test_back_references_short_input() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // 160 bytes total; max match = 152 < 258. Highly repetitive to ensure
        // the compressor emits back-references.
        let input: Vec<u8> = b"abcdefgh".iter().cloned().cycle().take(160).collect();
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&input).unwrap();
        let compressed = enc.finish().unwrap();

        let got = decompress_deflate64(&compressed).unwrap();
        assert_eq!(got, input);
    }

    #[test]
    fn test_multi_stored_blocks() {
        // Two stored DEFLATE blocks in a single stream — tests cross-block
        // window continuity without any Huffman encoding.
        let data1: Vec<u8> = (0u8..=255).collect();
        let data2: Vec<u8> = (0u8..=255).rev().collect();

        let mut stream = Vec::new();
        // Block 1: BFINAL=0, BTYPE=00
        stream.push(0x00u8);
        let l1 = data1.len() as u16;
        stream.extend_from_slice(&l1.to_le_bytes());
        stream.extend_from_slice(&(!l1).to_le_bytes());
        stream.extend_from_slice(&data1);
        // Block 2: BFINAL=1, BTYPE=00
        stream.push(0x01u8);
        let l2 = data2.len() as u16;
        stream.extend_from_slice(&l2.to_le_bytes());
        stream.extend_from_slice(&(!l2).to_le_bytes());
        stream.extend_from_slice(&data2);

        let got = decompress_deflate64(&stream).unwrap();
        let mut expected = data1;
        expected.extend_from_slice(&data2);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_multi_huffman_block_large() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // 70 KB of pseudo-random bytes: incompressible, so flate2 emits pure
        // literals — no back-references, therefore no length code 285.
        // The large size forces multiple DEFLATE blocks, exercising the
        // cross-block window and bit-reader state.
        let mut x = 0xdeadbeefu32;
        let input: Vec<u8> = (0..70_000)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 24) as u8
            })
            .collect();
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&input).unwrap();
        let compressed = enc.finish().unwrap();

        let got = decompress_deflate64(&compressed).unwrap();
        assert_eq!(got, input);
    }
}
