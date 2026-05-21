/// Deflate64 (Enhanced Deflate, ZIP method 9) decompressor.
///
/// Differences from standard DEFLATE (RFC 1951):
/// - 64 KB (65536-byte) sliding window instead of 32 KB
/// - Length code 285: 16 extra bits, base = 3  (max length 65538)
/// - Distance codes 30-31: 14 extra bits each  (max distance 65536)
use crate::error::{GzippyError, GzippyResult};
use std::io::Write;
use std::sync::OnceLock;

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

// ---------------------------------------------------------------------------
// Fixed Huffman tables — built once, reused across all fixed blocks.
// ---------------------------------------------------------------------------

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

static FIXED_LIT: OnceLock<HuffTable> = OnceLock::new();
static FIXED_DIST: OnceLock<HuffTable> = OnceLock::new();

fn fixed_lit_table() -> &'static HuffTable {
    FIXED_LIT.get_or_init(|| {
        HuffTable::build(&fixed_lit_lengths()).expect("fixed lit table is always valid")
    })
}

fn fixed_dist_table() -> &'static HuffTable {
    FIXED_DIST.get_or_init(|| {
        HuffTable::build(&fixed_dist_lengths()).expect("fixed dist table is always valid")
    })
}

// ---------------------------------------------------------------------------
// Huffman decoder — canonical codes, max 15-bit code lengths.
// ---------------------------------------------------------------------------

struct HuffTable {
    counts: [u16; 16], // how many codes have each bit-length
    symbols: Vec<u16>, // symbols sorted by (length, code value)
}

// SAFETY: HuffTable contains only [u16;16] and Vec<u16>, both Send+Sync.
unsafe impl Send for HuffTable {}
unsafe impl Sync for HuffTable {}

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
    /// Uses the puff.c canonical decode algorithm. `code`, `first`, and
    /// `index` are signed so that `code - count` can go negative when
    /// code < count — matching the C `int` arithmetic puff.c relies on.
    /// A bounds check guards against panics on malformed/oversubscribed trees.
    #[inline(always)]
    fn decode(&self, br: &mut BitReader) -> GzippyResult<u16> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;

        for bits in 1u8..=15 {
            code |= br.read_bit()? as i32;
            let count = self.counts[bits as usize] as i32;
            if code - count < first {
                let sym_idx = (index + (code - first)) as usize;
                return self.symbols.get(sym_idx).copied().ok_or_else(|| {
                    GzippyError::decompression("deflate64: Huffman tree index out of bounds")
                });
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

    /// Align to the next byte boundary and read LEN/NLEN for a stored block.
    fn read_u16_pair(&mut self) -> GzippyResult<(u16, u16)> {
        // Discard any partial-byte bits.
        let partial = self.avail % 8;
        if partial != 0 {
            self.buf >>= partial;
            self.avail -= partial;
        }
        // Rewind src cursor by however many full bytes are still buffered
        // so we read LEN/NLEN from the correct position.
        let buffered_bytes = (self.avail / 8) as usize;
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
    let mut window = vec![0u8; WINDOW];
    let mut wpos: usize = 0;
    // Bytes actually written into the window (saturates at WINDOW).
    // Used to reject back-references that predate the start of output.
    let mut filled: usize = 0;
    let mut total: u64 = 0;
    // Reusable scratch buffer — amortises allocation across all blocks.
    let mut flush_buf: Vec<u8> = Vec::with_capacity(65538);

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
                for &b in bytes {
                    window[wpos] = b;
                    wpos = (wpos + 1) % WINDOW;
                }
                filled = (filled + len as usize).min(WINDOW);
                total += len as u64;
            }
            0b01 => {
                // Fixed Huffman — tables are built once and cached.
                total += decode_block(
                    fixed_lit_table(),
                    fixed_dist_table(),
                    &mut br,
                    &mut window,
                    &mut wpos,
                    &mut filled,
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
                    &mut filled,
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

#[allow(clippy::too_many_arguments)]
fn decode_block<W: Write>(
    lit: &HuffTable,
    dist: &HuffTable,
    br: &mut BitReader<'_>,
    window: &mut [u8],
    wpos: &mut usize,
    filled: &mut usize,
    out: &mut W,
    flush_buf: &mut Vec<u8>,
) -> GzippyResult<u64> {
    let mut total = 0u64;
    // Flush threshold: drain literals to the writer in chunks of this size
    // to bound in-flight memory while keeping syscall overhead low.
    const LIT_FLUSH: usize = 32768;

    loop {
        let sym = lit.decode(br)?;

        if sym < 256 {
            // Literal — buffer for batched write.
            let b = sym as u8;
            flush_buf.push(b);
            window[*wpos] = b;
            *wpos = (*wpos + 1) % WINDOW;
            *filled = (*filled + 1).min(WINDOW);
            total += 1;

            if flush_buf.len() == LIT_FLUSH {
                out.write_all(flush_buf)?;
                flush_buf.clear();
            }
        } else if sym == 256 {
            // End of block — flush any buffered literals.
            if !flush_buf.is_empty() {
                out.write_all(flush_buf)?;
                flush_buf.clear();
            }
            break;
        } else {
            // Back-reference — flush buffered literals first.
            if !flush_buf.is_empty() {
                out.write_all(flush_buf)?;
                flush_buf.clear();
            }

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

            let distance = distance as usize;
            // Guard: distance must not exceed bytes actually written so far.
            if distance > *filled {
                return Err(GzippyError::decompression(
                    "deflate64: back-reference distance exceeds available output",
                ));
            }

            // Copy `length` bytes from the ring buffer, byte-by-byte so
            // overlapping copies (run-length expansion) work correctly.
            let mut copy_src = (*wpos + WINDOW - distance) % WINDOW;
            for _ in 0..length {
                let b = window[copy_src];
                flush_buf.push(b);
                window[*wpos] = b;
                *wpos = (*wpos + 1) % WINDOW;
                *filled = (*filled + 1).min(WINDOW);
                copy_src = (copy_src + 1) % WINDOW;
            }
            out.write_all(flush_buf)?;
            flush_buf.clear();
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

    // ── Bit-stream builder for hand-crafted Deflate64 test vectors ────────────

    /// Writes bits into a byte vector, LSB-first within each byte.
    struct BitWriter {
        data: Vec<u8>,
        cur: u8,
        nbits: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                data: Vec::new(),
                cur: 0,
                nbits: 0,
            }
        }

        fn write_bit(&mut self, bit: u8) {
            self.cur |= bit << self.nbits;
            self.nbits += 1;
            if self.nbits == 8 {
                self.data.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }

        /// Write `n` bits of `val`, LSB first — used for BFINAL/BTYPE and extra bits.
        fn write_lsb(&mut self, mut val: u64, n: u8) {
            for _ in 0..n {
                self.write_bit((val & 1) as u8);
                val >>= 1;
            }
        }

        /// Write `n` bits of `code`, MSB first — required for Huffman code words.
        fn write_msb(&mut self, code: u32, n: u8) {
            for i in (0..n).rev() {
                self.write_bit(((code >> i) & 1) as u8);
            }
        }

        /// Write a fixed lit/len symbol using RFC 1951 §3.2.6 code lengths.
        fn write_fixed_lit(&mut self, sym: u16) {
            match sym {
                0..=143 => self.write_msb(48 + sym as u32, 8),
                144..=255 => self.write_msb(400 + (sym - 144) as u32, 9),
                256..=279 => self.write_msb((sym - 256) as u32, 7),
                _ => self.write_msb(192 + (sym - 280) as u32, 8),
            }
        }

        /// Write a fixed distance symbol (all 32 codes have length 5, value = code).
        fn write_fixed_dist(&mut self, dist_code: u8) {
            self.write_msb(dist_code as u32, 5);
        }

        fn finish(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                self.data.push(self.cur);
            }
            self.data
        }
    }

    // ── Stored-block helpers ──────────────────────────────────────────────────

    fn make_stored_stream(data: &[u8]) -> Vec<u8> {
        let len = data.len() as u16;
        let nlen = !len;
        let mut out = vec![0x01u8]; // BFINAL=1, BTYPE=00
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    // ── Stored-block tests ────────────────────────────────────────────────────

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

    // ── Deflate64-specific golden tests (hand-crafted bit streams) ────────────

    /// Length code 285 with 16 extra bits — the primary Deflate64 extension.
    ///
    /// Stream: fixed-Huffman block containing literal 'a' followed by a
    /// back-reference (length=1000, distance=1) then EOB.
    /// Deflate64 length code 285: base=3, extra=16; extra value = 997.
    /// Expected output: 'a' × 1001.
    #[test]
    fn test_length_code_285_extended() {
        let mut bw = BitWriter::new();
        // Block header: BFINAL=1 (1 bit), BTYPE=01 fixed-Huffman (2 bits LSB).
        bw.write_lsb(1, 1);
        bw.write_lsb(1, 2); // BTYPE=01 → value 1, 2 bits LSB-first
                            // Literal 'a' (97): fixed 8-bit code = 48+97 = 145.
        bw.write_fixed_lit(97);
        // Length code 285: fixed 8-bit code = 192+(285-280) = 197.
        // In Deflate64: base=3, 16 extra bits; extra = 1000-3 = 997.
        bw.write_fixed_lit(285);
        bw.write_lsb(997, 16);
        // Distance code 0 (distance=1): 5-bit fixed code = 0, no extra bits.
        bw.write_fixed_dist(0);
        // EOB (256): fixed 7-bit code = 0.
        bw.write_fixed_lit(256);

        let stream = bw.finish();
        let got = decompress_deflate64(&stream).unwrap();
        assert_eq!(got, vec![b'a'; 1001]);
    }

    /// Distance code 30 (Deflate64 addition) and a back-reference that crosses
    /// the standard DEFLATE 32 KB window boundary.
    ///
    /// Stream:
    ///   stored block (BFINAL=0): 33 000 bytes of 'b'
    ///   fixed-Huffman block (BFINAL=1): back-ref (length=3, distance=33 000) + EOB
    /// Expected output: 'b' × 33 003.
    #[test]
    fn test_distance_code_30_and_large_window() {
        let n: usize = 33_000;
        let data_b = vec![b'b'; n];

        // Block 1: stored, BFINAL=0, BTYPE=00.
        let mut stream = vec![0x00u8]; // BFINAL=0, BTYPE=00
        let len16 = n as u16;
        stream.extend_from_slice(&len16.to_le_bytes());
        stream.extend_from_slice(&(!len16).to_le_bytes());
        stream.extend_from_slice(&data_b);

        // Block 2: fixed Huffman, BFINAL=1, BTYPE=01.
        // Length code 257 (length=3, 7-bit fixed code=1, 0 extra bits).
        // Distance code 30: base=32769, 14 extra bits; extra = 33000-32769 = 231.
        let mut bw = BitWriter::new();
        bw.write_lsb(1, 1); // BFINAL=1
        bw.write_lsb(1, 2); // BTYPE=01
        bw.write_fixed_lit(257); // length=3
        bw.write_fixed_dist(30); // dist code 30
        bw.write_lsb(231, 14); // extra bits for distance
        bw.write_fixed_lit(256); // EOB
        stream.extend_from_slice(&bw.finish());

        let got = decompress_deflate64(&stream).unwrap();
        let mut expected = data_b;
        expected.extend_from_slice(b"bbb");
        assert_eq!(got, expected);
    }

    // ── flate2-based correctness tests ───────────────────────────────────────
    //
    // Standard DEFLATE is a strict subset of Deflate64 *except* length code 285
    // (DEFLATE: fixed length 258; Deflate64: 16 extra bits).  flate2 only emits
    // code 285 when a match is exactly 258 bytes, which requires total input >
    // 258 bytes of repetitive data.  All inputs below are ≤ 258 bytes total so
    // code 285 cannot appear.  For multi-block coverage we use incompressible
    // data (no back-references, hence no code 285 regardless of size).

    #[test]
    fn test_fixed_huffman_literals() {
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

    #[test]
    fn test_dynamic_huffman_short_input() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

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

        // 160 bytes total; max match = 152 < 258.
        let input: Vec<u8> = b"abcdefgh".iter().cloned().cycle().take(160).collect();
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&input).unwrap();
        let compressed = enc.finish().unwrap();

        let got = decompress_deflate64(&compressed).unwrap();
        assert_eq!(got, input);
    }

    #[test]
    fn test_multi_stored_blocks() {
        let data1: Vec<u8> = (0u8..=255).collect();
        let data2: Vec<u8> = (0u8..=255).rev().collect();

        let mut stream = Vec::new();
        stream.push(0x00u8); // BFINAL=0, BTYPE=00
        let l1 = data1.len() as u16;
        stream.extend_from_slice(&l1.to_le_bytes());
        stream.extend_from_slice(&(!l1).to_le_bytes());
        stream.extend_from_slice(&data1);
        stream.push(0x01u8); // BFINAL=1, BTYPE=00
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

        // 70 KB of LCG pseudo-random bytes: incompressible, so flate2 emits
        // pure literals — no back-references, no length code 285.
        // The large size forces multiple DEFLATE blocks.
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
