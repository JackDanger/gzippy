/// Deflate64 (Enhanced Deflate, ZIP method 9) compressor.
///
/// Produces a raw Deflate64 bitstream.  Greedy LZ77 + dynamic Huffman per block.
/// Hash-3 chain match finder; window = 65536; chain depth = 32; max length = 65538.
use crate::error::{GzippyError, GzippyResult};

// ---------------------------------------------------------------------------
// Constants — identical to the decoder's tables.
// ---------------------------------------------------------------------------

const WINDOW: usize = 65536;
const BLOCK_TOKENS: usize = 16384;
const CHAIN_DEPTH: usize = 32;

#[rustfmt::skip]
const LENGTH_BASE: [u32; 29] = [
     3,  4,  5,  6,  7,  8,  9, 10,
    11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115,
    131, 163, 195, 227,
    3, // code 285 — Deflate64: base 3 + 16 extra bits
];

#[rustfmt::skip]
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 5,
    16, // code 285 — Deflate64 extension
];

#[rustfmt::skip]
const DIST_BASE: [u32; 32] = [
       1,    2,    3,    4,    5,    7,    9,   13,
      17,   25,   33,   49,   65,   97,  129,  193,
     257,  385,  513,  769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193,12289,16385,24577,
    32769, 49153,
];

#[rustfmt::skip]
const DIST_EXTRA: [u8; 32] = [
     0, 0, 0, 0, 1, 1, 2, 2,
     3, 3, 4, 4, 5, 5, 6, 6,
     7, 7, 8, 8, 9, 9,10,10,
    11,11,12,12,13,13,
    14, 14,
];

const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Token {
    Literal(u8),
    Match { length: u32, dist: u32 },
}

// ---------------------------------------------------------------------------
// Bit writer — LSB-first within bytes, MSB-first for Huffman codes.
// ---------------------------------------------------------------------------

struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    #[inline]
    fn write_bit(&mut self, bit: u8) {
        self.cur |= bit << self.nbits;
        self.nbits += 1;
        if self.nbits == 8 {
            self.out.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Write `n` bits of `val`, LSB first (block headers, extra bits, etc.).
    #[inline]
    fn write_lsb(&mut self, mut val: u64, n: u8) {
        for _ in 0..n {
            self.write_bit((val & 1) as u8);
            val >>= 1;
        }
    }

    /// Write a canonical Huffman code: `n` bits, MSB first.
    /// The encoder stores codes in natural (MSB-first) form; to emit via
    /// LSB-first writer we reverse the bits within the code length.
    #[inline]
    fn write_code(&mut self, code: u32, len: u8) {
        // Reverse `len` bits of `code` then emit LSB-first.
        let mut rev = 0u32;
        let mut v = code;
        for _ in 0..len {
            rev = (rev << 1) | (v & 1);
            v >>= 1;
        }
        self.write_lsb(rev as u64, len);
    }

    fn flush(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push(self.cur);
        }
        self.out
    }

    /// Flush only fully-completed bytes to `w`; the partial byte in `self.cur`
    /// is retained so subsequent blocks continue in the same bit stream.
    fn flush_to<W: std::io::Write>(&mut self, w: &mut W) -> GzippyResult<()> {
        w.write_all(&self.out)?;
        self.out.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Match finder — hash-3 chain, window = 65536, depth = 32.
// ---------------------------------------------------------------------------

struct MatchFinder {
    head: Vec<u32>, // hash -> most recent position (u32::MAX = none)
    prev: Vec<u32>, // prev[pos & (WINDOW-1)] -> previous position
    hash: u32,
}

const HASH_BITS: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: u32 = (HASH_SIZE - 1) as u32;

#[inline]
fn hash3(a: u8, b: u8, c: u8) -> u32 {
    // Knuth multiplicative hash over 3 bytes.
    let v = (a as u32) | ((b as u32) << 8) | ((c as u32) << 16);
    (v.wrapping_mul(0x9e3779b9)) >> (32 - HASH_BITS)
}

impl MatchFinder {
    fn new() -> Self {
        Self {
            head: vec![u32::MAX; HASH_SIZE],
            prev: vec![u32::MAX; WINDOW],
            hash: 0,
        }
    }

    /// Insert `pos` into the hash chain for the 3-byte sequence at `data[pos..]`.
    #[inline]
    fn insert(&mut self, data: &[u8], pos: usize) {
        if pos + 3 > data.len() {
            return;
        }
        let h = hash3(data[pos], data[pos + 1], data[pos + 2]) & HASH_MASK;
        self.hash = h;
        let slot = pos & (WINDOW - 1);
        self.prev[slot] = self.head[h as usize];
        self.head[h as usize] = pos as u32;
    }

    /// Find the longest match starting at `pos` in `data`.
    /// Returns `(length, distance)` or `(0, 0)` if no match >= 3.
    fn find_match(&mut self, data: &[u8], pos: usize) -> (u32, u32) {
        let n = data.len();
        if pos + 3 > n {
            return (0, 0);
        }
        let h = hash3(data[pos], data[pos + 1], data[pos + 2]) & HASH_MASK;
        self.hash = h;
        let slot = pos & (WINDOW - 1);
        self.prev[slot] = self.head[h as usize];
        self.head[h as usize] = pos as u32;

        let max_len = (n - pos).min(65538) as u32;
        let mut best_len = 0u32;
        let mut best_dist = 0u32;
        let mut cur = self.prev[slot];
        let mut steps = 0;

        while cur != u32::MAX && steps < CHAIN_DEPTH {
            let cur_pos = cur as usize;
            // Distance must fit in 65536.
            let dist = pos - cur_pos;
            if dist > WINDOW {
                break;
            }
            // Quick check: first bytes must match.
            if data[cur_pos] == data[pos] {
                let mut ml = 1u32;
                while ml < max_len && data[cur_pos + ml as usize] == data[pos + ml as usize] {
                    ml += 1;
                }
                if ml >= 3 && ml > best_len {
                    best_len = ml;
                    best_dist = dist as u32;
                    if best_len == max_len {
                        break;
                    }
                }
            }
            cur = self.prev[cur_pos & (WINDOW - 1)];
            steps += 1;
        }

        (best_len, best_dist)
    }
}

// ---------------------------------------------------------------------------
// Length/distance → code lookup
// ---------------------------------------------------------------------------

fn length_code(length: u32) -> (u32, u8, u32) {
    // Returns (code 257..=285, extra_bits_count, extra_value).
    if length >= 259 {
        // Deflate64 extension: code 285, 16 extra bits, base = 3.
        return (285, 16, length - 3);
    }
    // Binary search over LENGTH_BASE for codes 257..=284 (indices 0..28).
    let idx = LENGTH_BASE[..28]
        .partition_point(|&b| b <= length)
        .saturating_sub(1);
    let base = LENGTH_BASE[idx];
    let extra_bits = LENGTH_EXTRA[idx];
    (257 + idx as u32, extra_bits, length - base)
}

fn dist_code(dist: u32) -> (u32, u8, u32) {
    // Returns (dist_code 0..=31, extra_bits_count, extra_value).
    let idx = DIST_BASE.partition_point(|&b| b <= dist).saturating_sub(1);
    let base = DIST_BASE[idx];
    let extra_bits = DIST_EXTRA[idx];
    (idx as u32, extra_bits, dist - base)
}

// ---------------------------------------------------------------------------
// Package-merge length-limited Huffman (max_len bits, RFC 1951 §3.2.2).
// ---------------------------------------------------------------------------

fn package_merge(freqs: &[u32], max_len: u8) -> Vec<u8> {
    let n = freqs.len();
    // Symbols with freq > 0.
    let active: Vec<usize> = (0..n).filter(|&i| freqs[i] > 0).collect();
    let m = active.len();

    if m == 0 {
        return vec![0u8; n];
    }
    if m == 1 {
        // Single symbol — assign length 1 (can't have length 0 in a valid tree).
        let mut lens = vec![0u8; n];
        lens[active[0]] = 1;
        return lens;
    }

    // Package-merge requires per-symbol counts, not a set: the same symbol can
    // appear in multiple selected packages, and its code length = total appearances.
    // u16 fits 2^15 = 32768 (max possible count at level 15).
    let mut items: Vec<(u64, Vec<u16>)> = active
        .iter()
        .enumerate()
        .map(|(idx, &sym)| {
            let mut counts = vec![0u16; m];
            counts[idx] = 1;
            (freqs[sym] as u64, counts)
        })
        .collect();
    items.sort_by_key(|(w, _)| *w);

    let mut prev_level = items.clone();
    for _ in 0..(max_len - 1) {
        let mut packages: Vec<(u64, Vec<u16>)> = prev_level
            .chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| {
                let w = c[0].0.saturating_add(c[1].0);
                let mut counts = c[0].1.clone();
                for (a, b) in counts.iter_mut().zip(c[1].1.iter()) {
                    *a = a.saturating_add(*b);
                }
                (w, counts)
            })
            .collect();

        packages.extend_from_slice(&items);
        packages.sort_by_key(|(w, _)| *w);
        prev_level = packages;
    }

    let take = 2 * m - 2;
    let selected = &prev_level[..take.min(prev_level.len())];

    let mut totals = vec![0u32; m];
    for (_, counts) in selected {
        for (i, &c) in counts.iter().enumerate() {
            totals[i] += c as u32;
        }
    }

    let mut lens = vec![0u8; n];
    for (idx, &sym) in active.iter().enumerate() {
        lens[sym] = totals[idx] as u8;
    }
    lens
}

// ---------------------------------------------------------------------------
// Canonical code assignment from lengths (RFC 1951 §3.2.2).
// ---------------------------------------------------------------------------

fn assign_codes(lens: &[u8]) -> Vec<u32> {
    let max_len = *lens.iter().max().unwrap_or(&0) as usize;
    let mut bl_count = vec![0u32; max_len + 1];
    for &l in lens {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = vec![0u32; max_len + 2];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut codes = vec![0u32; lens.len()];
    for (i, &l) in lens.iter().enumerate() {
        if l > 0 {
            codes[i] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
    }
    codes
}

// ---------------------------------------------------------------------------
// RLE-encode combined lit+dist length array using the CL alphabet.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum ClSym {
    Lit(u8),      // 0..=15
    Repeat16(u8), // 16: repeat prev, extra = count-3 (0..=3)
    Zeros17(u8),  // 17: repeat zero, extra = count-3 (0..=7)
    Zeros18(u8),  // 18: repeat zero, extra = count-11 (0..=127)
}

fn rle_lengths(lens: &[u8]) -> Vec<ClSym> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lens.len() {
        let l = lens[i];
        if l == 0 {
            // Count run of zeros.
            let mut run = 1;
            while i + run < lens.len() && lens[i + run] == 0 && run < 138 {
                run += 1;
            }
            if run >= 11 {
                out.push(ClSym::Zeros18((run - 11) as u8));
            } else if run >= 3 {
                out.push(ClSym::Zeros17((run - 3) as u8));
            } else {
                for _ in 0..run {
                    out.push(ClSym::Lit(0));
                }
            }
            i += run;
        } else {
            out.push(ClSym::Lit(l));
            i += 1;
            // Count run of identical non-zero value.
            let mut run = 0;
            while i + run < lens.len() && lens[i + run] == l && run < 6 {
                run += 1;
            }
            if run >= 3 {
                out.push(ClSym::Repeat16((run - 3) as u8));
                i += run;
            }
        }
    }
    out
}

fn cl_sym_index(sym: &ClSym) -> usize {
    match sym {
        ClSym::Lit(l) => *l as usize,
        ClSym::Repeat16(_) => 16,
        ClSym::Zeros17(_) => 17,
        ClSym::Zeros18(_) => 18,
    }
}

// ---------------------------------------------------------------------------
// Block encoder — build tokens, Huffman tables, emit dynamic block.
// ---------------------------------------------------------------------------

fn emit_block(tokens: &[Token], bfinal: u8, bw: &mut BitWriter) -> GzippyResult<()> {
    // --- Frequency counting ---
    let mut lit_freq = [0u32; 286];
    let mut dist_freq = [0u32; 32];
    lit_freq[256] = 1; // EOB
    for &tok in tokens {
        match tok {
            Token::Literal(b) => lit_freq[b as usize] += 1,
            Token::Match { length, dist } => {
                let (lcode, _, _) = length_code(length);
                lit_freq[lcode as usize] += 1;
                let (dcode, _, _) = dist_code(dist);
                dist_freq[dcode as usize] += 1;
            }
        }
    }

    // Ensure at least one dist symbol so HuffTable::build succeeds.
    if dist_freq.iter().all(|&f| f == 0) {
        dist_freq[0] = 1;
    }

    // --- Build length-limited Huffman lengths ---
    let lit_lens = package_merge(&lit_freq, 15);
    let mut dist_lens = package_merge(&dist_freq, 15);

    // Guarantee at least one dist code with length > 0.
    if dist_lens.iter().all(|&l| l == 0) {
        dist_lens[0] = 1;
    }

    let lit_codes = assign_codes(&lit_lens);
    let dist_codes = assign_codes(&dist_lens);

    // --- Determine HLIT and HDIST ---
    let hlit = lit_lens[..286]
        .iter()
        .rposition(|&l| l > 0)
        .map(|p| p + 1)
        .unwrap_or(257)
        .max(257);
    let hdist = dist_lens[..32]
        .iter()
        .rposition(|&l| l > 0)
        .map(|p| p + 1)
        .unwrap_or(1)
        .max(1);

    // --- Combined length array for CL encoding ---
    let mut combined = Vec::with_capacity(hlit + hdist);
    combined.extend_from_slice(&lit_lens[..hlit]);
    combined.extend_from_slice(&dist_lens[..hdist]);
    let cl_syms = rle_lengths(&combined);

    // CL symbol frequencies.
    let mut cl_freq = [0u32; 19];
    for s in &cl_syms {
        cl_freq[cl_sym_index(s)] += 1;
    }
    let cl_lens_raw = package_merge(&cl_freq, 7);
    let cl_codes = assign_codes(&cl_lens_raw);

    // HCLEN: trim trailing zero CL codes (but at least 4).
    let hclen = CL_ORDER
        .iter()
        .rposition(|&i| cl_lens_raw[i] > 0)
        .map(|p| p + 1)
        .unwrap_or(4)
        .max(4);

    // --- Emit block header ---
    bw.write_lsb(bfinal as u64, 1);
    bw.write_lsb(0b10, 2); // BTYPE = dynamic Huffman

    bw.write_lsb((hlit - 257) as u64, 5);
    bw.write_lsb((hdist - 1) as u64, 5);
    bw.write_lsb((hclen - 4) as u64, 4);

    for i in 0..hclen {
        bw.write_lsb(cl_lens_raw[CL_ORDER[i]] as u64, 3);
    }

    // --- Emit RLE-encoded lengths ---
    for s in &cl_syms {
        let idx = cl_sym_index(s);
        bw.write_code(cl_codes[idx], cl_lens_raw[idx]);
        match s {
            ClSym::Lit(_) => {}
            ClSym::Repeat16(extra) => bw.write_lsb(*extra as u64, 2),
            ClSym::Zeros17(extra) => bw.write_lsb(*extra as u64, 3),
            ClSym::Zeros18(extra) => bw.write_lsb(*extra as u64, 7),
        }
    }

    // --- Emit tokens ---
    for &tok in tokens {
        match tok {
            Token::Literal(b) => {
                let sym = b as usize;
                bw.write_code(lit_codes[sym], lit_lens[sym]);
            }
            Token::Match { length, dist } => {
                let (lcode, lextra, lval) = length_code(length);
                bw.write_code(lit_codes[lcode as usize], lit_lens[lcode as usize]);
                if lextra > 0 {
                    bw.write_lsb(lval as u64, lextra);
                }
                let (dcode, dextra, dval) = dist_code(dist);
                bw.write_code(dist_codes[dcode as usize], dist_lens[dcode as usize]);
                if dextra > 0 {
                    bw.write_lsb(dval as u64, dextra);
                }
            }
        }
    }

    // EOB
    bw.write_code(lit_codes[256], lit_lens[256]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compress `data` as a raw Deflate64 bitstream, returning `Vec<u8>`.
#[allow(dead_code)]
pub fn compress_deflate64(data: &[u8]) -> GzippyResult<Vec<u8>> {
    let mut out = Vec::new();
    compress_deflate64_to_writer(data, &mut out)?;
    Ok(out)
}

/// Compress `data` as a raw Deflate64 bitstream, writing to `writer`.
/// Returns the number of compressed bytes written.
#[allow(dead_code)]
pub fn compress_deflate64_to_writer<W: std::io::Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    if data.is_empty() {
        // Empty stored block: BFINAL=1, BTYPE=00, LEN=0, NLEN=0xFFFF.
        let block = [0x01u8, 0x00, 0x00, 0xFF, 0xFF];
        writer.write_all(&block)?;
        return Ok(5);
    }

    let mut mf = MatchFinder::new();
    let mut bw = BitWriter::new();
    let mut tokens: Vec<Token> = Vec::with_capacity(BLOCK_TOKENS + 16);
    let n = data.len();
    let mut pos = 0usize;
    let mut written = 0u64;

    while pos < n {
        tokens.clear();

        while pos < n && tokens.len() < BLOCK_TOKENS {
            let (ml, md) = mf.find_match(data, pos);
            if ml >= 3 {
                tokens.push(Token::Match {
                    length: ml,
                    dist: md,
                });
                for k in 1..ml as usize {
                    mf.insert(data, pos + k);
                }
                pos += ml as usize;
            } else {
                tokens.push(Token::Literal(data[pos]));
                pos += 1;
            }
        }

        let is_last = pos >= n;
        emit_block(&tokens, if is_last { 1 } else { 0 }, &mut bw)
            .map_err(|e| GzippyError::compression(e.to_string()))?;

        // Flush complete bytes to the writer; partial byte stays in bw.
        let flushed = bw.out.len() as u64;
        bw.flush_to(writer)?;
        written += flushed;
    }

    // Emit final partial byte (if any).
    let remaining = bw.flush();
    written += remaining.len() as u64;
    writer.write_all(&remaining)?;
    Ok(written)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::deflate64::decompress_deflate64;

    fn roundtrip(data: &[u8]) {
        let compressed = compress_deflate64(data).expect("compress failed");
        let got = decompress_deflate64(&compressed).expect("decompress failed");
        assert_eq!(
            got,
            data,
            "roundtrip mismatch (compressed {} bytes)",
            compressed.len()
        );
    }

    #[test]
    fn test_two_block_boundary() {
        // Just over BLOCK_TOKENS=16384 bytes of unique data to test multi-block.
        let data: Vec<u8> = (0u8..=255).cycle().take(17_000).collect();
        roundtrip(&data);
    }

    #[test]
    fn test_lcg_small_sizes() {
        for size in [
            65000, 65500, 65536, 65537, 65600, 65700, 65800, 65900, 66000,
        ] {
            let mut x = 0xdeadbeefu32;
            let data: Vec<u8> = (0..size)
                .map(|_| {
                    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (x >> 24) as u8
                })
                .collect();
            let compressed = compress_deflate64(&data).expect("compress failed");
            let got = decompress_deflate64(&compressed).expect("decompress failed");
            if got != data {
                let first_diff = got
                    .iter()
                    .zip(data.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(data.len()));
                panic!(
                    "LCG size {} mismatch: first diff at byte {}, got[{}]={} want[{}]={}",
                    size, first_diff, first_diff, got[first_diff], first_diff, data[first_diff]
                );
            }
        }
    }

    #[test]
    fn test_empty() {
        roundtrip(b"");
    }

    #[test]
    fn test_short_ascii() {
        roundtrip(b"Hello, Deflate64!");
        roundtrip(b"abcdefghijklmnopqrstuvwxyz0123456789");
    }

    #[test]
    fn test_repeating_forces_code_285() {
        // 100 KB of 'a' — greedy will find matches up to 65538, exercising code 285.
        let data = vec![b'a'; 100_000];
        roundtrip(&data);
    }

    #[test]
    fn test_incompressible_multi_block() {
        // 70 KB of LCG pseudo-random bytes — essentially all literals, forces multi-block.
        let mut x = 0xdeadbeefu32;
        let data: Vec<u8> = (0..70_000)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 24) as u8
            })
            .collect();
        roundtrip(&data);
    }

    #[test]
    fn test_large_distance_codes_30_31() {
        // 50 KB of 'x' followed by 50 KB of 'x': best matches at distance > 32768
        // exercise dist codes 30 (32769..=49152) and 31 (49153..=65536).
        let data = vec![b'x'; 65536];
        roundtrip(&data);
    }

    #[test]
    fn test_mixed_content() {
        // Mix of repeated and unique bytes.
        let mut data = Vec::new();
        for i in 0..256u16 {
            data.push(i as u8);
        }
        data.extend_from_slice(&vec![b'Z'; 10_000]);
        data.extend((0u8..=255).cycle().take(5000));
        roundtrip(&data);
    }
}
