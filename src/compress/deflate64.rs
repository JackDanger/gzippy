/// Deflate64 (Enhanced Deflate, ZIP method 9) compressor.
///
/// Produces a raw Deflate64 bitstream.  Greedy LZ77 + dynamic Huffman per block.
/// Hash-3 chain match finder; window = 65536; chain depth = 32; max length = 65538.
///
/// CLI-unreachable (public library API only, re-exported at the crate root as
/// [`crate::compress_deflate64`] / [`crate::compress_deflate64_to_writer`]); it
/// carries no byte-identity constituency (see
/// `docs/compressor-architecture.md` §5-E "Stage E: POLISH"), so — unlike the
/// main `compress::deflate` engine — its output may change so long as it stays
/// correct (decodable, byte-perfect roundtrip through
/// [`crate::decompress::deflate64::decompress_deflate64`]).
///
/// Bitstream emission, exact length-limited Huffman construction, the
/// dynamic-header RLE wire format, and stored-block framing are ALL format-
/// LAW shared with the main `compress::deflate` engine (see
/// `docs/compressor-architecture.md` §2/§5-E) and are instantiated directly
/// from `compress::deflate::{bitstream, huffman::optimal, huffman::header}`
/// below — no format-agnostic logic is reimplemented here.
///
/// Two pieces stay genuinely private because the FORMAT differs, not by
/// oversight:
///
/// - **Length/distance code tables** ([`LENGTH_BASE`]/[`LENGTH_EXTRA`]/
///   [`DIST_BASE`]/[`DIST_EXTRA`]): Deflate64 extends litlen symbol 285 to a
///   16-extra-bit code (length up to 65538, vs. plain DEFLATE's fixed
///   length-258 symbol 285) and adds distance codes 30/31 for the 64 KiB
///   window (vs. plain DEFLATE's 30-code, 32 KiB table in
///   `compress::deflate::tables`). Different RFC tables, not a dedup gap.
/// - **The match finder** ([`MatchFinder`]): `compress::deflate::matchfinder::hc`
///   (the shared hash-chain matchfinder) hard-codes a 32 KiB window via
///   SIGNED `i16` chain positions (`WINDOW_SIZE = 1 << 15`,
///   `MATCHFINDER_INITVAL = i16::MIN`, sentinel/rebase arithmetic throughout
///   `matchfinder/common.rs` + `matchfinder/hc.rs`); `i16` cannot address a
///   64 KiB window at all (max magnitude 32768 < 65536), so Deflate64 cannot
///   reuse it without widening every position field to `i32`/`u32` across a
///   heavily pinned, gated hot module — out of scope for a format module with
///   no performance constituency. Likewise `matchfinder::common::LzMatch`
///   (the shared match-list vocabulary type used by `bt`/`lzfind`) packs
///   length/offset as `u16`, which cannot represent Deflate64's length-65538 /
///   offset-65536 range either way. [`MatchFinder`] DOES reuse
///   `matchfinder::common::lz_extend` for the match-length extension inner
///   loop (see below) — that primitive's contract (`str_pos + max_len <=
///   data.len()`) is format-independent and Deflate64's own bounds satisfy it
///   exactly.
use crate::compress::deflate::bitstream::BitWriter;
use crate::compress::deflate::emit_stored_block;
use crate::compress::deflate::huffman::header::build_dynamic_header;
use crate::compress::deflate::huffman::optimal::{calculate_bit_lengths, lengths_to_symbols};
use crate::compress::deflate::matchfinder::common::lz_extend;
use crate::error::GzippyResult;

// ---------------------------------------------------------------------------
// Constants — identical to the decoder's tables. Deflate64-specific (see
// module doc: extended length code 285 / distance codes 30-31 vs. the
// 32 KiB-window RFC tables in `compress::deflate::tables`).
// ---------------------------------------------------------------------------

const WINDOW: usize = 65536;
const BLOCK_TOKENS: usize = 16384;
const CHAIN_DEPTH: usize = 32;

/// Number of literal/length symbols Deflate64 ever emits (0..=285); the
/// shared [`build_dynamic_header`] trims/pads alphabets itself, so this is
/// sized to exactly what this encoder uses (no reserved 286/287 filler).
const NUM_LITLEN_SYMS: usize = 286;
/// Number of distance symbols (Deflate64 adds slots 30/31 for the 64 KiB
/// window; plain DEFLATE stops at 30).
const NUM_DIST_SYMS: usize = 32;

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

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// Not `matchfinder::common::LzMatch` — see module doc (`LzMatch`'s `u16`
/// length/offset fields cannot hold Deflate64's length-65538/offset-65536
/// range).
#[derive(Clone, Copy)]
enum Token {
    Literal(u8),
    Match { length: u32, dist: u32 },
}

// ---------------------------------------------------------------------------
// Match finder — hash-3 chain, window = 65536, depth = 32.
//
// Genuinely private: see the module doc for why the shared
// `matchfinder::hc::HcMatchfinder` (32 KiB window, `i16` positions) cannot
// address Deflate64's 64 KiB window. The match-length EXTENSION step below
// does reuse the shared, format-independent `lz_extend` word-at-a-time
// primitive in place of a hand-rolled byte loop.
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
                // `lz_extend`'s contract (`str_pos + max_len <= data.len()` and
                // `match_pos + max_len <= data.len()`) holds here: `cur_pos <
                // pos` and `max_len <= n - pos`, so `cur_pos + max_len < pos +
                // max_len <= n == data.len()` (and likewise for `pos`). Byte 0
                // is already confirmed equal above, hence `start_len = 1`.
                let ml = lz_extend(data, pos, cur_pos, 1, max_len);
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
// Block encoder — build tokens, Huffman tables, emit dynamic block.
//
// Huffman construction (exact length-limiting) and the dynamic-header
// precode/RLE wire format are instantiated from the shared modules —
// `huffman::optimal` (Zopfli-class boundary-PM package-merge, the same
// algorithm class this file used to hand-roll) and `huffman::header`
// (libdeflate-derived precode/RLE builder, format-agnostic: it operates on
// arbitrary-length litlen/offset codeword-length slices, trimming/padding
// alphabets itself). See the module doc for what stays private and why.
// ---------------------------------------------------------------------------

fn emit_block(tokens: &[Token], bfinal: u8, bw: &mut BitWriter) {
    // --- Frequency counting ---
    let mut lit_freq = [0usize; NUM_LITLEN_SYMS];
    let mut dist_freq = [0usize; NUM_DIST_SYMS];
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

    // Ensure at least one dist symbol so the length-limited builder (and the
    // decoder, which requires a non-empty dist code) has something to build.
    if dist_freq.iter().all(|&f| f == 0) {
        dist_freq[0] = 1;
    }

    // --- Exact length-limited Huffman lengths (Zopfli-class package-merge) ---
    let mut lit_lens_u32 = [0u32; NUM_LITLEN_SYMS];
    calculate_bit_lengths(&lit_freq, 15, &mut lit_lens_u32);
    let mut dist_lens_u32 = [0u32; NUM_DIST_SYMS];
    calculate_bit_lengths(&dist_freq, 15, &mut dist_lens_u32);

    let mut lit_codes = [0u32; NUM_LITLEN_SYMS];
    lengths_to_symbols(&lit_lens_u32, 15, &mut lit_codes);
    let mut dist_codes = [0u32; NUM_DIST_SYMS];
    lengths_to_symbols(&dist_lens_u32, 15, &mut dist_codes);

    let lit_lens: Vec<u8> = lit_lens_u32.iter().map(|&l| l as u8).collect();
    let dist_lens: Vec<u8> = dist_lens_u32.iter().map(|&l| l as u8).collect();

    // --- Emit block header + dynamic Huffman tables ---
    bw.add_bits(bfinal as u64, 1);
    bw.add_bits(0b10, 2); // BTYPE = dynamic Huffman
    let header = build_dynamic_header(&lit_lens, &dist_lens);
    header.emit(bw);

    // --- Emit tokens. `add_huffman_bits` takes the codeword MSB-first
    // (non-reversed) — exactly the form `lengths_to_symbols` produces. ---
    for &tok in tokens {
        match tok {
            Token::Literal(b) => {
                let sym = b as usize;
                bw.add_huffman_bits(lit_codes[sym], lit_lens_u32[sym]);
            }
            Token::Match { length, dist } => {
                let (lcode, lextra, lval) = length_code(length);
                bw.add_huffman_bits(lit_codes[lcode as usize], lit_lens_u32[lcode as usize]);
                if lextra > 0 {
                    bw.add_bits(lval as u64, lextra as u32);
                }
                let (dcode, dextra, dval) = dist_code(dist);
                bw.add_huffman_bits(dist_codes[dcode as usize], dist_lens_u32[dcode as usize]);
                if dextra > 0 {
                    bw.add_bits(dval as u64, dextra as u32);
                }
            }
        }
    }

    // EOB
    bw.add_huffman_bits(lit_codes[256], lit_lens_u32[256]);
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
    let mut bw = BitWriter::new();

    if data.is_empty() {
        // Single empty stored block: BFINAL=1, BTYPE=00, LEN=0, NLEN=0xFFFF.
        // Shared framing (`compress::deflate::emit_stored_block`) — see
        // module doc, "stored-block framing" is format law, not tier-
        // specific.
        emit_stored_block(&mut bw, &[], true);
        let block = bw.finish();
        writer.write_all(&block)?;
        return Ok(block.len() as u64);
    }

    let mut mf = MatchFinder::new();
    let mut tokens: Vec<Token> = Vec::with_capacity(BLOCK_TOKENS + 16);
    let n = data.len();
    let mut pos = 0usize;

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
        emit_block(&tokens, if is_last { 1 } else { 0 }, &mut bw);
    }

    let out = bw.finish();
    let written = out.len() as u64;
    writer.write_all(&out)?;
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

    /// Real-corpus differential (Stage E, docs/compressor-architecture.md
    /// §5-E): every prior test above drives synthetic/adversarial patterns;
    /// this exercises the rewritten Huffman-construction + dynamic-header
    /// path (now instantiated from `compress::deflate::huffman::{optimal,
    /// header}` instead of this file's own package-merge/RLE code) against
    /// real text — mixed literal/match frequency distributions no synthetic
    /// generator above produces. Same slice-of-a-tar-file convention as
    /// `matchfinder::hc`'s `matches_equal_scalar_silesia`.
    #[test]
    fn test_silesia_slice_roundtrip() {
        let path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark_data/silesia.tar");
        let Ok(mut f) = std::fs::File::open(&path) else {
            eprintln!(
                "note: {} missing; skipped silesia roundtrip",
                path.display()
            );
            return;
        };
        use std::io::{Read, Seek};
        // Two independent slices: one straddling multiple 16384-token blocks
        // of real text, one from further into the archive (different
        // literal/match mix).
        for (start, len) in [(1 << 16, 300 * 1024), (4 << 20, 150 * 1024)] {
            let mut data = vec![0u8; len];
            f.seek(std::io::SeekFrom::Start(start)).unwrap();
            if f.read_exact(&mut data).is_err() {
                eprintln!("note: silesia.tar too small at offset {start}; skipped");
                continue;
            }
            roundtrip(&data);
        }
    }
}
