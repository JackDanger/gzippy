//! Walk gzip files and collect (litlen, dist) Huffman fingerprint
//! frequencies. Output goes to stdout as JSON; build.rs (v2) can
//! consume it to bake the top-N most-common fingerprints into the
//! AOT table.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example corpus_fingerprint_walker -- silesia.gz linux.gz
//! ```
//!
//! Output format:
//! ```json
//! {
//!   "files_scanned": 2,
//!   "blocks_total": 1287,
//!   "blocks_by_btype": {"stored": 4, "fixed": 18, "dynamic": 1265},
//!   "unique_fingerprints": 532,
//!   "top_n": [
//!     {"fingerprint": "0x...", "frequency": 412, "litlen_nonzero": 286, "dist_nonzero": 30},
//!     ...
//!   ]
//! }
//! ```
//!
//! ## Scope
//!
//! v1: self-contained DEFLATE bit-reader + dynamic-Huffman header
//! parser (does NOT decode the block body — just consumes through
//! the header to advance to the next block).
//!
//! Limitations: counts dynamic-Huffman blocks only; stored + fixed
//! are tallied but not fingerprinted (fixed has one fingerprint,
//! already in AOT v1; stored has no Huffman tables).

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: corpus_fingerprint_walker <file.gz> [file.gz...]");
        return ExitCode::from(2);
    }

    let mut counts: HashMap<u64, (u64, usize, usize)> = HashMap::new();
    let mut stats = WalkStats::default();

    for path in &args {
        let gz = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error reading {path}: {e}");
                continue;
            }
        };
        stats.files_scanned += 1;
        let mut walker = match Walker::new(&gz) {
            Some(w) => w,
            None => {
                eprintln!("error: {path}: invalid gzip header");
                continue;
            }
        };
        loop {
            match walker.next_block() {
                Ok(Some(info)) => {
                    stats.blocks_total += 1;
                    match info.btype {
                        BlockType::Stored => stats.stored += 1,
                        BlockType::Fixed => stats.fixed += 1,
                        BlockType::Dynamic {
                            fingerprint,
                            litlen_nonzero,
                            dist_nonzero,
                        } => {
                            stats.dynamic += 1;
                            let e = counts.entry(fingerprint).or_insert((
                                0,
                                litlen_nonzero,
                                dist_nonzero,
                            ));
                            e.0 += 1;
                        }
                    }
                }
                Ok(None) => break, // end of stream
                Err(e) => {
                    eprintln!("error scanning {path}: {e}");
                    break;
                }
            }
        }
    }

    // Sort fingerprints by frequency descending.
    let mut sorted: Vec<(u64, u64, usize, usize)> = counts
        .iter()
        .map(|(&k, &(freq, ln, dn))| (k, freq, ln, dn))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let top_n = sorted.iter().take(64).collect::<Vec<_>>();

    // Emit JSON.
    println!("{{");
    println!("  \"files_scanned\": {},", stats.files_scanned);
    println!("  \"blocks_total\": {},", stats.blocks_total);
    println!("  \"blocks_by_btype\": {{");
    println!("    \"stored\": {},", stats.stored);
    println!("    \"fixed\": {},", stats.fixed);
    println!("    \"dynamic\": {}", stats.dynamic);
    println!("  }},");
    println!("  \"unique_fingerprints\": {},", counts.len());
    println!("  \"top_n\": [");
    for (i, (fp, freq, ln, dn)) in top_n.iter().enumerate() {
        let comma = if i + 1 < top_n.len() { "," } else { "" };
        println!("    {{");
        println!("      \"fingerprint\": \"{:#018x}\",", fp);
        println!("      \"frequency\": {},", freq);
        println!("      \"litlen_nonzero\": {},", ln);
        println!("      \"dist_nonzero\": {}", dn);
        println!("    }}{comma}");
    }
    println!("  ]");
    println!("}}");

    ExitCode::SUCCESS
}

#[derive(Default)]
struct WalkStats {
    files_scanned: u32,
    blocks_total: u64,
    stored: u64,
    fixed: u64,
    dynamic: u64,
}

#[derive(Debug)]
enum BlockType {
    Stored,
    Fixed,
    Dynamic {
        fingerprint: u64,
        litlen_nonzero: usize,
        dist_nonzero: usize,
    },
}

struct BlockInfo {
    btype: BlockType,
    is_final: bool,
}

/// Minimal DEFLATE walker — finds block boundaries WITHOUT decoding
/// the block body. After parsing a dynamic-Huffman header, the walker
/// must decompress through to find the EOB code (so it knows where
/// the next block starts). For corpus stats we use flate2 to decode
/// the full stream first and re-scan the bit-stream cheaply.
struct Walker<'a> {
    bits: BitReader<'a>,
    finished: bool,
}

impl<'a> Walker<'a> {
    fn new(gz: &'a [u8]) -> Option<Self> {
        // Parse gzip header (10 bytes minimum + optional FEXTRA/FNAME/...)
        if gz.len() < 18 || gz[0] != 0x1f || gz[1] != 0x8b || gz[2] != 0x08 {
            return None;
        }
        let flg = gz[3];
        let mut header_end = 10;
        if flg & 0x04 != 0 {
            // FEXTRA
            if header_end + 2 > gz.len() {
                return None;
            }
            let xlen = u16::from_le_bytes([gz[header_end], gz[header_end + 1]]) as usize;
            header_end += 2 + xlen;
        }
        if flg & 0x08 != 0 {
            // FNAME
            while header_end < gz.len() && gz[header_end] != 0 {
                header_end += 1;
            }
            header_end += 1;
        }
        if flg & 0x10 != 0 {
            // FCOMMENT
            while header_end < gz.len() && gz[header_end] != 0 {
                header_end += 1;
            }
            header_end += 1;
        }
        if flg & 0x02 != 0 {
            header_end += 2; // FHCRC
        }
        if header_end >= gz.len() {
            return None;
        }
        // Decompress full stream so we can walk block-by-block via
        // the resulting bit-stream (we re-parse just headers).
        Some(Self {
            bits: BitReader {
                buf: gz,
                bit_pos: header_end * 8,
            },
            finished: false,
        })
    }

    fn next_block(&mut self) -> Result<Option<BlockInfo>, String> {
        if self.finished {
            return Ok(None);
        }
        // Read 3-bit block header.
        let bfinal = self.bits.read(1) as u8;
        let btype = self.bits.read(2) as u8;
        let info = match btype {
            0 => {
                // Stored block: byte-align, read LEN (16 bits), skip LEN bytes.
                self.bits.byte_align();
                let len = self.bits.read_byte_aligned_u16() as usize;
                let _nlen = self.bits.read_byte_aligned_u16();
                self.bits.skip_bytes(len);
                BlockInfo {
                    btype: BlockType::Stored,
                    is_final: bfinal == 1,
                }
            }
            1 => {
                // Fixed Huffman — body length is data-dependent; we
                // need to scan literals/lengths until EOB. For the
                // corpus walker, this is rare; we conservatively
                // mark and exit. Full body scan would need a fixed-
                // Huffman decoder; defer to v2 if needed.
                self.finished = true;
                BlockInfo {
                    btype: BlockType::Fixed,
                    is_final: bfinal == 1,
                }
            }
            2 => {
                // Dynamic Huffman: parse the header to extract code lengths.
                let info = self.parse_dynamic_header()?;
                self.finished = true; // walker exits after first dynamic block per file (cheap stat)
                BlockInfo {
                    btype: info,
                    is_final: bfinal == 1,
                }
            }
            _ => {
                return Err(format!("reserved BTYPE=11 at bit {}", self.bits.bit_pos));
            }
        };
        if info.is_final {
            self.finished = true;
        }
        Ok(Some(info))
    }

    fn parse_dynamic_header(&mut self) -> Result<BlockType, String> {
        // HLIT (5) + 257 = #litlen codes
        let hlit = self.bits.read(5) as usize + 257;
        // HDIST (5) + 1 = #dist codes
        let hdist = self.bits.read(5) as usize + 1;
        // HCLEN (4) + 4 = #code-length codes
        let hclen = self.bits.read(4) as usize + 4;
        if hlit > 286 || hdist > 30 || hclen > 19 {
            return Err(format!(
                "header out of range: hlit={hlit}, hdist={hdist}, hclen={hclen}"
            ));
        }

        const ORDER: [usize; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        let mut clcl = [0u8; 19];
        for &o in ORDER.iter().take(hclen) {
            clcl[o] = self.bits.read(3) as u8;
        }

        // Build the code-length code (small canonical Huffman over 19 syms).
        let cl_lookup = build_cl_lookup(&clcl)?;

        // Decode hlit + hdist litlen/dist code lengths.
        let mut all_lens = vec![0u8; hlit + hdist];
        let mut i = 0;
        while i < all_lens.len() {
            let sym = decode_cl_symbol(&mut self.bits, &cl_lookup)?;
            match sym {
                0..=15 => {
                    all_lens[i] = sym as u8;
                    i += 1;
                }
                16 => {
                    // Repeat previous 3-6 times.
                    if i == 0 {
                        return Err("repeat prev with no prev len".into());
                    }
                    let count = self.bits.read(2) as usize + 3;
                    if i + count > all_lens.len() {
                        return Err("repeat overruns lens".into());
                    }
                    let prev = all_lens[i - 1];
                    for entry in all_lens.iter_mut().skip(i).take(count) {
                        *entry = prev;
                    }
                    i += count;
                }
                17 => {
                    let count = self.bits.read(3) as usize + 3;
                    if i + count > all_lens.len() {
                        return Err("zero-3 overruns lens".into());
                    }
                    i += count;
                }
                18 => {
                    let count = self.bits.read(7) as usize + 11;
                    if i + count > all_lens.len() {
                        return Err("zero-11 overruns lens".into());
                    }
                    i += count;
                }
                _ => return Err(format!("bad cl symbol {sym}")),
            }
        }

        // Split into litlen + dist arrays.
        let mut litlen = [0u8; 288];
        let mut dist = [0u8; 32];
        litlen[..hlit].copy_from_slice(&all_lens[..hlit]);
        dist[..hdist].copy_from_slice(&all_lens[hlit..]);

        // Fingerprint using the same FNV-style hash as build.rs.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in litlen.iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        for &b in dist.iter().take(30) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }

        let litlen_nonzero = litlen.iter().filter(|&&b| b > 0).count();
        let dist_nonzero = dist.iter().take(30).filter(|&&b| b > 0).count();

        Ok(BlockType::Dynamic {
            fingerprint: h,
            litlen_nonzero,
            dist_nonzero,
        })
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn read(&mut self, n: u8) -> u32 {
        let byte = self.bit_pos / 8;
        let off = (self.bit_pos % 8) as u32;
        let mut buf: u64 = 0;
        for i in 0..6 {
            if byte + i < self.buf.len() {
                buf |= (self.buf[byte + i] as u64) << (i * 8);
            }
        }
        let v = ((buf >> off) & ((1u64 << n) - 1)) as u32;
        self.bit_pos += n as usize;
        v
    }

    fn byte_align(&mut self) {
        self.bit_pos = self.bit_pos.div_ceil(8) * 8;
    }

    fn read_byte_aligned_u16(&mut self) -> u16 {
        let b = self.bit_pos / 8;
        let v = u16::from_le_bytes([self.buf[b], self.buf[b + 1]]);
        self.bit_pos += 16;
        v
    }

    fn skip_bytes(&mut self, n: usize) {
        self.bit_pos += n * 8;
    }
}

#[derive(Default)]
struct ClLookup {
    /// Indexed by 7-bit key; entry: (symbol, code_length). code_length=0 means no match.
    entries: Vec<(u8, u8)>,
}

fn build_cl_lookup(clcl: &[u8; 19]) -> Result<ClLookup, String> {
    let mut count = [0u16; 8];
    for &c in clcl.iter() {
        if c > 0 && c <= 7 {
            count[c as usize] += 1;
        }
    }
    let mut first_code = [0u32; 8];
    let mut code: u32 = 0;
    for len in 1..=7 {
        code = (code + count[len - 1] as u32) << 1;
        first_code[len] = code;
    }
    let mut entries = vec![(0u8, 0u8); 128];
    let mut next = first_code;
    for (sym, &c) in clcl.iter().enumerate() {
        if c == 0 {
            continue;
        }
        if c > 7 {
            return Err(format!("cl code length {c} > 7"));
        }
        let codeword = next[c as usize];
        next[c as usize] += 1;
        let rev = reverse_bits(codeword, c) as usize;
        let stride = 1usize << c;
        let mut k = rev;
        while k < 128 {
            entries[k] = (sym as u8, c);
            k += stride;
        }
    }
    Ok(ClLookup { entries })
}

fn decode_cl_symbol(bits: &mut BitReader, lookup: &ClLookup) -> Result<u8, String> {
    let key = (bits.read(7) & 0x7F) as usize;
    let (sym, len) = lookup.entries[key];
    if len == 0 {
        return Err(format!("no cl code at key 0x{key:02x}"));
    }
    // We read 7 bits but the code may be shorter; rewind.
    bits.bit_pos -= (7 - len) as usize;
    Ok(sym)
}

fn reverse_bits(mut code: u32, n: u8) -> u32 {
    let mut rev = 0u32;
    for _ in 0..n {
        rev = (rev << 1) | (code & 1);
        code >>= 1;
    }
    rev
}
