//! Foundation primitive — walk a gzip stream block-by-block and emit
//! per-block metadata (start bit, end bit, btype, fingerprint hash for
//! dynamic-Huffman blocks).

#![allow(dead_code)] // public surface used by examples + future AOT v2 build.rs
//!
//! Used by:
//! - `crates/gzippy-inflate/examples/corpus_fingerprint_walker` — collects
//!   fingerprint frequencies to feed AOT codegen (`build.rs`).
//! - Future Route C v3 dynasm-emit testbed — verifies the asm decoder
//!   produces the same output as the Rust reference for every block in
//!   a corpus.
//! - Future `gzippy-inflate` per-block bench harness.
//!
//! ## Why a separate primitive
//!
//! Multiple downstream consumers need "decode WITHOUT writing output"
//! semantics: they care about block boundaries + structural metadata,
//! not the bytes. Today's `decompress()` always materializes output.
//! Routing through this primitive keeps the foundation cheap to use
//! from sub-crate examples, fuzz harnesses, and AOT pipelines without
//! pulling in the full chunk-aware parallel-SM machinery.

use std::io::Read;
use std::io::Write;

use flate2::read::DeflateDecoder;

/// Per-block metadata.
#[derive(Debug, Clone)]
pub struct BlockMeta {
    /// Bit offset (in the gzip stream) of this block's 3-bit header.
    pub start_bit: u64,
    /// Bit offset of the END of this block (= start of next block's
    /// header, or first bit past the deflate stream's last block).
    pub end_bit: u64,
    /// BTYPE: 0=stored, 1=fixed-Huffman, 2=dynamic-Huffman.
    pub btype: u8,
    /// BFINAL: true if this is the last block in the stream.
    pub is_final: bool,
    /// For dynamic-Huffman blocks: FNV-style fingerprint hash of
    /// (litlen_code_lengths | dist_code_lengths). Matches
    /// `gzippy_inflate::aot::fingerprint_hash`.
    pub fingerprint: Option<u64>,
    /// For dynamic-Huffman blocks: count of non-zero litlen code lengths.
    pub litlen_nonzero: Option<u32>,
    /// For dynamic-Huffman blocks: count of non-zero dist code lengths.
    pub dist_nonzero: Option<u32>,
    /// Decoded byte count for this block (for sizing AOT cost models).
    pub decoded_bytes: u64,
}

/// Walk a gzip stream and emit per-block metadata.
///
/// Returns the list of blocks in stream order. Aborts on the first
/// decode error (the gzip header is malformed, or some block body is
/// corrupt).
///
/// Implementation: parses the gzip header manually, then uses flate2's
/// raw `DeflateDecoder` to decode the deflate body. Decoding through
/// flate2 gives correctness for free (vs re-implementing); we walk
/// the bit stream IN PARALLEL with the decoder to capture block
/// boundaries. Since flate2 doesn't expose block boundaries directly,
/// we use a heuristic: decode 1 byte at a time and observe the
/// underlying byte position of the bit-stream cursor; large jumps
/// indicate a new block header. For v1 corpus stats this is accurate
/// enough.
///
/// For v2 (precision): port the parent crate's `parse_dynamic_header_with_lens`
///     + a Huffman decoder that tracks bit position explicitly. Today's
///     flate2-based approach is the cheaper foundation; v2 lands when the
///     AOT pipeline needs exact bit boundaries for per-block asm hand-off.
pub fn walk_block_boundaries(gz: &[u8]) -> std::io::Result<Vec<BlockMeta>> {
    // Skip gzip header.
    if gz.len() < 18 || gz[0] != 0x1f || gz[1] != 0x8b || gz[2] != 0x08 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a gzip stream",
        ));
    }
    let flg = gz[3];
    let mut header_end = 10;
    if flg & 0x04 != 0 {
        let xlen = u16::from_le_bytes([gz[header_end], gz[header_end + 1]]) as usize;
        header_end += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        while header_end < gz.len() && gz[header_end] != 0 {
            header_end += 1;
        }
        header_end += 1;
    }
    if flg & 0x10 != 0 {
        while header_end < gz.len() && gz[header_end] != 0 {
            header_end += 1;
        }
        header_end += 1;
    }
    if flg & 0x02 != 0 {
        header_end += 2;
    }
    let deflate = &gz[header_end..gz.len() - 8]; // 8-byte trailer (CRC32 + ISIZE)
    let trailer_isize = u32::from_le_bytes([
        gz[gz.len() - 4],
        gz[gz.len() - 3],
        gz[gz.len() - 2],
        gz[gz.len() - 1],
    ]) as usize;

    // Decode the full payload first (oracle output) for byte counts.
    let mut decoded = Vec::with_capacity(trailer_isize);
    DeflateDecoder::new(deflate).read_to_end(&mut decoded)?;

    // Walk the bit stream block-by-block with our own minimal parser.
    let mut bits = BitWalker {
        buf: deflate,
        bit_pos: 0,
    };
    let mut blocks = Vec::new();
    let mut decoded_consumed = 0usize;
    loop {
        let start_bit = bits.bit_pos;
        let bfinal = bits.read(1) as u8;
        let btype = bits.read(2) as u8;
        let mut block = BlockMeta {
            start_bit,
            end_bit: 0,
            btype,
            is_final: bfinal == 1,
            fingerprint: None,
            litlen_nonzero: None,
            dist_nonzero: None,
            decoded_bytes: 0,
        };
        match btype {
            0 => {
                // Stored: byte-align, read LEN/NLEN, skip LEN bytes.
                bits.byte_align();
                let len = bits.read(16) as usize;
                let _nlen = bits.read(16);
                bits.bit_pos += (len as u64) * 8;
                block.decoded_bytes = len as u64;
                decoded_consumed += len;
            }
            1 => {
                // Fixed-Huffman: decode through to EOB using the same
                // canonical-Huffman walker as dynamic, with the RFC
                // 1951 fixed code lengths.
                let mut litlen = [0u8; 288];
                for entry in litlen.iter_mut().take(144) {
                    *entry = 8;
                }
                for entry in litlen.iter_mut().take(256).skip(144) {
                    *entry = 9;
                }
                for entry in litlen.iter_mut().take(280).skip(256) {
                    *entry = 7;
                }
                for entry in litlen.iter_mut().take(288).skip(280) {
                    *entry = 8;
                }
                let dist = [5u8; 30];
                let bytes = decode_block_body(&mut bits, &litlen[..], &dist[..])?;
                block.decoded_bytes = bytes as u64;
                decoded_consumed += bytes;
            }
            2 => {
                // Dynamic: parse header, capture fingerprint, decode body.
                let (litlen, dist, ll_nz, d_nz) = parse_dynamic_header(&mut bits)?;
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in litlen.iter() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
                for &b in dist.iter().take(30) {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
                block.fingerprint = Some(h);
                block.litlen_nonzero = Some(ll_nz);
                block.dist_nonzero = Some(d_nz);
                let bytes = decode_block_body(&mut bits, &litlen[..], &dist[..])?;
                block.decoded_bytes = bytes as u64;
                decoded_consumed += bytes;
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "reserved BTYPE=11",
                ));
            }
        }
        block.end_bit = bits.bit_pos;
        let is_final = block.is_final;
        blocks.push(block);
        if is_final {
            break;
        }
    }
    // Sanity: decoded_consumed should match trailer ISIZE (mod 2^32
    // for very large inputs).
    debug_assert_eq!(decoded.len(), decoded_consumed);
    let _ = decoded; // keep alive
    Ok(blocks)
}

struct BitWalker<'a> {
    buf: &'a [u8],
    bit_pos: u64,
}

impl BitWalker<'_> {
    fn read(&mut self, n: u8) -> u32 {
        let byte = (self.bit_pos / 8) as usize;
        let off = (self.bit_pos % 8) as u32;
        let mut buf: u64 = 0;
        for i in 0..6 {
            if byte + i < self.buf.len() {
                buf |= (self.buf[byte + i] as u64) << (i * 8);
            }
        }
        let v = ((buf >> off) & ((1u64 << n) - 1)) as u32;
        self.bit_pos += n as u64;
        v
    }

    fn byte_align(&mut self) {
        self.bit_pos = self.bit_pos.div_ceil(8) * 8;
    }
}

/// Decode a deflate block body (already past the 3-bit header) using
/// caller-supplied litlen + dist code lengths. Returns the decoded
/// byte count. Does NOT materialize output bytes (we only need block
/// boundaries + sizes for AOT/corpus stats).
fn decode_block_body(
    bits: &mut BitWalker,
    litlen_lens: &[u8],
    dist_lens: &[u8],
) -> std::io::Result<usize> {
    let lit_lookup = build_canonical_lookup(litlen_lens, 15)?;
    let dist_lookup = build_canonical_lookup(dist_lens, 15)?;
    const LENGTH_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LENGTH_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    let mut out_count = 0usize;
    loop {
        let (sym, len) = lookup_symbol(bits, &lit_lookup, 15)?;
        bits.bit_pos += len as u64;
        if sym < 256 {
            out_count += 1;
        } else if sym == 256 {
            return Ok(out_count);
        } else {
            let li = (sym - 257) as usize;
            let length = LENGTH_BASE[li] as usize + bits.read(LENGTH_EXTRA[li]) as usize;
            let (dsym, dlen) = lookup_symbol(bits, &dist_lookup, 15)?;
            bits.bit_pos += dlen as u64;
            let di = dsym as usize;
            if di >= 30 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid distance symbol {di}"),
                ));
            }
            let _distance = DIST_BASE[di] as usize + bits.read(DIST_EXTRA[di]) as usize;
            out_count += length;
        }
    }
}

/// Canonical Huffman lookup: 2^max_bits entries, each (symbol, length).
/// length == 0 means no code at this key.
fn build_canonical_lookup(code_lengths: &[u8], max_bits: u8) -> std::io::Result<Vec<(u16, u8)>> {
    let table_size = 1usize << max_bits;
    let mut entries = vec![(0u16, 0u8); table_size];
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
            entries[k] = (symbol as u16, len);
            k += stride;
        }
    }
    Ok(entries)
}

fn lookup_symbol(
    bits: &mut BitWalker,
    lookup: &[(u16, u8)],
    max_bits: u8,
) -> std::io::Result<(u16, u8)> {
    let mask = (1u32 << max_bits) - 1;
    // Read max_bits but don't consume — caller advances by `len`.
    let byte = (bits.bit_pos / 8) as usize;
    let off = (bits.bit_pos % 8) as u32;
    let mut buf: u64 = 0;
    for i in 0..6 {
        if byte + i < bits.buf.len() {
            buf |= (bits.buf[byte + i] as u64) << (i * 8);
        }
    }
    let key = ((buf >> off) as u32 & mask) as usize;
    let (sym, len) = lookup[key];
    if len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("no code at key 0x{key:04x}"),
        ));
    }
    Ok((sym, len))
}

fn parse_dynamic_header(bits: &mut BitWalker) -> std::io::Result<([u8; 288], [u8; 32], u32, u32)> {
    let hlit = bits.read(5) as usize + 257;
    let hdist = bits.read(5) as usize + 1;
    let hclen = bits.read(4) as usize + 4;
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
        clcl[o] = bits.read(3) as u8;
    }
    let cl_lookup = build_canonical_lookup(&clcl, 7)?;
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < all_lens.len() {
        let (sym, len) = lookup_symbol(bits, &cl_lookup, 7)?;
        bits.bit_pos += len as u64;
        match sym {
            0..=15 => {
                all_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "repeat with no prev",
                    ));
                }
                let count = bits.read(2) as usize + 3;
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
                let count = bits.read(3) as usize + 3;
                if i + count > all_lens.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "zero-3 overflow",
                    ));
                }
                i += count;
            }
            18 => {
                let count = bits.read(7) as usize + 11;
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
                    format!("bad cl sym {sym}"),
                ));
            }
        }
    }
    let mut litlen = [0u8; 288];
    let mut dist = [0u8; 32];
    litlen[..hlit].copy_from_slice(&all_lens[..hlit]);
    dist[..hdist].copy_from_slice(&all_lens[hlit..]);
    let ll_nz = litlen.iter().filter(|&&b| b > 0).count() as u32;
    let d_nz = dist.iter().take(30).filter(|&&b| b > 0).count() as u32;
    Ok((litlen, dist, ll_nz, d_nz))
}

fn reverse_bits(mut code: u32, n: u8) -> u32 {
    let mut rev = 0u32;
    for _ in 0..n {
        rev = (rev << 1) | (code & 1);
        code >>= 1;
    }
    rev
}

/// Convenience: produce a JSON line per block. Used by the AOT corpus
/// walker to spool to disk.
pub fn write_blocks_jsonl<W: Write>(w: &mut W, blocks: &[BlockMeta]) -> std::io::Result<()> {
    for b in blocks {
        let fp = b
            .fingerprint
            .map(|h| format!("\"{h:#018x}\""))
            .unwrap_or_else(|| "null".to_string());
        let ll = b
            .litlen_nonzero
            .map(|n| n.to_string())
            .unwrap_or_else(|| "null".to_string());
        let d = b
            .dist_nonzero
            .map(|n| n.to_string())
            .unwrap_or_else(|| "null".to_string());
        writeln!(
            w,
            r#"{{"start_bit":{},"end_bit":{},"btype":{},"is_final":{},"fingerprint":{},"litlen_nonzero":{},"dist_nonzero":{},"decoded_bytes":{}}}"#,
            b.start_bit, b.end_bit, b.btype, b.is_final, fp, ll, d, b.decoded_bytes
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gzip_at_level(payload: &[u8], level: u32) -> Vec<u8> {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        e.write_all(payload).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn walk_empty_payload() {
        let gz = gzip_at_level(b"", 6);
        let blocks = walk_block_boundaries(&gz).unwrap();
        assert!(!blocks.is_empty(), "even empty gzip has at least 1 block");
        let total: u64 = blocks.iter().map(|b| b.decoded_bytes).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn walk_text_round_trip() {
        let payload = b"the quick brown fox jumps over the lazy dog";
        let gz = gzip_at_level(payload, 6);
        let blocks = walk_block_boundaries(&gz).unwrap();
        let total: u64 = blocks.iter().map(|b| b.decoded_bytes).sum();
        assert_eq!(total, payload.len() as u64);
    }

    #[test]
    fn walk_repetitive_finds_dynamic_block_with_fingerprint() {
        let payload = vec![b'A'; 10000];
        let gz = gzip_at_level(&payload, 9);
        let blocks = walk_block_boundaries(&gz).unwrap();
        // Level 9 → expect dynamic Huffman.
        let dyn_blocks: Vec<_> = blocks
            .iter()
            .filter(|b| b.btype == 2 && b.fingerprint.is_some())
            .collect();
        assert!(
            !dyn_blocks.is_empty(),
            "level-9 repetitive payload should produce ≥1 dynamic block"
        );
        // Fingerprint is reproducible.
        let h1 = dyn_blocks[0].fingerprint.unwrap();
        let gz2 = gzip_at_level(&payload, 9);
        let blocks2 = walk_block_boundaries(&gz2).unwrap();
        let h2 = blocks2
            .iter()
            .find(|b| b.btype == 2)
            .unwrap()
            .fingerprint
            .unwrap();
        assert_eq!(h1, h2, "fingerprint is deterministic");
    }

    #[test]
    fn jsonl_writer_emits_lines() {
        let gz = gzip_at_level(b"hello world", 6);
        let blocks = walk_block_boundaries(&gz).unwrap();
        let mut buf = Vec::new();
        write_blocks_jsonl(&mut buf, &blocks).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let line_count = s.lines().count();
        assert_eq!(line_count, blocks.len());
        // Each line has the expected fields.
        for line in s.lines() {
            assert!(line.contains("\"start_bit\""));
            assert!(line.contains("\"btype\""));
            assert!(line.contains("\"decoded_bytes\""));
        }
    }
}
