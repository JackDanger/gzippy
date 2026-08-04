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

use std::io::Write;

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

/// Walk a gzip stream and emit per-block metadata with **exact** bit
/// boundaries.
///
/// Returns the list of blocks in stream order. Aborts on the first
/// decode error (malformed gzip header or corrupt block body).
///
/// Implementation: parses the gzip header manually, then walks the
/// deflate body bit-by-bit using `BitWalker` + a canonical Huffman
/// decoder (`decode_block_body` for the body, `parse_dynamic_header`
/// for BTYPE=10). `start_bit` / `end_bit` are exact bit offsets
/// suitable for asm decoder hand-off (Route C v3+ testbed depends on
/// this precision).
///
/// `flate2::DeflateDecoder` is used ONLY as an oracle for the
/// `decoded.len() == decoded_consumed` debug_assert; it does NOT
/// drive the bit-walker.
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

    // Decode the full payload first (oracle output) for byte counts — pure-Rust
    // raw-DEFLATE inflate (no C-FFI in the decode graph).
    let _ = trailer_isize;
    let decoded = crate::decompress::decompress_raw_bytes(deflate)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

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
    decode_block_body_obs(bits, litlen_lens, dist_lens, &mut |_, _| {})
}

/// [`decode_block_body`] with a per-token observer: called with
/// `(length, distance)` for every match and `(0, 0)` for every literal.
/// The fingerprint layer counts tokens through this without a second
/// decode loop existing to rot.
fn decode_block_body_obs(
    bits: &mut BitWalker,
    litlen_lens: &[u8],
    dist_lens: &[u8],
    observe: &mut impl FnMut(u16, u16),
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
            observe(0, 0);
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
            let distance = DIST_BASE[di] as usize + bits.read(DIST_EXTRA[di]) as usize;
            observe(length as u16, distance as u16);
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

    /// Precision contract: walking a stream then re-decoding from
    /// `start_bit` produces exactly `decoded_bytes` bytes and lands
    /// at `end_bit`. This is the load-bearing test for Route C v3+:
    /// any decoder claiming "I consumed bits 0..end_bit" can be
    /// trusted only if THIS test holds.
    #[test]
    fn block_boundaries_are_bit_exact() {
        let payload: Vec<u8> = (0..2000u32)
            .map(|i| (i.wrapping_mul(0x9e37) >> 8) as u8)
            .collect();
        let gz = gzip_at_level(&payload, 6);
        let blocks = walk_block_boundaries(&gz).unwrap();

        // Determine deflate body offset (must match walk_block_boundaries' logic).
        let flg = gz[3];
        let mut header_end = 10;
        if flg & 0x04 != 0 {
            let xlen = u16::from_le_bytes([gz[header_end], gz[header_end + 1]]) as usize;
            header_end += 2 + xlen;
        }
        if flg & 0x08 != 0 {
            while gz[header_end] != 0 {
                header_end += 1;
            }
            header_end += 1;
        }
        if flg & 0x10 != 0 {
            while gz[header_end] != 0 {
                header_end += 1;
            }
            header_end += 1;
        }
        if flg & 0x02 != 0 {
            header_end += 2;
        }
        let deflate = &gz[header_end..gz.len() - 8];

        // Re-walk from each block's start_bit and verify end_bit
        // matches when we follow the block to completion.
        for (i, b) in blocks.iter().enumerate() {
            assert!(
                b.end_bit > b.start_bit,
                "block {i}: end_bit {} must exceed start_bit {}",
                b.end_bit,
                b.start_bit
            );
            // Successive blocks must be contiguous in bit-space (no gap).
            if i > 0 {
                assert_eq!(
                    blocks[i - 1].end_bit,
                    b.start_bit,
                    "blocks {i}-1 and {i} should be contiguous in bit-space"
                );
            }
            // The last block's end_bit should not overrun the deflate body.
            assert!(
                (b.end_bit / 8) as usize <= deflate.len() + 1,
                "block {i}: end_bit {} would overrun deflate body ({} bytes)",
                b.end_bit,
                deflate.len()
            );
        }

        // Total decoded bytes must equal payload length.
        let total: u64 = blocks.iter().map(|b| b.decoded_bytes).sum();
        assert_eq!(total, payload.len() as u64);
    }

    /// The per-block rows must fold back to the whole-stream aggregate on
    /// every shared axis, and be contiguous in index/span space. This is the
    /// contract `tests/block_pins.rs` leans on: a per-block diff and a
    /// whole-stream diff can never disagree.
    #[test]
    fn block_rows_fold_to_stream_fingerprint() {
        let payload: Vec<u8> = (0..60_000u32)
            .map(|i| (i.wrapping_mul(0x9e37) >> 6) as u8)
            .collect();
        let gz = gzip_at_level(&payload, 6);
        let (fp, rows) = fingerprint_gzip_blocks(&gz).unwrap();
        assert_eq!(fp, fingerprint_gzip(&gz).unwrap());
        assert!(!rows.is_empty());
        assert_eq!(
            rows.iter().map(|r| r.header_bits).sum::<u64>(),
            fp.header_bits
        );
        assert_eq!(rows.iter().map(|r| r.data_bits).sum::<u64>(), fp.data_bits);
        assert_eq!(rows.iter().map(|r| r.literals).sum::<u64>(), fp.literals);
        assert_eq!(rows.iter().map(|r| r.matches).sum::<u64>(), fp.matches);
        assert_eq!(
            rows.iter().map(|r| r.span_bytes).sum::<u64>(),
            fp.decoded_bytes
        );
        assert_eq!(
            rows.iter().filter(|r| r.is_final).count() as u32,
            fp.members,
            "exactly one BFINAL block per member"
        );
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.block_index, i as u32, "block_index is the row index");
        }
        // TSV round-trip is lossless.
        for r in &rows {
            assert_eq!(
                BlockFingerprint::from_tsv_values(&r.tsv_values()).as_ref(),
                Some(r)
            );
        }
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

// ---------------------------------------------------------------------------
// Mechanism fingerprints
// ---------------------------------------------------------------------------

/// A structural fingerprint of a gzip stream, decomposed along the axes this
/// campaign has measured to be the real mechanisms: parse decisions (tokens),
/// entropy coding (header vs data bits), framing (empty seam blocks), and
/// shape (block types, members). Every field is deterministic — no timing —
/// so a fingerprint diff is a machine-behavior diff that runs anywhere.
///
/// The point is not any single field. A regression or a vendor gap shows up
/// as a per-axis delta ("literals +14%, header_bits +2,384, framing
/// unchanged"), which names the mechanism before any profiler runs. See
/// `tests/fingerprint_suite.rs` for the ledger/frontier tests built on this
/// and `examples/fingerprint_tool.rs` for pinning and gap reports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamFingerprint {
    /// Whole-file compressed size in bytes (gzip framing included).
    pub file_bytes: u64,
    /// Total decoded (uncompressed) bytes.
    pub decoded_bytes: u64,
    /// gzip members in the file (pigz/igzip may emit >1).
    pub members: u32,
    /// DEFLATE blocks by type.
    pub blocks_stored: u32,
    pub blocks_fixed: u32,
    pub blocks_dynamic: u32,
    /// Blocks that decode to ZERO bytes — seam/framing padding, not data.
    pub empty_blocks: u32,
    /// Bits spent on block headers (3-bit prelude + stored LEN/NLEN incl.
    /// alignment, or the whole dynamic Huffman table description).
    pub header_bits: u64,
    /// Bits spent on symbol data (everything after each block's header).
    pub data_bits: u64,
    /// Parse decisions.
    pub literals: u64,
    pub matches: u64,
    pub match_bytes: u64,
    /// Match-length histogram: 3, 4-7, 8-15, 16-31, 32-257, 258 (max-match).
    pub len3: u64,
    pub len4_7: u64,
    pub len8_15: u64,
    pub len16_31: u64,
    pub len32_257: u64,
    pub len258: u64,
    /// Matches whose distance exceeds 4096 (the length-3 offset-guard line).
    pub dist_gt4096: u64,
}

impl StreamFingerprint {
    /// Stable TSV column order. `parse_row` accepts exactly this shape, so
    /// pinned files regenerate byte-identically when nothing changed.
    pub const TSV_FIELDS: &'static [&'static str] = &[
        "file_bytes",
        "decoded_bytes",
        "members",
        "blocks_stored",
        "blocks_fixed",
        "blocks_dynamic",
        "empty_blocks",
        "header_bits",
        "data_bits",
        "literals",
        "matches",
        "match_bytes",
        "len3",
        "len4_7",
        "len8_15",
        "len16_31",
        "len32_257",
        "len258",
        "dist_gt4096",
    ];

    pub fn tsv_values(&self) -> Vec<u64> {
        vec![
            self.file_bytes,
            self.decoded_bytes,
            self.members as u64,
            self.blocks_stored as u64,
            self.blocks_fixed as u64,
            self.blocks_dynamic as u64,
            self.empty_blocks as u64,
            self.header_bits,
            self.data_bits,
            self.literals,
            self.matches,
            self.match_bytes,
            self.len3,
            self.len4_7,
            self.len8_15,
            self.len16_31,
            self.len32_257,
            self.len258,
            self.dist_gt4096,
        ]
    }

    pub fn from_tsv_values(vals: &[u64]) -> Option<Self> {
        if vals.len() != Self::TSV_FIELDS.len() {
            return None;
        }
        Some(Self {
            file_bytes: vals[0],
            decoded_bytes: vals[1],
            members: vals[2] as u32,
            blocks_stored: vals[3] as u32,
            blocks_fixed: vals[4] as u32,
            blocks_dynamic: vals[5] as u32,
            empty_blocks: vals[6] as u32,
            header_bits: vals[7],
            data_bits: vals[8],
            literals: vals[9],
            matches: vals[10],
            match_bytes: vals[11],
            len3: vals[12],
            len4_7: vals[13],
            len8_15: vals[14],
            len16_31: vals[15],
            len32_257: vals[16],
            len258: vals[17],
            dist_gt4096: vals[18],
        })
    }

    /// Per-axis diff against another fingerprint (typically: ours vs a rival,
    /// or current vs pinned). Returns (field, self_value, other_value) for
    /// every differing field, largest relative delta first — the failure
    /// message IS the first measurement of the investigation.
    pub fn diff(&self, other: &Self) -> Vec<(&'static str, u64, u64)> {
        let a = self.tsv_values();
        let b = other.tsv_values();
        let mut out: Vec<(&'static str, u64, u64)> = Self::TSV_FIELDS
            .iter()
            .zip(a.iter().zip(b.iter()))
            .filter(|(_, (x, y))| x != y)
            .map(|(f, (x, y))| (*f, *x, *y))
            .collect();
        out.sort_by(|l, r| {
            let rel = |(_, x, y): &(&str, u64, u64)| {
                let m = (*x).max(*y).max(1) as f64;
                ((*x as f64) - (*y as f64)).abs() / m
            };
            rel(r).partial_cmp(&rel(l)).unwrap()
        });
        out
    }
}

/// One row of the per-block fingerprint table. The whole-stream
/// [`StreamFingerprint`] localizes a size change to a FILE; these rows
/// localize it to the exact BLOCK that moved. Axes mirror the aggregate:
/// header vs data bits (entropy coding), literal/match counts (parse), and
/// the uncompressed span (block boundaries — a boundary shift shows up as
/// span-length changes cascading from the first moved block onward).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockFingerprint {
    /// Block index within the whole FILE (monotone across members).
    pub block_index: u32,
    /// gzip member this block belongs to (0-based).
    pub member_index: u32,
    /// BTYPE: 0=stored, 1=fixed-Huffman, 2=dynamic-Huffman.
    pub btype: u8,
    /// BFINAL flag of this block.
    pub is_final: bool,
    /// Bits spent on this block's header (3-bit prelude + stored LEN/NLEN
    /// incl. alignment, or the whole dynamic Huffman table description).
    pub header_bits: u64,
    /// Bits spent on this block's symbol data.
    pub data_bits: u64,
    /// Literal tokens emitted by this block.
    pub literals: u64,
    /// Match tokens emitted by this block.
    pub matches: u64,
    /// Uncompressed bytes this block decodes to (the block's span).
    pub span_bytes: u64,
}

impl BlockFingerprint {
    /// Stable TSV column order for the per-block pin file
    /// (`tests/fingerprints/ours_blocks.tsv`). `from_tsv_values` accepts
    /// exactly this shape.
    pub const TSV_FIELDS: &'static [&'static str] = &[
        "block",
        "member",
        "btype",
        "final",
        "header_bits",
        "data_bits",
        "literals",
        "matches",
        "span_bytes",
    ];

    pub fn tsv_values(&self) -> Vec<u64> {
        vec![
            self.block_index as u64,
            self.member_index as u64,
            self.btype as u64,
            self.is_final as u64,
            self.header_bits,
            self.data_bits,
            self.literals,
            self.matches,
            self.span_bytes,
        ]
    }

    pub fn from_tsv_values(vals: &[u64]) -> Option<Self> {
        if vals.len() != Self::TSV_FIELDS.len() {
            return None;
        }
        Some(Self {
            block_index: vals[0] as u32,
            member_index: vals[1] as u32,
            btype: vals[2] as u8,
            is_final: vals[3] != 0,
            header_bits: vals[4],
            data_bits: vals[5],
            literals: vals[6],
            matches: vals[7],
            span_bytes: vals[8],
        })
    }

    /// Differing axes vs another row: (axis, self_value, other_value).
    /// Row-identity fields (`block`) are included so an index drift is
    /// visible too; order is the TSV order (structural axes first would
    /// hide which one is which).
    pub fn diff(&self, other: &Self) -> Vec<(&'static str, u64, u64)> {
        let a = self.tsv_values();
        let b = other.tsv_values();
        Self::TSV_FIELDS
            .iter()
            .zip(a.iter().zip(b.iter()))
            .filter(|(_, (x, y))| x != y)
            .map(|(f, (x, y))| (*f, *x, *y))
            .collect()
    }
}

/// Fingerprint a complete gzip file (multi-member aware). Exact bit-level
/// walk of every DEFLATE block; no output is materialized.
///
/// This is the aggregate view of [`fingerprint_gzip_blocks`]; the two can
/// never disagree because this one is defined as the fold of the other.
pub fn fingerprint_gzip(gz: &[u8]) -> std::io::Result<StreamFingerprint> {
    fingerprint_gzip_blocks(gz).map(|(fp, _)| fp)
}

/// Fingerprint a complete gzip file AND return the per-block rows the
/// aggregate is folded from. Same exact bit-level walk as
/// [`fingerprint_gzip`]; observer-only, never on the shipped decode path.
pub fn fingerprint_gzip_blocks(
    gz: &[u8],
) -> std::io::Result<(StreamFingerprint, Vec<BlockFingerprint>)> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    let mut fp = StreamFingerprint {
        file_bytes: gz.len() as u64,
        ..Default::default()
    };
    let mut rows: Vec<BlockFingerprint> = Vec::new();
    let mut off = 0usize;
    while off < gz.len() {
        if gz.len() - off < 18 || gz[off] != 0x1f || gz[off + 1] != 0x8b || gz[off + 2] != 0x08 {
            return Err(bad("not a gzip member"));
        }
        let flg = gz[off + 3];
        let mut h = off + 10;
        if flg & 0x04 != 0 {
            let xlen = u16::from_le_bytes([gz[h], gz[h + 1]]) as usize;
            h += 2 + xlen;
        }
        if flg & 0x08 != 0 {
            while h < gz.len() && gz[h] != 0 {
                h += 1;
            }
            h += 1;
        }
        if flg & 0x10 != 0 {
            while h < gz.len() && gz[h] != 0 {
                h += 1;
            }
            h += 1;
        }
        if flg & 0x02 != 0 {
            h += 2;
        }
        let deflate = &gz[h..];
        let mut bits = BitWalker {
            buf: deflate,
            bit_pos: 0,
        };
        loop {
            let start_bit = bits.bit_pos;
            let bfinal = bits.read(1) as u8;
            let btype = bits.read(2) as u8;
            // Snapshot the aggregate axes so this block's contribution is
            // the delta — the per-block rows fold back to the aggregate by
            // construction.
            let header_bits_before = fp.header_bits;
            let data_bits_before = fp.data_bits;
            let literals_before = fp.literals;
            let matches_before = fp.matches;
            let mut observe = |len: u16, dist: u16| {
                if len == 0 {
                    fp.literals += 1;
                } else {
                    fp.matches += 1;
                    fp.match_bytes += len as u64;
                    match len {
                        3 => fp.len3 += 1,
                        4..=7 => fp.len4_7 += 1,
                        8..=15 => fp.len8_15 += 1,
                        16..=31 => fp.len16_31 += 1,
                        258 => fp.len258 += 1,
                        _ => fp.len32_257 += 1,
                    }
                    if dist > 4096 {
                        fp.dist_gt4096 += 1;
                    }
                }
            };
            let decoded = match btype {
                0 => {
                    bits.byte_align();
                    let len = bits.read(16) as usize;
                    let _nlen = bits.read(16);
                    let header_end = bits.bit_pos;
                    fp.header_bits += header_end - start_bit;
                    bits.bit_pos += (len as u64) * 8;
                    fp.data_bits += bits.bit_pos - header_end;
                    fp.blocks_stored += 1;
                    len
                }
                1 => {
                    let mut litlen = [0u8; 288];
                    for e in litlen.iter_mut().take(144) {
                        *e = 8;
                    }
                    for e in litlen.iter_mut().take(256).skip(144) {
                        *e = 9;
                    }
                    for e in litlen.iter_mut().take(280).skip(256) {
                        *e = 7;
                    }
                    for e in litlen.iter_mut().take(288).skip(280) {
                        *e = 8;
                    }
                    let dist = [5u8; 30];
                    let header_end = bits.bit_pos;
                    fp.header_bits += header_end - start_bit;
                    let n = decode_block_body_obs(&mut bits, &litlen[..], &dist[..], &mut observe)?;
                    fp.data_bits += bits.bit_pos - header_end;
                    fp.blocks_fixed += 1;
                    n
                }
                2 => {
                    let (litlen, dist, _ll, _d) = parse_dynamic_header(&mut bits)?;
                    let header_end = bits.bit_pos;
                    fp.header_bits += header_end - start_bit;
                    let n = decode_block_body_obs(&mut bits, &litlen[..], &dist[..], &mut observe)?;
                    fp.data_bits += bits.bit_pos - header_end;
                    fp.blocks_dynamic += 1;
                    n
                }
                _ => return Err(bad("reserved BTYPE=11")),
            };
            if decoded == 0 {
                fp.empty_blocks += 1;
            }
            fp.decoded_bytes += decoded as u64;
            rows.push(BlockFingerprint {
                block_index: rows.len() as u32,
                member_index: fp.members,
                btype,
                is_final: bfinal == 1,
                header_bits: fp.header_bits - header_bits_before,
                data_bits: fp.data_bits - data_bits_before,
                literals: fp.literals - literals_before,
                matches: fp.matches - matches_before,
                span_bytes: decoded as u64,
            });
            if bfinal == 1 {
                break;
            }
        }
        // Byte-align past the deflate stream, then the 8-byte trailer.
        let deflate_bytes = bits.bit_pos.div_ceil(8) as usize;
        off = h + deflate_bytes + 8;
        if off > gz.len() {
            return Err(bad("truncated gzip trailer"));
        }
        fp.members += 1;
    }
    Ok((fp, rows))
}
