//! Phase 1.1 corpus extractor (see plans/inner-loop-execution.md).
//!
//! Walks a gzip file (silesia or similar), decodes each DEFLATE block via
//! `gzippy::decompress::parallel::deflate_block::Block`, and dumps each
//! dynamic-Huffman block to its own corpus file.
//!
//! Per-block corpus file format (little-endian binary):
//!
//! ```text
//! Offset  Size    Field
//!  0      8       magic = b"GZBLK01\0"
//!  8      4       block_idx (u32)
//! 12      4       btype (u32; 2 = dynamic Huffman, others rejected at extract)
//! 16      8       bit_offset_in_compressed (u64; bit-offset into `compressed`)
//! 24      8       compressed_bit_len (u64; header + body length in bits)
//! 32      8       compressed_byte_len (u64; covers the bit range)
//! 40      4       decoded_len (u32; body output length in bytes)
//! 44      4       predecessor_len (u32; 0 or 32768)
//! 48      2       literal_code_count (u16; HLIT + 257)
//! 50      2       distance_code_count (u16; HDIST + 1)
//! 52      1       max_litlen_code_len (u8; diagnostic for LUT pressure)
//! 53      1       max_dist_code_len (u8)
//! 54      2       marker_count_during_decode (u16; 0 by construction)
//! 56      4       crc32_of_decoded (u32)
//! 60      4       reserved (u32; 0)
//! 64      ...     predecessor[predecessor_len] (u8)
//! +...    ...     compressed[compressed_byte_len] (u8)
//! +...    ...     decoded[decoded_len] (u8)
//! +...    ...     literal_cl[literal_code_count] (u8)
//! +...    ...     distance_cl[distance_code_count] (u8)
//! ```
//!
//! Output:
//! - `corpus/silesia_blocks/<NNN>.bin` per chosen block.
//! - `corpus/silesia_blocks/INDEX.json` human-readable summary.
//! - `corpus/silesia_blocks/README.md` selection-criteria doc.

use gzippy::decompress::inflate::consume_first_decode::Bits;
use gzippy::decompress::parallel::deflate_block::{Block, CompressionType};
use std::fs;
use std::io::Read;
use std::path::PathBuf;

const MAX_WINDOW_SIZE: usize = 32_768;
const CORPUS_VERSION: &[u8; 8] = b"GZBLK01\0";

#[derive(Debug, Clone)]
struct ExtractedBlock {
    block_idx: u32,
    bit_offset_in_compressed: u64,
    compressed_bit_len: u64,
    compressed_bytes: Vec<u8>,
    predecessor: Vec<u8>,
    decoded: Vec<u8>,
    literal_cl: Vec<u8>,
    distance_cl: Vec<u8>,
    max_litlen_code_len: u8,
    max_dist_code_len: u8,
}

impl ExtractedBlock {
    fn write_to(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(
            64 + self.predecessor.len()
                + self.compressed_bytes.len()
                + self.decoded.len()
                + self.literal_cl.len()
                + self.distance_cl.len(),
        );
        buf.extend_from_slice(CORPUS_VERSION);
        buf.extend_from_slice(&self.block_idx.to_le_bytes());
        // btype = 2 (dynamic) by construction (filter applied before write)
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&self.bit_offset_in_compressed.to_le_bytes());
        buf.extend_from_slice(&self.compressed_bit_len.to_le_bytes());
        buf.extend_from_slice(&(self.compressed_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.decoded.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.predecessor.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.literal_cl.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(self.distance_cl.len() as u16).to_le_bytes());
        buf.push(self.max_litlen_code_len);
        buf.push(self.max_dist_code_len);
        // marker_count_during_decode: 0 by construction (set_initial_window
        // for non-first blocks flips contains_marker_bytes=false; first
        // block is filtered out below if predecessor would have been empty
        // since first-of-member blocks aren't representative).
        buf.extend_from_slice(&0u16.to_le_bytes());
        let crc32 = crc32fast_of(&self.decoded);
        buf.extend_from_slice(&crc32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&self.predecessor);
        buf.extend_from_slice(&self.compressed_bytes);
        buf.extend_from_slice(&self.decoded);
        buf.extend_from_slice(&self.literal_cl);
        buf.extend_from_slice(&self.distance_cl);
        fs::write(path, buf)
    }
}

/// CRC32 of decoded output for corpus-file integrity check.
fn crc32fast_of(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

fn skip_gzip_header(input: &[u8]) -> std::io::Result<usize> {
    if input.len() < 10 || input[0] != 0x1f || input[1] != 0x8b {
        return Err(std::io::Error::other("not gzip"));
    }
    if input[2] != 8 {
        return Err(std::io::Error::other("compression method != deflate"));
    }
    let flg = input[3];
    let mut idx = 10usize;
    if flg & 0x04 != 0 {
        // FEXTRA
        if input.len() < idx + 2 {
            return Err(std::io::Error::other("truncated FEXTRA"));
        }
        let xlen = u16::from_le_bytes([input[idx], input[idx + 1]]) as usize;
        idx += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        // FNAME (zero-terminated)
        while idx < input.len() && input[idx] != 0 {
            idx += 1;
        }
        idx += 1;
    }
    if flg & 0x10 != 0 {
        // FCOMMENT
        while idx < input.len() && input[idx] != 0 {
            idx += 1;
        }
        idx += 1;
    }
    if flg & 0x02 != 0 {
        // FHCRC: 2 bytes
        idx += 2;
    }
    Ok(idx)
}

fn extract_all(deflate_bytes: &[u8]) -> Result<Vec<ExtractedBlock>, String> {
    let mut bits = Bits::new(deflate_bytes);
    let mut running_decoded: Vec<u8> = Vec::with_capacity(256 * 1024 * 1024);
    let mut block_idx: u32 = 0;
    let mut extracted = Vec::new();

    loop {
        // Predecessor window: last 32 KiB of running_decoded.
        let pred_len = running_decoded.len().min(MAX_WINDOW_SIZE);
        let predecessor: Vec<u8> =
            running_decoded[running_decoded.len() - pred_len..].to_vec();

        // Per Opus design call: fresh Block + set_initial_window per block.
        // This seeds the u16 ring with u8 bytes and flips
        // contains_marker_bytes=false so mid-block back-refs into the
        // predecessor resolve to clean literals (no marker emission).
        let mut block = Block::new();
        let mut out_buf: Vec<u16> = Vec::new();
        if !predecessor.is_empty() {
            block
                .set_initial_window(&mut out_buf, &predecessor)
                .map_err(|e| format!("set_initial_window block #{block_idx}: {e:?}"))?;
            // Discard the seeded-window output; we only want the body bytes.
            out_buf.clear();
        }
        // set_initial_window mutated decoded_bytes; reset its frame counter
        // so the body decode starts clean. Block::reset would do this, but
        // it also resets the seeded ring. We work around by using a separate
        // body-output buffer:
        let mut body_buf: Vec<u16> = Vec::new();

        let start_bit = bits.bit_position();
        if start_bit >= deflate_bytes.len() * 8 {
            break;
        }

        // read_header reads BFINAL + BTYPE + per-type sub-header. After it
        // returns, the block is ready for read() to decode the body.
        block
            .read_header(&mut bits, false)
            .map_err(|e| format!("read_header block #{block_idx}: {e:?}"))?;

        let btype = block.compression_type();
        let is_last = block.is_last_block();

        // Drive body decode until EOB.
        while !block.eob() {
            block
                .read(&mut bits, &mut body_buf, usize::MAX)
                .map_err(|e| format!("read body block #{block_idx}: {e:?}"))?;
        }

        let end_bit = bits.bit_position();
        let compressed_bit_len = (end_bit - start_bit) as u64;

        // Cast u16 → u8 (all values < 256 by set_initial_window contract
        // when predecessor non-empty; for first block of member with empty
        // predecessor, we just skip below since markers may fire).
        let decoded: Vec<u8> = body_buf
            .iter()
            .map(|&v| {
                debug_assert!(v < 256, "marker leak in block {block_idx}");
                v as u8
            })
            .collect();

        // Slice the input bytes covering this block's bit range.
        let byte_start = start_bit / 8;
        let byte_end = (end_bit + 7) / 8;
        let bit_offset_in_compressed = (start_bit % 8) as u64;
        let compressed_bytes = deflate_bytes[byte_start..byte_end].to_vec();

        // Append decoded body to running window.
        running_decoded.extend_from_slice(&decoded);

        // Filter: only dynamic-Huffman blocks with non-empty predecessor
        // are corpus candidates. Stored / fixed / first-of-member blocks
        // are skipped (representativeness reasons; see plan Phase 1.1).
        if btype == CompressionType::DynamicHuffman && !predecessor.is_empty() {
            // Snapshot Huffman code lengths from the just-decoded header.
            // `literal_cl` packs both litlen and distance code lengths:
            // litlen at [0..literal_code_count], distance at
            // [literal_code_count..literal_code_count+distance_code_count].
            let lit_count = block.literal_code_count;
            let dist_count = block.distance_code_count;
            let literal_cl = block.literal_cl[..lit_count].to_vec();
            let distance_cl =
                block.literal_cl[lit_count..lit_count + dist_count].to_vec();
            let max_litlen = *literal_cl.iter().max().unwrap_or(&0);
            let max_dist = *distance_cl.iter().max().unwrap_or(&0);

            extracted.push(ExtractedBlock {
                block_idx,
                bit_offset_in_compressed,
                compressed_bit_len,
                compressed_bytes,
                predecessor,
                decoded,
                literal_cl,
                distance_cl,
                max_litlen_code_len: max_litlen,
                max_dist_code_len: max_dist,
            });
        }

        block_idx += 1;
        if is_last {
            break;
        }
    }

    Ok(extracted)
}

fn stratified_pick(all: Vec<ExtractedBlock>, target: usize) -> Vec<ExtractedBlock> {
    // Per Opus's design call: stratify by max_litlen_code_len.
    //   shallow (≤9):  proxy for LUT-friendly
    //   typical (10-12): the bulk
    //   deep (≥13):    LUT-stressing
    // Plus a tail of "back-ref-heavy" blocks (decoded/compressed-bits ratio).
    let mut shallow: Vec<_> = all
        .iter()
        .filter(|b| b.max_litlen_code_len <= 9)
        .cloned()
        .collect();
    let mut typical: Vec<_> = all
        .iter()
        .filter(|b| (10..=12).contains(&b.max_litlen_code_len))
        .cloned()
        .collect();
    let mut deep: Vec<_> = all
        .iter()
        .filter(|b| b.max_litlen_code_len >= 13)
        .cloned()
        .collect();
    let mut backref_heavy: Vec<_> = {
        let mut v: Vec<_> = all.iter().cloned().collect();
        v.sort_by(|a, b| {
            let ra = a.decoded.len() as f64 / a.compressed_bit_len as f64;
            let rb = b.decoded.len() as f64 / b.compressed_bit_len as f64;
            rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    };

    let want_shallow = target / 4;
    let want_typical = target / 2;
    let want_deep = target / 4;
    let want_backref = target.saturating_sub(want_shallow + want_typical + want_deep);

    let mut picked = Vec::new();
    shallow.truncate(want_shallow);
    typical.truncate(want_typical);
    deep.truncate(want_deep);
    backref_heavy.truncate(want_backref);
    picked.extend(shallow);
    picked.extend(typical);
    picked.extend(deep);
    picked.extend(backref_heavy);
    // Dedup by block_idx (a back-ref-heavy block might also be in a
    // litlen bucket).
    let mut seen = std::collections::HashSet::new();
    picked.retain(|b| seen.insert(b.block_idx));
    picked.truncate(target);
    picked
}

fn verify_extraction(block: &ExtractedBlock) -> Result<(), String> {
    // Re-decode the corpus block via the same set_initial_window path and
    // assert byte equality with the snapshot. Catches extraction bugs.
    let mut decoder = Block::new();
    let mut out_buf: Vec<u16> = Vec::new();
    decoder
        .set_initial_window(&mut out_buf, &block.predecessor)
        .map_err(|e| format!("verify set_initial_window block #{}: {e:?}", block.block_idx))?;
    out_buf.clear();
    let mut bits = Bits::at_bit_offset(
        &block.compressed_bytes,
        block.bit_offset_in_compressed as usize,
    );
    let mut body_buf: Vec<u16> = Vec::new();
    decoder
        .read_header(&mut bits, false)
        .map_err(|e| format!("verify read_header block #{}: {e:?}", block.block_idx))?;
    while !decoder.eob() {
        decoder
            .read(&mut bits, &mut body_buf, usize::MAX)
            .map_err(|e| format!("verify read body block #{}: {e:?}", block.block_idx))?;
    }
    let re_decoded: Vec<u8> = body_buf
        .iter()
        .map(|&v| {
            if v >= 256 {
                panic!("verify: marker leak in re-decode of block {}", block.block_idx);
            }
            v as u8
        })
        .collect();
    if re_decoded != block.decoded {
        return Err(format!(
            "verify: re-decoded output mismatch in block #{} ({} bytes vs {} bytes)",
            block.block_idx,
            re_decoded.len(),
            block.decoded.len()
        ));
    }
    Ok(())
}

fn write_index_json(picked: &[ExtractedBlock], path: &PathBuf) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("{\n  \"version\": \"GZBLK01\",\n  \"count\": ");
    s.push_str(&picked.len().to_string());
    s.push_str(",\n  \"blocks\": [\n");
    for (i, b) in picked.iter().enumerate() {
        s.push_str("    {\"file\": \"");
        s.push_str(&format!("{:03}.bin", i));
        s.push_str(&format!(
            "\", \"block_idx\": {}, \"decoded_len\": {}, \"compressed_bit_len\": {}, ",
            b.block_idx,
            b.decoded.len(),
            b.compressed_bit_len
        ));
        s.push_str(&format!(
            "\"max_litlen_code_len\": {}, \"max_dist_code_len\": {}, ",
            b.max_litlen_code_len, b.max_dist_code_len
        ));
        s.push_str(&format!(
            "\"literal_code_count\": {}, \"distance_code_count\": {}",
            b.literal_cl.len(),
            b.distance_cl.len()
        ));
        s.push_str("}");
        if i + 1 < picked.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    fs::write(path, s)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gzippy-extract-blocks <input.gz> <output_dir> [target_count=40]");
        std::process::exit(2);
    }
    let input_path = &args[1];
    let output_dir = PathBuf::from(&args[2]);
    let target_count: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);

    let mut file = fs::File::open(input_path).expect("open input");
    let mut input_bytes = Vec::new();
    file.read_to_end(&mut input_bytes).expect("read input");

    let header_len = skip_gzip_header(&input_bytes).expect("parse gzip header");
    let deflate_bytes = &input_bytes[header_len..];

    eprintln!(
        "extracting blocks from {} ({} bytes after gzip header)",
        input_path,
        deflate_bytes.len()
    );

    let all = extract_all(deflate_bytes).expect("extract_all");
    eprintln!("found {} dynamic-Huffman blocks (with non-empty predecessor)", all.len());

    if all.is_empty() {
        eprintln!("no candidate blocks; aborting");
        std::process::exit(1);
    }

    let picked = stratified_pick(all, target_count);
    eprintln!("picked {} blocks via stratified sampling", picked.len());

    fs::create_dir_all(&output_dir).expect("create output_dir");

    for (i, block) in picked.iter().enumerate() {
        verify_extraction(block).expect("verify_extraction");
        let file_path = output_dir.join(format!("{:03}.bin", i));
        block.write_to(&file_path).expect("write block file");
    }

    write_index_json(&picked, &output_dir.join("INDEX.json")).expect("write INDEX.json");

    eprintln!(
        "wrote {} corpus files to {} (verified: all re-decode bit-perfect)",
        picked.len(),
        output_dir.display()
    );
}
