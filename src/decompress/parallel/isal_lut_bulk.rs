//! Stateless windowed-bulk DEFLATE decoder using the ISA-L LUT format
//! from `isal_huffman_pure.rs`.
//!
//! ## Why this exists
//!
//! The parallel-SM bulk phase (post-bootstrap, with a known 32 KiB
//! predecessor window) currently routes through `ResumableInflate2`
//! (`src/decompress/inflate/resumable.rs`). Per the 2026-05-27 neurotic
//! perf profile, ResumableInflate2's bulk hot loop accounts for
//!   - 9.39% CPU in `decode_huffman_body_resumable`
//!   - 6.04% CPU in `copy_match_windowed`
//!
//! ResumableInflate2 is heavily optimized (T0 raw-pointer literal writes,
//! T3 multi-literal lookahead, T4 yield-elide FASTLOOP) but carries
//! state-machine yield-check overhead from supporting the speculative-
//! decode use case. For the BULK windowed path, no yields are needed —
//! we have a clean output buffer, known input range, no marker emission.
//!
//! This module is the load-bearing piece for closing the remaining
//! pure-rust-vs-ISA-L gap (~10% paired median). The ISA-L bulk decoder
//! (which gzippy's `--features isal-compression` build uses) is
//! hand-tuned assembly we can't match cycle-for-cycle, but a tight
//! pure-Rust loop using the same LUT format (ISA-L's
//! `inflate_huff_code_large` with 12-bit + variable-long lookup, up to
//! 3 packed symbols per entry, pre-baked length extras) closes the
//! algorithmic-shape gap.
//!
//! ## Design
//!
//! - **Stateless.** No yield-on-output-fill, no resume tokens. The
//!   caller provides the entire (compressed) bit range AND the entire
//!   output buffer (sized to known bounds from the chunk config). The
//!   decoder either completes the block or errors.
//! - **Single-block.** Decodes ONE block at a time. The caller's
//!   block-iteration logic (in `gzip_chunk.rs` or `chunk_fetcher.rs`)
//!   loops to handle multi-block streams.
//! - **No markers.** The windowed path emits clean u8 directly — no
//!   u16 marker emission, no marker-zone tracking. Back-references
//!   resolve against the predecessor window OR the output already
//!   decoded in this chunk.
//! - **Pure Rust.** Uses `IsalLitLenCodePure::decode` (already a
//!   literal port of ISA-L's `HuffmanCodingISAL::decode`).
//!
//! ## Phasing
//!
//! - **Step 1 (this commit):** Scaffold + unit tests on simple
//!   fixtures. NOT wired into production.
//! - **Step 2 (next commit):** Wire as the bulk decoder in
//!   `gzip_chunk::decode_chunk_marker_bootstrap_then_isal` (replacing
//!   the ISA-L FFI call when not built with `--features isal-compression`).
//!   Silesia byte-perfect gate.
//! - **Step 3:** Bench on neurotic; verify gap closes.
//!
//! ## Correctness contract
//!
//! - Caller must pass `bits` positioned at the START of a deflate
//!   block's header.
//! - Caller must provide `output` sized to hold the full decoded
//!   block (bounded by RFC 1951 deflate semantics; in practice
//!   <= max_decoded_chunk_size per chunk).
//! - Caller must provide `predecessor_window` of length 32 KiB
//!   (or shorter for chunk-0 = zero window).
//! - Returns the number of output bytes written + the bit position
//!   immediately after the block trailer.
//! - On invalid Huffman code or out-of-bounds backref: returns
//!   `BulkDecodeError`.

#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
#![allow(dead_code)] // scaffold; production wiring lands in next commit

use crate::decompress::inflate::consume_first_decode::Bits;
use crate::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;
use crate::decompress::parallel::isal_huffman_pure::{
    IsalLitLenCodePure, LIT_LEN, MAX_HUFF_TREE_DEPTH,
};

/// Distance-decoder type — uses the proven `HuffmanCodingReversedBitsCached`
/// rather than `BulkDistDecoder` because (a) the dist code is only
/// 2.56% of CPU per profile (not the lever), and (b) the canonical
/// decoder's count semantics are well-tested via the existing canonical
/// hot path. Using `BulkDistDecoder` would require reconciling its
/// count_cumulative construction with `make_inflate_huff_code_dist`'s
/// expectation.
pub type BulkDistDecoder = HuffmanCodingReversedBitsCached<30>;

const MAX_WINDOW_SIZE: usize = 32 * 1024;
const MAX_MATCH_LENGTH: usize = 258;
const END_OF_BLOCK_SYMBOL: u32 = 256;
/// Vendor `DISTANCE_OFFSET` (deflate.hpp:1642): length codes are stored
/// post-expansion as `254 + actual_length` in the ISA-L LUT.
const LENGTH_BASE_OFFSET: u32 = 254;
const MAX_LIT_LEN_SYM: u32 = 512;

/// RFC 1951 distance-extra-bit count, indexed by distance code (0..30).
/// Mirror of `isal_huffman_pure`'s vendor table.
const DIST_EXTRA_BIT_COUNT: [u8; 32] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13, 0, 0,
];

/// RFC 1951 distance base values, indexed by distance code (0..30).
const DIST_START: [u32; 32] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0,
];

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BulkDecodeError {
    InvalidHuffmanCode,
    InvalidLookback,
    OutputOverflow,
    BlockTypeReserved,
    InvalidCodeLengths,
}

/// Successful block decode outcome.
#[derive(Debug, Copy, Clone)]
pub struct BulkBlockResult {
    /// Number of output bytes written by this block.
    pub bytes_written: usize,
    /// True if this was the final block (BFINAL=1).
    pub is_final_block: bool,
}

/// Decode one DEFLATE block using ISA-L's LUT format.
///
/// `bits` must be positioned at the start of the block's header (the
/// BFINAL bit). After return, `bits` is positioned just past the
/// block's data (at the next block's header or end-of-stream).
///
/// `output_start_offset` is the index in `output` where this block's
/// bytes begin. Back-references within `output_start_offset` bytes of
/// the start resolve into `predecessor_window`.
///
/// `predecessor_window` is 32 KiB of the bytes that immediately
/// precede `output[output_start_offset]`. For chunk-0 this is all
/// zeros (vendor convention; the gzip stream has no predecessor at
/// the start).
pub fn decode_block(
    bits: &mut Bits<'_>,
    output: &mut [u8],
    output_start_offset: usize,
    predecessor_window: &[u8],
    litlen_decoder: &mut IsalLitLenCodePure,
    dist_decoder: &mut BulkDistDecoder,
) -> Result<BulkBlockResult, BulkDecodeError> {
    // ── Block header: 3 bits = BFINAL (1) + BTYPE (2) ──────────────────
    bits.refill();
    let header_bits = bits.bitbuf & 0b111;
    bits.consume(3);
    let is_final_block = (header_bits & 0b1) != 0;
    let btype = (header_bits >> 1) & 0b11;

    let mut out_pos = output_start_offset;

    match btype {
        0b00 => {
            // Stored block: align to byte boundary, read LEN, NLEN,
            // then copy LEN bytes from input.
            return decode_stored_block(bits, output, out_pos, is_final_block);
        }
        0b01 => {
            // Fixed Huffman: build the fixed-tables version then
            // delegate to the dynamic decode loop. (Caller passes the
            // decoder structs as scratch.)
            build_fixed_huffman(litlen_decoder, dist_decoder)?;
        }
        0b10 => {
            // Dynamic Huffman: read code-length headers and build
            // the lit/len + dist tables in the caller's decoder
            // structs.
            build_dynamic_huffman(bits, litlen_decoder, dist_decoder)?;
        }
        _ => return Err(BulkDecodeError::BlockTypeReserved),
    }

    // ── Block body: decode symbols via the ISA-L LUT ────────────────────
    loop {
        bits.refill();
        let decoded = litlen_decoder.decode(bits);
        if decoded.bit_count == 0 {
            return Err(BulkDecodeError::InvalidHuffmanCode);
        }
        bits.consume(decoded.bit_count);

        let mut symbol = decoded.symbol;
        let mut sym_count = decoded.sym_count;
        if sym_count == 0 {
            return Err(BulkDecodeError::InvalidHuffmanCode);
        }

        while sym_count > 0 {
            let code = (symbol & 0xFFFF) as u16;
            if code as u32 <= 255 || sym_count > 1 {
                // Literal: write byte to output.
                if out_pos >= output.len() {
                    return Err(BulkDecodeError::OutputOverflow);
                }
                output[out_pos] = (code & 0xFF) as u8;
                out_pos += 1;
                symbol >>= 8;
                sym_count -= 1;
                continue;
            }

            if code as u32 == END_OF_BLOCK_SYMBOL {
                return Ok(BulkBlockResult {
                    bytes_written: out_pos - output_start_offset,
                    is_final_block,
                });
            }
            if code as u32 > MAX_LIT_LEN_SYM {
                return Err(BulkDecodeError::InvalidHuffmanCode);
            }

            // Length code (post-expansion in ISA-L LUT — actual length
            // = symbol - 254 per `LENGTH_BASE_OFFSET`).
            let length = (symbol - LENGTH_BASE_OFFSET) as usize;
            if length == 0 || length > MAX_MATCH_LENGTH {
                return Err(BulkDecodeError::InvalidHuffmanCode);
            }

            // Distance code lookup. `HuffmanCodingReversedBitsCached`
            // consumes bits internally via the LsbBitReader trait.
            use crate::decompress::parallel::huffman_base::LsbBitReader as _;
            let dist_sym = dist_decoder
                .decode(bits)
                .ok_or(BulkDecodeError::InvalidHuffmanCode)? as u32;
            if dist_sym >= 30 {
                return Err(BulkDecodeError::InvalidHuffmanCode);
            }
            // Distance extra bits.
            bits.refill();
            let extra_bits = DIST_EXTRA_BIT_COUNT[dist_sym as usize];
            let extra_val = if extra_bits > 0 {
                let v = (bits.bitbuf & ((1u64 << extra_bits) - 1)) as u32;
                bits.consume(extra_bits as u32);
                v
            } else {
                0
            };
            let distance = (DIST_START[dist_sym as usize] + extra_val) as usize;
            if distance == 0 || distance > MAX_WINDOW_SIZE {
                return Err(BulkDecodeError::InvalidLookback);
            }
            if out_pos + length > output.len() {
                return Err(BulkDecodeError::OutputOverflow);
            }

            // Copy `length` bytes from `distance` behind out_pos.
            // The source may be in the predecessor window (if it
            // extends before output_start_offset) or in `output`
            // already decoded.
            copy_match(
                output,
                out_pos,
                distance,
                length,
                output_start_offset,
                predecessor_window,
            );
            out_pos += length;

            symbol >>= 8;
            sym_count -= 1;
        }
    }
}

/// Decode a stored (uncompressed) DEFLATE block via the existing
/// `decode_stored_pub` helper in `consume_first_decode`. That helper
/// handles the LEN/NLEN header parsing + byte-aligned raw copy. We
/// don't reimplement it here because stored blocks are rare in real
/// workloads and the existing code is already correct.
fn decode_stored_block(
    bits: &mut Bits<'_>,
    output: &mut [u8],
    out_pos: usize,
    is_final_block: bool,
) -> Result<BulkBlockResult, BulkDecodeError> {
    use crate::decompress::inflate::consume_first_decode::decode_stored_pub;
    // `decode_stored_pub` returns the new `out_pos` (which the caller
    // uses to track total bytes); we convert to bytes_written.
    let new_pos =
        decode_stored_pub(bits, output, out_pos).map_err(|_| BulkDecodeError::OutputOverflow)?;
    Ok(BulkBlockResult {
        bytes_written: new_pos - out_pos,
        is_final_block,
    })
}

/// Copy `length` bytes from `out_pos - distance` to `out_pos` in
/// `output`. If the source extends before `out_start`, the bytes
/// come from `predecessor_window`.
#[inline]
fn copy_match(
    output: &mut [u8],
    out_pos: usize,
    distance: usize,
    length: usize,
    out_start: usize,
    predecessor_window: &[u8],
) {
    // Where the back-reference source begins (relative to out_pos).
    if distance <= out_pos - out_start {
        // All source bytes are within this chunk's output.
        // Use byte-by-byte copy to handle run-length encoding
        // (distance < length is valid DEFLATE).
        let src_start = out_pos - distance;
        for i in 0..length {
            output[out_pos + i] = output[src_start + i];
        }
    } else {
        // Source extends into the predecessor window.
        let chunk_local_avail = out_pos - out_start;
        let from_window = distance - chunk_local_avail;
        let window_src_start = predecessor_window.len() - from_window;
        let mut written = 0;
        // Phase 1: copy from window.
        let window_bytes = from_window.min(length);
        for i in 0..window_bytes {
            output[out_pos + i] = predecessor_window[window_src_start + i];
        }
        written += window_bytes;
        // Phase 2: copy from this chunk's output (now including the
        // bytes we just wrote — RLE-style).
        while written < length {
            output[out_pos + written] = output[out_pos + written - distance];
            written += 1;
        }
    }
}

/// Build fixed-Huffman tables per RFC 1951 §3.2.6 into the caller's
/// decoder structs. The fixed table uses code lengths:
///   - literals 0..143:  8 bits
///   - literals 144..255: 9 bits
///   - 256..279 (EOB + short length codes): 7 bits
///   - 280..285 (long length codes): 8 bits
///   - distances 0..29: 5 bits
fn build_fixed_huffman(
    litlen_decoder: &mut IsalLitLenCodePure,
    dist_decoder: &mut BulkDistDecoder,
) -> Result<(), BulkDecodeError> {
    let mut litlen_lens = vec![0u8; LIT_LEN];
    for sym in 0..144 {
        litlen_lens[sym] = 8;
    }
    for sym in 144..256 {
        litlen_lens[sym] = 9;
    }
    for sym in 256..280 {
        litlen_lens[sym] = 7;
    }
    for sym in 280..286 {
        litlen_lens[sym] = 8;
    }
    if !litlen_decoder.rebuild_from(&litlen_lens) {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    let mut dist_lens = vec![0u8; 30];
    for sym in 0..30 {
        dist_lens[sym] = 5;
    }
    let err = dist_decoder.initialize_from_lengths(&dist_lens, false);
    if err != crate::decompress::parallel::error::Error::None {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    Ok(())
}

/// Read the dynamic-Huffman block header and build litlen + dist
/// decoders. RFC 1951 §3.2.7.
///
/// Header structure:
///   HLIT (5 bits) = # literal/length codes - 257       → 257..286
///   HDIST (5 bits) = # distance codes - 1               → 1..32
///   HCLEN (4 bits) = # code-length codes - 4            → 4..19
///   HCLEN × 3-bit code-length-code lengths              → CL alphabet
///   HLIT + HDIST run-length-encoded code lengths        → lit/len + dist
fn build_dynamic_huffman(
    bits: &mut Bits<'_>,
    litlen_decoder: &mut IsalLitLenCodePure,
    dist_decoder: &mut BulkDistDecoder,
) -> Result<(), BulkDecodeError> {
    bits.refill();
    let hlit = ((bits.bitbuf & 0x1F) + 257) as usize;
    bits.consume(5);
    let hdist = ((bits.bitbuf & 0x1F) + 1) as usize;
    bits.consume(5);
    let hclen = ((bits.bitbuf & 0x0F) + 4) as usize;
    bits.consume(4);

    if hlit > 286 || hdist > 30 || hclen > 19 {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }

    // Code-length-code alphabet ordering per RFC 1951 §3.2.7.
    const CL_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut cl_lens = [0u8; 19];
    for i in 0..hclen {
        bits.refill();
        cl_lens[CL_ORDER[i]] = (bits.bitbuf & 0x07) as u8;
        bits.consume(3);
    }

    // Build a CL-alphabet decoder using the canonical reversed-bits
    // cache decoder (HuffmanCodingReversedBitsCached). The CL alphabet
    // has 19 symbols with max code length 7; the dist-format decoder
    // (`BulkDistDecoder`) has the wrong layout for this use case
    // (it bakes distance-extra-bit fields into entries).
    use crate::decompress::parallel::huffman_base::LsbBitReader as _;
    use crate::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;
    let mut cl_decoder = HuffmanCodingReversedBitsCached::<19>::new();
    let err = cl_decoder.initialize_from_lengths(&cl_lens, false);
    if err != crate::decompress::parallel::error::Error::None {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }

    // Now decode hlit + hdist code lengths using cl_decoder.
    let total = hlit + hdist;
    let mut all_lens = vec![0u8; total];
    let mut i = 0;
    while i < total {
        let sym = cl_decoder
            .decode(bits)
            .ok_or(BulkDecodeError::InvalidCodeLengths)? as u32;
        match sym {
            0..=15 => {
                all_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                // Repeat previous length 3..6 times (2 extra bits).
                if i == 0 {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                bits.refill();
                let extra = (bits.bitbuf & 0x3) as usize;
                bits.consume(2);
                let repeat = 3 + extra;
                if i + repeat > total {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                let prev = all_lens[i - 1];
                for _ in 0..repeat {
                    all_lens[i] = prev;
                    i += 1;
                }
            }
            17 => {
                // Repeat 0 length 3..10 times (3 extra bits).
                bits.refill();
                let extra = (bits.bitbuf & 0x7) as usize;
                bits.consume(3);
                let repeat = 3 + extra;
                if i + repeat > total {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                i += repeat;
            }
            18 => {
                // Repeat 0 length 11..138 times (7 extra bits).
                bits.refill();
                let extra = (bits.bitbuf & 0x7F) as usize;
                bits.consume(7);
                let repeat = 11 + extra;
                if i + repeat > total {
                    return Err(BulkDecodeError::InvalidCodeLengths);
                }
                i += repeat;
            }
            _ => return Err(BulkDecodeError::InvalidCodeLengths),
        }
    }

    let litlen_lens = &all_lens[..hlit];
    let dist_lens = &all_lens[hlit..];
    if !litlen_decoder.rebuild_from(litlen_lens) {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    let mut dist_padded = vec![0u8; 30];
    dist_padded[..dist_lens.len()].copy_from_slice(dist_lens);
    let derr = dist_decoder.initialize_from_lengths(&dist_padded, false);
    if derr != crate::decompress::parallel::error::Error::None {
        return Err(BulkDecodeError::InvalidCodeLengths);
    }
    let _ = MAX_HUFF_TREE_DEPTH; // silence unused-import warning
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a fixed-Huffman block produced by flate2 and verify
    /// byte-perfect output. This is the first correctness gate; the
    /// wiring commit adds the silesia byte-perfect differential.
    #[test]
    fn decode_block_round_trips_fixed_huffman() {
        // Build a small payload, compress as fixed-Huffman via flate2,
        // then decode via our bulk decoder.
        use std::io::Write;
        let payload = b"the quick brown fox jumps over the lazy dog".repeat(5);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(1));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();

        let mut output = vec![0u8; payload.len() + 1024];
        let mut bits = Bits::new(&deflate);
        bits.refill();
        let mut litlen = IsalLitLenCodePure::new_empty();
        let mut dist = BulkDistDecoder::new();
        let predecessor = [0u8; MAX_WINDOW_SIZE];

        let mut total = 0;
        loop {
            let result = decode_block(
                &mut bits,
                &mut output,
                total,
                &predecessor[..],
                &mut litlen,
                &mut dist,
            )
            .expect("fixed Huffman block must decode");
            total += result.bytes_written;
            if result.is_final_block {
                break;
            }
        }
        assert_eq!(&output[..total], &payload[..]);
    }

    /// Decode a dynamic-Huffman block produced by flate2 (Compression
    /// level 6 forces dynamic blocks).
    #[test]
    fn decode_block_round_trips_dynamic_huffman() {
        use std::io::Write;
        let payload = b"abc123def456ghi789jkl012mno345pqr678stu901vwx234yz".repeat(200);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();

        let mut output = vec![0u8; payload.len() + 1024];
        let mut bits = Bits::new(&deflate);
        bits.refill();
        let mut litlen = IsalLitLenCodePure::new_empty();
        let mut dist = BulkDistDecoder::new();
        let predecessor = [0u8; MAX_WINDOW_SIZE];

        let mut total = 0;
        loop {
            let result = decode_block(
                &mut bits,
                &mut output,
                total,
                &predecessor[..],
                &mut litlen,
                &mut dist,
            )
            .expect("dynamic Huffman block must decode");
            total += result.bytes_written;
            if result.is_final_block {
                break;
            }
        }
        assert_eq!(&output[..total], &payload[..]);
    }

    /// REAL-CORPUS correctness gate. Per memory rule
    /// `feedback-real-corpus-test-with-lever`: any new inner-decoder
    /// lever must include a silesia (or equivalent) differential test
    /// IN THE SAME COMMIT as the lever. Synthetic round-trips
    /// over-trust — silesia exercises real-world block diversity,
    /// packed-pair bit patterns, length-extra encodings, dynamic
    /// huffman header diversity, and multi-MB back-reference patterns
    /// that fixture-only tests miss.
    ///
    /// This test reads silesia-large.gz IF AVAILABLE, decompresses
    /// via flate2 (ground truth), then independently decompresses the
    /// raw DEFLATE stream via our bulk decoder in a multi-block loop,
    /// and asserts byte-equal output. Reads only the first single-gzip-
    /// member portion (no multi-member handling in this stateless
    /// decoder — caller orchestrates members in the production wiring).
    ///
    /// Skipped silently when silesia-large.gz isn't on the local
    /// machine (CI without corpus). Always runs on neurotic.
    #[test]
    fn decode_block_byte_perfect_silesia_if_available() {
        // Try several common locations for the silesia corpus.
        let candidates = [
            "benchmark_data/silesia-large.gz",
            "benchmark_data/silesia.tar.gz",
            "benchmark_data/silesia-gzip.tar.gz",
        ];
        let data = match candidates.iter().find_map(|p| std::fs::read(p).ok()) {
            Some(d) => d,
            None => {
                eprintln!("[silesia bulk test] no silesia corpus available, skipping");
                return;
            }
        };

        // Parse gzip header to find DEFLATE start. RFC 1952 §2.2:
        // 10-byte fixed header + optional FNAME / FCOMMENT / FEXTRA /
        // FHCRC. We use the existing gzip_format parser.
        let (hdr, hdr_size) =
            crate::decompress::parallel::gzip_format::read_header(&data).expect("gzip header");
        let _ = hdr;
        let deflate_start = hdr_size;
        // Footer is 8 bytes (CRC32 + ISIZE).
        let deflate_end = data.len() - 8;
        let deflate = &data[deflate_start..deflate_end];

        // Ground truth via flate2.
        use std::io::Read;
        let mut gz = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        gz.read_to_end(&mut expected).expect("flate2 decode");

        // Decompress via our bulk decoder in a multi-block loop.
        let mut output = vec![0u8; expected.len() + 4096];
        let mut bits = Bits::new(deflate);
        bits.refill();
        let mut litlen = IsalLitLenCodePure::new_empty();
        let mut dist = BulkDistDecoder::new();
        let predecessor = [0u8; MAX_WINDOW_SIZE];

        let mut total = 0;
        loop {
            let result = decode_block(
                &mut bits,
                &mut output,
                total,
                &predecessor[..],
                &mut litlen,
                &mut dist,
            )
            .unwrap_or_else(|e| panic!("silesia block decode failed at out_pos={total}: {e:?}"));
            total += result.bytes_written;
            if result.is_final_block {
                break;
            }
        }
        // Trim to actual length and compare.
        output.truncate(total);
        assert_eq!(
            output.len(),
            expected.len(),
            "decoded length mismatch: got {}, expected {}",
            output.len(),
            expected.len()
        );
        // Compare in chunks for clearer failure reports.
        const CHUNK: usize = 4096;
        for i in (0..output.len()).step_by(CHUNK) {
            let end = (i + CHUNK).min(output.len());
            assert_eq!(
                &output[i..end],
                &expected[i..end],
                "byte mismatch in chunk [{}..{}]",
                i,
                end
            );
        }
        eprintln!("[silesia bulk test] {} bytes byte-perfect", total);
    }

    /// Decode a stored (uncompressed) block.
    #[test]
    fn decode_block_round_trips_stored() {
        use std::io::Write;
        let payload = b"stored block test payload".to_vec();
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::none());
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();

        let mut output = vec![0u8; payload.len() + 1024];
        let mut bits = Bits::new(&deflate);
        bits.refill();
        let mut litlen = IsalLitLenCodePure::new_empty();
        let mut dist = BulkDistDecoder::new();
        let predecessor = [0u8; MAX_WINDOW_SIZE];

        let mut total = 0;
        loop {
            let result = decode_block(
                &mut bits,
                &mut output,
                total,
                &predecessor[..],
                &mut litlen,
                &mut dist,
            )
            .expect("stored block must decode");
            total += result.bytes_written;
            if result.is_final_block {
                break;
            }
        }
        assert_eq!(&output[..total], &payload[..]);
    }
}
