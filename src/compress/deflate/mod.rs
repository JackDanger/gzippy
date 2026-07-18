//! Pure-Rust DEFLATE encoder — Increment 1 (foundation substrate).
//!
//! This module is the entry point for a from-scratch, pure-Rust DEFLATE/gzip
//! compressor whose structure transliterates libdeflate
//! (`vendor/libdeflate/lib/deflate_compress.c`). Increment 1 lands the proven
//! substrate — constant [`tables`], the word-oriented [`bitstream`], the
//! length-limited canonical [`huffman`] builder + dynamic-block header, the
//! [`block_split`] statistic, and the shared [`matchfinder`] primitives — all
//! unit-tested, plus a **literals-only** engine wired end-to-end so the whole
//! stack is exercised against three independent decoders.
//!
//! The literals-only engine performs NO match finding: every input byte is
//! emitted as a literal in a DEFLATE **dynamic** Huffman block built from the
//! literal frequencies (a valid, fully-decodable DEFLATE stream), with a stored
//! block chosen when that is smaller (incompressible / tiny input). Real match
//! finding, greedy/lazy/near-optimal parsing, and static blocks arrive in later
//! increments.
//!
//! NOT wired into production routing yet (see CLAUDE.md): this increment only
//! adds the module tree + tests.
//!
//! Several primitives (offset-slot lookup, static-code frequencies, the
//! matchfinder hooks, block-splitting) are the substrate for later increments
//! and are not yet called from the literals-only path, so `dead_code` is
//! allowed module-wide for this increment.
#![allow(dead_code)]

pub mod bitstream;
pub mod block_split;
pub mod huffman;
pub mod matchfinder;
pub mod tables;

use bitstream::BitWriter;
use huffman::build_dynamic_header;
use tables::{
    DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, DEFLATE_BLOCKTYPE_UNCOMPRESSED, DEFLATE_END_OF_BLOCK,
    DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS, MAX_LITLEN_CODEWORD_LEN,
    MAX_OFFSET_CODEWORD_LEN,
};

/// Maximum input bytes per logical block. Kept below `2^18` so per-symbol
/// frequencies stay well under the `2^22` packing limit of the Huffman builder,
/// and so long inputs are split into several independently-coded blocks.
const BLOCK_CHUNK: usize = 1 << 18;

/// Largest payload of a single stored (BTYPE=00) sub-block.
const MAX_STORED_SUBBLOCK: usize = 65535;

/// Compress `data` into a raw DEFLATE stream (no gzip/zlib framing).
///
/// `level` is accepted for API compatibility but ignored in this increment
/// (the literals-only engine has no tunable parser yet).
pub fn compress_oneshot(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 64);
    compress_block(data, &[], level, &mut out);
    out
}

/// Compress `data` into a raw DEFLATE stream, appending to `out`.
///
/// `dict` is a preset-dictionary hook for later increments; the literals-only
/// engine emits no matches, so it is currently ignored.
pub fn compress_block(data: &[u8], _dict: &[u8], _level: u32, out: &mut Vec<u8>) {
    let mut bw = BitWriter::with_capacity(data.len() / 2 + 64);

    if data.is_empty() {
        emit_stored_block(&mut bw, &[], true);
        out.extend_from_slice(&bw.finish());
        return;
    }

    let mut off = 0usize;
    while off < data.len() {
        let end = (off + BLOCK_CHUNK).min(data.len());
        let chunk = &data[off..end];
        let is_final = end == data.len();
        emit_best_block(&mut bw, chunk, is_final);
        off = end;
    }

    out.extend_from_slice(&bw.finish());
}

/// Compress `data` into a gzip-framed stream (gzip header + DEFLATE + CRC32 +
/// ISIZE). This is the variant the roundtrip oracles consume.
pub fn compress_gzip(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 32);
    // Minimal gzip header: magic, CM=8 (deflate), FLG=0, MTIME=0, XFL=0,
    // OS=255 (unknown).
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]);

    compress_block(data, &[], level, &mut out);

    let crc = crc32fast::hash(data);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// Choose the cheaper of a dynamic Huffman block and a stored block for this
/// chunk, and emit it.
fn emit_best_block(bw: &mut BitWriter, chunk: &[u8], is_final: bool) {
    let plan = DynamicPlan::build(chunk);
    let stored_bits = stored_block_bits(chunk.len());

    if plan.total_bits() <= stored_bits {
        plan.emit(bw, chunk, is_final);
    } else {
        emit_stored_block(bw, chunk, is_final);
    }
}

/// A prepared dynamic Huffman block: the litlen/offset codes, the header, and
/// the precomputed encoded-data bit cost.
struct DynamicPlan {
    litcode: huffman::HuffmanCode,
    header: huffman::DynamicHeader,
    data_bits: u64,
}

impl DynamicPlan {
    fn build(chunk: &[u8]) -> DynamicPlan {
        let mut litlen_freqs = vec![0u32; DEFLATE_NUM_LITLEN_SYMS];
        litlen_freqs[DEFLATE_END_OF_BLOCK] = 1; // end-of-block symbol
        for &b in chunk {
            litlen_freqs[b as usize] += 1;
        }
        // Literals-only: no offset symbols are used.
        let offset_freqs = vec![0u32; DEFLATE_NUM_OFFSET_SYMS];

        let litcode = huffman::make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN,
            &litlen_freqs,
        );
        let offcode = huffman::make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN,
            &offset_freqs,
        );

        let header = build_dynamic_header(&litcode.lens, &offcode.lens);

        // Bits for the coded literals + the EOB symbol.
        let mut data_bits: u64 = 0;
        for &b in chunk {
            data_bits += litcode.lens[b as usize] as u64;
        }
        data_bits += litcode.lens[DEFLATE_END_OF_BLOCK] as u64;

        DynamicPlan {
            litcode,
            header,
            data_bits,
        }
    }

    /// Total bits: 3 (BFINAL+BTYPE) + header + coded data.
    fn total_bits(&self) -> u64 {
        3 + self.header.header_bits() + self.data_bits
    }

    fn emit(&self, bw: &mut BitWriter, chunk: &[u8], is_final: bool) {
        bw.add_bits(is_final as u64, 1);
        bw.add_bits(DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN as u64, 2);
        self.header.emit(bw);
        for &b in chunk {
            bw.add_bits(
                self.litcode.codewords[b as usize] as u64,
                self.litcode.lens[b as usize] as u32,
            );
        }
        bw.add_bits(
            self.litcode.codewords[DEFLATE_END_OF_BLOCK] as u64,
            self.litcode.lens[DEFLATE_END_OF_BLOCK] as u32,
        );
    }
}

/// Approximate bit cost of storing `len` bytes as stored (BTYPE=00) sub-blocks.
fn stored_block_bits(len: usize) -> u64 {
    let subblocks = (len / MAX_STORED_SUBBLOCK) + 1;
    // Per sub-block: 3 header bits + up to 7 pad + 32 bits LEN/NLEN, then the
    // raw bytes. Rounded to whole bytes.
    (8 * (len + 5 * subblocks)) as u64
}

/// Emit one or more stored (uncompressed, BTYPE=00) blocks covering `data`.
///
/// Port of the uncompressed-block emission in `deflate_flush_block` (~:1826).
/// A stored sub-block carries at most 65535 bytes, so long inputs use several;
/// `is_final` marks the last sub-block BFINAL.
fn emit_stored_block(bw: &mut BitWriter, data: &[u8], is_final: bool) {
    if data.is_empty() {
        write_stored_subblock(bw, &[], is_final);
        return;
    }
    let mut off = 0usize;
    while off < data.len() {
        let end = (off + MAX_STORED_SUBBLOCK).min(data.len());
        let last = end == data.len();
        write_stored_subblock(bw, &data[off..end], is_final && last);
        off = end;
    }
}

fn write_stored_subblock(bw: &mut BitWriter, sub: &[u8], bfinal: bool) {
    debug_assert!(sub.len() <= MAX_STORED_SUBBLOCK);
    bw.add_bits(bfinal as u64, 1);
    bw.add_bits(DEFLATE_BLOCKTYPE_UNCOMPRESSED as u64, 2);
    bw.align_to_byte();
    let len = sub.len() as u16;
    bw.write_u16_le(len);
    bw.write_u16_le(!len);
    bw.write_aligned_bytes(sub);
}
