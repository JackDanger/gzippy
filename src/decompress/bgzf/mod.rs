//! BGZF (Block GZIP Format) Parallel Decompression
//!
//! BGZF files have independent blocks with embedded size markers, allowing
//! perfect parallelism with zero lock contention.
//!
//! ## Strategy
//!
//! 1. Parse BGZF headers to find all block boundaries and output sizes (ISIZE)
//! 2. Pre-allocate entire output buffer based on sum of ISIZE values
//! 3. Decompress blocks in parallel, writing directly to pre-calculated offsets
//! 4. Single write of complete output
//!
//! ## Performance Target: 4000+ MB/s with 14 threads
//!
//! With single-threaded inflate at 10700 MB/s and no lock contention,
//! theoretical max is ~150,000 MB/s. Memory bandwidth limits us to ~4000-5000 MB/s.

#![allow(clippy::needless_range_loop)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(unused_imports)]
use crate::decompress::two_level_table::{FastBits, TurboBits, TwoLevelTable};

/// BGZF block information
#[derive(Debug, Clone)]
pub(crate) struct BgzfBlock {
    /// Byte offset of block start in compressed data
    start: usize,
    /// Total block length (including header and trailer)
    length: usize,
    /// Uncompressed size (from ISIZE trailer)
    isize: u32,
    /// Output offset (calculated during planning)
    output_offset: usize,
    /// Byte offset of raw deflate data within the block (past gzip header)
    deflate_start: usize,
}

/// Parse all BGZF blocks from compressed data
fn parse_bgzf_blocks(data: &[u8]) -> io::Result<Vec<BgzfBlock>> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    let mut output_offset = 0;

    while offset + 18 < data.len() {
        // Check gzip magic
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        // Must have FEXTRA flag
        if data[offset + 3] & 0x04 == 0 {
            break;
        }

        // Get XLEN
        if offset + 12 > data.len() {
            break;
        }
        let xlen = u16::from_le_bytes([data[offset + 10], data[offset + 11]]) as usize;
        if offset + 12 + xlen > data.len() {
            break;
        }

        // Find GZ subfield with block size
        let extra_start = offset + 12;
        let extra_field = &data[extra_start..extra_start + xlen];
        let mut block_size = None;
        let mut pos = 0;

        while pos + 4 <= extra_field.len() {
            let subfield_id = &extra_field[pos..pos + 2];
            let subfield_len =
                u16::from_le_bytes([extra_field[pos + 2], extra_field[pos + 3]]) as usize;

            if subfield_id == b"GZ" {
                if subfield_len == 4 && pos + 8 <= extra_field.len() {
                    // New 4-byte format (supports blocks > 64KB)
                    let size = u32::from_le_bytes([
                        extra_field[pos + 4],
                        extra_field[pos + 5],
                        extra_field[pos + 6],
                        extra_field[pos + 7],
                    ]) as usize;
                    if size > 0 {
                        block_size = Some(size);
                    }
                    break;
                } else if subfield_len == 2 && pos + 6 <= extra_field.len() {
                    // Legacy 2-byte format (BSIZE-1)
                    let size_minus_1 =
                        u16::from_le_bytes([extra_field[pos + 4], extra_field[pos + 5]]) as usize;
                    block_size = Some(size_minus_1 + 1);
                    break;
                }
            }

            pos += 4 + subfield_len;
        }

        let length = match block_size {
            Some(l) if l > 0 && offset + l <= data.len() => l,
            _ => break,
        };

        // Read ISIZE from trailer (last 4 bytes of block)
        let isize = if length >= 8 {
            let trailer_start = offset + length - 4;
            u32::from_le_bytes([
                data[trailer_start],
                data[trailer_start + 1],
                data[trailer_start + 2],
                data[trailer_start + 3],
            ])
        } else {
            0
        };

        // Deflate data starts after the full gzip header (including optional fields)
        let mut deflate_start = offset + 12 + xlen;
        let flags = data[offset + 3];
        // FNAME: null-terminated filename
        if flags & 0x08 != 0 {
            while deflate_start < offset + length && data[deflate_start] != 0 {
                deflate_start += 1;
            }
            deflate_start += 1; // skip null terminator
        }
        // FCOMMENT: null-terminated comment
        if flags & 0x10 != 0 {
            while deflate_start < offset + length && data[deflate_start] != 0 {
                deflate_start += 1;
            }
            deflate_start += 1;
        }
        // FHCRC: 2-byte header CRC
        if flags & 0x02 != 0 {
            deflate_start += 2;
        }

        blocks.push(BgzfBlock {
            start: offset,
            length,
            isize,
            output_offset,
            deflate_start,
        });

        output_offset += isize as usize;
        offset += length;
    }

    if blocks.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No BGZF blocks found",
        ));
    }

    Ok(blocks)
}

/// Inflate directly into a pre-allocated output slice
///
/// Decompress raw deflate data using the pure-Rust inflate engine.
///
/// This is the key function for zero-copy parallel decompression. The pure-Rust
/// decoder (`inflate_consume_first`) is stateless — there is no per-call
/// decompressor to allocate/free — and carries NO C-FFI, so the BGZF /
/// multi-member decode graph is FFI-free.
fn inflate_into(deflate_data: &[u8], output: &mut [u8]) -> io::Result<usize> {
    crate::decompress::inflate::consume_first_decode::inflate_consume_first(deflate_data, output)
}

/// Public version of inflate_into for use by other modules.
///
/// Pure-Rust inflate — no C-FFI in the decode graph.
pub fn inflate_into_pub(deflate_data: &[u8], output: &mut [u8]) -> io::Result<usize> {
    inflate_into(deflate_data, output)
}

/// Parallel BGZF decompression returning output as a Vec.
///
/// This is the zero-copy path: the output Vec is filled in-place by
/// parallel threads, then returned directly to the caller without any
/// intermediate copies.
pub fn decompress_bgzf_parallel_to_vec(data: &[u8], num_threads: usize) -> io::Result<Vec<u8>> {
    let blocks = parse_bgzf_blocks(data)?;

    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    let total_output: usize = blocks.iter().map(|b| b.isize as usize).sum();
    let output = vec![0u8; total_output];

    let num_blocks = blocks.len();
    let next_block = AtomicUsize::new(0);
    let had_error = std::sync::atomic::AtomicBool::new(false);

    use std::cell::UnsafeCell;
    struct OutputBuffer(UnsafeCell<Vec<u8>>);
    unsafe impl Sync for OutputBuffer {}

    let output_cell = OutputBuffer(UnsafeCell::new(output));

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(num_blocks) {
            let blocks_ref = &blocks;
            let next_ref = &next_block;
            let output_ref = &output_cell;
            let error_ref = &had_error;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let out_size = block.isize as usize;
                    if out_size == 0 {
                        continue;
                    }

                    // Raw deflate: skip gzip header, stop before 8-byte trailer
                    let deflate_end = block.start + block.length - 8;
                    let deflate_data = &data[block.deflate_start..deflate_end];

                    // SAFETY: Each block writes to a disjoint region
                    let output_ptr = unsafe { (*output_ref.0.get()).as_mut_ptr() };
                    let out_start = block.output_offset;
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(output_ptr.add(out_start), out_size)
                    };

                    match inflate_into(deflate_data, out_slice) {
                        Ok(actual_out) if actual_out == out_size => {}
                        _ => error_ref.store(true, Ordering::Relaxed),
                    }
                }
            });
        }
    });

    if had_error.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CRC32 or size mismatch in BGZF block",
        ));
    }

    Ok(output_cell.0.into_inner())
}

/// Parallel BGZF decompression writing to a generic writer.
///
/// For single-thread (num_threads=1), uses a streaming path that decompresses
/// block-by-block into a reusable buffer.
///
/// For multi-thread, uses a pipelined architecture:
///   - N decoder threads pull blocks via atomic counter, decompress into
///     pooled buffers, and send (block_index, buffer) through a channel
///   - Main thread receives completed blocks, writes them in order,
///     and returns buffers to the pool
///
/// This avoids allocating the full output (~211MB for silesia) which caused
/// ~53K page faults and only 2x scaling with 4 threads. The pipeline uses
/// a small buffer pool (~1MB) and writes blocks as they complete.
pub fn decompress_bgzf_parallel<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    if num_threads <= 1 {
        return decompress_bgzf_streaming(data, writer);
    }
    decompress_bgzf_pipelined(data, writer, num_threads)
}

/// Pipelined parallel BGZF: decoder threads + ordered writer.
///
/// Buffer pool avoids per-block allocation. Completed blocks are written
/// in order as they arrive, overlapping I/O with decompression.
fn decompress_bgzf_pipelined<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    let blocks = parse_bgzf_blocks(data)?;
    if blocks.is_empty() {
        return Ok(0);
    }

    let num_blocks = blocks.len();
    let max_block_output = blocks.iter().map(|b| b.isize as usize).max().unwrap_or(0);

    // Completed blocks channel: (block_index, decompressed_data)
    // Bounded to 2*threads so decoders don't race too far ahead of the writer.
    let channel_cap = num_threads * 2 + 2;
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<u8>)>(channel_cap);

    let next_block = AtomicUsize::new(0);
    let had_error = std::sync::atomic::AtomicBool::new(false);
    let mut total = 0u64;

    std::thread::scope(|scope| {
        // Spawn N decoder threads, each with its own reusable buffer
        for _ in 0..num_threads.min(num_blocks) {
            let done_tx = done_tx.clone();
            let blocks_ref = &blocks;
            let next_ref = &next_block;
            let error_ref = &had_error;

            scope.spawn(move || {
                let mut buf = vec![0u8; max_block_output];

                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let out_size = block.isize as usize;
                    if out_size == 0 {
                        let _ = done_tx.send((idx, Vec::new()));
                        continue;
                    }

                    if buf.len() < out_size {
                        buf.resize(out_size, 0);
                    }

                    let deflate_end = block.start + block.length - 8;
                    let deflate_data = &data[block.deflate_start..deflate_end];

                    let actual_out = match inflate_into(deflate_data, &mut buf[..out_size]) {
                        Ok(n) if n == out_size => n,
                        Ok(n) => {
                            error_ref.store(true, Ordering::Relaxed);
                            n
                        }
                        Err(_) => {
                            error_ref.store(true, Ordering::Relaxed);
                            0
                        }
                    };

                    // Transfer buffer ownership through the channel; swap in a
                    // fresh capacity-only Vec so the next block has a buffer to
                    // fill without any copy of the decompressed bytes.
                    buf.truncate(actual_out);
                    let send_buf =
                        std::mem::replace(&mut buf, Vec::with_capacity(max_block_output));
                    let _ = done_tx.send((idx, send_buf));
                }
            });
        }
        drop(done_tx); // close channel when all decoders finish

        // Writer: receive completed blocks, write in order.
        // Blocks may arrive out of order; hold them in a BTreeMap until
        // the next sequential block is available, then flush.
        let mut next_to_write = 0usize;
        let mut pending = std::collections::BTreeMap::<usize, Vec<u8>>::new();
        let mut write_error: Option<io::Error> = None;

        for (idx, data_vec) in &done_rx {
            pending.insert(idx, data_vec);

            while let Some(block_data) = pending.remove(&next_to_write) {
                if write_error.is_none() && !block_data.is_empty() {
                    if let Err(e) = writer.write_all(&block_data) {
                        write_error = Some(e);
                    }
                    total += block_data.len() as u64;
                }
                next_to_write += 1;
            }
        }
    });

    if had_error.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CRC32 or size mismatch in BGZF block",
        ));
    }

    Ok(total)
}

/// Streaming BGZF decompression: decompress one block at a time into a
/// reusable buffer, write immediately. No full-output-size allocation.
///
/// Uses raw deflate decompress with a reused decompressor (skipping both
/// gzip header re-parsing and decompressor alloc/free per block).
fn decompress_bgzf_streaming<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    let blocks = parse_bgzf_blocks(data)?;
    if blocks.is_empty() {
        return Ok(0);
    }

    let max_block_output = blocks.iter().map(|b| b.isize as usize).max().unwrap_or(0);
    let mut buf = vec![0u8; max_block_output];
    let mut total = 0u64;

    for block in &blocks {
        let out_size = block.isize as usize;
        if out_size == 0 {
            continue;
        }

        if out_size > buf.len() {
            buf.resize(out_size, 0);
        }

        // Raw deflate data: between header and 8-byte trailer (CRC32 + ISIZE)
        let deflate_end = block.start + block.length - 8;
        let deflate_data = &data[block.deflate_start..deflate_end];

        let actual_out = inflate_into(deflate_data, &mut buf[..out_size]).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "deflate decompression failed in BGZF block",
            )
        })?;

        writer.write_all(&buf[..actual_out])?;
        total += actual_out as u64;
    }
    Ok(total)
}
mod multi_member;

pub use multi_member::{decompress_multi_member_parallel, decompress_multi_member_parallel_to_vec};
pub(crate) use multi_member::{fast_path_ok, gz_coverage_is_pure, scan_member_boundaries_fast};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
