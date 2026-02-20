#![allow(dead_code)]
//! Two-Pass Parallel Single-Member Decompression
//!
//! Closes the performance gap with rapidgzip on single-member gzip files.
//!
//! ## Algorithm
//!
//! Pass 1 (scan): Decode the full deflate stream block-by-block, recording
//! checkpoints at regular intervals. Each checkpoint saves the bit-reader
//! state (byte position, bitbuf, bitsleft) and a 32KB window snapshot.
//!
//! Pass 2 (parallel re-decode): For each chunk between two consecutive
//! checkpoints, reconstruct a Bits reader from the checkpoint state,
//! pre-populate the output buffer with the 32KB dictionary, and decode
//! blocks until the expected output size is reached. Each chunk writes
//! to a disjoint region of the output buffer via lock-free parallel writes.

use crate::consume_first_decode::Bits;
use crate::scan_inflate::{scan_deflate, ScanCheckpoint};
use std::io::{self, Error, ErrorKind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const MIN_SIZE_FOR_PARALLEL: usize = 4 * 1024 * 1024;

/// Decompress a single-member gzip file using two-pass parallel strategy.
///
/// Returns the decompressed data, or None if the data is too small to
/// benefit from parallelism (caller should fall back to sequential).
pub fn decompress_two_pass_parallel(
    gzip_data: &[u8],
    num_threads: usize,
) -> io::Result<Option<Vec<u8>>> {
    let header_size = parse_gzip_header_size(gzip_data)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid gzip header"))?;

    if gzip_data.len() < header_size + 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Gzip data too short",
        ));
    }

    let deflate_data = &gzip_data[header_size..gzip_data.len() - 8];

    // Read ISIZE from trailer (mod 2^32 of uncompressed size)
    let isize_hint = u32::from_le_bytes([
        gzip_data[gzip_data.len() - 4],
        gzip_data[gzip_data.len() - 3],
        gzip_data[gzip_data.len() - 2],
        gzip_data[gzip_data.len() - 1],
    ]) as usize;

    if isize_hint < MIN_SIZE_FOR_PARALLEL || num_threads <= 1 {
        return Ok(None);
    }

    let num_chunks = (num_threads * 2).max(4);
    let checkpoint_interval = isize_hint / num_chunks;
    if checkpoint_interval < 64 * 1024 {
        return Ok(None);
    }

    // === PASS 1: Scan for block boundaries and windows ===
    let scan_result = scan_deflate(deflate_data, checkpoint_interval, isize_hint)?;

    if scan_result.checkpoints.is_empty() {
        return Ok(None);
    }

    let total_output = scan_result.total_output_size;

    // Build chunk descriptors
    let chunks = build_chunks(deflate_data, &scan_result.checkpoints, total_output);

    if chunks.len() < 2 {
        return Ok(None);
    }

    // === PASS 2: Parallel decode between checkpoints ===
    let output = vec![0u8; total_output];
    let next_chunk = AtomicUsize::new(0);
    let had_error = AtomicBool::new(false);
    let chunk_count = chunks.len();

    struct OutputBuffer(std::cell::UnsafeCell<Vec<u8>>);
    unsafe impl Sync for OutputBuffer {}

    let output_cell = OutputBuffer(std::cell::UnsafeCell::new(output));

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(chunk_count) {
            let chunks_ref = &chunks;
            let next_ref = &next_chunk;
            let output_ref = &output_cell;
            let error_ref = &had_error;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= chunk_count || error_ref.load(Ordering::Relaxed) {
                        break;
                    }

                    let chunk = &chunks_ref[idx];

                    match decode_chunk(deflate_data, chunk) {
                        Ok(chunk_data) => {
                            // SAFETY: each chunk writes to a disjoint output region
                            let output_ptr = unsafe { (*output_ref.0.get()).as_mut_ptr() };
                            let out_slice = unsafe {
                                std::slice::from_raw_parts_mut(
                                    output_ptr.add(chunk.output_offset),
                                    chunk.expected_output_size,
                                )
                            };
                            let copy_len = chunk_data.len().min(chunk.expected_output_size);
                            out_slice[..copy_len].copy_from_slice(&chunk_data[..copy_len]);
                        }
                        Err(_) => {
                            error_ref.store(true, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });

    if had_error.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Two-pass parallel decode failed on one or more chunks",
        ));
    }

    Ok(Some(output_cell.0.into_inner()))
}

/// Decode a single chunk between two checkpoints.
///
/// Reconstructs the Bits reader from the checkpoint state, pre-populates
/// the working buffer with the dictionary, and decodes block-by-block until
/// the expected output size is reached.
fn decode_chunk(deflate_data: &[u8], chunk: &ChunkDescriptor) -> io::Result<Vec<u8>> {
    let dict_size = chunk.dictionary.len();
    // The last deflate block before our target can produce up to ~64KB of output,
    // plus the fastloop needs 320 bytes of margin. Use 256KB to be safe.
    let buf_size = dict_size + chunk.expected_output_size + 256 * 1024;

    let mut buf = vec![0u8; buf_size];

    // Pre-populate dictionary in the output buffer
    if dict_size > 0 {
        buf[..dict_size].copy_from_slice(&chunk.dictionary);
    }

    // Reconstruct Bits reader from checkpoint state
    let mut bits = Bits {
        data: deflate_data,
        pos: chunk.input_byte_pos,
        bitbuf: chunk.input_bitbuf,
        bitsleft: chunk.input_bitsleft,
    };

    let mut out_pos = dict_size;
    let target = dict_size + chunk.expected_output_size;

    loop {
        if out_pos >= target {
            break;
        }

        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        match btype {
            0 => {
                out_pos =
                    crate::consume_first_decode::decode_stored_pub(&mut bits, &mut buf, out_pos)?;
            }
            1 => {
                out_pos =
                    crate::consume_first_decode::decode_fixed_pub(&mut bits, &mut buf, out_pos)?;
            }
            2 => {
                out_pos =
                    crate::consume_first_decode::decode_dynamic_pub(&mut bits, &mut buf, out_pos)?;
            }
            3 => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type")),
            _ => unreachable!(),
        }

        if bfinal {
            break;
        }
    }

    // Return only the chunk data (skip the dictionary prefix)
    let actual_size = (out_pos - dict_size).min(chunk.expected_output_size);
    Ok(buf[dict_size..dict_size + actual_size].to_vec())
}

/// A chunk to be decoded in the parallel pass.
struct ChunkDescriptor {
    /// Dictionary (32KB window from scan, empty for first chunk)
    dictionary: Vec<u8>,
    /// Byte position in deflate_data
    input_byte_pos: usize,
    /// Bit buffer state
    input_bitbuf: u64,
    input_bitsleft: u32,
    /// Offset in the final output buffer
    output_offset: usize,
    /// Expected decompressed size for this chunk
    expected_output_size: usize,
}

fn build_chunks(
    _deflate_data: &[u8],
    checkpoints: &[ScanCheckpoint],
    total_output: usize,
) -> Vec<ChunkDescriptor> {
    let mut chunks = Vec::with_capacity(checkpoints.len() + 1);

    // First chunk: starts at bit 0, no dictionary
    let first_cp = &checkpoints[0];
    chunks.push(ChunkDescriptor {
        dictionary: Vec::new(),
        input_byte_pos: 0,
        input_bitbuf: 0,
        input_bitsleft: 0,
        output_offset: 0,
        expected_output_size: first_cp.output_offset,
    });

    // Middle chunks: between consecutive checkpoints
    for i in 0..checkpoints.len() - 1 {
        let cp = &checkpoints[i];
        let next_cp = &checkpoints[i + 1];
        chunks.push(ChunkDescriptor {
            dictionary: cp.window.clone(),
            input_byte_pos: cp.input_byte_pos,
            input_bitbuf: cp.bitbuf,
            input_bitsleft: cp.bitsleft,
            output_offset: cp.output_offset,
            expected_output_size: next_cp.output_offset - cp.output_offset,
        });
    }

    // Last chunk: from last checkpoint to end of stream
    let last_cp = &checkpoints[checkpoints.len() - 1];
    chunks.push(ChunkDescriptor {
        dictionary: last_cp.window.clone(),
        input_byte_pos: last_cp.input_byte_pos,
        input_bitbuf: last_cp.bitbuf,
        input_bitsleft: last_cp.bitsleft,
        output_offset: last_cp.output_offset,
        expected_output_size: total_output - last_cp.output_offset,
    });

    chunks
}

fn parse_gzip_header_size(data: &[u8]) -> Option<usize> {
    if data.len() < 10 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }

    let flags = data[3];
    let mut pos = 10;

    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return None;
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    if flags & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }
    if flags & 0x02 != 0 {
        pos += 2;
    }

    Some(pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_two_pass_basic() {
        let original = b"Hello, world! This is a test of two-pass parallel decompression. \
                         We need enough data to make it worthwhile. Let's repeat this many times. ";
        let mut big_data = Vec::new();
        for _ in 0..100_000 {
            big_data.extend_from_slice(original);
        }

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&big_data).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = decompress_two_pass_parallel(&compressed, 4).unwrap();
        assert!(result.is_some(), "Should use parallel path for large data");
        let output = result.unwrap();
        assert_eq!(output.len(), big_data.len());
        assert_eq!(output, big_data);
    }

    #[test]
    fn test_two_pass_small_data_returns_none() {
        let original = b"Too small for parallel";
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = decompress_two_pass_parallel(&compressed, 4).unwrap();
        assert!(result.is_none(), "Small data should return None");
    }

    #[test]
    fn test_two_pass_correctness_diverse_data() {
        // Mix of repetitive and random-ish data to stress dynamic Huffman.
        // Must be >4MB to trigger parallel path.
        let mut data = Vec::new();
        for i in 0u64..5_000_000 {
            let val = (i
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)) as u8;
            data.push(val);
            if i % 100 == 0 {
                data.extend_from_slice(b"AAAAAAAAAA");
            }
        }

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&data).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = decompress_two_pass_parallel(&compressed, 4).unwrap();
        assert!(result.is_some());
        let output = result.unwrap();
        assert_eq!(output.len(), data.len());
        assert_eq!(output, data, "Output mismatch on diverse data");
    }
}
