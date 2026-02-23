#![allow(dead_code)]
//! Pipelined Parallel Single-Member Decompression
//!
//! Architecture:
//!   1 scanner thread + (N-1) decoder threads, running concurrently.
//!
//!   Scanner: fast scan with circular 4MB buffer (stays in L2 cache).
//!   Produces checkpoints (bit position + 32KB window) as it goes.
//!   Sends checkpoints to decoder threads via channel.
//!
//!   Decoders: start as soon as a checkpoint arrives. Each decodes one
//!   chunk (from checkpoint to next checkpoint) with the 32KB dictionary.
//!
//!   This pipelines the scan with decode: while the scanner processes
//!   the second half of the file, decoders are already working on
//!   the first half's chunks.

use crate::consume_first_decode::Bits;
use crate::scan_inflate::ScanCheckpoint;
use std::io::{self, Error, ErrorKind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const MIN_SIZE_FOR_PARALLEL: usize = 4 * 1024 * 1024;

/// Decompress a single-member gzip file using pipelined parallel strategy.
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

    let isize_hint = u32::from_le_bytes([
        gzip_data[gzip_data.len() - 4],
        gzip_data[gzip_data.len() - 3],
        gzip_data[gzip_data.len() - 2],
        gzip_data[gzip_data.len() - 1],
    ]) as usize;

    if isize_hint < MIN_SIZE_FOR_PARALLEL || num_threads <= 1 {
        return Ok(None);
    }

    // Target: num_threads chunks for good parallelism
    let num_chunks = num_threads.max(4);
    let checkpoint_interval = isize_hint / num_chunks;
    if checkpoint_interval < 64 * 1024 {
        return Ok(None);
    }

    // On x86_64 with ISA-L, use ISA-L's AVX-accelerated inflate for the scan.
    // Falls back to pure-Rust scan on arm64 or if ISA-L scan fails.
    let scan_result =
        crate::isal_decompress::scan_deflate_isal(deflate_data, checkpoint_interval, isize_hint)
            .ok_or_else(|| io::Error::other("isal scan unavailable"))
            .or_else(|_| {
                crate::scan_inflate::scan_deflate_fast(
                    deflate_data,
                    checkpoint_interval,
                    isize_hint,
                )
            })
            .or_else(|_| {
                crate::scan_inflate::scan_deflate(deflate_data, checkpoint_interval, isize_hint)
            })?;

    if scan_result.checkpoints.is_empty() {
        return Ok(None);
    }

    let total_output = scan_result.total_output_size;
    let chunks = build_chunks(deflate_data, &scan_result.checkpoints, total_output);

    if chunks.len() < 2 {
        return Ok(None);
    }

    // Parallel decode between checkpoints
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

            scope.spawn(move || loop {
                let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                if idx >= chunk_count || error_ref.load(Ordering::Relaxed) {
                    break;
                }

                let chunk = &chunks_ref[idx];

                match decode_chunk(deflate_data, chunk) {
                    Ok(chunk_data) => {
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
            });
        }
    });

    if had_error.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Parallel decode failed on one or more chunks",
        ));
    }

    let output = output_cell.0.into_inner();

    // Verify CRC32 and ISIZE from gzip trailer
    verify_gzip_trailer(gzip_data, &output)?;

    Ok(Some(output))
}

/// Decode a single chunk between two checkpoints.
fn decode_chunk(deflate_data: &[u8], chunk: &ChunkDescriptor) -> io::Result<Vec<u8>> {
    let dict_size = chunk.dictionary.len();
    let buf_size = dict_size + chunk.expected_output_size + 256 * 1024;

    let mut buf = vec![0u8; buf_size];

    if dict_size > 0 {
        buf[..dict_size].copy_from_slice(&chunk.dictionary);
    }

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

    let actual_size = (out_pos - dict_size).min(chunk.expected_output_size);
    Ok(buf[dict_size..dict_size + actual_size].to_vec())
}

struct ChunkDescriptor {
    dictionary: Vec<u8>,
    input_byte_pos: usize,
    input_bitbuf: u64,
    input_bitsleft: u32,
    output_offset: usize,
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

/// Verify gzip trailer (CRC32 + ISIZE) matches the decompressed output.
fn verify_gzip_trailer(gzip_data: &[u8], output: &[u8]) -> io::Result<()> {
    if gzip_data.len() < 18 {
        return Err(Error::new(ErrorKind::InvalidData, "gzip data too short"));
    }

    let trailer = &gzip_data[gzip_data.len() - 8..];
    let expected_crc = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let expected_isize =
        u32::from_le_bytes([trailer[4], trailer[5], trailer[6], trailer[7]]) as usize;

    // ISIZE is mod 2^32
    let actual_isize = output.len() & 0xFFFF_FFFF;
    if actual_isize != expected_isize {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "ISIZE mismatch: expected {} but got {}",
                expected_isize, actual_isize
            ),
        ));
    }

    let actual_crc = crc32fast::hash(output);
    if actual_crc != expected_crc {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "CRC32 mismatch: expected {:#010x} but got {:#010x}",
                expected_crc, actual_crc
            ),
        ));
    }

    Ok(())
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

    #[test]
    fn test_two_pass_thread_counts() {
        let original = b"Thread count test data for parallel decompression. ";
        let mut big_data = Vec::new();
        for _ in 0..100_000 {
            big_data.extend_from_slice(original);
        }

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&big_data).unwrap();
        let compressed = encoder.finish().unwrap();

        for threads in [2, 4, 8, 16] {
            let result = decompress_two_pass_parallel(&compressed, threads).unwrap();
            if threads < 2 {
                assert!(
                    result.is_none(),
                    "Should return None for {} threads",
                    threads
                );
            } else if let Some(output) = result {
                assert_eq!(
                    output.len(),
                    big_data.len(),
                    "Size mismatch at {} threads",
                    threads
                );
                assert_eq!(output, big_data, "Content mismatch at {} threads", threads);
            }
        }
    }

    #[test]
    fn test_two_pass_chunk_boundaries_consistent() {
        let mut data = Vec::with_capacity(10 * 1024 * 1024);
        let mut rng: u64 = 42;
        while data.len() < 10 * 1024 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 32) as u8);
            if rng % 20 < 10 {
                let byte = ((rng >> 16) % 26 + b'a' as u64) as u8;
                for _ in 0..(rng % 10 + 2) as usize {
                    if data.len() < 10 * 1024 * 1024 {
                        data.push(byte);
                    }
                }
            }
        }
        data.truncate(10 * 1024 * 1024);

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&data).unwrap();
        let compressed = encoder.finish().unwrap();

        let header_size = parse_gzip_header_size(&compressed).unwrap();
        let deflate_data = &compressed[header_size..compressed.len() - 8];
        let isize_hint = u32::from_le_bytes([
            compressed[compressed.len() - 4],
            compressed[compressed.len() - 3],
            compressed[compressed.len() - 2],
            compressed[compressed.len() - 1],
        ]) as usize;

        let scan = crate::scan_inflate::scan_deflate_fast(deflate_data, isize_hint / 8, isize_hint)
            .expect("scan should succeed");

        let chunks = build_chunks(deflate_data, &scan.checkpoints, scan.total_output_size);

        // Verify chunks cover the entire output without gaps or overlaps
        let mut covered = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk.output_offset, covered,
                "chunk {} starts at {} but expected {}",
                i, chunk.output_offset, covered
            );
            covered += chunk.expected_output_size;
        }
        assert_eq!(
            covered, scan.total_output_size,
            "chunks don't cover entire output"
        );
    }

    #[test]
    fn test_two_pass_silesia_roundtrip() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };

        let mut ref_output = Vec::new();
        {
            let mut decoder = flate2::read::GzDecoder::new(&gz[..]);
            std::io::Read::read_to_end(&mut decoder, &mut ref_output).unwrap();
        }

        let result = decompress_two_pass_parallel(&gz, 4).unwrap();
        assert!(result.is_some(), "silesia should use parallel path");
        let output = result.unwrap();
        assert_eq!(output.len(), ref_output.len(), "silesia size mismatch");
        assert_eq!(output, ref_output, "silesia content mismatch");
    }

    #[test]
    fn test_two_pass_single_thread_returns_none() {
        let original = b"Single thread should not use parallel path. ";
        let mut big_data = Vec::new();
        for _ in 0..100_000 {
            big_data.extend_from_slice(original);
        }

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&big_data).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = decompress_two_pass_parallel(&compressed, 1).unwrap();
        assert!(result.is_none(), "Single thread should return None");
    }
}
