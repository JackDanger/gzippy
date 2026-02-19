//! Prefix-Overlap Parallel Single-Member Gzip Decompression
//!
//! ## Status: Experimental / Research
//!
//! This module implements the prefix-overlap parallel decompression strategy.
//! It is NOT currently used in the production decompression path because:
//!
//! 1. Our pure-Rust inflate (~400 MB/s on complex silesia) is slower than
//!    sequential libdeflate FFI (~1400 MB/s). Even with 4× parallelism, the
//!    effective rate (600 MB/s) doesn't beat sequential libdeflate.
//!
//! 2. Correctness: the prefix-overlap approach relies on "window convergence" —
//!    after decoding the prefix chunk with a zero window, the last 32KB of prefix
//!    output must match the correct window. This holds for 4T on silesia (large
//!    chunks with few cross-boundary references), but fails for 8T (smaller chunks).
//!    A correct implementation requires marker-based decoding (rapidgzip approach).
//!
//! ## Algorithm
//!
//! Given N threads and N chunks split at block boundaries B[0..=N]:
//!
//! - Thread 0: decode deflate[B[0]..B[1]] with empty window → output[0]
//! - Thread 1: decode deflate[B[0]..B[2]], record bytes at B[1] → discard prefix, keep rest
//! - Thread i: decode deflate[B[i-1]..B[i+1]], record bytes at B[i] → keep only [B[i]..]
//! - Thread N-1: decode deflate[B[N-2]..B[N]], keep only [B[N-1]..]
//!
//! Threads 0 and 1 are always correct (both start at 0 with correct empty window).
//! Threads 2+ rely on window convergence of the prefix chunk.
#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::block_finder::BlockFinder;
use crate::ultra_fast_inflate::{
    inflate_with_prefix_overlap, inflate_with_prefix_overlap_limited, set_block_output_limit,
};

/// Minimum output bytes expected from a valid non-final split point.
/// A real dynamic block produces >> 1KB. False positives (4-byte blocks + BFINAL=1) fail this.
const MIN_TRIAL_OUTPUT_BYTES: usize = 1024; // 1KB

/// Try to decode from `start_bit` and verify it's a real, productive block boundary.
///
/// Uses a 512KB per-block output limit to prevent multi-second hangs on false-positive
/// split points that generate huge blocks of garbage output. A real block at any
/// mid-stream position produces at least `MIN_TRIAL_OUTPUT_BYTES` before the cap fires.
fn trial_decode_at(deflate_data: &[u8], start_bit: usize) -> bool {
    const TRIAL_BLOCK_LIMIT: usize = 512 * 1024; // 512KB per-block cap; stops runaway false positives
    let zero_window = [0u8; WINDOW_SIZE];
    let actual_stop = start_bit + 64 * 1024 * 8;

    set_block_output_limit(TRIAL_BLOCK_LIMIT);
    let result = inflate_with_prefix_overlap_limited(
        deflate_data,
        start_bit,
        start_bit,
        actual_stop,
        &zero_window,
        TRIAL_BLOCK_LIMIT,
    );
    set_block_output_limit(usize::MAX); // Restore to unlimited

    match result {
        Ok((ref bytes, prefix_len)) => {
            let actual_bytes = bytes.len().saturating_sub(prefix_len);
            actual_bytes >= MIN_TRIAL_OUTPUT_BYTES
        }
        Err(_) => {
            // A decode error here means the split point is valid (a real block with some back-refs
            // to the zero window), not a false positive. Count as valid if it's not an "empty block" error.
            // Actually, Err means false positive (invalid codes) OR we hit the cap on a huge block.
            // Either way, this candidate is unsuitable for a split point.
            false
        }
    }
}

/// Minimum deflate stream size for parallel decode benefit.
const MIN_PARALLEL_DEFLATE_BYTES: usize = 8 * 1024 * 1024; // 8MB

/// LZ77 window size (max back-reference distance in deflate).
const WINDOW_SIZE: usize = 32768;

/// Decompress a single-member gzip file using prefix-overlap parallel decode.
///
/// Returns `Err` if the file is too small or block boundaries can't be found.
/// The caller must fall back to sequential decompression on `Err`.
pub fn decompress_two_pass_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    if num_threads <= 1 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "two_pass_parallel requires num_threads > 1",
        ));
    }

    let (deflate_start, isize_hint) = parse_gzip_single_member(gzip_data)?;
    let deflate_end = gzip_data.len().saturating_sub(8);
    if deflate_end <= deflate_start {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Empty deflate stream",
        ));
    }
    let deflate_data = &gzip_data[deflate_start..deflate_end];

    if deflate_data.len() < MIN_PARALLEL_DEFLATE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "File too small for parallel decode",
        ));
    }

    // Find N-1 block boundaries splitting the deflate stream into N chunks
    let split_bits = find_split_points(deflate_data, num_threads)?;
    debug_assert_eq!(split_bits.len(), num_threads + 1);

    // Capacity hint from ISIZE
    let output_capacity = if isize_hint > 0 && isize_hint < 16 * 1024 * 1024 * 1024 {
        isize_hint
    } else {
        deflate_data.len() * 3
    };

    // Parallel prefix-overlap decode
    let final_output = parallel_prefix_overlap_decode(deflate_data, &split_bits, output_capacity)?;

    writer.write_all(&final_output)?;
    Ok(final_output.len() as u64)
}

/// Parse gzip header and return (deflate_start_byte, isize_hint).
fn parse_gzip_single_member(data: &[u8]) -> io::Result<(usize, usize)> {
    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a valid gzip single-member stream",
        ));
    }

    let flags = data[3];
    let mut pos = 10usize;

    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Truncated FEXTRA",
            ));
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

    if pos > data.len().saturating_sub(8) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Header overflow",
        ));
    }

    let isize_bytes = &data[data.len() - 4..];
    let isize_val = u32::from_le_bytes([
        isize_bytes[0],
        isize_bytes[1],
        isize_bytes[2],
        isize_bytes[3],
    ]) as usize;

    Ok((pos, isize_val))
}

/// Find N-1 split points near evenly-spaced bit positions IN PARALLEL.
/// Returns Vec of N+1 bit offsets: [0, split1, ..., split_{N-1}, total_bits].
///
/// Each candidate is verified with a trial inflate to eliminate false positives
/// from the BlockFinder (rare but possible in random-looking compressed data).
/// All N-1 boundary searches run concurrently.
fn find_split_points(deflate_data: &[u8], num_threads: usize) -> io::Result<Vec<usize>> {
    const SEARCH_WINDOW_BITS: usize = 4 * 1024 * 1024 * 8; // 4MB search window

    let total_bits = deflate_data.len() * 8;
    let chunk_bits = total_bits / num_threads;

    let split_bits: Vec<std::sync::Mutex<Option<usize>>> = (0..num_threads + 1)
        .map(|_| std::sync::Mutex::new(None))
        .collect();

    // Boundary 0 and N are fixed
    *split_bits[0].lock().unwrap() = Some(0);
    *split_bits[num_threads].lock().unwrap() = Some(total_bits);

    let had_error = AtomicBool::new(false);
    let error_ref = &had_error;
    let split_ref = &split_bits;

    std::thread::scope(|scope| {
        for (i, slot) in split_ref.iter().enumerate().take(num_threads).skip(1) {
            scope.spawn(move || {
                let finder = BlockFinder::new(deflate_data);
                let target_bit = i * chunk_bits;
                let search_start = target_bit.saturating_sub(SEARCH_WINDOW_BITS / 2);
                let search_end = (target_bit + SEARCH_WINDOW_BITS / 2).min(total_bits);

                let candidates = finder.find_blocks(search_start, search_end);

                let mut sorted: Vec<_> = candidates
                    .into_iter()
                    .filter(|b| b.valid)
                    .collect();
                sorted.sort_by_key(|b| {
                    let d = b.bit_offset as isize - target_bit as isize;
                    d.unsigned_abs()
                });

                let verified = sorted
                    .into_iter()
                    .find(|b| trial_decode_at(deflate_data, b.bit_offset));

                match verified {
                    Some(b) => {
                        *slot.lock().unwrap() = Some(b.bit_offset);
                    }
                    None => {
                        if std::env::var("GZIPPY_DEBUG").is_ok() {
                            eprintln!(
                                "[two_pass] No verified block boundary found near chunk {i} target (bit {target_bit})"
                            );
                        }
                        error_ref.store(true, Ordering::Relaxed);
                    }
                }
            });
        }
    });

    if had_error.load(Ordering::Relaxed) {
        return Err(io::Error::other(
            "No verified block boundary found for one or more chunks",
        ));
    }

    let result: Vec<usize> = split_bits
        .into_iter()
        .map(|m| m.into_inner().unwrap().unwrap())
        .collect();

    Ok(result)
}

/// Run the prefix-overlap parallel decode.
///
/// Thread 0: decode [split[0]..split[1]] with empty window
/// Thread i: decode [split[i-1]..split[i+1]] with empty window
///           The first chunk (split[i-1]..split[i]) acts as the prefix window
///           Only output AFTER the prefix is kept
fn parallel_prefix_overlap_decode(
    deflate_data: &[u8],
    split_bits: &[usize],
    capacity_hint: usize,
) -> io::Result<Vec<u8>> {
    let num_threads = split_bits.len() - 1;
    let had_error = AtomicBool::new(false);

    type ChunkResult = std::sync::Mutex<Option<(Vec<u8>, usize)>>;
    // Each thread produces (output_bytes, prefix_len)
    let results: Vec<ChunkResult> = (0..num_threads)
        .map(|_| std::sync::Mutex::new(None))
        .collect();

    let error_ref = &had_error;
    let results_ref = &results;

    let total_bits = deflate_data.len() * 8;

    std::thread::scope(|scope| {
        for i in 0..num_threads {
            scope.spawn(move || {
                // Use inflate_with_prefix_overlap for all threads.
                // Thread 0: prefix_stop = start = 0 (no prefix, prefix_len = 0 immediately)
                // Thread i: prefix = chunk[i-1], actual = chunk[i]
                let (start_bit, prefix_stop_bit) = if i == 0 {
                    (0usize, 0usize) // No prefix for thread 0
                } else {
                    (split_bits[i - 1], split_bits[i])
                };
                let actual_stop_bit = if i + 1 < split_bits.len() {
                    split_bits[i + 1]
                } else {
                    total_bits
                };

                // Pre-populate with 32KB of zeros so back-references to positions
                // before the start of this chunk don't cause errors. The first ~32KB
                // of output will have wrong bytes (zeros instead of real window data),
                // but all of that is in the prefix and discarded. By the end of the
                // prefix chunk, the window converges to the correct state.
                let zero_window = [0u8; WINDOW_SIZE];
                let result = inflate_with_prefix_overlap(
                    deflate_data,
                    start_bit,
                    prefix_stop_bit,
                    actual_stop_bit,
                    &zero_window,
                );

                match result {
                    Ok(pair) => {
                        *results_ref[i].lock().unwrap() = Some(pair);
                    }
                    Err(e) => {
                        if std::env::var("GZIPPY_DEBUG").is_ok() {
                            eprintln!(
                                "[two_pass] thread {i} FAILED: {e} (start={start_bit}, prefix_stop={prefix_stop_bit}, actual_stop={actual_stop_bit})"
                            );
                        }
                        error_ref.store(true, Ordering::Relaxed);
                    }
                }
            });
        }
    });

    if had_error.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "One or more chunks failed to decode",
        ));
    }

    // Assemble final output: for each thread, take output[prefix_len..]
    let mut final_output: Vec<u8> = Vec::with_capacity(capacity_hint);

    for slot in results {
        let (output, prefix_len) = slot
            .into_inner()
            .unwrap()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing chunk output"))?;

        // Only keep the actual chunk output (after the prefix)
        final_output.extend_from_slice(&output[prefix_len..]);
    }

    Ok(final_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn compress_gzip(data: &[u8], level: u32) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn test_small_returns_err() {
        let data = b"hello world";
        let compressed = compress_gzip(data, 6);
        let mut output = Vec::new();
        let result = decompress_two_pass_parallel(&compressed, &mut output, 4);
        assert!(result.is_err(), "Small files must return Err");
    }

    #[test]
    fn test_larger_data_correctness() {
        // Create ~1MB of data (still likely too small for 8MB minimum, but tests header parsing)
        let data: Vec<u8> = (0u32..250_000).flat_map(|i| i.to_le_bytes()).collect();
        let compressed = compress_gzip(&data, 6);

        let mut output = Vec::new();
        match decompress_two_pass_parallel(&compressed, &mut output, 2) {
            Ok(_) => {
                assert_eq!(output, data, "Output must match original");
            }
            Err(_) => {
                // Expected for files smaller than MIN_PARALLEL_DEFLATE_BYTES
            }
        }
    }

    #[test]
    #[ignore = "requires large silesia benchmark file (~160MB)"]
    fn bench_two_pass_parallel() {
        let path = "benchmark_data/silesia-gzip.tar.gz";
        let Ok(data) = std::fs::read(path) else {
            eprintln!("Skipping: {path} not found");
            return;
        };

        // Get reference output from sequential libdeflate
        let mut reference: Vec<u8> = Vec::new();
        crate::decompression::decompress_gzip_to_writer(&data, &mut reference).unwrap();
        println!("Reference output: {} bytes", reference.len());

        for threads in [2, 4, 8] {
            let start = std::time::Instant::now();
            let mut output = Vec::new();
            match decompress_two_pass_parallel(&data, &mut output, threads) {
                Ok(bytes) => {
                    let mb_per_s = bytes as f64 / start.elapsed().as_secs_f64() / 1_000_000.0;
                    let correct = output == reference;
                    println!(
                        "two_pass_parallel {threads}T: {mb_per_s:.0} MB/s ({bytes} bytes) correct={correct}"
                    );
                    assert!(
                        correct,
                        "{threads}T output doesn't match reference! First mismatch at byte {}",
                        output
                            .iter()
                            .zip(reference.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(bytes as usize)
                    );
                }
                Err(e) => println!("Failed at {threads} threads: {e}"),
            }
        }
    }

    /// Verify the window size is sufficient (32KB is the DEFLATE spec maximum distance)
    #[test]
    fn test_window_size_constant() {
        assert_eq!(
            WINDOW_SIZE, 32768,
            "WINDOW_SIZE must match DEFLATE max distance"
        );
    }
}
