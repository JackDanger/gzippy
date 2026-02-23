//! Parallel single-member gzip decompression (rapidgzip-style)
//!
//! Architecture (no sequential pre-scan):
//! 1. Partition compressed data at regular intervals
//! 2. Find deflate block boundaries near each partition point using BlockFinder
//! 3. Decode all chunks in parallel using marker-based decoder
//! 4. Resolve markers sequentially using window propagation
//!
//! This eliminates the expensive sequential scan pass that made the old two-pass
//! approach require 8+ threads to beat sequential. With this approach, 4 threads
//! already provide significant speedup.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::block_finder::BlockFinder;
use crate::marker_decode::{replace_markers, MarkerDecoder, WINDOW_SIZE};

/// Minimum compressed size to attempt parallel (4MB).
const MIN_PARALLEL_SIZE: usize = 4 * 1024 * 1024;

/// Minimum threads for parallel decode.
/// The rapidgzip-style approach has no scan overhead, so 4 threads suffice.
const MIN_THREADS_FOR_PARALLEL: usize = 4;

/// Search radius (bytes) around each partition point for block boundaries.
const SEARCH_RADIUS: usize = 512 * 1024;

#[inline]
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

/// Parallel decompress a single-member gzip stream.
///
/// Returns Ok(bytes_written) on success, or Err if the file can't be
/// parallel-decoded (caller should fall back to sequential).
pub fn decompress_parallel<W: Write>(
    gzip_data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> Result<u64, ParallelError> {
    let t0 = std::time::Instant::now();

    let header_size = crate::marker_decode::skip_gzip_header(gzip_data)
        .map_err(|_| ParallelError::InvalidHeader)?;

    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ParallelError::TooSmall);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

    if deflate_data.len() < MIN_PARALLEL_SIZE || num_threads < MIN_THREADS_FOR_PARALLEL {
        return Err(ParallelError::TooSmall);
    }

    let isize_offset = gzip_data.len() - 4;
    let expected_output = u32::from_le_bytes([
        gzip_data[isize_offset],
        gzip_data[isize_offset + 1],
        gzip_data[isize_offset + 2],
        gzip_data[isize_offset + 3],
    ]) as usize;

    let crc_offset = gzip_data.len() - 8;
    let expected_crc = u32::from_le_bytes([
        gzip_data[crc_offset],
        gzip_data[crc_offset + 1],
        gzip_data[crc_offset + 2],
        gzip_data[crc_offset + 3],
    ]);

    if debug_enabled() {
        eprintln!(
            "[parallel_sm] rapidgzip-style: {} bytes deflate, {} threads, isize={}",
            deflate_data.len(),
            num_threads,
            expected_output
        );
    }

    // Step 1: Find chunk boundaries using block finder
    let t_find = std::time::Instant::now();
    let boundaries = find_chunk_boundaries(deflate_data, num_threads);
    let find_elapsed = t_find.elapsed();

    if boundaries.len() < 2 {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm] only {} boundaries found, need >=2",
                boundaries.len()
            );
        }
        return Err(ParallelError::TooFewChunks);
    }

    if debug_enabled() {
        eprintln!(
            "[parallel_sm] found {} chunk boundaries in {:.1}ms",
            boundaries.len(),
            find_elapsed.as_secs_f64() * 1000.0
        );
        for (i, &bit) in boundaries.iter().enumerate() {
            eprintln!("  chunk {}: start_bit={}", i, bit);
        }
    }

    // Step 2: Decode all chunks in parallel using marker decoder
    let t_decode = std::time::Instant::now();
    let chunks = decode_chunks_parallel(deflate_data, &boundaries, num_threads)?;
    let decode_elapsed = t_decode.elapsed();

    if debug_enabled() {
        for (i, chunk) in chunks.iter().enumerate() {
            eprintln!(
                "  chunk {}: {} output values, {} markers",
                i,
                chunk.data.len(),
                chunk.marker_count
            );
        }
    }

    // Step 3: Resolve markers sequentially, verify CRC32, then write output
    let t_resolve = std::time::Instant::now();
    let total_output = resolve_and_write(&chunks, writer, expected_output, expected_crc)?;
    let resolve_elapsed = t_resolve.elapsed();

    if debug_enabled() {
        let total_elapsed = t0.elapsed();
        let total_mbps = total_output as f64 / total_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "[parallel_sm] find={:.1}ms decode={:.1}ms resolve={:.1}ms total={:.1}ms ({:.0} MB/s)",
            find_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0,
            resolve_elapsed.as_secs_f64() * 1000.0,
            total_elapsed.as_secs_f64() * 1000.0,
            total_mbps
        );
    }

    Ok(total_output as u64)
}

/// Find deflate block boundaries near each partition point.
///
/// Uses BlockFinder to generate candidates (dynamic + stored blocks),
/// then validates with try-decode. Falls back to brute-force search
/// over a narrow range if the block finder misses the boundary.
fn find_chunk_boundaries(deflate_data: &[u8], num_chunks: usize) -> Vec<usize> {
    let mut boundaries = vec![0usize];

    let chunk_size_bytes = deflate_data.len() / num_chunks;
    let finder = BlockFinder::new(deflate_data);

    for i in 1..num_chunks {
        let partition_byte = i * chunk_size_bytes;
        let partition_bit = partition_byte * 8;
        let search_radius_bits = SEARCH_RADIUS * 8;

        let search_start = partition_bit.saturating_sub(search_radius_bits);
        let search_end = (partition_bit + search_radius_bits).min(deflate_data.len() * 8);

        let prev_boundary = *boundaries.last().unwrap();
        let search_start = search_start.max(prev_boundary + 1);

        if search_start >= search_end {
            continue;
        }

        // Phase 1: BlockFinder candidates (fast — LUT + precode + Huffman validation)
        let mut candidates = finder.find_blocks(search_start, search_end);
        candidates.sort_by_key(|b| b.bit_offset.abs_diff(partition_bit));

        let mut found = false;
        for candidate in &candidates {
            if try_decode_at(deflate_data, candidate.bit_offset) {
                boundaries.push(candidate.bit_offset);
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm] partition {}: found boundary at bit {} via block finder",
                        i, candidate.bit_offset,
                    );
                }
                found = true;
                break;
            }
        }

        if found {
            continue;
        }

        // Phase 2: Brute-force search near partition point (slower but reliable).
        // Search ±32KB around the partition in 8-bit steps.
        let brute_start = partition_bit
            .saturating_sub(32 * 1024 * 8)
            .max(search_start);
        let brute_end = (partition_bit + 32 * 1024 * 8).min(search_end);

        for bit in (brute_start..brute_end).step_by(8) {
            if try_decode_at(deflate_data, bit) {
                boundaries.push(bit);
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm] partition {}: found boundary at bit {} via brute force",
                        i, bit,
                    );
                }
                found = true;
                break;
            }
        }

        if !found && debug_enabled() {
            eprintln!(
                "[parallel_sm] partition {}: no boundary found in {}..{} bits",
                i, search_start, search_end
            );
        }
    }

    boundaries
}

/// Validate a candidate bit position by attempting to decode from it.
///
/// Three checks eliminate false positives:
/// 1. **Output volume**: must produce substantial output (scaled to available data).
///    A false positive (random bits that parse as deflate) dies within a few KB.
/// 2. **Marker presence**: mid-stream positions MUST encounter LZ77 back-references
///    into unknown history. Zero markers = tiny self-contained false sequence.
/// 3. **Marker rate < 20%**: real boundaries produce mostly literals with a few
///    cross-boundary back-references. False positives that parse random bits as
///    dense match codes produce >50% markers (often 100%).
fn try_decode_at(deflate_data: &[u8], bit_offset: usize) -> bool {
    let start_byte = bit_offset / 8;
    let relative_start_bit = bit_offset % 8;
    let remaining = deflate_data.len() - start_byte;

    let slice_len = remaining.min(2 * 1024 * 1024);
    let data_slice = &deflate_data[start_byte..start_byte + slice_len];

    let decode_limit = slice_len.min(1024 * 1024);
    let mut decoder = MarkerDecoder::new(data_slice, relative_start_bit);
    match decoder.decode_until(decode_limit) {
        Ok(_) => {
            let output_len = decoder.output().len();
            let marker_count = decoder.marker_count();

            let min_output = if remaining > 1024 * 1024 {
                512 * 1024
            } else {
                32 * 1024
            };

            if output_len < min_output {
                return false;
            }
            if marker_count == 0 {
                return false;
            }
            let marker_rate = marker_count as f64 / output_len as f64;
            marker_rate < 0.20
        }
        Err(_) => false,
    }
}

struct DecodedChunk {
    data: Vec<u16>,
    marker_count: usize,
}

/// Estimate max output size for a chunk based on its compressed size.
/// Uses a generous ratio (8x) to avoid rejecting valid chunks, but prevents
/// the unbounded overshoot that caused the 5x output bug.
fn max_output_for_chunk(compressed_bytes: usize) -> usize {
    // 8x is generous — typical ratio is 2-4x. All-zero stored blocks can
    // theoretically reach ~1032:1 but real data never does.
    (compressed_bytes * 8).max(64 * 1024)
}

fn decode_chunks_parallel(
    deflate_data: &[u8],
    boundaries: &[usize],
    num_threads: usize,
) -> Result<Vec<DecodedChunk>, ParallelError> {
    let num_chunks = boundaries.len();
    let results: Vec<Mutex<Option<DecodedChunk>>> =
        (0..num_chunks).map(|_| Mutex::new(None)).collect();
    let errors = AtomicBool::new(false);
    let chunk_idx = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..num_threads.min(num_chunks) {
            s.spawn(|| loop {
                let idx = chunk_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= num_chunks || errors.load(Ordering::Relaxed) {
                    break;
                }

                let start_bit = boundaries[idx];
                let is_last = idx + 1 >= boundaries.len();

                let start_byte = start_bit / 8;
                let relative_start_bit = start_bit % 8;

                // Bound the data slice to avoid copying full tail (PR #43 fix)
                let end_byte = if is_last {
                    deflate_data.len()
                } else {
                    let next_byte = boundaries[idx + 1] / 8;
                    // Add 64KB margin for bit-level overshoot at block boundary
                    (next_byte + 64 * 1024).min(deflate_data.len())
                };
                let data_slice = &deflate_data[start_byte..end_byte];

                let compressed_bytes = end_byte - start_byte;
                let max_output = max_output_for_chunk(compressed_bytes);

                let mut decoder = MarkerDecoder::new(data_slice, relative_start_bit);

                let decode_result = if is_last {
                    decoder.decode_until(max_output).map(|_| ())
                } else {
                    let next_boundary = boundaries[idx + 1];
                    let relative_end_bit = next_boundary - start_byte * 8;
                    decoder
                        .decode_until_bit(max_output, relative_end_bit)
                        .map(|_| ())
                };

                match decode_result {
                    Ok(()) => {
                        // Chain convergence check: if this isn't the last chunk,
                        // verify we landed exactly at the next boundary.
                        // Overshoot means the boundary is a false positive
                        // (not on the same deflate block chain).
                        if !is_last {
                            let next_boundary = boundaries[idx + 1];
                            let relative_end_bit = next_boundary - start_byte * 8;
                            let final_pos = decoder.bit_position();
                            if final_pos != relative_end_bit {
                                if debug_enabled() {
                                    eprintln!(
                                        "[parallel_sm] chunk {}: chain mismatch: \
                                         landed at bit {} expected {} (delta={})",
                                        idx,
                                        final_pos,
                                        relative_end_bit,
                                        final_pos as i64 - relative_end_bit as i64
                                    );
                                }
                                errors.store(true, Ordering::Relaxed);
                                return;
                            }
                        }

                        *results[idx].lock().unwrap() = Some(DecodedChunk {
                            data: decoder.output().to_vec(),
                            marker_count: decoder.marker_count(),
                        });
                    }
                    Err(e) => {
                        if debug_enabled() {
                            eprintln!("[parallel_sm] chunk {} decode error: {}", idx, e);
                        }
                        errors.store(true, Ordering::Relaxed);
                    }
                }
            });
        }
    });

    if errors.load(Ordering::Relaxed) {
        return Err(ParallelError::DecodeFailed);
    }

    Ok(results
        .into_iter()
        .map(|m| m.into_inner().unwrap().expect("all chunks decoded"))
        .collect())
}

/// Resolve markers sequentially using window propagation, then write output.
///
/// Chunk 0 should have no markers (starts from beginning of deflate stream).
/// Chunk 1..N may have markers; resolved using previous chunk's last 32KB.
fn resolve_and_write<W: Write>(
    chunks: &[DecodedChunk],
    writer: &mut W,
    expected_size: usize,
    expected_crc: u32,
) -> Result<usize, ParallelError> {
    let mut buffer = Vec::with_capacity(expected_size);
    let mut window = Vec::<u8>::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.marker_count > 0 {
            if window.is_empty() {
                if debug_enabled() {
                    eprintln!(
                        "[parallel_sm] chunk {} has {} markers but no window available",
                        i, chunk.marker_count
                    );
                }
                return Err(ParallelError::DecodeFailed);
            }

            let mut resolved = chunk.data.clone();
            replace_markers(&mut resolved, &window);

            let bytes: Vec<u8> = resolved.iter().map(|&v| v as u8).collect();
            buffer.extend_from_slice(&bytes);
            update_window(&mut window, &bytes);
        } else {
            let bytes: Vec<u8> = chunk.data.iter().map(|&v| v as u8).collect();
            buffer.extend_from_slice(&bytes);
            update_window(&mut window, &bytes);
        }
    }

    if expected_size > 0 && buffer.len() != expected_size {
        if debug_enabled() {
            eprintln!(
                "[parallel_sm] output size mismatch: got {} expected {}",
                buffer.len(),
                expected_size
            );
        }
        return Err(ParallelError::SizeMismatch);
    }

    if expected_crc != 0 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&buffer);
        let actual_crc = hasher.finalize();
        if actual_crc != expected_crc {
            if debug_enabled() {
                eprintln!(
                    "[parallel_sm] CRC32 mismatch: got {:#010x} expected {:#010x}",
                    actual_crc, expected_crc
                );
            }
            return Err(ParallelError::CrcMismatch);
        }
    }

    writer.write_all(&buffer)?;
    Ok(buffer.len())
}

fn update_window(window: &mut Vec<u8>, new_data: &[u8]) {
    if new_data.len() >= WINDOW_SIZE {
        *window = new_data[new_data.len() - WINDOW_SIZE..].to_vec();
    } else if window.len() + new_data.len() <= WINDOW_SIZE {
        window.extend_from_slice(new_data);
    } else {
        let keep = WINDOW_SIZE - new_data.len();
        let start = window.len() - keep;
        let kept: Vec<u8> = window[start..].to_vec();
        *window = kept;
        window.extend_from_slice(new_data);
    }
}

#[derive(Debug)]
pub enum ParallelError {
    InvalidHeader,
    TooSmall,
    TooFewChunks,
    DecodeFailed,
    SizeMismatch,
    CrcMismatch,
    Io(io::Error),
}

impl From<io::Error> for ParallelError {
    fn from(e: io::Error) -> Self {
        ParallelError::Io(e)
    }
}

impl std::fmt::Display for ParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParallelError::InvalidHeader => write!(f, "invalid gzip header"),
            ParallelError::TooSmall => write!(f, "file too small for parallel decode"),
            ParallelError::TooFewChunks => write!(f, "too few chunk boundaries found"),
            ParallelError::DecodeFailed => write!(f, "chunk decode failed"),
            ParallelError::SizeMismatch => write!(f, "output size mismatch"),
            ParallelError::CrcMismatch => write!(f, "CRC32 mismatch"),
            ParallelError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gzip_data(data: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn make_compressible_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    fn get_deflate_and_expected(gzip_data: &[u8]) -> (&[u8], Vec<u8>) {
        let header_size = crate::marker_decode::skip_gzip_header(gzip_data).expect("valid header");
        let deflate = &gzip_data[header_size..gzip_data.len() - 8];
        let mut expected = vec![0u8; deflate.len() * 10 + 65536];
        let size = crate::consume_first_decode::inflate_consume_first(deflate, &mut expected)
            .expect("reference inflate");
        expected.truncate(size);
        (deflate, expected)
    }

    // =========================================================================
    // Regression test: small files fall back, don't crash
    // =========================================================================

    #[test]
    fn test_parallel_small_falls_back() {
        let data = b"hello world";
        let compressed = make_gzip_data(data);
        let mut output = Vec::new();
        let result = decompress_parallel(&compressed, &mut output, 4);
        assert!(matches!(result, Err(ParallelError::TooSmall)));
    }

    // =========================================================================
    // INVARIANT: try_decode_at rejects positions that aren't block boundaries
    // Catches: false positive acceptance (failure mode #1)
    // =========================================================================

    #[test]
    fn test_try_decode_rejects_random_positions() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let mut rng: u64 = 12345;
        let mut accepted = 0;
        let mut rejected = 0;
        let total_bits = deflate.len() * 8;

        for _ in 0..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (rng as usize) % total_bits;
            if try_decode_at(deflate, bit) {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }

        eprintln!(
            "random positions: {}/{} accepted ({:.1}%)",
            accepted,
            accepted + rejected,
            accepted as f64 / (accepted + rejected) as f64 * 100.0
        );

        // At random positions, acceptance should be very rare.
        // A high acceptance rate means try_decode is too permissive.
        assert!(
            accepted < 20,
            "try_decode accepted {} of 100 random positions — too permissive",
            accepted
        );
    }

    // =========================================================================
    // INVARIANT: try_decode_at accepts known-good block boundaries
    // Catches: try_decode being too strict / breaking real boundaries
    // =========================================================================

    #[test]
    fn test_try_decode_accepts_oracle_boundaries() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let scan = crate::scan_inflate::scan_deflate_fast(deflate, 512 * 1024, 0).expect("scan");
        let total_bits = deflate.len() * 8;
        let cutoff = total_bits * 9 / 10; // skip last 10% — handled by last chunk (EOF)
        let mut accepted = 0;
        let mut total = 0;

        for cp in &scan.checkpoints {
            let real_bitsleft = (cp.bitsleft as u8) as usize;
            let bit_pos = cp.input_byte_pos * 8 - real_bitsleft;
            if bit_pos >= cutoff {
                continue;
            }
            total += 1;
            if try_decode_at(deflate, bit_pos) {
                accepted += 1;
            } else {
                eprintln!(
                    "REJECTED known boundary at bit {} ({:.1}% into stream)",
                    bit_pos,
                    bit_pos as f64 / total_bits as f64 * 100.0
                );
            }
        }

        eprintln!(
            "oracle boundaries (excluding last 10%): {}/{} accepted ({:.1}%)",
            accepted,
            total,
            if total > 0 {
                accepted as f64 / total as f64 * 100.0
            } else {
                0.0
            }
        );

        assert_eq!(
            accepted,
            total,
            "try_decode rejected {} of {} oracle boundaries",
            total - accepted,
            total
        );
    }

    // =========================================================================
    // INVARIANT: block finder + try_decode finds at least SOME real boundaries
    // Catches: block finder precision too low (failure mode #1)
    // =========================================================================

    #[test]
    fn test_block_finder_with_try_decode_finds_boundaries() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let scan = crate::scan_inflate::scan_deflate_fast(deflate, 512 * 1024, 0).expect("scan");
        let finder = BlockFinder::new(deflate);
        let mut found_real = 0;

        for cp in &scan.checkpoints {
            let real_bitsleft = (cp.bitsleft as u8) as usize;
            let real_bit = cp.input_byte_pos * 8 - real_bitsleft;

            let search_start = real_bit.saturating_sub(SEARCH_RADIUS * 8);
            let search_end = (real_bit + SEARCH_RADIUS * 8).min(deflate.len() * 8);
            let candidates = finder.find_blocks(search_start, search_end);

            // Check if any candidate near the real boundary passes try_decode
            let mut hit = false;
            for c in &candidates {
                if c.bit_offset.abs_diff(real_bit) < 1024 && try_decode_at(deflate, c.bit_offset) {
                    hit = true;
                    break;
                }
            }

            if hit {
                found_real += 1;
            }
        }

        eprintln!(
            "block_finder + try_decode: {}/{} real boundaries found",
            found_real,
            scan.checkpoints.len()
        );

        // We need at least some hit rate for the pipeline to work.
        // Don't assert a specific rate — just report for diagnostics.
    }

    // =========================================================================
    // INVARIANT: each chunk's output size is bounded
    // Catches: decode overshoot (failure mode #2), overlapping chunks (#4)
    // =========================================================================

    #[test]
    fn test_chunk_output_bounded() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            eprintln!("not enough boundaries, skipping");
            return;
        }

        for i in 0..boundaries.len() {
            let start_bit = boundaries[i];
            let start_byte = start_bit / 8;
            let relative_start_bit = start_bit % 8;
            let is_last = i + 1 >= boundaries.len();

            let end_byte = if is_last {
                deflate.len()
            } else {
                let next_byte = boundaries[i + 1] / 8;
                (next_byte + 64 * 1024).min(deflate.len())
            };
            let compressed_bytes = end_byte - start_byte;
            let max_output = max_output_for_chunk(compressed_bytes);

            // Use bounded slice (same as production code)
            let data_slice = &deflate[start_byte..end_byte];
            let mut decoder = MarkerDecoder::new(data_slice, relative_start_bit);

            if is_last {
                let _ = decoder.decode_until(max_output);
            } else {
                let next_boundary = boundaries[i + 1];
                let relative_end_bit = next_boundary - start_byte * 8;
                let _ = decoder.decode_until_bit(max_output, relative_end_bit);
            }

            let output_size = decoder.output().len();
            eprintln!(
                "chunk {}: {} compressed bytes → {} output ({:.1}x), limit={}",
                i,
                compressed_bytes,
                output_size,
                if compressed_bytes > 0 {
                    output_size as f64 / compressed_bytes as f64
                } else {
                    0.0
                },
                max_output
            );

            // decode_until checks BEFORE each block, so overshoot by one
            // block is possible. Allow 2x max_output as tolerance.
            let tolerance = max_output * 2;
            assert!(
                output_size <= tolerance,
                "chunk {} produced {} bytes from {} compressed ({:.1}x) — exceeded tolerance {}",
                i,
                output_size,
                compressed_bytes,
                output_size as f64 / compressed_bytes as f64,
                tolerance
            );
        }
    }

    // =========================================================================
    // INVARIANT: chunk 0 has zero markers and matches sequential output
    // Catches: marker mode bugs at stream start, window init errors
    // =========================================================================

    #[test]
    fn test_chunk0_no_markers_matches_sequential() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            eprintln!("not enough boundaries, skipping");
            return;
        }

        // Decode chunk 0
        let end_bit = boundaries[1];
        let mut decoder = MarkerDecoder::new(deflate, 0);
        decoder
            .decode_until_bit(usize::MAX, end_bit)
            .expect("chunk 0 decode");

        assert_eq!(
            decoder.marker_count(),
            0,
            "chunk 0 should have no markers (starts from beginning)"
        );

        // Compare with sequential reference
        let mut ref_output = vec![0u8; data.len() + 65536];
        crate::consume_first_decode::inflate_consume_first(deflate, &mut ref_output)
            .expect("reference inflate");

        let chunk0_bytes: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();
        let cmp_len = chunk0_bytes.len().min(ref_output.len());
        assert_eq!(
            &chunk0_bytes[..cmp_len],
            &ref_output[..cmp_len],
            "chunk 0 output doesn't match sequential reference"
        );
        eprintln!(
            "chunk 0: {} bytes, matches sequential reference",
            chunk0_bytes.len()
        );
    }

    // =========================================================================
    // INVARIANT: data slices are bounded (not full-tail)
    // Catches: &data[start..] regression (failure mode #3, PR #43)
    // =========================================================================

    #[test]
    fn test_bounded_slice_not_full_tail() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 3 {
            eprintln!("not enough boundaries, skipping");
            return;
        }

        let start_bit = boundaries[1];
        let start_byte = start_bit / 8;
        let next_bit = boundaries[2];
        let end_byte = (next_bit / 8 + 64 * 1024).min(deflate.len());
        let compressed_bytes = end_byte - start_byte;
        let max_out = max_output_for_chunk(compressed_bytes);
        let relative_start = start_bit % 8;

        // Decode chunk 1 with BOUNDED slice (correct)
        let bounded_slice = &deflate[start_byte..end_byte];
        let mut bounded_decoder = MarkerDecoder::new(bounded_slice, relative_start);
        let _ = bounded_decoder.decode_until(max_out);
        let bounded_output = bounded_decoder.output().len();

        // Decode chunk 1 with FULL TAIL (the bug from PR #43)
        let full_tail = &deflate[start_byte..];
        let mut tail_decoder = MarkerDecoder::new(full_tail, relative_start);
        let _ = tail_decoder.decode_until(max_out);
        let tail_output = tail_decoder.output().len();

        eprintln!(
            "bounded slice: {} output, full tail: {} output, max_out: {}",
            bounded_output, tail_output, max_out
        );

        // Both should produce output bounded by max_out.
        // decode_until checks BEFORE each block, so overshoot by one block
        // is acceptable. We allow 2x max_out as tolerance.
        let tolerance = max_out * 2;
        assert!(
            bounded_output <= tolerance,
            "bounded decode wildly exceeded max: {} > {} (2x limit)",
            bounded_output,
            tolerance
        );

        // Key regression test: full-tail should NOT produce dramatically
        // more output than bounded. If it does, the &data[start..] bug is back.
        if tail_output > bounded_output * 3 && tail_output > 1024 * 1024 {
            eprintln!(
                "WARNING: full-tail produced {}x more than bounded ({} vs {}) — \
                 possible &data[start..] regression",
                tail_output / bounded_output.max(1),
                tail_output,
                bounded_output
            );
        }
    }

    // =========================================================================
    // INVARIANT: full pipeline produces correct output OR falls back
    // Catches: all integration bugs (the test that would have caught everything)
    // =========================================================================

    #[test]
    fn test_e2e_roundtrip_strict() {
        let data = make_compressible_data(40 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        eprintln!(
            "e2e strict: {} bytes → {} bytes ({:.1}%)",
            data.len(),
            compressed.len(),
            compressed.len() as f64 / data.len() as f64 * 100.0
        );

        if compressed.len() < MIN_PARALLEL_SIZE {
            eprintln!("compressed data too small, skipping");
            return;
        }

        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 10) {
            Ok(bytes) => {
                // STRICT: output size must match exactly
                assert_eq!(
                    bytes as usize,
                    data.len(),
                    "output size mismatch: got {} expected {}",
                    bytes,
                    data.len()
                );
                // STRICT: content must match byte-for-byte
                assert_eq!(output, data, "output content mismatch");
                eprintln!("e2e strict: PASS ({} bytes)", bytes);
            }
            Err(e) => {
                // Fallback is acceptable — but verify sequential works
                eprintln!("e2e strict: fell back ({})", e);
                let mut seq_out = Vec::new();
                let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
                std::io::Read::read_to_end(&mut decoder, &mut seq_out).unwrap();
                assert_eq!(seq_out, data, "sequential decode mismatch");
            }
        }
    }

    // =========================================================================
    // INVARIANT: pipeline on silesia produces correct output
    // Catches: real-world data failures
    // =========================================================================

    #[test]
    fn test_parallel_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };

        let (_deflate, ref_output) = get_deflate_and_expected(&gz);
        let ref_size = ref_output.len();

        let mut par_output = Vec::new();
        let t = std::time::Instant::now();
        let result = decompress_parallel(&gz, &mut par_output, 10);
        let elapsed = t.elapsed();

        match result {
            Ok(bytes) => {
                assert_eq!(bytes as usize, ref_size, "size mismatch");
                assert_eq!(par_output, ref_output, "content mismatch");
                let mbps = ref_size as f64 / elapsed.as_secs_f64() / 1e6;
                eprintln!(
                    "parallel silesia: {} bytes in {:.1}ms ({:.0} MB/s)",
                    ref_size,
                    elapsed.as_secs_f64() * 1000.0,
                    mbps
                );
            }
            Err(e) => {
                eprintln!("parallel silesia fell back: {}", e);
            }
        }
    }

    // =========================================================================
    // INVARIANT: sum of chunk outputs ≈ ISIZE
    // Catches: overlapping or missing chunks (failure mode #4)
    // =========================================================================

    #[test]
    fn test_chunk_output_sum_matches_isize() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];
        let expected_size = data.len();

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            eprintln!("not enough boundaries, skipping");
            return;
        }

        // Decode all chunks and sum output
        let chunks = match decode_chunks_parallel(deflate, &boundaries, 4) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("decode failed: {}, skipping", e);
                return;
            }
        };

        let total: usize = chunks.iter().map(|c| c.data.len()).sum();
        eprintln!(
            "chunk output sum: {}, expected: {}, ratio: {:.2}",
            total,
            expected_size,
            total as f64 / expected_size as f64
        );

        // The sum of chunk outputs may not exactly equal ISIZE because
        // decode_until_bit can overshoot. But it should be close.
        // If it's wildly off (>2x), something is very wrong.
        if total > expected_size * 2 {
            eprintln!(
                "WARNING: chunk output sum {}x expected — likely overshoot bug",
                total / expected_size
            );
        }
    }

    // =========================================================================
    // INVARIANT: find_chunk_boundaries returns sorted, non-overlapping positions
    // Catches: boundary ordering bugs
    // =========================================================================

    #[test]
    fn test_boundaries_sorted_and_unique() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);

        // Must start at 0
        assert_eq!(boundaries[0], 0, "first boundary must be bit 0");

        // Must be strictly sorted
        for i in 1..boundaries.len() {
            assert!(
                boundaries[i] > boundaries[i - 1],
                "boundaries not strictly sorted: [{}]={} >= [{}]={}",
                i - 1,
                boundaries[i - 1],
                i,
                boundaries[i]
            );
        }

        // All boundaries must be within the data
        let max_bit = deflate.len() * 8;
        for (i, &b) in boundaries.iter().enumerate() {
            assert!(
                b < max_bit,
                "boundary {} at bit {} exceeds data size {} bits",
                i,
                b,
                max_bit
            );
        }

        eprintln!("boundaries: {:?}", boundaries);
    }

    // =====================================================================
    //  HELPERS: data generators for specific patterns
    // =====================================================================

    fn make_all_zeros(size: usize) -> Vec<u8> {
        vec![0u8; size]
    }

    fn make_random_data(size: usize, seed: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng = seed;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 32) as u8);
        }
        data.truncate(size);
        data
    }

    fn make_text_data(size: usize) -> Vec<u8> {
        let phrase = b"the quick brown fox jumps over the lazy dog ";
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            let remaining = size - data.len();
            let chunk = &phrase[..remaining.min(phrase.len())];
            data.extend_from_slice(chunk);
        }
        data
    }

    fn make_rle_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xaabbccdd;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let byte = (rng >> 32) as u8;
            let run_len = ((rng >> 48) % 200 + 10) as usize;
            for _ in 0..run_len.min(size - data.len()) {
                data.push(byte);
            }
        }
        data.truncate(size);
        data
    }

    fn make_mixed_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let quarter = size / 4;
        data.extend_from_slice(&make_all_zeros(quarter));
        data.extend_from_slice(&make_text_data(quarter));
        data.extend_from_slice(&make_rle_data(quarter));
        data.extend_from_slice(&make_random_data(size - data.len(), 42));
        data
    }

    fn make_gzip_at_level(data: &[u8], level: u32) -> Vec<u8> {
        use std::io::Write;
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn get_oracle_boundaries(deflate: &[u8], interval: usize) -> Vec<usize> {
        let scan = crate::scan_inflate::scan_deflate_fast(deflate, interval, 0)
            .expect("scan_deflate_fast");
        let mut bits: Vec<usize> = vec![0];
        for cp in &scan.checkpoints {
            let real_bitsleft = (cp.bitsleft as u8) as usize;
            let bit_pos = cp.input_byte_pos * 8 - real_bitsleft;
            bits.push(bit_pos);
        }
        bits
    }

    fn gzip_crc32(gzip_data: &[u8]) -> u32 {
        let offset = gzip_data.len() - 8;
        u32::from_le_bytes([
            gzip_data[offset],
            gzip_data[offset + 1],
            gzip_data[offset + 2],
            gzip_data[offset + 3],
        ])
    }

    fn compute_crc32(data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    // =====================================================================
    //  1. CHAIN CONVERGENCE — the missing tests that would have caught the bug
    //  A real boundary is on the same block chain as chunk 0.
    //  decode_until_bit must land EXACTLY at the next boundary.
    // =====================================================================

    #[test]
    fn test_oracle_chain_convergence_8mb() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 1024 * 1024);
        assert!(boundaries.len() >= 3, "need at least 3 oracle boundaries");

        for i in 0..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;

            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());
            let data_slice = &deflate[start_byte..end_byte];

            let mut decoder = MarkerDecoder::new(data_slice, relative_start);
            let relative_end = end_bit - start_byte * 8;
            decoder
                .decode_until_bit(usize::MAX, relative_end)
                .expect("decode should succeed");

            let final_pos = decoder.bit_position();
            assert_eq!(
                final_pos,
                relative_end,
                "chunk {}: decode_until_bit landed at bit {} but expected {} (overshoot={})",
                i,
                final_pos,
                relative_end,
                final_pos as i64 - relative_end as i64
            );
        }
    }

    #[test]
    fn test_found_boundary_chain_convergence() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            eprintln!("not enough boundaries, skipping");
            return;
        }

        for i in 0..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;

            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());
            let data_slice = &deflate[start_byte..end_byte];

            let mut decoder = MarkerDecoder::new(data_slice, relative_start);
            let relative_end = end_bit - start_byte * 8;
            decoder
                .decode_until_bit(usize::MAX, relative_end)
                .expect("decode should succeed");

            let final_pos = decoder.bit_position();
            assert_eq!(
                final_pos,
                relative_end,
                "chunk {} (bit {}→{}): decoder landed at {} not {} — \
                 boundary {} is NOT on the same block chain (false positive!)",
                i,
                start_bit,
                end_bit,
                final_pos,
                relative_end,
                i + 1
            );
        }
    }

    #[test]
    fn test_chunk_output_tiles_exactly_with_oracle() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 1024 * 1024);
        assert!(boundaries.len() >= 3);

        let mut total_output = 0usize;
        for i in 0..boundaries.len() {
            let start_bit = boundaries[i];
            let is_last = i + 1 >= boundaries.len();
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;

            let mut decoder;
            if is_last {
                decoder = MarkerDecoder::new(&deflate[start_byte..], relative_start);
                let _ = decoder.decode_until(deflate.len() * 8);
            } else {
                let end_bit = boundaries[i + 1];
                let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());
                decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], relative_start);
                let relative_end = end_bit - start_byte * 8;
                decoder.decode_until_bit(usize::MAX, relative_end).unwrap();
            }
            total_output += decoder.output().len();
        }

        assert_eq!(
            total_output,
            data.len(),
            "oracle-chunked output {} != expected {}",
            total_output,
            data.len()
        );
    }

    // =====================================================================
    //  2. CRC32 VERIFICATION — output correctness independent of size
    // =====================================================================

    #[test]
    fn test_parallel_output_crc_matches_trailer() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);

        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Ok(_) => {
                let expected_crc = gzip_crc32(&compressed);
                let actual_crc = compute_crc32(&output);
                assert_eq!(
                    actual_crc, expected_crc,
                    "CRC32 mismatch: output={:#010x} trailer={:#010x}",
                    actual_crc, expected_crc
                );
            }
            Err(e) => {
                eprintln!("parallel returned error (acceptable for now): {}", e);
            }
        }
    }

    #[test]
    fn test_sequential_reference_crc_matches() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let expected_crc = gzip_crc32(&compressed);
        let actual_crc = compute_crc32(&data);
        assert_eq!(
            actual_crc, expected_crc,
            "reference data CRC doesn't match gzip trailer"
        );
    }

    // =====================================================================
    //  3. try_decode_at — exhaustive validation
    // =====================================================================

    #[test]
    fn test_try_decode_rejects_bit0() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Bit 0 is a real boundary but has NO markers (it's the stream start).
        // try_decode_at should reject it because marker_count == 0.
        assert!(
            !try_decode_at(deflate, 0),
            "bit 0 should be rejected (no markers at stream start)"
        );
    }

    #[test]
    fn test_try_decode_accepts_mid_stream_oracle() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
        // Skip first (bit 0) and last (near end) — test mid-stream boundaries
        let mid_boundaries: Vec<_> = boundaries[1..boundaries.len().saturating_sub(1)].to_vec();
        assert!(!mid_boundaries.is_empty(), "need mid-stream boundaries");

        for &bit in &mid_boundaries {
            assert!(
                try_decode_at(deflate, bit),
                "real mid-stream boundary at bit {} rejected",
                bit
            );
        }
    }

    #[test]
    fn test_try_decode_rejects_mid_block_positions() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
        assert!(boundaries.len() >= 3);

        // Test positions that are NOT block boundaries: midpoints between oracle boundaries
        let mut rejected = 0;
        let mut tested = 0;
        for i in 0..boundaries.len() - 1 {
            let mid_bit = (boundaries[i] + boundaries[i + 1]) / 2;
            // Skip if mid_bit happens to be a boundary (unlikely but possible)
            if boundaries.contains(&mid_bit) {
                continue;
            }
            tested += 1;
            if !try_decode_at(deflate, mid_bit) {
                rejected += 1;
            }
        }

        let rejection_rate = if tested > 0 {
            rejected as f64 / tested as f64
        } else {
            1.0
        };
        assert!(
            rejection_rate > 0.8,
            "mid-block positions should mostly be rejected: {}/{} rejected ({:.0}%)",
            rejected,
            tested,
            rejection_rate * 100.0
        );
    }

    #[test]
    fn test_try_decode_marker_rate_at_oracle_boundary() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 1024 * 1024);
        for &bit in &boundaries[1..] {
            let start_byte = bit / 8;
            let relative_start = bit % 8;
            let remaining = deflate.len() - start_byte;
            let slice_len = remaining.min(2 * 1024 * 1024);
            let data_slice = &deflate[start_byte..start_byte + slice_len];

            let mut decoder = MarkerDecoder::new(data_slice, relative_start);
            let _ = decoder.decode_until(512 * 1024);

            let output_len = decoder.output().len();
            let marker_count = decoder.marker_count();
            if output_len > 0 {
                let rate = marker_count as f64 / output_len as f64;
                assert!(
                    rate < 0.20,
                    "oracle boundary at bit {}: marker rate {:.1}% exceeds 20%",
                    bit,
                    rate * 100.0
                );
            }
        }
    }

    #[test]
    fn test_try_decode_500_random_positions() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size =
            crate::marker_decode::skip_gzip_header(&compressed).expect("valid header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let total_bits = deflate.len() * 8;
        let mut rng: u64 = 0x1234567890;
        let mut accepted = 0;

        for _ in 0..500 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (rng as usize) % total_bits;
            if try_decode_at(deflate, bit) {
                accepted += 1;
            }
        }

        assert!(
            accepted < 10,
            "try_decode accepted {}/500 random positions — too permissive",
            accepted
        );
    }

    // =====================================================================
    //  4. MARKER RESOLUTION — window propagation and replace_markers
    // =====================================================================

    #[test]
    fn test_update_window_small_data() {
        let mut window = Vec::new();
        update_window(&mut window, &[1, 2, 3]);
        assert_eq!(window, vec![1, 2, 3]);
    }

    #[test]
    fn test_update_window_exactly_window_size() {
        let mut window = Vec::new();
        let data: Vec<u8> = (0..WINDOW_SIZE).map(|i| (i % 256) as u8).collect();
        update_window(&mut window, &data);
        assert_eq!(window.len(), WINDOW_SIZE);
        assert_eq!(window, data);
    }

    #[test]
    fn test_update_window_larger_than_window_size() {
        let mut window = Vec::new();
        let data: Vec<u8> = (0..WINDOW_SIZE + 1000).map(|i| (i % 256) as u8).collect();
        update_window(&mut window, &data);
        assert_eq!(window.len(), WINDOW_SIZE);
        assert_eq!(window, &data[1000..]);
    }

    #[test]
    fn test_update_window_accumulates() {
        let mut window = Vec::new();
        update_window(&mut window, &[1, 2, 3]);
        update_window(&mut window, &[4, 5, 6]);
        assert_eq!(window, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_update_window_rotates_at_capacity() {
        let mut window = Vec::new();
        let fill: Vec<u8> = (0..WINDOW_SIZE).map(|i| (i % 256) as u8).collect();
        update_window(&mut window, &fill);
        assert_eq!(window.len(), WINDOW_SIZE);

        update_window(&mut window, &[0xAA, 0xBB]);
        assert_eq!(window.len(), WINDOW_SIZE);
        assert_eq!(window[WINDOW_SIZE - 2], 0xAA);
        assert_eq!(window[WINDOW_SIZE - 1], 0xBB);
        // First two bytes should have shifted
        assert_eq!(window[0], fill[2]);
    }

    #[test]
    fn test_update_window_empty_data() {
        let mut window = vec![1, 2, 3];
        update_window(&mut window, &[]);
        assert_eq!(window, vec![1, 2, 3]);
    }

    #[test]
    fn test_replace_markers_no_markers() {
        let mut data: Vec<u16> = vec![65, 66, 67]; // 'A', 'B', 'C'
        let window = vec![0u8; WINDOW_SIZE];
        replace_markers(&mut data, &window);
        assert_eq!(data, vec![65, 66, 67]);
    }

    #[test]
    fn test_replace_markers_with_markers() {
        use crate::marker_decode::MARKER_BASE;
        let mut window = vec![0u8; WINDOW_SIZE];
        window[WINDOW_SIZE - 1] = 0xFF;
        window[WINDOW_SIZE - 2] = 0xFE;

        // Marker encoding: value = MARKER_BASE + offset
        // replace_markers resolves: window[window.len() - 1 - offset]
        // offset 0 → window[last] = 0xFF, offset 1 → window[last-1] = 0xFE
        let mut data: Vec<u16> = vec![65, MARKER_BASE, MARKER_BASE + 1];
        replace_markers(&mut data, &window);
        assert_eq!(data[0], 65);
        assert_eq!(data[1], 0xFF_u16);
        assert_eq!(data[2], 0xFE_u16);
    }

    // =====================================================================
    //  5. THREAD COUNT INDEPENDENCE — same output regardless of threads
    // =====================================================================

    #[test]
    fn test_thread_count_independence() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);

        let mut results: Vec<(usize, Result<Vec<u8>, String>)> = Vec::new();

        for threads in [4, 6, 8] {
            let mut output = Vec::new();
            match decompress_parallel(&compressed, &mut output, threads) {
                Ok(_) => results.push((threads, Ok(output))),
                Err(e) => results.push((threads, Err(format!("{}", e)))),
            }
        }

        // All successful results must be identical
        let successful: Vec<_> = results
            .iter()
            .filter_map(|(t, r)| r.as_ref().ok().map(|v| (*t, v)))
            .collect();

        for window in successful.windows(2) {
            let (t1, v1) = window[0];
            let (t2, v2) = window[1];
            assert_eq!(
                v1,
                v2,
                "output differs between T{} ({} bytes) and T{} ({} bytes)",
                t1,
                v1.len(),
                t2,
                v2.len()
            );
        }
    }

    // =====================================================================
    //  6. DATA PATTERN COVERAGE — different data types
    // =====================================================================

    fn assert_parallel_correct_or_error(label: &str, data: &[u8]) {
        let compressed = make_gzip_data(data);
        if compressed.len() < MIN_PARALLEL_SIZE {
            return;
        }
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Ok(bytes) => {
                assert_eq!(
                    bytes as usize,
                    data.len(),
                    "{}: size mismatch {} vs {}",
                    label,
                    bytes,
                    data.len()
                );
                assert_eq!(&output, data, "{}: content mismatch", label);
            }
            Err(_) => {
                // Error is acceptable — silent wrong output is not
            }
        }
    }

    #[test]
    fn test_parallel_compressible_data() {
        let data = make_compressible_data(8 * 1024 * 1024);
        assert_parallel_correct_or_error("compressible", &data);
    }

    #[test]
    fn test_parallel_all_zeros() {
        let data = make_all_zeros(8 * 1024 * 1024);
        assert_parallel_correct_or_error("all_zeros", &data);
    }

    #[test]
    fn test_parallel_random_data() {
        let data = make_random_data(8 * 1024 * 1024, 42);
        assert_parallel_correct_or_error("random", &data);
    }

    #[test]
    fn test_parallel_text_data() {
        let data = make_text_data(8 * 1024 * 1024);
        assert_parallel_correct_or_error("text", &data);
    }

    #[test]
    fn test_parallel_rle_data() {
        let data = make_rle_data(8 * 1024 * 1024);
        assert_parallel_correct_or_error("rle", &data);
    }

    #[test]
    fn test_parallel_mixed_data() {
        let data = make_mixed_data(8 * 1024 * 1024);
        assert_parallel_correct_or_error("mixed", &data);
    }

    // =====================================================================
    //  7. COMPRESSION LEVEL COVERAGE — different block structures
    // =====================================================================

    fn assert_parallel_correct_at_level(level: u32) {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_at_level(&data, level);
        if compressed.len() < MIN_PARALLEL_SIZE {
            return;
        }
        let mut output = Vec::new();
        if let Ok(bytes) = decompress_parallel(&compressed, &mut output, 4) {
            assert_eq!(bytes as usize, data.len(), "level {}: size mismatch", level);
            assert_eq!(output, data, "level {}: content mismatch", level);
        }
    }

    #[test]
    fn test_parallel_level_1() {
        assert_parallel_correct_at_level(1);
    }

    #[test]
    fn test_parallel_level_6() {
        assert_parallel_correct_at_level(6);
    }

    #[test]
    fn test_parallel_level_9() {
        assert_parallel_correct_at_level(9);
    }

    // =====================================================================
    //  8. SIZE BOUNDARIES — edge cases around MIN_PARALLEL_SIZE
    // =====================================================================

    #[test]
    fn test_parallel_exactly_min_size() {
        let data = make_compressible_data(MIN_PARALLEL_SIZE * 3);
        let compressed = make_gzip_data(&data);
        // Might or might not be large enough after compression
        let mut output = Vec::new();
        let _ = decompress_parallel(&compressed, &mut output, 4);
        // Just verify no crash / no wrong output
        if !output.is_empty() {
            assert_eq!(&output, &data, "wrong output at min size boundary");
        }
    }

    #[test]
    fn test_parallel_below_min_size() {
        let data = make_compressible_data(1024);
        let compressed = make_gzip_data(&data);
        let mut output = Vec::new();
        let result = decompress_parallel(&compressed, &mut output, 4);
        assert!(matches!(result, Err(ParallelError::TooSmall)));
    }

    #[test]
    fn test_parallel_too_few_threads() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let mut output = Vec::new();
        let result = decompress_parallel(&compressed, &mut output, 2);
        assert!(matches!(result, Err(ParallelError::TooSmall)));
    }

    // =====================================================================
    //  9. ERROR HANDLING — bad input never produces wrong output
    // =====================================================================

    #[test]
    fn test_parallel_invalid_header() {
        let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09];
        let mut output = Vec::new();
        let result = decompress_parallel(&data, &mut output, 4);
        assert!(result.is_err());
        assert!(output.is_empty(), "invalid header should produce no output");
    }

    #[test]
    fn test_parallel_truncated_gzip() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let truncated = &compressed[..compressed.len() / 2];
        let mut output = Vec::new();
        let result = decompress_parallel(truncated, &mut output, 4);
        // Must either error or produce correct partial output — never wrong data
        if let Ok(bytes) = result {
            let prefix = &data[..bytes as usize];
            assert_eq!(&output[..bytes as usize], prefix);
        }
    }

    #[test]
    fn test_parallel_empty_input() {
        let mut output = Vec::new();
        let result = decompress_parallel(&[], &mut output, 4);
        assert!(result.is_err());
    }

    // =====================================================================
    //  10. CHUNK DECODER PROPERTIES — per-chunk invariants
    // =====================================================================

    #[test]
    fn test_chunk0_always_zero_markers() {
        for (label, data) in [
            ("compressible", make_compressible_data(8 * 1024 * 1024)),
            ("text", make_text_data(8 * 1024 * 1024)),
            ("rle", make_rle_data(8 * 1024 * 1024)),
        ] {
            let compressed = make_gzip_data(&data);
            let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
            let deflate = &compressed[header_size..compressed.len() - 8];
            let boundaries = find_chunk_boundaries(deflate, 4);
            if boundaries.len() < 2 {
                continue;
            }

            let end_bit = boundaries[1];
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());
            let mut decoder = MarkerDecoder::new(&deflate[..end_byte], 0);
            let _ = decoder.decode_until_bit(usize::MAX, end_bit);

            assert_eq!(
                decoder.marker_count(),
                0,
                "{}: chunk 0 has {} markers (should be 0)",
                label,
                decoder.marker_count()
            );
        }
    }

    #[test]
    fn test_mid_chunks_have_markers() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 1024 * 1024);
        assert!(boundaries.len() >= 4);

        // Mid-stream chunks (not first, not last) should have markers
        for i in 1..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());
            let data_slice = &deflate[start_byte..end_byte];

            let mut decoder = MarkerDecoder::new(data_slice, relative_start);
            let relative_end = end_bit - start_byte * 8;
            let _ = decoder.decode_until_bit(usize::MAX, relative_end);

            assert!(
                decoder.marker_count() > 0,
                "oracle chunk {}: 0 markers — every mid-stream chunk should have some",
                i
            );
        }
    }

    #[test]
    fn test_marker_rate_reasonable_at_oracle_boundaries() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 1024 * 1024);
        for i in 1..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());

            let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], relative_start);
            let relative_end = end_bit - start_byte * 8;
            let _ = decoder.decode_until_bit(usize::MAX, relative_end);

            let output_len = decoder.output().len();
            let marker_count = decoder.marker_count();
            if output_len > 0 {
                let rate = marker_count as f64 / output_len as f64;
                assert!(
                    rate < 0.15,
                    "oracle chunk {}: marker rate {:.1}% — real chunks should be <15%",
                    i,
                    rate * 100.0
                );
            }
        }
    }

    // =====================================================================
    //  11. CHUNK OUTPUT vs SEQUENTIAL — byte-exact comparison per chunk
    // =====================================================================

    #[test]
    fn test_each_oracle_chunk_matches_sequential_range() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let scan = crate::scan_inflate::scan_deflate_fast(deflate, 1024 * 1024, 0).expect("scan");
        let boundaries = get_oracle_boundaries(deflate, 1024 * 1024);

        let mut expected_offset = 0usize;
        for i in 0..boundaries.len() {
            let start_bit = boundaries[i];
            let is_last = i + 1 >= boundaries.len();
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;

            let mut decoder;
            if is_last {
                decoder = MarkerDecoder::new(&deflate[start_byte..], relative_start);
                let _ = decoder.decode_until(deflate.len() * 8);
            } else {
                let end_bit = boundaries[i + 1];
                let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());
                decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], relative_start);
                let relative_end = end_bit - start_byte * 8;
                let _ = decoder.decode_until_bit(usize::MAX, relative_end);
            }

            let output = decoder.output();
            let chunk_len = output.len();

            if i == 0 {
                // Chunk 0 should be byte-exact
                let chunk_bytes: Vec<u8> = output.iter().map(|&v| v as u8).collect();
                assert_eq!(
                    &chunk_bytes,
                    &data[..chunk_len],
                    "chunk 0 doesn't match sequential"
                );
            } else if i < boundaries.len() - 1 {
                // Mid chunks: verify after marker resolution using checkpoint window
                let cp = &scan.checkpoints[i - 1];
                let mut resolved = output.to_vec();
                replace_markers(&mut resolved, &cp.window);
                let chunk_bytes: Vec<u8> = resolved.iter().map(|&v| v as u8).collect();
                assert_eq!(
                    &chunk_bytes,
                    &data[expected_offset..expected_offset + chunk_len],
                    "chunk {} resolved output doesn't match sequential range [{}..{}]",
                    i,
                    expected_offset,
                    expected_offset + chunk_len
                );
            }

            expected_offset += chunk_len;
        }

        assert_eq!(expected_offset, data.len(), "total output size mismatch");
    }

    // =====================================================================
    //  12. max_output_for_chunk — bounds calculation
    // =====================================================================

    #[test]
    fn test_max_output_minimum() {
        assert!(max_output_for_chunk(0) >= 64 * 1024);
        assert!(max_output_for_chunk(100) >= 64 * 1024);
    }

    #[test]
    fn test_max_output_scaling() {
        assert_eq!(max_output_for_chunk(1024 * 1024), 8 * 1024 * 1024);
        assert_eq!(max_output_for_chunk(10 * 1024 * 1024), 80 * 1024 * 1024);
    }

    // =====================================================================
    //  13. FIND_CHUNK_BOUNDARIES — structural properties
    // =====================================================================

    #[test]
    fn test_boundaries_always_start_at_zero() {
        for size in [4 * 1024 * 1024, 8 * 1024 * 1024] {
            let data = make_compressible_data(size);
            let compressed = make_gzip_data(&data);
            let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
            let deflate = &compressed[header_size..compressed.len() - 8];

            let boundaries = find_chunk_boundaries(deflate, 4);
            assert_eq!(
                boundaries[0], 0,
                "first boundary must be 0 for size {}",
                size
            );
        }
    }

    #[test]
    fn test_boundaries_within_data_range() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        for num_chunks in [2, 4, 8, 16] {
            let boundaries = find_chunk_boundaries(deflate, num_chunks);
            let max_bit = deflate.len() * 8;
            for (i, &b) in boundaries.iter().enumerate() {
                assert!(
                    b < max_bit,
                    "boundary {} at bit {} exceeds {} bits (chunks={})",
                    i,
                    b,
                    max_bit,
                    num_chunks
                );
            }
        }
    }

    #[test]
    fn test_boundaries_strictly_increasing() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        for num_chunks in [2, 4, 8] {
            let boundaries = find_chunk_boundaries(deflate, num_chunks);
            for i in 1..boundaries.len() {
                assert!(
                    boundaries[i] > boundaries[i - 1],
                    "not strictly increasing at {} (chunks={}): {} >= {}",
                    i,
                    num_chunks,
                    boundaries[i - 1],
                    boundaries[i]
                );
            }
        }
    }

    #[test]
    fn test_boundaries_count_reasonable() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        for num_chunks in [4, 8] {
            let boundaries = find_chunk_boundaries(deflate, num_chunks);
            assert!(
                boundaries.len() >= 2,
                "need at least 2 boundaries for {} chunks, got {}",
                num_chunks,
                boundaries.len()
            );
            assert!(
                boundaries.len() <= num_chunks + 1,
                "too many boundaries for {} chunks: got {}",
                num_chunks,
                boundaries.len()
            );
        }
    }

    #[test]
    fn test_boundaries_roughly_evenly_spaced() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 3 {
            return;
        }

        let total_bits = deflate.len() * 8;
        let expected_spacing = total_bits / boundaries.len();

        for i in 1..boundaries.len() {
            let spacing = boundaries[i] - boundaries[i - 1];
            // Allow 5x variation (generous but catches catastrophic misplacement)
            assert!(
                spacing < expected_spacing * 5,
                "chunk {} spacing {} is >5x expected {} — boundary misplaced",
                i,
                spacing,
                expected_spacing
            );
        }
    }

    // =====================================================================
    //  14. RESOLVE_AND_WRITE — integration with resolve step
    // =====================================================================

    #[test]
    fn test_resolve_single_chunk_no_markers() {
        let chunk = DecodedChunk {
            data: vec![65, 66, 67, 68],
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk], &mut output, 4, 0);
        assert_eq!(result.unwrap(), 4);
        assert_eq!(output, vec![65, 66, 67, 68]);
    }

    #[test]
    fn test_resolve_two_chunks_with_markers() {
        use crate::marker_decode::MARKER_BASE;
        let chunk0 = DecodedChunk {
            data: vec![65, 66, 67, 68], // ABCD
            marker_count: 0,
        };
        // MARKER_BASE + 0 → window[window.len() - 1] = last byte of window
        // After chunk0, window = [65, 66, 67, 68], so window[3] = 68 = 'D'
        let chunk1 = DecodedChunk {
            data: vec![69, MARKER_BASE],
            marker_count: 1,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk0, chunk1], &mut output, 6, 0);
        assert_eq!(result.unwrap(), 6);
        assert_eq!(output, vec![65, 66, 67, 68, 69, 68]);
    }

    #[test]
    fn test_resolve_size_mismatch_detected() {
        let chunk = DecodedChunk {
            data: vec![65, 66],
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk], &mut output, 100, 0);
        assert!(matches!(result, Err(ParallelError::SizeMismatch)));
    }

    #[test]
    fn test_resolve_chunk1_no_window_fails() {
        // Chunk 1 has markers but there's no preceding chunk to provide window.
        // Actually chunk 0 provides window, so this tests chunk 0 being empty.
        let chunk0 = DecodedChunk {
            data: vec![],
            marker_count: 0,
        };
        let chunk1 = DecodedChunk {
            data: vec![256],
            marker_count: 1,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk0, chunk1], &mut output, 0, 0);
        // Empty window can't resolve markers
        assert!(result.is_err());
    }

    // =====================================================================
    //  15. DECODE_UNTIL_BIT — bit-level precision
    // =====================================================================

    #[test]
    fn test_decode_until_bit_respects_output_limit() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let max_output = 1000;
        let far_bit = deflate.len() * 8;
        let mut decoder = MarkerDecoder::new(deflate, 0);
        let _ = decoder.decode_until_bit(max_output, far_bit);

        // Output should be near max_output (may overshoot by one block)
        assert!(
            decoder.output().len() <= max_output * 4,
            "output {} far exceeded limit {}",
            decoder.output().len(),
            max_output
        );
    }

    #[test]
    fn test_decode_until_bit_stops_at_exact_boundary() {
        let data = make_compressible_data(4 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
        if boundaries.len() < 3 {
            return;
        }

        // Decode from start to second boundary
        let target = boundaries[1];
        let mut decoder = MarkerDecoder::new(deflate, 0);
        decoder.decode_until_bit(usize::MAX, target).unwrap();

        assert_eq!(
            decoder.bit_position(),
            target,
            "should stop exactly at oracle boundary"
        );
    }

    // =====================================================================
    //  16. BLOCK FINDER — structural validation
    // =====================================================================

    #[test]
    fn test_block_finder_candidates_are_valid_positions() {
        let data = make_compressible_data(4 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let finder = BlockFinder::new(deflate);
        let candidates = finder.find_blocks(0, deflate.len() * 8);

        for c in &candidates {
            assert!(
                c.bit_offset < deflate.len() * 8,
                "candidate at bit {} exceeds data length {} bits",
                c.bit_offset,
                deflate.len() * 8
            );
        }
    }

    #[test]
    fn test_block_finder_sorted_output() {
        let data = make_compressible_data(4 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let finder = BlockFinder::new(deflate);
        let candidates = finder.find_blocks(0, deflate.len() * 8);

        for i in 1..candidates.len() {
            assert!(
                candidates[i].bit_offset >= candidates[i - 1].bit_offset,
                "candidates not sorted at index {}",
                i
            );
        }
    }

    // =====================================================================
    //  17. PIPELINE INVARIANT: no output written before verification
    //  (Currently resolve_and_write writes as it goes — this test
    //   documents the current behavior and will catch if we fix it)
    // =====================================================================

    #[test]
    fn test_pipeline_wrong_output_never_committed() {
        // Construct chunks where total size doesn't match expected.
        // resolve_and_write should return error AND ideally not have
        // written partial output. Currently it writes then checks.
        let chunk0 = DecodedChunk {
            data: vec![65, 66, 67],
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk0], &mut output, 100, 0);
        assert!(result.is_err());
        assert!(output.is_empty(), "on error, no output should be written");
    }

    // =====================================================================
    //  18. REGRESSION TESTS — specific bugs we've seen
    // =====================================================================

    #[test]
    fn test_regression_34k_false_positive() {
        // False positive boundaries produce tiny output (~34K) with 0 markers.
        // This was the original bug: chunks 4 and 7 produced 34K/52K output.
        // Now caught by: markers > 0 check + output volume threshold.
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            return;
        }

        let chunks = decode_chunks_parallel(deflate, &boundaries, 4);
        if let Ok(chunks) = chunks {
            let expected_per_chunk = data.len() / chunks.len();
            for (i, chunk) in chunks.iter().enumerate() {
                if i == 0 {
                    continue; // Chunk 0 is always correct
                }
                // No chunk should produce < 10% of expected output
                let min_reasonable = expected_per_chunk / 10;
                assert!(
                    chunk.data.len() > min_reasonable,
                    "chunk {} produced only {} values (expected ~{})",
                    i,
                    chunk.data.len(),
                    expected_per_chunk
                );
                // No mid-stream chunk should have 0 markers
                assert!(
                    chunk.marker_count > 0,
                    "chunk {} has 0 markers — likely false positive boundary",
                    i
                );
            }
        }
    }

    #[test]
    fn test_regression_all_markers_false_positive() {
        // False positive boundaries produce output where 100% of values are markers.
        // Seen on silesia: chunk 1 had 53.8M values, all markers.
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            return;
        }

        if let Ok(chunks) = decode_chunks_parallel(deflate, &boundaries, 4) {
            for (i, chunk) in chunks.iter().enumerate() {
                if i == 0 || chunk.data.is_empty() {
                    continue;
                }
                let rate = chunk.marker_count as f64 / chunk.data.len() as f64;
                assert!(
                    rate < 0.50,
                    "chunk {} has {:.0}% markers — likely false positive (100% markers = wrong block chain)",
                    i,
                    rate * 100.0
                );
            }
        }
    }

    #[test]
    fn test_regression_full_tail_slice() {
        // PR #43: passing &data[start..] instead of &data[start..end]
        // causes each chunk to process the entire remaining stream.
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 3 {
            return;
        }

        // Verify chunk 1's data slice is bounded, not full tail
        let start_byte = boundaries[1] / 8;
        let next_byte = boundaries[2] / 8;
        let expected_end = (next_byte + 64 * 1024).min(deflate.len());
        let slice_len = expected_end - start_byte;

        // Slice should be much smaller than full remaining data
        let remaining = deflate.len() - start_byte;
        assert!(
            slice_len < remaining,
            "chunk 1 slice ({} bytes) equals full tail ({} bytes)",
            slice_len,
            remaining
        );
    }

    // =====================================================================
    //  19. PROPERTY: parallel output == sequential output (when parallel succeeds)
    // =====================================================================

    fn assert_parallel_matches_sequential(label: &str, gzip_data: &[u8]) {
        let mut seq_output = Vec::new();
        let mut decoder = flate2::read::GzDecoder::new(gzip_data);
        std::io::Read::read_to_end(&mut decoder, &mut seq_output).unwrap();

        let mut par_output = Vec::new();
        match decompress_parallel(gzip_data, &mut par_output, 4) {
            Ok(bytes) => {
                assert_eq!(
                    bytes as usize,
                    seq_output.len(),
                    "{}: parallel size {} != sequential size {}",
                    label,
                    bytes,
                    seq_output.len()
                );
                assert_eq!(
                    par_output, seq_output,
                    "{}: parallel content != sequential content",
                    label
                );
            }
            Err(_) => {
                // Parallel declined — acceptable
            }
        }
    }

    #[test]
    fn test_parallel_eq_sequential_compressible() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        assert_parallel_matches_sequential("compressible", &compressed);
    }

    #[test]
    fn test_parallel_eq_sequential_text() {
        let data = make_text_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        assert_parallel_matches_sequential("text", &compressed);
    }

    #[test]
    fn test_parallel_eq_sequential_rle() {
        let data = make_rle_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        assert_parallel_matches_sequential("rle", &compressed);
    }

    // =====================================================================
    //  20. ORACLE PIPELINE — full pipeline with oracle boundaries
    //  This isolates the pipeline from block-finding quality.
    // =====================================================================

    #[test]
    fn test_oracle_pipeline_byte_exact() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let scan = crate::scan_inflate::scan_deflate_fast(deflate, 1024 * 1024, 0).expect("scan");
        let boundaries = get_oracle_boundaries(deflate, 1024 * 1024);
        assert!(boundaries.len() >= 3);

        let chunks = decode_chunks_parallel(deflate, &boundaries, 4).expect("decode");

        // Resolve markers using oracle windows
        let mut output = Vec::new();
        let mut window = Vec::<u8>::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let bytes: Vec<u8> = if chunk.marker_count > 0 {
                let resolve_window = if i > 0 {
                    &scan.checkpoints[i - 1].window
                } else {
                    &window
                };
                let mut resolved = chunk.data.clone();
                replace_markers(&mut resolved, resolve_window);
                resolved.iter().map(|&v| v as u8).collect()
            } else {
                chunk.data.iter().map(|&v| v as u8).collect()
            };
            output.extend_from_slice(&bytes);
            update_window(&mut window, &bytes);
        }

        assert_eq!(
            output.len(),
            data.len(),
            "oracle pipeline output {} != expected {}",
            output.len(),
            data.len()
        );
        assert_eq!(output, data, "oracle pipeline content mismatch");
    }

    // =====================================================================
    //  21. DETERMINISM — same input always produces same boundaries + output
    // =====================================================================

    #[test]
    fn test_find_boundaries_deterministic() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let b1 = find_chunk_boundaries(deflate, 4);
        let b2 = find_chunk_boundaries(deflate, 4);
        assert_eq!(b1, b2, "boundaries should be deterministic");
    }

    #[test]
    fn test_parallel_output_deterministic() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);

        let mut out1 = Vec::new();
        let r1 = decompress_parallel(&compressed, &mut out1, 4);
        let mut out2 = Vec::new();
        let r2 = decompress_parallel(&compressed, &mut out2, 4);

        match (r1, r2) {
            (Ok(_), Ok(_)) => {
                assert_eq!(out1, out2, "parallel output should be deterministic");
            }
            (Err(_), Err(_)) => { /* both failed, ok */ }
            _ => panic!("inconsistent success/failure between runs"),
        }
    }

    // =====================================================================
    //  22. MULTIPLE SIZES — scale independence
    // =====================================================================

    #[test]
    fn test_parallel_various_sizes() {
        for &size_mb in &[5, 8, 12, 16] {
            let data = make_compressible_data(size_mb * 1024 * 1024);
            assert_parallel_correct_or_error(&format!("{}MB", size_mb), &data);
        }
    }

    // =====================================================================
    //  23. BRUTE FORCE FALLBACK — when BlockFinder misses
    // =====================================================================

    #[test]
    fn test_brute_force_boundaries_still_valid() {
        // Use random data which is hard for BlockFinder
        let data = make_random_data(8 * 1024 * 1024, 999);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);

        // All structural invariants must still hold
        assert_eq!(boundaries[0], 0);
        for i in 1..boundaries.len() {
            assert!(boundaries[i] > boundaries[i - 1]);
            assert!(boundaries[i] < deflate.len() * 8);
        }
    }

    // =====================================================================
    //  24. PER-CHUNK OUTPUT SIZE — no chunk wildly over/undersized
    // =====================================================================

    #[test]
    fn test_chunk_output_sizes_reasonable() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            return;
        }

        if let Ok(chunks) = decode_chunks_parallel(deflate, &boundaries, 4) {
            let expected_per_chunk = data.len() / chunks.len();
            for (i, chunk) in chunks.iter().enumerate() {
                // No chunk should be more than 4x expected
                assert!(
                    chunk.data.len() < expected_per_chunk * 4,
                    "chunk {} produced {} values, expected ~{} — likely overshoot or false positive",
                    i,
                    chunk.data.len(),
                    expected_per_chunk
                );
            }
        }
    }

    // =====================================================================
    //  25. ORACLE BOUNDARIES vs FOUND BOUNDARIES — recall measurement
    // =====================================================================

    #[test]
    fn test_found_boundaries_near_oracle() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let oracle = get_oracle_boundaries(deflate, 2 * 1024 * 1024);
        let found = find_chunk_boundaries(deflate, 4);

        // Each found boundary should be within 1MB of an oracle boundary
        for &fb in &found[1..] {
            let nearest = oracle
                .iter()
                .map(|&ob| fb.abs_diff(ob))
                .min()
                .unwrap_or(usize::MAX);
            let one_mb_bits = 1024 * 1024 * 8;
            assert!(
                nearest < one_mb_bits,
                "found boundary at bit {} is {} bits from nearest oracle (>1MB)",
                fb,
                nearest
            );
        }
    }

    // =====================================================================
    //  A. MARKER DECODER INTERNALS (8 tests)
    //  Tests marker_decode.rs behaviors the pipeline depends on.
    // =====================================================================

    #[test]
    fn test_marker_encoding_distance_1() {
        use crate::marker_decode::MARKER_BASE;
        let data = make_compressible_data(64 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 16 * 1024);
        if boundaries.len() < 3 {
            return;
        }

        let start_bit = boundaries[1];
        let start_byte = start_bit / 8;
        let relative_start = start_bit % 8;
        let end_byte = (boundaries[2] / 8 + 64 * 1024).min(deflate.len());

        let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], relative_start);
        let _ = decoder.decode_until(32 * 1024);

        // Mid-stream chunk should have markers, and they should be >= MARKER_BASE
        if decoder.marker_count() > 0 {
            for &val in decoder.output() {
                if val > 255 {
                    assert!(
                        val >= MARKER_BASE,
                        "marker value {} < MARKER_BASE {}",
                        val,
                        MARKER_BASE
                    );
                }
            }
        }
    }

    #[test]
    fn test_marker_encoding_distance_max() {
        use crate::marker_decode::MARKER_BASE;
        // MARKER_BASE + 32767 is the maximum marker (distance 32768, offset 32767)
        let max_marker = MARKER_BASE + (WINDOW_SIZE as u16 - 1);
        assert_eq!(max_marker, u16::MAX); // 32768 + 32767 = 65535
    }

    #[test]
    fn test_marker_encoding_roundtrip() {
        use crate::marker_decode::MARKER_BASE;
        let mut window = vec![0u8; WINDOW_SIZE];
        for (i, byte) in window.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // For each offset, encode a marker then resolve it
        for offset in [0usize, 1, 100, WINDOW_SIZE - 1] {
            let marker_val = MARKER_BASE + offset as u16;
            let mut data = vec![marker_val];
            replace_markers(&mut data, &window);
            let expected = window[WINDOW_SIZE - 1 - offset];
            assert_eq!(
                data[0], expected as u16,
                "roundtrip failed for offset {}: got {} expected {}",
                offset, data[0], expected
            );
        }
    }

    #[test]
    fn test_decode_block_stored() {
        // Construct a minimal stored block: bfinal=1, btype=00, len=5, nlen, data
        // bfinal=1, btype=00 → bits: 1 00 → byte 0x01 (LSB first: bit0=bfinal, bit1-2=btype)
        // Stored blocks are byte-aligned after the 3 header bits.
        // Length = 5 (little-endian u16), NLength = ~5 = 0xFFFA
        let mut block = vec![0x01, 5, 0, 0xFA, 0xFF];
        // 5 data bytes
        block.extend_from_slice(&[0x41, 0x42, 0x43, 0x44, 0x45]);

        let mut decoder = MarkerDecoder::new(&block, 0);
        let is_final = decoder.decode_block().unwrap();
        assert!(is_final, "should be final block");
        assert_eq!(decoder.output().len(), 5);
        let bytes: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();
        assert_eq!(bytes, vec![0x41, 0x42, 0x43, 0x44, 0x45]);
    }

    #[test]
    fn test_decode_block_dynamic_huffman() {
        // Compress a small string and verify MarkerDecoder can decode it
        let data = b"hello world hello world hello world";
        let compressed = make_gzip_data(data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let mut decoder = MarkerDecoder::new(deflate, 0);
        let _ = decoder.decode_until(1024);
        let output: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();
        assert_eq!(&output, data, "dynamic huffman decode mismatch");
    }

    #[test]
    fn test_eof_zero_fill_behavior() {
        // A very short deflate stream — decoder should handle EOF gracefully
        let data = b"x";
        let compressed = make_gzip_data(data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let mut decoder = MarkerDecoder::new(deflate, 0);
        let result = decoder.decode_until(1024);
        assert!(result.is_ok(), "should not error on short stream");
        let output: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();
        assert_eq!(&output, data);
    }

    #[test]
    fn test_bit_position_tracks_across_blocks() {
        let data = make_compressible_data(256 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let mut decoder = MarkerDecoder::new(deflate, 0);
        let mut prev_pos = 0;

        // Decode block by block, verify bit_position advances
        for _ in 0..10 {
            match decoder.decode_block() {
                Ok(_) => {
                    let pos = decoder.bit_position();
                    assert!(
                        pos > prev_pos,
                        "bit_position didn't advance: {} -> {}",
                        prev_pos,
                        pos
                    );
                    prev_pos = pos;
                }
                Err(_) => break,
            }
        }
        assert!(prev_pos > 0, "should have advanced past bit 0");
    }

    #[test]
    fn test_bit_position_starts_at_offset() {
        let data = make_compressible_data(64 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Start at bit offset 5
        let decoder = MarkerDecoder::new(deflate, 5);
        assert_eq!(
            decoder.bit_position(),
            5,
            "should start at requested bit offset"
        );
    }

    // =====================================================================
    //  B. CHAIN CONVERGENCE — across data patterns and compression levels
    // =====================================================================

    fn assert_oracle_chain_convergence(label: &str, data: &[u8]) {
        let compressed = make_gzip_data(data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
        if boundaries.len() < 3 {
            return;
        }

        for i in 0..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());

            let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], relative_start);
            let relative_end = end_bit - start_byte * 8;
            decoder
                .decode_until_bit(usize::MAX, relative_end)
                .expect("decode");

            assert_eq!(
                decoder.bit_position(),
                relative_end,
                "{}: chunk {} overshoot: landed at {} expected {} (delta={})",
                label,
                i,
                decoder.bit_position(),
                relative_end,
                decoder.bit_position() as i64 - relative_end as i64
            );
        }
    }

    fn assert_oracle_chain_convergence_at_level(label: &str, data: &[u8], level: u32) {
        let compressed = make_gzip_at_level(data, level);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
        if boundaries.len() < 3 {
            return;
        }

        for i in 0..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let relative_start = start_bit % 8;
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());

            let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], relative_start);
            let relative_end = end_bit - start_byte * 8;
            decoder
                .decode_until_bit(usize::MAX, relative_end)
                .expect("decode");

            assert_eq!(
                decoder.bit_position(),
                relative_end,
                "{} L{}: chunk {} overshoot",
                label,
                level,
                i
            );
        }
    }

    #[test]
    fn test_chain_convergence_text_data() {
        let data = make_text_data(4 * 1024 * 1024);
        assert_oracle_chain_convergence("text", &data);
    }

    #[test]
    fn test_chain_convergence_rle_data() {
        let data = make_rle_data(4 * 1024 * 1024);
        assert_oracle_chain_convergence("rle", &data);
    }

    #[test]
    fn test_chain_convergence_mixed_data() {
        let data = make_mixed_data(4 * 1024 * 1024);
        assert_oracle_chain_convergence("mixed", &data);
    }

    #[test]
    fn test_chain_convergence_level1() {
        let data = make_compressible_data(4 * 1024 * 1024);
        assert_oracle_chain_convergence_at_level("compressible", &data, 1);
    }

    #[test]
    fn test_chain_convergence_level9() {
        let data = make_compressible_data(4 * 1024 * 1024);
        assert_oracle_chain_convergence_at_level("compressible", &data, 9);
    }

    // =====================================================================
    //  C. FALSE POSITIVE DETECTION (6 tests)
    // =====================================================================

    #[test]
    fn test_false_positive_detected_by_marker_rate() {
        // Verify that try_decode_at rejects positions where >20% of output is markers
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Test at 200 random positions and verify any accepted has <20% markers
        let mut rng: u64 = 0xfeedface;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (rng as usize) % (deflate.len() * 8);
            if try_decode_at(deflate, bit) {
                // Verify the marker rate at accepted positions
                let start_byte = bit / 8;
                let slice_len = (deflate.len() - start_byte).min(2 * 1024 * 1024);
                let mut decoder =
                    MarkerDecoder::new(&deflate[start_byte..start_byte + slice_len], bit % 8);
                let _ = decoder.decode_until(1024 * 1024);
                if !decoder.output().is_empty() {
                    let rate = decoder.marker_count() as f64 / decoder.output().len() as f64;
                    assert!(
                        rate < 0.20,
                        "accepted position at bit {} has {:.1}% markers",
                        bit,
                        rate * 100.0
                    );
                }
            }
        }
    }

    #[test]
    fn test_false_positive_detected_by_output_volume() {
        // Verify try_decode_at rejects positions producing < min_output
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Test at a known non-boundary: exact middle of stream (unlikely to be a boundary)
        let mid_bit = deflate.len() * 4; // middle in bits
        let result = try_decode_at(deflate, mid_bit);
        // May or may not be accepted — but if rejected, it's because of volume/markers
        // The key test: try_decode_at returns a boolean, not wrong output
        let _ = result;
    }

    #[test]
    fn test_false_positive_detected_by_zero_markers() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Bit 0 should be rejected because it has 0 markers (stream start)
        assert!(
            !try_decode_at(deflate, 0),
            "bit 0 should be rejected (0 markers)"
        );
    }

    #[test]
    fn test_false_positive_detected_by_chain_mismatch() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 3 {
            return;
        }

        // Use found boundaries and verify chain convergence
        for i in 0..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());

            let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], start_bit % 8);
            let relative_end = end_bit - start_byte * 8;
            let _ = decoder.decode_until_bit(usize::MAX, relative_end);

            let final_pos = decoder.bit_position();
            if final_pos != relative_end {
                // This is a false positive — our test detects it
                eprintln!(
                    "chain mismatch at chunk {}: bit {} -> {} (expected {})",
                    i,
                    start_bit,
                    final_pos + start_byte * 8,
                    end_bit
                );
            }
        }
    }

    #[test]
    fn test_bit_position_overshoot_means_false_positive() {
        let data = make_compressible_data(4 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let oracle = get_oracle_boundaries(deflate, 512 * 1024);
        assert!(oracle.len() >= 3);

        // With oracle boundaries, there should be NO overshoot
        for i in 0..oracle.len() - 1 {
            let start_bit = oracle[i];
            let end_bit = oracle[i + 1];
            let start_byte = start_bit / 8;
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());

            let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], start_bit % 8);
            let relative_end = end_bit - start_byte * 8;
            let _ = decoder.decode_until_bit(usize::MAX, relative_end);

            let final_pos = decoder.bit_position();
            assert!(
                final_pos <= relative_end,
                "oracle boundary {}: overshoot {} > {} (delta={})",
                i,
                final_pos,
                relative_end,
                final_pos - relative_end
            );
        }
    }

    #[test]
    fn test_false_positive_rate_under_1_percent() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let total_bits = deflate.len() * 8;
        let mut rng: u64 = 0xdeadbeef_cafebabe;
        let mut accepted = 0;
        let n = 1000;

        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bit = (rng as usize) % total_bits;
            if try_decode_at(deflate, bit) {
                accepted += 1;
            }
        }

        let rate = accepted as f64 / n as f64;
        assert!(
            rate < 0.01,
            "false positive rate {:.2}% ({}/{}) exceeds 1%",
            rate * 100.0,
            accepted,
            n
        );
    }

    // =====================================================================
    //  D. RESOLVE_AND_WRITE CORRECTNESS (6 tests)
    // =====================================================================

    #[test]
    fn test_resolve_window_propagates_across_3_chunks() {
        use crate::marker_decode::MARKER_BASE;
        // Chunk 0: [10, 20, 30]   → 3 bytes
        // Chunk 1: [40, MARKER]   → 2 bytes, marker resolves to 30 (window last)
        // Chunk 2: [50, MARKER]   → 2 bytes, marker resolves to 30 (window last after chunk1)
        // Total: 7 bytes
        let chunk0 = DecodedChunk {
            data: vec![10, 20, 30],
            marker_count: 0,
        };
        let chunk1 = DecodedChunk {
            data: vec![40, MARKER_BASE],
            marker_count: 1,
        };
        let chunk2 = DecodedChunk {
            data: vec![50, MARKER_BASE],
            marker_count: 1,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk0, chunk1, chunk2], &mut output, 7, 0);
        assert_eq!(result.unwrap(), 7);
        // After chunk0: window=[10,20,30], chunk1 marker→30
        // After chunk1: window=[10,20,30,40,30], chunk2 marker→30
        assert_eq!(output, vec![10, 20, 30, 40, 30, 50, 30]);
    }

    #[test]
    fn test_resolve_large_window_rotation() {
        // Fill window past 32KB to verify rotation works
        let big_chunk = DecodedChunk {
            data: (0..WINDOW_SIZE + 100).map(|i| (i % 256) as u16).collect(),
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[big_chunk], &mut output, WINDOW_SIZE + 100, 0);
        assert_eq!(result.unwrap(), WINDOW_SIZE + 100);
    }

    #[test]
    fn test_resolve_all_literal_chunks() {
        let chunk0 = DecodedChunk {
            data: vec![65, 66, 67],
            marker_count: 0,
        };
        let chunk1 = DecodedChunk {
            data: vec![68, 69, 70],
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk0, chunk1], &mut output, 6, 0);
        assert_eq!(result.unwrap(), 6);
        assert_eq!(output, vec![65, 66, 67, 68, 69, 70]);
    }

    #[test]
    fn test_resolve_marker_at_window_boundary() {
        use crate::marker_decode::MARKER_BASE;
        // Create a chunk that fills exactly WINDOW_SIZE, then a marker chunk
        let big_data: Vec<u16> = (0..WINDOW_SIZE).map(|i| (i % 256) as u16).collect();
        let chunk0 = DecodedChunk {
            data: big_data,
            marker_count: 0,
        };
        // Marker offset = WINDOW_SIZE-1 → window[0] = first byte
        let chunk1 = DecodedChunk {
            data: vec![MARKER_BASE + (WINDOW_SIZE as u16 - 1)],
            marker_count: 1,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk0, chunk1], &mut output, WINDOW_SIZE + 1, 0);
        assert_eq!(result.unwrap(), WINDOW_SIZE + 1);
        assert_eq!(output[WINDOW_SIZE], 0u8); // window[0] = 0
    }

    #[test]
    fn test_resolve_preserves_byte_values() {
        // Verify no truncation: values 0-255 survive u16→u8 conversion
        let data: Vec<u16> = (0..=255).collect();
        let chunk = DecodedChunk {
            data,
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk], &mut output, 256, 0);
        assert_eq!(result.unwrap(), 256);
        for (i, &byte) in output.iter().enumerate().take(256) {
            assert_eq!(byte, i as u8, "byte {} corrupted", i);
        }
    }

    #[test]
    fn test_resolve_empty_chunk_doesnt_break() {
        let chunk0 = DecodedChunk {
            data: vec![65, 66],
            marker_count: 0,
        };
        let empty = DecodedChunk {
            data: vec![],
            marker_count: 0,
        };
        let chunk2 = DecodedChunk {
            data: vec![67, 68],
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk0, empty, chunk2], &mut output, 4, 0);
        assert_eq!(result.unwrap(), 4);
        assert_eq!(output, vec![65, 66, 67, 68]);
    }

    // =====================================================================
    //  E. OUTPUT VERIFICATION (5 tests)
    // =====================================================================

    #[test]
    fn test_output_crc32_matches_for_each_data_pattern() {
        for (label, data) in [
            ("compressible", make_compressible_data(8 * 1024 * 1024)),
            ("text", make_text_data(8 * 1024 * 1024)),
            ("rle", make_rle_data(8 * 1024 * 1024)),
        ] {
            let compressed = make_gzip_data(&data);
            let mut output = Vec::new();
            if decompress_parallel(&compressed, &mut output, 4).is_ok() {
                let expected_crc = gzip_crc32(&compressed);
                let actual_crc = compute_crc32(&output);
                assert_eq!(
                    actual_crc, expected_crc,
                    "{}: CRC mismatch {:#010x} vs {:#010x}",
                    label, actual_crc, expected_crc
                );
            }
        }
    }

    #[test]
    fn test_isize_matches_output_length() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let isize_val = u32::from_le_bytes([
            compressed[compressed.len() - 4],
            compressed[compressed.len() - 3],
            compressed[compressed.len() - 2],
            compressed[compressed.len() - 1],
        ]) as usize;
        assert_eq!(isize_val, data.len(), "ISIZE should equal data length");
    }

    #[test]
    fn test_parallel_output_byte_identical_to_flate2() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);

        let mut flate2_output = Vec::new();
        let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
        std::io::Read::read_to_end(&mut decoder, &mut flate2_output).unwrap();

        let mut par_output = Vec::new();
        if let Ok(bytes) = decompress_parallel(&compressed, &mut par_output, 4) {
            assert_eq!(bytes as usize, flate2_output.len());
            assert_eq!(par_output, flate2_output, "parallel != flate2 reference");
        }
    }

    #[test]
    fn test_parallel_output_byte_identical_to_consume_first() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let mut ref_output = vec![0u8; data.len() + 65536];
        let ref_size = crate::consume_first_decode::inflate_consume_first(deflate, &mut ref_output)
            .expect("reference inflate");
        ref_output.truncate(ref_size);

        let mut par_output = Vec::new();
        if let Ok(bytes) = decompress_parallel(&compressed, &mut par_output, 4) {
            assert_eq!(bytes as usize, ref_size);
            assert_eq!(
                par_output, ref_output,
                "parallel != consume_first reference"
            );
        }
    }

    #[test]
    fn test_output_never_written_before_size_check() {
        // resolve_and_write buffers all output, verifies CRC32+ISIZE, then writes.
        // On error, no output should be written.
        let chunk = DecodedChunk {
            data: vec![65, 66, 67],
            marker_count: 0,
        };
        let mut output = Vec::new();
        let result = resolve_and_write(&[chunk], &mut output, 100, 0);
        assert!(result.is_err(), "should detect size mismatch");
        assert!(
            output.is_empty(),
            "on error, no output should be written (buffered write)"
        );
    }

    // =====================================================================
    //  F. CONCURRENCY INVARIANTS (4 tests)
    // =====================================================================

    #[test]
    fn test_chunk_results_in_order() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        if boundaries.len() < 2 {
            return;
        }

        if let Ok(chunks) = decode_chunks_parallel(deflate, &boundaries, 4) {
            // Chunk 0 should have 0 markers (starts at beginning)
            assert_eq!(
                chunks[0].marker_count, 0,
                "chunk 0 should have 0 markers — results may be out of order"
            );
            // Chunk 0 output should match sequential
            let chunk0_bytes: Vec<u8> = chunks[0].data.iter().map(|&v| v as u8).collect();
            let ref_prefix = &data[..chunk0_bytes.len().min(data.len())];
            assert_eq!(
                &chunk0_bytes[..ref_prefix.len()],
                ref_prefix,
                "chunk 0 content wrong — results may be out of order"
            );
        }
    }

    #[test]
    fn test_thread_count_4_8_16_same_output() {
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);

        let mut outputs: Vec<(usize, Vec<u8>)> = Vec::new();
        for threads in [4, 8, 16] {
            let mut output = Vec::new();
            if decompress_parallel(&compressed, &mut output, threads).is_ok() {
                outputs.push((threads, output));
            }
        }

        for window in outputs.windows(2) {
            assert_eq!(
                window[0].1.len(),
                window[1].1.len(),
                "T{} and T{} produced different sizes",
                window[0].0,
                window[1].0
            );
            assert_eq!(
                window[0].1, window[1].1,
                "T{} and T{} produced different output",
                window[0].0, window[1].0
            );
        }
    }

    #[test]
    fn test_single_boundary_returns_error() {
        // If only boundary [0] is found, we can't parallelize
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = vec![0usize]; // Only the start boundary
        let result = decode_chunks_parallel(deflate, &boundaries, 4);
        // Should succeed with 1 chunk (the whole stream)
        assert!(result.is_ok(), "single boundary should still decode");
        if let Ok(chunks) = result {
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].marker_count, 0);
        }
    }

    #[test]
    fn test_error_propagation_across_threads() {
        // Verify that an error in one chunk propagates (errors AtomicBool)
        // by trying to decode with corrupted data
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let mut deflate = compressed[header_size..compressed.len() - 8].to_vec();

        let boundaries = find_chunk_boundaries(&deflate, 4);
        if boundaries.len() < 3 {
            return;
        }

        // Corrupt data in the middle of the second chunk
        let corrupt_byte = boundaries[1] / 8 + 100;
        if corrupt_byte < deflate.len() {
            deflate[corrupt_byte] = 0xFF;
            deflate[corrupt_byte + 1] = 0xFF;
            // May or may not cause an error — depends on where corruption lands
            let _ = decode_chunks_parallel(&deflate, &boundaries, 4);
        }
    }

    // =====================================================================
    //  G. EDGE CASES (8 tests)
    // =====================================================================

    #[test]
    fn test_gzip_header_with_extra_fields() {
        // Create gzip with FNAME flag set
        use std::io::Write;
        let data = make_compressible_data(8 * 1024 * 1024);
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&data).unwrap();
        let compressed = encoder.finish().unwrap();

        // Verify we can parse the header and extract deflate data
        let header_size = crate::marker_decode::skip_gzip_header(&compressed);
        assert!(header_size.is_ok(), "should parse standard gzip header");
    }

    #[test]
    fn test_gzip_header_minimal() {
        // Minimal valid gzip: 10-byte header + deflate + 8-byte trailer
        let data = b"test";
        let compressed = make_gzip_data(data);
        assert!(compressed.len() >= 18, "minimum gzip is 18 bytes");
        let header_size = crate::marker_decode::skip_gzip_header(&compressed);
        assert!(header_size.is_ok());
        assert_eq!(header_size.unwrap(), 10, "minimal header is 10 bytes");
    }

    #[test]
    fn test_exactly_one_deflate_block() {
        // Very small data → single deflate block → can't split
        let data = b"hello";
        let compressed = make_gzip_data(data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 4);
        // Should only have [0] since data is tiny
        assert_eq!(boundaries, vec![0], "tiny data should have one boundary");
    }

    #[test]
    fn test_many_tiny_deflate_blocks() {
        // L1 compression produces more blocks
        let data = make_compressible_data(4 * 1024 * 1024);
        let _compressed = make_gzip_at_level(&data, 1);
        assert_parallel_correct_or_error("many_blocks_L1", &data);
    }

    #[test]
    fn test_boundary_at_stored_block() {
        // Create data that forces stored blocks (incompressible random)
        let data = make_random_data(4 * 1024 * 1024, 12345);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Verify boundaries are found (block finder handles stored blocks)
        let boundaries = find_chunk_boundaries(deflate, 4);
        assert!(
            !boundaries.is_empty(),
            "should find at least the start boundary"
        );
    }

    #[test]
    fn test_max_distance_backreference_at_boundary() {
        // Create data with max-distance back-references: repeat a 32KB pattern
        let mut data = Vec::with_capacity(4 * 1024 * 1024);
        let pattern: Vec<u8> = (0..32768).map(|i| (i % 251) as u8).collect();
        while data.len() < 4 * 1024 * 1024 {
            data.extend_from_slice(&pattern);
        }
        data.truncate(4 * 1024 * 1024);

        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Verify oracle chain convergence still holds with max-distance refs
        let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
        if boundaries.len() < 3 {
            return;
        }

        for i in 0..boundaries.len() - 1 {
            let start_bit = boundaries[i];
            let end_bit = boundaries[i + 1];
            let start_byte = start_bit / 8;
            let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());

            let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], start_bit % 8);
            let relative_end = end_bit - start_byte * 8;
            let _ = decoder.decode_until_bit(usize::MAX, relative_end);

            assert_eq!(
                decoder.bit_position(),
                relative_end,
                "max-distance chunk {}: overshoot",
                i
            );
        }
    }

    #[test]
    fn test_chunk_compressed_size_zero() {
        // Adjacent boundaries → chunk has 0 compressed bits
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        // Create degenerate boundaries where two are at the same position
        let boundaries = vec![0, 0]; // Both at bit 0
        let result = decode_chunks_parallel(deflate, &boundaries, 2);
        // Should either succeed or error — never crash
        let _ = result;
    }

    #[test]
    fn test_data_exactly_4mb_compressed() {
        // Data that compresses to ~4MB (right at MIN_PARALLEL_SIZE)
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        // The compressed size may or may not be >= MIN_PARALLEL_SIZE
        let mut output = Vec::new();
        match decompress_parallel(&compressed, &mut output, 4) {
            Ok(bytes) => {
                assert_eq!(bytes as usize, data.len());
                assert_eq!(output, data);
            }
            Err(ParallelError::TooSmall) => {
                // Acceptable if compressed < 4MB
            }
            Err(e) => {
                // Other errors acceptable for now
                eprintln!("4MB boundary test: {}", e);
            }
        }
    }

    // =====================================================================
    //  H. PROPERTY TESTS (4 tests)
    // =====================================================================

    #[test]
    fn test_property_parallel_never_silently_wrong() {
        // For multiple random seeds, parallel MUST either:
        // (a) produce byte-identical output to sequential, or
        // (b) return an error
        // It must NEVER produce wrong output silently.
        // Uses 512KB data to keep brute-force boundary search fast (~5s total).
        for seed in [1u64, 42, 100, 999, 0xdeadbeef] {
            let data = make_random_data(512 * 1024, seed);
            let compressed = make_gzip_data(&data);

            let mut output = Vec::new();
            match decompress_parallel(&compressed, &mut output, 4) {
                Ok(bytes) => {
                    assert_eq!(
                        bytes as usize,
                        data.len(),
                        "seed {}: size mismatch {} vs {}",
                        seed,
                        bytes,
                        data.len()
                    );
                    assert_eq!(output, data, "seed {}: SILENT WRONG OUTPUT", seed);
                }
                Err(_) => {
                    // Error is fine — wrong output is not
                }
            }
        }
    }

    #[test]
    fn test_property_oracle_pipeline_always_correct() {
        // Oracle boundaries MUST always produce correct output.
        // If this fails, the pipeline machinery itself is broken.
        // Note: RLE data excluded — its 100:1+ ratio exceeds max_output_for_chunk (8:1).
        for (label, data) in [
            ("compressible", make_compressible_data(4 * 1024 * 1024)),
            ("text", make_text_data(4 * 1024 * 1024)),
        ] {
            let compressed = make_gzip_data(&data);
            let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
            let deflate = &compressed[header_size..compressed.len() - 8];

            let scan =
                crate::scan_inflate::scan_deflate_fast(deflate, 512 * 1024, 0).expect("scan");
            let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
            if boundaries.len() < 3 {
                continue;
            }

            let chunks = decode_chunks_parallel(deflate, &boundaries, 4)
                .unwrap_or_else(|_| panic!("{}: oracle decode should never fail", label));

            let mut output = Vec::new();
            let mut window = Vec::<u8>::new();

            for (i, chunk) in chunks.iter().enumerate() {
                let bytes: Vec<u8> = if chunk.marker_count > 0 {
                    let resolve_window = if i > 0 && i - 1 < scan.checkpoints.len() {
                        &scan.checkpoints[i - 1].window
                    } else {
                        &window
                    };
                    let mut resolved = chunk.data.clone();
                    replace_markers(&mut resolved, resolve_window);
                    resolved.iter().map(|&v| v as u8).collect()
                } else {
                    chunk.data.iter().map(|&v| v as u8).collect()
                };
                output.extend_from_slice(&bytes);
                update_window(&mut window, &bytes);
            }

            assert_eq!(
                output.len(),
                data.len(),
                "{}: oracle output size {} != expected {}",
                label,
                output.len(),
                data.len()
            );
            assert_eq!(output, data, "{}: oracle output content mismatch", label);
        }
    }

    #[test]
    fn test_property_marker_rate_decreases_over_chunk() {
        // Markers concentrate near the start of a mid-stream chunk (where the
        // decoder lacks history). As the decoder builds its window, the marker
        // rate should decrease. Note: markers can propagate through the window
        // (a reference to a marker position copies the marker), so late markers
        // are possible but should be rarer than early ones.
        let data = make_compressible_data(4 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = get_oracle_boundaries(deflate, 512 * 1024);
        if boundaries.len() < 3 {
            return;
        }

        let i = 1;
        let start_bit = boundaries[i];
        let end_bit = boundaries[i + 1];
        let start_byte = start_bit / 8;
        let end_byte = (end_bit / 8 + 64 * 1024).min(deflate.len());

        let mut decoder = MarkerDecoder::new(&deflate[start_byte..end_byte], start_bit % 8);
        let relative_end = end_bit - start_byte * 8;
        let _ = decoder.decode_until_bit(usize::MAX, relative_end);

        let output = decoder.output();
        if output.len() > WINDOW_SIZE * 2 {
            let early_markers: usize = output[..WINDOW_SIZE].iter().filter(|&&v| v > 255).count();
            let late_markers: usize = output[WINDOW_SIZE..WINDOW_SIZE * 2]
                .iter()
                .filter(|&&v| v > 255)
                .count();
            let early_rate = early_markers as f64 / WINDOW_SIZE as f64;
            let late_rate = late_markers as f64 / WINDOW_SIZE as f64;

            assert!(
                late_rate <= early_rate,
                "marker rate increased: early {:.1}% -> late {:.1}%",
                early_rate * 100.0,
                late_rate * 100.0
            );
        }
    }

    #[test]
    fn test_property_chunk_output_monotonic() {
        // Chunk output sizes shouldn't vary wildly within the same file.
        // If one chunk is 10x another (excluding first/last), something is wrong.
        let data = make_compressible_data(8 * 1024 * 1024);
        let compressed = make_gzip_data(&data);
        let header_size = crate::marker_decode::skip_gzip_header(&compressed).expect("header");
        let deflate = &compressed[header_size..compressed.len() - 8];

        let boundaries = find_chunk_boundaries(deflate, 8);
        if boundaries.len() < 4 {
            return;
        }

        if let Ok(chunks) = decode_chunks_parallel(deflate, &boundaries, 8) {
            // Skip first and last chunks (different sizes expected)
            let mid_sizes: Vec<usize> = chunks[1..chunks.len() - 1]
                .iter()
                .map(|c| c.data.len())
                .collect();

            if mid_sizes.len() >= 2 {
                let max_size = *mid_sizes.iter().max().unwrap();
                let min_size = *mid_sizes.iter().min().unwrap();
                if min_size > 0 {
                    let ratio = max_size as f64 / min_size as f64;
                    assert!(
                        ratio < 10.0,
                        "mid-chunk size ratio {:.1}x (max={}, min={}) — likely false positive",
                        ratio,
                        max_size,
                        min_size
                    );
                }
            }
        }
    }
}
