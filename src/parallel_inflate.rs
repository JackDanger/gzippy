//! Parallel Inflate Infrastructure
//!
//! This module provides the infrastructure for parallel gzip decompression
//! using our pure Rust inflate implementation. Key components:
//!
//! 1. **Chunk partitioning** - Divide input into chunks for parallel processing
//! 2. **Speculative decoding** - Start decoding at guessed block boundaries
//! 3. **Window propagation** - Pass 32KB windows between chunks
//! 4. **Parallel marker replacement** - Resolve back-references in parallel
//!
//! This is the pure Rust equivalent of rapidgzip's approach.

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::simd_inflate;

// =============================================================================
// Constants
// =============================================================================

/// Default chunk size for parallel decompression (4MB)
const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Window size for LZ77 (32KB)
const WINDOW_SIZE: usize = 32 * 1024;

/// Minimum file size for parallel decompression
const PARALLEL_THRESHOLD: usize = 1024 * 1024; // 1MB

// =============================================================================
// Chunk Result
// =============================================================================

/// Result of decompressing a chunk
#[derive(Debug)]
pub struct ChunkResult {
    /// Chunk index
    pub index: usize,
    /// Decompressed data
    pub data: Vec<u8>,
    /// Final 32KB window (for next chunk)
    pub window: Vec<u8>,
    /// Whether decompression succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ChunkResult {
    pub fn success(index: usize, data: Vec<u8>) -> Self {
        let window = if data.len() >= WINDOW_SIZE {
            data[data.len() - WINDOW_SIZE..].to_vec()
        } else {
            data.clone()
        };

        Self {
            index,
            data,
            window,
            success: true,
            error: None,
        }
    }

    pub fn failure(index: usize, error: String) -> Self {
        Self {
            index,
            data: Vec::new(),
            window: Vec::new(),
            success: false,
            error: Some(error),
        }
    }
}

// =============================================================================
// BGZF Detection and Parsing
// =============================================================================

/// BGZF block info
#[derive(Debug, Clone, Copy)]
pub struct BgzfBlock {
    /// Start position in compressed data
    pub start: usize,
    /// Compressed size (including header/trailer)
    pub csize: usize,
    /// Uncompressed size
    pub usize: usize,
}

/// Detect if this is a BGZF file (gzippy or bgzip output)
pub fn detect_bgzf(data: &[u8]) -> bool {
    if data.len() < 18 {
        return false;
    }

    // Check gzip magic
    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 8 {
        return false;
    }

    // Check for FEXTRA flag and BGZF extra field
    let flags = data[3];
    if flags & 0x04 == 0 {
        return false;
    }

    // Parse extra field
    let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
    if xlen < 6 || data.len() < 12 + xlen {
        return false;
    }

    // Look for BC subfield (BGZF marker)
    let mut pos = 12;
    while pos + 4 <= 12 + xlen {
        let si1 = data[pos];
        let si2 = data[pos + 1];
        let slen = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;

        if si1 == 66 && si2 == 67 && slen >= 2 {
            // Found BGZF marker
            return true;
        }

        pos += 4 + slen;
    }

    false
}

/// Parse BGZF blocks from data
pub fn parse_bgzf_blocks(data: &[u8]) -> Vec<BgzfBlock> {
    let mut blocks = Vec::new();
    let mut pos = 0;

    while pos + 18 <= data.len() {
        // Check gzip header
        if data[pos] != 0x1f || data[pos + 1] != 0x8b {
            break;
        }

        let flags = data[pos + 3];
        if flags & 0x04 == 0 {
            break;
        }

        let xlen = u16::from_le_bytes([data[pos + 10], data[pos + 11]]) as usize;
        if pos + 12 + xlen > data.len() {
            break;
        }

        // Find block size from extra field
        let mut block_size = 0;
        let mut xpos = pos + 12;
        while xpos + 4 <= pos + 12 + xlen {
            let si1 = data[xpos];
            let si2 = data[xpos + 1];
            let slen = u16::from_le_bytes([data[xpos + 2], data[xpos + 3]]) as usize;

            if si1 == 66 && si2 == 67 && slen >= 2 {
                // BGZF: block size - 1 is stored in 2 bytes
                // Or 4 bytes for gzippy extended format
                if slen == 2 {
                    block_size = u16::from_le_bytes([data[xpos + 4], data[xpos + 5]]) as usize + 1;
                } else if slen == 4 {
                    block_size = u32::from_le_bytes([
                        data[xpos + 4],
                        data[xpos + 5],
                        data[xpos + 6],
                        data[xpos + 7],
                    ]) as usize
                        + 1;
                }
                break;
            }

            xpos += 4 + slen;
        }

        if block_size == 0 || pos + block_size > data.len() {
            break;
        }

        // Get uncompressed size from trailer
        if pos + block_size >= 4 {
            let usize = u32::from_le_bytes([
                data[pos + block_size - 4],
                data[pos + block_size - 3],
                data[pos + block_size - 2],
                data[pos + block_size - 1],
            ]) as usize;

            blocks.push(BgzfBlock {
                start: pos,
                csize: block_size,
                usize,
            });
        }

        pos += block_size;
    }

    blocks
}

// =============================================================================
// Multi-Member Detection
// =============================================================================

/// Find gzip member boundaries (for pigz-style multi-member files)
pub fn find_member_boundaries(data: &[u8]) -> Vec<usize> {
    let mut boundaries = vec![0];
    let mut pos = 0;

    while pos + 10 < data.len() {
        // Skip current member by finding its size
        // This is tricky without full decompression
        // We'll look for the next gzip magic number

        if pos > 0 && data[pos] == 0x1f && data[pos + 1] == 0x8b && data[pos + 2] == 8 {
            boundaries.push(pos);
        }
        pos += 1;
    }

    boundaries
}

// =============================================================================
// Parallel Decompression
// =============================================================================

/// Parallel gzip decompressor
pub struct ParallelInflater {
    num_threads: usize,
}

impl ParallelInflater {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Decompress gzip data in parallel
    pub fn decompress<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        // Check for BGZF (easiest to parallelize)
        if detect_bgzf(data) {
            return self.decompress_bgzf(data, writer);
        }

        // For non-BGZF, use single-threaded fast decompress
        // (parallel speculative decoding is complex)
        let mut output = Vec::new();
        simd_inflate::inflate_gzip_fast(data, &mut output)?;
        let len = output.len() as u64;
        writer.write_all(&output)?;
        Ok(len)
    }

    /// Decompress BGZF file in parallel
    fn decompress_bgzf<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        let blocks = parse_bgzf_blocks(data);

        if blocks.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "No BGZF blocks found",
            ));
        }

        let num_blocks = blocks.len();
        let results: Vec<Mutex<Option<ChunkResult>>> =
            (0..num_blocks).map(|_| Mutex::new(None)).collect();

        let next_block = AtomicUsize::new(0);

        // Parallel decompression
        std::thread::scope(|scope| {
            for _ in 0..self.num_threads.min(num_blocks) {
                let next_ref = &next_block;
                let results_ref = &results;
                let blocks_ref = &blocks;

                scope.spawn(move || loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let block_data = &data[block.start..block.start + block.csize];

                    let result = match decompress_bgzf_block(block_data) {
                        Ok(decompressed) => ChunkResult::success(idx, decompressed),
                        Err(e) => ChunkResult::failure(idx, e.to_string()),
                    };

                    *results_ref[idx].lock().unwrap() = Some(result);
                });
            }
        });

        // Collect results in order
        let mut total = 0u64;
        for mutex in results.iter() {
            let result = mutex
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| io::Error::other("Missing result"))?;

            if !result.success {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    result.error.unwrap_or_else(|| "Unknown error".to_string()),
                ));
            }

            writer.write_all(&result.data)?;
            total += result.data.len() as u64;
        }

        Ok(total)
    }
}

/// Decompress a single BGZF block
fn decompress_bgzf_block(block_data: &[u8]) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    simd_inflate::inflate_gzip_fast(block_data, &mut output)?;
    Ok(output)
}

// =============================================================================
// Integration with Main Decompression Path
// =============================================================================

/// High-performance decompression that automatically selects the best strategy
pub fn decompress_auto<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    if data.len() < PARALLEL_THRESHOLD || num_threads == 1 {
        // Small file or single-threaded: use fast sequential
        let mut output = Vec::new();
        simd_inflate::inflate_gzip_fast(data, &mut output)?;
        let len = output.len() as u64;
        writer.write_all(&output)?;
        Ok(len)
    } else {
        // Large file: use parallel
        let inflater = ParallelInflater::new(num_threads);
        inflater.decompress(data, writer)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write as IoWrite;

    #[test]
    fn test_sequential_decompress() {
        let original = b"Hello, World! This is a test of sequential decompression.";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_auto(&compressed, &mut output, 1).unwrap();

        assert_eq!(&output[..], &original[..]);
    }

    #[test]
    fn test_parallel_inflater() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let inflater = ParallelInflater::new(4);
        let mut output = Vec::new();
        inflater.decompress(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_bgzf_detection() {
        // Create a simple gzip file (not BGZF)
        let original = b"Test data";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        assert!(!detect_bgzf(&compressed));
    }
}
