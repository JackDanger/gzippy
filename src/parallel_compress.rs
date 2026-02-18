//! High-performance parallel gzip compression
//!
//! This module implements parallel compression using memory-mapped I/O and
//! our custom zero-overhead scheduler (no rayon).
//!
//! For large files, we use block-based parallel compression where each block
//! is a complete gzip member that can be concatenated.
//!
//! Key optimizations:
//! - Memory-mapped files for zero-copy access (no read_to_end latency)
//! - Custom scheduler with streaming output (no bulk collection)
//! - Thread-local buffer reuse to minimize allocations
//! - 128KB fixed blocks (matches pigz default)
//! - libdeflate for L1-L6 (30-50% faster than zlib-ng)
//! - BGZF-style block size markers in FEXTRA for fast parallel decompression

use crate::scheduler::compress_parallel_independent;
use flate2::Compression;
use memmap2::Mmap;
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// BGZF-style subfield ID for block size markers
/// Using "GZ" to identify gzippy-compressed blocks with embedded block sizes
pub const GZ_SUBFIELD_ID: [u8; 2] = [b'G', b'Z'];

/// Metadata for gzip header FNAME and MTIME fields
#[derive(Clone, Debug, Default)]
pub struct GzipHeaderInfo {
    /// Original filename (basename only) for FNAME field
    pub filename: Option<String>,
    /// File modification time as Unix timestamp for MTIME field
    pub mtime: u32,
    /// Optional comment for FCOMMENT field
    pub comment: Option<String>,
}

/// Adjust compression level for backend compatibility
///
/// For L1-L6, we use libdeflate which doesn't have zlib-ng's L1 RLE issue.
/// For L7-L9, we use zlib-ng which needs L1→L2 mapping.
///
/// However, since L7-L9 never uses L1, this mapping is only relevant for
/// the pipelined compressor which uses zlib-ng directly.
#[inline]
pub fn adjust_compression_level(level: u32) -> u32 {
    // Only needed for zlib-ng (pipelined compressor at L7-L9)
    // Parallel compressor uses libdeflate for L1-L6, no adjustment needed
    if level == 1 {
        2 // Map L1→L2 for zlib-ng (RLE produces 33% larger files)
    } else {
        level
    }
}

/// Get optimal block size based on compression level and file size
///
/// At lower levels (L1-L2), we use larger blocks to reduce per-block overhead
/// since compression is fast and synchronization becomes the bottleneck.
/// At higher levels (L3-L6), we use smaller blocks for better parallelization
/// since compression takes longer and can better utilize the parallelism.
///
/// Block size is also scaled based on file size to ensure enough blocks for
/// parallelism on small files while reducing overhead on large files.
#[inline]
pub fn get_block_size_for_level(level: u32) -> usize {
    match level {
        // L1-L2: Use 128KB blocks as baseline
        // This gives enough parallelism for small files
        // Note: 128KB > BGZF u16 limit, so BGZF markers will be disabled
        1 | 2 => 128 * 1024,
        // L3-L6: Use 64KB blocks - fits BGZF, enables parallel decompression
        3..=6 => DEFAULT_BLOCK_SIZE,
        // L7-L9: Handled by pipelined compressor (shouldn't reach here)
        7..=9 => 128 * 1024,
        // L10-L12: Ultra compression needs larger blocks for better context
        // libdeflate's exhaustive search benefits from more data to find matches
        // Use 512KB blocks - still enables parallelism but gives good ratio
        _ => 512 * 1024,
    }
}

/// Get optimal block size considering both level and file size
/// This ensures we have enough blocks for parallelism (minimum 4*num_threads)
/// while not having too many blocks that synchronization overhead dominates.
#[inline]
pub fn get_optimal_block_size(level: u32, file_size: usize, num_threads: usize) -> usize {
    let base_block_size = get_block_size_for_level(level);

    // For L1-L2, dynamically size blocks based on file size
    // Goal: ~8 blocks per thread for good load balancing, max 256KB blocks
    if level <= 2 {
        let target_blocks = num_threads * 8;
        let dynamic_size = file_size / target_blocks;
        // Clamp between 64KB (minimum for efficiency) and 256KB (maximum for L1-L2)
        let clamped = dynamic_size.clamp(64 * 1024, 256 * 1024);
        // Round up to 64KB boundary for alignment
        (clamped + 65535) & !65535
    } else {
        base_block_size
    }
}

/// Check if a data sample is highly compressible (ratio < 5%).
/// Used to skip parallelism when block overhead exceeds speedup.
fn is_highly_compressible(sample: &[u8], level: u32) -> bool {
    if sample.is_empty() {
        return false;
    }
    let lvl = match libdeflater::CompressionLvl::new(level as i32) {
        Ok(l) => l,
        Err(_) => return false,
    };
    let mut compressor = libdeflater::Compressor::new(lvl);
    let max_size = compressor.deflate_compress_bound(sample.len());
    let mut output = vec![0u8; max_size];
    match compressor.deflate_compress(sample, &mut output) {
        Ok(size) => (size as f64 / sample.len() as f64) < 0.05,
        Err(_) => false,
    }
}

// Thread-local compression buffer to avoid per-block allocation
thread_local! {
    static COMPRESS_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    // Cache libdeflate Compressor by level to avoid per-block allocation
    // Tuple is (level, compressor) - we only cache one level per thread
    static LIBDEFLATE_COMPRESSOR: RefCell<Option<(i32, libdeflater::Compressor)>> = const { RefCell::new(None) };
}

/// Default block size for parallel compression
/// BGZF format stores block size as u16, so max is 65535 bytes
/// We use 64KB to stay within this limit while maximizing parallelism
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

/// Parallel gzip compression using custom scheduler
pub struct ParallelGzEncoder {
    compression_level: u32,
    num_threads: usize,
    header_info: GzipHeaderInfo,
}

impl ParallelGzEncoder {
    pub fn new(compression_level: u32, num_threads: usize) -> Self {
        Self {
            compression_level,
            num_threads,
            header_info: GzipHeaderInfo::default(),
        }
    }

    pub fn set_header_info(&mut self, info: GzipHeaderInfo) {
        self.header_info = info;
    }

    /// Compress data in parallel and write to output
    pub fn compress<R: Read, W: Write + Send>(&self, mut reader: R, writer: W) -> io::Result<u64> {
        // Read all input data
        let mut input_data = Vec::new();
        let bytes_read = reader.read_to_end(&mut input_data)? as u64;

        if input_data.is_empty() {
            // Write empty gzip file - use level 9 for zlib (L10+ aren't supported by zlib)
            let zlib_level = self.compression_level.min(9);
            let encoder = self.gz_builder().write(
                writer,
                Compression::new(adjust_compression_level(zlib_level)),
            );
            encoder.finish()?;
            return Ok(0);
        }

        // Calculate optimal block size
        let block_size = self.calculate_block_size(input_data.len());

        // For small files or single thread, choose the fastest single-member path
        // For L10-L12, always use libdeflate blocks for better compression
        if self.compression_level <= 9 && (input_data.len() <= block_size || self.num_threads == 1)
        {
            let mut w = writer;
            return self
                .compress_single_stream(&input_data, &mut w)
                .map(|_| bytes_read);
        }

        // Ratio probe: skip parallelism for highly compressible data at fast levels.
        // At L1-L5, compression is so fast that block overhead dominates for
        // highly compressible data (e.g., software dataset: Tmax < T1).
        if self.compression_level <= 5 && self.num_threads > 1 {
            let sample = &input_data[..block_size.min(input_data.len())];
            if is_highly_compressible(sample, self.compression_level) {
                let mut w = writer;
                return self
                    .compress_single_stream(&input_data, &mut w)
                    .map(|_| bytes_read);
            }
        }

        // Large file with multiple threads, or L10-L12: compress blocks in parallel with libdeflate
        let _ = self.compress_parallel(&input_data, block_size, writer)?;

        Ok(bytes_read)
    }

    /// Build a flate2 GzBuilder with FNAME/MTIME/FCOMMENT from header_info
    fn gz_builder(&self) -> flate2::GzBuilder {
        let mut builder = flate2::GzBuilder::new();
        if let Some(ref name) = self.header_info.filename {
            builder = builder.filename(name.as_bytes());
        }
        builder = builder.mtime(self.header_info.mtime);
        if let Some(ref comment) = self.header_info.comment {
            builder = builder.comment(comment.as_bytes());
        }
        builder
    }

    /// Calculate optimal block size based on file size and thread count
    fn calculate_block_size(&self, file_size: usize) -> usize {
        get_optimal_block_size(self.compression_level, file_size, self.num_threads)
    }

    /// Compress a file using memory-mapped I/O for zero-copy access
    pub fn compress_file<P: AsRef<Path>, W: Write + Send>(
        &self,
        path: P,
        mut writer: W,
    ) -> io::Result<u64> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;

        if file_len == 0 {
            // Write empty gzip file - use level 9 for zlib (L10+ aren't supported by zlib)
            let zlib_level = self.compression_level.min(9);
            let encoder = self.gz_builder().write(
                &mut writer,
                Compression::new(adjust_compression_level(zlib_level)),
            );
            encoder.finish()?;
            return Ok(0);
        }

        // Memory-map the file for zero-copy access
        let mmap = unsafe { Mmap::map(&file)? };

        // For small files or single thread, choose the fastest single-member path
        // For L10-L12, always use libdeflate blocks for better compression
        let block_size = self.calculate_block_size(file_len);
        if self.compression_level <= 9 && (file_len <= block_size || self.num_threads == 1) {
            return self.compress_single_stream(&mmap, &mut writer);
        }

        // Ratio probe: for multi-threaded L1-L5, check if data is highly compressible.
        // When compression ratio < 5%, parallelism overhead (per-block headers, thread
        // coordination, CRC32) exceeds speedup. Fall back to single-threaded streaming.
        if self.compression_level <= 5 && self.num_threads > 1 {
            let sample = &mmap[..block_size.min(file_len)];
            if is_highly_compressible(sample, self.compression_level) {
                return self.compress_single_stream(&mmap, &mut writer);
            }
        }

        // Large file with multiple threads, or L10-L12: compress blocks in parallel with libdeflate
        let _ = self.compress_parallel(&mmap, block_size, writer)?;

        Ok(file_len as u64)
    }

    /// Single-stream compression using the best available backend
    fn compress_single_stream<W: Write>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        // For L0-L3, use ISA-L when available (fastest on x86 with AVX2)
        if self.compression_level <= 3 && crate::isal_compress::is_available() {
            return crate::isal_compress::compress_gzip_to_writer(
                data,
                &mut *writer,
                self.compression_level,
            );
        }

        // For L1-L5, use libdeflate/ISA-L single member (faster than zlib-ng)
        if self.compression_level >= 1 && self.compression_level <= 5 {
            compress_single_member(
                &mut *writer,
                data,
                self.compression_level,
                &self.header_info,
            )?;
            return Ok(data.len() as u64);
        }

        // L6-L9: flate2/zlib-ng streaming (better ratio at higher levels)
        let mut encoder = self.gz_builder().write(
            &mut *writer,
            Compression::new(adjust_compression_level(self.compression_level)),
        );
        encoder.write_all(data)?;
        encoder.finish()?;
        Ok(data.len() as u64)
    }

    /// Parallel compression using custom scheduler with streaming output
    fn compress_parallel<W: Write + Send>(
        &self,
        data: &[u8],
        block_size: usize,
        writer: W,
    ) -> io::Result<W> {
        let compression_level = self.compression_level;
        let header_info = self.header_info.clone();

        // Probe compression ratio on first block to detect highly compressible data.
        // When ratio < 10%, parallel coordination overhead exceeds the benefit of
        // multi-threading — use flate2/zlib-ng streaming instead (faster for such data
        // due to smaller buffer allocations and better cache behavior).
        if compression_level <= 5 && data.len() > block_size && self.num_threads > 1 {
            let probe_block = &data[..block_size];
            let mut probe_output = Vec::new();
            compress_block_bgzf_libdeflate(
                &mut probe_output,
                probe_block,
                compression_level,
                &header_info,
            );
            let ratio = probe_output.len() as f64 / probe_block.len() as f64;
            if ratio < 0.10 {
                let adjusted_level = adjust_compression_level(compression_level.min(9));
                let mut writer = writer;
                let mut encoder = self
                    .gz_builder()
                    .write(&mut writer, Compression::new(adjusted_level));
                encoder.write_all(data)?;
                encoder.finish()?;
                return Ok(writer);
            }
        }

        // Use ISA-L for levels 0-3 when available (3-5x faster on x86)
        let use_isal = compression_level <= 3 && crate::isal_compress::is_available();

        // Use custom scheduler - dedicated writer thread for max parallelism
        compress_parallel_independent(
            data,
            block_size,
            self.num_threads,
            writer,
            |block, output| {
                if use_isal {
                    compress_block_bgzf_isal(output, block, compression_level, &header_info);
                } else {
                    compress_block_bgzf_libdeflate(output, block, compression_level, &header_info);
                }
            },
        )
    }
}

/// Compress entire input as a single gzip member using the fastest available backend.
///
/// For L0-L3: Uses ISA-L (AVX2/NEON assembly) if available, otherwise libdeflate.
/// For L4-L12: Uses libdeflate.
///
/// The output includes a standard gzip header with FNAME/MTIME and a BGZF-compatible
/// block size marker in FEXTRA. This is valid gzip readable by any decompressor.
pub fn compress_single_member<W: Write>(
    writer: &mut W,
    input: &[u8],
    compression_level: u32,
    header_info: &GzipHeaderInfo,
) -> io::Result<u64> {
    if input.is_empty() {
        let encoder =
            flate2::GzBuilder::new().write(writer, Compression::new(compression_level.min(9)));
        encoder.finish()?;
        return Ok(0);
    }

    let bytes = input.len() as u64;
    let mut output = Vec::with_capacity(input.len());

    // Use ISA-L for L0-L3 if available (3-5x faster on x86 with AVX2)
    if compression_level <= 3 && crate::isal_compress::is_available() {
        compress_block_bgzf_isal(&mut output, input, compression_level, header_info);
    } else {
        compress_block_bgzf_libdeflate(&mut output, input, compression_level, header_info);
    }

    writer.write_all(&output)?;
    Ok(bytes)
}

/// Compress a block with BGZF-style gzip header containing block size
///
/// The header includes:
/// - Standard gzip magic (0x1f 0x8b)
/// - FEXTRA flag set (0x04), optionally FNAME (0x08) and FCOMMENT (0x10)
/// - "GZ" subfield with compressed block size (allows parallel decompression)
/// - MTIME from file metadata (when available)
/// - Original filename (when available)
///
/// This is compatible with all gzip decompressors (they ignore unknown subfields)
/// but enables gzippy to find block boundaries without inflating.
///
/// Uses libdeflate for L1-L6 (faster, no dictionary needed).
fn compress_block_bgzf_libdeflate(
    output: &mut Vec<u8>,
    block: &[u8],
    compression_level: u32,
    header_info: &GzipHeaderInfo,
) {
    use libdeflater::{CompressionLvl, Compressor};

    output.clear();

    // Reserve space for header (we'll write block size later)
    let header_start = output.len();

    // Build flags: FEXTRA always set, optionally FNAME and FCOMMENT
    let mut flags: u8 = 0x04; // FEXTRA
    if header_info.filename.is_some() {
        flags |= 0x08; // FNAME
    }
    if header_info.comment.is_some() {
        flags |= 0x10; // FCOMMENT
    }

    // Write gzip header
    output.extend_from_slice(&[
        0x1f, 0x8b, // Magic
        0x08, // Compression method (deflate)
        flags,
    ]);
    output.extend_from_slice(&header_info.mtime.to_le_bytes()); // MTIME
    output.extend_from_slice(&[
        0x00, // XFL (no extra flags)
        0xff, // OS (unknown)
    ]);

    // FEXTRA: XLEN + subfield data
    // XLEN: 8 bytes (2 byte ID + 2 byte len + 4 byte block size)
    output.extend_from_slice(&[8, 0]);

    // Subfield: "GZ" + 2 bytes len + 4 bytes block size (placeholder)
    // Using 4 bytes allows block sizes up to 4GB
    output.extend_from_slice(&GZ_SUBFIELD_ID);
    output.extend_from_slice(&[4, 0]); // Subfield data length (4 bytes)
    let block_size_offset = output.len();
    output.extend_from_slice(&[0, 0, 0, 0]); // Placeholder for block size

    // FNAME (after FEXTRA, per RFC 1952 order)
    if let Some(ref name) = header_info.filename {
        output.extend_from_slice(name.as_bytes());
        output.push(0); // null terminator
    }

    // FCOMMENT (after FNAME)
    if let Some(ref comment) = header_info.comment {
        output.extend_from_slice(comment.as_bytes());
        output.push(0); // null terminator
    }

    // Get or create compressor from thread-local cache
    let level = compression_level as i32;
    let max_compressed_size = LIBDEFLATE_COMPRESSOR.with(|cache| {
        let mut cache = cache.borrow_mut();

        let compressor = match cache.as_mut() {
            Some((cached_level, comp)) if *cached_level == level => comp,
            _ => {
                let lvl = CompressionLvl::new(level).unwrap_or_default();
                *cache = Some((level, Compressor::new(lvl)));
                &mut cache.as_mut().unwrap().1
            }
        };

        compressor.deflate_compress_bound(block.len())
    });

    // Resize output buffer
    let deflate_start = output.len();
    output.resize(deflate_start + max_compressed_size, 0);

    // Do the actual compression
    let actual_len = LIBDEFLATE_COMPRESSOR.with(|cache| {
        let mut cache = cache.borrow_mut();
        let compressor = &mut cache.as_mut().unwrap().1;
        compressor
            .deflate_compress(block, &mut output[deflate_start..])
            .expect("libdeflate compression failed")
    });

    output.truncate(deflate_start + actual_len);

    // Compute CRC32 of uncompressed data
    let crc32 = crc32fast::hash(block);

    // Write gzip trailer: CRC32 + ISIZE (uncompressed size mod 2^32)
    output.extend_from_slice(&crc32.to_le_bytes());
    output.extend_from_slice(&(block.len() as u32).to_le_bytes());

    // Now write the total block size (including header and trailer)
    let total_block_size = output.len() - header_start;

    // Block size stored as u32 (no overflow possible for reasonable blocks)
    output[block_size_offset..block_size_offset + 4]
        .copy_from_slice(&(total_block_size as u32).to_le_bytes());
}

/// Compress a block using ISA-L for levels 0-3, with BGZF-style header.
/// Uses the same header format as compress_block_bgzf_libdeflate so blocks
/// from either compressor can be mixed.
fn compress_block_bgzf_isal(
    output: &mut Vec<u8>,
    block: &[u8],
    compression_level: u32,
    header_info: &GzipHeaderInfo,
) {
    // ISA-L produces complete gzip members, but we need BGZF-style headers
    // with block size markers. So we:
    // 1. Use ISA-L to get the raw deflate data
    // 2. Wrap it in our BGZF header format
    match crate::isal_compress::compress_deflate(block, compression_level) {
        Some(deflate_data) => {
            output.clear();
            let header_start = output.len();

            // Build flags: FEXTRA always set, optionally FNAME and FCOMMENT
            let mut flags: u8 = 0x04; // FEXTRA
            if header_info.filename.is_some() {
                flags |= 0x08;
            }
            if header_info.comment.is_some() {
                flags |= 0x10;
            }

            // Write gzip header
            output.extend_from_slice(&[0x1f, 0x8b, 0x08, flags]);
            output.extend_from_slice(&header_info.mtime.to_le_bytes());
            output.extend_from_slice(&[0x00, 0xff]);

            // FEXTRA: 8 bytes (2 ID + 2 len + 4 block size)
            output.extend_from_slice(&[8, 0]);
            output.extend_from_slice(&GZ_SUBFIELD_ID);
            output.extend_from_slice(&[4, 0]);
            let block_size_offset = output.len();
            output.extend_from_slice(&[0, 0, 0, 0]); // placeholder

            // FNAME
            if let Some(ref name) = header_info.filename {
                output.extend_from_slice(name.as_bytes());
                output.push(0);
            }

            // FCOMMENT
            if let Some(ref comment) = header_info.comment {
                output.extend_from_slice(comment.as_bytes());
                output.push(0);
            }

            // Deflate data from ISA-L
            output.extend_from_slice(&deflate_data);

            // CRC32 + ISIZE trailer
            let crc32 = crc32fast::hash(block);
            output.extend_from_slice(&crc32.to_le_bytes());
            output.extend_from_slice(&(block.len() as u32).to_le_bytes());

            // Write total block size
            let total_block_size = output.len() - header_start;
            output[block_size_offset..block_size_offset + 4]
                .copy_from_slice(&(total_block_size as u32).to_le_bytes());
        }
        None => {
            // Fall back to libdeflate if ISA-L fails
            compress_block_bgzf_libdeflate(output, block, compression_level, header_info);
        }
    }
}

/// Split data into rsyncable blocks using a rolling hash.
/// Block boundaries are determined by content, so small input changes
/// only affect nearby blocks — ideal for rsync workflows.
///
/// Uses a simple Adler-style rolling hash with a window of 8KB.
/// When the hash's low bits match a trigger mask, a block boundary is created.
/// Target block size is ~128KB (mask = 0x1FFFF = 128K-1).
pub fn split_rsyncable(data: &[u8]) -> Vec<&[u8]> {
    const WINDOW: usize = 8192;
    const MASK: u32 = 0x1FFFF; // ~128KB average block size
    const MIN_BLOCK: usize = 32 * 1024; // 32KB minimum
    const MAX_BLOCK: usize = 512 * 1024; // 512KB maximum

    if data.len() <= MIN_BLOCK {
        return vec![data];
    }

    let mut blocks = Vec::new();
    let mut block_start = 0;
    let mut hash: u32 = 0;

    for i in 0..data.len() {
        // Add new byte to hash
        hash = hash.wrapping_add(data[i] as u32);

        // Remove byte leaving the window
        if i >= WINDOW {
            hash = hash.wrapping_sub(data[i - WINDOW] as u32);
        }

        let block_len = i - block_start + 1;

        // Check for boundary: hash hits trigger AND block is big enough
        if block_len >= MIN_BLOCK && (hash & MASK == MASK || block_len >= MAX_BLOCK) {
            blocks.push(&data[block_start..block_start + block_len]);
            block_start += block_len;
        }
    }

    // Last block
    if block_start < data.len() {
        blocks.push(&data[block_start..]);
    }

    blocks
}

/// Compress data with rsyncable block boundaries.
/// Each content-determined block becomes an independent gzip member.
pub fn compress_rsyncable<W: Write + Send>(
    data: &[u8],
    compression_level: u32,
    num_threads: usize,
    header_info: &GzipHeaderInfo,
    mut writer: W,
) -> io::Result<u64> {
    let blocks = split_rsyncable(data);

    if blocks.is_empty() {
        return Ok(0);
    }

    // For single block or single thread, compress sequentially
    if blocks.len() == 1 || num_threads <= 1 {
        let mut total = 0u64;
        for block in &blocks {
            let mut output = Vec::new();
            compress_block_bgzf_libdeflate(&mut output, block, compression_level, header_info);
            writer.write_all(&output)?;
            total += block.len() as u64;
        }
        return Ok(total);
    }

    // Parallel: compress blocks using thread pool
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let num_blocks = blocks.len();
    let next_block = AtomicUsize::new(0);

    // Pre-allocate output slots
    let outputs: Vec<std::sync::Mutex<Vec<u8>>> = (0..num_blocks)
        .map(|_| std::sync::Mutex::new(Vec::new()))
        .collect();

    thread::scope(|scope| {
        for _ in 0..num_threads.min(num_blocks) {
            scope.spawn(|| loop {
                let idx = next_block.fetch_add(1, Ordering::Relaxed);
                if idx >= num_blocks {
                    break;
                }
                let mut output = outputs[idx].lock().unwrap();
                compress_block_bgzf_libdeflate(
                    &mut output,
                    blocks[idx],
                    compression_level,
                    header_info,
                );
            });
        }
    });

    // Write outputs in order
    let mut total = 0u64;
    for (i, slot) in outputs.iter().enumerate() {
        let output = slot.lock().unwrap();
        writer.write_all(&output)?;
        total += blocks[i].len() as u64;
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parallel_compress_small() {
        let data = b"Hello, world!";
        let encoder = ParallelGzEncoder::new(6, 4);

        let mut output = Vec::new();
        encoder
            .compress(Cursor::new(&data[..]), &mut output)
            .unwrap();

        // Verify output is valid gzip by decompressing
        let mut decoder = flate2::read::GzDecoder::new(&output[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_parallel_compress_large() {
        let data = b"Hello, world! ".repeat(100000); // ~1.4MB
        let encoder = ParallelGzEncoder::new(6, 4);

        let mut output = Vec::new();
        encoder.compress(Cursor::new(&data), &mut output).unwrap();

        // Verify output is valid gzip by decompressing
        // Note: flate2's GzDecoder handles concatenated gzip members
        let mut decoder = flate2::read::MultiGzDecoder::new(&output[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }
}
