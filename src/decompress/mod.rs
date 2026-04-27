//! Gzip decompression engine — pure bytes-in / bytes-out.
//!
//! Entry points for the I/O layer are in `io`. This module handles:
//! classify → route → decompress. All functions are `pub(crate)` so tests
//! can reach them directly.

pub mod bgzf;
pub mod combined_lut;
pub mod format;
pub mod inflate;
pub mod inflate_tables;
pub mod io;
pub mod packed_lut;
pub mod parallel;
pub mod scan_inflate;
pub mod simd_copy;
pub mod simd_huffman;
pub mod two_level_table;

use std::io::Write;

use crate::decompress::format::{has_bgzf_markers, is_likely_multi_member, read_gzip_isize};
use crate::error::{GzippyError, GzippyResult};

const STREAM_BUFFER_SIZE: usize = 1024 * 1024;

#[cfg(target_os = "macos")]
const CACHE_LINE_SIZE: usize = 128;
#[cfg(not(target_os = "macos"))]
const CACHE_LINE_SIZE: usize = 64;

#[inline]
fn alloc_aligned_buffer(size: usize) -> Vec<u8> {
    let aligned = (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
    vec![0u8; aligned]
}

// =============================================================================
// Routing
// =============================================================================

/// The decompression path selected for a given input.
///
/// This is the canonical routing table. `classify_gzip` returns one of these;
/// `decompress_gzip_libdeflate` dispatches on it. To add a new path: add a variant
/// here, a condition in `classify_gzip`, and a dispatch arm below.
///
/// Current paths (in priority order):
///   GzippyParallel   — gzippy-produced multi-block files ("GZ" FEXTRA subfield)
///   MultiMemberPar   — pigz-style multi-member, Tmax threads
///   MultiMemberSeq   — pigz-style multi-member, T1
///   IsalSingle       — x86_64 single-member via ISA-L (fastest sequential)
///   StreamingSingle  — single-member > 1GB, no ISA-L (avoids huge allocation)
///   LibdeflateSingle — default single-member via libdeflate one-shot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    GzippyParallel,
    MultiMemberPar,
    MultiMemberSeq,
    IsalSingle,
    StreamingSingle,
    LibdeflateSingle,
}

/// Classify a gzip input into the optimal `DecodePath`.
///
/// Single source of truth for routing. All classification logic lives here;
/// `decompress_gzip_libdeflate` only dispatches on the result.
pub fn classify_gzip(data: &[u8], num_threads: usize) -> DecodePath {
    if has_bgzf_markers(data) {
        return DecodePath::GzippyParallel;
    }
    if is_likely_multi_member(data) {
        return if num_threads > 1 {
            DecodePath::MultiMemberPar
        } else {
            DecodePath::MultiMemberSeq
        };
    }
    if crate::backends::isal_decompress::is_available() {
        return DecodePath::IsalSingle;
    }
    if data.len() > 1024 * 1024 * 1024 {
        return DecodePath::StreamingSingle;
    }
    DecodePath::LibdeflateSingle
}

// =============================================================================
// Public entry point
// =============================================================================

/// Decompress gzip data to an arbitrary writer. Used by --test mode and
/// callers that need direct access without going through the file I/O layer.
pub fn decompress_gzip_to_writer<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    decompress_gzip_libdeflate(data, writer, num_threads)
}

// =============================================================================
// Core engine — all pub(crate) so tests can reach them directly
// =============================================================================

/// Route and decompress a gzip byte slice to `writer`.
pub(crate) fn decompress_gzip_libdeflate<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(0);
    }

    let path = classify_gzip(data, num_threads);

    if crate::utils::debug_enabled() {
        eprintln!(
            "[gzippy] path={:?} threads={} bytes={}",
            path,
            num_threads,
            data.len()
        );
    }

    match path {
        DecodePath::GzippyParallel => {
            let bytes =
                crate::decompress::bgzf::decompress_bgzf_parallel(data, writer, num_threads)?;
            Ok(bytes)
        }
        DecodePath::MultiMemberPar => {
            let bytes = crate::decompress::bgzf::decompress_multi_member_parallel(
                data,
                writer,
                num_threads,
            )?;
            Ok(bytes)
        }
        DecodePath::MultiMemberSeq => decompress_multi_member_sequential(data, writer),
        DecodePath::IsalSingle | DecodePath::StreamingSingle | DecodePath::LibdeflateSingle => {
            decompress_single_member(data, writer, num_threads)
        }
    }
}

/// Decompress gzip to an owned Vec. Used by the I/O layer for parallel paths
/// that benefit from a Vec intermediate (e.g. multi-member Tmax).
pub(crate) fn decompress_gzip_to_vec(data: &[u8], num_threads: usize) -> GzippyResult<Vec<u8>> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(Vec::new());
    }
    if has_bgzf_markers(data) {
        return Ok(crate::decompress::bgzf::decompress_bgzf_parallel_to_vec(
            data,
            num_threads,
        )?);
    }
    if num_threads > 1 && is_likely_multi_member(data) {
        return Ok(
            crate::decompress::bgzf::decompress_multi_member_parallel_to_vec(data, num_threads)?,
        );
    }
    let mut output = Vec::new();
    decompress_gzip_libdeflate(data, &mut output, num_threads)?;
    Ok(output)
}

/// Route to the best available single-member decoder.
pub(crate) fn decompress_single_member<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> GzippyResult<u64> {
    // Parallel single-member path: speculative block-boundary search + ISA-L
    // re-decode per confirmed chunk via inflatePrime. x86_64 + ISA-L only.
    // Arm64: sequential libdeflate (~14,000 MB/s) beats parallel zlib-ng (4 × 600 = 2,400 MB/s).
    const MIN_PARALLEL_COMPRESSED: usize = 10 * 1024 * 1024;
    if crate::backends::isal_decompress::is_available()
        && num_threads > 1
        && data.len() > MIN_PARALLEL_COMPRESSED
    {
        match crate::decompress::parallel::single_member::decompress_parallel(
            data,
            writer,
            num_threads,
        ) {
            Ok(n) => {
                writer.flush()?;
                return Ok(n);
            }
            Err(_) => {
                if crate::utils::debug_enabled() {
                    eprintln!("[gzippy] parallel single-member failed, falling back to sequential");
                }
            }
        }
    }

    if crate::backends::isal_decompress::is_available() {
        if let Some(bytes) = crate::backends::isal_decompress::decompress_gzip_stream(data, writer)
        {
            writer.flush()?;
            return Ok(bytes);
        }
        if crate::utils::debug_enabled() {
            eprintln!(
                "[gzippy] WARNING: ISA-L decompress failed on {} bytes, using libdeflate",
                data.len()
            );
        }
    }
    if !crate::backends::isal_decompress::is_available() && data.len() > 1024 * 1024 * 1024 {
        if crate::utils::debug_enabled() {
            eprintln!("[gzippy] streaming zlib-ng decode: {} bytes", data.len());
        }
        return decompress_single_member_streaming(data, writer);
    }
    decompress_single_member_libdeflate(data, writer)
}

/// Streaming single-member decompress via flate2/zlib-ng. Fixed 1MB buffer —
/// avoids the page-fault overhead of allocating a full-size output buffer.
pub(crate) fn decompress_single_member_streaming<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(data);
    let mut buf = vec![0u8; STREAM_BUFFER_SIZE];
    let mut total = 0u64;
    loop {
        let n = decoder.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        total += n as u64;
    }
    writer.flush()?;
    Ok(total)
}

/// Single-member decompress via libdeflate FFI (fastest path).
/// Uses the ISIZE trailer hint for initial buffer sizing; grows and retries
/// on InsufficientSpace.
pub(crate) fn decompress_single_member_libdeflate<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    use crate::backends::libdeflate::{DecompressError, DecompressorEx};
    let mut decompressor = DecompressorEx::new();
    let isize_hint = read_gzip_isize(data).unwrap_or(0) as usize;
    let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(64 * 1024)
    };
    let mut output = alloc_aligned_buffer(initial_size);
    loop {
        match decompressor.gzip_decompress_ex(data, &mut output) {
            Ok(result) => {
                writer.write_all(&output[..result.output_size])?;
                writer.flush()?;
                return Ok(result.output_size as u64);
            }
            Err(DecompressError::InsufficientSpace) => {
                let new_size = output.len().saturating_mul(2);
                output.resize(new_size, 0);
            }
            Err(DecompressError::BadData) => {
                return Err(GzippyError::invalid_argument(
                    "invalid gzip data".to_string(),
                ));
            }
        }
    }
}

/// Sequential multi-member decompress via libdeflate. Uses `gzip_decompress_ex`
/// which returns `input_consumed` so we can step through members without
/// re-scanning.
pub(crate) fn decompress_multi_member_sequential<W: Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    use crate::backends::libdeflate::{DecompressError, DecompressorEx};
    let mut decompressor = DecompressorEx::new();
    let mut total_bytes = 0u64;
    let mut offset = 0;
    let mut member_count = 0u32;
    let isize_hint = read_gzip_isize(data).unwrap_or(0) as usize;
    let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(256 * 1024)
    };
    let mut output_buf = alloc_aligned_buffer(initial_size);

    while offset < data.len() {
        if data.len() - offset < 10 {
            break;
        }
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }
        let remaining = &data[offset..];
        let min_size = remaining.len().max(128 * 1024);
        if output_buf.len() < min_size {
            output_buf.resize(min_size, 0);
        }
        let mut success = false;
        loop {
            match decompressor.gzip_decompress_ex(remaining, &mut output_buf) {
                Ok(result) => {
                    member_count += 1;
                    if crate::utils::debug_enabled() {
                        eprintln!(
                            "[gzippy] sequential member {}: in_consumed={} out_size={} offset={}/{}",
                            member_count,
                            result.input_consumed,
                            result.output_size,
                            offset,
                            data.len()
                        );
                    }
                    writer.write_all(&output_buf[..result.output_size])?;
                    total_bytes += result.output_size as u64;
                    offset += result.input_consumed;
                    success = true;
                    break;
                }
                Err(DecompressError::InsufficientSpace) => {
                    let new_size = output_buf.len().saturating_mul(2);
                    output_buf.resize(new_size, 0);
                }
                Err(DecompressError::BadData) => {
                    break;
                }
            }
        }
        if !success {
            break;
        }
    }
    writer.flush()?;
    Ok(total_bytes)
}

/// Decompress zlib data (2-byte header + deflate + 4-byte Adler32).
pub(crate) fn decompress_zlib_turbo<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    if data.len() < 6 {
        return Err(GzippyError::invalid_argument(
            "Zlib data too short".to_string(),
        ));
    }
    let deflate_data = &data[2..data.len() - 4];
    let mut output_buf = vec![0u8; data.len().saturating_mul(4).max(64 * 1024)];
    match crate::decompress::bgzf::inflate_into_pub(deflate_data, &mut output_buf) {
        Ok(size) => {
            writer.write_all(&output_buf[..size])?;
            writer.flush()?;
            Ok(size as u64)
        }
        Err(e) => Err(GzippyError::invalid_argument(format!(
            "zlib decompression failed: {}",
            e
        ))),
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
    use std::io::Write;

    #[test]
    fn test_decompress_multi_member_file() {
        let part1: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let part2: Vec<u8> = (0..100_000).map(|i| ((i + 50) % 256) as u8).collect();

        let mut enc1 = GzEncoder::new(Vec::new(), Compression::default());
        enc1.write_all(&part1).unwrap();
        let compressed1 = enc1.finish().unwrap();

        let mut enc2 = GzEncoder::new(Vec::new(), Compression::default());
        enc2.write_all(&part2).unwrap();
        let compressed2 = enc2.finish().unwrap();

        let mut multi = compressed1;
        multi.extend_from_slice(&compressed2);

        let mut output = Vec::new();
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        decompress_gzip_libdeflate(&multi, &mut output, num_threads).unwrap();

        let mut expected = part1;
        expected.extend_from_slice(&part2);
        assert_eq!(output, expected);
    }
}
