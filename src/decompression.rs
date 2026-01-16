//! Ultra-fast decompression using libdeflate + zlib-ng
//!
//! Strategy:
//! - Single-member gzip: libdeflate (30-50% faster than zlib)
//! - Multi-member gzip: zlib-ng via flate2 (reliable member boundary handling)
//! - Stdin streaming: flate2 MultiGzDecoder
//!
//! Key insight: Deflate streams can contain bytes that look like gzip headers
//! (0x1f 0x8b 0x08), so we can't reliably detect member boundaries by scanning.
//! Instead, we use flate2's GzDecoder which properly parses each member.

use std::fs::File;
use std::io::{self, stdin, stdout, BufReader, BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::cli::RigzArgs;
use crate::error::{RigzError, RigzResult};
use crate::format::CompressionFormat;
use crate::utils::strip_compression_extension;

/// Output buffer size for streaming
const STREAM_BUFFER_SIZE: usize = 128 * 1024;

pub fn decompress_file(filename: &str, args: &RigzArgs) -> RigzResult<i32> {
    if filename == "-" {
        return decompress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(RigzError::FileNotFound(filename.to_string()));
    }

    if input_path.is_dir() {
        return Err(RigzError::invalid_argument(format!(
            "{} is a directory",
            filename
        )));
    }

    let output_path = if args.stdout {
        None
    } else {
        Some(get_output_filename(input_path, args))
    };

    if let Some(ref output_path) = output_path {
        if output_path.exists() && !args.force {
            return Err(RigzError::invalid_argument(format!(
                "Output file {} already exists",
                output_path.display()
            )));
        }
    }

    let input_file = File::open(input_path)?;
    let file_size = input_file.metadata()?.len();
    let mmap = unsafe { Mmap::map(&input_file)? };

    let format = detect_compression_format_from_path(input_path)?;

    let result = if args.stdout {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        decompress_mmap_libdeflate(&mmap, &mut writer, format)
    } else {
        let output_path = output_path.clone().unwrap();
        let output_file = File::create(&output_path)?;
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, output_file);
        decompress_mmap_libdeflate(&mmap, &mut writer, format)
    };

    match result {
        Ok(output_size) => {
            if args.verbosity > 0 && !args.quiet {
                print_decompression_stats(file_size, output_size, input_path);
            }
            if !args.keep && !args.stdout {
                std::fs::remove_file(input_path)?;
            }
            Ok(0)
        }
        Err(e) => {
            if !args.stdout {
                let cleanup_path = get_output_filename(input_path, args);
                if cleanup_path.exists() {
                    let _ = std::fs::remove_file(&cleanup_path);
                }
            }
            Err(e)
        }
    }
}

pub fn decompress_stdin(_args: &RigzArgs) -> RigzResult<i32> {
    use flate2::read::MultiGzDecoder;
    
    let stdin = stdin();
    let input = BufReader::with_capacity(STREAM_BUFFER_SIZE, stdin.lock());
    let stdout = stdout();
    let mut output = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
    
    let mut decoder = MultiGzDecoder::new(input);
    io::copy(&mut decoder, &mut output)?;
    output.flush()?;
    
    Ok(0)
}

/// Decompress using libdeflate (fastest for in-memory data)
fn decompress_mmap_libdeflate<W: Write>(
    mmap: &Mmap,
    writer: &mut W,
    format: CompressionFormat,
) -> RigzResult<u64> {
    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            decompress_gzip_libdeflate(&mmap[..], writer)
        }
        CompressionFormat::Zlib => {
            decompress_zlib_libdeflate(&mmap[..], writer)
        }
    }
}

/// Quick check if data contains multiple gzip members
/// Uses SIMD-accelerated search via memchr (10-50x faster than byte-by-byte)
/// Only scans first 256KB to detect parallel-compressed files
#[inline]
fn is_multi_member_quick(data: &[u8]) -> bool {
    use memchr::memmem;
    
    const SCAN_LIMIT: usize = 256 * 1024;
    const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b, 0x08];
    
    let scan_end = data.len().min(SCAN_LIMIT);
    
    // Skip past the first gzip header (minimum 10 bytes)
    // and look for another gzip magic sequence
    if scan_end <= 10 {
        return false;
    }
    
    // memmem uses SIMD (AVX2/NEON) internally for fast searching
    memmem::find(&data[10..scan_end], GZIP_MAGIC).is_some()
}

/// Decompress gzip - chooses optimal strategy based on content
/// 
/// - Single member: libdeflate (fastest, 30-50% faster than zlib)
/// - Multi member: zlib-ng via flate2 (reliable member boundary handling)
fn decompress_gzip_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(0);
    }

    // Fast path: check if this is likely multi-member (from parallel compression)
    // Only scan first 256KB - if no second header found, use direct single-member path
    if !is_multi_member_quick(data) {
        return decompress_single_member_libdeflate(data, writer);
    }

    // Multi-member file: use zlib-ng which correctly handles member boundaries
    // by actually inflating each stream (can't reliably detect boundaries otherwise)
    decompress_multi_member_zlibng(data, writer)
}

/// Decompress single-member gzip using libdeflate (fastest path)
fn decompress_single_member_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use libdeflater::{Decompressor, DecompressionError};
    
    let mut decompressor = Decompressor::new();
    
    // Estimate output size: start with 4x input, grow if needed
    let initial_size = data.len().saturating_mul(4).max(64 * 1024);
    let mut output_buf = vec![0u8; initial_size];
    
    loop {
        match decompressor.gzip_decompress(data, &mut output_buf) {
            Ok(decompressed_size) => {
                writer.write_all(&output_buf[..decompressed_size])?;
                writer.flush()?;
                return Ok(decompressed_size as u64);
            }
            Err(DecompressionError::InsufficientSpace) => {
                // Grow buffer and retry
                let new_size = output_buf.len().saturating_mul(2);
                output_buf.resize(new_size, 0);
                continue;
            }
            Err(_) => {
                return Err(RigzError::invalid_argument("gzip decompression failed".to_string()));
            }
        }
    }
}

/// Decompress multi-member gzip using zlib-ng (via flate2)
/// Uses MultiGzDecoder which is optimized for concatenated gzip streams
fn decompress_multi_member_zlibng<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use flate2::bufread::MultiGzDecoder;
    use std::io::Read;
    
    let mut total_bytes = 0u64;
    let mut decoder = MultiGzDecoder::new(data);
    
    // Use larger buffer for better throughput (matches pigz's internal buffer)
    let mut buf = vec![0u8; 256 * 1024];
    
    loop {
        match decoder.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                writer.write_all(&buf[..n])?;
                total_bytes += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(RigzError::Io(e)),
        }
    }
    
    writer.flush()?;
    Ok(total_bytes)
}

/// Decompress zlib using libdeflate
fn decompress_zlib_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use libdeflater::{Decompressor, DecompressionError};
    
    let mut decompressor = Decompressor::new();
    let mut output_buf = vec![0u8; data.len().saturating_mul(4).max(64 * 1024)];
    
    loop {
        match decompressor.zlib_decompress(data, &mut output_buf) {
            Ok(decompressed_size) => {
                writer.write_all(&output_buf[..decompressed_size])?;
                writer.flush()?;
                return Ok(decompressed_size as u64);
            }
            Err(DecompressionError::InsufficientSpace) => {
                let new_size = output_buf.len().saturating_mul(2);
                output_buf.resize(new_size, 0);
                continue;
            }
            Err(_) => {
                return Err(RigzError::invalid_argument("zlib decompression failed".to_string()));
            }
        }
    }
}

fn detect_compression_format_from_path(path: &Path) -> RigzResult<CompressionFormat> {
    if let Some(format) = crate::utils::detect_format_from_file(path) {
        Ok(format)
    } else {
        Ok(CompressionFormat::Gzip)
    }
}

fn get_output_filename(input_path: &Path, args: &RigzArgs) -> std::path::PathBuf {
    if args.stdout {
        return input_path.to_path_buf();
    }
    let mut output_path = strip_compression_extension(input_path);
    if output_path == input_path {
        output_path = input_path.to_path_buf();
        let current_name = output_path.file_name().unwrap().to_str().unwrap();
        output_path.set_file_name(format!("{}.out", current_name));
    }
    output_path
}

fn print_decompression_stats(input_size: u64, output_size: u64, path: &Path) {
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("<unknown>");
    
    let ratio = if output_size > 0 {
        input_size as f64 / output_size as f64
    } else {
        1.0
    };
    
    let (in_size, in_unit) = format_size(input_size);
    let (out_size, out_unit) = format_size(output_size);
    
    eprintln!(
        "{}: {:.1}{} → {:.1}{} ({:.1}x expansion)",
        filename, in_size, in_unit, out_size, out_unit, 1.0 / ratio
    );
}

fn format_size(bytes: u64) -> (f64, &'static str) {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    
    if bytes >= GB {
        (bytes as f64 / GB as f64, "GB")
    } else if bytes >= MB {
        (bytes as f64 / MB as f64, "MB")
    } else if bytes >= KB {
        (bytes as f64 / KB as f64, "KB")
    } else {
        (bytes as f64, "B")
    }
}
