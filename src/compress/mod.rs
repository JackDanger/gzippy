//! Compression engine — pure bytes-in / bytes-out dispatch.
//!
//! Entry points for the I/O layer are in `io`. This module contains
//! `compress_with_pipeline` (routing engine) and `compress_bytes` (library API).

pub mod io;
pub mod optimization;
pub mod parallel;
pub mod pipelined;
pub mod simple;
pub mod zopfli;

use std::io::{Read, Write};

use crate::cli::GzippyArgs;
use crate::compress::optimization::OptimizationConfig;
use crate::compress::parallel::GzipHeaderInfo;
use crate::compress::simple::SimpleOptimizer;
use crate::error::GzippyResult;

/// Select the fastest available compression backend and drive it to completion.
///
/// Routing (matches CLAUDE.md compression table):
///   L11 / -F / -I / -J       → Zopfli (true zopfli compression)
///   T1 L0–L3 ISA-L available → ISA-L streaming
///   T1 L1–L5                 → libdeflate one-shot (ratio probe) or flate2 streaming
///   T1 L6–L9                 → flate2/zlib-ng streaming
///   T>1                      → SimpleOptimizer (parallel_compress / pipelined_compress)
pub(crate) fn compress_with_pipeline<R: Read, W: Write + Send>(
    mut reader: R,
    writer: W,
    args: &GzippyArgs,
    opt_config: &OptimizationConfig,
    header_info: &GzipHeaderInfo,
) -> GzippyResult<u64> {
    // Zopfli path: L11 or any zopfli tuning flag triggers true zopfli
    if args.use_zopfli() {
        if args.verbosity >= 2 {
            eprintln!(
                "gzippy: using zopfli compression ({} iterations)",
                args.zopfli_iterations.unwrap_or(15)
            );
        }
        let tuning = crate::backends::zopfli_compress::ZopfliTuning::from_args(args);
        // thread_count + block_size are intentionally not passed: the
        // zopfli path is single-member by ratio mandate (plan.md
        // Phase 11.1.A). Intra-block parallelism inside `deflate_part`
        // still uses the machine.
        let mut encoder = crate::compress::zopfli::ZopfliGzEncoder::new(tuning);
        encoder.set_header_info(header_info.clone());
        return encoder.compress(reader, writer).map_err(|e| e.into());
    }

    if opt_config.thread_count == 1 && args.compression_level <= 9 {
        // ISA-L: T1 L0–L3 on x86_64 with AVX2
        if args.compression_level <= 3
            && !args.huffman
            && !args.rle
            && crate::backends::isal_compress::is_available()
        {
            if args.verbosity >= 2 {
                eprintln!("gzippy: using ISA-L single-threaded streaming compression");
            }
            let bytes = crate::backends::isal_compress::compress_gzip_stream_direct(
                &mut reader,
                writer,
                args.compression_level as u32,
            )?;
            return Ok(bytes);
        }

        // libdeflate one-shot: T1 L1–L5 (ratio probe decides vs flate2)
        if args.compression_level >= 1 && args.compression_level <= 5 && !args.huffman && !args.rle
        {
            let mut probe_buf = Vec::with_capacity(65536);
            reader.by_ref().take(65536).read_to_end(&mut probe_buf)?;

            let use_libdeflate = if probe_buf.len() >= 65536 {
                let lvl = libdeflater::CompressionLvl::new(args.compression_level as i32)
                    .unwrap_or_default();
                let mut comp = libdeflater::Compressor::new(lvl);
                let bound = comp.deflate_compress_bound(probe_buf.len());
                let mut out = vec![0u8; bound];
                let actual = comp
                    .deflate_compress(&probe_buf, &mut out)
                    .unwrap_or(probe_buf.len());
                (actual as f64 / probe_buf.len() as f64) >= 0.10
            } else {
                true
            };

            if use_libdeflate {
                if args.verbosity >= 2 {
                    eprintln!("gzippy: using libdeflate single-threaded path");
                }
                let mut input_data = probe_buf;
                reader.read_to_end(&mut input_data)?;
                let bytes = input_data.len() as u64;
                let mut writer = writer;
                crate::compress::parallel::compress_single_member(
                    &mut writer,
                    &input_data,
                    args.compression_level as u32,
                    header_info,
                )?;
                return Ok(bytes);
            }

            // Highly compressible data: stream through flate2/zlib-ng
            if args.verbosity >= 2 {
                eprintln!("gzippy: using flate2 single-threaded path (highly compressible)");
            }
            let adjusted_level = if args.compression_level == 1 {
                2
            } else {
                args.compression_level
            };
            let compression = flate2::Compression::new(adjusted_level as u32);
            let mut builder = flate2::GzBuilder::new();
            if let Some(ref name) = header_info.filename {
                builder = builder.filename(name.as_bytes());
            }
            builder = builder.mtime(header_info.mtime);
            if let Some(ref comment) = header_info.comment {
                builder = builder.comment(comment.as_bytes());
            }
            let mut chained = std::io::Cursor::new(probe_buf).chain(reader);
            let mut encoder = builder.write(writer, compression);
            let bytes = std::io::copy(&mut chained, &mut encoder)?;
            encoder.finish()?;
            return Ok(bytes);
        }

        // T1 L6–L9: flate2/zlib-ng streaming
        if args.verbosity >= 2 {
            eprintln!("gzippy: using direct flate2 single-threaded path");
        }
        let compression = if args.huffman || args.rle {
            flate2::Compression::new(1)
        } else {
            let adjusted_level = if args.compression_level == 1 {
                2
            } else {
                args.compression_level
            };
            flate2::Compression::new(adjusted_level as u32)
        };
        let mut builder = flate2::GzBuilder::new();
        if let Some(ref name) = header_info.filename {
            builder = builder.filename(name.as_bytes());
        }
        builder = builder.mtime(header_info.mtime);
        if let Some(ref comment) = header_info.comment {
            builder = builder.comment(comment.as_bytes());
        }
        let mut encoder = builder.write(writer, compression);
        let bytes = std::io::copy(&mut reader, &mut encoder)?;
        encoder.finish()?;
        return Ok(bytes);
    }

    // Multi-threaded: SimpleOptimizer dispatches to ParallelGzEncoder or PipelinedGzEncoder
    if args.verbosity >= 2 {
        eprintln!(
            "gzippy: using parallel backend with {} threads",
            opt_config.thread_count,
        );
    }
    let optimizer = SimpleOptimizer::new(opt_config.clone()).with_header_info(header_info.clone());
    optimizer.compress(reader, writer).map_err(|e| e.into())
}

// =============================================================================
// Library API
// =============================================================================

/// Compress data using gzippy's full routing table.
///
/// Selects the fastest backend for the given `level` (0–12) and `threads`:
/// - T1 L0–3 + ISA-L  → ISA-L SIMD streaming
/// - T1 L1–5          → libdeflate one-shot (ratio probe) or flate2/zlib-ng
/// - T1 L6–9          → flate2/zlib-ng streaming
/// - T>1 L0–5         → `ParallelGzEncoder` (gzippy "GZ" multi-block format)
/// - T>1 L6–9         → `PipelinedGzEncoder` (single-member gzip-compatible)
/// - L11              → Zopfli
#[allow(dead_code)] // called from lib.rs; unused in the binary
pub fn compress_bytes<R: Read, W: Write + Send>(
    reader: R,
    writer: W,
    level: u8,
    threads: usize,
) -> GzippyResult<u64> {
    use crate::compress::optimization::{ContentType, OptimizationConfig};

    let level = level.clamp(0, 12);
    let threads = threads.max(1);

    let args = GzippyArgs {
        compression_level: level,
        processes: threads,
        ..GzippyArgs::default()
    };

    // Large sentinel so OptimizationConfig::new treats this as a "big file" and
    // enables parallel paths regardless of actual input length (unknown at call time).
    const LARGE_FILE_SENTINEL: u64 = u64::MAX;
    let opt_config =
        OptimizationConfig::new(threads, LARGE_FILE_SENTINEL, level, ContentType::Binary);
    let header_info = GzipHeaderInfo::default();

    compress_with_pipeline(reader, writer, &args, &opt_config, &header_info)
}
