//! Compression engine — pure bytes-in / bytes-out dispatch.
//!
//! Entry points for the I/O layer (`compress_io`) are in that module.
//! This module only contains `compress_with_pipeline`, which selects and
//! drives the best available backend for a given (level, threads) pair.

use std::io::{self, Read, Write};

use crate::cli::GzippyArgs;
use crate::error::GzippyResult;
use crate::optimization::OptimizationConfig;
use crate::parallel_compress::GzipHeaderInfo;
use crate::simple_optimizations::SimpleOptimizer;

/// Select the fastest available compression backend and drive it to completion.
///
/// Routing (matches CLAUDE.md compression table):
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
    if opt_config.thread_count == 1 && args.compression_level <= 9 {
        // ISA-L: T1 L0–L3 on x86_64 with AVX2
        if args.compression_level <= 3
            && !args.huffman
            && !args.rle
            && crate::isal_compress::is_available()
        {
            if args.verbosity >= 2 {
                eprintln!("gzippy: using ISA-L single-threaded streaming compression");
            }
            let bytes = crate::isal_compress::compress_gzip_stream_direct(
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
                crate::parallel_compress::compress_single_member(
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
            let adjusted_level = if args.compression_level == 1 { 2 } else { args.compression_level };
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
            let bytes = io::copy(&mut chained, &mut encoder)?;
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
            let adjusted_level = if args.compression_level == 1 { 2 } else { args.compression_level };
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
        let bytes = io::copy(&mut reader, &mut encoder)?;
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
