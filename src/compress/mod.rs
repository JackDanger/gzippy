//! Compression engine — pure bytes-in / bytes-out dispatch.
//!
//! Entry points for the I/O layer are in `io`. This module contains
//! `compress_with_pipeline` (routing engine) and `compress_bytes` (library API).

pub mod deflate;
pub mod deflate64;
pub mod io;
pub mod optimization;
pub mod parallel;
pub mod pipelined;
// Increment 7: `SimpleOptimizer` is the C-FFI (flate2 / libdeflate / ISA-L)
// parallel-compress dispatcher. It is OFF the production routing graph and kept
// only as a differential oracle behind `ffi-oracle`.
#[cfg(any(test, feature = "ffi-oracle"))]
pub mod simple;
pub mod zopfli;

use std::io::{Read, Write};

use crate::cli::GzippyArgs;
use crate::compress::optimization::OptimizationConfig;
use crate::compress::parallel::GzipHeaderInfo;
use crate::error::GzippyResult;

/// Drive the pure-Rust compression engine to completion (Increment 7 — the sole
/// production compress path; no C-FFI compressor in the routing graph).
///
/// Routing:
///   -F / -I / -J (explicit zopfli tuning) → zopfli (pure `zopfli_pure`)
///   T1  L0–L12  → `deflate::compress_gzip` (pure single-member gzip)
///   T>1 L0–L12  → `PipelinedGzEncoder::compress_buffer_pure` (pure parallel,
///                 standard single-member gzip, byte-identical across T)
pub(crate) fn compress_with_pipeline<R: Read, W: Write + Send>(
    mut reader: R,
    writer: W,
    args: &GzippyArgs,
    opt_config: &OptimizationConfig,
    header_info: &GzipHeaderInfo,
) -> GzippyResult<u64> {
    // Explicit zopfli tuning flags (-F iterations / -I no-split / -J split-max)
    // force the true zopfli encoder (the pure-Rust `zopfli_pure` port — NO
    // C-FFI). Plain `-11` does NOT: it falls through to the pure near-optimal
    // DEFLATE engine below, matching the pre-Inc7 ordering intent.
    let explicit_zopfli =
        args.zopfli_iterations.is_some() || args.zopfli_no_split || args.zopfli_split_max.is_some();
    if explicit_zopfli {
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

    // Increment 7: the pure-Rust DEFLATE encoder is the SOLE production
    // single-thread compress path for EVERY level 0–12. All C-FFI compressors
    // (ISA-L / libdeflate one-shot / flate2-zlib-ng) have been removed from the
    // production routing graph; they remain compilable only behind the dev
    // `ffi-oracle` feature as differential fuzz oracles. `compress_gzip` picks
    // the cheapest of stored / static-Huffman / dynamic-Huffman per block, so it
    // subsumes the old L0 stored passthrough and the `--huffman` / `--rle`
    // fall-throughs while always emitting a valid single-member gzip stream.
    if opt_config.thread_count == 1 {
        if args.verbosity >= 2 {
            eprintln!(
                "gzippy: using pure-Rust DEFLATE encoder (T1 L{})",
                args.compression_level
            );
        }
        let mut input = Vec::new();
        reader.read_to_end(&mut input)?;
        let bytes = input.len() as u64;
        // Pad the read buffer in place with the matchfinder's trailing slack so
        // the compressor parses IN PLACE — no second full-input work buffer.
        let logical_len = input.len();
        input.resize(logical_len + crate::compress::deflate::INPLACE_TAIL_PAD, 0);
        let gz = crate::compress::deflate::compress_gzip_padded(
            &input,
            logical_len,
            args.compression_level as u32,
        );
        let mut writer = writer;
        writer.write_all(&gz)?;
        writer.flush()?;
        return Ok(bytes);
    }

    // Increment 7: multi-threaded pure-Rust parallel DEFLATE encoder — now the
    // SOLE production T>1 compress path (the `pure-rust-encoder` gate is removed;
    // Inc6 landed it alongside SimpleOptimizer, Inc7 makes it unconditional).
    // Produces a single-member STANDARD gzip stream (header + concatenated
    // per-chunk DEFLATE with sync-flush seams + combined CRC/ISIZE) via
    // `PipelinedGzEncoder::compress_buffer_pure` — NOT the "GZ" multiblock
    // format. The former SimpleOptimizer → ParallelGzEncoder / PipelinedGzEncoder
    // C-FFI path (flate2/libdeflate/ISA-L) is retained only behind the dev
    // `ffi-oracle` feature as a differential oracle. `compress_buffer_pure`
    // covers every level 0–12 and the `--huffman` / `--rle` modes (the pure
    // engine picks the cheapest per-block encoding regardless).
    if args.verbosity >= 2 {
        eprintln!(
            "gzippy: using pure-Rust parallel DEFLATE encoder ({} threads)",
            opt_config.thread_count,
        );
    }
    let mut input = Vec::new();
    reader.read_to_end(&mut input)?;
    let bytes = input.len() as u64;
    let mut encoder = crate::compress::pipelined::PipelinedGzEncoder::new(
        args.compression_level as u32,
        opt_config.thread_count,
    );
    encoder.set_header_info(header_info.clone());
    encoder.compress_buffer_pure(&input, writer)?;
    Ok(bytes)
}

// =============================================================================
// Library API
// =============================================================================

/// Compress `data` to raw DEFLATE (RFC 1951) at `level` — no gzip framing.
///
/// Increment 7: routes to the pure-Rust DEFLATE encoder for every level 0–12
/// (the sole production compress path; no C-FFI compressor). `level` is clamped
/// to `0..=12`.
#[allow(dead_code)] // called from lib.rs; unused in the binary
pub fn compress_raw_bytes(data: &[u8], level: u8) -> crate::error::GzippyResult<Vec<u8>> {
    let level = level.clamp(0, 12);
    Ok(crate::compress::deflate::compress_oneshot(
        data,
        level as u32,
    ))
}

/// Compress data using gzippy's full routing table.
///
/// Increment 7: routes through the pure-Rust DEFLATE engine for the given
/// `level` (0–12) and `threads` — T1 single-member, T>1 pure parallel
/// single-member — with explicit zopfli tuning (-F/-I/-J) taking the pure
/// zopfli path. No C-FFI compressor is on this graph.
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
        // `processes` flows into ZopfliTuning::thread_budget for the L11 path;
        // non-zopfli routing uses opt_config.thread_count instead.
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
