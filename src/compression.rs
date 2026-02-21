//! File compression module
//!
//! Uses system zlib for identical output to gzip at ALL compression levels.

use std::fs::File;
use std::io::{self, stdin, stdout, BufWriter, Cursor, Read, Write};
use std::path::Path;

use crate::cli::GzippyArgs;
use crate::error::{GzippyError, GzippyResult};
use crate::optimization::{detect_content_type, ContentType, OptimizationConfig};
use crate::parallel_compress::GzipHeaderInfo;
use crate::simple_optimizations::SimpleOptimizer;

pub fn compress_file(filename: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    if filename == "-" {
        return compress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(GzippyError::FileNotFound(filename.to_string()));
    }

    // Handle directory recursion
    if input_path.is_dir() {
        return if args.recursive {
            compress_directory(filename, args)
        } else {
            Err(GzippyError::invalid_argument(format!(
                "{} is a directory",
                filename
            )))
        };
    }

    // Skip symlinks (unless -f, which follows them)
    if input_path.is_symlink() && !args.force {
        if !args.quiet {
            eprintln!(
                "gzippy: {}: is a symbolic link -- skipping (use -f to force)",
                filename
            );
        }
        return Ok(2);
    }

    // Skip special files (devices, FIFOs, sockets)
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        let ft = std::fs::symlink_metadata(input_path)?.file_type();
        if ft.is_block_device() || ft.is_char_device() || ft.is_fifo() || ft.is_socket() {
            if !args.quiet {
                eprintln!("gzippy: {}: is not a regular file -- skipping", filename);
            }
            return Ok(2);
        }
    }

    // Skip files with multiple hard links (unless -f)
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(metadata) = std::fs::metadata(input_path) {
            if metadata.nlink() > 1 && !args.force {
                if !args.quiet {
                    eprintln!(
                        "gzippy: {}: has {} other links -- skipping (use -f to force)",
                        filename,
                        metadata.nlink() - 1
                    );
                }
                return Ok(2); // Exit code 2 = warning
            }
        }
    }

    // Determine output filename
    let output_path = if args.stdout {
        None
    } else {
        Some(get_output_filename(input_path, args))
    };

    // Check if output file exists and handle force/prompt
    if let Some(ref output_path) = output_path {
        if output_path.exists() && !args.force {
            use std::io::IsTerminal;
            if std::io::stdin().is_terminal() {
                eprint!(
                    "gzippy: {} already exists; do you wish to overwrite (y or n)? ",
                    output_path.display()
                );
                let mut response = String::new();
                std::io::stdin().read_line(&mut response)?;
                if !response.trim().eq_ignore_ascii_case("y") {
                    eprintln!("\tnot overwritten");
                    return Ok(2);
                }
            } else {
                return Err(GzippyError::invalid_argument(format!(
                    "Output file {} already exists",
                    output_path.display()
                )));
            }
        }
    }

    // Open input file once and get metadata from the handle (fewer syscalls)
    let input_file = File::open(input_path)?;
    let file_size = input_file.metadata()?.len();

    // Skip content detection for single-threaded mode - no benefit and adds overhead
    // For multi-threaded with L4-L9, detect content type for optimization decisions
    let content_type = if args.processes <= 1 || args.compression_level <= 3 {
        // Fast path: skip content detection
        ContentType::Binary
    } else {
        // Multi-threaded with higher compression levels: detect for optimization
        let mut sample_file = File::open(input_path)?;
        detect_content_type(&mut sample_file).unwrap_or(ContentType::Binary)
    };

    // Create optimization configuration
    // --independent forces L6 behavior (independent blocks) even at L7-L9
    let effective_level =
        if args.independent && args.compression_level >= 7 && args.compression_level <= 9 {
            6
        } else {
            args.compression_level
        };
    let opt_config =
        OptimizationConfig::new(args.processes, file_size, effective_level, content_type);

    if args.verbosity >= 2 {
        eprintln!(
            "gzippy: optimizing for {:?} content, {} threads, {}KB buffer, {:?} backend",
            content_type,
            opt_config.thread_count,
            opt_config.buffer_size / 1024,
            opt_config.backend
        );
    }

    // Build gzip header metadata from file
    let header_info = build_header_info(input_path, args);

    // Use mmap for multi-threaded compression
    // On Linux, mmap is faster even for small files due to zero-copy and kernel page cache
    // Threshold: 128KB (one block) - below this, overhead exceeds benefit
    let use_mmap = opt_config.thread_count > 1 && file_size > 128 * 1024;

    // Register output file for signal handler cleanup
    if let Some(ref output_path) = output_path {
        crate::set_output_file(Some(output_path.to_string_lossy().to_string()));
    }

    let result = if args.rsyncable && use_mmap {
        // RSYNCABLE PATH: Content-determined block boundaries for rsync-friendly output
        if args.verbosity >= 2 {
            eprintln!("gzippy: using rsyncable compression");
        }
        let mmap = unsafe { memmap2::Mmap::map(&File::open(input_path)?)? };
        if args.stdout {
            crate::parallel_compress::compress_rsyncable(
                &mmap,
                args.compression_level as u32,
                opt_config.thread_count,
                &header_info,
                stdout(),
            )
            .map_err(|e| e.into())
        } else {
            let output_path = output_path.clone().unwrap();
            let output_file = BufWriter::new(File::create(&output_path)?);
            crate::parallel_compress::compress_rsyncable(
                &mmap,
                args.compression_level as u32,
                opt_config.thread_count,
                &header_info,
                output_file,
            )
            .map_err(|e| e.into())
        }
    } else if use_mmap {
        // MMAP PATH: Zero-copy parallel compression for large files
        if args.verbosity >= 2 {
            eprintln!(
                "gzippy: using mmap parallel backend with {} threads",
                opt_config.thread_count,
            );
        }
        let optimizer =
            SimpleOptimizer::new(opt_config.clone()).with_header_info(header_info.clone());
        if args.stdout {
            let out = BufWriter::with_capacity(1024 * 1024, stdout());
            optimizer
                .compress_file(input_path, out)
                .map_err(|e| e.into())
        } else {
            let output_path = output_path.clone().unwrap();
            let output_file = BufWriter::new(File::create(&output_path)?);
            optimizer
                .compress_file(input_path, output_file)
                .map_err(|e| e.into())
        }
    } else if args.stdout {
        let out = BufWriter::with_capacity(1024 * 1024, stdout());
        compress_with_pipeline(input_file, out, args, &opt_config, &header_info)
    } else {
        let output_path = output_path.clone().unwrap();
        let output_file = BufWriter::new(File::create(&output_path)?);
        compress_with_pipeline(input_file, output_file, args, &opt_config, &header_info)
    };

    // Clear signal handler's output file reference
    crate::set_output_file(None);

    match result {
        Ok(_) => {
            // Preserve file permissions and timestamps on output file
            if !args.stdout {
                let output_path = get_output_filename(input_path, args);
                preserve_metadata(input_path, &output_path);

                // Synchronous: fsync after write
                if args.synchronous {
                    if let Ok(f) = File::open(&output_path) {
                        let _ = f.sync_all();
                    }
                }
            }

            // Print stats if verbose (get actual compressed size from output file)
            if args.verbosity > 0 && !args.quiet && !args.stdout {
                let output_path = get_output_filename(input_path, args);
                if let Ok(metadata) = std::fs::metadata(&output_path) {
                    print_compression_stats(file_size, metadata.len(), input_path, args);
                }
            }

            // Delete original file if not keeping it
            if !args.keep && !args.stdout {
                std::fs::remove_file(input_path)?;
            }

            Ok(0)
        }
        Err(e) => {
            // Clean up output file on error if we created one
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

pub fn compress_stdin(args: &GzippyArgs) -> GzippyResult<i32> {
    let can_parallelize = args.processes > 1;

    // For multi-threaded: try to mmap stdin (zero-copy, all threads share).
    // For single-threaded: read_to_end is faster (sequential read-ahead avoids page faults).
    #[cfg(unix)]
    let mmap_data: Option<memmap2::Mmap> = if can_parallelize {
        use std::os::unix::io::FromRawFd;
        let stdin_fd = 0;
        let meta =
            std::fs::File::from(unsafe { std::os::unix::io::OwnedFd::from_raw_fd(stdin_fd) });
        let is_regular = meta
            .metadata()
            .map(|m| m.file_type().is_file())
            .unwrap_or(false);
        let result = if is_regular {
            unsafe { memmap2::Mmap::map(&meta) }.ok()
        } else {
            None
        };
        std::mem::forget(meta);
        result
    } else {
        None
    };
    #[cfg(not(unix))]
    let mmap_data: Option<memmap2::Mmap> = None;

    let mut buffer_vec = Vec::new();
    let input_data: &[u8] = if let Some(ref mmap) = mmap_data {
        &mmap[..]
    } else {
        let mut input = stdin();
        let mut sample = vec![0u8; 8192];
        let bytes_read = input.read(&mut sample)?;
        if bytes_read > 0 {
            sample.truncate(bytes_read);
            buffer_vec.extend_from_slice(&sample);
            input.read_to_end(&mut buffer_vec)?;
        }
        &buffer_vec
    };

    let file_size = input_data.len() as u64;
    let content_type = if input_data.len() >= 8192 {
        crate::optimization::analyze_content_type(&input_data[..8192])
    } else if !input_data.is_empty() {
        crate::optimization::analyze_content_type(input_data)
    } else {
        ContentType::Binary
    };

    let opt_config = OptimizationConfig::new(
        args.processes,
        file_size,
        args.compression_level,
        content_type,
    );

    let header_info = GzipHeaderInfo::default();
    let compression_level = args.compression_level as u32;
    let output = BufWriter::with_capacity(1024 * 1024, stdout());

    if opt_config.thread_count > 1 {
        if args.compression_level >= 6 && args.compression_level <= 9 {
            let mut encoder = crate::pipelined_compress::PipelinedGzEncoder::new(
                compression_level,
                opt_config.thread_count,
            );
            encoder.set_header_info(header_info);
            encoder.compress_buffer(input_data, output)?;
        } else {
            let mut encoder = crate::parallel_compress::ParallelGzEncoder::new(
                compression_level,
                opt_config.thread_count,
            );
            encoder.set_header_info(header_info);
            encoder.compress_buffer(input_data, output)?;
        }
        return Ok(0);
    }

    // T1: go through compress_with_pipeline which has ratio probe optimization.
    // Drop references so we can move buffer_vec into the Cursor without copying.
    drop(mmap_data);
    let cursor = Cursor::new(buffer_vec);
    let result = compress_with_pipeline(cursor, output, args, &opt_config, &header_info);

    match result {
        Ok(_) => Ok(0),
        Err(e) => Err(e),
    }
}

fn compress_directory(dirname: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    use walkdir::WalkDir;

    let mut exit_code = 0;

    for entry in WalkDir::new(dirname) {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let path_str = path.to_string_lossy();
            match compress_file(&path_str, args) {
                Ok(code) => {
                    if code != 0 {
                        exit_code = code;
                    }
                }
                Err(e) => {
                    eprintln!("gzippy: {}: {}", path_str, e);
                    exit_code = 1;
                }
            }
        }
    }

    Ok(exit_code)
}

fn compress_with_pipeline<R: Read, W: Write + Send>(
    mut reader: R,
    writer: W,
    args: &GzippyArgs,
    opt_config: &OptimizationConfig,
    header_info: &GzipHeaderInfo,
) -> GzippyResult<u64> {
    // FAST PATH: Single-threaded goes directly to the fastest available backend
    // Exception: L10-L12 use libdeflate even single-threaded for ultra compression
    if opt_config.thread_count == 1 && args.compression_level <= 9 {
        // For L0-L3, use ISA-L when available (fastest on x86 with AVX2)
        if args.compression_level <= 3
            && !args.huffman
            && !args.rle
            && crate::isal_compress::is_available()
        {
            if args.verbosity >= 2 {
                eprintln!("gzippy: using ISA-L single-threaded compression");
            }
            let bytes = crate::isal_compress::compress_gzip_stream(
                &mut reader,
                writer,
                args.compression_level as u32,
            )?;
            return Ok(bytes);
        }

        // L1-L5 (no ISA-L): Use libdeflate for ~2x faster compression than zlib-ng,
        // unless the data is highly compressible (ratio < 10%) where zlib-ng's
        // streaming mode is faster (avoids large buffer allocation).
        if args.compression_level >= 1 && args.compression_level <= 5 && !args.huffman && !args.rle
        {
            // Read first 64KB to probe compression ratio without buffering all data
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
                true // Small files: libdeflate is fine
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

            // Highly compressible: stream through flate2/zlib-ng (no full buffering)
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
            let bytes = io::copy(&mut chained, &mut encoder)?;
            encoder.finish()?;
            return Ok(bytes);
        }

        // L6-L9: Use flate2/zlib-ng streaming (better ratio at higher levels)
        use flate2::Compression;

        if args.verbosity >= 2 {
            eprintln!("gzippy: using direct flate2 single-threaded path");
        }

        // Handle compression strategy flags
        let compression = if args.huffman || args.rle {
            Compression::new(1) // Huffman-only / RLE approximation (fastest)
        } else {
            // zlib-ng level 1 uses a different strategy that produces 2-5x larger output
            // on repetitive data. Map level 1 → 2 for better compression ratio.
            let adjusted_level = if args.compression_level == 1 {
                2
            } else {
                args.compression_level
            };
            Compression::new(adjusted_level as u32)
        };

        // Use GzBuilder for FNAME/MTIME/FCOMMENT
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

    // MULTI-THREADED PATH: Use optimizer for parallel compression
    let optimizer = SimpleOptimizer::new(opt_config.clone()).with_header_info(header_info.clone());

    if args.verbosity >= 2 {
        eprintln!(
            "gzippy: using parallel backend with {} threads",
            opt_config.thread_count,
        );
    }

    optimizer.compress(reader, writer).map_err(|e| e.into())
}

fn get_output_filename(input_path: &Path, args: &GzippyArgs) -> std::path::PathBuf {
    let mut output_path = input_path.to_path_buf();

    // Remove existing compression extensions if forcing
    if args.force {
        let mut stem = input_path
            .file_stem()
            .unwrap_or(input_path.as_os_str())
            .to_str()
            .unwrap();
        if stem.ends_with(".tar") {
            stem = &stem[..stem.len() - 4];
        }
        output_path.set_file_name(stem);
    }

    // Add appropriate extension
    let current_extension = output_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("");
    let new_extension = if current_extension.is_empty() {
        args.suffix.trim_start_matches('.')
    } else {
        &format!("{}{}", current_extension, args.suffix)
    };

    output_path.set_extension(new_extension);
    output_path
}

fn print_compression_stats(input_size: u64, output_size: u64, path: &Path, args: &GzippyArgs) {
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("<unknown>");

    let ratio = if input_size > 0 {
        output_size as f64 / input_size as f64
    } else {
        1.0
    };
    let saved_pct = (1.0 - ratio) * 100.0;

    // Format sizes nicely
    let (in_size, in_unit) = format_size(input_size);
    let (out_size, out_unit) = format_size(output_size);

    if args.processes > 1 {
        eprintln!(
            "{}: {:.1}{} → {:.1}{} ({:.1}% saved, {} threads)",
            filename, in_size, in_unit, out_size, out_unit, saved_pct, args.processes
        );
    } else {
        eprintln!(
            "{}: {:.1}{} → {:.1}{} ({:.1}% saved)",
            filename, in_size, in_unit, out_size, out_unit, saved_pct
        );
    }
}

/// Copy file permissions and timestamps from source to destination.
/// Errors are silently ignored (best-effort, matching gzip behavior).
fn preserve_metadata(src: &Path, dst: &Path) {
    if let Ok(metadata) = std::fs::metadata(src) {
        // Copy permissions (mode bits on Unix)
        let _ = std::fs::set_permissions(dst, metadata.permissions());

        // Copy modification time
        if let Ok(mtime) = metadata.modified() {
            let _ = filetime::set_file_mtime(dst, filetime::FileTime::from_system_time(mtime));
        }
    }
}

/// Build gzip header metadata from a file path and CLI args
fn build_header_info(path: &Path, args: &GzippyArgs) -> GzipHeaderInfo {
    let filename = if !args.no_name {
        path.file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    let mtime = if !args.no_time {
        std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0)
    } else {
        0
    };

    GzipHeaderInfo {
        filename,
        mtime,
        comment: args.comment.clone(),
    }
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
