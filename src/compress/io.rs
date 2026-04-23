//! File, stdin, and directory I/O for compression.
//!
//! Entry points: `compress_file` and `compress_stdin`.
//! All compression logic is in `compression.rs`; this module only handles
//! filesystem concerns: file reading, output path selection, metadata
//! preservation, stats printing, and signal-handler registration.

use std::fs::File;
use std::io::{self, stdin, stdout, BufWriter, Cursor, Read, Write};
use std::path::Path;

struct CountingWriter<W: Write> {
    inner: W,
    count: u64,
}
impl<W: Write> CountingWriter<W> {
    fn new(inner: W) -> Self {
        Self { inner, count: 0 }
    }
}
impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

use crate::cli::GzippyArgs;
use crate::compress::optimization::{detect_content_type, ContentType, OptimizationConfig};
use crate::compress::parallel::GzipHeaderInfo;
use crate::compress::simple::SimpleOptimizer;
use crate::error::{GzippyError, GzippyResult};
use crate::utils::{debug_enabled, preserve_metadata};

pub fn compress_file(filename: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    if filename == "-" {
        return compress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(GzippyError::FileNotFound(filename.to_string()));
    }
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
    if input_path.is_symlink() && !args.force {
        if !args.quiet {
            eprintln!(
                "gzippy: {}: is a symbolic link -- skipping (use -f to force)",
                filename
            );
        }
        return Ok(2);
    }
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
                return Ok(2);
            }
        }
    }

    let output_path = if args.stdout {
        None
    } else {
        Some(get_output_filename(input_path, args))
    };

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

    let input_file = File::open(input_path)?;
    let file_size = input_file.metadata()?.len();

    let content_type = if args.processes <= 1 || args.compression_level <= 3 {
        ContentType::Binary
    } else {
        let mut sample_file = File::open(input_path)?;
        detect_content_type(&mut sample_file).unwrap_or(ContentType::Binary)
    };

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

    let header_info = build_header_info(input_path, args);
    let use_mmap = opt_config.thread_count > 1 && file_size > 128 * 1024;

    if let Some(ref output_path) = output_path {
        crate::set_output_file(Some(output_path.to_string_lossy().to_string()));
    }

    let result = if args.rsyncable && use_mmap {
        if args.verbosity >= 2 {
            eprintln!("gzippy: using rsyncable compression");
        }
        let mmap = unsafe { memmap2::Mmap::map(&File::open(input_path)?)? };
        if args.stdout {
            crate::compress::parallel::compress_rsyncable(
                &mmap,
                args.compression_level as u32,
                opt_config.thread_count,
                &header_info,
                stdout(),
            )
            .map_err(|e| e.into())
        } else {
            let output_file = BufWriter::new(File::create(output_path.as_ref().unwrap())?);
            crate::compress::parallel::compress_rsyncable(
                &mmap,
                args.compression_level as u32,
                opt_config.thread_count,
                &header_info,
                output_file,
            )
            .map_err(|e| e.into())
        }
    } else if use_mmap {
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
            let output_file = BufWriter::new(File::create(output_path.as_ref().unwrap())?);
            optimizer
                .compress_file(input_path, output_file)
                .map_err(|e| e.into())
        }
    } else if args.stdout {
        let out = BufWriter::with_capacity(1024 * 1024, stdout());
        crate::compress::compress_with_pipeline(input_file, out, args, &opt_config, &header_info)
    } else {
        let output_file = BufWriter::new(File::create(output_path.as_ref().unwrap())?);
        crate::compress::compress_with_pipeline(
            input_file,
            output_file,
            args,
            &opt_config,
            &header_info,
        )
    };

    crate::set_output_file(None);

    match result {
        Ok(_) => {
            if !args.stdout {
                let output_path = get_output_filename(input_path, args);
                preserve_metadata(input_path, &output_path);
                if args.synchronous {
                    if let Ok(f) = File::open(&output_path) {
                        let _ = f.sync_all();
                    }
                }
            }
            if args.verbosity > 0 && !args.quiet && !args.stdout {
                let output_path = get_output_filename(input_path, args);
                if let Ok(metadata) = std::fs::metadata(&output_path) {
                    print_stats(file_size, metadata.len(), input_path, args);
                }
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

pub fn compress_stdin(args: &GzippyArgs) -> GzippyResult<i32> {
    let can_parallelize = args.processes > 1;
    let verbose = args.verbose && !args.quiet;

    // T1 L0-L3 streaming fast path: compress directly from stdin with ~2MB memory.
    // Must happen BEFORE read_to_end to avoid buffering the entire input.
    if !can_parallelize
        && args.compression_level <= 3
        && !args.huffman
        && !args.rle
        && crate::backends::isal_compress::is_available()
    {
        let mut input = stdin();
        let mut counted = CountingWriter::new(BufWriter::with_capacity(1024 * 1024, stdout()));
        let compression_level = args.compression_level as u32;
        let in_bytes = if debug_enabled() {
            let t0 = std::time::Instant::now();
            let bytes = crate::backends::isal_compress::compress_gzip_stream_direct(
                &mut input,
                &mut counted,
                compression_level,
            )?;
            let elapsed = t0.elapsed();
            eprintln!(
                "[gzippy] compress T1 ISA-L L{} streaming: {:.1}ms, {:.1} MB/s ({} bytes in)",
                compression_level,
                elapsed.as_secs_f64() * 1000.0,
                bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0,
                bytes
            );
            bytes
        } else {
            crate::backends::isal_compress::compress_gzip_stream_direct(
                &mut input,
                &mut counted,
                compression_level,
            )?
        };
        counted.flush()?;
        if verbose {
            print_stdin_stats(in_bytes, counted.count, args);
        }
        return Ok(0);
    }

    #[cfg(unix)]
    let mmap_data: Option<memmap2::Mmap> = if can_parallelize {
        use std::os::unix::io::FromRawFd;
        let meta = std::fs::File::from(unsafe {
            std::os::unix::io::OwnedFd::from_raw_fd(0 /* stdin */)
        });
        let is_regular = meta
            .metadata()
            .map(|m| m.file_type().is_file())
            .unwrap_or(false);
        let result = if is_regular {
            let m = unsafe { memmap2::Mmap::map(&meta) }.ok();
            if let Some(ref mmap) = m {
                let _ = mmap.advise(memmap2::Advice::Sequential);
            }
            m
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

    let in_bytes = input_data.len() as u64;
    let content_type = if input_data.len() >= 8192 {
        crate::compress::optimization::analyze_content_type(&input_data[..8192])
    } else if !input_data.is_empty() {
        crate::compress::optimization::analyze_content_type(input_data)
    } else {
        ContentType::Binary
    };

    let opt_config = OptimizationConfig::new(
        args.processes,
        in_bytes,
        args.compression_level,
        content_type,
    );

    let header_info = GzipHeaderInfo::default();
    let compression_level = args.compression_level as u32;
    let mut counted = CountingWriter::new(BufWriter::with_capacity(1024 * 1024, stdout()));

    if opt_config.thread_count > 1 {
        if args.compression_level >= 6 && args.compression_level <= 9 {
            let mut encoder = crate::compress::pipelined::PipelinedGzEncoder::new(
                compression_level,
                opt_config.thread_count,
            );
            encoder.set_header_info(header_info);
            encoder.compress_buffer(input_data, &mut counted)?;
        } else {
            let mut encoder = crate::compress::parallel::ParallelGzEncoder::new(
                compression_level,
                opt_config.thread_count,
            );
            encoder.set_header_info(header_info);
            encoder.compress_buffer(input_data, &mut counted)?;
        }
        counted.flush()?;
        if verbose {
            print_stdin_stats(in_bytes, counted.count, args);
        }
        return Ok(0);
    }

    if args.compression_level <= 3
        && !args.huffman
        && !args.rle
        && crate::backends::isal_compress::is_available()
    {
        if args.verbosity >= 2 {
            eprintln!("gzippy: using ISA-L single-threaded compression (direct)");
        }
        if debug_enabled() {
            let t0 = std::time::Instant::now();
            crate::backends::isal_compress::compress_gzip_to_writer(
                input_data,
                &mut counted,
                compression_level,
            )?;
            let elapsed = t0.elapsed();
            eprintln!(
                "[gzippy] compress T1 ISA-L L{}: {:.1}ms, {:.1} MB/s ({} bytes in)",
                compression_level,
                elapsed.as_secs_f64() * 1000.0,
                input_data.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0,
                input_data.len()
            );
        } else {
            crate::backends::isal_compress::compress_gzip_to_writer(
                input_data,
                &mut counted,
                compression_level,
            )?;
        }
        counted.flush()?;
        if verbose {
            print_stdin_stats(in_bytes, counted.count, args);
        }
        return Ok(0);
    }

    drop(mmap_data);
    let cursor = Cursor::new(buffer_vec);
    match crate::compress::compress_with_pipeline(
        cursor,
        &mut counted,
        args,
        &opt_config,
        &header_info,
    ) {
        Ok(_) => {
            counted.flush()?;
            if verbose {
                print_stdin_stats(in_bytes, counted.count, args);
            }
            Ok(0)
        }
        Err(e) => Err(e),
    }
}

fn print_stdin_stats(in_bytes: u64, out_bytes: u64, args: &GzippyArgs) {
    let ratio = if in_bytes > 0 {
        out_bytes as f64 / in_bytes as f64
    } else {
        1.0
    };
    let saved_pct = (1.0 - ratio) * 100.0;
    let (in_size, in_unit) = human_size(in_bytes);
    let (out_size, out_unit) = human_size(out_bytes);
    if args.processes > 1 {
        eprintln!(
            "(stdin): {:.1}{} → {:.1}{} ({:.1}% saved, {} threads)",
            in_size, in_unit, out_size, out_unit, saved_pct, args.processes
        );
    } else {
        eprintln!(
            "(stdin): {:.1}{} → {:.1}{} ({:.1}% saved)",
            in_size, in_unit, out_size, out_unit, saved_pct
        );
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

fn get_output_filename(input_path: &Path, args: &GzippyArgs) -> std::path::PathBuf {
    let mut output_path = input_path.to_path_buf();
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

fn print_stats(input_size: u64, output_size: u64, path: &Path, args: &GzippyArgs) {
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
    let (in_size, in_unit) = human_size(input_size);
    let (out_size, out_unit) = human_size(output_size);
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

pub(crate) fn build_header_info(path: &Path, args: &GzippyArgs) -> GzipHeaderInfo {
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

fn human_size(bytes: u64) -> (f64, &'static str) {
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
