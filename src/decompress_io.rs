//! File, stdin, and directory I/O for decompression.
//!
//! Entry points: `decompress_file` and `decompress_stdin`.
//! All decompression logic is in `decompression.rs`; this module only handles
//! filesystem concerns: mmap, output path selection, metadata preservation,
//! stats printing, and signal-handler registration.

use std::fs::File;
use std::io::{stdin, stdout, BufReader, BufWriter, Read, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::cli::GzippyArgs;
use crate::error::{GzippyError, GzippyResult};
use crate::format::CompressionFormat;
use crate::gzip_format::{
    extract_gzip_fname, extract_gzip_mtime, has_bgzf_markers, is_likely_multi_member,
};
use crate::utils::{debug_enabled, preserve_metadata, strip_compression_extension};

const STREAM_BUFFER_SIZE: usize = 1024 * 1024;

pub fn decompress_file(filename: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    if filename == "-" {
        return decompress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(GzippyError::FileNotFound(filename.to_string()));
    }
    if input_path.is_dir() {
        return if args.recursive {
            decompress_directory(filename, args)
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

    let input_file = File::open(input_path)?;
    let file_size = input_file.metadata()?.len();
    let mmap = unsafe { Mmap::map(&input_file)? };
    let _ = mmap.advise(memmap2::Advice::Sequential);

    let is_compressed =
        mmap.len() >= 2 && ((mmap[0] == 0x1f && mmap[1] == 0x8b) || mmap[0] == 0x78);

    if args.force && args.stdout && !is_compressed {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        writer.write_all(&mmap)?;
        writer.flush()?;
        return Ok(0);
    }
    if !is_compressed {
        return Err(GzippyError::invalid_argument(format!(
            "{}: not in gzip format",
            filename
        )));
    }

    let output_path = if args.stdout {
        None
    } else {
        Some(get_output_filename(input_path, args, &mmap))
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

    let format = detect_format(input_path);

    if let Some(ref output_path) = output_path {
        crate::set_output_file(Some(output_path.to_string_lossy().to_string()));
    }

    let result = if args.stdout {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        let r = decompress_to_writer(&mmap, &mut writer, format, args);
        writer.flush()?;
        r
    } else {
        let output_path = output_path.clone().unwrap();
        let output_file = File::create(&output_path)?;
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, output_file);
        let r = decompress_to_writer(&mmap, &mut writer, format, args);
        writer.flush()?;
        r
    };

    crate::set_output_file(None);

    match result {
        Ok(output_size) => {
            if !args.stdout {
                if let Some(ref output_path) = output_path {
                    preserve_metadata(input_path, output_path);
                    if args.name {
                        if let Some(mtime) = extract_gzip_mtime(&mmap) {
                            if mtime != 0 {
                                let _ = filetime::set_file_mtime(
                                    output_path,
                                    filetime::FileTime::from_unix_time(mtime as i64, 0),
                                );
                            }
                        }
                    }
                    if args.synchronous {
                        if let Ok(f) = File::open(output_path) {
                            let _ = f.sync_all();
                        }
                    }
                }
            }
            if args.verbosity > 0 && !args.quiet {
                print_stats(file_size, output_size, input_path);
            }
            if !args.keep && !args.stdout {
                std::fs::remove_file(input_path)?;
            }
            Ok(0)
        }
        Err(e) => {
            if !args.stdout {
                let cleanup_path = get_output_filename(input_path, args, &mmap);
                if cleanup_path.exists() {
                    let _ = std::fs::remove_file(&cleanup_path);
                }
            }
            Err(e)
        }
    }
}

pub fn decompress_stdin(args: &GzippyArgs) -> GzippyResult<i32> {
    #[cfg(unix)]
    let mmap_data: Option<Mmap> = {
        use std::os::unix::io::FromRawFd;
        let meta = std::fs::File::from(unsafe {
            std::os::unix::io::OwnedFd::from_raw_fd(0 /* stdin */)
        });
        let is_regular = meta.metadata().map(|m| m.file_type().is_file()).unwrap_or(false);
        let result = if is_regular {
            let m = unsafe { Mmap::map(&meta) }.ok();
            if let Some(ref mmap) = m {
                let _ = mmap.advise(memmap2::Advice::Sequential);
            }
            m
        } else {
            None
        };
        std::mem::forget(meta);
        result
    };
    #[cfg(not(unix))]
    let mmap_data: Option<Mmap> = None;

    let input_data_vec;
    let input_data: &[u8] = if let Some(ref mmap) = mmap_data {
        if debug_enabled() {
            eprintln!("[gzippy] stdin mmap'd: {} bytes", mmap.len());
        }
        &mmap[..]
    } else {
        let stdin_handle = stdin();
        let mut data = Vec::new();
        {
            let mut reader = BufReader::with_capacity(STREAM_BUFFER_SIZE, stdin_handle.lock());
            reader.read_to_end(&mut data)?;
        }
        input_data_vec = data;
        &input_data_vec
    };

    if input_data.is_empty() {
        return Ok(0);
    }

    let is_gzip = input_data.len() >= 2 && input_data[0] == 0x1f && input_data[1] == 0x8b;
    let is_zlib = input_data.len() >= 2 && input_data[0] == 0x78;

    if args.force && !is_gzip && !is_zlib {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        writer.write_all(input_data)?;
        writer.flush()?;
        return Ok(0);
    }

    let format = if is_gzip {
        CompressionFormat::Gzip
    } else if is_zlib {
        CompressionFormat::Zlib
    } else {
        CompressionFormat::Gzip
    };

    let stdout = stdout();
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());

    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            let is_bgzf = has_bgzf_markers(input_data);
            let is_multi = !is_bgzf && is_likely_multi_member(input_data);
            let can_parallelize = args.processes > 1 && (is_bgzf || is_multi);

            if debug_enabled() {
                eprintln!(
                    "[gzippy] decompress_stdin: len={} bgzf={} multi={} parallel={} procs={}",
                    input_data.len(),
                    is_bgzf,
                    is_multi,
                    can_parallelize,
                    args.processes
                );
            }

            if is_bgzf {
                let threads = if can_parallelize { args.processes } else { 1 };
                crate::bgzf::decompress_bgzf_parallel(input_data, &mut writer, threads)?;
            } else if can_parallelize {
                let output = crate::decompression::decompress_gzip_to_vec(input_data, args.processes)?;
                writer.write_all(&output)?;
            } else {
                crate::decompression::decompress_single_member(input_data, &mut writer, args.processes)?;
            }
        }
        CompressionFormat::Zlib => {
            crate::decompression::decompress_zlib_turbo(input_data, &mut writer)?;
        }
    }

    writer.flush()?;
    Ok(0)
}

fn decompress_directory(dirname: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    use walkdir::WalkDir;
    let mut exit_code = 0;
    for entry in WalkDir::new(dirname) {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && crate::utils::is_compressed_file(path) {
            let path_str = path.to_string_lossy();
            match decompress_file(&path_str, args) {
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

/// Dispatch a memory-mapped buffer to the correct decompressor.
fn decompress_to_writer<W: Write>(
    mmap: &Mmap,
    writer: &mut W,
    format: CompressionFormat,
    args: &GzippyArgs,
) -> GzippyResult<u64> {
    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            let is_gzip = mmap.len() >= 2 && mmap[0] == 0x1f && mmap[1] == 0x8b;
            if !is_gzip {
                return Ok(0);
            }
            let bgzf = has_bgzf_markers(&mmap[..]);
            let multi = is_likely_multi_member(&mmap[..]);
            let can_parallelize = args.processes > 1 && (bgzf || multi);

            if debug_enabled() {
                eprintln!(
                    "[gzippy] decompress_file: len={} bgzf={} multi={} parallel={} procs={}",
                    mmap.len(), bgzf, multi, can_parallelize, args.processes
                );
            }

            if bgzf {
                let threads = if can_parallelize { args.processes } else { 1 };
                let bytes = crate::bgzf::decompress_bgzf_parallel(&mmap[..], writer, threads)?;
                Ok(bytes)
            } else if can_parallelize {
                let output = crate::decompression::decompress_gzip_to_vec(&mmap[..], args.processes)?;
                let len = output.len() as u64;
                writer.write_all(&output)?;
                Ok(len)
            } else {
                crate::decompression::decompress_single_member(&mmap[..], writer, args.processes)
            }
        }
        CompressionFormat::Zlib => {
            crate::decompression::decompress_zlib_turbo(&mmap[..], writer)
        }
    }
}

fn detect_format(path: &Path) -> CompressionFormat {
    crate::utils::detect_format_from_file(path).unwrap_or(CompressionFormat::Gzip)
}

fn get_output_filename(input_path: &Path, args: &GzippyArgs, data: &[u8]) -> std::path::PathBuf {
    if args.stdout {
        return input_path.to_path_buf();
    }
    if args.name {
        if let Some(fname) = extract_gzip_fname(data) {
            if !fname.is_empty() {
                let mut output = input_path.to_path_buf();
                output.set_file_name(&fname);
                return output;
            }
        }
    }
    if args.suffix != ".gz" {
        let suffix = args.suffix.trim_start_matches('.');
        if let Some(name) = input_path.file_name().and_then(|n| n.to_str()) {
            let lower = name.to_lowercase();
            let suffix_with_dot = format!(".{}", suffix);
            if lower.ends_with(&suffix_with_dot) {
                let mut output = input_path.to_path_buf();
                output.set_file_name(&name[..name.len() - suffix_with_dot.len()]);
                return output;
            }
        }
    }
    let mut output_path = strip_compression_extension(input_path);
    if output_path == input_path {
        output_path = input_path.to_path_buf();
        let current_name = output_path.file_name().unwrap().to_str().unwrap();
        output_path.set_file_name(format!("{}.out", current_name));
    }
    output_path
}

fn print_stats(input_size: u64, output_size: u64, path: &Path) {
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
    let (in_size, in_unit) = human_size(input_size);
    let (out_size, out_unit) = human_size(output_size);
    eprintln!(
        "{}: {:.1}{} → {:.1}{} ({:.1}x expansion)",
        filename, in_size, in_unit, out_size, out_unit, 1.0 / ratio
    );
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
