//! gzippy - The fastest parallel gzip
//!
//! A drop-in replacement for gzip that uses multiple processors for compression.
//! Inspired by [pigz](https://zlib.net/pigz/) by Mark Adler.

use std::env;
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[macro_use]
mod test_utils;
mod benchmark_datasets;
mod bgzf;
mod block_finder;
mod block_finder_lut;
mod bmi2;
mod cli;
mod combined_lut;
mod compression;
mod consume_first_decode;
mod consume_first_table;
mod decompression;
mod double_literal;
mod error;
mod format;
mod golden_tests;
mod inflate_tables;
mod isal;
mod isal_compress;
mod jit_decode;
mod libdeflate_decode;
mod libdeflate_entry;
mod libdeflate_ext;
mod marker_decode;
mod optimization;
mod packed_lut;
mod parallel_compress;
mod pipelined_compress;
mod scheduler;
mod simd_copy;
mod simd_huffman;
mod simple_optimizations;
mod specialized_decode;
mod thread_pool;
mod two_level_table;
mod ultra_fast_inflate;
mod ultra_inflate;
mod utils;
mod vector_huffman;

use cli::GzippyArgs;
use error::GzippyError;

const VERSION: &str = concat!("gzippy ", env!("CARGO_PKG_VERSION"));

/// Track the current output file so signal handlers can clean it up.
/// When set, an incomplete output file exists that should be deleted on abort.
static OUTPUT_FILE: Mutex<Option<String>> = Mutex::new(None);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Set the current output file path for signal handler cleanup.
pub fn set_output_file(path: Option<String>) {
    if let Ok(mut guard) = OUTPUT_FILE.lock() {
        *guard = path;
    }
}

fn install_signal_handlers() {
    unsafe {
        // SIGINT (Ctrl-C), SIGTERM, SIGHUP: clean up and exit
        for &sig in &[libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            libc::signal(sig, signal_handler as *const () as libc::sighandler_t);
        }
        // SIGPIPE: exit quietly (e.g., piping to head)
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

extern "C" fn signal_handler(sig: libc::c_int) {
    // Mark as interrupted (atomic, signal-safe)
    INTERRUPTED.store(true, Ordering::SeqCst);

    // Try to clean up the output file.
    // Mutex::lock may not be signal-safe, but try_lock is better.
    // In the worst case we just skip cleanup.
    if let Ok(guard) = OUTPUT_FILE.try_lock() {
        if let Some(ref path) = *guard {
            let _ = std::fs::remove_file(path);
        }
    }

    // Restore default handler and re-raise so parent sees correct signal
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

fn main() {
    install_signal_handlers();

    let result = run();

    match result {
        Ok(exit_code) => process::exit(exit_code),
        Err(e) => {
            eprintln!("gzippy: {}", e);
            process::exit(1);
        }
    }
}

fn run() -> Result<i32, GzippyError> {
    let args = GzippyArgs::parse()?;

    if args.version {
        println!("{}", VERSION);
        return Ok(0);
    }

    if args.help {
        print_help();
        return Ok(0);
    }

    if args.license {
        print_license();
        return Ok(0);
    }

    // Support gunzip/ungzippy/zcat/gzcat symlinks
    let program_path = env::args().next().unwrap_or_else(|| "gzippy".to_string());
    let program_name = Path::new(&program_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("gzippy");

    let decompress = args.decompress
        || program_name.contains("ungzippy")
        || program_name.contains("gunzip")
        || program_name == "zcat"
        || program_name == "gzcat";

    // zcat/gzcat imply decompress-to-stdout
    let stdout_mode = args.stdout || program_name == "zcat" || program_name == "gzcat";

    // Refuse to write compressed binary data to a terminal (unless -f)
    if !decompress && stdout_mode && !args.force {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            eprintln!(
                "gzippy: compressed data not written to a terminal. Use -f to force compression."
            );
            return Ok(1);
        }
    }

    // Apply stdout_mode back to args for downstream use
    let mut args = args;
    if stdout_mode {
        args.stdout = true;
    }

    // --test implies decompress mode
    let decompress = decompress || args.test;

    let mut exit_code = 0;

    // Handle --list mode
    if args.list {
        if args.files.is_empty() {
            eprintln!("gzippy: --list does not support stdin");
            return Ok(1);
        }
        if args.verbose {
            println!(
                "method    crc     date  time  {:>12} {:>12}  ratio  uncompressed_name",
                "compressed", "uncompressed"
            );
        } else {
            println!("  compressed  uncompressed  ratio  uncompressed_name");
        }
        let mut total_comp = 0u64;
        let mut total_uncomp = 0u64;
        for file in &args.files {
            match list_file(file, &args) {
                Ok((comp, uncomp)) => {
                    total_comp += comp;
                    total_uncomp += uncomp;
                }
                Err(e) => {
                    eprintln!("gzippy: {}: {}", file, e);
                    exit_code = 1;
                }
            }
        }
        if args.files.len() > 1 {
            let ratio = if total_uncomp > 0 {
                (1.0 - total_comp as f64 / total_uncomp as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "{:>12}  {:>12}  {:4.1}%  (totals)",
                total_comp, total_uncomp, ratio
            );
        }
        return Ok(exit_code);
    }

    if args.files.is_empty() {
        // Process stdin
        if decompress {
            if args.test {
                exit_code = test_stdin(&args)?;
            } else {
                exit_code = decompression::decompress_stdin(&args)?;
            }
        } else {
            exit_code = compression::compress_stdin(&args)?;
        }
    } else {
        // Process files
        for file in &args.files {
            let result = if args.test {
                test_file(file, &args)
            } else if decompress {
                decompression::decompress_file(file, &args)
            } else {
                compression::compress_file(file, &args)
            };

            match result {
                Ok(code) => {
                    if code != 0 {
                        exit_code = code;
                    }
                }
                Err(e) => {
                    eprintln!("gzippy: {}: {}", file, e);
                    exit_code = 1;
                }
            }
        }
    }

    Ok(exit_code)
}

/// Test integrity of a compressed file by decompressing to a sink
fn test_file(filename: &str, args: &GzippyArgs) -> Result<i32, GzippyError> {
    use memmap2::Mmap;
    use std::fs::File;

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(GzippyError::FileNotFound(filename.to_string()));
    }

    let input_file = File::open(input_path)?;
    let mmap = unsafe { Mmap::map(&input_file)? };

    // Decompress into a Vec (discarded after) to verify integrity
    let mut sink = Vec::new();
    let result = decompression::decompress_gzip_to_writer(&mmap, &mut sink);

    match result {
        Ok(_) => {
            if !args.quiet {
                eprintln!("{}: OK", filename);
            }
            Ok(0)
        }
        Err(e) => {
            eprintln!("{}: {}", filename, e);
            Ok(1)
        }
    }
}

/// Test integrity of compressed data from stdin
fn test_stdin(args: &GzippyArgs) -> Result<i32, GzippyError> {
    use std::io::Read;

    let stdin = std::io::stdin();
    let mut input_data = Vec::new();
    {
        let mut reader = std::io::BufReader::new(stdin.lock());
        reader.read_to_end(&mut input_data)?;
    }

    let mut sink = Vec::new();
    let result = decompression::decompress_gzip_to_writer(&input_data, &mut sink);

    match result {
        Ok(_) => {
            if !args.quiet {
                eprintln!("stdin: OK");
            }
            Ok(0)
        }
        Err(e) => {
            eprintln!("stdin: {}", e);
            Ok(1)
        }
    }
}

/// List compressed file information (gzip -l format)
fn list_file(filename: &str, args: &GzippyArgs) -> Result<(u64, u64), GzippyError> {
    use std::fs;

    let metadata =
        fs::metadata(filename).map_err(|_| GzippyError::FileNotFound(filename.to_string()))?;
    let compressed_size = metadata.len();

    // Read the gzip file
    let data = fs::read(filename).map_err(GzippyError::Io)?;

    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b {
        return Err(GzippyError::invalid_argument(format!(
            "{}: not in gzip format",
            filename
        )));
    }

    // ISIZE is last 4 bytes of the gzip file
    let isize_bytes = &data[data.len() - 4..];
    let uncompressed_size = u32::from_le_bytes([
        isize_bytes[0],
        isize_bytes[1],
        isize_bytes[2],
        isize_bytes[3],
    ]) as u64;

    // CRC32 is 4 bytes before ISIZE
    let crc_bytes = &data[data.len() - 8..data.len() - 4];
    let crc32 = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);

    let ratio = if uncompressed_size > 0 {
        (1.0 - compressed_size as f64 / uncompressed_size as f64) * 100.0
    } else {
        0.0
    };

    // Get the output name - check for FNAME in header first
    let fname = extract_list_fname(&data);
    let display_name = if let Some(ref name) = fname {
        name.as_str()
    } else {
        let output_name = crate::utils::strip_compression_extension(Path::new(filename));
        // Use a leaked string since we need a &str that outlives the function
        // This is fine since list_file is called a limited number of times
        let name = output_name
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(filename)
            .to_string();
        // We need to handle the borrow differently for verbose vs non-verbose
        Box::leak(name.into_boxed_str())
    };

    if args.verbose {
        // Parse header for verbose info
        let mtime = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let date_str = if mtime > 0 {
            // Format as "Jan 01 2024 12:00" using basic conversion
            format_unix_timestamp(mtime)
        } else {
            "                ".to_string()
        };

        println!(
            "defla {:08x} {} {:>12} {:>12} {:4.1}%  {}",
            crc32, date_str, compressed_size, uncompressed_size, ratio, display_name
        );
    } else {
        println!(
            "{:>12}  {:>12}  {:4.1}%  {}",
            compressed_size, uncompressed_size, ratio, display_name
        );
    }

    Ok((compressed_size, uncompressed_size))
}

/// Extract FNAME from gzip header for list display
fn extract_list_fname(data: &[u8]) -> Option<String> {
    if data.len() < 10 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }
    let flags = data[3];
    if flags & 0x08 == 0 {
        return None;
    }
    let mut pos = 10;
    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return None;
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    let start = pos;
    while pos < data.len() && data[pos] != 0 {
        pos += 1;
    }
    if pos >= data.len() {
        return None;
    }
    String::from_utf8(data[start..pos].to_vec()).ok()
}

/// Format a Unix timestamp into a basic date/time string
fn format_unix_timestamp(timestamp: u32) -> String {
    const MONTHS: &[&str] = &[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    // Simple Unix timestamp to date conversion
    let secs = timestamp as u64;
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;

    // Calculate year/month/day from days since epoch
    let mut year = 1970u32;
    let mut remaining_days = days;

    loop {
        let days_in_year =
            if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) {
                366
            } else {
                365
            };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_months: &[u64] = if leap {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 0usize;
    for (i, &dim) in days_in_months.iter().enumerate() {
        if remaining_days < dim {
            month = i;
            break;
        }
        remaining_days -= dim;
    }
    let day = remaining_days + 1;

    format!(
        "{} {:2} {} {:02}:{:02}",
        MONTHS[month], day, year, hours, minutes
    )
}

fn print_help() {
    println!("Usage: gzippy [OPTION]... [FILE]...");
    println!();
    println!("Compress or decompress FILEs (by default, compress in place).");
    println!("Uses multiple processors for parallel compression.");
    println!();
    println!("Options:");
    println!("  -1..-9              Compression level (1=fast, 9=best, default=6)");
    println!("  --level N           Set compression level 1-12");
    println!("  --ultra             Ultra compression (level 11, near-zopfli)");
    println!("  --max               Maximum compression (level 12, closest to zopfli)");
    println!("  -c, --stdout        Write to stdout, keep original files");
    println!("  -d, --decompress    Decompress");
    println!("  -f, --force         Force overwrite / compress links / pass-through");
    println!("  -k, --keep          Keep original file");
    println!("  -l, --list          List compressed file info");
    println!("  -t, --test          Test compressed file integrity");
    println!("  -n, --no-name       Don't save/restore original name and timestamp");
    println!("  -N, --name          Save/restore original name and timestamp");
    println!("  -m, --no-time       Don't save/restore modification time");
    println!("  -M, --time          Save/restore modification time (pigz)");
    println!("  -p, --processes N   Number of threads (default: all CPUs)");
    println!("  -b, --blocksize N   Block size for parallel compression");
    println!("  -r, --recursive     Recurse into directories");
    println!("  -R, --rsyncable     Make output rsync-friendly");
    println!("  -S, --suffix .suf   Use suffix .suf instead of .gz");
    println!("  -Y, --synchronous   Synchronous output (fsync after write)");
    println!("  -i, --independent   Force independent blocks (parallel decompress)");
    println!("  -C, --comment TEXT  Add comment to gzip header");
    println!("  -H, --huffman       Huffman-only compression");
    println!("  -U, --rle           Run-length encoding compression");
    println!("  -q, --quiet         Suppress output");
    println!("  -v, --verbose       Verbose output");
    println!("  -h, --help          Show this help");
    println!("  -V, --version       Show version");
    println!("  -L, --license       Show license");
    println!();
    println!("Compression levels:");
    println!("  1-6              Fast (libdeflate, parallel decompress)");
    println!("  7-9              Balanced (zlib-ng, gzip-compatible)");
    println!("  10-12            Ultra (libdeflate high, near-zopfli ratio)");
    println!();
    println!("Examples:");
    println!("  gzippy file.txt          Compress file.txt -> file.txt.gz");
    println!("  gzippy -d file.txt.gz    Decompress file.txt.gz -> file.txt");
    println!("  gzippy -p4 -9 file.txt   Compress with 4 threads, best compression");
    println!("  cat file | gzippy > out  Compress stdin to stdout");
}

fn print_license() {
    println!("gzippy - The fastest gzip");
    println!();
    println!("Inspired by pigz by Mark Adler, Copyright (C) 2007-2023");
    println!();
    println!("zlib License - see LICENSE file for details.");
}
