//! gzippy - The fastest parallel gzip
//!
//! A drop-in replacement for gzip that uses multiple processors for compression.
//! Inspired by [pigz](https://zlib.net/pigz/) by Mark Adler.

use std::env;
use std::path::Path;
use std::process;

mod algebraic_decode;
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
mod fast_inflate;
mod format;
mod golden_tests;
mod hyper_parallel;
mod inflate_tables;
mod isal;
mod jit_decode;
mod libdeflate_decode;
mod libdeflate_entry;
mod libdeflate_ext;
mod marker_decode;
mod multi_symbol;
mod optimization;
mod packed_lut;
mod parallel_compress;
mod parallel_decompress;
mod parallel_inflate;
mod pipelined_compress;
mod precomputed_decode;
mod rapidgzip_decoder;
mod scheduler;
mod simd_copy;
mod simd_huffman;
mod simd_inflate;
mod simd_parallel_decode;
mod simple_optimizations;
mod specialized_decode;
mod speculative_batch;
mod thread_pool;
mod turbo_inflate;
mod two_level_table;
mod ultimate_decode;
mod ultra_decoder;
mod ultra_decompress;
mod ultra_fast_inflate;
mod ultra_inflate;
mod unified_table;
mod utils;
mod vector_huffman;

use cli::GzippyArgs;
use error::GzippyError;

const VERSION: &str = concat!("gzippy ", env!("CARGO_PKG_VERSION"));

fn main() {
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
        println!("  compressed  uncompressed  ratio  uncompressed_name");
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
fn list_file(filename: &str, _args: &GzippyArgs) -> Result<(u64, u64), GzippyError> {
    use std::fs;

    let metadata =
        fs::metadata(filename).map_err(|_| GzippyError::FileNotFound(filename.to_string()))?;
    let compressed_size = metadata.len();

    // Read the gzip trailer to get ISIZE (uncompressed size mod 2^32)
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

    let ratio = if uncompressed_size > 0 {
        (1.0 - compressed_size as f64 / uncompressed_size as f64) * 100.0
    } else {
        0.0
    };

    // Get the output name (what the file would decompress to)
    let output_name = crate::utils::strip_compression_extension(Path::new(filename));
    let display_name = output_name
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(filename);

    println!(
        "{:>12}  {:>12}  {:4.1}%  {}",
        compressed_size, uncompressed_size, ratio, display_name
    );

    Ok((compressed_size, uncompressed_size))
}

fn print_help() {
    println!("Usage: gzippy [OPTION]... [FILE]...");
    println!();
    println!("Compress or decompress FILEs (by default, compress in place).");
    println!("Uses multiple processors for parallel compression.");
    println!();
    println!("Options:");
    println!("  -1..-9           Compression level (1=fast, 9=best, default=6)");
    println!("  --level N        Set compression level 1-12");
    println!("  --ultra          Ultra compression (level 11, near-zopfli)");
    println!("  --max            Maximum compression (level 12, closest to zopfli)");
    println!("  -c, --stdout     Write to stdout, keep original files");
    println!("  -d, --decompress Decompress");
    println!("  -f, --force      Force overwrite of output file");
    println!("  -k, --keep       Keep original file");
    println!("  -p, --processes  Number of threads (default: all CPUs)");
    println!("  -r, --recursive  Recurse into directories");
    println!("  -q, --quiet      Suppress output");
    println!("  -v, --verbose    Verbose output");
    println!("  -h, --help       Show this help");
    println!("  -V, --version    Show version");
    println!("  -L, --license    Show license");
    println!();
    println!("Compression levels:");
    println!("  1-6              Fast (libdeflate, parallel decompress)");
    println!("  7-9              Balanced (zlib-ng, gzip-compatible)");
    println!("  10-12            Ultra (libdeflate high, near-zopfli ratio)");
    println!();
    println!("Examples:");
    println!("  gzippy file.txt          Compress file.txt → file.txt.gz");
    println!("  gzippy -d file.txt.gz    Decompress file.txt.gz → file.txt");
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
