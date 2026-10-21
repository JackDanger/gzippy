//! gzippy - The fastest parallel gzip
//!
//! A drop-in replacement for gzip that uses multiple processors for compression.
//! Inspired by [pigz](https://zlib.net/pigz/) by Mark Adler.

// Global rpmalloc experiment — gated behind `global-rpmalloc` feature.
// Prior history (below) noted mimalloc + jemalloc both regressed. rpmalloc
// has different characteristics (lock-free per-thread heaps); worth one
// fresh data point especially since vendor uses rpmalloc as their per-Vec
// allocator. NOT recommended for production — vendor explicitly avoids
// global rpmalloc per `ChunkData.hpp:24` ("memory slab reuse issues").
#[cfg(feature = "global-rpmalloc")]
#[global_allocator]
static GLOBAL: rpmalloc::RpMalloc = rpmalloc::RpMalloc;

// Global allocator: system default (glibc on Linux).
//
// History note (2026-05-19): tried mimalloc and jemalloc as global
// allocators to mirror vendor's rpmalloc (`core/FasterVector.hpp:38-64`).
// mimalloc: wall 0.94s → 1.78s (sys 1.7s → 8.4s) — aggressive page
// release via madvise hurt on memory-pressured host.
// jemalloc: SM throughput 560 → 462 MB/s at 20-trial median (CPUs
// utilized jumped to 5.9 but throughput dropped — fighting itself).
//
// Per vendor's actual usage: `FasterVector.hpp:38-42` "rpmalloc as a
// custom allocator in the few places we know to be performance-critical",
// NOT global. Vendor's rpnew.h global override is explicitly avoided
// per `ChunkData.hpp:24` ("memory slab reuse issues in rpmalloc").
//
// Conclusion: global allocator swap is not the right tool. The right
// port of vendor's pattern would be a per-`Vec` allocator on the
// `Vec<u8>` and `Vec<u16>` inside ChunkData, which stable Rust does
// not support without unstable `Allocator` API. Alternative paths:
// (a) explicit mmap + MAP_POPULATE for chunk buffers; (b) larger
// chunk_buffer_pool cap so warm Vecs always available.

use std::env;
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

// ── Core infrastructure ───────────────────────────────────────────────────────
mod analyze;
mod cli;
mod error;
mod format;
mod index_mode;
mod utils;

// ── Compression & decompression stacks ───────────────────────────────────────
mod compress; // engine (mod.rs) + io, parallel, pipelined, optimization, simple
mod coz_probe; // Coz causal-profiling probe shim (no-op without --features coz)
mod decompress; // engine (mod.rs) + io, format, bgzf, SIMD tables, scan_inflate

// ── Hardware backends (ISA-L, libdeflate FFI) ─────────────────────────────────
mod backends; // isal, isal_compress, isal_decompress, libdeflate

// ── Threading infrastructure ──────────────────────────────────────────────────
mod infra; // thread_pool, scheduler

// ── Test infrastructure ───────────────────────────────────────────────────────
#[cfg(test)]
mod fixtures; // frozen mechanism fixtures (the lib exports these as `gzippy::fixtures`)
#[cfg(test)]
mod holdout; // NEVER-TUNE holdout generators + sha256 (fixtures' pins use it)
#[cfg(test)]
mod tests;

use cli::GzippyArgs;
use error::GzippyError;

/// Compile-time build flavor string: "parallel-sm+isal", "parallel-sm+pure", or
/// "legacy-serial". Derived from cfg predicates in build.rs and surfaced via
/// `--version` output and the first `GZIPPY_DEBUG=1` line.
pub(crate) const BUILD_FLAVOR: &str = env!("BUILD_FLAVOR");

const VERSION: &str = concat!(
    "gzippy ",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("BUILD_FLAVOR"),
    ")",
);

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

    // `anatomy-counters` feature (default OFF, compiles to nothing when off):
    // one machine-parsable line on stderr at process end, ahead of both exit
    // paths below (including the fast `libc::_exit` success path, which skips
    // normal userspace teardown).
    #[cfg(feature = "anatomy-counters")]
    compress::deflate::anatomy_counters::flush_to_stderr();

    // `anatomy-wall` feature (default OFF, compiles to nothing when off):
    // the wall-clock phase-timer sibling of `anatomy-counters` above — see
    // src/compress/deflate/anatomy_wall.rs module docs.
    #[cfg(feature = "anatomy-wall")]
    compress::deflate::anatomy_wall::flush_to_stderr();

    match result {
        Ok(exit_code) => fast_exit_success(exit_code),
        Err(e) => {
            eprintln!("gzippy: {}", e);
            process::exit(1);
        }
    }
}

/// Verified-success process exit.
///
/// On the clean-success path (exit_code == 0) EVERY side effect is already
/// complete before we get here: all output has been written AND flushed to the
/// OS by the decompress/compress entry points (their `BufWriter`s call
/// `flush()?`), CRC32+ISIZE were verified against the gzip trailer inside the
/// decode driver, the source file was removed and metadata preserved (see
/// `decompress::io::decompress_file`'s `Ok` branch), and — because `run()`
/// processes every requested file before returning — this fires only after the
/// LAST unit of work. There is nothing left to do in userspace except tear
/// down the address space, which the kernel's `exit_group` does regardless.
///
/// So we `libc::_exit` directly, skipping the redundant *userspace* teardown
/// that `process::exit`'s `libc::exit()` would run first: the allocator's
/// `madvise(MADV_DONTNEED)` span-decommit flood, C static destructors from the
/// linked codecs, and any atexit handlers. That teardown cost is proportional
/// to peak RSS (the measured AMD-Zen2 mid-T loss vs rapidgzip) and is pure
/// overhead once the output is durable.
///
/// Byte-transparency guarantees:
///  - Only `exit_code == 0` (fully-verified success) fast-exits; every non-zero
///    Ok code (skipped symlink=2, per-file error=1) and every `Err` takes the
///    normal `process::exit`, which preserves exit codes AND lets
///    `decompress_file` delete a partial dest on CRC/size failure.
///  - We flush the process-wide `stdout`/`stderr` handles first, covering any
///    `println!`/`eprintln!` (stats, `-v`) still sitting in the global
///    `LineWriter` buffer. The bulk data path is already flushed to the fd.
fn fast_exit_success(exit_code: i32) -> ! {
    if exit_code == 0 {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // SAFETY: `_exit` is async-signal-safe and simply issues the
        // exit_group syscall. All userspace state we care about (output bytes)
        // is already flushed to the OS above and inside the entry points.
        unsafe { libc::_exit(0) };
    }
    process::exit(exit_code);
}

fn run() -> Result<i32, GzippyError> {
    let args = GzippyArgs::parse()?;

    // First GZIPPY_DEBUG=1 line: build flavor (before any routing or path prints).
    if crate::utils::debug_enabled() {
        eprintln!("[gzippy] build-flavor={BUILD_FLAVOR}");
    }

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

    // `--tune <spec>` (l1-tune feature only): apply the L1 matchfinder
    // config override for every compress call this process makes. Applied
    // once, up front — matches the "one process, one candidate config" shape
    // `fulcrum l1search` drives (shells out to an l1-tune-built gzippy per
    // (config, corpus) pair rather than looping `tune::set` in-process the
    // way the deleted `examples/l1_search.rs` did).
    #[cfg(feature = "l1-tune")]
    if let Some(spec) = &args.tune {
        let cfg = compress::deflate::parse::tune::parse_spec(spec)
            .map_err(GzippyError::invalid_argument)?;
        compress::deflate::parse::tune::set(cfg);
    }

    // --analyze short-circuits the normal compress/decompress flow.
    if let Some(code) = analyze::maybe_run(&args) {
        return Ok(code);
    }

    // --index and --seek short-circuit the normal decompress flow.
    if let Some(code) = index_mode::maybe_run(&args) {
        return Ok(code);
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

    // `--verbose` flows to the parallel-SM driver as a real `verbose: bool`
    // parameter threaded through the decode call graph (decompress::io ->
    // decompress::mod -> parallel::single_member -> parallel::sm_driver ->
    // parallel::chunk_fetcher::drive_impl), which drains FetcherStatistics +
    // ChunkFetcherStatistics + the deletion-trap counters to stderr at
    // end-of-decode. Mirror of vendor's `args.verbose` ->
    // `setStatisticsEnabled(true)` at vendor/rapidgzip/librapidarchive/src/tools/rapidgzip.cpp:164. (Previously an
    // internal env-var round-trip; replaced batch 4g.)

    // zcat/gzcat imply decompress-to-stdout
    let stdout_mode = args.stdout || program_name == "zcat" || program_name == "gzcat";

    // Refuse to write compressed binary data to a terminal (unless -f).
    // Applies to: explicit -c/stdout mode, OR no files given (stdin→stdout compress).
    if !decompress && !args.test && !args.force && (stdout_mode || args.files.is_empty()) {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            // gzip's treat_stdin message verbatim (gzip.c:610-620; probed via
            // a pty): two lines, then do_exit(ERROR) = exit 1.
            eprintln!(
                "gzippy: compressed data not written to a terminal. Use -f to force compression.\n\
                 For help, type: gzippy -h"
            );
            return Ok(1);
        }
    }
    // Refuse to read compressed data from a terminal (unless -f).
    if (decompress || args.test) && !args.force && args.files.is_empty() {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() {
            eprintln!(
                "gzippy: compressed data not read from a terminal. Use -f to force decompression.\n\
                 For help, type: gzippy -h"
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

    // ── Flag honesty (issues #302–#305) ──────────────────────────────────────
    // Rival flags we PARSE but whose documented semantics we do not deliver
    // are rejected LOUDLY before any work starts, instead of exiting 0 with
    // output that silently lacks the property the flag exists to provide
    // (non-rsyncable "rsyncable" archives, gzip bytes under .zip names,
    // strategy flags that change nothing). Compress mode only: on
    // decompress/test/list these flags are inert for the rivals too, so a
    // `-d` pipeline that happens to carry them keeps working. Exit 2
    // (warning-class): nothing was attempted, nothing was written.
    if !decompress && !args.list {
        if args.rsyncable {
            eprintln!(
                "gzippy: -R/--rsyncable: rsyncable output is not yet implemented (see issue #302)"
            );
            return Ok(2);
        }
        match args.format {
            format::CompressionFormat::Zlib => {
                // Implemented (issue #303): -z emits a real RFC 1950 zlib
                // container. The only flag that cannot be honoured is a
                // comment: zlib has no FCOMMENT field, so accepting `-C`
                // would silently drop what the user typed.
                if args.comment.is_some() {
                    eprintln!(
                        "gzippy: -z/--zlib cannot store a comment (zlib containers have no \
                         comment field); refusing to silently drop it — write gzippy gzip \
                         output with -C, or drop -C"
                    );
                    return Ok(2);
                }
            }
            format::CompressionFormat::Zip => {
                eprintln!("gzippy: -K/--zip: zip container output is not yet implemented (see issue #303)");
                return Ok(2);
            }
            format::CompressionFormat::Gzip => {}
        }
        if args.alias.is_some() {
            eprintln!("gzippy: -A/--alias is only meaningful with --zip, which is not yet implemented (see issue #303)");
            return Ok(2);
        }
        if args.huffman {
            eprintln!("gzippy: -H/--huffman (Huffman-only strategy) is not implemented; refusing to silently compress normally");
            return Ok(2);
        }
        if args.rle {
            eprintln!("gzippy: -U/--rle (run-length-encoding strategy) is not implemented; refusing to silently compress normally");
            return Ok(2);
        }
        if args.independent {
            eprintln!("gzippy: -i/--independent block independence is not implemented; refusing to silently ignore it");
            return Ok(2);
        }
        if args.comment.is_some() && (args.stdout || args.files.is_empty()) {
            eprintln!("gzippy: -C/--comment is not stored on the stdout path; write to a file, or drop -C");
            return Ok(2);
        }
        // Not a rejection — a LOUD divergence notice. -0 keeps its gzip/pigz
        // meaning (STORE, no compression). igzip's level scale is 0..=3 where
        // -0 is its FASTEST REAL COMPRESSOR, so an igzip script pointed here
        // would otherwise silently get output larger than its input (issue
        // #305). pigz -0 users get exactly what pigz gives them, plus one
        // stderr line (suppressed by -q).
        if args.compression_level == 0 && !args.quiet {
            eprintln!(
                "gzippy: warning: -0 stores without compression (gzip/pigz semantics); \
                 igzip's -0 is its fastest compression level (see issue #305)"
            );
        }
    }

    let mut exit_code = 0;

    // Handle --list mode
    if args.list {
        if args.files.is_empty() {
            eprintln!("gzippy: --list does not support stdin");
            return Ok(1);
        }
        // gzip.c 553-556: `-q` sets `quiet = 1; verbose = 0` — mirror that
        // ratio/verbose interplay locally for -l headers and totals (member
        // rows print regardless). gzippy's parser keeps the two flags
        // independent; last-flag-wins ordering is not mirrored.
        let verbose = args.verbose && !args.quiet;
        // gzip.c do_list (1504-1605): the header is first_time's `if (!quiet)`
        // row; member rows print regardless of -q — and the header prints only
        // when an actual row is about to print (do_list's first_time is set
        // inside the per-file call, so `gzip -l /nope` prints NOTHING on
        // stdout). gzippy's established -v header shape is kept as-is; the
        // non-verbose header is copied verbatim from the vendored format
        // string "%*.*s %*.*s  ratio uncompressed_name" at
        // positive_off_t_width. `header_done` mirrors `first_time`. The -v
        // header carries the fixed 9-column shape used since the gzip -v era.
        let mut header_done = false;
        let mut total_comp = 0u64;
        let mut total_uncomp = 0u64;
        let mut last_header_bytes = 0u64;
        for file in &args.files {
            if Path::new(file).is_dir() {
                // Same contract as compress/decompress/-t (see comment on
                // compress_file): exit 2, this exact message, skip the file,
                // keep processing the rest.
                if !args.quiet {
                    eprintln!("gzippy: {} is a directory -- ignored", file);
                }
                crate::utils::bump_exit_code(&mut exit_code, 2);
                continue;
            }
            match list_file(file, &args, verbose, &mut header_done) {
                Ok(Some((comp, uncomp, header_bytes))) => {
                    total_comp += comp;
                    total_uncomp += uncomp;
                    last_header_bytes = header_bytes;
                }
                Ok(None) => {
                    // The failure was already rendered inside list_file (in
                    // gzip's progerror shape); credit the ERROR class.
                    crate::utils::bump_exit_code(&mut exit_code, 1);
                }
                Err(e) => {
                    // Per-file rendering: gzip's get_method format-class
                    // messages print with a LEADING blank line
                    // (gzip.c:1473-1475); everything else is plain
                    // "program: file: message" (gzip.c:389-394).
                    let lead = if e.has_leading_blank_line() { "\n" } else { "" };
                    eprintln!("{}gzippy: {}: {}", lead, file, e);
                    crate::utils::bump_exit_code(&mut exit_code, 1);
                }
            }
        }
        // gzip.c 574: `if (list && !quiet && file_count > 1) do_list(-1, -1);`
        // do_list(-1) starts `if (total_in <= 0 || total_out <= 0) return;`
        if args.files.len() > 1 && !args.quiet && !(total_comp == 0 || total_uncomp == 0) {
            // do_list(-1) re-reads the CURRENT global header_bytes, i.e. the
            // last listed row's: "header_bytes is not meaningful but used to
            // ensure the same ratio if there is a single file" (gzip.c 1545).
            let num = total_uncomp as i128 - (total_comp as i128 - last_header_bytes as i128);
            let ratio = 100.0 * num as f64 / total_uncomp as f64;
            // Verbose totals carry a 28-space strip ("method  crc     date  time  "
            // length, gzip.c 1537) and drop the sizes only under -q
            // (`if (verbose || !quiet)`); non-verbose totals always print them.
            let mut row = String::new();
            if verbose {
                row.push_str("                            ");
            }
            if verbose || !args.quiet {
                row.push_str(&format!(
                    "{:>width$} {:>width$} ",
                    total_comp,
                    total_uncomp,
                    width = positive_off_t_width()
                ));
            }
            println!("{}{:5.1}% (totals)", row, ratio);
        }
        return Ok(exit_code);
    }

    if args.files.is_empty() {
        // Process stdin
        if decompress {
            if args.test {
                exit_code = test_stdin(&args)?;
            } else {
                exit_code = decompress::io::decompress_stdin(&args)?;
            }
        } else {
            exit_code = compress::io::compress_stdin(&args)?;
        }
    } else {
        // Process files
        for file in &args.files {
            let result = if args.test {
                test_file(file, &args)
            } else if decompress {
                decompress::io::decompress_file(file, &args)
            } else {
                compress::io::compress_file(file, &args)
            };

            match result {
                Ok(code) => {
                    crate::utils::bump_exit_code(&mut exit_code, code);
                }
                Err(e) => {
                    // Per-file rendering: gzip's get_method-class messages
                    // ("not in gzip format", "unexpected end of file") print
                    // with a LEADING blank line (gzip.c:1473-1475,
                    // util.c:394-402); everything else is plain
                    // "program: file: message" (gzip.c:389-394).
                    let lead = if e.has_leading_blank_line() { "\n" } else { "" };
                    eprintln!("{}gzippy: {}: {}", lead, file, e);
                    crate::utils::bump_exit_code(&mut exit_code, 1);
                }
            }
        }
    }

    Ok(exit_code)
}

/// Test integrity of a compressed file by decompressing to a sink
fn test_file(filename: &str, args: &GzippyArgs) -> Result<i32, GzippyError> {
    use memmap2::Mmap;

    // gzip's -t open semantics: internally `case 't': test = decompress =
    // to_stdout = 1` (gzip.c:464-465), so -t DOES follow symlinks
    // (open_and_stat's O_NOFOLLOW applies only when NOT to_stdout,
    // gzip.c:1022-1025 — probed: `gzip -t link-to-real.gz` tests the target)
    // and a missing input rides the SAME decompress suffix ladder as -d
    // (open_input_file, gzip.c:1100-1146).
    let (input_file, resolved_path) = match crate::utils::open_gzip_input(
        Path::new(filename),
        true, // -t is internally DECOMPRESS: the suffix ladder applies
        true, // -t sets to_stdout: follow links
        args.force,
        &args.suffix,
    ) {
        Ok((f, p)) => (f, p),
        Err(failure) => return Ok(crate::utils::report_open_failure(failure, args.quiet)),
    };
    let name = resolved_path.to_string_lossy().into_owned();

    // Directory arm with the OPEN's stat (gzip.c:731-732): WARNING class,
    // exit 2, this exact message — established by execution (gzip 1.12/1.14:
    // `gzip -t somedir` → "is a directory -- ignored", exit 2).
    let istat = input_file.metadata().map_err(|e| GzippyError::os(&e))?;
    if istat.is_dir() {
        if !args.quiet {
            eprintln!("gzippy: {} is a directory -- ignored", name);
        }
        return Ok(2);
    }

    let mmap = unsafe { Mmap::map(&input_file).map_err(|e| GzippyError::os(&e))? };

    // Container-boundary classification (BEFORE any decode) mirrors gzip's
    // get_method exactly: EOF before a complete magic ("unexpected end of
    // file" — probed 0- and 1-byte inputs; util.c:394-402) and 2+ bytes with
    // no gzip/zlib shape ("not in gzip format"; get_method gzip.c:1473-1475).
    // WITHOUT this gate the parallel decode path reports OK for garbage (exit
    // 0, nothing printed), inverting `gzip -t`'s integrity verdict.
    let is_gzip = mmap.len() >= 2 && mmap[0] == 0x1f && mmap[1] == 0x8b;
    let cont = mmap.len() >= 2 && mmap[0] == 0x78;
    if !is_gzip && !cont {
        // gzip get_method (gzip.c:1461-1462): force + to_stdout copies the
        // input through unchanged instead of failing — `-tf <junk>` exits 0
        // silently (probed; -t sets to_stdout internally, so this holds for
        // every -t with -f on non-container bytes).
        if args.force {
            return Ok(0);
        }
        // ERROR class prints unconditionally (get_method has no -q gate).
        if mmap.len() < 2 {
            eprintln!("\ngzippy: {}: unexpected end of file", name);
        } else {
            eprintln!("\ngzippy: {}: not in gzip format", name);
        }
        return Ok(1);
    }
    // get_method's header walk applies to -t exactly like -d (the
    // decompress_file block carries the citations): EOF inside the header is
    // read_error's blank-lined "unexpected end of file"; a CM byte other
    // than DEFLATED is "unknown method N -- not supported". A `-t` verdict
    // must not come back OK from the parallel decoder for either. ERROR
    // class: unconditional.
    if is_gzip {
        use crate::decompress::format::FirstMemberBoundary;
        match crate::decompress::format::classify_first_member(&mmap) {
            FirstMemberBoundary::UnexpectedEof => {
                eprintln!("\ngzippy: {}: unexpected end of file", name);
                return Ok(1);
            }
            FirstMemberBoundary::UnknownMethod(cm) => {
                eprintln!("gzippy: {}: unknown method {} -- not supported", name, cm);
                return Ok(1);
            }
            FirstMemberBoundary::Header => {}
        }
    }
    // get_method's header walk applies to -t exactly like -d (the
    // decompress_file block carries the citations): EOF inside the header is
    // read_error's blank-lined "unexpected end of file"; a CM byte other
    // than DEFLATED is "unknown method N -- not supported". A `-t` verdict
    // must not come back OK from the parallel decoder for either.
    if is_gzip {
        use crate::decompress::format::FirstMemberBoundary;
        match crate::decompress::format::classify_first_member(&mmap) {
            FirstMemberBoundary::UnexpectedEof => {
                if !args.quiet {
                    eprintln!("\ngzippy: {}: unexpected end of file", name);
                }
                return Ok(1);
            }
            FirstMemberBoundary::UnknownMethod(cm) => {
                if !args.quiet {
                    eprintln!("gzippy: {}: unknown method {} -- not supported", name, cm);
                }
                return Ok(1);
            }
            FirstMemberBoundary::Header => {}
        }
    }

    // Decompress into a Vec (discarded after) to verify integrity. Route by
    // CONTENT: pigz -t verifies its own -z (.zz) files too, so a zlib stream
    // goes to the zlib decoder; everything else keeps the gzip path (and its
    // decode-internal error for foreign/truncated bytes).
    let mut sink = Vec::new();
    let result = if crate::compress::zlib_container::is_zlib_stream(&mmap) {
        crate::decompress::decompress_zlib_turbo(&mmap, &mut sink)
    } else {
        decompress::decompress_gzip_to_writer(&mmap, &mut sink)
    };

    match result {
        Ok(_) => {
            // gzip's contract, established by execution: `gzip -t` is SILENT on
            // success and only `gzip -tv` reports. We printed unconditionally
            // (unless --quiet), which inverts the standard failure check
            // `gzip -t f 2>&1 | grep -q .` for every drop-in user. Errors below
            // still report unconditionally. (pigz is silent even with -tv; we
            // follow gzip, the primary drop-in target.)
            if args.verbose {
                eprintln!("{}:\t OK", name);
            }
            Ok(0)
        }
        Err(e) => {
            eprintln!("gzippy: {}: {}", name, e);
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

    // Same container-boundary classification as the file path (gzip's
    // get_method taxonomy, probed): EOF before a complete magic is
    // "unexpected end of file"; 2+ bytes with no gzip/zlib shape is "not in
    // gzip format"; -f pass-through (gzip.c:1461-1462, "pass input
    // unchanged" — `gzip -tf <junk>` exits 0 silently). gzip names the peer
    // "stdin" itself (treat_stdin: strcpy(ifname, "stdin"), gzip.c:623).
    let is_gzip = input_data.len() >= 2 && input_data[0] == 0x1f && input_data[1] == 0x8b;
    let cont = input_data.len() >= 2 && input_data[0] == 0x78;
    if !is_gzip && !cont {
        if args.force {
            return Ok(0);
        }
        // ERROR class prints unconditionally (get_method has no -q gate).
        if input_data.len() < 2 {
            eprintln!("\ngzippy: stdin: unexpected end of file");
        } else {
            eprintln!("\ngzippy: stdin: not in gzip format");
        }
        return Ok(1);
    }
    // get_method's header walk on stdin too (decompress_file's block holds
    // the citations): read_error EOF in the header, "unknown method" for a
    // foreign CM byte — same texts, peer named "stdin". ERROR class:
    // unconditional.
    if is_gzip {
        use crate::decompress::format::FirstMemberBoundary;
        match crate::decompress::format::classify_first_member(&input_data) {
            FirstMemberBoundary::UnexpectedEof => {
                eprintln!("\ngzippy: stdin: unexpected end of file");
                return Ok(1);
            }
            FirstMemberBoundary::UnknownMethod(cm) => {
                eprintln!("gzippy: stdin: unknown method {} -- not supported", cm);
                return Ok(1);
            }
            FirstMemberBoundary::Header => {}
        }
    }
    // get_method's header walk on stdin too (decompress_file's block holds
    // the citations): read_error EOF in the header, "unknown method" for a
    // foreign CM byte — same texts, peer named "stdin".
    if is_gzip {
        use crate::decompress::format::FirstMemberBoundary;
        match crate::decompress::format::classify_first_member(&input_data) {
            FirstMemberBoundary::UnexpectedEof => {
                if !args.quiet {
                    eprintln!("\ngzippy: stdin: unexpected end of file");
                }
                return Ok(1);
            }
            FirstMemberBoundary::UnknownMethod(cm) => {
                if !args.quiet {
                    eprintln!("gzippy: stdin: unknown method {} -- not supported", cm);
                }
                return Ok(1);
            }
            FirstMemberBoundary::Header => {}
        }
    }

    let mut sink = Vec::new();
    let result = if crate::compress::zlib_container::is_zlib_stream(&input_data) {
        crate::decompress::decompress_zlib_turbo(&input_data, &mut sink)
    } else {
        decompress::decompress_gzip_to_writer(&input_data, &mut sink)
    };

    match result {
        Ok(_) => {
            // See the note on the file path above: silent on success unless -v.
            if args.verbose {
                eprintln!("stdin:\t OK");
            }
            Ok(0)
        }
        Err(e) => {
            eprintln!("gzippy: stdin: {}", e);
            Ok(1)
        }
    }
}

/// List compressed file information (gzip -l format).
///
/// Contract evidence — vendored gzip source (vendor/gzip):
/// - `gzip.c do_list` (1504-1605): ONE row per file; `bytes_in` = input file
///   size (stat), `bytes_out` = ISIZE read from the last 8 bytes
///   (`bytes_in = lseek(ifd, (off_t)(-8), SEEK_END)` + `read(buf, 8)` in the
///   RECORD_IO branch; the body never inflates).
/// - `gzip.c get_method` (1431-1432): the reader's per-member overhead is
///   `header_bytes = inptr + 2*4` — every byte the header parse consumed
///   (fixed 10-byte header + FEXTRA + FNAME + FCOMMENT + FHCRC) plus the
///   8 trailer bytes.
/// - `util.c display_ratio` (416-422): prints `"%5.1f%%"` of
///   `den == 0 ? 0 : 100.0 * num / den` where the caller passes
///   `num = bytes_out - (bytes_in - header_bytes)`, `den = bytes_out`.
///   NO clamping — expansion prints negative values as-is.
/// - Name column = `ofname` generated by `make_ofname` (gzip.c 1173-1248)
///   from the on-disk input name via `get_suffix` (gzip.c 982-1022): the
///   gzip suffix stripped case-insensitively, ".tgz"/".taz" mapped to
///   ".tar", names without a gzip suffix printed verbatim. The header FNAME
///   is consulted ONLY for an explicit -N: with the -l default
///   `no_name = decompress` (gzip.c 530) get_method's ORIG_NAME branch
///   discards the stored name (gzip.c 1384).
/// - Reads are two-point by construction: the ~192 KB header prefix from
///   the front and 8 bytes at EOF-8.
///
/// gzip's -l open semantics: internally `case 'l': list = decompress =
/// to_stdout = 1` (gzip.c:462), so links are followed (no O_NOFOLLOW) and
/// a missing input rides the SAME decompress suffix ladder as -d,
/// including the resolved naming (`gzip -l nofile` reports
/// "nofile.gz: No such file or directory" — probed).
///
/// Known divergence (documented, not papered over): gzip 1.12 runs the
/// decompression loop in -l mode and re-parses EVERY member header, so
/// `part_nb > 1` zeroes header_bytes for concatenated multi-member FILES;
/// this cheap reader parses only the first header and always charges that
/// member's overhead. Single-member files — the target of every pinned
/// row — are identical.
///
/// Returns `Ok(Some((compressed, uncompressed, header_bytes)))` on a listed
/// row; `Ok(None)` when the failure was already REPORTED here (in gzip's
/// fopen-error shape) and the caller should just credit `1`; `Err` is
/// reserved for errors the caller renders (`NotInGzipFormat`,
/// `UnexpectedEof`, head-truncation IO).
fn list_file(
    filename: &str,
    args: &GzippyArgs,
    verbose: bool,
    header_done: &mut bool,
) -> Result<Option<(u64, u64, u64)>, GzippyError> {
    use std::io::{Read, Seek, SeekFrom};

    // gzip's -l open semantics: internally `case 'l': list = decompress =
    // to_stdout = 1` (gzip.c:462), so links are followed (no O_NOFOLLOW) and
    // a missing input rides the SAME decompress suffix ladder as -d,
    // including the resolved naming (`gzip -l nofile` reports
    // "nofile.gz: No such file or directory" — probed).
    let (mut handle, _resolved_path) = match crate::utils::open_gzip_input(
        Path::new(filename),
        true, // -l is internally DECOMPRESS: the suffix ladder applies
        true, // -l sets to_stdout: follow links
        args.force,
        &args.suffix,
    ) {
        Ok((f, p)) => (f, p),
        Err(failure) => {
            // Already rendered in gzip's progerror shape; callers credit the
            // ERROR class (1) without printing again.
            let _ = crate::utils::report_open_failure(failure, args.quiet);
            return Ok(None);
        }
    };
    // do_list's bytes_in is the size of the file that was actually opened
    // (gzip stats ifname after the suffix ladder resolves it).
    let compressed_size = handle.metadata().map_err(|e| GzippyError::os(&e))?.len();

    // Point read 1 — the header prefix. Far upper bound of a gzip header:
    // 10 fixed + 2+XLEN + FNAME + FCOMMENT (each string <= 65535 + NUL)
    // + FHCRC. The deflate body below it is NEVER read — the two-point
    // structure is the point of -l (gzip.c 1556-1561).
    const LIST_HEADER_MAX: usize = 10 + 2 + 65_535 + 65_536 + 65_536 + 2;
    let prefix_len = (compressed_size as usize).min(LIST_HEADER_MAX);
    let mut prefix = vec![0u8; prefix_len];
    let mut got = 0usize;
    while got < prefix_len {
        match handle.read(&mut prefix[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(GzippyError::Io(e)),
        }
    }
    prefix.truncate(got);

    // get_method's two first-member failure classes (both probed: gzip -l of
    // an empty/2-byte .gz prints "unexpected end of file" via read_error —
    // util.c:394-402 — while garbage bytes reach "not in gzip format" —
    // gzip.c:1473-1475). Both are ERROR class with the leading blank line;
    // the per-file loop prepends "program: file:".
    if prefix.len() < 2 {
        return Err(GzippyError::UnexpectedEof);
    }
    let is_gzip = prefix[0] == 0x1f && prefix[1] == 0x8b;
    if !is_gzip {
        return Err(GzippyError::NotInGzipFormat);
    }
    if compressed_size < 18 {
        // Good magic with an incomplete member: gzip's header scan runs out
        // of input and read_error fires (probed: `gzip -l` of a 2/9/20-byte
        // header fragment → "unexpected end of file", exit 1).
        return Err(GzippyError::UnexpectedEof);
    }

    let header = parse_list_header(&prefix);
    // display_ratio (util.c 416-422) with do_list's caller arguments:
    // num = bytes_out - (bytes_in - header_bytes), den = bytes_out.
    let header_bytes = header.header_bytes.unwrap_or((10 + 8) as u64);

    // Point read 2 — the trailer: CRC32 + ISIZE at EOF-8.
    handle.seek(SeekFrom::End(-8)).map_err(GzippyError::Io)?;
    let mut trailer = [0u8; 8];
    handle.read_exact(&mut trailer).map_err(GzippyError::Io)?;
    let crc32 = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let uncompressed_size =
        u32::from_le_bytes([trailer[4], trailer[5], trailer[6], trailer[7]]) as u64;

    let num = uncompressed_size as i128 - (compressed_size as i128 - header_bytes as i128);
    let ratio = if uncompressed_size == 0 {
        0.0
    } else {
        100.0 * num as f64 / uncompressed_size as f64
    };

    // uncompressed_name: gzip's ofname — on-disk name minus the gzip suffix;
    // header FNAME only for an explicit -N. See make_ofname/get_method.
    let fname = header.fname;
    let display: String = if args.name_explicit && fname.as_deref().is_some_and(|f| !f.is_empty()) {
        // get_method's ORIG_NAME branch (gzip.c 1388-1401): copy the
        // stored name over the on-disk directory prefix intact, then
        // strip the header name's own directory component (basename).
        let dir = Path::new(filename)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let base = fname
            .as_deref()
            .unwrap()
            .rsplit('/')
            .next()
            .unwrap_or_default();
        if dir.is_empty() {
            base.to_string()
        } else {
            format!("{}/{}", dir, base)
        }
    } else {
        list_output_name(filename, &args.suffix)
    };

    // done_header (do_list's first_time): the header prints only when the
    // first actual ROW is about to print — a failed listing leaves stdout
    // empty (`gzip -l /nope/nope.gz 2>/dev/null | wc -c` == 0, probed).
    if !*header_done {
        if verbose {
            println!(
                "method  crc     date  time    compressed uncompressed  ratio uncompressed_name"
            );
        } else if !args.quiet {
            println!("{}", list_header_line());
        }
        *header_done = true;
    }

    // Same -q-clears-verbose mirror as the -l orchestration above (gzip.c
    // 553-556): a quieted run lists plain rows.
    if args.verbose && !args.quiet {
        let date_str = if header.mtime > 0 {
            format_unix_timestamp(header.mtime)
        } else {
            "            ".to_string() // 12 spaces matching "MMM DD HH:MM"
        };
        println!(
            "defla {:08x} {} {:>12} {:>12} {:4.1}% {}",
            crc32, date_str, compressed_size, uncompressed_size, ratio, display
        );
    } else {
        println!(
            "{:>width$} {:>width$} {:5.1}% {}",
            compressed_size,
            uncompressed_size,
            ratio,
            display,
            width = positive_off_t_width()
        );
    }

    Ok(Some((compressed_size, uncompressed_size, header_bytes)))
}

/// The first-header facts -l needs: how many bytes the header parse consumed
/// (None when the header does not parse — flags fields run off the read
/// window, unterminated strings, CM != DEFLATED), plus FNAME and MTIME.
/// Field order mirrors gzip's get_method: FEXTRA -> FNAME -> FCOMMENT ->
/// FHCRC.
struct ListHeader {
    header_bytes: Option<u64>,
    fname: Option<String>,
    mtime: u32,
}

impl ListHeader {
    fn unparsable(mtime: u32) -> Self {
        ListHeader {
            header_bytes: None,
            fname: None,
            mtime,
        }
    }
}

fn parse_list_header(prefix: &[u8]) -> ListHeader {
    // RFC 1952 flag bits (vendor names: gzip.h FEXTRA etc).
    const FEXTRA: u8 = 0x04;
    const FNAME_FLAG: u8 = 0x08;
    const FCOMMENT: u8 = 0x10;
    const FHCRC: u8 = 0x02;

    if prefix.len() < 10 || prefix[0] != 0x1f || prefix[1] != 0x8b {
        return ListHeader::unparsable(0);
    }
    let mtime = u32::from_le_bytes([prefix[4], prefix[5], prefix[6], prefix[7]]);
    // CM != 8 is gzip's "unknown method -- not supported" (gzip.c 1319-1326);
    // for an existing -l row we fall back to the bare fixed-header overhead.
    if prefix[2] != 0x08 {
        return ListHeader::unparsable(mtime);
    }
    let flags = prefix[3];
    let mut pos = 10usize;

    if flags & FEXTRA != 0 {
        if pos + 2 > prefix.len() {
            return ListHeader::unparsable(mtime);
        }
        let xlen = u16::from_le_bytes([prefix[pos], prefix[pos + 1]]) as usize;
        pos += 2 + xlen;
        if pos > prefix.len() {
            return ListHeader::unparsable(mtime);
        }
    }
    let mut fname = None;
    if flags & FNAME_FLAG != 0 {
        let nul = match prefix[pos..].iter().position(|&b| b == 0) {
            Some(n) => pos + n,
            None => return ListHeader::unparsable(mtime),
        };
        fname = String::from_utf8(prefix[pos..nul].to_vec()).ok();
        pos = nul + 1;
    }
    if flags & FCOMMENT != 0 {
        let nul = match prefix[pos..].iter().position(|&b| b == 0) {
            Some(n) => pos + n,
            None => return ListHeader::unparsable(mtime),
        };
        pos = nul + 1;
    }
    if flags & FHCRC != 0 {
        pos += 2;
    }
    if pos > prefix.len() {
        return ListHeader::unparsable(mtime);
    }
    ListHeader {
        header_bytes: Some((pos + 8) as u64),
        fname,
        mtime,
    }
}

/// gzip's get_suffix (gzip.c 982-1022) + make_ofname's decompress branch
/// (gzip.c 1181-1201): the last min(len, MAX_SUFFIX+2) bytes of the name are
/// compared lowercased against the known gzip suffixes; a matching suffix
/// must sit at the name's very end with a non-separator character before it.
/// ".tgz"/".taz" map to ".tar" (make_ofname 1195-1198), every other match is
/// stripped; a name without a gzip suffix is listed under its own name
/// (make_ofname 1183-1186: `-l` tries all files).
fn list_output_name(input: &str, z_suffix: &str) -> String {
    // vendor known_suffixes (Unix tailoring): z_suffix first (gzip.c 995),
    // then the fixed set (gzip.c 988); the bare "z" candidate is
    // MSDOS/VMS-only tailoring (MAX_EXT_CHARS) and absent on Unix.
    const KNOWN: [&str; 7] = [".gz", ".z", ".taz", ".tgz", "-gz", "-z", "_z"];
    let name_len = input.len();
    // MAX_SUFFIX == 30 on Unix (tailor.h); strlwr window = MAX_SUFFIX + 2.
    let window_len = name_len.min(32);
    let lower: Vec<u8> = input.as_bytes()[name_len - window_len..]
        .iter()
        .map(|b| b.to_ascii_lowercase())
        .collect();
    let mut hit: Option<&str> = None;
    for cand in std::iter::once(z_suffix).chain(KNOWN.iter().copied()) {
        let s = cand.len();
        if s == 0 || window_len <= s {
            continue;
        }
        let tail = &lower[window_len - s..];
        // get_suffix lowercases the window before every compare (1011).
        let pre = lower[window_len - s - 1];
        if tail == cand.as_bytes() && pre != b'/' {
            hit = Some(cand);
            break;
        }
    }
    let Some(cand) = hit else {
        // No gzip suffix: make_ofname returns OK with ofname unchanged.
        return input.to_string();
    };
    let Some(stem) = input.get(..name_len - cand.len()) else {
        // Guard: an ASCII suffix can never end inside a multi-byte char.
        return input.to_string();
    };
    // Decompress branch: strlwr(suff) first, so the mapping is case-free;
    // .tgz/.taz become .tar, everything else is dropped (1195-1201).
    if cand.eq_ignore_ascii_case(".tgz") || cand.eq_ignore_ascii_case(".taz") {
        format!("{}.tar", stem)
    } else {
        stem.to_string()
    }
}

/// do_list's header at positive_off_t_width (gzip.c 1530):
/// "%*.*s %*.*s  ratio uncompressed_name" over "compressed"/"uncompressed".
fn list_header_line() -> String {
    format!(
        "{:>width$} {:>width$}  ratio uncompressed_name",
        "compressed",
        "uncompressed",
        width = positive_off_t_width()
    )
}

/// gzip.c do_list (1517-1522): decimal digits of OFF_T_MAX
/// (`for (o = OFF_T_MAX; 9 < o; o /= 10) positive_off_t_width++;`); off_t is
/// i64 on our platforms, so i64::MAX = 2^63-1 = 19 digits.
fn positive_off_t_width() -> usize {
    let mut width = 1usize;
    let mut o = i64::MAX;
    while o > 9 {
        o /= 10;
        width += 1;
    }
    width
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

    // gzip format: "Jan  1 12:00" — no year, day right-aligned in 2 chars
    format!("{} {:2} {:02}:{:02}", MONTHS[month], day, hours, minutes)
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
    println!("  --ultra             True zopfli (level 11, single-member; -p tunes intra-block parallelism only)");
    println!("  --max               Maximum compression (level 12, near-optimal parse, near-zopfli ratio)");
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
    println!("  -b, --blocksize N   Block size in KiB for parallel compression (pigz");
    println!("                      semantics; k/m/g suffixes accepted)");
    println!("  -r, --recursive     Recurse into directories");
    println!("  -S, --suffix .suf   Use suffix .suf instead of .gz");
    println!("  -Y, --synchronous   Synchronous output (fsync after write)");
    println!("  -C, --comment TEXT  Add comment to gzip header (file output only)");
    println!("  -q, --quiet         Suppress output");
    println!("  -v, --verbose       Verbose output");
    println!("  -h, --help          Show this help");
    println!("  -V, --version       Show version");
    println!("  -L, --license       Show license");
    println!();
    println!("Compression levels (one pure-Rust DEFLATE engine, no C-FFI compressor):");
    println!("  0                Store only, no compression (gzip/pigz semantics)");
    println!("  1-9              Fast/balanced (hash-chain + lazy parse; ratio tracks libdeflate)");
    println!("  10,12            Near-optimal parse (near-zopfli ratio, parallel)");
    println!("  11               True zopfli (single-member, slowest, best ratio)");
    println!();
    println!("igzip compatibility: gzippy follows the gzip/pigz flag contract where they");
    println!("collide with igzip's. igzip's 0..3 level scale is NOT emulated: -0 stores");
    println!("(igzip's -0 is its fastest compressor) and -2/-3 are weak-fast gzip levels");
    println!("(igzip's default/best). -z takes pigz's meaning (zlib container, RFC 1950),");
    println!("not igzip's 'compress' no-op. See issue #305.");
    println!();
    println!("Not yet implemented (rejected loudly rather than silently misbehaving):");
    println!("  -R/--rsyncable (#302), -K/--zip and -A/--alias (#303),");
    println!("  -H/--huffman, -U/--rle, -i/--independent");
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
