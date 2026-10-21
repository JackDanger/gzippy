//! File, stdin, and directory I/O for decompression.
//!
//! Entry points: `decompress_file` and `decompress_stdin`.
//! All decompression logic is in `decompression.rs`; this module only handles
//! filesystem concerns: mmap, output path selection, metadata preservation,
//! stats printing, and signal-handler registration.

use std::fs::File;
use std::io::{self, stdin, stdout, BufReader, BufWriter, Read, Write};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::Path;

#[cfg(unix)]
#[inline]
fn out_fd_of<F: AsRawFd>(f: &F) -> Option<i32> {
    Some(f.as_raw_fd())
}
#[cfg(not(unix))]
#[inline]
fn out_fd_of<F>(_f: &F) -> Option<i32> {
    None
}

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

use memmap2::Mmap;

use crate::cli::GzippyArgs;
use crate::decompress::format::{
    extract_gzip_fname, extract_gzip_mtime, has_bgzf_markers, is_likely_multi_member,
};

use crate::error::{GzippyError, GzippyResult};
use crate::format::CompressionFormat;
use crate::utils::{debug_enabled, preserve_metadata, strip_compression_extension};

const STREAM_BUFFER_SIZE: usize = 1024 * 1024;

pub fn decompress_file(filename: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    if filename == "-" {
        return decompress_stdin(args);
    }

    // gzip's whole OPEN contract for a decompress input — ELOOP link refusal
    // ("Too many levels of symbolic links", ERROR class), the missing-file
    // suffix ladder (a missing bare name resolves against `<name>.gz` etc.
    // like gzip's open_input_file), and the RESOLVED input path for all
    // downstream naming — is one helper (gzip.c:1072-1163; see utils docs).
    let (input_file, resolved) = match crate::utils::open_gzip_input(
        Path::new(filename),
        true, // -d decompress: the suffix ladder applies (gzip.c:1100+)
        args.stdout,
        args.force,
        &args.suffix,
    ) {
        Ok((f, p)) => (f, p),
        Err(failure) => return Ok(crate::utils::report_open_failure(failure, args.quiet)),
    };
    let input_path: &Path = resolved.as_path();
    let name = input_path.to_string_lossy().into_owned();

    // stat of the OPENED file (gzip's fstat-after-open order, treat_file:717):
    // a mode-000 DIRECTORY fails open first ("Permission denied") before the
    // directory notice is reachable, exactly as in gzip.
    let istat = match input_file.metadata() {
        Ok(m) => m,
        Err(e) => return Err(GzippyError::os(&e)),
    };
    let file_size = istat.len();

    if istat.is_dir() {
        // gzip treats a directory as an argument to recurse into (with -r) or
        // ignores it as a WARNING (gzip.h:263-264 WARN bumps exit_code only
        // when it is currently OK): "is a directory -- ignored", exit 2,
        // treat_file gzip.c:728-733. Established by execution (gzip -d/-dc/-t/-l
        // on a real directory), not by reading the source. Multi-file runs keep
        // processing the rest.
        if args.recursive {
            return decompress_directory(filename, args);
        }
        if !args.quiet {
            eprintln!("gzippy: {} is a directory -- ignored", name);
        }
        return Ok(2);
    }

    // gzip.c:737-768: anything that is not a regular file (device, FIFO,
    // socket) is skipped with this exact know-when-NOT-to-fire message —
    // WARNING class ("-- ignored"), exit 2, and ONLY when not going to
    // stdout (to_stdout reads through: `gzip -c /dev/null` compresses). -t/-l
    // also set gzip's to_stdout, but they never reach this function.
    if !args.stdout && !istat.is_file() {
        if !args.quiet {
            eprintln!(
                "gzippy: {} is not a directory or a regular file - ignored",
                name
            );
        }
        return Ok(2);
    }
    // gzip.c:740-768, continued — mode-bit refusals AFTER the regular-file
    // check: setuid and setgid are refused regardless of -f, while the
    // sticky bit and a hardlink count of 2+ yield to -f; same ladder as the
    // compress arm (see that comment for the message-text generation note).
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // Normalize u32 std mode() to mode_t (u16 on Darwin) so the libc
        // S_IS* constants compile on both Unix flavors.
        let mode = istat.mode() as libc::mode_t;
        if !args.stdout {
            if mode & libc::S_ISUID != 0 {
                if !args.quiet {
                    eprintln!("gzippy: {} is set-user-ID on execution - ignored", name);
                }
                return Ok(2);
            }
            if mode & libc::S_ISGID != 0 {
                if !args.quiet {
                    eprintln!("gzippy: {} is set-group-ID on execution - ignored", name);
                }
                return Ok(2);
            }
            if !args.force {
                if mode & libc::S_ISVTX != 0 {
                    if !args.quiet {
                        eprintln!("gzippy: {} has the sticky bit set - file ignored", name);
                    }
                    return Ok(2);
                }
                if istat.nlink() > 1 {
                    if !args.quiet {
                        let other = istat.nlink() - 1;
                        eprintln!(
                            "gzippy: {} has {} other {} -- file ignored",
                            name,
                            other,
                            if other == 1 { "link" } else { "links" },
                        );
                    }
                    return Ok(2);
                }
            }
        }
    }
    let mmap = unsafe { Mmap::map(&input_file).map_err(|e| GzippyError::os(&e))? };
    let _ = mmap.advise(memmap2::Advice::Sequential);

    let is_compressed =
        mmap.len() >= 2 && ((mmap[0] == 0x1f && mmap[1] == 0x8b) || mmap[0] == 0x78);

    // gzip.c:1461-1462: force + to_stdout passes the input through unchanged.
    if args.force && args.stdout && !is_compressed {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        writer.write_all(&mmap)?;
        writer.flush()?;
        return Ok(0);
    }

    // make_ofname (gzip.c:1184-1200), decompress arm — runs BEFORE any
    // content read or decode (treat_file calls make_ofname at 774, get_method
    // is only reached in the work loop): an input without a recognized gzip
    // suffix is skipped as a WARNING — "unknown suffix -- ignored" — exit 2.
    // -t/-l bypass this (make_ofname returns OK for them) and never reach
    // this function; -c has no output NAME to protect (ofname := "stdout",
    // gzip.c:1444-1445). The gzip.c:1189 gate on verbosity: "Avoid annoying
    // messages with -r".
    if !args.stdout && crate::utils::known_gzip_suffix(&name, &args.suffix).is_none() {
        if args.verbose || (!args.recursive && !args.quiet) {
            eprintln!("gzippy: {}: unknown suffix -- ignored", name);
        }
        return Ok(2);
    }

    // Container-boundary classification (BEFORE any decode), mirroring
    // gzip's get_method taxonomy — both messages here are ERROR class,
    // printed with a leading blank line exactly as gzip prints them
    // (get_method gzip.c:1473-1475; read_error util.c:394-402; both probed
    // against gzip 1.12):
    //
    //   * fewer than 2 bytes: EOF before a complete magic → "unexpected end
    //     of file" (0- and 1-byte inputs);
    //   * 2+ bytes that are neither a gzip magic (old or new) nor a zlib
    //     container: "not in gzip format" (get_method's part_nb == 1 arm).
    //
    // With the magic present, get_method keeps consuming the variable
    // header: EOF before the header's end stays read_error ("unexpected end
    // of file", probed: `gzip -d 2-byte-1f8b` → exit 1), and a CM byte
    // other than 8 is get_method's own "unknown method N -- not supported"
    // (no blank line). Encode both with classify_first_member below;
    // decode-internal corruption messages (truncated BODY, bad deflate
    // data) are the decoder's own texts and are NOT re-mapped here.
    if !is_compressed {
        // ERROR-class messages print UNCONDITIONALLY: read_error and
        // get_method's fprintf have no quiet gate (only WARN-class messages
        // honor -q — gzip.h:263-265).
        if mmap.len() < 2 {
            eprintln!("\ngzippy: {}: unexpected end of file", name);
        } else {
            eprintln!("\ngzippy: {}: not in gzip format", name);
        }
        return Ok(1);
    }
    if mmap.len() >= 2 && mmap[0] == 0x1f && mmap[1] == 0x8b {
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
                // Same contract as compress_file (see the comment there):
                // non-interactive gzip does not error, it prints this exact
                // shape and exits 2 (WARNING), never blocking on input and
                // never returning 1. Verified by executing `gzip -d` on a
                // real file with an existing decompression target.
                if !args.quiet {
                    eprintln!(
                        "gzippy: {} already exists;\tnot overwritten",
                        output_path.display()
                    );
                }
                return Ok(2);
            }
        }
    }

    // Content complement (issue #303): the extension usually routes (.zz →
    // Zlib), but `-z` output redirected to any other name — or foreign zlib
    // data under any name — previously reached the gzip arm, which emits an
    // EMPTY output and still deletes the input. A valid zlib header (CM=8,
    // no FDICT, FCHECK ok — same sniff the gzip path's `is_compressed` uses
    // in one direction) routes to the zlib decoder; every other content
    // keeps the extension-based verdict unchanged.
    let format = if crate::compress::zlib_container::is_zlib_stream(&mmap) {
        CompressionFormat::Zlib
    } else {
        detect_format(input_path)
    };

    if let Some(ref output_path) = output_path {
        crate::set_output_file(Some(output_path.to_string_lossy().to_string()));
    }

    let result = if args.stdout {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        writer.flush()?;
        let out_fd = out_fd_of(writer.get_ref());
        let r = decompress_to_writer(&mmap, &mut writer, out_fd, format, args);
        writer.flush()?;
        r
    } else {
        let output_path = output_path.clone().unwrap();
        let output_file = File::create(&output_path)?;
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, output_file);
        writer.flush()?;
        let out_fd = out_fd_of(writer.get_ref());
        let r = decompress_to_writer(&mmap, &mut writer, out_fd, format, args);
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
                print_stats(
                    file_size,
                    output_size,
                    input_path,
                    output_path.as_deref(),
                    args,
                );
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
        let is_regular = meta
            .metadata()
            .map(|m| m.file_type().is_file())
            .unwrap_or(false);
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

    // Container-boundary classification, mirroring gzip's treat_stdin +
    // get_method (both probed): EMPTY stdin decompresses to "unexpected end
    // of file" (exit 1) unless -f passes it through; 2+ bytes with no
    // gzip/zlib shape do NOT decode silently — "not in gzip format", exit 1
    // (gzip names the peer "stdin"; the leading blank line is part of the
    // observable message, get_method gzip.c:1473-1475 / util.c:394-402).
    // -f copies the bytes through unchanged (gzip.c:1461-1462, `work = copy`)
    // and exits 0.
    let is_gzip = input_data.len() >= 2 && input_data[0] == 0x1f && input_data[1] == 0x8b;
    let is_zlib = input_data.len() >= 2 && input_data[0] == 0x78;

    if args.force && !is_gzip && !is_zlib {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        writer.write_all(input_data)?;
        writer.flush()?;
        return Ok(0);
    }

    if !is_gzip && !is_zlib {
        // ERROR class: unconditional (read_error / get_method have no -q gate).
        if input_data.len() < 2 {
            eprintln!("\ngzippy: stdin: unexpected end of file");
        } else {
            eprintln!("\ngzippy: stdin: not in gzip format");
        }
        return Ok(1);
    }
    // With the magic present, the same get_method header walk as the file
    // path applies to stdin (see that block for the citations): EOF before
    // the header's end is read_error's blank-lined "unexpected end of
    // file", a foreign CM byte is "unknown method N -- not supported".
    // ERROR class: unconditional.
    if is_gzip {
        use crate::decompress::format::FirstMemberBoundary;
        match crate::decompress::format::classify_first_member(input_data) {
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

    let verbose = args.verbose && !args.quiet;
    let in_bytes = input_data.len() as u64;

    // Exactly one of the two container shapes holds here (both garbage and
    // empty stdin were routed above), so the format is unambiguous.
    let format = if is_gzip {
        CompressionFormat::Gzip
    } else {
        CompressionFormat::Zlib
    };

    let stdout = stdout();
    let mut counted =
        CountingWriter::new(BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock()));

    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            // A first-member "GZ" hit only takes the BGZF fast path when the
            // WHOLE file is pure GZ (every member carries the subfield, walk ends
            // at EOF). A mixed concatenation (GZ ++ plain) is treated as
            // multi-member so it routes through the cross-member-capable path
            // instead of desyncing the BGZF block walk. [R2-#3]
            // Compute the two format-detection scans ONCE and reuse them both for
            // this routing decision AND the single-member classifier (threaded via
            // `decompress_single_member_prescanned`), so the ~16 MiB
            // `is_likely_multi_member` memmem walk is not repeated inside
            // `classify_gzip`. `raw_multi` is the bare scan result;
            // `is_multi` keeps the existing bgzf-exclusive routing semantics.
            let bgzf_markers = has_bgzf_markers(input_data);
            let is_bgzf = bgzf_markers && crate::decompress::bgzf::gz_coverage_is_pure(input_data);
            let raw_multi = is_likely_multi_member(input_data);
            let is_multi = !is_bgzf && raw_multi;
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
                crate::decompress::bgzf::decompress_bgzf_parallel(
                    input_data,
                    &mut counted,
                    threads,
                )?;
            } else if can_parallelize {
                let output = crate::decompress::decompress_gzip_to_vec(input_data, args.processes)?;
                counted.write_all(&output)?;
            } else {
                crate::decompress::decompress_single_member_prescanned(
                    input_data,
                    &mut counted,
                    args.processes,
                    args.verbose,
                    bgzf_markers,
                    raw_multi,
                )?;
            }
        }
        CompressionFormat::Zlib => {
            crate::decompress::decompress_zlib_turbo(input_data, &mut counted)?;
        }
    }

    counted.flush()?;
    if verbose {
        let out_bytes = counted.count;
        let ratio = if in_bytes > 0 {
            out_bytes as f64 / in_bytes as f64
        } else {
            1.0
        };
        let (in_size, in_unit) = human_size(in_bytes);
        let (out_size, out_unit) = human_size(out_bytes);
        eprintln!(
            "(stdin): {:.1}{} → {:.1}{} ({:.1}x expansion)",
            in_size, in_unit, out_size, out_unit, ratio
        );
    }
    Ok(0)
}

fn decompress_directory(dirname: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    use walkdir::WalkDir;
    let mut exit_code = 0;
    // Same treat_dir contract as compress_directory (see the comment there):
    // every readdir entry flows through — here gated by the archive-suffix
    // convention (is_compressed_file is extension-only, never opening the
    // path, so dangling links stay safe) — and a symlink INTO an archive is
    // therefore handed to decompress_file and refused with open's ELOOP
    // (ERROR class), like gzip's -r, instead of being silently skipped.
    for entry in WalkDir::new(dirname) {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type().is_dir() && crate::utils::is_compressed_file(path) {
            let path_str = path.to_string_lossy();
            match decompress_file(&path_str, args) {
                Ok(code) => {
                    // gzip's WARN-escalation ledger (gzip.h:263-264), same
                    // rule as the top-level file loop: ERROR (1) wins
                    // regardless of arrival order, WARNING (2) only sticks
                    // when nothing worse is recorded.
                    crate::utils::bump_exit_code(&mut exit_code, code);
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
    out_fd: Option<i32>,
    format: CompressionFormat,
    args: &GzippyArgs,
) -> GzippyResult<u64> {
    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            let is_gzip = mmap.len() >= 2 && mmap[0] == 0x1f && mmap[1] == 0x8b;
            if !is_gzip {
                return Ok(0);
            }
            // Pure-GZ only takes the BGZF fast path; a mixed GZ ++ plain
            // concatenation routes as multi-member. [R2-#3]
            // Compute both format-detection scans ONCE; reuse for this dispatch AND
            // the single-member classifier (via `decompress_single_member_fd_prescanned`)
            // so the ~16 MiB `is_likely_multi_member` walk is not repeated inside
            // `classify_gzip`. `raw_multi` is the bare scan; `multi` keeps the existing
            // bgzf-exclusive routing semantics.
            let bgzf_markers = has_bgzf_markers(&mmap[..]);
            let bgzf = bgzf_markers && crate::decompress::bgzf::gz_coverage_is_pure(&mmap[..]);
            let raw_multi = is_likely_multi_member(&mmap[..]);
            let multi = raw_multi;
            let can_parallelize = args.processes > 1 && (bgzf || multi);

            if debug_enabled() {
                eprintln!(
                    "[gzippy] decompress_file: len={} bgzf={} multi={} parallel={} procs={}",
                    mmap.len(),
                    bgzf,
                    multi,
                    can_parallelize,
                    args.processes
                );
            }

            if bgzf {
                let threads = if can_parallelize { args.processes } else { 1 };
                let bytes =
                    crate::decompress::bgzf::decompress_bgzf_parallel(&mmap[..], writer, threads)?;
                Ok(bytes)
            } else if can_parallelize {
                let output = crate::decompress::decompress_gzip_to_vec(&mmap[..], args.processes)?;
                let len = output.len() as u64;
                writer.write_all(&output)?;
                Ok(len)
            } else {
                crate::decompress::decompress_single_member_fd_prescanned(
                    &mmap[..],
                    writer,
                    out_fd,
                    args.processes,
                    args.verbose,
                    bgzf_markers,
                    raw_multi,
                )
            }
        }
        CompressionFormat::Zlib => crate::decompress::decompress_zlib_turbo(&mmap[..], writer),
    }
}

fn detect_format(path: &Path) -> CompressionFormat {
    crate::utils::detect_format_from_file(path).unwrap_or(CompressionFormat::Gzip)
}

fn get_output_filename(input_path: &Path, args: &GzippyArgs, data: &[u8]) -> std::path::PathBuf {
    if args.stdout {
        return input_path.to_path_buf();
    }
    // gzip(1) contract, issue #268: on decompression the stored FNAME is
    // honoured ONLY when the user passed -N/--name. `name` defaults to true —
    // that is the compress-side default (store the original name by default) —
    // so it cannot distinguish "user asked for name restoration" from "never
    // asked"; `name_explicit` can. Plain -d names its output after the on-disk
    // file minus the suffix below and ignores the header, and the gunzip
    // alias is plain -d and keeps that default.
    if args.name_explicit {
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

fn print_stats(
    input_size: u64,
    output_size: u64,
    input_path: &Path,
    output_path: Option<&Path>,
    args: &GzippyArgs,
) {
    // ratio: how much the original was compressed (positive = shrank, negative = grew)
    let saved_pct = if output_size > 0 {
        (1.0_f64 - input_size as f64 / output_size as f64) * 100.0
    } else {
        0.0
    };
    if args.verbosity >= 2 {
        // gzippy detail format for -vv
        let name = input_path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("<unknown>");
        let (in_sz, in_u) = human_size(input_size);
        let (out_sz, out_u) = human_size(output_size);
        let expansion = output_size as f64 / input_size.max(1) as f64;
        eprintln!(
            "{}: {:.1}{} → {:.1}{} ({:.1}x expansion)",
            name, in_sz, in_u, out_sz, out_u, expansion
        );
    } else {
        // gzip-compatible format for -v: "path:   X.X% -- replaced with outpath"
        let in_name = input_path.to_str().unwrap_or("<unknown>");
        match output_path {
            Some(out) => eprintln!(
                "{}:\t{:7.1}% -- replaced with {}",
                in_name,
                saved_pct.clamp(-99.9, 99.9),
                out.to_str().unwrap_or("<unknown>")
            ),
            None => eprintln!("{}:\t{:7.1}%", in_name, saved_pct.clamp(-99.9, 99.9)),
        }
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
