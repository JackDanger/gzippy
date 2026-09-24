//! T-SCOPE: **DISPATCH** — chooses the driver; runs at every thread count.
//!
//! File, stdin, and directory I/O for compression.
//!
//! Entry points: `compress_file` and `compress_stdin`.
//! All compression logic is in `compression.rs`; this module only handles
//! filesystem concerns: file reading, output path selection, metadata
//! preservation, stats printing, and signal-handler registration.

use std::fs::File;
use std::io::{self, stdin, stdout, BufWriter, Cursor, Write};
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
use crate::error::{GzippyError, GzippyResult};
use crate::utils::preserve_metadata;

/// Splices a full FNAME/MTIME gzip header over the minimal header the T1
/// encoders emit (issue #309: gzip's contract stores FNAME and MTIME when
/// compressing a named FILE; `gzip -l`/`gzip -dN` rely on them).
///
/// The T1 encoders write header + DEFLATE stream + trailer through one
/// writer, always starting with the level-specific minimal header. This
/// adapter swallows those 10 bytes, emits `replacement` in their place, and
/// passes everything after them through untouched — the DEFLATE bytes and
/// trailer cannot change, so T1 `-c` output (every graded board/tie-guard
/// cell) is not routed through this type at all and file output differs from
/// it ONLY in the header. Fails closed: if the swallowed bytes are not the
/// expected minimal header, it errors rather than emit a corrupt member.
struct HeaderSpliceWriter<W: Write> {
    inner: W,
    replacement: Vec<u8>,
    expected: [u8; 10],
    /// Bytes of the minimal header consumed so far (0..=10).
    seen: usize,
}

impl<W: Write> HeaderSpliceWriter<W> {
    fn new(inner: W, replacement: Vec<u8>, expected: [u8; 10]) -> Self {
        Self {
            inner,
            replacement,
            expected,
            seen: 0,
        }
    }
}

impl<W: Write> Write for HeaderSpliceWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.seen < self.expected.len() {
            let take = (self.expected.len() - self.seen).min(buf.len());
            if buf[..take] != self.expected[self.seen..self.seen + take] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "T1 encoder did not emit the fixed minimal gzip header; \
                     refusing to splice FNAME/MTIME over unknown bytes",
                ));
            }
            if self.seen == 0 && take > 0 {
                self.inner.write_all(&self.replacement)?;
            }
            self.seen += take;
            if take == buf.len() {
                return Ok(take);
            }
            let n = self.inner.write(&buf[take..])?;
            return Ok(take + n);
        }
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Map `input_file`, asking for [`TAIL_PAD`] bytes of readable slack past EOF so
/// the encoder can parse IN PLACE instead of copying the whole input.
///
/// Returns `(mapping, logical_len)`. The mapping is `logical_len + TAIL_PAD`
/// bytes when the slack was obtainable and exactly `logical_len` otherwise.
///
/// WHY. The padded encoder requires TAIL_PAD trailing bytes so the matchfinder's
/// speculative 4/8-byte loads stay in bounds. Without slack, the only way to
/// supply them is to copy the entire input — a 51 MB memcpy on monorepo.tar to
/// append 16 bytes. Measured 2026-08-21: explicit allocations ran at exactly
/// 1.50x the input on every corpus file regardless of compressibility, and peak
/// RSS at 2.5-2.7x, against pigz's flat ~2 MB.
///
/// HOW IT IS SAFE. A file's last page is partial whenever `len % page != 0`, and
/// the kernel zero-fills the remainder of that page; reading it is defined, and
/// mapping into it does not fault. Mapping into a page BEYOND the last one is
/// what raises SIGBUS, so we only ask for slack that fits inside the final
/// partial page, and fall back to the plain map (and the copy) otherwise —
/// including the exact-multiple-of-page case, where there is no partial page at
/// all.
fn map_with_tail_pad(file: &File, len: usize) -> std::io::Result<(memmap2::Mmap, usize)> {
    const TAIL_PAD: usize = crate::compress::deflate::INPLACE_TAIL_PAD;
    let page = page_size();
    let slack_in_last_page = if page == 0 { 0 } else { page - (len % page) };
    // `len % page == 0` gives slack_in_last_page == page, but there is no
    // partial page then — the next byte is in a page the file does not own.
    let can_extend = page != 0 && !len.is_multiple_of(page) && slack_in_last_page >= TAIL_PAD;
    if can_extend {
        // SAFETY: same contract as the plain `Mmap::map` below — the file is
        // opened read-only and not mutated concurrently by gzippy. The extra
        // TAIL_PAD bytes lie inside the file's final partial page, which the
        // kernel zero-fills, so they are readable and read as zero.
        let m = unsafe { memmap2::MmapOptions::new().len(len + TAIL_PAD).map(file) };
        if let Ok(m) = m {
            return Ok((m, len));
        }
        // Fall through to the plain map if the kernel refused the longer
        // mapping for any reason; correctness never depends on the fast path.
    }
    // SAFETY: as above.
    let m = unsafe { memmap2::Mmap::map(file)? };
    Ok((m, len))
}

#[cfg(unix)]
fn page_size() -> usize {
    // SAFETY: `sysconf` with a valid name is always safe; a negative return
    // means "no limit", which we treat as "no slack available".
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        0
    }
}

#[cfg(not(unix))]
fn page_size() -> usize {
    0
}

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
            // Contract established by execution (gzip 1.14 / pigz 2.8, macOS
            // + Linux): `gzip somedir` (no -r) prints exactly this shape and
            // exits 2 (WARNING, not ERROR) — never touches the directory,
            // never halts a multi-file invocation's remaining good files.
            // pigz disagrees (exit 1, "skipping: X is a directory"); gzip is
            // the primary drop-in target so we match gzip.
            if !args.quiet {
                eprintln!("gzippy: {} is a directory -- ignored", filename);
            }
            Ok(2)
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

    // Precondition ordering below is established BY EXECUTION against real
    // gzip 1.14 (never inferred from reading gzip's source): permission
    // (openability) beats EVERY other precondition and is never overridden
    // by `-f`; hardlink-refusal beats the already-suffix notice and the
    // refuse-overwrite notice, and IS overridden by `-f`. Concretely,
    // `gzip already.gz` where `already.gz` is mode 000 reports "Permission
    // denied" (exit 1), and where `already.gz` has a second hardlink reports
    // "has 1 other link -- file ignored" (exit 2) — in BOTH cases gzip never
    // reaches its "already has .gz suffix" no-op notice, even though the
    // filename triggers that check too. gzippy previously ran the
    // already-suffix check (and the hardlink check) before ever attempting
    // to open the file, so a mode-000 or hardlinked file already ending in
    // the target suffix short-circuited to the wrong (successful, no-op)
    // exit code instead of reporting the real precondition failure — a
    // compound-fixture divergence the drop-in census caught
    // (`fulcrum dropin`, fixture `already-named.gz` x {mode000, hardlink}).
    //
    // Opening the file here (rather than at its old, later position) is
    // what actually performs the permission check; the handle is reused
    // below instead of being opened a second time.
    let input_file = File::open(input_path)?;

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

    // Refuse to compress files that already carry the target suffix in-place
    // (e.g. foo.gz -> foo.gz.gz). Contract established by execution (gzip
    // 1.14, pigz 2.8 on macOS + Linux):
    //   - `-c`: there is no output-FILENAME collision to protect (output
    //     goes to stdout) so gzip ALWAYS compresses through regardless of
    //     this file's name, with or without -f. (pigz disagrees here — it
    //     skips and emits empty stdout — but gzip is the primary drop-in
    //     target, so we match gzip: skip this check entirely when
    //     `args.stdout`.)
    //   - in-place, without -f: gzip leaves the file COMPLETELY UNCHANGED
    //     and exits 0 -- a no-op NOTICE, not an error (both gzip and pigz
    //     agree: real `gzip already.gz` prints this message and exits 0).
    //   - in-place, with -f: `args.force` already short-circuits this
    //     branch (the condition below), so -f compresses anyway
    //     (foo.gz -> foo.gz.gz), matching gzip.
    if !args.stdout && !args.force && filename.ends_with(args.suffix.as_str()) {
        if !args.quiet {
            eprintln!(
                "gzippy: {}: already has {} suffix -- unchanged",
                filename, args.suffix
            );
        }
        return Ok(0);
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
                // Contract established by execution (gzip 1.14, non-interactive
                // stdin — e.g. a script or a pipeline): gzip does NOT treat this
                // as an error. It prints the exact same "already exists;\tnot
                // overwritten" shape it would after a declined prompt and exits
                // 2 (WARNING) -- it never actually blocks waiting for input, and
                // it never returns 1. gzippy previously returned an
                // InvalidArgument Err here (exit 1), which is wrong on two
                // counts: wrong exit CLASS, and it would abort a multi-file
                // invocation's remaining good files via the main-loop Err path
                // instead of just skipping this one. pigz disagrees (exit 1,
                // "skipping: F exists") -- gzip is the primary drop-in target.
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

    let file_size = input_file.metadata()?.len();

    let content_type = if args.processes <= 1 || args.compression_level <= 3 {
        ContentType::Binary
    } else {
        let mut sample_file = File::open(input_path)?;
        detect_content_type(&mut sample_file).unwrap_or(ContentType::Binary)
    };

    // (-i/--independent used to cap levels 7-9 to an L6 OptimizationConfig
    // here; -i is now rejected up front in run() — flag honesty, it never
    // delivered pigz's independence property — so the cap was dead code.)
    let opt_config = OptimizationConfig::new(
        args.processes,
        file_size,
        args.compression_level,
        content_type,
    );

    // -C/--comment: run() already rejects -C on the stdout path (minimal
    // header there, deliberately — see the tie cage). Both single-thread
    // FILE routes below splice the full FNAME/MTIME/FCOMMENT header via
    // `HeaderSpliceWriter` (issue #309), so the comment IS stored at every
    // thread count in file mode and the former -p1 refusal is gone.

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

    // Increment 7: the pure-Rust parallel DEFLATE encoder is the SOLE production
    // T>1 compress path — it produces a STANDARD single-member gzip stream for
    // every level 0–12 (and `--huffman` / `--rle`). Explicit zopfli tuning
    // (-F/-I/-J) still routes single-member through `compress_with_pipeline`;
    // `--rsyncable` keeps its content-defined split (also pure now). The former
    // mmap SimpleOptimizer (flate2/libdeflate C-FFI) path is gone from routing
    // and survives only behind the dev `ffi-oracle` feature as an oracle.
    let explicit_zopfli =
        args.zopfli_iterations.is_some() || args.zopfli_no_split || args.zopfli_split_max.is_some();
    let use_pure_parallel = use_mmap && !args.rsyncable && !explicit_zopfli;

    // T1 FILE inputs above the same threshold: mmap and parse whole-buffer
    // style over the map. The vendor's answer to input-side fault cost is
    // mmap, not streaming (libdeflate-gzip maps its input —
    // vendor/libdeflate/programs/gzip.c). There is no streaming T1 route
    // anymore (deleted 2026-08-30 — `ldx` is whole-buffer by construction),
    // so the mmap route's point is simply in-place parse with no input copy:
    // the stdin/pipe route reads to end into a Vec and parses the same way.
    // Output is byte-identical to the Read-based T1 route at every level.
    let use_t1_mmap = opt_config.thread_count == 1
        && file_size > 128 * 1024
        && !args.rsyncable
        && !explicit_zopfli;

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
    } else if use_pure_parallel {
        if args.verbosity >= 2 {
            eprintln!(
                "gzippy: using pure-Rust parallel DEFLATE encoder with {} threads",
                opt_config.thread_count,
            );
        }
        // SAFETY: the input file is opened read-only and mapped for the
        // duration of this compression; it is not mutated concurrently by
        // gzippy, matching every other mmap read path in this module.
        let mmap = unsafe { memmap2::Mmap::map(&File::open(input_path)?)? };
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);
        let mut encoder = crate::compress::pipelined::PipelinedGzEncoder::new(
            args.compression_level as u32,
            opt_config.thread_count,
        );
        // Honour `-b` ONLY when the user typed it — `block_size` carries a 128 KiB default,
        // so passing it unconditionally would change every existing T>1 output.
        encoder.set_block_size_override(args.block_size_explicit.then_some(args.block_size));
        encoder.set_header_info(header_info.clone());
        encoder.set_minimal_gzip_header(args.stdout);
        if args.stdout {
            let out = BufWriter::with_capacity(1024 * 1024, stdout());
            encoder
                .compress_buffer_pure(&mmap, out)
                .map_err(|e| e.into())
        } else {
            let output_file = BufWriter::new(File::create(output_path.as_ref().unwrap())?);
            encoder
                .compress_buffer_pure(&mmap, output_file)
                .map_err(|e| e.into())
        }
    } else if use_t1_mmap {
        if args.verbosity >= 2 {
            eprintln!(
                "gzippy: using pure-Rust DEFLATE encoder (T1 mmap L{})",
                args.compression_level
            );
        }
        // SAFETY: the input file is opened read-only and mapped for the
        // duration of this compression; it is not mutated concurrently by
        // gzippy, matching every other mmap read path in this module.
        let (mmap, logical_len) = map_with_tail_pad(&input_file, file_size as usize)?;
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);
        // Gate-4 route assertion at the call site of the encoder about to run
        // (both writer arms below invoke the same function unconditionally).
        crate::compress::route::emit(
            crate::compress::route::PURE_T1_MMAP,
            args.compression_level as u32,
            1,
        );
        let encode = |mut w: &mut dyn Write| -> GzippyResult<u64> {
            let bytes = crate::anatomy_wall_cli!({
                crate::compress::deflate::encode_gzip_unpadded_slice_to_writer(
                    &mmap,
                    logical_len,
                    &mut w,
                    args.compression_level as u32,
                )?
            });
            w.flush()?;
            Ok(bytes)
        };
        if args.stdout {
            let mut out = BufWriter::with_capacity(1024 * 1024, stdout());
            encode(&mut out)
        } else {
            // File output: gzip's contract stores FNAME + MTIME (issue #309).
            // The graded `-c` output above is untouched.
            let mut out = HeaderSpliceWriter::new(
                BufWriter::new(File::create(output_path.as_ref().unwrap())?),
                header_info.to_member_header(),
                crate::compress::deflate::minimal_gzip_header(args.compression_level as u32),
            );
            encode(&mut out)
        }
    } else if args.stdout {
        let out = BufWriter::with_capacity(1024 * 1024, stdout());
        crate::compress::compress_with_pipeline_sized(
            input_file,
            out,
            args,
            &opt_config,
            &header_info,
            Some(file_size as usize),
        )
    } else {
        let output_file = BufWriter::new(File::create(output_path.as_ref().unwrap())?);
        if opt_config.thread_count == 1 && !explicit_zopfli {
            // The T1 branch inside `compress_with_pipeline_sized` writes the
            // fixed minimal header; splice the full FNAME/MTIME header over
            // it (issue #309). The zopfli and T>1 branches build their own
            // full header from `header_info` and must NOT be wrapped.
            crate::compress::compress_with_pipeline_sized(
                input_file,
                HeaderSpliceWriter::new(
                    output_file,
                    header_info.to_member_header(),
                    crate::compress::deflate::minimal_gzip_header(args.compression_level as u32),
                ),
                args,
                &opt_config,
                &header_info,
                Some(file_size as usize),
            )
        } else {
            crate::compress::compress_with_pipeline_sized(
                input_file,
                output_file,
                args,
                &opt_config,
                &header_info,
                Some(file_size as usize),
            )
        }
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
                    print_stats(file_size, metadata.len(), input_path, &output_path, args);
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

    // Increment 7: the single-thread stdin fast path uses the pure-Rust DEFLATE
    // encoder (the sole production compress path). The former T1 L0–L3 ISA-L
    // streaming shortcut was C-FFI and has been removed from the routing graph;
    // it falls through to `compress_with_pipeline` (pure) below. ISA-L compress
    // survives only behind the dev `ffi-oracle` feature as a differential oracle.

    // Try to mmap stdin when it's a regular file (< file redirection).
    // For pipes, mmap_data stays None and we fall through to the
    // whole-buffer `compress_with_pipeline` route.
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

    let header_info = GzipHeaderInfo::default();
    let mut counted = CountingWriter::new(BufWriter::with_capacity(1024 * 1024, stdout()));

    let in_bytes = if let Some(ref mmap) = mmap_data {
        // Regular-file stdin (< file): multi-threaded parallel compression.
        let input_data = &mmap[..];
        let file_size = input_data.len() as u64;
        let content_type = if input_data.len() >= 8192 {
            crate::compress::optimization::analyze_content_type(&input_data[..8192])
        } else if !input_data.is_empty() {
            crate::compress::optimization::analyze_content_type(input_data)
        } else {
            ContentType::Binary
        };
        let opt_config = OptimizationConfig::new(
            args.processes,
            file_size,
            args.compression_level,
            content_type,
        );
        let compression_level = args.compression_level as u32;
        // Increment 7: the pure-Rust parallel DEFLATE encoder is the SOLE
        // production T>1 compress path (standard single-member gzip, every level
        // 0–12). Explicit zopfli tuning (-F/-I/-J) still routes single-member
        // through `compress_with_pipeline` so the zopfli encoder runs. The former
        // PipelinedGzEncoder / ParallelGzEncoder C-FFI split (flate2/libdeflate)
        // is retained only behind the dev `ffi-oracle` feature as an oracle.
        let explicit_zopfli = args.zopfli_iterations.is_some()
            || args.zopfli_no_split
            || args.zopfli_split_max.is_some();

        if opt_config.thread_count > 1 && !explicit_zopfli {
            let mut encoder = crate::compress::pipelined::PipelinedGzEncoder::new(
                compression_level,
                opt_config.thread_count,
            );
            encoder.set_block_size_override(args.block_size_explicit.then_some(args.block_size));
            encoder.set_header_info(header_info.clone());
            encoder.set_minimal_gzip_header(true);
            encoder.compress_buffer_pure(input_data, &mut counted)?;
            counted.flush()?;
            if verbose {
                print_stdin_stats(file_size, counted.count, args);
            }
            return Ok(0);
        }
        // Single-threaded (or explicit zopfli) with mmap'd file: stream through
        // compress_with_pipeline.
        let opt_config_t1 =
            OptimizationConfig::new(1, file_size, args.compression_level, content_type);
        crate::compress::compress_with_pipeline_sized(
            Cursor::new(input_data),
            &mut counted,
            args,
            &opt_config_t1,
            &header_info,
            Some(input_data.len()),
        )?
    } else {
        // Pipe stdin: whole-buffer (read-to-end into one Vec, then the T1
        // parse — `ldx` is whole-buffer by construction; the 2026-08-30
        // streaming deletion made buffering the honest contract). Single
        // worker, so it is the cheapest T1 route for unknown-length input.
        let opt_config = OptimizationConfig::new(1, 0, args.compression_level, ContentType::Binary);
        crate::compress::compress_with_pipeline(
            stdin(),
            &mut counted,
            args,
            &opt_config,
            &header_info,
        )?
    };

    counted.flush()?;
    if verbose {
        print_stdin_stats(in_bytes, counted.count, args);
    }
    Ok(0)
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
    let current_extension = output_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("");
    let new_extension = if current_extension.is_empty() {
        args.suffix.trim_start_matches('.').to_string()
    } else {
        format!("{}{}", current_extension, args.suffix)
    };
    output_path.set_extension(&new_extension);
    output_path
}

fn print_stats(
    input_size: u64,
    output_size: u64,
    input_path: &Path,
    output_path: &Path,
    args: &GzippyArgs,
) {
    let saved_pct = if input_size > 0 {
        (1.0_f64 - output_size as f64 / input_size as f64) * 100.0
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
        if args.processes > 1 {
            eprintln!(
                "{}: {:.1}{} → {:.1}{} ({:.1}% saved, {} threads)",
                name, in_sz, in_u, out_sz, out_u, saved_pct, args.processes
            );
        } else {
            eprintln!(
                "{}: {:.1}{} → {:.1}{} ({:.1}% saved)",
                name, in_sz, in_u, out_sz, out_u, saved_pct
            );
        }
    } else {
        // gzip-compatible format for -v: "path:   X.X% -- replaced with outpath"
        let in_name = input_path.to_str().unwrap_or("<unknown>");
        let out_name = output_path.to_str().unwrap_or("<unknown>");
        eprintln!(
            "{}:\t{:7.1}% -- replaced with {}",
            in_name,
            saved_pct.clamp(-99.9, 99.9),
            out_name
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

#[cfg(test)]
mod map_with_tail_pad_tests {
    use super::*;
    use std::io::Write;

    /// The fast path in `encode_gzip_unpadded_slice_to_writer` is only correct
    /// if the bytes past EOF in the final partial page are readable AND ZERO.
    /// That is a kernel guarantee, not a gzippy one, so pin it: a release build
    /// compiles the `debug_assert` away, and a non-zero pad would silently
    /// corrupt the tail of every compressed file.
    #[test]
    fn extended_map_slack_is_readable_and_zero() {
        let page = page_size();
        assert!(page > 0, "test needs a real page size");
        let dir = tempfile::tempdir().expect("tempdir");
        // Sizes chosen around the page boundary, where the branch decides.
        for &len in &[
            1usize,
            17,
            page - 17,
            page - 16,
            page - 15,
            page - 1,
            page,
            page + 1,
            3 * page - 16,
            3 * page,
            3 * page + 7,
        ] {
            let path = dir.path().join(format!("f{len}"));
            let mut f = std::fs::File::create(&path).expect("create");
            // Non-zero content, so a pad read as zero cannot be content bleed.
            f.write_all(&vec![0xABu8; len]).expect("write");
            f.sync_all().expect("sync");
            drop(f);

            let f = std::fs::File::open(&path).expect("open");
            let (m, logical) = map_with_tail_pad(&f, len).expect("map");
            assert_eq!(logical, len, "logical_len must be the true file length");
            assert_eq!(
                &m[..len],
                &vec![0xABu8; len][..],
                "content changed, len={len}"
            );

            let extended = m.len() > len;
            assert!(
                m.len() == len || m.len() == len + crate::compress::deflate::INPLACE_TAIL_PAD,
                "mapping is either exact or exactly padded, got {} for len={len}",
                m.len()
            );
            if extended {
                assert!(
                    m[len..].iter().all(|&b| b == 0),
                    "slack past EOF must read as zero, len={len}"
                );
            }
            // A file ending exactly on a page boundary has no partial page to
            // borrow from, so it MUST fall back rather than map a page it does
            // not own.
            if len.is_multiple_of(page) {
                assert!(
                    !extended,
                    "must not extend past a full final page, len={len}"
                );
            }
        }
    }
}
