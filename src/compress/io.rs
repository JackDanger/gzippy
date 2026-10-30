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

/// Map `input_file`, handing the encoder TAIL_PAD readable zero bytes past `len`
/// so the encoder can parse IN PLACE instead of copying the whole input.
///
/// Returns `(mapping, logical_len)`. On unix every mapping here carries at
/// least `len + [`INPLACE_TAIL_PAD`]` readable bytes; a mapping of exactly `len`
/// (no pad) survives only where neither route below could mmap at all.
///
/// WHY. The padded encoder requires TAIL_PAD trailing bytes so the matchfinder's
/// speculative 4/8-byte loads stay in bounds. Without slack, the only way to
/// supply them used to be to copy the entire input — a 51 MB memcpy on
/// monorepo.tar to append 16 bytes. Measured 2026-08-21: explicit allocations
/// ran at exactly 1.50x the input on every corpus file regardless of
/// compressibility, and peak RSS at 2.5-2.7x, against pigz's flat ~2 MB — the
/// 1.0x share was this copy, and it fired for every input whose length is an
/// exact multiple of the page size (~1/4096 of arbitrary files) or whose final
/// partial page left < TAIL_PAD slack.
///
/// THE THREE ROUTES, best first:
///
/// 1. `len % page != 0 && slack >= TAIL_PAD` — the file's final partial page
///    already owns `slack` readable bytes past EOF (the kernel zero-fills the
///    tail of a partial page): map `len + TAIL_PAD` and those bytes ARE the
///    pad. ~99.6% of arbitrary files.
/// 2. THE overlay (this is where the whole-input copy used to live): `len %
///    page == 0` (no partial page to borrow from) or `slack < TAIL_PAD` (the
///    pad would cross into a page the file does not own), plus any route-1
///    attempt the kernel refused. Anonymous PROT_READ pages cover the whole
///    reserved range first, then MAP_FIXED installs a private file mapping of
///    exactly `len` over the head: file bytes stay file-backed, the pad stays
///    fresh anonymous zero.
/// 3. plain exact-length map — only when mmap failed everywhere above (e.g.
///    non-unix targets with no usable page size, or a kernel that refused
///    every attempt); `encode_gzip_unpadded_slice_to_writer` then pays the one
///    whole-input copy. Never reached for regular unix files with instant mmap.
///
/// HOW IT IS SAFE. Route 1: a file's last page is partial whenever
/// `len % page != 0`, and the kernel zero-fills the remainder of that page;
/// reading it is defined, and mapping into it does not fault — only a page
/// BEYOND the file's last one raises SIGBUS, so route 1 only asks for slack
/// inside that partial page. Route 2's file mapping is exactly `len` bytes
/// (never a page the file does not own); for the partial-page shapes the
/// kernel still rounds the length up to the file's own page boundary, where
/// reads past EOF stay zero-filled — the same guarantee route 1 is built on.
/// Everything past the file's last page is anonymous fresh pages, which read
/// as zero before any write.
fn map_with_tail_pad(file: &File, len: usize) -> std::io::Result<(InputMap, usize)> {
    const TAIL_PAD: usize = crate::compress::deflate::INPLACE_TAIL_PAD;
    let page = page_size();
    if page == 0 || len == 0 {
        // Mainframe case (non-unix: no page size knowledge) or the empty file
        // (nothing to pad past): the plain map and the encode-site copy.
        let m = unsafe { memmap2::Mmap::map(file)? };
        return Ok((m.into(), len));
    }
    let slack_in_last_page = page - (len % page);
    // `len % page == 0` gives slack_in_last_page == page, but there is no
    // partial page then — the next byte is in a page the file does not own;
    // route 2 (the overlay) covers it instead.
    let can_extend = !len.is_multiple_of(page) && slack_in_last_page >= TAIL_PAD;
    if can_extend {
        // SAFETY: same contract as the plain `Mmap::map` below — the file is
        // opened read-only and not mutated concurrently by gzippy. The extra
        // TAIL_PAD bytes lie inside the file's final partial page, which the
        // kernel zero-fills, so they are readable and read as zero.
        let extended = unsafe { memmap2::MmapOptions::new().len(len + TAIL_PAD).map(file) };
        if let Ok(m) = extended {
            return Ok((m.into(), len));
        }
        // The kernel refused the longer mapping: fall through to the overlay,
        // which pads the tail anonymously and works wherever the plain map can.
    }
    let reserved_len = (len + TAIL_PAD).div_ceil(page) * page;
    if let Some(m) = map_with_tail_pad_overlay(file, len, reserved_len) {
        return Ok((m, len));
    }
    // Both pad routes failed to mmap (kernel refused every attempt). The
    // encoder at the call site pays the one whole-input copy; the diagnostic
    // mirrors `route::emit`'s GZIPPY_DEBUG gate so a fallback run is visible.
    if crate::utils::debug_enabled() {
        eprintln!("[gzippy] tail-pad overlay failed; falling back to the whole-input copy path");
    }
    // SAFETY: `file` is opened read-only and is not mutated concurrently by
    // gzippy; the mapping is never written.
    let m = unsafe { memmap2::Mmap::map(file)? };
    Ok((m.into(), len))
}

/// The read-only `[u8]` window the T1 mmap route parses over. Either a plain
/// `memmap2::Mmap`, or the two-mapping overlay of `map_with_tail_pad`'s route 2
/// (anonymous pad tail under a MAP_FIXED file head), which memmap2 cannot
/// express by itself.
///
/// The overlay's raw pointer exists only because the joined range spans two
/// kernel VMAs; memory owns it for exactly this struct's lifetime (one munmap
/// in Drop, after which the `[u8]` handed out by Deref is dead — callers (the
/// encoder) never outlive the value on any in-crate path.
pub(crate) struct InputMap {
    /// The MAPPED byte count. Either exactly the file length (route 3, no pad
    /// obtainable), `len + TAIL_PAD` (route 1), or the page-rounded route-2
    /// reservation (>= `len + TAIL_PAD`).
    len: usize,
    kind: InputMapKind,
}

enum InputMapKind {
    Simple(memmap2::Mmap),
    #[cfg(unix)]
    Overlay {
        /// Start of the joined (file head + anonymous tail) reservation.
        base: *const u8,
    },
}

impl From<memmap2::Mmap> for InputMap {
    fn from(m: memmap2::Mmap) -> Self {
        Self {
            len: m.len(),
            kind: InputMapKind::Simple(m),
        }
    }
}

impl std::ops::Deref for InputMap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match &self.kind {
            InputMapKind::Simple(m) => m,
            #[cfg(unix)]
            InputMapKind::Overlay { base } => unsafe {
                // SAFETY: `[base, base + len)` is fully installed, read-only,
                // and lives until this value's Drop; every borrow handed out
                // through Deref is shorter than that lifetime on any in-crate
                // path (the encoder borrows it for one synchronous call).
                std::slice::from_raw_parts(*base, self.len)
            },
        }
    }
}

impl InputMap {
    /// Configure the kernel's readahead for the whole window. Errors are
    /// dropped, matching the `let _ = mmap.advise(..)` at every call site.
    pub(crate) fn advise_sequential(&self) {
        match &self.kind {
            InputMapKind::Simple(m) => {
                let _ = m.advise(memmap2::Advice::Sequential);
            }
            #[cfg(unix)]
            InputMapKind::Overlay { base } => {
                // SAFETY: `[base, base + len)` is a live, readable mapping;
                // madvise only annotates it.
                let _ = unsafe {
                    libc::madvise(
                        base.cast_mut().cast::<libc::c_void>(),
                        self.len,
                        libc::MADV_SEQUENTIAL,
                    )
                };
            }
        }
    }
}

impl Drop for InputMap {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let InputMapKind::Overlay { base } = &self.kind {
            // SAFETY: exactly one munmap, at owner death, over the whole
            // reservation this type created (it removes BOTH the file mapping
            // and the anonymous tail; the `Simple` arm is freed by its own
            // `Mmap` drop below).
            unsafe { libc::munmap(base.cast_mut().cast::<libc::c_void>(), self.len) };
        }
    }
}

/// Route 2 of `map_with_tail_pad`: one address range whose head is a private
/// file mapping of exactly `len` bytes and whose tail reads as fresh zero
/// pages. Returns None (keeping `map_with_tail_pad`'s plain-map fallback) if
/// any kernel call fails — correctness never depends on this route.
#[cfg(unix)]
fn map_with_tail_pad_overlay(file: &File, len: usize, reserved_len: usize) -> Option<InputMap> {
    use std::os::unix::io::AsRawFd;

    // SAFETY: kernel-chosen address; on failure the result is exactly
    // MAP_FAILED (checked below), never an errno-as-pointer.
    let base = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            reserved_len,
            libc::PROT_READ,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        return None;
    }
    let base: *const u8 = base.cast();

    // SAFETY: `base` is this thread's freshly mapped reservation — the ONLY
    // pages MAP_FIXED consumes are pages this function created two lines
    // above; no other thread can hold a reference to them. The file mapping
    // (fd of the read-only input) then covers `[base, base + len)`, rounded
    // up by the kernel to the file's own page boundary, where reads past EOF
    // stay zero-filled.
    let head = unsafe {
        libc::mmap(
            base.cast_mut().cast::<libc::c_void>(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE | libc::MAP_FIXED,
            file.as_raw_fd(),
            0,
        )
    };
    if head.is_null() || head != base.cast_mut().cast::<libc::c_void>() {
        // MAP_FIXED failed atomically: the anonymous reservation is intact.
        // SAFETY: the range is unmapped exactly once, right here.
        unsafe { libc::munmap(base.cast_mut().cast::<libc::c_void>(), reserved_len) };
        return None;
    }
    Some(InputMap {
        len: reserved_len,
        kind: InputMapKind::Overlay { base },
    })
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
            crate::compress::rsyncable::compress_rsyncable(
                &mmap,
                args.compression_level as u32,
                opt_config.thread_count,
                &header_info,
                stdout(),
            )
            .map_err(|e| e.into())
        } else {
            let output_file = BufWriter::new(File::create(output_path.as_ref().unwrap())?);
            crate::compress::rsyncable::compress_rsyncable(
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
        mmap.advise_sequential();
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
    use crate::compress::deflate::encode_gzip_unpadded_slice_to_writer;
    use std::io::Write;

    /// Every route a regular unix file can take must hand the encoder its
    /// TAIL_PAD pad window past EOF, and the pad must read as ZERO — the fast
    /// path in `encode_gzip_unpadded_slice_to_writer` is only correct if the
    /// bytes past EOF are readable AND zero. Zero-fill is a kernel promise
    /// (the tail of a partial final page, and fresh anonymous pages), not a
    /// gzippy one, so pin it: a release build compiles the `debug_assert`
    /// away, and a non-zero pad would silently corrupt the tail of every
    /// compressed file.
    ///
    /// The exact-multiple sizes (`page`, `3 * page`) and the thin-slack band
    /// (`page - 15 .. page - 1`) are the sizes that keep the whole-input-copy
    /// fallback alive (route 3): if `map_with_tail_pad`'s overlay ever stops
    /// covering them (mmap refusal, regression), the plain map lands here with
    /// `m.len() == len` and both assertions below go RED with it.
    #[test]
    fn extended_map_slack_is_readable_and_zero() {
        let page = page_size();
        assert!(page > 0, "test needs a real page size");
        let tail_pad = crate::compress::deflate::INPLACE_TAIL_PAD;
        let dir = tempfile::tempdir().expect("tempdir");
        // Sizes chosen around the page boundary, where the route decides:
        // route 1 (slack >= tail_pad), the whole thin-slack band (slack
        // 1..15: slack -1 is `page`, slack.. every reduction), and the
        // exact-multiple case where there is no partial page at all.
        let mut sizes = vec![
            1usize,
            17,
            page - 17,
            page - 16,
            page,
            page + 1,
            3 * page - 16,
            3 * page,
            3 * page + 7,
            3 * page + 8,
        ];
        // Every thin-slack shape in [page - tail_pad + 1, page): the pad
        // window would cross into a page the file does not own.
        sizes.extend((1..tail_pad).map(|s| page - s));
        sizes.sort_unstable();
        sizes.dedup();
        for &len in &sizes {
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
            // Every in-place route: the mapping carries the TAIL_PAD window
            // and it reads as zero. The only shape allowed to miss this is
            // the route-3 plain map — which, when reached, means BOTH pad
            // arrangements above failed and the encode site pays the copy.
            assert!(
                m.len() >= len + tail_pad,
                "mapping must carry the {}-byte pad window past EOF, got {} \
                 readable bytes for len={len} (route-2 overlay failed and \
                 route 3 ran — the whole-input copy is live again)",
                tail_pad,
                m.len()
            );
            assert!(
                m[len..].iter().all(|&b| b == 0),
                "pad past EOF must read as zero, len={len} (mapping {} bytes)",
                m.len()
            );
            // The mapping never extends beyond a page past the pad; anything
            // larger is a kernel-ABI surprise to catch here rather than
            // discover as an ASan reports on the encoder's read window.
            assert!(
                m.len() <= (len + tail_pad).div_ceil(page) * page,
                "mapping {} bytes is past the pad's page for len={len}",
                m.len()
            );
        }
    }

    /// Whatever pad route a file takes, the ENCODED BYTES must be identical:
    /// the in-place parse over a mapping whose pad is anonymous zero (route 2,
    /// incl. the exact-page-multiple and thin-slack shapes) must equal the
    /// historical whole-input copy the exact-multiple files used to pay for —
    /// same parser state, pad filling zero either way. Guards the T1
    /// tie-cage contract (output identity) against a pad that is readable but
    /// not zero, or a logical length that drifted.
    #[test]
    fn overlay_route_output_equals_the_copy_path_output_bytes() {
        let page = page_size();
        assert!(page > 0, "test needs a real page size");
        let tail_pad = crate::compress::deflate::INPLACE_TAIL_PAD;
        let dir = tempfile::tempdir().expect("tempdir");
        // The overlay shapes: exact multiples (page, 3 * page) and the whole
        // thin-slack band. 3 * page spans several parser blocks so the tail
        // byte state is exercised across block boundaries, not just once.
        for &len in &[1usize, page, 3 * page, 3 * page - tail_pad, 3 * page - 1] {
            let path = dir.path().join(format!("id{len}"));
            let content = {
                // Deterministic, structured-enough data (literals + matches),
                // non-zero, and NOT page-indifferent: any pad bleed would
                // change the bytes.
                let mut v = vec![0u8; len];
                for (i, b) in v.iter_mut().enumerate() {
                    let s = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
                    *b = if (i / 4096) % 2 == 0 {
                        (s & 0xFF) as u8
                    } else {
                        b'a' + (s % 26) as u8
                    };
                }
                v
            };
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(&content).expect("write");
            f.sync_all().expect("sync");
            drop(f);
            let f = std::fs::File::open(&path).expect("open");

            // (a) the in-place route: parse straight over the padded mapping.
            let (m, logical) = map_with_tail_pad(&f, len).expect("map");
            let mut src_a = Vec::new();
            encode_gzip_unpadded_slice_to_writer(&m, logical, &mut src_a, 6)
                .expect("encode over the mapping");

            // (b) the historical copy path: whole input + zero pad, logical
            // length unchanged. References the SAME encoder one level under.
            let mut copied = content.clone();
            copied.resize(len + tail_pad, 0);
            let mut src_b = Vec::new();
            encode_gzip_unpadded_slice_to_writer(&copied, len, &mut src_b, 6)
                .expect("encode over the copy");

            assert_eq!(
                src_a, src_b,
                "in-place (mapping) bytes differ from the copy path, len={len}"
            );
        }
    }
}
