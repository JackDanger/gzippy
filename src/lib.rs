//! gzippy — embed the world's fastest gzip in your Rust program.
//!
//! Every function routes through the same backend-selection logic as the
//! gzippy CLI, so you automatically get ISA-L SIMD, libdeflate, parallel
//! multi-block, Zopfli, and multi-member decompression without any extra
//! configuration.
//!
//! # Quick start
//!
//! ```rust
//! let data = b"hello, world!".repeat(1000);
//! let compressed = gzippy::compress(&data, 6).unwrap();
//! let decompressed = gzippy::decompress(&compressed).unwrap();
//! assert_eq!(decompressed, data);
//! ```
//!
//! # Choosing a compression level
//!
//! | Level | Backend | Notes |
//! |-------|---------|-------|
//! | 0     | Store (no compression) | |
//! | 1–3   | ISA-L SIMD (x86_64), libdeflate/zlib-ng (other) | fastest |
//! | 4–5   | libdeflate one-shot | balanced |
//! | 6     | zlib-ng streaming | gzip default |
//! | 7–9   | zlib-ng streaming | high ratio |
//! | 10,12 | libdeflate ultra | near-zopfli ratio |
//! | 11    | Zopfli | best ratio, very slow |
//!
//! # Threading and output format
//!
//! `threads = 1` always produces a **standard single-member gzip** stream
//! decompressible by any tool.
//!
//! `threads > 1` behaviour depends on level:
//! - **L0–5**: [`ParallelGzEncoder`] produces a gzippy "GZ" multi-block
//!   stream. This is **not** decompressible by standard tools (gunzip, pigz,
//!   etc.) — only by gzippy itself (CLI or this library). Use it when both
//!   ends of the pipe run gzippy.
//! - **L6–9**: [`PipelinedGzEncoder`] produces a standard single-member
//!   stream that any tool can decompress.
//!
//! [`ParallelGzEncoder`]: compress::parallel::ParallelGzEncoder
//! [`PipelinedGzEncoder`]: compress::pipelined::PipelinedGzEncoder
//!
//! # Decompression
//!
//! The decompressor handles all gzip variants automatically:
//! - gzippy "GZ" multi-block streams (parallel bgzf path)
//! - Standard multi-member streams (e.g. `cat a.gz b.gz`)
//! - Single-member streams (standard gzip output)
//!
//! **Non-gzip input:** if `data` does not begin with the gzip magic bytes
//! (`0x1f 0x8b`), every decompress function returns `Ok(empty)` rather than
//! an error — consistent with CLI sniffing behavior.

// ── Shared infrastructure (same module tree as the binary) ───────────────────
mod backends;
mod cli;
mod format;
mod infra;
mod utils;

#[cfg(test)]
mod tests;

// `compress::io` and `decompress::io` call `crate::set_output_file` to register
// the in-progress output path for signal-handler cleanup. In the library there
// is no signal handler, so this is a no-op.
#[doc(hidden)]
pub fn set_output_file(_path: Option<String>) {}

// ── Engine modules ────────────────────────────────────────────────────────────
// `#[doc(hidden)]` marks these as internal: rustdoc will not render them,
// and the public contract is the six top-level functions + three types below.
// Items under `compress::` / `decompress::` are not covered by semver.
#[doc(hidden)]
pub mod compress;
#[doc(hidden)]
pub mod decompress;
#[doc(hidden)]
pub mod error;

// ── Stable public surface ─────────────────────────────────────────────────────
pub use decompress::DecodePath;
pub use error::{GzippyError, GzippyResult};

// =============================================================================
// Compression API
// =============================================================================

/// Compress `data` to gzip format at `level` using all available CPUs.
///
/// `level` is clamped to `0..=12` (see the [level table](crate#choosing-a-compression-level)).
///
/// When all CPUs are used and `level` is in 0–5, the output uses gzippy's
/// parallel "GZ" multi-block format. Use [`compress_with_threads`]`(data, level, 1)`
/// if you need standard gzip output that any tool can read.
pub fn compress(data: &[u8], level: u8) -> GzippyResult<Vec<u8>> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    compress_with_threads(data, level, threads)
}

/// Compress `data` to gzip format at `level` using exactly `threads` threads.
///
/// **`threads = 1`** — standard single-member gzip; decompressible by any tool.
///
/// **`threads > 1, level 0–5`** — gzippy "GZ" multi-block format; only
/// decompressible by gzippy (CLI or this library). Faster for large inputs.
///
/// **`threads > 1, level 6–9`** — pipelined single-member gzip; any tool
/// can decompress.
pub fn compress_with_threads(data: &[u8], level: u8, threads: usize) -> GzippyResult<Vec<u8>> {
    let mut out = Vec::new();
    compress::compress_bytes(std::io::Cursor::new(data), &mut out, level, threads)?;
    Ok(out)
}

/// Compress data from `reader` into `writer` at `level` using all available CPUs.
///
/// Suitable for large inputs you don't want to buffer entirely in memory. The
/// same threading and format rules as [`compress`] apply.
///
/// Returns the number of **uncompressed** bytes consumed from `reader`.
pub fn compress_to_writer<R: std::io::Read, W: std::io::Write + Send>(
    reader: R,
    writer: W,
    level: u8,
) -> GzippyResult<u64> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    compress_to_writer_with_threads(reader, writer, level, threads)
}

/// Compress data from `reader` into `writer` at `level` with explicit thread count.
///
/// The same threading and format rules as [`compress_with_threads`] apply.
///
/// Returns the number of **uncompressed** bytes consumed from `reader`.
pub fn compress_to_writer_with_threads<R: std::io::Read, W: std::io::Write + Send>(
    reader: R,
    writer: W,
    level: u8,
    threads: usize,
) -> GzippyResult<u64> {
    compress::compress_bytes(reader, writer, level, threads)
}

// =============================================================================
// Decompression API
// =============================================================================

/// Decompress a gzip stream using all available CPUs.
///
/// Automatically selects the best path — parallel bgzf, parallel
/// multi-member, ISA-L single-member, or libdeflate one-shot — based on
/// the input format and available hardware.
///
/// **Non-gzip input:** returns `Ok(Vec::new())`.
pub fn decompress(data: &[u8]) -> GzippyResult<Vec<u8>> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    decompress_with_threads(data, threads)
}

/// Decompress a gzip stream with explicit thread count.
///
/// Set `threads = 1` for deterministic single-threaded decompression (useful
/// in constrained or benchmark contexts).
///
/// **Non-gzip input:** returns `Ok(Vec::new())`.
pub fn decompress_with_threads(data: &[u8], threads: usize) -> GzippyResult<Vec<u8>> {
    let mut out = Vec::new();
    decompress::decompress_bytes(data, &mut out, threads)?;
    Ok(out)
}

/// Decompress a gzip stream into `writer` using all available CPUs.
///
/// Useful when streaming output to a file or network socket without an
/// intermediate allocation. For explicit thread control use
/// [`decompress_to_writer_with_threads`].
///
/// Returns the number of decompressed bytes written.
///
/// **Non-gzip input:** writes nothing and returns `Ok(0)`.
pub fn decompress_to_writer<W: std::io::Write + Send>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    decompress::decompress_bytes(data, writer, threads)
}

/// Decompress a gzip stream into `writer` with explicit thread count.
///
/// Mirrors [`decompress_with_threads`] for the writer API.
///
/// Returns the number of decompressed bytes written.
///
/// **Non-gzip input:** writes nothing and returns `Ok(0)`.
pub fn decompress_to_writer_with_threads<W: std::io::Write + Send>(
    data: &[u8],
    writer: &mut W,
    threads: usize,
) -> GzippyResult<u64> {
    decompress::decompress_bytes(data, writer, threads)
}

// =============================================================================
// Raw DEFLATE API (no gzip framing)
// =============================================================================

/// Compress `data` to raw DEFLATE (RFC 1951) at `level` — no gzip header or trailer.
///
/// `level` is clamped to `0..=12`. Uses the same backend hierarchy as [`compress`]:
/// ISA-L SIMD on x86_64 for levels 0–3, then libdeflate one-shot for all levels.
///
/// Use this when the framing (CRC32, size) is handled by the caller, for example
/// when embedding deflate streams in ZIP, 7z, or zlib containers.
pub fn compress_raw(data: &[u8], level: u8) -> GzippyResult<Vec<u8>> {
    compress::compress_raw_bytes(data, level)
}

/// Decompress a raw DEFLATE stream (RFC 1951) — no gzip header or trailer expected.
///
/// Uses libdeflate for speed, growing the output buffer as needed. Falls back to
/// a flate2/zlib-ng streaming decoder if the output exceeds 1 GiB.
///
/// Returns an error if `data` is not valid DEFLATE.
pub fn decompress_raw(data: &[u8]) -> GzippyResult<Vec<u8>> {
    decompress::decompress_raw_bytes(data)
}

/// Alias for [`compress_raw`] — used by 7zippy's Deflate coder.
pub use self::compress_raw as deflate_encode;

/// Alias for [`decompress_raw`].
pub use self::decompress_raw as deflate_decode;

// =============================================================================
// Routing inspection
// =============================================================================

/// Return the [`DecodePath`] gzippy would choose for `data` with `threads`.
///
/// Useful for tests and diagnostics. Does not allocate or decompress.
pub fn classify(data: &[u8], threads: usize) -> DecodePath {
    decompress::classify_gzip(data, threads)
}
