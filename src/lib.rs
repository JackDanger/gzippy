//! gzippy — embed the world's fastest gzip in your Rust program.
//!
//! Thin, zero-overhead wrappers over gzippy's actual compression and
//! decompression engines. Every function in this crate routes through the
//! same backend selection logic as the CLI, so you get ISA-L, libdeflate,
//! parallel multi-block, and Zopfli paths automatically.
//!
//! # Quick start
//!
//! ```rust
//! // Round-trip at level 6 using all available CPUs.
//! let data = b"hello, world!".repeat(1000);
//! let compressed = gzippy::compress(&data, 6).unwrap();
//! let decompressed = gzippy::decompress(&compressed).unwrap();
//! assert_eq!(decompressed, data);
//! ```
//!
//! # Compression levels
//!
//! | Level | Backend |
//! |-------|---------|
//! | 0     | Store (no compression) |
//! | 1–3   | ISA-L SIMD (x86_64), libdeflate/zlib-ng (other) |
//! | 4–5   | libdeflate one-shot |
//! | 6–9   | zlib-ng streaming |
//! | 10,12 | libdeflate ultra (near-zopfli ratio, reasonable speed) |
//! | 11    | Zopfli (maximum ratio, very slow) |
//!
//! Multi-threading (`compress_with_threads` with `threads > 1`) routes
//! L0–5 through `ParallelGzEncoder` (gzippy "GZ" multi-block format) and
//! L6–9 through `PipelinedGzEncoder` (standard single-member gzip).

// ── Shared infrastructure (same module tree as the binary) ───────────────────
mod backends;
mod cli;
mod format;
mod infra;
mod utils;

#[cfg(test)]
mod tests;

// `compress::io` and `decompress::io` call `crate::set_output_file` to track
// the in-progress output path for signal-handler cleanup.  In the library
// there is no signal handler, so this is a no-op.
pub fn set_output_file(_path: Option<String>) {}

// ── Public engine modules ─────────────────────────────────────────────────────
pub mod compress;
pub mod decompress;
pub mod error;

// ── Re-exports ────────────────────────────────────────────────────────────────
pub use decompress::{classify_gzip, DecodePath};
pub use error::{GzippyError, GzippyResult};

// =============================================================================
// Top-level convenience API
// =============================================================================

/// Compress `data` to gzip format at `level` using all available CPUs.
///
/// `level` is clamped to `0..=12`. Pass `11` for Zopfli.
/// Output is a valid gzip stream decompressible by any gzip tool.
pub fn compress(data: &[u8], level: u8) -> GzippyResult<Vec<u8>> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    compress_with_threads(data, level, threads)
}

/// Compress `data` to gzip format at `level` using exactly `threads` threads.
///
/// Set `threads = 1` to force single-threaded output (standard gzip member).
/// Set `threads > 1` to enable gzippy's parallel multi-block format for
/// levels 0–5, or pipelined single-member for levels 6–9.
pub fn compress_with_threads(data: &[u8], level: u8, threads: usize) -> GzippyResult<Vec<u8>> {
    let mut out = Vec::new();
    compress::compress_bytes(std::io::Cursor::new(data), &mut out, level, threads)?;
    Ok(out)
}

/// Decompress a gzip stream using all available CPUs.
///
/// Automatically selects the best path: parallel bgzf, parallel multi-member,
/// ISA-L single-member, or libdeflate one-shot.
///
/// **Non-gzip input:** if `data` does not start with the gzip magic bytes
/// (`0x1f 0x8b`), returns `Ok(Vec::new())` rather than an error — consistent
/// with CLI behavior when sniffing stdin. Check the return length if you need
/// to distinguish "empty gzip" from "not gzip".
pub fn decompress(data: &[u8]) -> GzippyResult<Vec<u8>> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    decompress_with_threads(data, threads)
}

/// Decompress a gzip stream with explicit thread count.
///
/// Set `threads = 1` for deterministic single-threaded output (useful in
/// constrained or benchmark contexts).
///
/// **Non-gzip input:** returns `Ok(Vec::new())` — see [`decompress`].
pub fn decompress_with_threads(data: &[u8], threads: usize) -> GzippyResult<Vec<u8>> {
    let mut out = Vec::new();
    decompress::decompress_bytes(data, &mut out, threads)?;
    Ok(out)
}

/// Decompress a gzip stream to an arbitrary writer using all available CPUs.
///
/// Useful when you want to stream output directly to a file or network socket
/// without an intermediate allocation. For explicit thread control, use
/// [`decompress_to_writer_with_threads`].
pub fn decompress_to_writer<W: std::io::Write + Send>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    decompress::decompress_bytes(data, writer, threads)
}

/// Decompress a gzip stream to an arbitrary writer with explicit thread count.
///
/// Mirrors [`decompress_with_threads`] for the writer API: set `threads = 1`
/// for deterministic single-threaded decompression.
pub fn decompress_to_writer_with_threads<W: std::io::Write + Send>(
    data: &[u8],
    writer: &mut W,
    threads: usize,
) -> GzippyResult<u64> {
    decompress::decompress_bytes(data, writer, threads)
}

/// Return the [`DecodePath`] gzippy would choose for this input and thread count.
///
/// Useful for tests and diagnostics — lets callers assert that specific data
/// takes the expected backend without actually decompressing.
pub fn classify(data: &[u8], threads: usize) -> DecodePath {
    decompress::classify_gzip(data, threads)
}
