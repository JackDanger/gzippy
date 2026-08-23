//! gzippy — embed the world's fastest gzip in your Rust program.
//!
//! Every function routes through the same backend-selection logic as the
//! gzippy CLI: a single pure-Rust DEFLATE engine serves every level and
//! thread count, with no C-FFI compressor on the routing graph.
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
//! | Level | Notes |
//! |-------|-------|
//! | 0     | Store (no compression) |
//! | 1–9   | Pure-Rust DEFLATE engine (hash-chain / lazy parse); ratio tracks libdeflate |
//! | 10,12 | Pure-Rust DEFLATE engine, near-optimal parse — near-zopfli ratio |
//! | 11    | Same near-optimal parse as 10/12 — level 11 alone does **not** invoke the much slower Zopfli "crown" engine, which this library API does not currently expose (CLI-only, via `-F`/`-I`/`-J`) |
//!
//! # Threading and output format
//!
//! Every level and thread count produces a **standard single-member gzip**
//! stream, decompressible by any tool:
//! - **`threads = 1`**: [`compress::deflate::encode_gzip_slack_padded_to_vec`] (the T1
//!   entry point).
//! - **`threads > 1`**: [`PipelinedGzEncoder`] (`compress_buffer_pure`) —
//!   a pure parallel encoder whose output is byte-identical across thread
//!   counts.
//!
//! gzippy's own "GZ" multi-block format ([`ParallelGzEncoder`]) still
//! exists and is still decompressible (see below), but it is not reachable
//! from this crate's compress routing table — it is retained only as a
//! differential test oracle behind the `ffi-oracle` feature.
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
// Doc-hidden re-export so the `engine_isolation` bench can reach the ISA-L
// from-bit FFI oracle (`backends::isal_decompress::decompress_deflate_from_bit`).
// Measurement-only surface; does NOT touch the decode routing graph.
#[doc(hidden)]
pub use backends::isal_decompress as isal_decompress_oracle;
mod cli;
mod format;
#[doc(hidden)]
pub mod infra;
mod utils;

#[cfg(test)]
mod tests;

// Lib-only oracle corpus test. See note in
// `src/compress/deflate/parse/ultra/mod.rs` for why this is declared here
// rather than under the `mod compress;` tree.
#[cfg(all(test, feature = "oracle"))]
#[path = "compress/deflate/parse/ultra/oracle_tests.rs"]
mod zopfli_oracle_tests;

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
pub mod coz_probe;
#[doc(hidden)]
pub mod decompress;
#[doc(hidden)]
pub mod error;
pub mod fixtures;
pub mod holdout;

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
/// Output is **standard single-member gzip at every level and thread count** — any
/// tool reads it.
///
/// ⭐ This used to claim that all-CPUs + level 0–5 produced a gzippy-only "GZ"
/// multi-block format. That has not been true for some time: verified 2026-08-23 by
/// decompressing L0–L5 at `-p4` with stock `gzip -dc` (all six round-trip, one member
/// each). Owner review flagged it as "stale and internally contradictory".
pub fn compress(data: &[u8], level: u8) -> GzippyResult<Vec<u8>> {
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    compress_with_threads(data, level, threads)
}

/// Compress `data` to gzip format at `level` using up to `threads` compression
/// workers.
///
/// Output is **standard single-member gzip at every level and thread count**.
///
/// # `threads` is a worker count, not a total
///
/// It is clamped to the available parallelism, and the parallel path additionally
/// runs a dedicated writer thread, so the process may use `threads + 1`. This
/// previously read "exactly `threads` threads", which owner review flagged as
/// untrue — it "oversubscribes a machine already configured with N CPUs".
///
/// # `threads > 1` buffers the whole input
///
/// The parallel encoder is whole-buffer. See [`compress_to_writer`] for the
/// streaming (single-threaded) path.
pub fn compress_with_threads(data: &[u8], level: u8, threads: usize) -> GzippyResult<Vec<u8>> {
    let mut out = Vec::new();
    compress::compress_bytes(std::io::Cursor::new(data), &mut out, level, threads)?;
    Ok(out)
}

/// Compress data from `reader` into `writer` at `level`, **genuinely streaming**:
/// output begins before the input has been fully read, and memory does not scale
/// with input size.
///
/// # Threading
///
/// This is **single-threaded**, and that is the point: the parallel encoder is
/// whole-buffer (it needs random access to the input for inter-block dictionaries),
/// so any thread count above 1 must `read_to_end` first. Use
/// [`compress_to_writer_with_threads`] if you want parallelism and can afford to
/// buffer the whole input; use this when you cannot.
///
/// ⭐ OWNER REVIEW, 2026-08-23 — this function used to default to all CPUs while its
/// own documentation promised "suitable for large inputs you don't want to buffer
/// entirely in memory". It then called `read_to_end` before emitting a single byte:
///
///   "The library's 'streaming' writer API buffers the entire input whenever it uses
///    more than one thread. This violates both the API docs and README promise for
///    large inputs."
///
/// The CLI already had the right rule for the same situation and it was never applied
/// here — `compress::io` on pipe stdin: "stream directly without buffering all input
/// first. Single-threaded so output begins immediately without OOM risk."
///
/// Enforced by `tests/streaming_api_is_honest.rs`, which fails if the first byte of
/// output requires the whole input to have been read.
///
/// Returns the number of **uncompressed** bytes consumed from `reader`.
pub fn compress_to_writer<R: std::io::Read, W: std::io::Write + Send>(
    reader: R,
    writer: W,
    level: u8,
) -> GzippyResult<u64> {
    compress_to_writer_with_threads(reader, writer, level, 1)
}

/// Compress data from `reader` into `writer` at `level` with explicit thread count.
///
/// The same threading and format rules as [`compress_with_threads`] apply.
///
/// # ⚠ `threads > 1` BUFFERS THE ENTIRE INPUT
///
/// The parallel encoder is whole-buffer — workers index the input directly and each
/// block's dictionary is the preceding 32 KiB — so with more than one thread this
/// reads the reader to end before emitting any output. Peak memory is therefore
/// `input + O(threads * block_size)`. If you need output to begin before input ends,
/// or memory that does not scale with input, use [`compress_to_writer`], which is
/// single-threaded and genuinely streams.
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
// Deflate64 API (ZIP method 9 / Enhanced Deflate)
// =============================================================================

/// Decompress a raw Deflate64 stream (ZIP method 9 / Enhanced Deflate).
///
/// Deflate64 extends standard DEFLATE with a 64 KB sliding window,
/// length codes up to 65 538 bytes, and distance codes up to 65 536 bytes.
/// It is used as compression method 9 in ZIP archives.
///
/// `data` must be a raw Deflate64 bitstream — no ZIP local-file header,
/// no gzip framing.  Returns the decompressed bytes.
///
/// Returns an error if `data` is not valid Deflate64.
pub fn decompress_deflate64(data: &[u8]) -> GzippyResult<Vec<u8>> {
    decompress::deflate64::decompress_deflate64(data)
}

/// Decompress a raw Deflate64 stream into `writer`.
///
/// Streaming variant of [`decompress_deflate64`] — avoids the intermediate
/// allocation when the caller already has a [`Write`] target.
///
/// Returns the number of decompressed bytes written.
pub fn decompress_deflate64_to_writer<W: std::io::Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    decompress::deflate64::decompress_deflate64_to_writer(data, writer)
}

/// Compress `data` as a raw Deflate64 bitstream, returning `Vec<u8>`.
///
/// Produces a valid Deflate64 (ZIP method 9 / Enhanced Deflate) raw stream.
/// No gzip or ZIP container is added.
pub fn compress_deflate64(data: &[u8]) -> GzippyResult<Vec<u8>> {
    compress::deflate64::compress_deflate64(data)
}

/// Compress `data` as a raw Deflate64 bitstream, writing to `writer`.
///
/// Returns the number of compressed bytes written.
pub fn compress_deflate64_to_writer<W: std::io::Write>(
    data: &[u8],
    writer: &mut W,
) -> GzippyResult<u64> {
    compress::deflate64::compress_deflate64_to_writer(data, writer)
}

// =============================================================================
// Routing inspection
// =============================================================================

/// Return the [`DecodePath`] gzippy would choose for `data` with `threads`.
///
/// Useful for tests and diagnostics. Does not allocate or decompress.
pub fn classify(data: &[u8], threads: usize) -> DecodePath {
    decompress::classify_gzip(data, threads)
}
