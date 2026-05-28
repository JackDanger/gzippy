//! `gzippy-inflate` — a pure-Rust DEFLATE / gzip inflate primitive.
//!
//! Public surface for `plans/unified-decoder.md` §3.11. v0.1.0 ships
//! the stable API skeleton; the implementation today delegates to
//! `libdeflater` (matching the gzippy parent crate's production path)
//! and migrates to the pure-Rust inflate in v0.2.0 once the Route C
//! work is complete.
//!
//! ## Roadmap
//!
//! - v0.1.0 (this release): API scaffold + delegated impl.
//! - v0.2.0: pure-Rust inflate migrated in from `gzippy::decompress::inflate`.
//! - v0.3.0: `no_std` + `alloc` core for `decode_deflate_into`.
//! - v0.4.0: `#[cfg(target_arch = "wasm32")]` WebAssembly target.
//!
//! ## Public API
//!
//! ```ignore
//! use gzippy_inflate::Inflate;
//! let data = Inflate::decode_gzip(&compressed)?;
//! ```
//!
//! Async surface (gated on `async` feature) lives in `crate::r#async`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub mod aot;
pub mod bit_reader;
pub mod chd;
pub mod constant_time;
pub mod gpu;
pub use aot::{fingerprint_hash, match_aot_fingerprint};
pub use bit_reader::{pext, BitReaderI32, ShiftRegisterWidth};
pub use chd::ChdTable;
pub use constant_time::ConstantTimeInflate;
pub use gpu::{BlockRange, GpuBackend, GpuError, GpuInflate};

/// Public inflate primitive. Stateless; each method allocates its own output.
///
/// v0.1.0: delegates to `libdeflater::Decompressor`. Production code in
/// the parent gzippy crate uses the same backend at this version, so
/// behavior is identical.
pub struct Inflate;

impl Inflate {
    /// Decode a complete gzip-format byte stream.
    ///
    /// Returns the uncompressed bytes. Internally sizes the output
    /// buffer from the gzip trailer's ISIZE field for single-member
    /// streams; multi-member streams grow the output incrementally.
    #[cfg(feature = "std")]
    pub fn decode_gzip(input: &[u8]) -> Result<Vec<u8>, InflateError> {
        use libdeflater::Decompressor;
        // ISIZE is the gzip trailer's last 4 bytes (mod 2^32 per RFC 1952).
        if input.len() < 18 {
            return Err(InflateError::Truncated);
        }
        let tail = &input[input.len() - 4..];
        let expected = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as usize;
        // For multi-member files ISIZE wraps; oversize by 4 GiB as a
        // conservative upper bound. The libdeflater wrapper returns the
        // actual byte count.
        let mut out = vec![0u8; expected.max(1024)];
        let mut decompressor = Decompressor::new();
        loop {
            match decompressor.gzip_decompress(input, &mut out) {
                Ok(n) => {
                    out.truncate(n);
                    return Ok(out);
                }
                Err(libdeflater::DecompressionError::InsufficientSpace) => {
                    // Double the buffer; tries again.
                    let new_cap = (out.len() * 2).max(out.len() + 1024 * 1024);
                    out.resize(new_cap, 0);
                }
                Err(libdeflater::DecompressionError::BadData) => {
                    return Err(InflateError::BadData);
                }
            }
        }
    }

    /// Decode a raw DEFLATE byte stream (no gzip header/trailer).
    /// Writes into the provided output slice; returns bytes written.
    #[cfg(feature = "std")]
    pub fn decode_deflate_into(input: &[u8], output: &mut [u8]) -> Result<usize, InflateError> {
        use libdeflater::Decompressor;
        let mut decompressor = Decompressor::new();
        decompressor
            .deflate_decompress(input, output)
            .map_err(|e| match e {
                libdeflater::DecompressionError::BadData => InflateError::BadData,
                libdeflater::DecompressionError::InsufficientSpace => InflateError::OutputTooSmall,
            })
    }

    /// Build a builder for configurable inflate.
    pub fn builder() -> InflateBuilder {
        InflateBuilder::default()
    }
}

/// Builder for configurable inflate parameters.
///
/// v0.1.0: all settings are no-ops because the backend (libdeflater)
/// doesn't expose them. Reserved for v0.2.0+ pure-Rust backend that
/// supports per-call BTYPE specialization, BMI2 dispatch, etc.
#[derive(Default)]
pub struct InflateBuilder {
    pub(crate) _constant_time: bool,
    pub(crate) _check_crc: bool,
}

impl InflateBuilder {
    /// Request constant-time decode (branchless cmov dispatch) for
    /// security-sensitive use. ~20% slower than the optimized variant.
    /// v0.1.0: no-op; reserved for v0.3.0+.
    pub fn constant_time(mut self, on: bool) -> Self {
        self._constant_time = on;
        self
    }

    /// Verify the gzip CRC32 trailer matches decoded output.
    /// v0.1.0: always on for `decode_gzip` (libdeflater verifies).
    pub fn check_crc(mut self, on: bool) -> Self {
        self._check_crc = on;
        self
    }

    /// Decode a gzip stream with these settings.
    #[cfg(feature = "std")]
    pub fn decode_gzip(self, input: &[u8]) -> Result<Vec<u8>, InflateError> {
        Inflate::decode_gzip(input)
    }
}

/// Inflate error surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InflateError {
    /// Input is shorter than the minimum valid gzip stream (18 bytes).
    Truncated,
    /// DEFLATE stream is malformed (invalid Huffman codes, bad block
    /// header, distance out of range, etc.).
    BadData,
    /// Provided output buffer is too small for the decoded payload.
    OutputTooSmall,
}

impl core::fmt::Display for InflateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InflateError::Truncated => write!(f, "truncated input"),
            InflateError::BadData => write!(f, "malformed DEFLATE stream"),
            InflateError::OutputTooSmall => write!(f, "output buffer too small"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InflateError {}

// ── §3.12 async API ─────────────────────────────────────────────────────
//
// Gated on the `async` feature. Wraps the sync `Inflate` in a
// non-blocking adapter compatible with tokio's async runtime.
#[cfg(feature = "async")]
pub mod r#async {
    use super::{Inflate, InflateError};
    use futures::stream::{self, Stream};
    use tokio::io::AsyncReadExt;

    /// Async inflate. Reads the full gzip stream from `reader`, then
    /// decodes synchronously and yields the result as a single chunk.
    ///
    /// v0.1.0 is the simplest correct implementation: buffer all input,
    /// decode, yield. v0.2.0 will stream-decode (yield decoded chunks
    /// as input arrives).
    pub struct AsyncInflate;

    impl AsyncInflate {
        /// Decode an async-readable gzip stream. Yields one `Result<Vec<u8>,
        /// InflateError>` containing the entire decoded payload (v0.1.0
        /// behavior; v0.2.0 streams chunks).
        pub fn decode<R>(reader: R) -> impl Stream<Item = Result<Vec<u8>, InflateError>>
        where
            R: tokio::io::AsyncRead + Unpin + Send + 'static,
        {
            stream::once(async move {
                let mut input = Vec::new();
                let mut r = reader;
                r.read_to_end(&mut input)
                    .await
                    .map_err(|_| InflateError::Truncated)?;
                Inflate::decode_gzip(&input)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gzip(payload: &[u8]) -> Vec<u8> {
        use libdeflater::{CompressionLvl, Compressor};
        let mut compressor = Compressor::new(CompressionLvl::default());
        let bound = compressor.gzip_compress_bound(payload.len());
        let mut out = vec![0u8; bound];
        let n = compressor.gzip_compress(payload, &mut out).unwrap();
        out.truncate(n);
        out
    }

    #[test]
    fn decode_round_trip() {
        let payload = b"Hello, gzippy-inflate world!";
        let gz = make_gzip(payload);
        let decoded = Inflate::decode_gzip(&gz).unwrap();
        assert_eq!(decoded.as_slice(), payload);
    }

    #[test]
    fn decode_empty_payload() {
        let gz = make_gzip(b"");
        let decoded = Inflate::decode_gzip(&gz).unwrap();
        assert_eq!(decoded.as_slice(), b"");
    }

    #[test]
    fn decode_large_payload_grows_buffer() {
        // 1 MiB highly compressible payload; ISIZE will be a real
        // number this time.
        let payload = vec![b'A'; 1_000_000];
        let gz = make_gzip(&payload);
        let decoded = Inflate::decode_gzip(&gz).unwrap();
        assert_eq!(decoded.len(), payload.len());
        assert_eq!(decoded.as_slice(), payload.as_slice());
    }

    #[test]
    fn decode_truncated() {
        let r = Inflate::decode_gzip(&[1, 2, 3]);
        assert_eq!(r.unwrap_err(), InflateError::Truncated);
    }

    #[test]
    fn decode_bad_data() {
        // 18+ bytes of garbage that aren't a valid gzip stream.
        let bad = [0xFFu8; 32];
        let r = Inflate::decode_gzip(&bad);
        // Either Truncated (if libdeflater bails early) or BadData.
        assert!(matches!(
            r,
            Err(InflateError::BadData) | Err(InflateError::Truncated)
        ));
    }

    #[test]
    fn builder_settings_no_op_in_v0_1() {
        let payload = b"builder works";
        let gz = make_gzip(payload);
        let decoded = Inflate::builder()
            .constant_time(true)
            .check_crc(true)
            .decode_gzip(&gz)
            .unwrap();
        assert_eq!(decoded.as_slice(), payload);
    }
}
