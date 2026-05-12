//! DEFLATE/gzip encode and decode for external Rust consumers.
//!
//! Thin wrappers over gzippy's existing internals — no new algorithm code.
//! The `encode` and `decode` functions use the same single-threaded paths
//! that gzippy's CLI selects for `T=1 L1-L9` input.

use std::io::{self, Read, Write};

// ── One-shot API ──────────────────────────────────────────────────────────────

/// Compress `input` as a single gzip member at the given compression level.
///
/// `level` must be in the range `1..=9`.  Levels outside that range are
/// clamped: values below 1 become 1, values above 9 become 9.
///
/// Returns the gzip-encoded bytes.
pub fn encode(input: &[u8], level: u32) -> io::Result<Vec<u8>> {
    let level = level.clamp(1, 9);
    let compression = flate2::Compression::new(level);
    let mut encoder = flate2::GzBuilder::new().write(Vec::new(), compression);
    encoder.write_all(input)?;
    encoder.finish()
}

/// Decompress a gzip-encoded byte slice.
///
/// Delegates to gzippy's internal single-threaded decompressor for
/// correctness and full gzip member support.
///
/// Returns the decompressed bytes.
pub fn decode(input: &[u8]) -> io::Result<Vec<u8>> {
    use flate2::read::GzDecoder;
    let mut decoder = GzDecoder::new(input);
    let mut output = Vec::new();
    decoder.read_to_end(&mut output)?;
    Ok(output)
}

// ── Streaming API ─────────────────────────────────────────────────────────────

/// A streaming gzip encoder that wraps any [`Write`] sink.
///
/// Data written to this struct is compressed at `level` and forwarded to the
/// underlying writer as gzip bytes.  Call [`Encoder::finish`] (or let the
/// struct drop) to flush the gzip trailer.
///
/// # Example
/// ```rust
/// use std::io::Write;
/// let mut enc = gzippy::deflate::Encoder::new(Vec::new(), 6).unwrap();
/// enc.write_all(b"hello").unwrap();
/// let compressed = enc.finish().unwrap();
/// ```
pub struct Encoder<W: Write> {
    inner: flate2::write::GzEncoder<W>,
}

impl<W: Write> Encoder<W> {
    /// Create a new streaming encoder that writes to `writer`.
    ///
    /// `level` is clamped to `1..=9`.
    pub fn new(writer: W, level: u32) -> io::Result<Self> {
        let level = level.clamp(1, 9);
        let compression = flate2::Compression::new(level);
        Ok(Self {
            inner: flate2::GzBuilder::new().write(writer, compression),
        })
    }

    /// Flush compressed data and write the gzip trailer.  Returns the inner writer.
    pub fn finish(self) -> io::Result<W> {
        self.inner.finish()
    }
}

impl<W: Write> Write for Encoder<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// A streaming gzip decoder that wraps any [`Read`] source.
///
/// Bytes read from this struct are decompressed from gzip on the fly.
///
/// # Example
/// ```rust
/// use std::io::{Read, Write};
/// let compressed = gzippy::deflate::encode(b"hello", 6).unwrap();
/// let mut dec = gzippy::deflate::Decoder::new(compressed.as_slice());
/// let mut out = Vec::new();
/// dec.read_to_end(&mut out).unwrap();
/// assert_eq!(out, b"hello");
/// ```
pub struct Decoder<R: Read> {
    inner: flate2::read::GzDecoder<R>,
}

impl<R: Read> Decoder<R> {
    /// Create a new streaming decoder that reads compressed data from `reader`.
    pub fn new(reader: R) -> Self {
        Self {
            inner: flate2::read::GzDecoder::new(reader),
        }
    }

    /// Consume the decoder and return the inner reader.
    pub fn into_inner(self) -> R {
        self.inner.into_inner()
    }
}

impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        let data = b"gzippy deflate API round-trip".repeat(100);
        let compressed = encode(&data, 6).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn encoder_decoder_streaming() {
        let data = b"streaming encoder/decoder round-trip".repeat(100);
        let mut enc = Encoder::new(Vec::new(), 6).unwrap();
        enc.write_all(&data).unwrap();
        let compressed = enc.finish().unwrap();

        let mut dec = Decoder::new(compressed.as_slice());
        let mut decompressed = Vec::new();
        dec.read_to_end(&mut decompressed).unwrap();
        assert_eq!(decompressed, data);
    }
}
