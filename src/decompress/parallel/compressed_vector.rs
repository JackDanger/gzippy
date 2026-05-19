//! Literal port of `rapidgzip::CompressedVector`
//! (vendor/.../CompressedVector.hpp:113-246).
//!
//! Stores a byte vector in compressed form (deflate/zlib/gzip/none) for
//! memory-efficient window caching. Decompresses on demand. Used by
//! `WindowMap` (Step 13 wiring) so the cache holds 32 KiB windows in
//! ~1-10 KiB each on highly-compressible data.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::sync::Arc;

/// Subset of rapidgzip's `CompressionType` enum
/// (CompressedVector.hpp:21-34). We support the four types our pipeline
/// actually needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    Deflate,
    Zlib,
    Gzip,
}

impl CompressionType {
    pub fn as_str(self) -> &'static str {
        match self {
            CompressionType::None => "NONE",
            CompressionType::Deflate => "Deflate",
            CompressionType::Zlib => "ZLIB",
            CompressionType::Gzip => "GZIP",
        }
    }
}

/// Literal port of `rapidgzip::CompressedVector<Container>`
/// (CompressedVector.hpp:113-246). Always uses `Vec<u8>` for the
/// container (rapidgzip's `FasterVector<uint8_t>` analogue).
pub struct CompressedVector {
    compression_type: CompressionType,
    decompressed_size: usize,
    data: Arc<Vec<u8>>,
}

impl CompressedVector {
    /// Compress `to_compress` into the given `compression_type`.
    /// `CompressionType::None` stores the bytes verbatim.
    pub fn from_bytes(to_compress: &[u8], compression_type: CompressionType) -> Self {
        let decompressed_size = to_compress.len();
        if to_compress.is_empty() {
            return Self {
                compression_type,
                decompressed_size: 0,
                data: Arc::new(Vec::new()),
            };
        }
        let data = if matches!(compression_type, CompressionType::None) {
            to_compress.to_vec()
        } else {
            compress(to_compress, compression_type).unwrap_or_else(|_| to_compress.to_vec())
        };
        Self {
            compression_type,
            decompressed_size,
            data: Arc::new(data),
        }
    }

    /// Construct from already-compressed bytes (e.g., loaded from an
    /// index file). Mirrors the third rapidgzip constructor at
    /// CompressedVector.hpp:140-147.
    pub fn from_compressed(
        compressed_data: Vec<u8>,
        decompressed_size: usize,
        compression_type: CompressionType,
    ) -> Self {
        Self {
            compression_type,
            decompressed_size,
            data: Arc::new(compressed_data),
        }
    }

    pub fn compression_type(&self) -> CompressionType {
        self.compression_type
    }

    pub fn compressed_data(&self) -> Arc<Vec<u8>> {
        self.data.clone()
    }

    /// Borrow the stored bytes. For `CompressionType::None` these are
    /// the verbatim source bytes; for other types they are the
    /// compressed payload. Used by `WindowMap::get`'s None-fast-path to
    /// skip the `decompress()` Vec clone.
    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    pub fn decompressed_size(&self) -> usize {
        self.decompressed_size
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Decompress and return the original bytes.
    pub fn decompress(&self) -> Vec<u8> {
        if self.data.is_empty() {
            return Vec::new();
        }
        match self.compression_type {
            CompressionType::None => self.data.as_ref().clone(),
            CompressionType::Zlib => zlib_decompress(&self.data, self.decompressed_size),
            CompressionType::Gzip => gzip_decompress(&self.data, self.decompressed_size),
            CompressionType::Deflate => deflate_decompress(&self.data, self.decompressed_size),
        }
    }
}

impl std::fmt::Debug for CompressedVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressedVector")
            .field("compression_type", &self.compression_type)
            .field("decompressed_size", &self.decompressed_size)
            .field("compressed_size", &self.data.len())
            .finish()
    }
}

fn compress(input: &[u8], kind: CompressionType) -> std::io::Result<Vec<u8>> {
    use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
    use flate2::Compression;
    match kind {
        CompressionType::None => Ok(input.to_vec()),
        CompressionType::Zlib => {
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
            enc.write_all(input)?;
            enc.finish()
        }
        CompressionType::Gzip => {
            let mut enc = GzEncoder::new(Vec::new(), Compression::default());
            enc.write_all(input)?;
            enc.finish()
        }
        CompressionType::Deflate => {
            let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
            enc.write_all(input)?;
            enc.finish()
        }
    }
}

fn zlib_decompress(input: &[u8], expected: usize) -> Vec<u8> {
    use flate2::read::ZlibDecoder;
    let mut out = Vec::with_capacity(expected);
    let mut dec = ZlibDecoder::new(input);
    let _ = dec.read_to_end(&mut out);
    out
}

fn gzip_decompress(input: &[u8], expected: usize) -> Vec<u8> {
    use flate2::read::GzDecoder;
    let mut out = Vec::with_capacity(expected);
    let mut dec = GzDecoder::new(input);
    let _ = dec.read_to_end(&mut out);
    out
}

fn deflate_decompress(input: &[u8], expected: usize) -> Vec<u8> {
    use flate2::read::DeflateDecoder;
    let mut out = Vec::with_capacity(expected);
    let mut dec = DeflateDecoder::new(input);
    let _ = dec.read_to_end(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_zlib() {
        let input: Vec<u8> = (0..1024u16).map(|i| (i % 251) as u8).collect();
        let cv = CompressedVector::from_bytes(&input, CompressionType::Zlib);
        assert_eq!(cv.decompressed_size(), input.len());
        assert!(cv.compressed_size() <= input.len());
        let out = cv.decompress();
        assert_eq!(out, input);
    }

    #[test]
    fn round_trips_through_gzip() {
        let input = b"hello world hello world hello world".to_vec();
        let cv = CompressedVector::from_bytes(&input, CompressionType::Gzip);
        assert_eq!(cv.decompress(), input);
    }

    #[test]
    fn none_keeps_bytes_verbatim() {
        let input = b"identity".to_vec();
        let cv = CompressedVector::from_bytes(&input, CompressionType::None);
        assert_eq!(cv.compressed_size(), input.len());
        assert_eq!(cv.decompress(), input);
    }

    #[test]
    fn empty_vector_decompresses_to_empty() {
        let cv = CompressedVector::from_bytes(&[], CompressionType::Zlib);
        assert!(cv.is_empty());
        assert!(cv.decompress().is_empty());
    }

    #[test]
    fn from_compressed_preserves_metadata() {
        let compressed = vec![0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01];
        let cv = CompressedVector::from_compressed(compressed.clone(), 0, CompressionType::Zlib);
        assert_eq!(cv.compressed_size(), compressed.len());
        assert_eq!(cv.decompressed_size(), 0);
        assert_eq!(cv.compression_type(), CompressionType::Zlib);
    }
}
