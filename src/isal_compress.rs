//! ISA-L Compression Backend
//!
//! Uses Intel ISA-L for fast compression at levels 0-3.
//! ISA-L uses AVX2/AVX-512 for the LZ77 matching step,
//! achieving 2-3 GB/s — 3-5x faster than zlib-ng at these levels.
//!
//! Falls back to zlib-ng when ISA-L is not available (ARM, or feature disabled).

/// Check if ISA-L compression is available at compile time
pub fn is_available() -> bool {
    cfg!(feature = "isal-compression")
}

/// Compress data using ISA-L gzip at the given level (0-3).
/// Returns the compressed data including gzip header and trailer.
///
/// Returns None if ISA-L is not available, allowing the caller to fall back.
#[cfg(feature = "isal-compression")]
#[allow(dead_code)]
pub fn compress_gzip(data: &[u8], level: u32) -> Option<Vec<u8>> {
    use isal::write::Encoder;
    use isal::{Codec, CompressionLevel};
    use std::io::Write;

    // ISA-L only supports levels 0, 1, 3 (no level 2)
    let isal_level = match level {
        0 => CompressionLevel::Zero,
        1 => CompressionLevel::One,
        _ => CompressionLevel::Three,
    };

    let mut encoder = Encoder::new(Vec::new(), isal_level, Codec::Gzip);
    encoder.write_all(data).ok()?;
    encoder.finish().ok()
}

#[cfg(not(feature = "isal-compression"))]
#[allow(dead_code)]
pub fn compress_gzip(_data: &[u8], _level: u32) -> Option<Vec<u8>> {
    None
}

/// Compress a block of data using ISA-L deflate (raw, no gzip wrapper).
/// Returns None if ISA-L is not available.
#[cfg(feature = "isal-compression")]
pub fn compress_deflate(data: &[u8], level: u32) -> Option<Vec<u8>> {
    use isal::write::Encoder;
    use isal::{Codec, CompressionLevel};
    use std::io::Write;

    // ISA-L only supports levels 0, 1, 3 (no level 2)
    let isal_level = match level {
        0 => CompressionLevel::Zero,
        1 => CompressionLevel::One,
        _ => CompressionLevel::Three,
    };

    let mut encoder = Encoder::new(Vec::new(), isal_level, Codec::Deflate);
    encoder.write_all(data).ok()?;
    encoder.finish().ok()
}

#[cfg(not(feature = "isal-compression"))]
pub fn compress_deflate(_data: &[u8], _level: u32) -> Option<Vec<u8>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isal_available() {
        // ISA-L should be available when compiled with the feature
        if cfg!(feature = "isal-compression") {
            assert!(is_available());
        }
    }

    #[test]
    #[cfg(feature = "isal-compression")]
    fn test_compress_gzip_roundtrip() {
        let original = b"Hello, World! This is a test of ISA-L compression.";
        let compressed = compress_gzip(original, 1).expect("compression failed");

        // Verify it's valid gzip
        assert!(compressed.len() >= 10);
        assert_eq!(compressed[0], 0x1f);
        assert_eq!(compressed[1], 0x8b);

        // Decompress with libdeflate to verify
        let mut decompressor = libdeflater::Decompressor::new();
        let mut output = vec![0u8; original.len() + 1024];
        let size = decompressor
            .gzip_decompress(&compressed, &mut output)
            .expect("decompression failed");
        assert_eq!(&output[..size], &original[..]);
    }

    #[test]
    #[cfg(feature = "isal-compression")]
    fn test_compress_all_levels() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        for level in 0..=3 {
            let compressed = compress_gzip(&data, level).expect("compression failed");
            assert!(
                compressed.len() < data.len(),
                "level {} should compress",
                level
            );

            // Verify roundtrip
            let mut decompressor = libdeflater::Decompressor::new();
            let mut output = vec![0u8; data.len() + 1024];
            let size = decompressor
                .gzip_decompress(&compressed, &mut output)
                .expect("decompression failed");
            assert_eq!(&output[..size], &data[..]);
        }
    }
}
