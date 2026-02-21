//! ISA-L Decompression Backend
//!
//! Uses Intel ISA-L for fast gzip decompression on x86_64.
//! ISA-L uses AVX2/AVX-512 for Huffman decode, vpclmulqdq for CRC32,
//! and optimized RLE match copy — 45-56% faster than libdeflate on
//! repetitive data (logs, software archives).
//!
//! Falls back to libdeflate when ISA-L is not available (ARM, or feature disabled).

/// Check if ISA-L decompression is available and beneficial.
/// Only true on x86_64 with the isal-compression feature enabled.
#[inline]
pub fn is_available() -> bool {
    cfg!(all(feature = "isal-compression", target_arch = "x86_64"))
}

/// Decompress a full gzip stream (header + deflate + trailer) using ISA-L.
/// Returns the decompressed data, or None if ISA-L is unavailable or fails.
///
/// The caller should provide a size hint (e.g., from ISIZE trailer) to avoid
/// repeated buffer growth.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decompress_gzip(input: &[u8], size_hint: usize) -> Option<Vec<u8>> {
    let initial_size = if size_hint > 0 && size_hint < 2 * 1024 * 1024 * 1024 {
        size_hint
    } else {
        input.len().saturating_mul(4).max(256 * 1024)
    };

    let mut output = vec![0u8; initial_size];
    match isal::decompress_into(input, &mut output, isal::Codec::Gzip) {
        Ok(size) => {
            output.truncate(size);
            Some(output)
        }
        Err(_) => {
            // Retry with larger buffer if output was too small
            let mut retry_size = initial_size.saturating_mul(2);
            for _ in 0..4 {
                output.resize(retry_size, 0);
                match isal::decompress_into(input, &mut output, isal::Codec::Gzip) {
                    Ok(size) => {
                        output.truncate(size);
                        return Some(output);
                    }
                    Err(_) => {
                        retry_size = retry_size.saturating_mul(2);
                    }
                }
            }
            None
        }
    }
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decompress_gzip(_input: &[u8], _size_hint: usize) -> Option<Vec<u8>> {
    None
}

/// Decompress a full gzip stream into a pre-allocated output buffer.
/// Returns the number of bytes written, or None on failure.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn decompress_gzip_into(input: &[u8], output: &mut [u8]) -> Option<usize> {
    isal::decompress_into(input, output, isal::Codec::Gzip).ok()
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn decompress_gzip_into(_input: &[u8], _output: &mut [u8]) -> Option<usize> {
    None
}

/// Decompress raw deflate data into a pre-allocated output buffer.
/// Returns the number of bytes written, or None on failure.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn decompress_deflate_into(input: &[u8], output: &mut [u8]) -> Option<usize> {
    isal::decompress_into(input, output, isal::Codec::Deflate).ok()
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn decompress_deflate_into(_input: &[u8], _output: &mut [u8]) -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_isal_decompress_roundtrip() {
        use super::*;

        let original = b"Hello, World! This is a test of ISA-L decompression. \
                         Repeated data for compression: AAAAAAAAAAAAAAAAAAA \
                         More repeated data: BBBBBBBBBBBBBBBBBBBBBBBB";

        // Compress with libdeflate
        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let max_size = compressor.gzip_compress_bound(original.len());
        let mut compressed = vec![0u8; max_size];
        let size = compressor
            .gzip_compress(original, &mut compressed)
            .expect("compression failed");
        compressed.truncate(size);

        // Decompress with ISA-L
        let result = decompress_gzip(&compressed, original.len()).expect("decompression failed");
        assert_eq!(&result, &original[..]);
    }

    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_isal_decompress_into() {
        use super::*;

        let original: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let max_size = compressor.gzip_compress_bound(original.len());
        let mut compressed = vec![0u8; max_size];
        let size = compressor
            .gzip_compress(&original, &mut compressed)
            .expect("compression failed");
        compressed.truncate(size);

        let mut output = vec![0u8; original.len()];
        let bytes = decompress_gzip_into(&compressed, &mut output).expect("decompression failed");
        assert_eq!(bytes, original.len());
        assert_eq!(&output[..bytes], &original[..]);
    }
}
