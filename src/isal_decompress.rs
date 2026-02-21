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

/// Stream-decompress a gzip stream using ISA-L, writing directly to the writer.
/// Avoids allocating a buffer for the entire decompressed output.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decompress_gzip_stream<W: std::io::Write>(input: &[u8], writer: &mut W) -> Option<u64> {
    use std::io::Read;
    let cursor = std::io::Cursor::new(input);
    let mut decoder = isal::read::Decoder::new(cursor, isal::Codec::Gzip);

    let mut buf = vec![0u8; 256 * 1024];
    let mut total = 0u64;
    loop {
        match decoder.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if writer.write_all(&buf[..n]).is_err() {
                    return None;
                }
                total += n as u64;
            }
            Err(_) => return None,
        }
    }
    Some(total)
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decompress_gzip_stream<W: std::io::Write>(_input: &[u8], _writer: &mut W) -> Option<u64> {
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

        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let max_size = compressor.gzip_compress_bound(original.len());
        let mut compressed = vec![0u8; max_size];
        let size = compressor
            .gzip_compress(original, &mut compressed)
            .expect("compression failed");
        compressed.truncate(size);

        let mut result = Vec::new();
        let bytes = decompress_gzip_stream(&compressed, &mut result).expect("decompression failed");
        assert_eq!(bytes as usize, original.len());
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
