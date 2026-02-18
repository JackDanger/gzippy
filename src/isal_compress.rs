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

/// Map gzippy compression level to ISA-L CompressionLevel.
/// ISA-L only supports levels 0, 1, 3 (no level 2).
#[cfg(feature = "isal-compression")]
#[inline]
fn to_isal_level(level: u32) -> isal::CompressionLevel {
    match level {
        0 => isal::CompressionLevel::Zero,
        1 => isal::CompressionLevel::One,
        _ => isal::CompressionLevel::Three,
    }
}

/// Compress data using ISA-L gzip at the given level (0-3).
/// Uses stateless single-shot compression for maximum throughput.
///
/// Returns None if ISA-L is not available, allowing the caller to fall back.
#[cfg(feature = "isal-compression")]
#[allow(dead_code)]
pub fn compress_gzip(data: &[u8], level: u32) -> Option<Vec<u8>> {
    // Worst case: incompressible data + gzip header/trailer overhead
    let max_size = data.len() + data.len() / 10 + 256;
    let mut output = vec![0u8; max_size];
    let size =
        isal::compress_into(data, &mut output, to_isal_level(level), isal::Codec::Gzip).ok()?;
    output.truncate(size);
    Some(output)
}

#[cfg(not(feature = "isal-compression"))]
#[allow(dead_code)]
pub fn compress_gzip(_data: &[u8], _level: u32) -> Option<Vec<u8>> {
    None
}

/// Compress a block of data using ISA-L deflate (raw, no gzip wrapper).
/// Uses stateless single-shot compression for maximum throughput.
/// Returns None if ISA-L is not available.
#[cfg(feature = "isal-compression")]
pub fn compress_deflate(data: &[u8], level: u32) -> Option<Vec<u8>> {
    let max_size = data.len() + data.len() / 10 + 256;
    let mut output = vec![0u8; max_size];
    let size = isal::compress_into(
        data,
        &mut output,
        to_isal_level(level),
        isal::Codec::Deflate,
    )
    .ok()?;
    output.truncate(size);
    Some(output)
}

#[cfg(not(feature = "isal-compression"))]
pub fn compress_deflate(_data: &[u8], _level: u32) -> Option<Vec<u8>> {
    None
}

/// Gzip compression using ISA-L's single-shot stateless API.
/// Buffers all input, then compresses in one call for maximum throughput.
/// Returns bytes read from the input on success.
#[cfg(feature = "isal-compression")]
pub fn compress_gzip_stream<R: std::io::Read, W: std::io::Write>(
    reader: &mut R,
    mut writer: W,
    level: u32,
) -> std::io::Result<u64> {
    let mut input = Vec::new();
    reader.read_to_end(&mut input)?;
    let bytes = input.len() as u64;

    if input.is_empty() {
        // Write minimal gzip for empty input
        let compressed = compress_gzip(&input, level)
            .ok_or_else(|| std::io::Error::other("ISA-L compression failed"))?;
        writer.write_all(&compressed)?;
        return Ok(0);
    }

    let max_size = input.len() + input.len() / 10 + 256;
    let mut output = vec![0u8; max_size];
    let size = isal::compress_into(&input, &mut output, to_isal_level(level), isal::Codec::Gzip)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    writer.write_all(&output[..size])?;
    Ok(bytes)
}

/// Compress a slice directly to a writer using ISA-L gzip.
/// Avoids the copy that compress_gzip_stream does when data is already in memory.
#[cfg(feature = "isal-compression")]
pub fn compress_gzip_to_writer<W: std::io::Write>(
    data: &[u8],
    mut writer: W,
    level: u32,
) -> std::io::Result<u64> {
    let max_size = data.len() + data.len() / 10 + 256;
    let mut output = vec![0u8; max_size];
    let size = isal::compress_into(data, &mut output, to_isal_level(level), isal::Codec::Gzip)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    writer.write_all(&output[..size])?;
    Ok(data.len() as u64)
}

#[cfg(not(feature = "isal-compression"))]
pub fn compress_gzip_to_writer<W: std::io::Write>(
    _data: &[u8],
    _writer: W,
    _level: u32,
) -> std::io::Result<u64> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "ISA-L not available",
    ))
}

#[cfg(not(feature = "isal-compression"))]
pub fn compress_gzip_stream<R: std::io::Read, W: std::io::Write>(
    _reader: &mut R,
    _writer: W,
    _level: u32,
) -> std::io::Result<u64> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "ISA-L not available",
    ))
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
