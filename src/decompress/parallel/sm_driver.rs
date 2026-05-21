//! Production entry for parallel single-member gzip decompression.
//!
//! Parses the gzip envelope, runs [`super::chunk_fetcher::drive`] on the
//! raw deflate body, then verifies CRC32 + ISIZE. `single_member` is a thin
//! classifier-routed wrapper around [`read_parallel_sm`].

/// Decompress one single-member gzip buffer with the parallel chunk pipeline.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn read_parallel_sm<W: std::io::Write>(
    gzip_data: &[u8],
    writer: &mut W,
    parallelization: usize,
    target_compressed_chunk_bytes: usize,
) -> Result<ReadResult, ReadParallelSmError> {
    use crate::decompress::parallel::chunk_data::ChunkConfiguration;
    use crate::decompress::parallel::chunk_fetcher;
    use crate::decompress::parallel::gzip_format;

    let (_hdr, header_size) =
        gzip_format::read_header(gzip_data).map_err(|_| ReadParallelSmError::InvalidHeader)?;
    let trailer_size = 8;
    if gzip_data.len() < header_size + trailer_size {
        return Err(ReadParallelSmError::InvalidFormat);
    }
    let deflate_data = &gzip_data[header_size..gzip_data.len() - trailer_size];

    let footer = gzip_format::read_footer(gzip_data, gzip_data.len() - trailer_size)
        .map_err(|_| ReadParallelSmError::InvalidFormat)?;
    let expected_crc = footer.crc32;
    let expected_size = footer.uncompressed_size as usize;

    let configuration = ChunkConfiguration {
        split_chunk_size: target_compressed_chunk_bytes,
        max_decoded_chunk_size: 20 * target_compressed_chunk_bytes,
        crc32_enabled: true,
    };

    let (total_crc, total_size) =
        chunk_fetcher::drive(deflate_data, writer, parallelization, configuration)
            .map_err(|e| ReadParallelSmError::DecodeFailed(format!("{e:?}")))?;

    if total_size != expected_size {
        return Err(ReadParallelSmError::SizeMismatch {
            expected: expected_size,
            actual: total_size,
        });
    }
    if total_crc != expected_crc {
        return Err(ReadParallelSmError::CrcMismatch {
            expected: expected_crc,
            actual: total_crc,
        });
    }

    Ok(ReadResult {
        total_crc,
        total_size,
    })
}

// `total_crc` is verified against the trailer inside `read_parallel_sm`; the
// field is kept on the result for completeness even though the current caller
// reads only `total_size`. The allow is unconditional: non-x86 builds never
// construct `ReadResult`, and x86 builds construct it but read only the size.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ReadResult {
    pub total_crc: u32,
    pub total_size: usize,
}

#[cfg_attr(
    not(all(feature = "isal-compression", target_arch = "x86_64")),
    allow(dead_code)
)]
#[derive(Debug)]
pub enum ReadParallelSmError {
    InvalidHeader,
    InvalidFormat,
    DecodeFailed(String),
    SizeMismatch { expected: usize, actual: usize },
    CrcMismatch { expected: u32, actual: u32 },
}

impl std::fmt::Display for ReadParallelSmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadParallelSmError::InvalidHeader => write!(f, "invalid gzip header"),
            ReadParallelSmError::InvalidFormat => write!(f, "input below parallel SM minimum"),
            ReadParallelSmError::DecodeFailed(s) => write!(f, "chunk decode failed: {s}"),
            ReadParallelSmError::SizeMismatch { expected, actual } => {
                write!(f, "output size mismatch: expected {expected}, got {actual}")
            }
            ReadParallelSmError::CrcMismatch { expected, actual } => {
                write!(
                    f,
                    "CRC32 mismatch: expected {expected:08x}, got {actual:08x}"
                )
            }
        }
    }
}

impl std::error::Error for ReadParallelSmError {}

#[cfg(all(test, feature = "isal-compression", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_gzip(payload: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn read_parallel_sm_roundtrip() {
        let payload: Vec<u8> = (0..64 * 1024).map(|i| (i % 251) as u8).collect();
        let gzip = make_gzip(&payload);
        let mut out = Vec::new();
        let result = read_parallel_sm(&gzip, &mut out, 4, 512 * 1024).unwrap();
        assert_eq!(out, payload);
        assert_eq!(result.total_size, payload.len());
    }
}
