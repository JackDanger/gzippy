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

/// Stream-decompress a gzip stream using ISA-L's raw stateful inflate,
/// writing directly to the writer. Bypasses the isal-rs Decoder wrapper
/// to eliminate Cursor and 16KB internal buffer copy overhead.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decompress_gzip_stream<W: std::io::Write>(input: &[u8], writer: &mut W) -> Option<u64> {
    use isal::isal_sys::igzip_lib as isal_raw;

    let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
    unsafe { isal_raw::isal_inflate_init(&mut state) };
    state.crc_flag = isal_raw::IGZIP_GZIP;

    state.avail_in = input.len() as u32;
    state.next_in = input.as_ptr() as *mut u8;

    let mut out_buf = vec![0u8; 1024 * 1024];
    let mut total = 0u64;

    loop {
        state.avail_out = out_buf.len() as u32;
        state.next_out = out_buf.as_mut_ptr();

        let ret = unsafe { isal_raw::isal_inflate(&mut state) };
        if ret != 0 {
            return None;
        }

        let written = out_buf.len() - state.avail_out as usize;
        if written > 0 {
            if writer.write_all(&out_buf[..written]).is_err() {
                return None;
            }
            total += written as u64;
        }

        if state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH {
            break;
        }
        if written == 0 && state.avail_in == 0 {
            break;
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

/// Scan a deflate stream using ISA-L, capturing checkpoints at block boundaries.
///
/// Faster than pure-Rust `scan_deflate_fast` on x86_64 because ISA-L uses
/// AVX2/AVX-512 for Huffman decode. Produces `ScanCheckpoint` objects
/// compatible with `two_pass_parallel`'s parallel decode phase.
///
/// Returns `None` if ISA-L is not available or an error occurs.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn scan_deflate_isal(
    deflate_data: &[u8],
    checkpoint_interval: usize,
    expected_output_size: usize,
) -> Option<crate::scan_inflate::ScanResult> {
    use crate::scan_inflate::{ScanCheckpoint, ScanResult, WINDOW_SIZE};
    use isal::isal_sys::igzip_lib as isal_raw;

    let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
    unsafe { isal_raw::isal_inflate_init(&mut state) };
    state.crc_flag = isal_raw::ISAL_DEFLATE;

    state.avail_in = deflate_data.len() as u32;
    state.next_in = deflate_data.as_ptr() as *mut u8;

    let buf_size = if expected_output_size > 0 {
        expected_output_size + 256 * 1024
    } else {
        deflate_data.len().saturating_mul(32).max(1024 * 1024)
    };
    let mut output = vec![0u8; buf_size];

    let mut out_pos: usize = 0;
    let mut checkpoints = Vec::new();
    let mut next_checkpoint_at = checkpoint_interval;

    let chunk_size: u32 = 256 * 1024;

    loop {
        let remaining_out = output.len() - out_pos;
        if remaining_out == 0 {
            output.resize(output.len() * 2, 0);
        }
        let this_chunk = remaining_out.min(chunk_size as usize);

        state.avail_out = this_chunk as u32;
        state.next_out = output[out_pos..].as_mut_ptr();

        let ret = unsafe { isal_raw::isal_inflate(&mut state) };
        if ret != 0 {
            return None;
        }

        let written = this_chunk - state.avail_out as usize;
        out_pos += written;

        if state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_NEW_HDR
            && out_pos >= next_checkpoint_at
            && out_pos >= WINDOW_SIZE
        {
            let input_byte_pos = deflate_data.len() - state.avail_in as usize;
            let window_start = out_pos - WINDOW_SIZE;
            checkpoints.push(ScanCheckpoint {
                input_byte_pos,
                bitbuf: state.read_in,
                bitsleft: state.read_in_length as u32,
                output_offset: out_pos,
                window: output[window_start..out_pos].to_vec(),
            });
            next_checkpoint_at = out_pos + checkpoint_interval;
        }

        if state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH {
            break;
        }
        if written == 0 && state.avail_in == 0 {
            break;
        }
    }

    Some(ScanResult {
        checkpoints,
        total_output_size: out_pos,
    })
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn scan_deflate_isal(
    _deflate_data: &[u8],
    _checkpoint_interval: usize,
    _expected_output_size: usize,
) -> Option<crate::scan_inflate::ScanResult> {
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

    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_isal_decompress_corrupt_returns_none() {
        use super::*;

        let original: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();
        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let max_size = compressor.gzip_compress_bound(original.len());
        let mut compressed = vec![0u8; max_size];
        let size = compressor
            .gzip_compress(&original, &mut compressed)
            .unwrap();
        compressed.truncate(size);

        // Flip a byte in the middle of the compressed data
        let mid = compressed.len() / 2;
        compressed[mid] ^= 0xFF;

        let mut output = Vec::new();
        let result = decompress_gzip_stream(&compressed, &mut output);
        assert!(result.is_none(), "corrupt data must return None");
    }

    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_isal_decompress_truncated_returns_none() {
        use super::*;

        let original: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();
        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let max_size = compressor.gzip_compress_bound(original.len());
        let mut compressed = vec![0u8; max_size];
        let size = compressor
            .gzip_compress(&original, &mut compressed)
            .unwrap();
        compressed.truncate(size);

        // Truncate to half
        let truncated = &compressed[..compressed.len() / 2];

        let mut output = Vec::new();
        let result = decompress_gzip_stream(truncated, &mut output);
        assert!(result.is_none(), "truncated data must return None");
    }

    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_isal_decompress_empty_returns_none() {
        use super::*;

        let mut output = Vec::new();
        let result = decompress_gzip_stream(&[], &mut output);
        assert!(result.is_none(), "empty input must return None");
    }

    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_isal_decompress_large_roundtrip() {
        use super::*;

        let mut data = Vec::with_capacity(4 * 1024 * 1024);
        let mut rng: u64 = 0xdeadbeef;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
        ];
        while data.len() < 4 * 1024 * 1024 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 2 {
                data.push((rng >> 16) as u8);
            } else {
                let phrase = phrases[((rng >> 24) as usize) % phrases.len()];
                let remaining = 4 * 1024 * 1024 - data.len();
                data.extend_from_slice(&phrase[..remaining.min(phrase.len())]);
            }
        }
        data.truncate(4 * 1024 * 1024);

        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let max_size = compressor.gzip_compress_bound(data.len());
        let mut compressed = vec![0u8; max_size];
        let size = compressor.gzip_compress(&data, &mut compressed).unwrap();
        compressed.truncate(size);

        let mut result = Vec::new();
        let bytes = decompress_gzip_stream(&compressed, &mut result).expect("ISA-L failed on 4MB");
        assert_eq!(bytes as usize, data.len());
        assert_eq!(result, data);
    }

    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_isal_decompress_into_corrupt_returns_none() {
        use super::*;

        let original: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
        let max_size = compressor.gzip_compress_bound(original.len());
        let mut compressed = vec![0u8; max_size];
        let size = compressor
            .gzip_compress(&original, &mut compressed)
            .unwrap();
        compressed.truncate(size);

        let mid = compressed.len() / 2;
        compressed[mid] ^= 0xFF;

        let mut output = vec![0u8; original.len() + 1024];
        let result = decompress_gzip_into(&compressed, &mut output);
        assert!(
            result.is_none(),
            "corrupt data into fixed buffer must return None"
        );
    }

    // Tests that work on ALL architectures (ISA-L or not)

    #[test]
    fn test_isal_is_available_consistent() {
        use super::*;
        let available = is_available();
        if cfg!(all(feature = "isal-compression", target_arch = "x86_64")) {
            assert!(available, "ISA-L must be available on x86_64 with feature");
        } else {
            assert!(
                !available,
                "ISA-L must not be available without feature/arch"
            );
        }
    }

    #[test]
    fn test_isal_stub_returns_none_when_unavailable() {
        use super::*;
        if is_available() {
            return;
        }
        let valid_gzip = {
            use std::io::Write;
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(b"hello world").unwrap();
            enc.finish().unwrap()
        };
        let mut output = Vec::new();
        assert!(decompress_gzip_stream(&valid_gzip, &mut output).is_none());
    }

    #[test]
    fn test_scan_deflate_isal_stub_when_unavailable() {
        use super::*;
        if is_available() {
            return;
        }
        let deflate = {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(b"hello world, this is enough data to test")
                .unwrap();
            enc.finish().unwrap()
        };
        assert!(scan_deflate_isal(&deflate, 1024, 0).is_none());
    }

    #[test]
    fn test_scan_deflate_isal_large_data() {
        use super::*;
        if !is_available() {
            eprintln!("skipping (ISA-L not available)");
            return;
        }

        let mut data = Vec::new();
        for i in 0..200_000u64 {
            data.extend_from_slice(format!("Line {}: some test data for scan\n", i).as_bytes());
        }

        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &data).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = scan_deflate_isal(&compressed, 256 * 1024, data.len())
            .expect("ISA-L scan should succeed");
        assert_eq!(result.total_output_size, data.len());
        assert!(!result.checkpoints.is_empty());

        for cp in &result.checkpoints {
            assert_eq!(cp.window.len(), crate::scan_inflate::WINDOW_SIZE);
            assert!(cp.output_offset >= crate::scan_inflate::WINDOW_SIZE);
            assert!(cp.input_byte_pos <= compressed.len());
        }
    }

    #[test]
    fn test_scan_deflate_isal_matches_pure_rust() {
        use super::*;
        if !is_available() {
            eprintln!("skipping (ISA-L not available)");
            return;
        }

        let mut data = Vec::new();
        for i in 0..200_000u64 {
            data.extend_from_slice(format!("Data item {}: payload\n", i).as_bytes());
        }

        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &data).unwrap();
        let compressed = encoder.finish().unwrap();

        let isal_result = scan_deflate_isal(&compressed, 128 * 1024, data.len())
            .expect("ISA-L scan should succeed");
        let rust_result =
            crate::scan_inflate::scan_deflate_fast(&compressed, 128 * 1024, data.len())
                .expect("pure Rust scan should succeed");

        assert_eq!(isal_result.total_output_size, rust_result.total_output_size);

        // Windows at matching output positions must be identical
        for isal_cp in &isal_result.checkpoints {
            for rust_cp in &rust_result.checkpoints {
                if isal_cp.output_offset == rust_cp.output_offset {
                    assert_eq!(
                        isal_cp.window, rust_cp.window,
                        "Window mismatch at offset {}",
                        isal_cp.output_offset
                    );
                }
            }
        }
    }
}
