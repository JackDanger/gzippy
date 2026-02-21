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
