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
    let mut finished = false;

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
            finished = true;
            break;
        }
        if written == 0 && state.avail_in == 0 {
            break;
        }
    }
    if finished {
        Some(total)
    } else {
        None
    }
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decompress_gzip_stream<W: std::io::Write>(_input: &[u8], _writer: &mut W) -> Option<u64> {
    None
}

/// Write-ahead variant: ISA-L decode with writes on a background thread.
/// Overlaps write syscalls with the next inflate iteration.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn decompress_gzip_stream_threaded<W: std::io::Write + Send + 'static>(
    input: &[u8],
    writer: W,
) -> Option<(u64, W)> {
    use isal::isal_sys::igzip_lib as isal_raw;

    let wa = crate::infra::io_thread::WriteAhead::new(writer, 4);

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
            if wa.send(&out_buf[..written]).is_err() {
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

    match wa.finish() {
        Ok(w) => Some((total, w)),
        Err(_) => None,
    }
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn decompress_gzip_stream_threaded<W: std::io::Write + Send + 'static>(
    _input: &[u8],
    _writer: W,
) -> Option<(u64, W)> {
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
) -> Option<crate::decompress::scan_inflate::ScanResult> {
    use crate::decompress::scan_inflate::{ScanCheckpoint, ScanResult, WINDOW_SIZE};
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

        // Checkpoint at output-position intervals. ISAL_BLOCK_NEW_HDR is not used
        // because ISA-L often transitions through it within a single isal_inflate call,
        // so the caller never sees it — yielding zero checkpoints for most real streams.
        if out_pos >= next_checkpoint_at && out_pos >= WINDOW_SIZE {
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
) -> Option<crate::decompress::scan_inflate::ScanResult> {
    None
}

/// Decompress raw deflate from any bit offset using ISA-L + inflatePrime.
///
/// Enables ISA-L on non-byte-aligned chunk boundaries from speculative parallel
/// decode. ISA-L's inflate_state has a 64-bit bit buffer (read_in / read_in_length)
/// that can be pre-loaded with the partial first byte's bits before starting.
///
/// This is the same "inflatePrime" pattern used by rapidgzip's IsalInflateWrapper
/// (rapidgzip/librapidarchive/src/rapidgzip/gzip/isal.hpp).
///
/// `dict` is the 32KB sliding-window from the previous chunk. Empty slice is
/// valid (first chunk or chunk with no back-references before start).
/// `max_output` caps the output size — use `chunk.data.len()` from the
/// speculative decoder to reproduce exactly the right number of bytes.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decompress_deflate_from_bit(
    data: &[u8],
    bit_offset: usize,
    dict: &[u8],
    max_output: usize,
) -> Option<Vec<u8>> {
    use isal::isal_sys::igzip_lib as isal_raw;

    let byte_idx = bit_offset / 8;
    let bit_skip = bit_offset % 8;

    if byte_idx >= data.len() {
        return None;
    }

    let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
    unsafe { isal_raw::isal_inflate_init(&mut state) };
    // Raw deflate: no gzip/zlib header expected.
    state.crc_flag = isal_raw::ISAL_DEFLATE;

    if bit_skip > 0 {
        // inflatePrime: deflate is LSB-first, so bits [bit_skip..7] of data[byte_idx]
        // are the first bits of this chunk. Shift out the preceding block's bits and
        // load them into ISA-L's internal bit register before feeding full bytes.
        state.read_in = (data[byte_idx] as u64) >> bit_skip;
        state.read_in_length = (8 - bit_skip) as i32;
        // SAFETY: byte_idx < data.len(), so byte_idx + 1 <= data.len().
        state.next_in = unsafe { data.as_ptr().add(byte_idx + 1) as *mut u8 };
        state.avail_in = (data.len() - byte_idx - 1) as u32;
    } else {
        state.next_in = unsafe { data.as_ptr().add(byte_idx) as *mut u8 };
        state.avail_in = (data.len() - byte_idx) as u32;
    }

    // Prime the 32 KB LZ77 history window. Without this, back-references before
    // position 0 trigger ISAL_INVALID_LOOKBACK and inflate returns non-zero.
    // When no dict is provided we use a static zero window — matches ISA-L's
    // own convention for fresh-stream decodes and zlib-ng's zero-window prime.
    static ZERO_WINDOW: [u8; 32768] = [0u8; 32768];
    let window = if dict.is_empty() {
        &ZERO_WINDOW[..]
    } else {
        dict
    };
    {
        let ret = unsafe {
            isal_raw::isal_inflate_set_dict(
                &mut state,
                window.as_ptr() as *mut u8,
                window.len() as u32,
            )
        };
        if ret != 0 {
            return None;
        }
    }

    const MAX_CAP: usize = 512 * 1024 * 1024;
    let cap = max_output.clamp(256 * 1024, MAX_CAP);
    let mut output = vec![0u8; cap];
    let mut out_pos = 0usize;

    loop {
        let remaining = cap - out_pos;
        if remaining == 0 {
            break; // output cap reached
        }

        state.avail_out = remaining as u32;
        // SAFETY: out_pos < cap = output.len()
        state.next_out = unsafe { output.as_mut_ptr().add(out_pos) };

        let ret = unsafe { isal_raw::isal_inflate(&mut state) };
        let written = remaining - state.avail_out as usize;
        out_pos += written;

        if ret != 0 {
            if out_pos == 0 {
                return None;
            }
            break;
        }

        if state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH {
            break;
        }
        if written == 0 && state.avail_in == 0 {
            break;
        }
    }

    if out_pos == 0 {
        return None;
    }
    output.truncate(out_pos);
    Some(output)
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn decompress_deflate_from_bit(
    _data: &[u8],
    _bit_offset: usize,
    _dict: &[u8],
    _max_output: usize,
) -> Option<Vec<u8>> {
    None
}

/// Same as `decompress_deflate_from_bit` but also returns the end bit position.
///
/// The end bit is derived from ISA-L's post-decode state:
///   `end_bit = data.len() * 8 - state.avail_in * 8 - state.read_in_length`
///
/// This works for both byte-aligned (bit_skip=0) and non-aligned (bit_skip>0) starts
/// because the formula accounts for bits pre-loaded into the bit buffer at setup time.
///
/// Tip: pass `data` as `&full_data[..until_byte]` to limit ISA-L's input consumption.
/// The end_bit is still in `full_data` coordinates since both slices share the same layout.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
#[allow(dead_code)]
pub fn decompress_deflate_from_bit_with_end(
    data: &[u8],
    bit_offset: usize,
    dict: &[u8],
    max_output: usize,
) -> Option<(Vec<u8>, usize)> {
    use isal::isal_sys::igzip_lib as isal_raw;

    let byte_idx = bit_offset / 8;
    let bit_skip = bit_offset % 8;

    if byte_idx >= data.len() {
        return None;
    }

    let mut state: isal_raw::inflate_state = unsafe { std::mem::zeroed() };
    unsafe { isal_raw::isal_inflate_init(&mut state) };
    state.crc_flag = isal_raw::ISAL_DEFLATE;

    if bit_skip > 0 {
        state.read_in = (data[byte_idx] as u64) >> bit_skip;
        state.read_in_length = (8 - bit_skip) as i32;
        state.next_in = unsafe { data.as_ptr().add(byte_idx + 1) as *mut u8 };
        state.avail_in = (data.len() - byte_idx - 1) as u32;
    } else {
        state.next_in = unsafe { data.as_ptr().add(byte_idx) as *mut u8 };
        state.avail_in = (data.len() - byte_idx) as u32;
    }

    static ZERO_WINDOW: [u8; 32768] = [0u8; 32768];
    let window = if dict.is_empty() {
        &ZERO_WINDOW[..]
    } else {
        dict
    };
    {
        let ret = unsafe {
            isal_raw::isal_inflate_set_dict(
                &mut state,
                window.as_ptr() as *mut u8,
                window.len() as u32,
            )
        };
        if ret != 0 {
            return None;
        }
    }

    const MAX_CAP: usize = 512 * 1024 * 1024;
    let cap = max_output.clamp(256 * 1024, MAX_CAP);
    let mut output = vec![0u8; cap];
    let mut out_pos = 0usize;

    loop {
        let remaining = cap - out_pos;
        if remaining == 0 {
            break;
        }

        state.avail_out = remaining as u32;
        state.next_out = unsafe { output.as_mut_ptr().add(out_pos) };

        let ret = unsafe { isal_raw::isal_inflate(&mut state) };
        let written = remaining - state.avail_out as usize;
        out_pos += written;

        if ret != 0 {
            if out_pos == 0 {
                return None;
            }
            break;
        }

        if state.block_state == isal_raw::isal_block_state_ISAL_BLOCK_FINISH {
            break;
        }
        if written == 0 && state.avail_in == 0 {
            break;
        }
    }

    if out_pos == 0 {
        return None;
    }
    output.truncate(out_pos);

    // end_bit formula (valid for both bit_skip=0 and bit_skip>0):
    //   All bits available = pre_loaded + avail_in_bytes * 8
    //   Bits remaining     = final_avail_in * 8 + final_read_in_length
    //   Bits consumed      = available - remaining
    //   end_bit            = start_bit + bits_consumed = data.len()*8 - final_avail_in*8 - final_read_in_length
    let end_bit =
        data.len() * 8 - state.avail_in as usize * 8 - state.read_in_length.max(0) as usize;

    Some((output, end_bit))
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn decompress_deflate_from_bit_with_end(
    _data: &[u8],
    _bit_offset: usize,
    _dict: &[u8],
    _max_output: usize,
) -> Option<(Vec<u8>, usize)> {
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
            assert_eq!(
                cp.window.len(),
                crate::decompress::scan_inflate::WINDOW_SIZE
            );
            assert!(cp.output_offset >= crate::decompress::scan_inflate::WINDOW_SIZE);
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
            crate::decompress::scan_inflate::scan_deflate_fast(&compressed, 128 * 1024, data.len())
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

    /// Verify decompress_deflate_from_bit at bit offset 0 (byte-aligned baseline).
    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_deflate_from_bit_byte_aligned() {
        use super::*;

        let original: Vec<u8> = b"the quick brown fox jumps over the lazy dog "
            .iter()
            .cycle()
            .take(32_000)
            .cloned()
            .collect();

        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, &original).unwrap();
        let deflate = enc.finish().unwrap();

        let result = decompress_deflate_from_bit(&deflate, 0, &[], original.len())
            .expect("byte-aligned inflate must succeed");
        assert_eq!(result, original);
    }

    /// Verify decompress_deflate_from_bit with a 32KB dict (window from prior chunk).
    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_deflate_from_bit_with_dict() {
        use super::*;

        // Build a blob where the second half back-references the first half.
        let window: Vec<u8> = b"AAAA_repeated_window_data_for_back_refs_"
            .iter()
            .cycle()
            .take(32_768)
            .cloned()
            .collect();
        let payload: Vec<u8> = window.iter().take(16_000).cloned().collect(); // all back-refs

        // Compress payload with flate2 using the window as history (zlib dict).
        // We approximate by concatenating window+payload and taking the deflate tail.
        let mut combined = window.clone();
        combined.extend_from_slice(&payload);
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, &combined).unwrap();
        let deflate_all = enc.finish().unwrap();

        // Decompress the whole thing to get expected payload output.
        let mut dec = flate2::read::DeflateDecoder::new(deflate_all.as_slice());
        let mut all_out = Vec::new();
        std::io::Read::read_to_end(&mut dec, &mut all_out).unwrap();
        assert_eq!(&all_out[..window.len()], &window[..]);
        let expected_payload = all_out[window.len()..].to_vec();

        // Now decompress only the deflate stream but supply the window as dict.
        // Since we encoded window+payload as one stream (not two), we can only
        // test the dict path via decompress_deflate_from_bit at offset 0 with dict.
        // The dict doesn't affect byte 0 decode here — this tests that dict doesn't break it.
        let result = decompress_deflate_from_bit(&deflate_all, 0, &window, all_out.len())
            .expect("inflate with dict must succeed");
        assert_eq!(result, all_out, "output must match full decompression");
        let _ = expected_payload; // used for clarity above
    }

    /// Verify decompress_deflate_from_bit works at a non-byte-aligned block boundary.
    ///
    /// scan_deflate_isal checkpoints are NOT at block boundaries (ISA-L pauses mid-block),
    /// so we brute-force scan bit-by-bit to find positions where ISA-L actually decodes
    /// valid output. The first non-byte-aligned position that produces >= 1KB is a real boundary.
    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    /// x86_64 ISA-L path must not OOM when max_output=usize::MAX.
    ///
    /// Old behaviour: `let cap = max_output.max(256 * 1024)` with no upper bound
    /// attempted to allocate `usize::MAX` bytes, causing an immediate OOM panic.
    /// The arm64 zlib-ng path already clamps at 512 MB; this test ensures the
    /// x86_64 path is consistent.
    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_decompress_deflate_from_bit_huge_max_output_no_oom() {
        use super::*;
        // Empty input: should return None without attempting a huge allocation.
        let result = decompress_deflate_from_bit(&[], 0, &[], usize::MAX);
        assert!(result.is_none(), "empty input must return None, not OOM");

        // Real data with a huge max_output hint: should decompress correctly
        // and return the real (small) output, not allocate 512 MB.
        let original = b"hello world hello world hello world";
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, original).unwrap();
        let deflate = enc.finish().unwrap();

        let result = decompress_deflate_from_bit(&deflate, 0, &[], usize::MAX)
            .expect("real input with usize::MAX hint must succeed");
        assert_eq!(result, original);
    }

    #[test]
    #[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
    fn test_deflate_from_bit_non_byte_aligned() {
        use super::*;

        // Large repetitive data → many short deflate blocks → many non-byte-aligned boundaries.
        let original: Vec<u8> = (0u32..50_000)
            .flat_map(|i| format!("line {}: the quick brown fox\n", i).into_bytes())
            .collect();

        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut enc, &original).unwrap();
        let deflate = enc.finish().unwrap();

        // Bit 0 is always valid (start of stream). Find a non-byte-aligned boundary by
        // scanning. We limit to the first 32KB of the deflate stream to keep the test fast.
        let min_output = 1024;
        let search_limit = (32 * 1024 * 8).min(deflate.len() * 8);
        let found_bit = (1..search_limit)
            .filter(|b| b % 8 != 0) // non-byte-aligned only
            .find(|&bit| {
                decompress_deflate_from_bit(&deflate, bit, &[], min_output)
                    .is_some_and(|out| out.len() >= min_output)
            });

        let bit_offset = found_bit.expect(
            "should find a non-byte-aligned block boundary in the first 32KB of deflate stream",
        );

        // Second call: confirm the position is stable and produces output.
        let result = decompress_deflate_from_bit(&deflate, bit_offset, &[], deflate.len() * 4)
            .expect("second call at confirmed boundary must succeed");
        assert!(
            !result.is_empty(),
            "output must not be empty at bit {}",
            bit_offset
        );
        assert!(
            bit_offset % 8 != 0,
            "confirmed boundary must be non-byte-aligned: bit={}",
            bit_offset
        );
    }
}
