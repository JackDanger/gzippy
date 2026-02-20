//! Scan Inflate - Block-boundary-aware deflate decoder for two-pass parallel.
//!
//! Decodes a deflate stream block-by-block using the proven production decode
//! path, recording checkpoints (input bit position + 32KB window) at regular
//! intervals. These checkpoints enable the second pass to re-decode chunks
//! in parallel with dictionaries.
//!
//! Current implementation uses a full output buffer for correctness.
//! Future optimization: circular-buffer decode that fits in L1 cache.

use crate::consume_first_decode::Bits;
use std::io::{Error, ErrorKind, Result};

const WINDOW_SIZE: usize = 32768;

/// A checkpoint recorded at a deflate block boundary during the scan pass.
#[derive(Clone)]
pub struct ScanCheckpoint {
    /// Byte position in the input (deflate) data at this block boundary.
    pub input_byte_pos: usize,
    /// Bit buffer state at this block boundary.
    pub bitbuf: u64,
    pub bitsleft: u32,
    /// Total decompressed bytes at this point.
    pub output_offset: usize,
    /// 32KB window snapshot (the last 32KB of decompressed output).
    pub window: Vec<u8>,
}

/// Result of the scan pass.
pub struct ScanResult {
    /// Checkpoints at block boundaries near desired split points.
    pub checkpoints: Vec<ScanCheckpoint>,
    /// Total decompressed size of the deflate stream.
    pub total_output_size: usize,
}

/// Scan a deflate stream, recording checkpoints at block boundaries.
///
/// Decodes block-by-block using the proven production decode path.
/// At block boundaries near each `checkpoint_interval` bytes of output,
/// snapshots the bit-reader state and the last 32KB of output (the window).
///
/// `expected_output_size`: hint for pre-allocation (0 = auto-estimate).
pub fn scan_deflate(
    deflate_data: &[u8],
    checkpoint_interval: usize,
    expected_output_size: usize,
) -> Result<ScanResult> {
    let mut bits = Bits::new(deflate_data);
    let mut checkpoints = Vec::new();
    let mut next_checkpoint_at = checkpoint_interval;

    let estimated_size = if expected_output_size > 0 {
        // Pre-allocate with some margin for safety
        expected_output_size + 256 * 1024
    } else {
        deflate_data.len().saturating_mul(32).max(1024 * 1024)
    };
    let mut output = vec![0u8; estimated_size];
    let mut out_pos = 0;

    loop {
        // Record checkpoint at block boundary if we've crossed an interval
        if out_pos >= next_checkpoint_at && out_pos >= WINDOW_SIZE {
            let window_start = out_pos - WINDOW_SIZE;
            checkpoints.push(ScanCheckpoint {
                input_byte_pos: bits.pos,
                bitbuf: bits.bitbuf,
                bitsleft: bits.bitsleft,
                output_offset: out_pos,
                window: output[window_start..out_pos].to_vec(),
            });
            next_checkpoint_at = out_pos + checkpoint_interval;
        }

        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        // Ensure enough buffer for the next block. Deflate blocks can produce
        // arbitrarily large output (e.g., highly compressible data), so keep
        // a generous margin. The decode functions take &mut [u8], so we must
        // pre-allocate before calling them.
        if out_pos + 4 * 1024 * 1024 > output.len() {
            output.resize((output.len() * 2).max(out_pos + 8 * 1024 * 1024), 0);
        }

        match btype {
            0 => {
                out_pos =
                    crate::consume_first_decode::decode_stored_pub(&mut bits, &mut output, out_pos)?
            }
            1 => {
                out_pos =
                    crate::consume_first_decode::decode_fixed_pub(&mut bits, &mut output, out_pos)?
            }
            2 => {
                out_pos = crate::consume_first_decode::decode_dynamic_pub(
                    &mut bits,
                    &mut output,
                    out_pos,
                )?
            }
            3 => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type")),
            _ => unreachable!(),
        }

        if bfinal {
            break;
        }
    }

    Ok(ScanResult {
        checkpoints,
        total_output_size: out_pos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_inflate_basic() {
        let original = b"Hello, world! This is a test of the scan inflate module.";
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = scan_deflate(&compressed, 1024, 0).unwrap();
        assert_eq!(result.total_output_size, original.len());
    }

    #[test]
    fn test_scan_inflate_large() {
        let mut original = Vec::new();
        for i in 0..100_000 {
            original.extend_from_slice(format!("Line {}: some test data\n", i).as_bytes());
        }

        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Verify reference decoder works
        let mut ref_output = vec![0u8; original.len() + 1024];
        let ref_size =
            crate::consume_first_decode::inflate_consume_first(&compressed, &mut ref_output)
                .expect("inflate_consume_first should work");
        assert_eq!(ref_size, original.len(), "Reference inflate size mismatch");

        let result = scan_deflate(&compressed, 256 * 1024, original.len()).unwrap();
        assert_eq!(result.total_output_size, original.len());
        assert!(!result.checkpoints.is_empty());

        // Verify each checkpoint has a valid 32KB window
        for cp in &result.checkpoints {
            assert_eq!(cp.window.len(), WINDOW_SIZE);
            assert!(cp.output_offset >= WINDOW_SIZE);
            assert!(cp.input_byte_pos <= compressed.len());
        }
    }

    #[test]
    fn test_scan_checkpoint_windows_match_output() {
        let mut original = Vec::new();
        for i in 0..200_000 {
            original.extend_from_slice(format!("Data item {}: payload\n", i).as_bytes());
        }

        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = scan_deflate(&compressed, 128 * 1024, original.len()).unwrap();

        // Verify windows match what a full decode produces
        let mut full_output = vec![0u8; original.len() + 1024];
        let full_size =
            crate::consume_first_decode::inflate_consume_first(&compressed, &mut full_output)
                .unwrap();
        assert_eq!(full_size, result.total_output_size);

        for cp in &result.checkpoints {
            let window_start = cp.output_offset - WINDOW_SIZE;
            let expected_window = &full_output[window_start..cp.output_offset];
            assert_eq!(
                cp.window.as_slice(),
                expected_window,
                "Window mismatch at offset {}",
                cp.output_offset
            );
        }
    }
}
