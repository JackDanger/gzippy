#![allow(dead_code)]
//! Scan Inflate - Block-boundary-aware deflate decoder for two-pass parallel.
//!
//! Decodes a deflate stream block-by-block using the proven production decode
//! path, recording checkpoints (input bit position + 32KB window) at regular
//! intervals. These checkpoints enable the second pass to re-decode chunks
//! in parallel with dictionaries.
//!
//! Current implementation uses a full output buffer for correctness.
//! Future optimization: circular-buffer decode that fits in L1 cache.

use crate::decompress::inflate::consume_first_decode::Bits;
use std::io::{Error, ErrorKind, Result};

pub const WINDOW_SIZE: usize = 32768;

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
                out_pos = crate::decompress::inflate::consume_first_decode::decode_stored_pub(
                    &mut bits,
                    &mut output,
                    out_pos,
                )?
            }
            1 => {
                out_pos = crate::decompress::inflate::consume_first_decode::decode_fixed_pub(
                    &mut bits,
                    &mut output,
                    out_pos,
                )?
            }
            2 => {
                out_pos = crate::decompress::inflate::consume_first_decode::decode_dynamic_pub(
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

/// Fast scan: small circular buffer that stays in L2 cache.
///
/// Instead of allocating a full output buffer (211MB → page faults, cache misses),
/// uses a 4MB buffer and resets at block boundaries, keeping only the last 32KB
/// (the LZ77 window). This makes all output writes hit L2 cache, ~1.5-2x faster
/// than full-output decode.
pub fn scan_deflate_fast(
    deflate_data: &[u8],
    checkpoint_interval: usize,
    _expected_output_size: usize,
) -> Result<ScanResult> {
    // 8MB buffer: large enough for any single deflate block, small enough
    // that the hot working set (after reset) fits in L2/L3 cache.
    const INITIAL_BUF: usize = 8 * 1024 * 1024;
    const RESET_THRESHOLD: usize = 4 * 1024 * 1024;

    let mut bits = Bits::new(deflate_data);
    let mut checkpoints = Vec::new();
    let mut next_checkpoint_at = checkpoint_interval;

    let mut output = vec![0u8; INITIAL_BUF];
    let mut out_pos: usize = 0;
    let mut virtual_pos: usize = 0;

    loop {
        // Reset buffer between blocks to keep working set in cache.
        // Preserve the last 32KB (LZ77 window) for match distances.
        if out_pos > RESET_THRESHOLD {
            let keep = WINDOW_SIZE.min(out_pos);
            output.copy_within(out_pos - keep..out_pos, 0);
            out_pos = keep;
        }

        // Record checkpoint at block boundary when we cross an interval
        if virtual_pos >= next_checkpoint_at && virtual_pos >= WINDOW_SIZE {
            let window_start = out_pos.saturating_sub(WINDOW_SIZE);
            let window_end = out_pos;
            checkpoints.push(ScanCheckpoint {
                input_byte_pos: bits.pos,
                bitbuf: bits.bitbuf,
                bitsleft: bits.bitsleft,
                output_offset: virtual_pos,
                window: output[window_start..window_end].to_vec(),
            });
            next_checkpoint_at = virtual_pos + checkpoint_interval;
        }

        // Ensure enough buffer for the next block
        if out_pos + 4 * 1024 * 1024 > output.len() {
            output.resize((output.len() * 2).max(out_pos + 8 * 1024 * 1024), 0);
        }

        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        let old_out_pos = out_pos;
        match btype {
            0 => {
                out_pos = crate::decompress::inflate::consume_first_decode::decode_stored_pub(
                    &mut bits,
                    &mut output,
                    out_pos,
                )?
            }
            1 => {
                out_pos = crate::decompress::inflate::consume_first_decode::decode_fixed_pub(
                    &mut bits,
                    &mut output,
                    out_pos,
                )?
            }
            2 => {
                out_pos = crate::decompress::inflate::consume_first_decode::decode_dynamic_pub(
                    &mut bits,
                    &mut output,
                    out_pos,
                )?
            }
            3 => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type")),
            _ => unreachable!(),
        }

        virtual_pos += out_pos - old_out_pos;

        if bfinal {
            break;
        }
    }

    Ok(ScanResult {
        checkpoints,
        total_output_size: virtual_pos,
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
        let ref_size = crate::decompress::inflate::consume_first_decode::inflate_consume_first(
            &compressed,
            &mut ref_output,
        )
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
    fn test_scan_fast_basic() {
        let original = b"Hello, world! This is a test of the fast scan inflate module.";
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = scan_deflate_fast(&compressed, 1024, 0).unwrap();
        assert_eq!(result.total_output_size, original.len());
    }

    #[test]
    fn test_scan_fast_large() {
        let mut original = Vec::new();
        for i in 0..100_000 {
            original.extend_from_slice(format!("Line {}: some test data\n", i).as_bytes());
        }

        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let result = scan_deflate_fast(&compressed, 256 * 1024, original.len()).unwrap();
        assert_eq!(result.total_output_size, original.len());
        assert!(!result.checkpoints.is_empty());

        for cp in &result.checkpoints {
            assert_eq!(cp.window.len(), WINDOW_SIZE);
            assert!(cp.output_offset >= WINDOW_SIZE);
            assert!(cp.input_byte_pos <= compressed.len());
        }
    }

    #[test]
    fn test_scan_fast_matches_full_scan() {
        let mut original = Vec::new();
        for i in 0..200_000 {
            original.extend_from_slice(format!("Data item {}: payload\n", i).as_bytes());
        }

        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let full = scan_deflate(&compressed, 128 * 1024, original.len()).unwrap();
        let fast = scan_deflate_fast(&compressed, 128 * 1024, original.len()).unwrap();

        assert_eq!(full.total_output_size, fast.total_output_size);
        assert_eq!(full.checkpoints.len(), fast.checkpoints.len());

        for (f, s) in full.checkpoints.iter().zip(fast.checkpoints.iter()) {
            assert_eq!(f.input_byte_pos, s.input_byte_pos);
            assert_eq!(f.output_offset, s.output_offset);
            assert_eq!(
                f.window, s.window,
                "Window mismatch at offset {}",
                f.output_offset
            );
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
        let full_size = crate::decompress::inflate::consume_first_decode::inflate_consume_first(
            &compressed,
            &mut full_output,
        )
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

    /// Head-to-head: scan_deflate_fast vs sequential libdeflate on silesia.
    /// This answers: "Is scan cheap enough to make two-pass parallel viable?"
    ///
    /// If scan_ratio > 0.8 → two-pass is hopeless (scan ≈ sequential)
    /// If scan_ratio ≈ 0.5 → two-pass breaks even at 4 threads
    /// If scan_ratio < 0.3 → two-pass wins at 4 threads
    #[test]
    fn bench_scan_vs_sequential_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping (silesia not found)");
                return;
            }
        };

        let header_size = crate::decompress::parallel::marker_decode::skip_gzip_header(&gz)
            .expect("valid header");
        let deflate = &gz[header_size..gz.len() - 8];
        let isize_val = u32::from_le_bytes([
            gz[gz.len() - 4],
            gz[gz.len() - 3],
            gz[gz.len() - 2],
            gz[gz.len() - 1],
        ]) as usize;

        // Warmup
        let _ = scan_deflate_fast(deflate, isize_val / 8, isize_val);
        let mut out = vec![0u8; isize_val + 256 * 1024];
        let _ = crate::decompress::inflate::consume_first_decode::inflate_consume_first(
            deflate, &mut out,
        );

        // Measure scan_deflate_fast (5 trials)
        let mut scan_times = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let r = scan_deflate_fast(deflate, isize_val / 8, isize_val).unwrap();
            scan_times.push(t.elapsed());
            assert_eq!(r.total_output_size, isize_val);
        }

        // Measure sequential inflate_consume_first (5 trials)
        let mut seq_times = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let sz = crate::decompress::inflate::consume_first_decode::inflate_consume_first(
                deflate, &mut out,
            )
            .unwrap();
            seq_times.push(t.elapsed());
            assert_eq!(sz, isize_val);
        }

        // Measure sequential libdeflate FFI (5 trials)
        let mut ffi_times = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let mut ffi_out = Vec::new();
            crate::decompress::decompress_single_member_libdeflate(&gz, &mut ffi_out).unwrap();
            ffi_times.push(t.elapsed());
            assert_eq!(ffi_out.len(), isize_val);
        }

        let scan_median = median_duration(&scan_times);
        let seq_median = median_duration(&seq_times);
        let ffi_median = median_duration(&ffi_times);

        let scan_mbps = isize_val as f64 / scan_median.as_secs_f64() / 1e6;
        let seq_mbps = isize_val as f64 / seq_median.as_secs_f64() / 1e6;
        let ffi_mbps = isize_val as f64 / ffi_median.as_secs_f64() / 1e6;

        let scan_vs_seq = scan_median.as_secs_f64() / seq_median.as_secs_f64();
        let scan_vs_ffi = scan_median.as_secs_f64() / ffi_median.as_secs_f64();

        eprintln!(
            "=== Scan vs Sequential on silesia ({:.1} MB) ===",
            isize_val as f64 / 1e6
        );
        eprintln!(
            "  scan_deflate_fast:    {:.1} MB/s ({:.1}ms)",
            scan_mbps,
            scan_median.as_secs_f64() * 1000.0
        );
        eprintln!(
            "  inflate_consume_first:{:.1} MB/s ({:.1}ms)",
            seq_mbps,
            seq_median.as_secs_f64() * 1000.0
        );
        eprintln!(
            "  libdeflate FFI:       {:.1} MB/s ({:.1}ms)",
            ffi_mbps,
            ffi_median.as_secs_f64() * 1000.0
        );
        eprintln!("  scan/sequential ratio: {:.2}x", scan_vs_seq);
        eprintln!("  scan/FFI ratio:        {:.2}x", scan_vs_ffi);
        eprintln!();

        // Two-pass viability math (4 threads):
        // total = scan_time + sequential_time / N
        // speedup = sequential_time / total = 1 / (scan_ratio + 1/N)
        for n in [2, 4, 8] {
            let two_pass_time = scan_median.as_secs_f64() + ffi_median.as_secs_f64() / n as f64;
            let speedup = ffi_median.as_secs_f64() / two_pass_time;
            let two_pass_mbps = isize_val as f64 / two_pass_time / 1e6;
            eprintln!(
                "  Two-pass T{}: {:.1} MB/s ({:.2}x vs FFI sequential) {}",
                n,
                two_pass_mbps,
                speedup,
                if speedup > 1.0 {
                    "← FASTER"
                } else {
                    "← SLOWER"
                }
            );
        }
    }

    fn median_duration(times: &[std::time::Duration]) -> std::time::Duration {
        let mut sorted: Vec<_> = times.to_vec();
        sorted.sort();
        sorted[sorted.len() / 2]
    }
}
