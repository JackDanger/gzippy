//! Route C testbed harness — extract per-block test cases from a corpus
//! and provide a diff API so every incremental dynasm-emit commit can
//! verify byte-perfect output against an oracle.
//!
//! Foundation primitive. Compounds every future Route C v3+ commit:
//! - Add a new asm-emit tweak → run testbed → see exactly which block
//!   indices diverge.
//! - Add a new block-type code path → testbed reports per-block
//!   pass/fail count.
//! - Track per-block timing → identify the slow path's hottest fingerprint.
//!
//! ## Design
//!
//! Test cases are produced from a real gzip corpus (silesia, logs, etc.)
//! by:
//! 1. Walking the gzip stream with `gzippy::decompress::block_walker`
//!    to get per-block (start_bit, end_bit, decoded_bytes) metadata.
//! 2. Decoding the full stream via the oracle (flate2) to get the
//!    full expected output.
//! 3. Slicing the oracle output per block (cumulative decoded_bytes
//!    gives the per-block output range).
//!
//! Each `TestCase` is then independently runnable: feed the gzip bytes
//! + start_bit to any decoder; compare its output to the slice.

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// CPU-capability requirements a decoder declares for itself. Tests
/// auto-skip when the host doesn't satisfy `requires`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArchCaps {
    pub requires_bmi2: bool,
    pub requires_avx2: bool,
    pub requires_avx512: bool,
    pub requires_neon: bool,
}

impl ArchCaps {
    /// Is the current process host's CPU compatible with this cap set?
    pub fn host_supports(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            if self.requires_bmi2 && !std::is_x86_feature_detected!("bmi2") {
                return false;
            }
            if self.requires_avx2 && !std::is_x86_feature_detected!("avx2") {
                return false;
            }
            if self.requires_avx512 && !std::is_x86_feature_detected!("avx512f") {
                return false;
            }
            if self.requires_neon {
                return false; // can't have NEON on x86_64
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.requires_bmi2 || self.requires_avx2 || self.requires_avx512 {
                return false; // no x86_64 features on aarch64
            }
            if self.requires_neon && !std::arch::is_aarch64_feature_detected!("neon") {
                return false;
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            if self.requires_bmi2
                || self.requires_avx2
                || self.requires_avx512
                || self.requires_neon
            {
                return false;
            }
        }
        true
    }
}

/// One block extracted from a corpus.
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Block index in the source file (0-based, in stream order).
    pub block_index: usize,
    /// Bit offset (in the deflate body, NOT the gzip header) where
    /// this block's 3-bit header starts.
    pub start_bit_in_deflate: u64,
    /// Bit offset of the END of this block.
    pub end_bit_in_deflate: u64,
    /// BTYPE: 0=stored, 1=fixed-Huffman, 2=dynamic-Huffman.
    pub btype: u8,
    /// Fingerprint for dynamic-Huffman blocks (None for stored/fixed).
    pub fingerprint: Option<u64>,
    /// Decoded bytes this block emits.
    pub expected_output: Vec<u8>,
}

/// Comparison result for one decoder run against one test case.
#[derive(Debug, Clone)]
pub struct DiffResult {
    pub case_index: usize,
    pub btype: u8,
    pub fingerprint: Option<u64>,
    pub bytes_match: bool,
    pub end_bit_match: bool,
    /// First byte offset where decoder output differs from expected.
    /// `None` if bytes match exactly.
    pub first_diff_byte: Option<usize>,
    /// Wall-time of the decoder call in nanoseconds.
    pub decode_ns: u64,
}

/// Aggregate report across N runs.
#[derive(Debug, Clone, Default)]
pub struct TestbedReport {
    pub total_cases: usize,
    pub passed: usize,
    pub failed: usize,
    pub total_decode_ns: u64,
    pub total_bytes: u64,
    pub failures: Vec<DiffResult>,
}

impl TestbedReport {
    pub fn pass_rate(&self) -> f64 {
        if self.total_cases == 0 {
            0.0
        } else {
            self.passed as f64 / self.total_cases as f64
        }
    }

    pub fn throughput_mbps(&self) -> f64 {
        if self.total_decode_ns == 0 {
            0.0
        } else {
            (self.total_bytes as f64) / (self.total_decode_ns as f64) * 1000.0
        }
    }
}

#[cfg(feature = "std")]
mod corpus {
    use super::*;
    use libdeflater::Decompressor;

    /// Extract test cases from a gzip file's bytes.
    ///
    /// Uses libdeflater (already a sub-crate dependency) as the oracle
    /// to materialize per-block expected output by slicing the full
    /// decoded stream at cumulative decoded_bytes boundaries.
    pub fn extract_cases(
        gz: &[u8],
        _deflate_body_offset: usize,
        blocks: &[BlockSummary],
    ) -> std::io::Result<Vec<TestCase>> {
        // ISIZE-sized buffer; libdeflater needs an exact-or-larger hint.
        let isize_field = u32::from_le_bytes([
            gz[gz.len() - 4],
            gz[gz.len() - 3],
            gz[gz.len() - 2],
            gz[gz.len() - 1],
        ]) as usize;
        let mut oracle = vec![0u8; isize_field.max(1024)];
        let mut dec = Decompressor::new();
        let n = loop {
            match dec.gzip_decompress(gz, &mut oracle) {
                Ok(n) => break n,
                Err(libdeflater::DecompressionError::InsufficientSpace) => {
                    oracle.resize(oracle.len() * 2, 0);
                }
                Err(libdeflater::DecompressionError::BadData) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "oracle decode failed",
                    ));
                }
            }
        };
        oracle.truncate(n);

        let mut cases = Vec::with_capacity(blocks.len());
        let mut cum_bytes = 0usize;
        for (i, b) in blocks.iter().enumerate() {
            let end = cum_bytes + b.decoded_bytes as usize;
            if end > oracle.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("block {i} decoded_bytes overrun: {end} > {}", oracle.len()),
                ));
            }
            cases.push(TestCase {
                block_index: i,
                start_bit_in_deflate: b.start_bit,
                end_bit_in_deflate: b.end_bit,
                btype: b.btype,
                fingerprint: b.fingerprint,
                expected_output: oracle[cum_bytes..end].to_vec(),
            });
            cum_bytes = end;
        }
        Ok(cases)
    }
}

/// Per-block metadata the testbed accepts. Mirrors the shape of
/// `gzippy::decompress::block_walker::BlockMeta` but doesn't import
/// from the parent crate so this module compiles in `no_std`.
#[derive(Debug, Clone, Copy)]
pub struct BlockSummary {
    pub start_bit: u64,
    pub end_bit: u64,
    pub btype: u8,
    pub fingerprint: Option<u64>,
    pub decoded_bytes: u64,
}

#[cfg(feature = "std")]
pub use corpus::extract_cases;

/// Run a candidate decoder against a slice of test cases and produce
/// a report. The decoder receives (deflate body, start_bit, btype,
/// fingerprint, expected_output_capacity) and returns the produced
/// bytes + end_bit. Failures are appended to the report with
/// first-diff-byte location.
///
/// `caps` declares what CPU features the decoder requires. If the
/// host doesn't support them, `run_testbed` returns an empty report
/// without invoking the decoder (the Cargo target may be a CI runner
/// without AVX-512, etc).
#[cfg(feature = "std")]
pub fn run_testbed_with_caps<F>(
    cases: &[TestCase],
    deflate_body: &[u8],
    caps: ArchCaps,
    decoder: F,
) -> TestbedReport
where
    F: FnMut(&[u8], u64, u8, Option<u64>, usize) -> std::io::Result<(u64, Vec<u8>)>,
{
    if !caps.host_supports() {
        return TestbedReport {
            total_cases: 0,
            ..Default::default()
        };
    }
    run_testbed(cases, deflate_body, decoder)
}

/// Run a candidate decoder against a slice of test cases. No cap
/// gating (assumes the decoder is portable scalar).
#[cfg(feature = "std")]
pub fn run_testbed<F>(cases: &[TestCase], deflate_body: &[u8], mut decoder: F) -> TestbedReport
where
    F: FnMut(&[u8], u64, u8, Option<u64>, usize) -> std::io::Result<(u64, Vec<u8>)>,
{
    let mut report = TestbedReport {
        total_cases: cases.len(),
        ..Default::default()
    };
    for (idx, case) in cases.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let r = decoder(
            deflate_body,
            case.start_bit_in_deflate,
            case.btype,
            case.fingerprint,
            case.expected_output.len(),
        );
        let decode_ns = t0.elapsed().as_nanos() as u64;
        report.total_decode_ns += decode_ns;
        report.total_bytes += case.expected_output.len() as u64;

        match r {
            Ok((end_bit, output)) => {
                let bytes_match = output == case.expected_output;
                let end_bit_match = end_bit == case.end_bit_in_deflate;
                if bytes_match && end_bit_match {
                    report.passed += 1;
                } else {
                    report.failed += 1;
                    let first_diff = output
                        .iter()
                        .zip(case.expected_output.iter())
                        .position(|(a, b)| a != b);
                    report.failures.push(DiffResult {
                        case_index: idx,
                        btype: case.btype,
                        fingerprint: case.fingerprint,
                        bytes_match,
                        end_bit_match,
                        first_diff_byte: first_diff,
                        decode_ns,
                    });
                }
            }
            Err(_) => {
                report.failed += 1;
                report.failures.push(DiffResult {
                    case_index: idx,
                    btype: case.btype,
                    fingerprint: case.fingerprint,
                    bytes_match: false,
                    end_bit_match: false,
                    first_diff_byte: None,
                    decode_ns,
                });
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: empty test-case list produces an empty report.
    #[test]
    fn empty_testbed_report() {
        let cases: Vec<TestCase> = Vec::new();
        let deflate = b"";
        let report = run_testbed(&cases, deflate, |_, start, _, _, _| Ok((start, Vec::new())));
        assert_eq!(report.total_cases, 0);
        assert_eq!(report.pass_rate(), 0.0);
    }

    /// An always-correct decoder passes every case.
    #[test]
    fn always_correct_decoder_passes() {
        let cases = vec![
            TestCase {
                block_index: 0,
                start_bit_in_deflate: 0,
                end_bit_in_deflate: 8,
                btype: 1,
                fingerprint: None,
                expected_output: vec![1, 2, 3],
            },
            TestCase {
                block_index: 1,
                start_bit_in_deflate: 8,
                end_bit_in_deflate: 16,
                btype: 2,
                fingerprint: Some(0xdead_beef),
                expected_output: vec![4, 5],
            },
        ];
        let deflate = b"";
        let report = run_testbed(&cases, deflate, |_, _, _, _, _| Ok((8, vec![1, 2, 3])));
        assert_eq!(report.total_cases, 2);
        assert_eq!(report.passed, 1, "first case passes (expects 1,2,3)");
        assert_eq!(report.failed, 1, "second case fails (expects 4,5)");
        assert_eq!(report.failures.len(), 1);
        let f = &report.failures[0];
        assert_eq!(f.case_index, 1);
        assert!(!f.bytes_match);
        assert!(!f.end_bit_match);
    }

    // `extract_cases` integration is exercised in the parent crate's
    // tests where flate2 is already a dependency.

    /// Throughput calc is sane.
    #[test]
    fn throughput_mbps_arithmetic() {
        let r = TestbedReport {
            total_cases: 1,
            passed: 1,
            failed: 0,
            total_decode_ns: 1_000_000, // 1ms
            total_bytes: 1_000_000,     // 1 MB
            failures: Vec::new(),
        };
        // 1MB in 1ms = 1000 MB/s
        assert!((r.throughput_mbps() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn pass_rate_zero_total_is_zero() {
        let r = TestbedReport::default();
        assert_eq!(r.pass_rate(), 0.0);
    }

    /// On every supported host, `ArchCaps::default()` (no requirements)
    /// passes `host_supports`.
    #[test]
    fn arch_caps_default_supports_any_host() {
        assert!(ArchCaps::default().host_supports());
    }

    /// AVX-512 requirement should fail on a typical Mac arm64 dev box
    /// but pass on a host that actually has AVX-512.
    #[test]
    fn arch_caps_avx512_gating_matches_host() {
        let caps = ArchCaps {
            requires_avx512: true,
            ..Default::default()
        };
        #[cfg(target_arch = "aarch64")]
        assert!(!caps.host_supports(), "aarch64 can't have AVX-512");
        #[cfg(target_arch = "x86_64")]
        {
            let expected = std::is_x86_feature_detected!("avx512f");
            assert_eq!(caps.host_supports(), expected);
        }
    }

    /// `run_testbed_with_caps` skips the decoder when caps unmet.
    #[test]
    fn run_testbed_skips_on_unsupported_caps() {
        let cases = vec![TestCase {
            block_index: 0,
            start_bit_in_deflate: 0,
            end_bit_in_deflate: 8,
            btype: 2,
            fingerprint: None,
            expected_output: vec![1, 2, 3],
        }];
        // Use a caps set we know fails on every host (NEON on x86_64
        // or BMI2 on aarch64 — pick whichever is foreign).
        #[cfg(target_arch = "aarch64")]
        let caps = ArchCaps {
            requires_bmi2: true,
            ..Default::default()
        };
        #[cfg(target_arch = "x86_64")]
        let caps = ArchCaps {
            requires_neon: true,
            ..Default::default()
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let caps = ArchCaps {
            requires_avx2: true,
            ..Default::default()
        };
        let mut called = false;
        let report = run_testbed_with_caps(&cases, b"", caps, |_, _, _, _, _| {
            called = true;
            Ok((0, vec![]))
        });
        assert!(!called, "decoder should not have been called");
        assert_eq!(report.total_cases, 0);
    }
}
