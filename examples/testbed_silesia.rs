//! Route C v3+ development driver — load a corpus, run a candidate
//! decoder via `gzippy_inflate::route_c_testbed`, report per-block
//! pass/fail + throughput.
//!
//! Today the candidate decoder is the parent crate's `decompress::
//! block_walker::decode_block_body` (Rust reference). When Route C
//! v3 lands, swap in the dynasm-emitted decoder and the diff API
//! reports which fingerprints diverge.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example testbed_silesia -- benchmark_data/silesia-gzip9.gz
//! ```
//!
//! ## Output
//!
//! ```
//! Loaded 3350 blocks from silesia-gzip9.gz (211968000 oracle bytes)
//! Testbed report (Rust reference decoder):
//!   total_cases:    3350
//!   passed:         3350
//!   failed:         0
//!   pass_rate:      100.00%
//!   throughput:     1234 MB/s
//! ```

use std::env;
use std::fs;
use std::process::ExitCode;

use gzippy::decompress::block_walker::{walk_block_boundaries, BlockMeta};
#[cfg(all(target_arch = "x86_64", feature = "route-c-dynasm"))]
use gzippy_inflate::route_c_dynamic::decode_dynamic_block_hybrid_with_window;
use gzippy_inflate::route_c_dynamic::{
    decode_dynamic_block_layered_with_window, parse_dynamic_header, BitReader, LayeredLut, LutRole,
};
use gzippy_inflate::route_c_testbed::{extract_cases, run_testbed, BlockSummary};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: testbed_silesia <file.gz>");
        return ExitCode::from(2);
    }
    let path = &args[0];

    let gz = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let blocks = match walk_block_boundaries(&gz) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error walking {path}: {e}");
            return ExitCode::from(2);
        }
    };

    // Convert BlockMeta -> BlockSummary (cross-crate boundary).
    let summaries: Vec<BlockSummary> = blocks
        .iter()
        .map(|b: &BlockMeta| BlockSummary {
            start_bit: b.start_bit,
            end_bit: b.end_bit,
            btype: b.btype,
            fingerprint: b.fingerprint,
            decoded_bytes: b.decoded_bytes,
        })
        .collect();

    // Deflate body starts after the gzip header.
    let deflate_offset = compute_deflate_offset(&gz);
    let deflate_body = &gz[deflate_offset..gz.len() - 8];

    let cases = match extract_cases(&gz, deflate_offset, &summaries) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error extracting test cases: {e}");
            return ExitCode::from(2);
        }
    };
    let oracle_bytes: u64 = cases.iter().map(|c| c.expected_output.len() as u64).sum();
    eprintln!(
        "Loaded {} blocks from {path} ({oracle_bytes} oracle bytes)",
        cases.len()
    );

    // Route C v3 baseline: Rust reference decoder from the sub-crate.
    // Each subsequent Route C v3+ commit will replace pieces of this
    // with dynasm-emitted asm and the testbed will report regressions
    // per case_index.
    // Scratch LUTs reused across all 3350 blocks. Saves ~50 MB of
    // allocator churn per silesia run.
    let mut lut_ll = LayeredLut::default();
    let mut lut_d = LayeredLut::default();

    // GZIPPY_TESTBED_DECODER=hybrid → use v3.7 asm+Rust hybrid.
    // Default = pure-Rust reference (decode_dynamic_block_layered_with_window).
    let use_hybrid = std::env::var("GZIPPY_TESTBED_DECODER")
        .map(|v| v == "hybrid")
        .unwrap_or(false);
    eprintln!(
        "Decoder: {}",
        if use_hybrid {
            "v3.7 asm+Rust hybrid"
        } else {
            "pure-Rust reference"
        }
    );

    let report = run_testbed(
        &cases,
        deflate_body,
        |deflate, start_bit, btype, _fp, expected_len, window| {
            let after_header = start_bit + 3;
            let mut out_buf = vec![0u8; expected_len];
            match btype {
                0 => {
                    let aligned = after_header.div_ceil(8) * 8;
                    let mut br = BitReader {
                        buf: deflate,
                        bit_pos: aligned,
                    };
                    let len = br.read(16) as usize;
                    let _nlen = br.read(16);
                    let byte = (br.bit_pos / 8) as usize;
                    if byte + len > deflate.len() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "stored block overruns deflate body",
                        ));
                    }
                    out_buf[..len].copy_from_slice(&deflate[byte..byte + len]);
                    Ok((br.bit_pos + (len as u64) * 8, out_buf[..len].to_vec()))
                }
                1 => {
                    let mut ll = [0u8; 288];
                    for e in ll.iter_mut().take(144) {
                        *e = 8;
                    }
                    for e in ll.iter_mut().take(256).skip(144) {
                        *e = 9;
                    }
                    for e in ll.iter_mut().take(280).skip(256) {
                        *e = 7;
                    }
                    for e in ll.iter_mut().take(288).skip(280) {
                        *e = 8;
                    }
                    let dl = [5u8; 32];
                    lut_ll.build_into_with_role(&ll, LutRole::Litlen);
                    lut_d.build_into_with_role(&dl, LutRole::Dist);
                    let (end_bit, bytes) = decode_dynamic_block_layered_with_window(
                        deflate,
                        after_header,
                        &lut_ll,
                        &lut_d,
                        &mut out_buf,
                        0,
                        window,
                    )?;
                    out_buf.truncate(bytes);
                    Ok((end_bit, out_buf))
                }
                2 => {
                    let (after_hdr, ll, dl) = parse_dynamic_header(deflate, after_header)?;
                    lut_ll.build_into_with_role(&ll, LutRole::Litlen);
                    lut_d.build_into_with_role(&dl, LutRole::Dist);
                    #[cfg(all(target_arch = "x86_64", feature = "route-c-dynasm"))]
                    let (end_bit, bytes) = if use_hybrid {
                        decode_dynamic_block_hybrid_with_window(
                            deflate,
                            after_hdr,
                            &lut_ll,
                            &lut_d,
                            &mut out_buf,
                            0,
                            window,
                        )?
                    } else {
                        decode_dynamic_block_layered_with_window(
                            deflate,
                            after_hdr,
                            &lut_ll,
                            &lut_d,
                            &mut out_buf,
                            0,
                            window,
                        )?
                    };
                    #[cfg(not(all(target_arch = "x86_64", feature = "route-c-dynasm")))]
                    let (end_bit, bytes) = decode_dynamic_block_layered_with_window(
                        deflate,
                        after_hdr,
                        &lut_ll,
                        &lut_d,
                        &mut out_buf,
                        0,
                        window,
                    )?;
                    out_buf.truncate(bytes);
                    Ok((end_bit, out_buf))
                }
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("reserved BTYPE={btype}"),
                )),
            }
        },
    );

    println!("Testbed report (Rust reference decoder — Route C v3 baseline):");
    println!("  total_cases:    {}", report.total_cases);
    println!("  passed:         {}", report.passed);
    println!("  failed:         {}", report.failed);
    println!("  pass_rate:      {:.2}%", report.pass_rate() * 100.0);
    println!("  total_bytes:    {}", report.total_bytes);
    println!("  decode_ns:      {}", report.total_decode_ns);
    if report.total_decode_ns > 0 {
        println!("  throughput:     {:.0} MB/s", report.throughput_mbps());
    }

    // First-5 failures triage.
    if !report.failures.is_empty() {
        eprintln!("\nFirst 5 failures:");
        for f in report.failures.iter().take(5) {
            eprintln!(
                "  case {} btype {} fp {:?} bytes_match {} end_bit_match {} first_diff {:?}",
                f.case_index,
                f.btype,
                f.fingerprint,
                f.bytes_match,
                f.end_bit_match,
                f.first_diff_byte
            );
        }
    }

    // Histogram of fingerprint frequency (top 10) — preview of where
    // Route C v3's per-block JIT would spend its build budget.
    let mut fp_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for c in &cases {
        if let Some(fp) = c.fingerprint {
            *fp_counts.entry(fp).or_insert(0) += 1;
        }
    }
    let mut sorted: Vec<_> = fp_counts.iter().collect();
    sorted.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
    eprintln!(
        "\nTop-10 fingerprint frequencies (out of {}):",
        fp_counts.len()
    );
    for (fp, n) in sorted.iter().take(10) {
        eprintln!("  {fp:#018x}  ×{n}");
    }

    ExitCode::SUCCESS
}

fn compute_deflate_offset(gz: &[u8]) -> usize {
    if gz.len() < 18 || gz[0] != 0x1f || gz[1] != 0x8b || gz[2] != 0x08 {
        panic!("not a gzip stream");
    }
    let flg = gz[3];
    let mut p = 10;
    if flg & 0x04 != 0 {
        let xlen = u16::from_le_bytes([gz[p], gz[p + 1]]) as usize;
        p += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        while p < gz.len() && gz[p] != 0 {
            p += 1;
        }
        p += 1;
    }
    if flg & 0x10 != 0 {
        while p < gz.len() && gz[p] != 0 {
            p += 1;
        }
        p += 1;
    }
    if flg & 0x02 != 0 {
        p += 2;
    }
    p
}
