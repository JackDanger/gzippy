//! Thread-count validity pins for the production encoder.
//!
//! ⚠ THIS FILE USED TO PIN BYTE-IDENTITY: `file_compression_is_byte_identical_across_thread_counts`
//! asserted that T>1 emits the same member as T1 on every real file input. That rule
//! was RETRACTED THREE TIMES — CLAUDE.md STEP 2: "T>1 may emit different bytes than T1.
//! THE ONLY CORRECTNESS BAR, at every thread count, is VALID GZIP: roundtrip sha256
//! through our decoder plus one independent decoder." The branch's own
//! `perf(parallel): enable parallel pipeline for T>1 L0-L12` commit re-routed T>1 into
//! the chunked pipeline whose bytes differ by construction (its message states L9 is
//! ~5% smaller at T4), so the old pin was RED BY CONSTRUCTION — it re-legislated the
//! banned "T1==T4" rule from a leaf test.
//!
//! What this file pins now (the STEP-2 bar, EXECUTED):
//!   * T1 stays the size reference: byte-deterministic across runs (the size board
//!     compares bytes — a flaky T1 would make every verdict noise), and valid gzip.
//!   * T>1 output at each thread count is VALID GZIP: it round-trips to the input
//!     byte-exactly through our decoder (`gzippy -d`) AND through one independent
//!     decoder (the system `gzip -d`), and is byte-deterministic at that thread
//!     count.
//!   * Nothing here asserts T>1 == T1 bytes. A future change may make T>1 smaller or
//!     larger than T1; it may not make it invalid or flaky.

use gzippy::fixtures;
use std::process::Command;

const LEVELS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

fn compress(bin: &str, path: &std::path::Path, level: u32, threads: u32) -> Vec<u8> {
    let output = Command::new(bin)
        .args([
            &format!("-{level}"),
            "-p",
            &threads.to_string(),
            "-c",
            path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "gzippy failed on {} L{level} T{threads}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Write `gz` to a decoder's stdin and return its stdout.
///
/// The write runs on its own thread: for incompressible input the decoder's
/// output is as large as the input, so an inline `write_all` fills the stdout
/// pipe (64 KiB) → the decoder blocks on `write` → it stops reading stdin →
/// the parent blocks on `write` — a cross-deadlock (the 2026-08-30 receipt
/// for this test's 30-minute timeout). Closing our end of stdin is equally
/// load-bearing: without it the decoder blocks in `read()` waiting for EOF.
fn run_stdin_decoder(cmd: &str, args: &[&str], gz: &[u8]) -> std::process::Output {
    use std::io::Write;
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {cmd} {args:?}: {e}"));
    let mut stdin = child.stdin.take().unwrap();
    // Scoped so the writer may borrow `gz` without a copy: the scope exits
    // only after `wait_with_output` has drained stdout AND the write finished.
    let output = std::thread::scope(|s| {
        let writer = s.spawn(move || {
            // Dropping `stdin` (end of this closure) is the EOF signal.
            stdin.write_all(gz).map_err(|e| e.to_string())
        });
        // `wait_with_output` drains stdout concurrently with the stdin write.
        let output = child
            .wait_with_output()
            .unwrap_or_else(|e| panic!("wait {cmd}: {e}"));
        writer
            .join()
            .unwrap_or_else(|_| panic!("decoder write thread panicked"))
            .unwrap_or_else(|e| panic!("write to {cmd} stdin: {e}"));
        output
    });
    output
}

/// Our decoder: `gzippy -d -c` from stdin.
fn decompress_ours(bin: &str, gz: &[u8]) -> Vec<u8> {
    let output = run_stdin_decoder(bin, &["-d", "-c"], gz);
    assert!(
        output.status.success(),
        "gzippy -d failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Independent decoder: the system `gzip -d -c` (GNU gzip — a different
/// implementation, the strongest independence available in a test environment).
fn decompress_independent(gz: &[u8]) -> Vec<u8> {
    let output = run_stdin_decoder("gzip", &["-d", "-c"], gz);
    assert!(
        output.status.success(),
        "independent decoder (gzip -d) REJECTED the stream — it is not valid gzip: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn t1_is_deterministic_and_every_thread_count_emits_valid_gzip() {
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();

    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let path = dir.path().join(name);
        std::fs::write(&path, &data).unwrap();

        for &level in LEVELS {
            // T1: the size reference — deterministic across runs, valid gzip.
            let t1a = compress(bin, &path, level, 1);
            let t1b = compress(bin, &path, level, 1);
            assert_eq!(
                t1a, t1b,
                "{name} L{level} T1: run-to-run output changed (non-deterministic)"
            );
            assert_eq!(
                decompress_ours(bin, &t1a),
                data,
                "{name} L{level} T1: our-decoder roundtrip"
            );
            assert_eq!(
                decompress_independent(&t1a),
                data,
                "{name} L{level} T1: independent-decoder roundtrip"
            );

            for threads in [2, 4, 8, 16] {
                let a = compress(bin, &path, level, threads);
                let b = compress(bin, &path, level, threads);
                assert_eq!(
                    a, b,
                    "{name} L{level} T{threads}: run-to-run output changed (non-deterministic)"
                );
                assert_eq!(
                    decompress_ours(bin, &a),
                    data,
                    "{name} L{level} T{threads}: our-decoder roundtrip"
                );
                assert_eq!(
                    decompress_independent(&a),
                    data,
                    "{name} L{level} T{threads}: independent-decoder roundtrip"
                );
            }
        }
    }
}
