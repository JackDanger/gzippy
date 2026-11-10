//! Pipe-parallelism regression guard.
//!
//! The contract under test: `gzippy -p N` reading stdin must produce
//! BYTE-IDENTICAL output to `gzippy -p N -c file` on the same bytes. This was
//! a known defect once ("stdin silently routes T1" — pipes got
//! single-threaded output and single-threaded wall while file arguments got
//! the parallel path), and it was fixed — but, as this suite discovered on
//! 2026-08-09, ONLY FOR SEEKABLE STDIN. There are two distinct stdin cases
//! and they routed differently from then until 2026-11-10:
//!
//!   1. REDIRECT stdin (`gzippy -p4 < file`): fd 0 is a regular file.
//!      `compress_stdin` (src/compress/io.rs) mmaps it and takes the same
//!      parallel path as a file argument. Byte-identical at p4/p8, equal
//!      wall. FIXED — and `redirect_stdin_matches_file_arg_at_p4` below
//!      guards it against regression.
//!
//!   2. TRUE PIPE stdin (`cat file | gzippy -p4`): fd 0 is a FIFO. The mmap
//!      probe failed and `compress_stdin` fell into its explicit
//!      "Pipe stdin: ..." branch, which hardcoded
//!      `OptimizationConfig::new(1, 0, ..)` and called the single-thread
//!      pipeline: `-p` was IGNORED, output was T1 bytes, and wall was T1
//!      wall (measured 2026-08-09: 64 MiB Cargo.lock-repeat, M1:
//!      truepipe@p4 0.269 s == truepipe@p1 0.269 s vs file@p4 0.087 s at L1;
//!      0.323 vs 0.148 s at L6). FIXED 2026-11-10: the pipe branch reads
//!      stdin to end and feeds the whole-buffer bytes through
//!      `compress_with_pipeline_sized` with a real OptimizationConfig
//!      (`args.processes` + true input length), so it routes exactly like a
//!      file argument at the same level and `-p`. The former #[ignore] pin
//!      `true_pipe_matches_file_arg_at_p4` below is now the guard.
//!
//! Why byte-identity with the FILE path proves the stdin path is PARALLEL:
//! the file-arg path at -p 4 is known-parallel and its output differs from
//! T1 on most fixtures (the t4 pins prove it; this suite re-checks on its own
//! grid), so identity cannot be satisfied by both quietly routing T1.
//!
//! Grid: every synthetic fixture (src/fixtures.rs, 1 MiB each) x levels
//! {1, 6, 9} x -p 4, through the real binary (CARGO_BIN_EXE_gzippy). Stdin
//! outputs also roundtrip through flate2 (an independent decoder), so a
//! corrupt-but-identical pair cannot pass.

use gzippy::fixtures;
use std::io::Read;
use std::io::Write;
use std::process::{Command, Stdio};

const LEVELS: &[u32] = &[1, 6, 9];
const THREADS: u32 = 4;

/// Compress a file argument through the real binary: `gzippy -L -p T -c path`.
fn compress_file(path: &std::path::Path, level: u32, threads: u32) -> Vec<u8> {
    let o = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args([
            &format!("-{level}"),
            "-p",
            &threads.to_string(),
            "-c",
            path.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "gzippy failed on file arg {} L{level} T{threads}: {}",
        path.display(),
        String::from_utf8_lossy(&o.stderr)
    );
    o.stdout
}

/// `gzippy -L -p T < path`: stdin is the regular file itself (seekable).
fn compress_redirect(path: &std::path::Path, level: u32, threads: u32) -> Vec<u8> {
    let o = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args([&format!("-{level}"), "-p", &threads.to_string(), "-c"])
        .stdin(Stdio::from(std::fs::File::open(path).unwrap()))
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "gzippy failed on redirect stdin L{level} T{threads}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    o.stdout
}

/// `producer | gzippy -L -p T`: stdin is a TRUE FIFO pipe. The writer runs on
/// its own thread so a large output cannot deadlock against the pipe.
fn compress_pipe(data: &[u8], level: u32, threads: u32) -> Vec<u8> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args([&format!("-{level}"), "-p", &threads.to_string(), "-c"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let data_owned = data.to_vec();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&data_owned).expect("write stdin");
        // drop closes the pipe
    });
    let o = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(
        o.status.success(),
        "gzippy failed on FIFO-pipe stdin L{level} T{threads}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    o.stdout
}

/// Decode a (possibly multi-member) gzip stream through flate2 — the
/// independent-decoder leg of the correctness bar.
fn flate2_decode(gz: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(gz)
        .read_to_end(&mut out)
        .expect("flate2 failed to decode the stdin-path output");
    out
}

/// Shared engine: compare a stdin flavor against the file-arg path across the
/// whole grid; roundtrip every stdin output; require p4 != p1 somewhere so
/// the identity cannot be trivially satisfied by both sides routing T1.
fn assert_stdin_matches_file(
    flavor: &str,
    compress_stdin_flavor: &dyn Fn(&std::path::Path, &[u8], u32) -> Vec<u8>,
) {
    let dir = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    let mut p4_differs_from_p1 = 0usize;
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let path = dir.path().join(name);
        std::fs::write(&path, &data).unwrap();
        for &level in LEVELS {
            let via_file = compress_file(&path, level, THREADS);
            let via_stdin = compress_stdin_flavor(&path, &data, level);
            let via_file_t1 = compress_file(&path, level, 1);
            if via_file != via_file_t1 {
                p4_differs_from_p1 += 1;
            }
            // Correctness holds regardless of routing.
            let decoded = flate2_decode(&via_stdin);
            assert_eq!(
                decoded, data,
                "{flavor} stdin output for {name}:L{level} decoded to the wrong bytes — corrupt"
            );
            if via_stdin != via_file {
                let first = via_stdin
                    .iter()
                    .zip(via_file.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| via_stdin.len().min(via_file.len()));
                failures.push(format!(
                    "STDIN ROUTING BUG ({flavor}) at {name}:L{level}:-p{THREADS}: stdin output \
                     ({} B) != file-arg output ({} B), first divergence at offset {first}. \
                     The file path is known-parallel, so this stdin flavor is NOT taking \
                     the parallel path. This is the 'stdin silently routes T1' defect — \
                     fix the routing in compress_stdin (src/compress/io.rs), do not \
                     weaken this test.{}",
                    via_stdin.len(),
                    via_file.len(),
                    if via_stdin == via_file_t1 {
                        " (Confirmed: the stdin bytes MATCH the -p 1 output — routed T1.)"
                    } else {
                        " (The stdin bytes match NEITHER -p 4 nor -p 1 file output — a \
                         third config; check what OptimizationConfig the stdin route built.)"
                    }
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}\n", failures.join("\n\n"));
    assert!(
        p4_differs_from_p1 > 0,
        "-p 4 file output was byte-identical to -p 1 on EVERY cell — the identity \
         check above proved nothing about parallelism. Either the parallel path \
         did not engage at all, or seams have been eliminated entirely; know which \
         before trusting this guard."
    );
}

/// THE REGRESSION GUARD (passes today): `gzippy -p4 < file` == `gzippy -p4
/// -c file`, byte for byte, on every fixture x level. Seekable stdin takes
/// the parallel path; this pins the fix so it cannot silently rot.
#[test]
fn redirect_stdin_matches_file_arg_at_p4() {
    assert_stdin_matches_file("redirect", &|path, _data, level| {
        compress_redirect(path, level, THREADS)
    });
}

/// THE BUG, FIXED 2026-11-10 and now the permanent guard: `cat file | gzippy
/// -p4` used to route the explicit single-threaded pipe branch of
/// `compress_stdin`, ignoring -p entirely (every fixture x level {1,6,9}
/// failed the identity, and wall equalled -p1 wall). The pipe branch now
/// routes whole-buffer bytes through `compress_with_pipeline_sized` with the
/// real `-p` config, so this must stay green the same way the redirect twin
/// above does.
#[test]
fn true_pipe_matches_file_arg_at_p4() {
    assert_stdin_matches_file("FIFO-pipe", &|_path, data, level| {
        compress_pipe(data, level, THREADS)
    });
}

/// Pipe output must stay valid and deterministic: it roundtrips through
/// flate2 (asserted for every cell here) and two runs emit identical bytes.
/// Pins correctness, not routing — the identity twin above guards routing.
#[test]
fn true_pipe_output_roundtrips_and_is_deterministic() {
    let mut cells = 0usize;
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        for &level in LEVELS {
            let a = compress_pipe(&data, level, THREADS);
            let b = compress_pipe(&data, level, THREADS);
            assert_eq!(
                a, b,
                "FIFO-pipe output is nondeterministic on {name}:L{level} — find the race"
            );
            assert_eq!(
                flate2_decode(&a),
                data,
                "FIFO-pipe output for {name}:L{level} decoded to the wrong bytes — corrupt"
            );
            cells += 1;
        }
    }
    assert_eq!(cells, fixtures::NAMES.len() * LEVELS.len());
}
