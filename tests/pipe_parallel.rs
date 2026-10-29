//! Pipe-parallelism regression guard.
//!
//! The contract under test: `gzippy -p N` reading stdin must produce
//! BYTE-IDENTICAL output to `gzippy -p N -c file` on the same bytes. This was
//! a known defect once ("stdin silently routes T1" — pipes got
//! single-threaded output and single-threaded wall while file arguments got
//! the parallel path), and it was fixed — but, as this suite discovered on
//! 2026-08-09, ONLY FOR SEEKABLE STDIN. There are two distinct stdin cases
//! and they route the same way as of wave-b (2026-10-29, branch
//! `wave-b/stdin-parallel`):
//!
//!   1. REDIRECT stdin (`gzippy -p4 < file`): fd 0 is a regular file.
//!      `compress_stdin` (src/compress/io.rs) mmaps it and takes the same
//!      parallel path as a file argument. Byte-identical at p4/p8, equal
//!      wall. FIXED 2026-08-09 — and `redirect_stdin_matches_file_arg_at_p4`
//!      below guards it against regression. At T1 the same rule now holds
//!      above the 128 KiB mmap threshold: `< file` at -p1 takes the map arm
//!      the file path takes (`PureT1Mmap`, in-place parse, byte-identical).
//!
//!   2. TRUE PIPE stdin (`cat file | gzippy -p4`): fd 0 is a FIFO. WAS the
//!      live routing bug this suite pinned on 2026-08-09 (the pipe branch
//!      hardcoded `OptimizationConfig::new(1, ...)`, `-p` was IGNORED, and
//!      wall was -p1 wall: truepipe@p4 0.269 s == truepipe@p1 0.269 s vs
//!      file@p4 0.087 s at L1; 0.323 vs 0.148 s at L6). FIXED in wave-b: the
//!      pipe branch now reads to end and runs ONE parallel
//!      PipelinedGzEncoder encode — the same engine the redirect arm uses —
//!      so `cat f | gzippy -pN` output is byte-identical to
//!      `gzippy -pN -c f`. T1 pipes keep the exact streaming route and bytes
//!      they always shipped. `true_pipe_matches_file_arg_at_p4` below (its
//!      #[ignore] removed) is now the guard.
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
                         third config; check what compress_stdin routes for this flavor.)"
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

/// THE GUARD (was the live bug pin, #[ignore]d 2026-08-09; un-ignored when
/// wave-b routed the pipe branch): `cat file | gzippy -p4` reads to end and
/// runs ONE parallel PipelinedGzEncoder encode, so its bytes equal
/// `gzippy -p4 -c file` on every fixture x level, and wall is parallel wall.
#[test]
fn true_pipe_matches_file_arg_at_p4() {
    assert_stdin_matches_file("FIFO-pipe", &|_path, data, level| {
        compress_pipe(data, level, THREADS)
    });
}

/// Even now that pipes route parallel, pipe output must stay VALID and
/// deterministic: it roundtrips through flate2 (asserted for every cell
/// here) and two runs emit identical bytes. This stays green across any
/// future re-routing; it pins correctness, not routing.
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

/// EXECUTED-route marker (the byte-identity test above proves PARALLELISM
/// indirectly; this proves it observably): with `GZIPPY_DEBUG=1`, a `cat f |
/// gzippy -6 -p4` run must print the route `route::emit` line naming
/// `PureParallelPipeline` with the actual worker count — the same Gate-4
/// marker the file-arg path prints (`t1_mmap_route.rs` style). The marker is
/// printed INSIDE `compress_buffer_pure`, so seeing it means the parallel
/// encoder executed, not merely that a branch was chosen.
#[test]
fn true_pipe_at_p4_prints_the_parallel_route_marker() {
    let data = fixtures::generate("text");
    let mut child = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args(["-6", "-p", "4", "-c"])
        .env("GZIPPY_DEBUG", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let data_owned = data.clone();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&data_owned).expect("write stdin");
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(out.status.success(), "pipe run failed: {out:?}");
    assert_eq!(flate2_decode(&out.stdout), data, "roundtrip");
    let stderr = String::from_utf8_lossy(&out.stderr);
    for token in [
        "compress path=PureParallelPipeline",
        "encode-path=PureParallelPipeline",
        "threads=4",
    ] {
        assert!(
            stderr.contains(token),
            "expected `{token}` in GZIPPY_DEBUG stderr of a -p4 pipe run:\n{stderr}"
        );
    }
    // And the OLD T1 marker must be gone from the pipe route: the former
    // pipe branch was exactly this bug (PureT1 printed on every -p run).
    assert!(
        !stderr.contains("encode-path=PureT1 "),
        "-p4 pipe must not execute the T1 route.\n{stderr}"
    );
}

/// T1 redirect stdin (`gzippy -p1 < file`) above the 128 KiB mmap threshold
/// must run the same map arm the file path runs (`PureT1Mmap`, in-place
/// parse) and emit byte-identical output to `gzippy -p1 -c file`; below the
/// threshold it keeps the read-to-end `PureT1` route — the same threshold
/// split the file path owns (pinned in t1_mmap_route.rs).
#[test]
fn redirect_stdin_at_t1_takes_the_mmap_route_above_the_threshold() {
    let mut corpus = Vec::with_capacity(2 * 1024 * 1024);
    let mut s: u32 = 0x9E37_79B9;
    for _ in 0..2 * 1024 * 1024 {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let r = (s >> 16) as u8;
        corpus.push(if (corpus.len() / 4096) % 3 == 0 {
            r
        } else {
            b'a' + (r % 26)
        });
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.bin");
    std::fs::write(&path, &corpus).unwrap();

    // Above the threshold: mmap route, bytes identical to the file -c route.
    let via_file = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args(["-6", "-p1", "-c"])
        .arg(&path)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(via_file.status.success());
    let gdfd = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args(["-6", "-p1", "-c"])
        .env("GZIPPY_DEBUG", "1")
        .stdin(Stdio::from(std::fs::File::open(&path).unwrap()))
        .output()
        .unwrap();
    assert!(gdfd.status.success(), "redirect run failed: {gdfd:?}");
    let stderr = String::from_utf8_lossy(&gdfd.stderr).to_string();
    assert!(
        stderr.contains("encode-path=PureT1Mmap level=6 threads=1"),
        "redirect `< file` at -p1 above the threshold must execute PureT1Mmap:\n{stderr}"
    );
    assert_eq!(
        gdfd.stdout, via_file.stdout,
        "redirect stdin at -p1 must be byte-identical to `gzippy -p1 -c file`"
    );
    assert_eq!(flate2_decode(&gdfd.stdout), corpus, "roundtrip");

    // Below the threshold: the read-to-end route (no mmap), like a file arg.
    let small = &corpus[..64 * 1024];
    let small_path = dir.path().join("small.bin");
    std::fs::write(&small_path, small).unwrap();
    let via_file_small = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args(["-6", "-p1", "-c"])
        .arg(&small_path)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(via_file_small.status.success());
    let red = Command::new(env!("CARGO_BIN_EXE_gzippy"))
        .args(["-6", "-p1", "-c"])
        .env("GZIPPY_DEBUG", "1")
        .stdin(Stdio::from(std::fs::File::open(&small_path).unwrap()))
        .output()
        .unwrap();
    assert!(red.status.success(), "small redirect failed: {red:?}");
    let stderr = String::from_utf8_lossy(&red.stderr).to_string();
    assert!(
        stderr.contains("encode-path=PureT1 level=6 threads=1"),
        "sub-threshold redirect must stay on PureT1 (read-to-end):\n{stderr}"
    );
    assert!(
        !stderr.contains("PureT1Mmap"),
        "sub-threshold redirect must not take the mmap route:\n{stderr}"
    );
    assert_eq!(flate2_decode(&red.stdout), small, "small roundtrip");
    assert_eq!(
        red.stdout, via_file_small.stdout,
        "sub-threshold redirect must stay byte-identical to the file -c route"
    );
}
