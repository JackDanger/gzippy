//! `-vv` chunk-grid observability pins (issue #259).
//!
//! The compress path reports its chunk grid on ONE stderr line at `-vv`:
//! the T>1 pipeline reports route name, effective thread count, per-chunk
//! block size (and whether it came from an explicit `-b`), chunk count and
//! avg + max chunk size; the T1 whole-buffer routes report their equivalent
//! single line (`chunks=1 (whole-buffer)`). Plain `-v` (verbosity 1) must
//! emit NOTHING of the kind — no pre-existing verbose output changes shape.
//!
//! The T>1 numbers are not hard-coded: they are derived from
//! `pipelined_block_size`, the same pure function the encoder's grid uses
//! (the thread-free grid is host-independent, so the values below hold on
//! any box), so the pin fails if the report ever drifts from the numbers
//! that actually decided the encode — and only then. Default (non `-vv`)
//! runs are not exercised here; `tests/thread_byte_parity.rs` pins their
//! output bytes and routes separately.

use assert_cmd::Command;
use gzippy::compress::pipelined::{pipelined_block_size, GRID_REF_THREADS};

const LEVEL: u32 = 6;

fn corpus(len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut s: u32 = 0x9E37_79B9;
    for i in 0..len {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let r = (s >> 16) as u8;
        v.push(if (i / 4096) % 3 == 0 {
            r
        } else {
            b'a' + (r % 26)
        });
    }
    v
}

fn gzippy() -> Command {
    Command::cargo_bin("gzippy").expect("gzippy binary")
}

fn chunk_grid_lines(stderr: &str) -> Vec<&str> {
    stderr
        .lines()
        .filter(|l| l.contains("chunk grid:"))
        .collect()
}

fn assert_one_chunk_line(stderr: &[u8]) -> String {
    let s = String::from_utf8_lossy(stderr).to_string();
    let lines = chunk_grid_lines(&s);
    assert_eq!(
        lines.len(),
        1,
        "exactly ONE chunk-grid line at -vv (non-noisy), got {lines:?} in:\n{s}"
    );
    lines[0].to_string()
}

/// The report must restate the numbers the grid actually used: derive them
/// from `pipelined_block_size` — the same pure function the pipeline calls —
/// rather than hard-coding constants, so this pin tracks the report-vs-grid
/// agreement, not the grid values themselves (already pinned elsewhere).
/// `num_chunks` is the hand-computed expectation, asserted against the
/// derivation so a reading error here fails loudly.
fn expected_parallel_tail(input_len: usize, num_chunks: usize) -> String {
    let block = pipelined_block_size(input_len, GRID_REF_THREADS, LEVEL);
    let chunks = input_len.div_ceil(block);
    assert_eq!(chunks, num_chunks, "derived grid vs hand-computed");
    format!(
        " block={} block-source=computed chunks={} avg={} max={}",
        block,
        chunks,
        input_len / chunks,
        block.min(input_len),
    )
}

const PARALLEL_PREFIX_VV: &str = "gzippy: chunk grid: route=PureParallelPipeline threads=";

#[test]
fn vv_reports_the_chunk_grid_for_parallel_file_input() {
    let len = 1024 * 1024; // above the 128 KiB mmap threshold: the mmap fast path
    let data = corpus(len);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args([
            "-vv",
            format!("-{LEVEL}").as_str(),
            "-p4",
            "-c",
            path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let line = assert_one_chunk_line(&run.get_output().stderr);
    assert!(
        line.starts_with(PARALLEL_PREFIX_VV),
        "chunk-grid line must name the pipeline route and threads, got: {line}"
    );
    let threads = line[PARALLEL_PREFIX_VV.len()..]
        .split(' ')
        .next()
        .expect("threads field present");
    assert!(
        !threads.is_empty() && threads.bytes().all(|b| b.is_ascii_digit()),
        "threads must be the effective count, got: {line}"
    );
    assert!(
        line.ends_with(&expected_parallel_tail(len, 4)),
        "chunk-grid numbers must be the grid's own, got: {line}"
    );

    use std::io::Read;
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(run.get_output().stdout.as_slice())
        .read_to_end(&mut out)
        .expect("valid gzip stream");
    assert_eq!(out, data, "-vv report must not change the encode");
}

#[test]
fn plain_v_prints_stats_but_no_chunk_grid_line() {
    let len = 1024 * 1024;
    let data = corpus(len);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args([
            "-v",
            format!("-{LEVEL}").as_str(),
            "-p4",
            "-k",
            path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&run.get_output().stderr);
    assert!(
        stderr.contains("replaced with"),
        "plain -v must keep its stats line, got: {stderr}"
    );
    assert!(
        chunk_grid_lines(&stderr).is_empty(),
        "plain -v must NOT report the chunk grid, got: {stderr}"
    );
}

#[test]
fn vv_marks_an_explicit_b_block_size() {
    let len = 1024 * 1024;
    let data = corpus(len);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args([
            "-vv",
            format!("-{LEVEL}").as_str(),
            "-p4",
            "-b",
            "256",
            "-c",
            path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let line = assert_one_chunk_line(&run.get_output().stderr);
    // -b 256: a bare number is KIBIBYTES (pigz semantics) = 262,144 bytes —
    // what `set_block_size_override` receives via `block_size_explicit`.
    // 4 uniform chunks over 1 MiB.
    assert!(
        line.ends_with(&format!(
            " block={} block-source=explicit-b chunks={chunks} avg={avg} max={max}",
            256 * 1024,
            chunks = len / (256 * 1024),
            avg = len / (len / (256 * 1024)),
            max = 256 * 1024,
        )),
        "an explicit -b must be reported as explicit-b with the user's size, got: {line}"
    );
}

#[test]
fn vv_reports_one_whole_buffer_chunk_for_the_t1_routes() {
    let dir = tempfile::tempdir().unwrap();

    // FILE input above the 128 KiB mmap threshold: the `PureT1Mmap` route.
    let data = corpus(256 * 1024);
    let path = dir.path().join("mmap.bin");
    std::fs::write(&path, &data).unwrap();
    let run = gzippy()
        .args([
            "-vv",
            format!("-{LEVEL}").as_str(),
            "-p1",
            "-c",
            path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let line = assert_one_chunk_line(&run.get_output().stderr);
    assert_eq!(
        line, "gzippy: chunk grid: route=PureT1Mmap threads=1 chunks=1 (whole-buffer)",
        "T1 mmap route must report its whole-buffer summary, got: {line}"
    );

    // FILE input below the threshold: read-to-end `PureT1` (the T1 branch of
    // `compress_with_pipeline_sized`).
    let data = corpus(64 * 1024);
    let path = dir.path().join("small.bin");
    std::fs::write(&path, &data).unwrap();
    let run = gzippy()
        .args([
            "-vv",
            format!("-{LEVEL}").as_str(),
            "-p1",
            "-c",
            path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let line = assert_one_chunk_line(&run.get_output().stderr);
    assert_eq!(
        line, "gzippy: chunk grid: route=PureT1 threads=1 chunks=1 (whole-buffer)",
        "read-to-end T1 route must report its whole-buffer summary, got: {line}"
    );

    // PIPED stdin: same whole-buffer `PureT1` summary.
    let data = corpus(256 * 1024);
    let run = gzippy()
        .args(["-vv", format!("-{LEVEL}").as_str(), "-p1", "-c"])
        .write_stdin(data)
        .assert()
        .success();
    let line = assert_one_chunk_line(&run.get_output().stderr);
    assert_eq!(
        line, "gzippy: chunk grid: route=PureT1 threads=1 chunks=1 (whole-buffer)",
        "piped stdin T1 route must report its whole-buffer summary, got: {line}"
    );
}

#[test]
fn vv_reports_the_chunk_grid_for_the_subthreshold_parallel_fallback() {
    // A file at or below the 128 KiB threshold (with -p4) skips the mmap fast
    // path and reaches the same pipeline through
    // `compress_with_pipeline_sized`; the report must be identical in shape.
    let len = 64 * 1024;
    let data = corpus(len);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args([
            "-vv",
            format!("-{LEVEL}").as_str(),
            "-p4",
            "-c",
            path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let line = assert_one_chunk_line(&run.get_output().stderr);
    assert!(
        line.starts_with(PARALLEL_PREFIX_VV),
        "chunk-grid line must name the pipeline route and threads, got: {line}"
    );
    assert!(
        line.ends_with(&expected_parallel_tail(len, 1)),
        "chunk-grid numbers must be the grid's own, got: {line}"
    );
}
