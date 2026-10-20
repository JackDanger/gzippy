//! Issue #258 — route assertions for EVERY `compress::route::emit` constant.
//!
//! The shipped binary prints one `[gzippy] compress path=<NAME>
//! encode-path=<NAME> level=L threads=T` line per EXECUTING encoder when
//! `GZIPPY_DEBUG=1` (`src/compress/mod.rs` `route::emit` at the call site of
//! the encoder that actually runs). CLAUDE.md's measurement PROTOCOL requires
//! any claim about the shipped binary to be proven by executing it, so every
//! constant in that closed set must have a spawned-binary assertion somewhere
//! in `tests/`. This file is the inventory; its two new tests close the gaps.
//!
//! # Audit of the five `route::emit` constants (at main = 4d2e2dea)
//!
//! | `route::*` constant | spawn+`GZIPPY_DEBUG=1` assertion BEFORE this file | assertion site |
//! |---|---|---|
//! | `Zopfli` | **none** | NEW: `zopfli_route_executes_on_explicit_tuning` (below) |
//! | `PureT1` | yes — pipe stdin at L1/6/9 and sub-threshold file at L6 (`tests/t1_mmap_route.rs`) | `tests/t1_mmap_route.rs`; cross-referenced, not duplicated |
//! | `PureT1Mmap` | yes — 2 MiB file at L1/6/9 and 1 MiB in-place (`tests/t1_mmap_route.rs`) | `tests/t1_mmap_route.rs`; cross-referenced, not duplicated |
//! | `PureParallelPipeline` | **none** | NEW: `parallel_pipeline_route_executes_at_p2` (below) |
//! | `Rsyncable` | **none — unreachable by design since the flag-honesty rejection**: compress-mode `-R`/`--rsyncable` is refused with exit 2 and stderr naming `rsyncable` + issue #302 (`tests/rival_flags.rs`, `pigz_rsyncable_rejected_loudly`). The `route::RSYNCABLE` emit in `io.rs` can only be reached through the flag that never reaches it, so a positive route assertion would have to resurrect the flag and would fail open. The authoritative pin is the REJECTION, cross-referenced — not duplicated here. |
//!
//! Decode-side path names (e.g. `path=ParallelSM`) are a separate half: the
//! binary's decode route is asserted by CI Gate-4 (`ci.yml` "Gate-4
//! production-path assertion", working on `encode-path=`/decode `path=`
//! tokens), and single-member decode selection is pinned through the library
//! API in `tests/lib_api.rs`. The files the issue names for that half
//! (`tests/trace_parity.rs`, `tests/selector_byte_transparency.rs`) do not
//! exist at this ref; the audit above covers the compress-side `route::emit`
//! set only.
//!
//! Input-size gates (`src/compress/io.rs`): the mmap routes force
//! `file_size > 128 * 1024` (the threshold shared by the T>1 mmap gate and the
//! T1 mmap gate); the T>1 parallel pipeline additionally needs
//! `thread_count > 1`, which for a regular FILE input means `-pN` above the
//! same threshold. The inputs below sit well clear of both.

use assert_cmd::Command;

/// Deterministic compressible-but-not-trivial bytes (same generator shape as
/// `tests/t1_mmap_route.rs`): pseudo-random noise blended with low-entropy
/// text-like runs.
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
    let mut c = Command::cargo_bin("gzippy").expect("gzippy binary");
    c.env("GZIPPY_DEBUG", "1");
    c
}

fn roundtrip(gz: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(gz)
        .read_to_end(&mut out)
        .expect("valid gzip stream");
    out
}

/// stderr must name the executed route — the exact token `route::emit`
/// prints, so a test that greps `encode-path=` observes the encoder that ran,
/// never a routing decision. Threads passed separately so the T>1 row can pin
/// the requested `-pN` (the printed value is the pipeline's thread count, not
/// always the CLI's).
fn assert_route(stderr: &[u8], route: &str, level: u32, threads: usize) {
    let s = String::from_utf8_lossy(stderr);
    let needle = format!("encode-path={route} level={level} threads={threads}");
    assert!(
        s.contains(&needle),
        "expected `{needle}` in stderr, got:\n{s}"
    );
}

/// `-F 1` (zopfli iteration override) forces the zopfli encoder on ANY input
/// (`explicit_zopfli` excludes both mmap routes and the plain T1 route,
/// `src/compress/mod.rs`). One iteration on a 16 KiB input keeps runtime well
/// under a second. `-p1` pins the printed thread count (the zopfli route is
/// single-member by ratio mandate and reports `args.processes.max(1)`).
#[test]
fn zopfli_route_executes_on_explicit_tuning() {
    let data = corpus(16 * 1024);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args(["-6", "-p1", "-F", "1", "-c", path.to_str().unwrap()])
        .assert()
        .success();

    assert_route(&run.get_output().stderr, "Zopfli", 6, 1);
    assert_eq!(
        roundtrip(&run.get_output().stdout),
        data,
        "zopfli roundtrip"
    );
}

/// `-p2` on a file above the 128 KiB threshold takes the T>1 parallel
/// pipeline (`use_mmap && !rsyncable && !explicit_zopfli`), which prints its
/// route from inside `PipelinedGzEncoder::compress_buffer_pure` — the same
/// line whichever call site reached it. The `threads=2` tail assumes a runner
/// with at least 2 cores (GitHub's linux runners qualify): thread honoring is
/// part of the observable contract this row asserts.
#[test]
fn parallel_pipeline_route_executes_at_p2() {
    let data = corpus(2 * 1024 * 1024);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args(["-1", "-p2", "-c", path.to_str().unwrap()])
        .assert()
        .success();

    assert_route(&run.get_output().stderr, "PureParallelPipeline", 1, 2);
    assert_eq!(
        roundtrip(&run.get_output().stdout),
        data,
        "parallel route roundtrip"
    );
}
