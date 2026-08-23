//! Route assertions for the T1 FILE mmap path — EXECUTED, not inferred.
//!
//! The route under test: T1 file inputs above the 128 KiB mmap threshold take
//! `PureT1Mmap` (whole-buffer-equivalent parse over a read-only mmap,
//! `deflate::encode_gzip_unpadded_slice_to_writer`), while non-seekable T1
//! inputs (stdin/pipes) keep the streaming `PureT1` route. The route name is
//! printed by `compress::route::emit` at the call site of the encoder that
//! actually runs (`GZIPPY_DEBUG=1`), so these tests observe the executed path
//! rather than re-deriving the routing table from source — the same Gate-4
//! discipline the T>1 routes already carry.
//!
//! Byte-identity between the two routes is asserted here too: the mmap route
//! exists to change WALL, never bytes, so file-input output must equal
//! stdin-input output for the same data at every level checked.

use assert_cmd::Command;

/// Deterministic compressible-but-not-trivial bytes, comfortably above the
/// 128 KiB mmap threshold and above the ~375 KB pass-2 tail so the mmap
/// encoder's bulk pass genuinely runs.
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

/// stderr must name the mmap route, with T1 and the requested level — the
/// exact token `route::emit` prints, including the trailing space that keeps
/// `PureT1` (a prefix of `PureT1Mmap`) from matching by accident.
fn assert_route(stderr: &[u8], route: &str, level: u32) {
    let s = String::from_utf8_lossy(stderr);
    let needle = format!("encode-path={route} level={level} threads=1");
    assert!(
        s.contains(&needle),
        "expected `{needle}` in stderr, got:\n{s}"
    );
}

#[test]
fn t1_file_to_stdout_takes_the_mmap_route_and_matches_stdin_bytes() {
    let data = corpus(2 * 1024 * 1024);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.bin");
    std::fs::write(&path, &data).unwrap();

    for level in [1u32, 6, 9] {
        // FILE input: must execute PureT1Mmap.
        let file_run = gzippy()
            .args([&format!("-{level}"), "-p1", "-c", path.to_str().unwrap()])
            .assert()
            .success();
        assert_route(&file_run.get_output().stderr, "PureT1Mmap", level);
        let file_gz = file_run.get_output().stdout.clone();
        assert_eq!(roundtrip(&file_gz), data, "L{level}: file-route roundtrip");

        // PIPE stdin: must keep the streaming PureT1 route...
        let stdin_run = gzippy()
            .args([&format!("-{level}"), "-p1", "-c"])
            .write_stdin(data.clone())
            .assert()
            .success();
        let stderr = String::from_utf8_lossy(&stdin_run.get_output().stderr).to_string();
        assert_route(&stdin_run.get_output().stderr, "PureT1", level);
        assert!(
            !stderr.contains("PureT1Mmap"),
            "L{level}: pipe stdin must not take the mmap route:\n{stderr}"
        );

        // ...and the two routes must emit IDENTICAL bytes.
        assert_eq!(
            file_gz,
            stdin_run.get_output().stdout,
            "L{level}: mmap (file) and streaming (stdin) outputs differ"
        );
    }
}

#[test]
fn t1_in_place_file_flow_takes_the_mmap_route() {
    let data = corpus(1024 * 1024);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inplace.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args(["-1", "-p1", "-k", path.to_str().unwrap()])
        .assert()
        .success();
    assert_route(&run.get_output().stderr, "PureT1Mmap", 1);

    let gz = std::fs::read(dir.path().join("inplace.bin.gz")).expect("output file written");
    assert_eq!(roundtrip(&gz), data, "in-place mmap-route roundtrip");

    // The -c flow over the same input runs the same encoder behind the same
    // route, so the compressed BODY must be identical. The headers differ by
    // contract (issue #309): file output stores FNAME+MTIME the way gzip
    // does, while -c keeps the minimal 10-byte header (what libdeflate-gzip
    // emits on -c; every graded invocation is -c).
    let stdout_run = gzippy()
        .args(["-1", "-p1", "-c", path.to_str().unwrap()])
        .assert()
        .success();
    let cgz = stdout_run.get_output().stdout.clone();
    assert_eq!(
        &cgz[..10],
        &[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x04, 0xff]
    );
    assert_eq!(
        gz[3] & 0x08,
        0x08,
        "file output must set FLG.FNAME (issue #309)"
    );
    let fname_end = 10
        + gz[10..]
            .iter()
            .position(|&b| b == 0)
            .expect("NUL-terminated FNAME");
    assert_eq!(
        &gz[10..fname_end],
        b"inplace.bin",
        "FNAME must be the input basename"
    );
    assert_eq!(
        &gz[fname_end + 1..],
        &cgz[10..],
        "in-place and -c outputs must differ ONLY in the gzip header"
    );
}

/// Files at or below the 128 KiB threshold keep the streaming route — the
/// threshold is shared with the T>1 mmap gate, and this pins which side of it
/// each route owns.
#[test]
fn t1_small_file_keeps_the_streaming_route() {
    let data = corpus(64 * 1024);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.bin");
    std::fs::write(&path, &data).unwrap();

    let run = gzippy()
        .args(["-6", "-p1", "-c", path.to_str().unwrap()])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&run.get_output().stderr).to_string();
    assert_route(&run.get_output().stderr, "PureT1", 6);
    assert!(
        !stderr.contains("PureT1Mmap"),
        "sub-threshold file must not take the mmap route:\n{stderr}"
    );
    assert_eq!(roundtrip(&run.get_output().stdout), data);
}
