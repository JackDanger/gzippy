//! DEFLATE-validity contract for the decoder.
//!
//! Contract (gzip(1) and libdeflate-gzip as the reference decoders): on any
//! input both of them REJECT, gzippy must exit non-zero PROMPTLY — never hang,
//! never print fabricated bytes with exit 0. Every input here was rejected by
//! GNU gzip 1.14 and libdeflate-gzip when the test was written (probed with
//! `gzip -dc F; echo $?` and `libdeflate-gzip -dc F; echo $?`, both exit 1).
//!
//! Inputs are the decoder audit's fuzz reproducers, embedded as hex so the
//! suite carries no external files.

use std::io::{Read as _, Write as _};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn gzippy() -> &'static str {
    env!("CARGO_BIN_EXE_gzippy")
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2));
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// `n` zero bytes as hex (the hang286 reproducers are mostly zero runs).
fn z(n: usize) -> String {
    "00".repeat(n)
}

/// "Promptly": every input here is under 200 bytes, so a correct reject takes
/// milliseconds even in a debug build. The audit saw these spin for 15 s+.
const DEADLINE: Duration = Duration::from_secs(5);

struct Outcome {
    code: Option<i32>,
    stdout: Vec<u8>,
    timed_out: bool,
}

/// Run gzippy with `args`, feeding `stdin_bytes` through a PIPE; kill it
/// after DEADLINE.
fn run_bounded(args: &[&str], stdin_bytes: &[u8]) -> Outcome {
    run_bounded_with(args, Some(stdin_bytes), None)
}

/// `gzippy -dc < FILE`: stdin is a regular file, which takes the mmap'd
/// stdin route rather than the pipe route.
fn decode_stdin_file(bytes: &[u8]) -> Outcome {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("in.z");
    std::fs::write(&p, bytes).unwrap();
    let f = std::fs::File::open(&p).unwrap();
    run_bounded_with(&["-dc"], None, Some(f))
}

fn run_bounded_with(
    args: &[&str],
    stdin_bytes: Option<&[u8]>,
    stdin_file: Option<std::fs::File>,
) -> Outcome {
    let stdin_cfg = match stdin_file {
        Some(f) => Stdio::from(f),
        None => Stdio::piped(),
    };
    let mut child = Command::new(gzippy())
        .args(args)
        .stdin(stdin_cfg)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn gzippy");
    let stdin = child.stdin.take();
    let input = stdin_bytes.unwrap_or(&[]).to_vec();
    let writer = std::thread::spawn(move || {
        if let Some(mut stdin) = stdin {
            let _ = stdin.write_all(&input);
        }
    });
    let mut stdout = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = stdout.read_to_end(&mut v);
        v
    });
    let start = Instant::now();
    let (code, timed_out) = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break (st.code(), false);
        }
        if start.elapsed() > DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            break (None, true);
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let _ = writer.join();
    let stdout = reader.join().unwrap();
    Outcome {
        code,
        stdout,
        timed_out,
    }
}

/// Write `bytes` to a temp file and decode it with `-dc -pN FILE`.
fn decode_file(bytes: &[u8], threads: usize) -> Outcome {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("in.gz");
    std::fs::write(&p, bytes).unwrap();
    let t = format!("-p{threads}");
    run_bounded(&["-dc", &t, p.to_str().unwrap()], &[])
}

fn assert_rejected(name: &str, o: &Outcome) {
    assert!(
        !o.timed_out,
        "{name}: gzippy HUNG (killed after {DEADLINE:?}); gzip/libdeflate reject it"
    );
    assert!(
        matches!(o.code, Some(c) if c != 0),
        "{name}: gzippy exited {:?} with {} stdout bytes; gzip/libdeflate reject it",
        o.code,
        o.stdout.len()
    );
}

fn assert_file_rejected_all_threads(name: &str, bytes: &[u8]) {
    for t in [1usize, 2, 4] {
        let o = decode_file(bytes, t);
        assert_rejected(&format!("{name} -p{t}"), &o);
    }
}

// --- Finding 1: empty non-final blocks fabricated from past-EOF zero padding.

#[test]
fn empty_fixed_block_then_zero_padding_terminates() {
    let f = unhex("1f8b0800000000000003020000000000000000");
    assert_file_rejected_all_threads("gz_emptyfixed_isize0", &f);
}

#[test]
fn parallel_sm_19_byte_fuzz_case_terminates() {
    let f = unhex("1f8b08000000003100f50253d0ff94f88301e9");
    assert_file_rejected_all_threads("t_parallelsm_19b", &f);
}

// --- Finding 2: HLIT > 286 / HDIST > 30 / no EOB code / symbols 286-287.

fn hang286_body() -> String {
    // dynamic header declaring HLIT=288 whose code uses symbol 286
    format!("fde00190244992244902{}01{}10{}2060", z(32), z(94), z(14))
}

#[test]
fn gzip_hlit_288_symbol_286_terminates() {
    let f = unhex(&format!(
        "1f8b08{}03{}{}01000000",
        z(6),
        hang286_body(),
        z(4)
    ));
    assert_eq!(f.len(), 172, "embedded gz_hang286 reproducer");
    assert_file_rejected_all_threads("gz_hang286", &f);
}

#[test]
fn fuzz_37_byte_dynamic_header_terminates() {
    let f = unhex("1f8b080052525252525252ffffff57ff88000052ffffffffffffff52523452525252520700");
    assert_file_rejected_all_threads("fuzz2_timeout_b", &f);
}

#[test]
fn zlib_stdin_8_byte_body_terminates() {
    let f = unhex("789cfcff210003002a0000000000");
    let o = run_bounded(&["-dc"], &f);
    assert_rejected("zlib_hang_8b stdin", &o);
    let o = decode_stdin_file(&f);
    assert_rejected("zlib_hang_8b < file", &o);
}

#[test]
fn zlib_stdin_hlit_288_symbol_286_terminates() {
    let f = unhex(&format!("789c{}{}", hang286_body(), z(4)));
    assert_eq!(f.len(), 160, "embedded zlib_hang286 reproducer");
    let o = run_bounded(&["-dc"], &f);
    assert_rejected("zlib_hang286 stdin", &o);
    let o = decode_stdin_file(&f);
    assert_rejected("zlib_hang286 < file", &o);
}

#[test]
fn raw_deflate_library_8_byte_body_terminates() {
    let f = unhex("fcff210003002a00");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(gzippy::decompress_raw(&f).is_err());
    });
    match rx.recv_timeout(DEADLINE) {
        Ok(is_err) => assert!(is_err, "decompress_raw accepted an invalid stream"),
        Err(_) => panic!("decompress_raw HUNG on an 8-byte invalid stream"),
    }
}

// --- Finding 13: litlen symbol 286 (reserved, RFC 1951 3.2.6) is accepted.

#[test]
fn fixed_huffman_symbol_286_is_rejected() {
    // 'a' then fixed-code symbol 286, then EOB; CRC/ISIZE match "a".
    let f = unhex("1f8b08000000000000034b1c030043beb7e801000000");
    assert_file_rejected_all_threads("fixed_sym286", &f);
}

#[test]
fn zlib_fixed_huffman_symbol_286_is_rejected() {
    let f = unhex(&format!("789c{}0c{}", hang286_body(), z(8)));
    assert_eq!(f.len(), 165, "embedded zlib_sym286 reproducer");
    let o = run_bounded(&["-dc"], &f);
    assert_rejected("zlib_sym286 stdin", &o);
    let o = decode_stdin_file(&f);
    assert_rejected("zlib_sym286 < file", &o);
}

// --- Finding 7: back-references before the start of the member.

#[test]
fn backref_before_stream_start_is_rejected() {
    // Fixed block whose first symbol is a 258-byte match at distance 32768:
    // there are no prior bytes. main printed 258 zero bytes, exit 0.
    let f = unhex("1f8b08000000000000031bbdff1f009abf69ae02010000");
    assert_file_rejected_all_threads("window_underflow_small", &f);
}

#[test]
fn backref_into_previous_member_is_rejected() {
    // Member 2 is a back-reference into member 1's bytes (each gzip member is
    // an independent DEFLATE stream; its CRC was crafted to match). gzip and
    // libdeflate print member 1 and exit 1.
    let f = unhex(concat!(
        "1f8b08000000000002ffcb48cdc9c957c84dcd4d4a2d52c8cf4bd55148cd4bc9cc4b57c8cc5388",
        "0200adb90bdf1d0000001f8b0800000000000003030200407f805503000000",
    ));
    assert_file_rejected_all_threads("xmember_backref", &f);
}

// --- Findings 11/14: truncation at the end of the file.

/// `printf 'hello\n' | gzip -n` (GNU gzip 1.14).
const HELLO_GZ: &str = "1f8b0800000000000003cb48cdc9c9e7020020303a3606000000";

#[test]
fn trailer_truncated_by_one_byte_is_rejected() {
    // empty.gz (FNAME "null") cut to 24 of 25 bytes: the "trailer" read from
    // the last 8 bytes overlapped the deflate data. gzip: unexpected end of file.
    let f = unhex("1f8b08080000000000036e756c6c00030000000000000000");
    assert_file_rejected_all_threads("empty.gz cut to 24 bytes", &f);
}

#[test]
fn trailing_gzip_magic_that_is_not_a_complete_member_is_rejected() {
    // gzip 1.14: bytes after a member that start with 1f 8b (or are a lone
    // 1f at EOF) are a member: truncated -> "unexpected end of file", bad
    // method -> "unknown method", exit 1 either way.
    for tail in ["1f", "1f8b", "1f8b08", "1f8b0800000000000003", "1f8b78797a"] {
        let f = unhex(&format!("{HELLO_GZ}{tail}"));
        assert_file_rejected_all_threads(&format!("hello.gz + {tail}"), &f);
    }
}

#[test]
fn trailing_zeros_or_non_magic_garbage_still_decode() {
    // Not a member: zero padding (gzip: silent) and other bytes (gzip: "trailing
    // garbage ignored") keep the decoded member and do not fail.
    for tail in ["00000000", "1f00", "001f8b08", "67617262616765"] {
        let f = unhex(&format!("{HELLO_GZ}{tail}"));
        for t in [1usize, 4] {
            let o = decode_file(&f, t);
            assert!(!o.timed_out, "hello.gz + {tail} -p{t} hung");
            assert_eq!(o.code, Some(0), "hello.gz + {tail} -p{t}: exit");
            assert_eq!(o.stdout, b"hello\n", "hello.gz + {tail} -p{t}: output");
        }
    }
}
