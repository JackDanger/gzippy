//! CLI drop-in contract tests.
//!
//! gzippy's charter is drop-in replacement for gzip (and pigz /
//! libdeflate-gzip / igzip): same commands, same observable behaviour. The
//! CLI-behaviour axis has a box-side tool (`fulcrum dropin`) but had no
//! laptop-speed guard; this suite is that guard. Every assertion names the
//! contract clause it tests — gzip(1) as documented by GNU gzip, since gzip
//! itself is the contract (POSIX specifies only `compress`, not gzip).
//!
//! Behaviour was probed against GNU gzip 1.14 before writing each pin. Where
//! gzippy DIVERGES from the gzip contract the test is `#[ignore]` with a
//! `DIVERGENCE from gzip contract:` comment — those are triage candidates,
//! not endorsed behaviour. Current divergence list:
//!
//!   1. `plain_decompress_names_output_from_on_disk_filename` — plain `-d`
//!      restores the header-stored original filename; GNU gzip derives the
//!      output name from the on-disk filename minus suffix unless -N is given.
//!
//! All inputs are small synthetic bytes generated in-test; no corpus data.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn gzippy() -> &'static str {
    env!("CARGO_BIN_EXE_gzippy")
}

/// Run gzippy with `args` in `dir`, stdin closed. Panics only on spawn
/// failure; callers assert on the Output.
fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(gzippy())
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("failed to spawn gzippy")
}

/// Run gzippy with `stdin_bytes` piped in; returns the Output (stdout captured).
fn run_piped(dir: &Path, args: &[&str], stdin_bytes: &[u8]) -> Output {
    let mut child = Command::new(gzippy())
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn gzippy");
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(stdin_bytes)
        .expect("write stdin");
    child.wait_with_output().expect("wait gzippy")
}

/// A few KB of compressible-but-not-trivial synthetic bytes.
fn sample_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(4096);
    let mut x: u64 = 0x243f_6a88_85a3_08d3;
    while out.len() < 4096 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(b"the quick brown fox ");
        out.push(b'a' + (x % 26) as u8);
        out.push(b'\n');
    }
    out
}

fn tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("create tempdir")
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, bytes).expect("write input file");
    p
}

/// Compress `name` in `dir` with `-k -c`-free default args plus `extra`,
/// asserting success. Used as setup, not as the behaviour under test.
fn compress_ok(dir: &Path, extra: &[&str], name: &str) {
    let mut args: Vec<&str> = extra.to_vec();
    args.push(name);
    let out = run(dir, &args);
    assert!(
        out.status.success(),
        "setup: gzippy {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------
// Exit codes
// ---------------------------------------------------------------------------

#[test]
fn compress_success_exits_zero() {
    // gzip(1): "Exit status is normally 0".
    let d = tempdir();
    write_file(d.path(), "foo", &sample_bytes());
    let out = run(d.path(), &["foo"]);
    assert!(
        out.status.success(),
        "gzip(1) exit-status clause: successful compression must exit 0; got {:?}, stderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn decompress_of_garbage_exits_nonzero() {
    // gzip(1): "if an error occurs ... exit status is 1" — invalid input is
    // an error, and the invalid file must be left in place.
    let d = tempdir();
    let garbage = b"this is not a gzip stream at all, not even close";
    write_file(d.path(), "garbage.gz", garbage);
    let out = run(d.path(), &["-d", "garbage.gz"]);
    assert!(
        !out.status.success(),
        "gzip(1) error-exit clause: -d of non-gzip bytes must exit nonzero"
    );
    assert_eq!(
        fs::read(d.path().join("garbage.gz")).expect("garbage.gz still present"),
        garbage,
        "gzip(1) error clause: a failed decompression must leave the input file untouched"
    );
}

#[test]
fn test_flag_on_truncated_stream_exits_nonzero() {
    // gzip(1) -t: "Test. Check the compressed file integrity" — a truncated
    // member fails the integrity check.
    let d = tempdir();
    write_file(d.path(), "foo", &sample_bytes());
    compress_ok(d.path(), &[], "foo");
    let full = fs::read(d.path().join("foo.gz")).expect("read foo.gz");
    assert!(
        full.len() > 20,
        "setup: compressed stream too short to truncate"
    );
    write_file(d.path(), "trunc.gz", &full[..full.len() / 2]);
    let out = run(d.path(), &["-t", "trunc.gz"]);
    assert!(
        !out.status.success(),
        "gzip(1) -t integrity clause: a truncated stream must fail -t with nonzero exit"
    );
}

#[test]
fn test_flag_on_valid_stream_exits_zero_and_writes_no_stdout() {
    // gzip(1) -t: integrity check only; nothing is decompressed to stdout.
    let d = tempdir();
    write_file(d.path(), "foo", &sample_bytes());
    compress_ok(d.path(), &["-k"], "foo");
    let out = run(d.path(), &["-t", "foo.gz"]);
    assert!(
        out.status.success(),
        "gzip(1) -t clause: a valid stream must pass -t with exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "gzip(1) -t clause: -t checks integrity only and must write nothing to stdout; got {} bytes",
        out.stdout.len()
    );
}

// ---------------------------------------------------------------------------
// In-place semantics: -c, default remove, -k, -d
// ---------------------------------------------------------------------------

#[test]
fn stdout_flag_writes_stdout_and_keeps_input() {
    // gzip(1) -c: "Write output on standard output; keep original files
    // unchanged."
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    let out = run(d.path(), &["-c", "foo"]);
    assert!(out.status.success(), "setup: -c compression failed");
    assert!(
        !out.stdout.is_empty(),
        "gzip(1) -c clause: compressed output must go to stdout"
    );
    assert_eq!(
        fs::read(d.path().join("foo")).expect("foo still present"),
        data,
        "gzip(1) -c clause: -c must keep the original file unchanged"
    );
    // The stdout bytes are a valid stream: pipe them back through -dc.
    let back = run_piped(d.path(), &["-dc"], &out.stdout);
    assert!(back.status.success(), "-c output must decompress cleanly");
    assert_eq!(
        back.stdout, data,
        "gzip(1) -c clause: the stdout stream must roundtrip to the input bytes"
    );
}

#[test]
fn default_compress_creates_gz_and_removes_input() {
    // gzip(1): "gzip ... replaces each file whose name ends in .gz ... Each
    // file is replaced by one with the extension .gz" — compression is
    // in-place: foo becomes foo.gz and foo is removed.
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    let out = run(d.path(), &["foo"]);
    assert!(out.status.success(), "setup: compression failed");
    assert!(
        d.path().join("foo.gz").exists(),
        "gzip(1) in-place clause: compressing foo must create foo.gz"
    );
    assert!(
        !d.path().join("foo").exists(),
        "gzip(1) in-place clause: compressing foo without -k/-c must remove foo"
    );
}

#[test]
fn keep_flag_keeps_input() {
    // gzip(1) -k: "Keep (don't delete) input files during compression or
    // decompression."
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    let out = run(d.path(), &["-k", "foo"]);
    assert!(out.status.success(), "setup: -k compression failed");
    assert!(
        d.path().join("foo.gz").exists(),
        "gzip(1) -k clause: -k still creates foo.gz"
    );
    assert_eq!(
        fs::read(d.path().join("foo")).expect("foo kept"),
        data,
        "gzip(1) -k clause: -k must keep the input file"
    );
}

#[test]
fn decompress_restores_file_and_removes_gz() {
    // gzip(1): "gunzip takes a list of files ... and replaces each file whose
    // name ends with .gz ... with a decompressed file without the original
    // extension."
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    compress_ok(d.path(), &[], "foo");
    let out = run(d.path(), &["-d", "foo.gz"]);
    assert!(out.status.success(), "setup: decompression failed");
    assert_eq!(
        fs::read(d.path().join("foo")).expect("foo restored"),
        data,
        "gzip(1) in-place clause: -d foo.gz must restore foo byte-exactly"
    );
    assert!(
        !d.path().join("foo.gz").exists(),
        "gzip(1) in-place clause: -d without -k/-c must remove foo.gz"
    );
}

// ---------------------------------------------------------------------------
// -f overwrite semantics
// ---------------------------------------------------------------------------

#[test]
fn without_force_refuses_to_overwrite_existing_gz() {
    // gzip(1): "If the output file already exists gzip would ask to overwrite
    // ... if ... not a terminal, the file is not overwritten" and the warning
    // exit path applies (GNU gzip exits 2 here; we require nonzero).
    let d = tempdir();
    write_file(d.path(), "foo", &sample_bytes());
    let sentinel = b"pre-existing target bytes".to_vec();
    write_file(d.path(), "foo.gz", &sentinel);
    let out = run(d.path(), &["foo"]);
    assert!(
        !out.status.success(),
        "gzip(1) overwrite clause: without -f and a non-tty stdin, an existing foo.gz must not be overwritten and the exit code must be nonzero"
    );
    assert_eq!(
        fs::read(d.path().join("foo.gz")).expect("foo.gz still present"),
        sentinel,
        "gzip(1) overwrite clause: the refused target must be untouched"
    );
    assert!(
        d.path().join("foo").exists(),
        "gzip(1) overwrite clause: the refused input must be left in place"
    );
}

#[test]
fn force_overwrites_existing_gz() {
    // gzip(1) -f: "Force compression or decompression even if ... the
    // corresponding file already exists".
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    write_file(d.path(), "foo.gz", b"stale target");
    let out = run(d.path(), &["-f", "foo"]);
    assert!(
        out.status.success(),
        "gzip(1) -f clause: -f must overwrite the existing target and exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The overwritten target is a real stream that restores the input.
    let dec = run(d.path(), &["-dc", "foo.gz"]);
    assert!(dec.status.success(), "-f output must decompress cleanly");
    assert_eq!(
        dec.stdout, data,
        "gzip(1) -f clause: the forced target must hold the newly compressed input"
    );
}

// ---------------------------------------------------------------------------
// Levels and thread flag
// ---------------------------------------------------------------------------

#[test]
fn levels_1_through_9_all_parse_and_roundtrip() {
    // gzip(1) -# : "Regulate the speed of compression using the specified
    // digit #, where -1 ... -9". Every level must be accepted and produce a
    // valid stream.
    let d = tempdir();
    let data = sample_bytes();
    for level in 1..=9u32 {
        let flag = format!("-{level}");
        let name = format!("f{level}");
        write_file(d.path(), &name, &data);
        let out = run(d.path(), &[&flag, "-c", &name]);
        assert!(
            out.status.success(),
            "gzip(1) level clause: {flag} must be accepted; stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let back = run_piped(d.path(), &["-dc"], &out.stdout);
        assert!(
            back.status.success() && back.stdout == data,
            "gzip(1) level clause: {flag} output must roundtrip byte-exactly"
        );
    }
}

#[test]
fn processes_flag_accepted_with_file() {
    // pigz(1) -p: "Allow up to n processes" — part of the drop-in surface for
    // pigz; the output is still ordinary gzip.
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    let out = run(d.path(), &["-p", "4", "-c", "foo"]);
    assert!(
        out.status.success(),
        "pigz(1) -p clause: '-p 4' with a file must be accepted; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let back = run_piped(d.path(), &["-dc"], &out.stdout);
    assert!(
        back.status.success() && back.stdout == data,
        "pigz(1) -p clause: -p 4 output must roundtrip byte-exactly"
    );
}

// ---------------------------------------------------------------------------
// Multi-member streams
// ---------------------------------------------------------------------------

#[test]
fn concatenated_members_decompress_to_concatenation() {
    // gzip(1) ADVANCED USAGE: "Multiple compressed files can be concatenated.
    // In this case, gunzip will extract all members at once."
    let d = tempdir();
    let a = b"first member payload AAAA\n".to_vec();
    let b = b"second member payload BBBB\n".to_vec();
    write_file(d.path(), "a", &a);
    write_file(d.path(), "b", &b);
    let ga = run(d.path(), &["-c", "a"]);
    let gb = run(d.path(), &["-c", "b"]);
    assert!(
        ga.status.success() && gb.status.success(),
        "setup: member compression failed"
    );
    let mut cat = ga.stdout.clone();
    cat.extend_from_slice(&gb.stdout);
    write_file(d.path(), "cat.gz", &cat);
    let out = run(d.path(), &["-dc", "cat.gz"]);
    assert!(
        out.status.success(),
        "gzip(1) multi-member clause: -d of concatenated members must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut expected = a;
    expected.extend_from_slice(&b);
    assert_eq!(
        out.stdout, expected,
        "gzip(1) multi-member clause: decompressing catted members must yield the concatenated payloads"
    );
}

// ---------------------------------------------------------------------------
// stdin/stdout piping
// ---------------------------------------------------------------------------

#[test]
fn stdin_stdout_pipe_roundtrips() {
    // gzip(1): "If no files are specified ... the standard input is
    // compressed to the standard output."
    let d = tempdir();
    let data = sample_bytes();
    let comp = run_piped(d.path(), &["-c"], &data);
    assert!(
        comp.status.success() && !comp.stdout.is_empty(),
        "gzip(1) stdin clause: 'gzippy -c < in' must compress stdin to stdout; stderr: {}",
        String::from_utf8_lossy(&comp.stderr)
    );
    let dec = run_piped(d.path(), &["-dc"], &comp.stdout);
    assert!(
        dec.status.success(),
        "gzip(1) stdin clause: 'gzippy -dc < out.gz' must decompress stdin; stderr: {}",
        String::from_utf8_lossy(&dec.stderr)
    );
    assert_eq!(
        dec.stdout, data,
        "gzip(1) stdin clause: pipe compress-then-decompress must roundtrip byte-exactly"
    );
}

// ---------------------------------------------------------------------------
// Original-filename semantics (-N / -n are documented in `gzippy --help`)
// ---------------------------------------------------------------------------

#[test]
fn compress_then_decompress_yields_original_name() {
    // gzip(1) in-place naming: compressing foo then decompressing foo.gz
    // yields foo again.
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    compress_ok(d.path(), &[], "foo");
    let out = run(d.path(), &["-d", "foo.gz"]);
    assert!(out.status.success(), "setup: decompression failed");
    assert_eq!(
        fs::read(d.path().join("foo")).expect("foo present"),
        data,
        "gzip(1) naming clause: compress foo / decompress foo.gz must yield foo with the original bytes"
    );
}

#[test]
fn dash_upper_n_restores_stored_name_on_decompress() {
    // gzip(1) -N: "When decompressing ... restore the original file name ...
    // if present". Both -N and -n are documented in `gzippy --help`, so this
    // surface is claimed.
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "orig", &data);
    compress_ok(d.path(), &["-N"], "orig");
    fs::rename(d.path().join("orig.gz"), d.path().join("renamed.gz")).expect("rename");
    let out = run(d.path(), &["-d", "-N", "renamed.gz"]);
    assert!(
        out.status.success(),
        "gzip(1) -N clause: -dN must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        fs::read(d.path().join("orig")).expect("stored name restored"),
        data,
        "gzip(1) -N clause: -dN must restore the file under its stored original name"
    );
    assert!(
        !d.path().join("renamed").exists(),
        "gzip(1) -N clause: -dN must not also create a file named after the .gz file"
    );
}

#[test]
fn dash_lower_n_omits_name_so_decompress_uses_on_disk_name() {
    // gzip(1) -n: "When compressing, do not save the original file name ...
    // When decompressing ... the name of the file being decompressed is used"
    // (minus the suffix). With no stored name there is nothing to restore, so
    // this holds on both sides of the divergence below.
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "orig", &data);
    compress_ok(d.path(), &["-n"], "orig");
    fs::rename(d.path().join("orig.gz"), d.path().join("renamed.gz")).expect("rename");
    let out = run(d.path(), &["-d", "renamed.gz"]);
    assert!(out.status.success(), "setup: -d of -n stream failed");
    assert_eq!(
        fs::read(d.path().join("renamed")).expect("on-disk-derived name"),
        data,
        "gzip(1) -n clause: with no stored name, -d must name the output after the .gz file minus suffix"
    );
}

#[test]
#[ignore]
// DIVERGENCE from gzip contract: plain `gzippy -d renamed.gz` restores the
// header-stored original filename ("orig"), while GNU gzip 1.14 (probed)
// derives the output name from the on-disk filename minus suffix ("renamed")
// unless -N is given. gzip(1): name restoration on decompression is the -N
// behaviour, not the default. Un-ignore when gzippy matches; the current
// behaviour produces a different filename than gzip for the same command
// line, which can surprise scripts.
fn plain_decompress_names_output_from_on_disk_filename() {
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "orig", &data);
    compress_ok(d.path(), &[], "orig"); // default compress stores the name
    fs::rename(d.path().join("orig.gz"), d.path().join("renamed.gz")).expect("rename");
    let out = run(d.path(), &["-d", "renamed.gz"]);
    assert!(out.status.success(), "setup: decompression failed");
    assert_eq!(
        fs::read(d.path().join("renamed")).expect("on-disk-derived name"),
        data,
        "gzip(1) default naming clause: plain -d must name the output after the .gz file minus suffix, not the stored header name"
    );
    assert!(
        !d.path().join("orig").exists(),
        "gzip(1) default naming clause: plain -d must not restore the stored header name without -N"
    );
}

// ---------------------------------------------------------------------------
// Suffix handling
// ---------------------------------------------------------------------------

#[test]
fn compressing_a_gz_file_warns_and_leaves_it_unchanged() {
    // gzip(1): "gzip ... will not compress a file that already has a .gz
    // suffix" — GNU gzip 1.14 (probed) prints "already has .gz suffix --
    // unchanged" and exits 0 with the input intact. gzippy matches: same
    // warning shape, exit 0, input untouched.
    let d = tempdir();
    let data = sample_bytes();
    write_file(d.path(), "foo", &data);
    compress_ok(d.path(), &["-k"], "foo");
    let gz_before = fs::read(d.path().join("foo.gz")).expect("read foo.gz");
    let out = run(d.path(), &["foo.gz"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "gzip(1) suffix clause: compressing foo.gz must be a no-op warning with exit 0 (GNU gzip 1.14 probed behaviour); stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        fs::read(d.path().join("foo.gz")).expect("foo.gz still present"),
        gz_before,
        "gzip(1) suffix clause: the .gz input must be left byte-identical"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("already has .gz suffix"),
        "gzip(1) suffix clause: the refusal must be announced on stderr; got: {stderr}"
    );
    assert!(
        !d.path().join("foo.gz.gz").exists(),
        "gzip(1) suffix clause: no double-suffixed output may be created"
    );
}
