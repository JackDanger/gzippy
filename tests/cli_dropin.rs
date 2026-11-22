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
//! not endorsed behaviour. Current divergence list: NONE. The last one —
//! plain `-d` restoring the header-stored filename — was fixed as issue #268:
//! on decompression plain `-d` names its output after the on-disk file minus
//! the suffix, and only an explicit `-N` consults the stored FNAME.
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
//
// gzip(1) contract (probed; issue #268): on decompression the stored FNAME is
// consulted ONLY under an explicit -N; plain -d derives the output name from
// the on-disk file minus the suffix. Compression still stores the original
// name by default, so `gzippy foo` echoes gzip's header bytes.
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
    // `-dN` would name it identically here.
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
// gzip(1) default naming clause (probed 1.14; issue #268): plain `-d` names
// the output after the on-disk file minus the suffix — the header-stored
// FNAME is ignored unless -N is passed. gzippy used to consult the FNAME
// by default, restoring "orig" where gzip produces "renamed"; scripts
// recompressing under a different filename got a different output filename
// than gzip for the same command line.
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

// ── `-v` route-assert stderr contract (the parity/Gate-4 census needle) ──────
//
// scripts/campaign/parity-census.sh requires every T>1 encode to NAME the
// requested thread count on **-v stderr** ("threads=N"), and Gate-4's
// shipped-flavor check the `path=`/`encode-path=` spellings. The print is
// gated `GZIPPY_DEBUG || -v` (src/utils.rs route_assert_enabled). The
// duplicate-storage falsifier this guards against is asserted IN A CHILD
// PROCESS (assert_cmd): a process-global OnceLock cannot be probed in-process
// (the test fleet shares it; 2026-11-12 CI run 43276266526 failed exactly
// that way in the first version of this guard).

#[test]
fn v_flag_prints_route_line_on_stderr() {
    let bytes = sample_bytes();
    // BB-sized input: the parallel pipeline's own -v output names the count.
    let mut big = bytes.clone();
    while big.len() < 400_000 {
        big.extend_from_slice(&big.clone());
    }
    let dir = tempdir();
    let file = write_file(dir.path(), "vroute.bin", &big);
    let dirp = dir.path();

    let with_v = run_piped(
        dirp,
        &["-6", "-p2", "-v", "-c", file.to_str().unwrap()],
        &[],
    );
    assert!(
        with_v.status.success(),
        "-v compress must succeed; stderr: {}",
        String::from_utf8_lossy(&with_v.stderr)
    );
    let err = String::from_utf8_lossy(&with_v.stderr);
    assert!(
        err.contains("threads=2") || err.contains("(2 threads)"),
        "-v stderr must name the requested thread count; got: {}",
        err
    );
    assert!(
        err.contains("path=") || err.contains("encode-path=") || err.contains("route="),
        "-v stderr must carry the route spelling Gate-4/parity parse; got: {}",
        err
    );

    let without_v = run_piped(dirp, &["-6", "-p2", "-c", file.to_str().unwrap()], &[]);
    assert_eq!(
        String::from_utf8_lossy(&without_v.stderr).len(),
        0,
        "plain compress must stay silent on stderr (no -v, no GZIPPY_DEBUG)"
    );
}

// ---------------------------------------------------------------------------
// -N must never let the header FNAME choose a directory (path traversal).
//
// gzip(1) 1.14 `get_method` ORIG_NAME branch: the stored name is reduced to
// its base name (`gzip_base_name`) before it replaces the on-disk file name,
// so the output always lands beside the input. Probed: `gzip -dNk a/b/x.gz`
// with FNAME "../../file" writes `a/b/file`. An archive is untrusted input —
// before this clause, `gzippy -dN` wrote attacker-chosen bytes to any
// relative or absolute path the user could write, then deleted the input.
// ---------------------------------------------------------------------------

/// A gzip member of `data` whose header carries `fname` verbatim.
fn gz_with_fname(dir: &Path, data: &[u8], fname: &[u8]) -> Vec<u8> {
    write_file(dir, "payload", data);
    compress_ok(dir, &["-n"], "payload");
    let gz = fs::read(dir.join("payload.gz")).expect("payload.gz");
    fs::remove_file(dir.join("payload.gz")).expect("rm payload.gz");
    assert_eq!(
        gz[3] & 0x1e,
        0,
        "setup: -n header must carry no optional fields"
    );
    let mut out = gz[..10].to_vec();
    out[3] |= 0x08; // FNAME
    out.extend_from_slice(fname);
    out.push(0);
    out.extend_from_slice(&gz[10..]);
    out
}

/// Every regular file under `root`, relative to it.
fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).expect("read_dir") {
            let p = e.expect("entry").path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p.strip_prefix(root).unwrap().to_path_buf());
            }
        }
    }
    out.sort();
    out
}

fn assert_fname_confined(fname: &[u8], expect_name: &str) {
    let d = tempdir();
    let inner = d.path().join("a").join("b");
    fs::create_dir_all(&inner).expect("mkdir a/b");
    let data = sample_bytes();
    let gz = gz_with_fname(&inner, &data, fname);
    fs::write(inner.join("evil.gz"), gz).expect("write evil.gz");
    let out = run(&inner, &["-d", "-N", "evil.gz"]);
    let shown = String::from_utf8_lossy(fname);
    assert!(
        out.status.success(),
        "-dN with FNAME {shown:?} must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let expect = PathBuf::from("a").join("b").join(expect_name);
    assert_eq!(
        files_under(d.path()),
        vec![expect.clone()],
        "-dN with FNAME {shown:?}: output must land beside the input as {expect:?} and nowhere else"
    );
    assert_eq!(fs::read(d.path().join(&expect)).unwrap(), data);
}

#[test]
fn dash_upper_n_relative_traversal_fname_stays_beside_input() {
    assert_fname_confined(b"../../escaped", "escaped");
}

#[test]
fn dash_upper_n_absolute_fname_stays_beside_input() {
    let outside = tempdir();
    let target = outside.path().join("absolute-target");
    assert_fname_confined(target.to_str().unwrap().as_bytes(), "absolute-target");
    assert!(
        !target.exists(),
        "-dN must never write the absolute FNAME path"
    );
}

#[test]
fn dash_upper_n_dotdot_or_dir_only_fname_falls_back_to_on_disk_name() {
    // A stored name with no usable base name ("..", ".", "x/") restores
    // nothing; the output is named from the on-disk file, as with no FNAME.
    for f in [&b".."[..], b".", b"dir/", b"../"] {
        assert_fname_confined(f, "evil");
    }
}

// ---------------------------------------------------------------------------
// Output-file creation (gzip.c create_outfile / check_ofname / copy_stat).
//
// The output is opened O_CREAT|O_EXCL, mode 0600, so a symlink or hardlink
// planted at the output name is never written through; under -f (or a "y")
// the planted entry is unlinked — the link, never its target. Metadata is
// copied on the output FD, never by path. Each case was a write-through bug
// before this clause (probed against gzip 1.14, which leaves the victim
// intact in every case).
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn planted_output_link_is_never_written_through() {
    use std::os::unix::fs::symlink;
    let data = sample_bytes();
    // (args, input name, output name, prepare input)
    for (force, decompress) in [(false, false), (true, false), (false, true), (true, true)] {
        for hard in [false, true] {
            if hard && !force {
                continue; // an existing regular file is simply "already exists"
            }
            let d = tempdir();
            let victim = d.path().join("victim");
            fs::write(&victim, b"VICTIM").unwrap();
            write_file(d.path(), "foo", &data);
            let (input, output) = if decompress {
                compress_ok(d.path(), &[], "foo");
                ("foo.gz", "foo")
            } else {
                ("foo", "foo.gz")
            };
            let out_path = d.path().join(output);
            if hard {
                fs::hard_link(&victim, &out_path).unwrap();
            } else if force {
                symlink(&victim, &out_path).unwrap();
            } else {
                // dangling: Path::exists() reads it as absent
                symlink(d.path().join("created-through-link"), &out_path).unwrap();
            }
            let mut args = vec![];
            if decompress {
                args.push("-d");
            }
            if force {
                args.push("-f");
            }
            args.push(input);
            let out = run(d.path(), &args);
            let case = format!("args={args:?} hardlink={hard}");
            assert_eq!(
                fs::read(&victim).unwrap(),
                b"VICTIM",
                "{case}: victim written through"
            );
            assert!(
                !d.path().join("created-through-link").exists(),
                "{case}: file created through a dangling link"
            );
            if force {
                assert!(
                    out.status.success(),
                    "{case}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let meta = fs::symlink_metadata(&out_path).unwrap();
                assert!(
                    meta.file_type().is_file(),
                    "{case}: output must be a fresh regular file"
                );
            } else {
                assert_eq!(
                    out.status.code(),
                    Some(2),
                    "{case}: must refuse, gzip exit 2"
                );
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn output_takes_input_group_and_mode_on_the_fd() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let d = tempdir();
    write_file(d.path(), "f", &sample_bytes());
    fs::set_permissions(d.path().join("f"), fs::Permissions::from_mode(0o640)).unwrap();
    let istat = fs::metadata(d.path().join("f")).unwrap();
    compress_ok(d.path(), &[], "f");
    let o = fs::metadata(d.path().join("f.gz")).unwrap();
    assert_eq!(o.mode() & 0o7777, 0o640, "copy_stat: mode");
    assert_eq!(
        o.gid(),
        istat.gid(),
        "copy_stat: group (fchown before fchmod)"
    );
}

#[cfg(unix)]
#[test]
fn dash_upper_n_onto_own_archive_is_refused_and_keeps_the_input() {
    let d = tempdir();
    let gz = gz_with_fname(d.path(), &sample_bytes(), b"arc.gz");
    fs::write(d.path().join("arc.gz"), &gz).unwrap();
    let out = run(d.path(), &["-d", "-N", "-f", "arc.gz"]);
    assert_eq!(out.status.code(), Some(1), "gzip.c same_file: ERROR class");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("cannot decompress onto itself"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        fs::read(d.path().join("arc.gz")).unwrap(),
        gz,
        "input must survive intact"
    );
}

#[test]
fn plain_dash_d_ignores_header_mtime() {
    // gzip(1): the stored MTIME, like FNAME, is restored only under -N
    // (issue #268's clause). Plain -d keeps the .gz file's own mtime.
    let d = tempdir();
    let mut gz = gz_with_fname(d.path(), &sample_bytes(), b"x");
    gz[4..8].copy_from_slice(&1u32.to_le_bytes());
    fs::write(d.path().join("t.gz"), &gz).unwrap();
    let gz_mtime = fs::metadata(d.path().join("t.gz"))
        .unwrap()
        .modified()
        .unwrap();
    let out = run(d.path(), &["-dk", "t.gz"]);
    assert!(out.status.success());
    assert_eq!(
        fs::metadata(d.path().join("t"))
            .unwrap()
            .modified()
            .unwrap(),
        gz_mtime
    );
    fs::remove_file(d.path().join("x")).ok();
    let out = run(d.path(), &["-dkN", "t.gz"]);
    assert!(out.status.success());
    let restored = fs::metadata(d.path().join("x"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(
        restored,
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(1)
    );
}

#[cfg(unix)]
#[test]
fn index_output_link_is_never_written_through() {
    let d = tempdir();
    write_file(d.path(), "ix", &sample_bytes());
    compress_ok(d.path(), &[], "ix");
    let victim = d.path().join("victim");
    fs::write(&victim, b"VICTIM").unwrap();
    std::os::unix::fs::symlink(&victim, d.path().join("ix.gz.gzidx")).unwrap();
    let out = run(d.path(), &["--index", "ix.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(fs::read(&victim).unwrap(), b"VICTIM");
    assert!(fs::symlink_metadata(d.path().join("ix.gz.gzidx"))
        .unwrap()
        .is_file());
}

// ---------------------------------------------------------------------------
// -S validation and compress-side naming (gzip.c main / make_ofname).
// Probed gzip 1.14 for every expectation below.
// ---------------------------------------------------------------------------

#[test]
fn suffix_is_validated_up_front_like_gzip() {
    let d = tempdir();
    write_file(d.path(), "foo", b"a\n");
    let long = "a".repeat(31);
    for (s, msg) in [
        ("/x", "suffix contains '/'"),
        ("a/b", "suffix contains '/'"),
        ("", "invalid suffix ''"),
        (long.as_str(), "invalid suffix"),
    ] {
        for mode in [&[][..], &["-d"][..], &["-c"][..]] {
            let mut args: Vec<&str> = mode.to_vec();
            args.extend(["-S", s, "foo"]);
            let out = run(d.path(), &args);
            assert_eq!(
                out.status.code(),
                Some(1),
                "-S {s:?} {mode:?}: gzip exits 1"
            );
            assert!(
                String::from_utf8_lossy(&out.stderr).contains(msg),
                "-S {s:?} {mode:?}: stderr {:?}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    assert_eq!(
        fs::read_dir(d.path()).unwrap().count(),
        1,
        "nothing may be written"
    );
}

#[test]
fn compress_appends_the_suffix_verbatim() {
    // strcat(ofname, z_suffix): no extension arithmetic.
    let d = tempdir();
    for (name, extra, want) in [
        ("foo.", &[][..], "foo..gz"),
        ("bar", &["-S", "_z"][..], "bar_z"),
        ("baz", &["-S", "-z"][..], "baz-z"),
        ("quz", &["-S", "x"][..], "quzx"),
        ("a.tar", &[][..], "a.tar.gz"),
    ] {
        write_file(d.path(), name, b"a\n");
        let mut args = vec!["-k"];
        args.extend_from_slice(extra);
        args.push(name);
        let out = run(d.path(), &args);
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(d.path().join(want).is_file(), "{args:?} must write {want}");
    }
}

#[cfg(unix)]
#[test]
fn recursive_force_survives_a_symlink_loop() {
    // -rf follows symlinked directories (no O_NOFOLLOW under -f); a link
    // back to an ancestor used to recurse until the stack overflowed
    // (SIGABRT, exit 134) before compressing anything.
    let d = tempdir();
    let sub = d.path().join("tree").join("sub");
    fs::create_dir_all(&sub).unwrap();
    fs::write(d.path().join("tree").join("o"), b"o\n").unwrap();
    fs::write(sub.join("s"), b"s\n").unwrap();
    std::os::unix::fs::symlink("..", sub.join("loop")).unwrap();
    let out = run(d.path(), &["-rf", "tree"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("directory loop"));
    assert!(d.path().join("tree/o.gz").is_file());
    assert!(sub.join("s.gz").is_file());
}

// ---------------------------------------------------------------------------
// Redirected stdin honours the fd's current offset (POSIX read semantics;
// gzip reads from wherever the shell left the file). The T>1 route mmapped
// stdin from byte 0 and so compressed bytes another reader had already
// consumed — a VALID gzip of the WRONG content.
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn redirected_stdin_starts_at_the_fd_offset() {
    use std::io::{Seek, SeekFrom};
    let d = tempdir();
    let mut data = Vec::new();
    while data.len() < 300_000 {
        data.extend_from_slice(&sample_bytes());
    }
    let path = write_file(d.path(), "in", &data);
    for p in ["-p1", "-p4"] {
        let mut f = fs::File::open(&path).unwrap();
        f.seek(SeekFrom::Start(1000)).unwrap();
        let out = Command::new(gzippy())
            .args(["-c", "-6", p])
            .stdin(Stdio::from(f))
            .output()
            .unwrap();
        assert!(out.status.success());
        let gz = write_file(d.path(), "o.gz", &out.stdout);
        let back = run(
            d.path(),
            &["-dc", gz.file_name().unwrap().to_str().unwrap()],
        );
        assert!(
            back.stdout == data[1000..],
            "{p}: must compress from the fd offset"
        );
    }
}
