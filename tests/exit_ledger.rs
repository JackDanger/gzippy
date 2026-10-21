//! Exit-code ledger and error-output contract vs GNU gzip's CLI.
//!
//! gzip.1 documents the exit-status classes ("Exit status is normally 0; if
//! an error occurs, exit status is 1. If a warning occurs, exit status is
//! 2." — vendored `vendor/gzip/gzip.1:395-396`), and the vendored source
//! fixes the message shapes: `progerror` = `program: pathname:
//! strerror(errno)` (gzip.c:389-394); the WARN ledger escalation
//! (`if (exit_code == OK) exit_code = WARNING;` — gzip.h:263-264 — while
//! ERROR is set unconditionally); `get_method`'s container errors
//! (gzip.c:1473-1475); `read_error`'s EOF-in-magic (util.c:394-402);
//! `open_input_file`'s decompress-mode suffix ladder (gzip.c:1072-1163); and
//! `open_and_stat`'s O_NOFOLLOW link refusal (gzip.c:1022-1049).
//!
//! EVERY pin below was probed against the local `gzip` binary (1.12, this
//! box) BEFORE being written; where vendored source and the local binary
//! agree the pin is exact (message text), where they diverge the case lives
//! in the triage list instead. The probe transcript (each case: gzip's
//! observed stderr/exit ↔ gzippy's after the fix) is in the commit message
//! of the change that introduced this suite.
//!
//! ## Changed (this suite pins the fix)
//!   * missing input → `gzippy: <path>: No such file or directory` (pure
//!     strerror text, no duplicated pathname), exit 1 — compress names the
//!     GIVEN path; -d/-t/-l retry the gzip suffix ladder and name the LAST
//!     TRIED path (`-dc nofile` reports "nofile.gz", gzip.c:1147-1151).
//!   * symlink refusal without -c/-f is ELOOP's strerror text ("Too many
//!     levels of symbolic links"), ERROR class — exit 1, not the old
//!     skip-warning at exit 2 (open_and_stat, gzip.c:1022-1049). -c/-f/-t
//!     follow the link.
//!   * `unknown suffix -- ignored` is gzip's WARNING (exit 2) for -d on a
//!     suffix-less input; previously we tried to decode and exited 1 with a
//!     duplicated "not in gzip format" message.
//!   * "not in gzip format" (get_method gzip.c:1473-1475) and "unexpected
//!     end of file" (util.c:394-402) are classified at the container
//!     boundary (no dup prefix, gzip's leading blank line) for -d/-dc/-t
//!     and -l; and the exit-class integrity bug `gzippy -t junk` returning
//!     0 is pinned to gzip's error exit.
//!   * compress "already has SUFFIX suffix -- unchanged" recognizes gzip's
//!     full get_suffix table (case-insensitive `.gz .z .taz .tgz -gz -z
//!     _z`, gzip.c:987-991), prints the matched tail in ORIGINAL case
//!     without a colon after the name, exits 0, and does NOT bump the
//!     ledger (make_ofname prints without WARN on purpose, gzip.c:1204-1215).
//!   * non-regular-file skip uses gzip's text ("is not a directory or a
//!     regular file - ignored", WARNING) and only fires when NOT going to
//!     stdout (`gzip -c /dev/null` compresses zero bytes, probed).
//!   * usage errors render GNU getopt's exact lines plus `Try 'gzippy
//!     --help' for more information.` and exit 1 (getopt + try_help,
//!     gzip.c:315-319).
//!   * the empty-invocation TTY refusal carries gzip's "For help, type:
//!     gzippy -h" second line (treat_stdin, gzip.c:610-620).
//!   * the per-file ledger escalates EXACTLY like gzip's (error 1 always
//!     wins over warning 2; warnings only from OK), including inside -r.
//!   * the treat_file mode-bit ladder (gzip.c:737-768) fires entirely under
//!     `!to_stdout`: set-user-ID / set-group-ID refusals ("is set-user-ID on
//!     execution - ignored", exit 2, no -f escape), then -f-escapable sticky
//!     ("has the sticky bit set - file ignored") and hardlink ("has N other
//!     link[s] -- file ignored") refusals — compress AND decompress; -c, -t
//!     and -l read through. The old gzippy had only a misworded hardlink
//!     check in compress ("has N other links -- skipping (use -f to force)",
//!     wrong shape: gzip has no colon after the name) and none of the rest,
//!     and old code re-statted the path instead of using the open's stat.
//!     Message text follows the RESTORED generation (gzip 1.12+: "-- file
//!     ignored"; the vendored 1.6 copy still has the pre-2022 "-- unchanged"
//!     wording, same class and same "link"/"links" pluralization).
//!   * `gzip -l` of a truncated or too-short archive reports read_error's
//!     class "unexpected end of file" (probed 2/9/20-byte headers), not the
//!     old blanket "not in gzip format"; garbage magic keeps "not in gzip
//!     format"; the ≥18-byte ISIZE-slice panic window is gone.
//!   * with the gzip magic present, get_method's own header WALK stays
//!     ERROR-class faithfully: EOF before the variable header's end is
//!     read_error's blank-lined "unexpected end of file" (probed a 2-byte
//!     `\x1f\x8b`: the parallel decoder must not report "invalid gzip
//!     header"; it used to), and a CM byte other than 8 is "unknown method
//!     N -- not supported" — no blank line (gzip.c:1449-1456, probed).
//!   * ERROR-class messages print UNCONDITIONALLY: `-q` silences only WARN
//!     messages (gzip.h:263-265 — the WARN macro carries "if (!quiet)",
//!     read_error/get_method do not; probed `-q -d fake.gz` still prints).
//!   * a build without the dev-only `l1-tune` feature rejects `--tune=x`
//!     exactly like any other unknown long option ("unrecognized option
//!     '--tune=x'" + try_help), not the old bespoke "Unknown option" line.
//!
//! ## Already-matching pins (accepted before this change, guarded now)
//!   * `is a directory -- ignored` → exit 2 with this exact text — for the
//!     compress arm and every -d/-dc/-t/-l shape.
//!   * already-exists refusal (`already exists;\tnot overwritten`) → exit 2
//!     (failure_ux.rs covers the file-state side).
//!   * -V/-h precede files and exit 0.
//!   * permission-denied → "Permission denied", exit 1 (failure_ux.rs pins
//!     the file-state side; here the perror-grade TEXT is pinned).
//!
//! ## triage: skipped, reason X
//!   * decode-INTERNAL corruption messages (truncated tail, bad deflate
//!     data): ours come from the decoder ("Decompression error: ..."),
//!     gzip's from its decoder ("invalid compressed data--format
//!     violated"); the exit classes (1) match, the texts are our decoder's
//!     own — out of the CLI layer this change owns.
//!   * `-l` truncation BLIND SPOT: an 18+ byte .gz whose HEADER
//!     is incomplete (FNAME/FEXTRA cut off) still lists fabricated ISIZE
//!     values (exit 0) — real gzip read_errors on it. Closing this needs a
//!     full header PARSE in do_list, which is a feature-shaped change, not
//!     an exit-code message fix; left as documented divergence.
//!   * `-Z`/`--lzw` output unsupported — a FEATURE gap (gzip 1.12+ supports
//!     LZW), not an error-text one; the getopt refusal stays (exit 1).
//!   * gzip's PACK/LZH/zip container support (magic 0x1f9d/0x1fa0/PK\003\004
//!     decode to .taz/.Z work-alikes) is a feature gap: our container check
//!     reports those "not in gzip format" (exit 1) like pigz does.
//!
//! Runtime budget: no large inputs, no signals — well under 60 s.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::process::{Command, Output, Stdio};

fn gzippy() -> &'static str {
    env!("CARGO_BIN_EXE_gzippy")
}

fn tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("create tempdir")
}

/// Run gzippy with `args` in `dir`, stdin closed, C locale (strerror text is
/// locale-dependent).
fn run(dir: &std::path::Path, args: &[&str]) -> Output {
    Command::new(gzippy())
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .env("LC_ALL", "C")
        .output()
        .expect("failed to spawn gzippy")
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Compressible-but-not-trivial synthetic payload.
fn payload() -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog repeatedly. ".repeat(1024)
}

/// A tiny hand-built gzip member (fixed header, one STORED deflate block)
/// whose CRC32/ISIZE are correct, so any decoder accepts it. Used only for
/// fixtures; the SUITES THAT PIN CLI SHAPES must not depend on gzippy's own
/// encoder, and this avoids spawning a plugin dependency.
fn build_gzip_fixture(payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() <= 65_535, "fixture must fit one stored block");
    let mut out = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
    let len = payload.len() as u16;
    out.push(0x01); // BFINAL=1, BTYPE=00 (stored)
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc32(payload).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// 1. Missing file: strerror text, ladder naming
// ---------------------------------------------------------------------------

/// gzip (probed): `gzip -dc nofile.gz` → "gzip: nofile.gz: No such file or
/// directory", exit 1 — progerror's `program: pathname: strerror(errno)`
/// shape (gzip.c:389-394), NOT a gzippy-specific wording, never a duplicated
/// pathname. The compress arm names the GIVEN path (open_input_file never
/// retries suffixes when !decompress — gzip.c:1100-1101).
#[test]
fn missing_file_error_is_perror_text_exit_1() {
    let dir = tempdir();

    let out = run(dir.path(), &["-dc", "nofile.gz"]);
    assert_eq!(out.status.code(), Some(1), "gzip exits 1 (ERROR class)");
    assert_eq!(
        stderr_of(&out),
        "gzippy: nofile.gz: No such file or directory\n",
        "progerror shape, strerror text, no duplicates"
    );

    let out = run(dir.path(), &["nofile.txt"]); // compress arm
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "gzippy: nofile.txt: No such file or directory\n"
    );
}

/// gzip (probed): in DECOMPRESS mode a missing SUFFIX-LESS name is retried
/// against the suffix ladder — `gzip -dc nofile` reports
/// "nofile.gz: No such file or directory", naming the z_suffix-appended path
/// ("No suffix found, complain using z_suffix", gzip.c:1147-1151). Same for
/// -d, -l and -t.
#[test]
fn decompress_missing_bare_name_reports_the_suffix_appended_ladder() {
    let dir = tempdir();
    for args in [
        ["-dc", "nofile"],
        ["-d", "nofile"],
        ["-l", "nofile"],
        ["-t", "nofile"],
    ] {
        let out = run(dir.path(), &args);
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        assert_eq!(
            stderr_of(&out),
            "gzippy: nofile.gz: No such file or directory\n",
            "{args:?}: the complaint names the z_suffix attempt (gzip.c:1151)"
        );
    }
}

/// `-S .foo`: the ladder starts with z_suffix and the complaint names it
/// (probed: `gzip -S .foo -dc nofile` → "gzip: nofile.foo: No such file or
/// directory").
#[test]
fn missing_bare_name_with_custom_suffix_names_that_suffix() {
    let dir = tempdir();
    let out = run(dir.path(), &["-S", ".foo", "-dc", "nofile"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "gzippy: nofile.foo: No such file or directory\n"
    );
}

/// The ladder's other side: `gzip -d plain.txt` where ONLY plain.txt.gz
/// exists opens the suffixed archive and decompresses it to plain.txt (the
/// resolved name becomes the operand for output naming). Before the fix we
/// misreported "File not found: plain.txt" even though the archive was
/// addressable under gzip's contract (probed: real gzip decompresses).
#[test]
fn decompress_finds_the_archive_under_the_suffix_appended_name() {
    let dir = tempdir();
    let payload = payload();
    let src = dir.path().join("plain.txt");
    fs::write(
        dir.path().join("plain.txt.gz"),
        build_gzip_fixture(&payload),
    )
    .expect("write fixture archive");

    let out = run(dir.path(), &["-d", "plain.txt"]);
    assert!(
        out.status.success(),
        "-d must find plain.txt.gz: {:?}",
        out.status.code()
    );
    assert_eq!(fs::read(&src).expect("read restored"), payload);
    assert!(
        !dir.path().join("plain.txt.gz").exists(),
        "archive consumed (gzip's in-place -d contract)"
    );
}

// ---------------------------------------------------------------------------
// 2. Symlinks: ELOOP refusal (ERROR class), follow under -c/-f/-t
// ---------------------------------------------------------------------------

/// gzip (probed): `gzip link` and `gzip -d link.gz` = "Too many levels of
/// symbolic links", exit 1 — the O_NOFOLLOW refusal is an ERROR, NOT the
/// warning-at-2 the old gzippy printed; a DANGLING link hits ELOOP before
/// ENOENT (open_and_stat with O_NOFOLLOW, gzip.c:1022-1049).
#[test]
fn symlink_refusal_is_the_eloop_error_not_a_skip_warning() {
    let dir = tempdir();
    fs::write(dir.path().join("real.txt"), payload()).expect("real.txt");
    fs::write(dir.path().join("real.gz"), build_gzip_fixture(&payload())).expect("real.gz");
    symlink("real.txt", dir.path().join("use-link.txt")).expect("symlink");
    symlink("real.gz", dir.path().join("use-link.gz")).expect("symlink .gz");
    symlink("nowhere", dir.path().join("dangling.txt")).expect("dangling");

    let out = run(dir.path(), &["use-link.txt"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "compress of a link = ERROR class"
    );
    assert_eq!(
        stderr_of(&out),
        "gzippy: use-link.txt: Too many levels of symbolic links\n"
    );

    let out = run(dir.path(), &["-d", "use-link.gz"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "decompress of a link = ERROR class"
    );
    assert_eq!(
        stderr_of(&out),
        "gzippy: use-link.gz: Too many levels of symbolic links\n"
    );

    let out = run(dir.path(), &["dangling.txt"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "gzippy: dangling.txt: Too many levels of symbolic links\n",
        "dangling link: ELOOP precedes ENOENT (the link refusal comes first)"
    );
}

/// gzip (probed): under -c or -f the link IS followed and the operation
/// proceeds (`gzip -c link` compresses the target, exit 0), matching
/// open_and_stat's to_stdout/force escape (gzip.c:1022-1025). -t sets
/// to_stdout internally (gzip.c:464-465), so `gzip -t link.gz` tests the
/// TARGET.
#[test]
fn links_are_followed_under_force_stdout_and_test() {
    let dir = tempdir();
    let payload_bytes = payload();
    fs::write(dir.path().join("real.txt"), &payload_bytes).expect("real.txt");
    fs::write(
        dir.path().join("realtext.gz"),
        build_gzip_fixture(&payload_bytes),
    )
    .expect("archive");
    symlink("real.txt", dir.path().join("link-c.txt")).expect("link for -c");
    symlink("realtext.gz", dir.path().join("link-t.gz")).expect("link for -t");

    // -c compresses THROUGH the link (to stdout here) and exits 0.
    let out = run(dir.path(), &["-c", "link-c.txt"]);
    assert!(out.status.success(), "-c follows links: {out:?}");
    assert!(!out.stdout.is_empty(), "-c wrote output through the link");
    assert!(
        out.stdout.len() < payload_bytes.len(),
        "-c wrote compressed (smaller) bytes through the link"
    );
    assert!(dir.path().join("real.txt").exists(), "-c keeps the target");

    // -t tests the archive behind the link → OK, silent.
    let out = run(dir.path(), &["-t", "link-t.gz"]);
    assert!(out.status.success(), "-t follows links (probed): {out:?}");
    assert!(
        !stderr_of(&out).contains("symbolic"),
        "-t must not emit the old link-skip warning"
    );

    // -f compresses through the link (the `gzip -f link` flow).
    let out = run(dir.path(), &["-f", "link-c.txt"]);
    assert!(out.status.success(), "-f follows links: {out:?}");
}

// ---------------------------------------------------------------------------
// 3. unknown suffix / already has suffix (get_suffix table)
// ---------------------------------------------------------------------------

/// gzip (probed): `gzip -d plain.txt` (existing, no .gz suffix) is skipped
/// as a WARNING — "unknown suffix -- ignored", exit 2 — and the file is
/// never read or touched. Even `-f` does not silence it (probed
/// `gzip -df plain.txt` warns, exit 2 — make_ofname has no force gate).
#[test]
fn decompress_of_suffixless_input_is_a_warning_not_a_decode_failure() {
    let dir = tempdir();
    let payload_bytes = payload();
    fs::write(dir.path().join("plain.txt"), &payload_bytes).expect("plain.txt");

    let out = run(dir.path(), &["-d", "plain.txt"]);
    assert_eq!(out.status.code(), Some(2), "WARNING class (probed)");
    assert_eq!(
        stderr_of(&out),
        "gzippy: plain.txt: unknown suffix -- ignored\n"
    );
    assert!(dir.path().join("plain.txt").exists(), "file left alone");
    assert_eq!(
        fs::read(dir.path().join("plain.txt")).unwrap(),
        payload_bytes
    );

    // -f keeps the unknown-suffix refusal for plain -d.
    let out = run(dir.path(), &["-d", "-f", "plain.txt"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr_of(&out).contains("unknown suffix -- ignored"));
}

/// gzip (probed): the compress-side no-op recognizes the FULL get_suffix
/// table case-insensitively and prints the matched tail in ORIGINAL case:
/// `gzip FOO.GZ` → "already has .GZ suffix -- unchanged" (exit 0), `gzip
/// old.z` → ".z suffix"; and there is NO colon after the name
/// (make_ofname compress arm, gzip.c:1207-1211).
#[test]
fn already_has_suffix_recognizes_the_whole_get_suffix_table() {
    let dir = tempdir();
    let payload_bytes = payload();

    for (name, expected) in [
        ("a.GZ", "gzippy: a.GZ already has .GZ suffix -- unchanged"),
        ("a.z", "gzippy: a.z already has .z suffix -- unchanged"),
        (
            "a.tgz",
            "gzippy: a.tgz already has .tgz suffix -- unchanged",
        ),
    ] {
        fs::write(dir.path().join(name), &payload_bytes).expect(name);
        let out = run(dir.path(), &[name]);
        assert_eq!(out.status.code(), Some(0), "{name}: no-op exits 0 (probed)");
        assert_eq!(stderr_of(&out), format!("{expected}\n"), "{name}");
        assert_eq!(
            fs::read(dir.path().join(name)).unwrap(),
            payload_bytes,
            "{name}: file unchanged"
        );
    }
}

/// The multi-file ledger variant of the no-op: "already has suffix" does NOT
/// bump the exit code (make_ofname prints WITHOUT WARN on purpose —
/// gzip.c:1213 "Don't use WARN, as it affects exit status"), so
/// `gzip x.GZ somedir` settles at the directory's WARNING=2 (not 0 and not
/// anything else).
#[test]
fn already_has_suffix_notice_does_not_bump_exit_ledger() {
    let dir = tempdir();
    fs::write(dir.path().join("x.GZ"), payload()).expect("x.GZ");
    fs::create_dir(dir.path().join("sub")).expect("subdir");

    let out = run(dir.path(), &["x.GZ", "sub"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "no-op is neutral; dir warning = 2"
    );
    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("already has .GZ suffix -- unchanged"),
        "{stderr:?}"
    );
    assert!(stderr.contains("is a directory -- ignored"), "{stderr:?}");
}

// ---------------------------------------------------------------------------
// 4. not in gzip format / unexpected end of file (get_method + read_error)
// ---------------------------------------------------------------------------

/// gzip (probed): first-member magic failure = "\ngzip: PATH: not in gzip
/// format" (a LEADING BLANK LINE then the message), exit 1 — for -dc, -t
/// and -l alike. (This also once fired with a doubled pathname:
/// "gzippy: f: f: not in gzip format".)
#[test]
fn not_in_gzip_format_matches_gzips_leading_blank_line_shape() {
    let dir = tempdir();
    let bad = b"not gzip data at all\n";
    fs::write(dir.path().join("fakegz.gz"), bad).expect("fakegz.gz");
    fs::write(dir.path().join("fake.txt"), bad).expect("plain file");

    for args in [
        ["-dc", "fakegz.gz"],
        ["-t", "fakegz.gz"],
        ["-l", "fakegz.gz"],
    ] {
        let out = run(dir.path(), &args);
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        assert_eq!(
            stderr_of(&out),
            "\ngzippy: fakegz.gz: not in gzip format\n",
            "{args:?}"
        );
    }

    // A suffix-less non-gzip name under -dc/-t: same shape, own name.
    for args in [["-dc", "fake.txt"], ["-t", "fake.txt"]] {
        let out = run(dir.path(), &args);
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        assert_eq!(
            stderr_of(&out),
            "\ngzippy: fake.txt: not in gzip format\n",
            "{args:?}"
        );
    }
}

/// gzip (probed): 0- and 1-byte inputs = "\ngzip: PATH: unexpected end of
/// file" (util.c:394-402), exit 1. The -t side used to IGNORE them (exit 0,
/// silence) — pinning the corrected verdict for -t and for -d/-dc alike.
#[test]
fn tiny_and_empty_inputs_report_unexpected_eof() {
    let dir = tempdir();
    fs::write(dir.path().join("e.txt"), b"").expect("empty file");
    fs::write(dir.path().join("one.bin"), b"a").expect("one byte");

    let out = run(dir.path(), &["-t", "e.txt"]);
    assert_eq!(out.status.code(), Some(1), "0-byte input is an ERROR");
    assert_eq!(stderr_of(&out), "\ngzippy: e.txt: unexpected end of file\n");

    let out = run(dir.path(), &["-t", "one.bin"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "\ngzippy: one.bin: unexpected end of file\n"
    );

    let out = run(dir.path(), &["-dc", "one.bin"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "\ngzippy: one.bin: unexpected end of file\n"
    );
}

/// gzip (probed): with the gzip MAGIC present, get_method keeps consuming
/// the variable header — EOF before the header's end is STILL read_error
/// ("unexpected end of file", exit 1), NOT a decoder message (probed:
/// `gzip -d` of exactly "\x1f\x8b" → "\ngzip: tiny.gz: unexpected end of
/// file"). gzippy used to hand such bytes to the parallel decoder, which
/// reported its own "invalid gzip header" instead.
#[test]
fn eof_inside_the_member_header_is_read_error() {
    let dir = tempdir();
    let good = build_gzip_fixture(&payload());
    fs::write(dir.path().join("two.gz"), &good[..2]).expect("2-byte header prefix");
    fs::write(dir.path().join("four.gz"), &good[..4]).expect("4-byte header prefix");

    for name in ["two.gz", "four.gz"] {
        let expected = format!("\ngzippy: {name}: unexpected end of file\n");
        for args in [vec!["-dc", name], vec!["-d", name], vec!["-t", name]] {
            let out = run(dir.path(), &args);
            assert_eq!(out.status.code(), Some(1), "{args:?}: ERROR class");
            assert_eq!(stderr_of(&out), expected, "{args:?}");
        }
    }

    // Same classification from stdin (gzip's peer name is "stdin").
    let mut child = Command::new(gzippy())
        .args(["-d"])
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    use std::io::Write as _;
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&good[..2])
        .expect("write 2 bytes");
    let out = child.wait_with_output().expect("wait");
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stderr_of(&out), "\ngzippy: stdin: unexpected end of file\n");
}

/// gzip (probed): a gzip-magic input whose CM byte is not 8 = "unknown
/// method %d -- not supported", exit 1 — get_method's own fprintf, NO
/// leading blank line, printed even under -q (ERROR classes carry no quiet
/// gate; only WARNs are silent under -q, gzip.h:263-265).
#[test]
fn unknown_method_byte_is_get_methods_own_refusal() {
    let dir = tempdir();
    // \x1f\x8b + CM=9, then a full 10-byte fixed header: the header
    // STRUCTURE would be complete if CM were DEFLATED(8).
    fs::write(
        dir.path().join("cm9.gz"),
        [0x1f, 0x8b, 0x09, 0x00, 0, 0, 0, 0, 0x00, 0xff],
    )
    .expect("cm9.gz");

    for args in [
        vec!["-d", "cm9.gz"],
        vec!["-dc", "cm9.gz"],
        vec!["-t", "cm9.gz"],
    ] {
        let out = run(dir.path(), &args);
        assert_eq!(out.status.code(), Some(1), "{args:?}: ERROR class");
        assert_eq!(
            stderr_of(&out),
            "gzippy: cm9.gz: unknown method 9 -- not supported\n",
            "{args:?}: plain fprintf, no blank line (gzip.c:1449-1456)"
        );
    }

    // -q does NOT silence ERROR-class messages (probed).
    let out = run(dir.path(), &["-q", "-d", "cm9.gz"]);
    assert_eq!(
        stderr_of(&out),
        "gzippy: cm9.gz: unknown method 9 -- not supported\n"
    );

    // ... nor the container errors (probed `gzip -q -d fake.gz` prints).
    fs::write(dir.path().join("fake.gz"), b"junk").expect("fake");
    let out = run(dir.path(), &["-q", "-d", "fake.gz"]);
    assert_eq!(
        stderr_of(&out),
        "\ngzippy: fake.gz: not in gzip format\n",
        "-q keeps ERROR-class stderr"
    );

    // While a WARN stays quiet under -q (probed `gzip -q /dev/null` → exit
    // 2, silence).
    let out = run(dir.path(), &["-q", "/dev/null"]);
    assert_eq!(out.status.code(), Some(2), "WARN class still exits 2");
    assert!(stderr_of(&out).is_empty(), "WARN silenced by -q");
}

/// gzip (probed): `gzip -tf junk` passes the bytes through with exit 0 and
/// silence (get_method's force pass-through, gzip.c:1461-1462); WITHOUT -f,
/// `gzip -t junk` errors. Our -t used to misreport garbage as OK even
/// without -f — both directions pinned here.
#[test]
fn test_integrity_of_garbage_errors_unless_forced() {
    let dir = tempdir();
    fs::write(dir.path().join("junk.txt"), b"plain text bytes\n").expect("junk");

    let out = run(dir.path(), &["-t", "junk.txt"]);
    assert_eq!(out.status.code(), Some(1), "gzip -t junk errors");
    assert!(stderr_of(&out).contains("not in gzip format"));

    let out = run(dir.path(), &["-tf", "junk.txt"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "force-test passes through silently (probed)"
    );
    assert!(stderr_of(&out).is_empty());
    assert!(dir.path().join("junk.txt").exists(), "input untouched");
}

/// gzip (probed): decompressing garbage / EMPTY stdin: "stdin: not in gzip
/// format" / "stdin: unexpected end of file" with the leading blank line,
/// exit 1 — the empty case previously exited 0 silently, and garbage hit
/// the decoder's own "invalid gzip header" text.
#[test]
fn decompress_stdin_classifies_containers_like_gzip() {
    let dir = tempdir();

    // Empty stdin, no force: EOF before the magic.
    let out = Command::new(gzippy())
        .args(["-d"])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .env("LC_ALL", "C")
        .output()
        .expect("spawn");
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stderr_of(&out), "\ngzippy: stdin: unexpected end of file\n");

    // Compressed pipe data in -d mode: sanity (exit 0, correct payload).
    use std::io::Write as _;
    let gz = build_gzip_fixture(b"the payload\n");
    let mut child = Command::new(gzippy())
        .args(["-d"])
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(&gz)
        .expect("write archive");
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "-d of a valid pipe must succeed");
    assert_eq!(out.stdout, b"the payload\n");
}

// ---------------------------------------------------------------------------
// 5. open-class failures carry the exact strerror text
// ---------------------------------------------------------------------------

/// gzip (probed): `gzip noread.txt` (mode 000) → "Permission denied", exit
/// 1. failure_ux.rs pins the file-state side of this failure; this pins the
/// program: path: strerror(errno) MESSAGE text.
#[test]
fn unreadable_file_reports_permission_denied_verbatim() {
    let dir = tempdir();
    let input = dir.path().join("noread.txt");
    fs::write(&input, payload()).expect("write");
    fs::set_permissions(&input, fs::Permissions::from_mode(0o000)).expect("chmod 000");

    let out = run(dir.path(), &["noread.txt"]);
    // Restore before asserting so tempdir cleanup works.
    fs::set_permissions(&input, fs::Permissions::from_mode(0o644)).expect("restore mode");

    assert_eq!(out.status.code(), Some(1), "error class");
    assert_eq!(stderr_of(&out), "gzippy: noread.txt: Permission denied\n");
}

/// gzip (probed): `gzip /dev/null` (char device, no -c) → "is not a
/// directory or a regular file - ignored", WARNING, exit 2 — gzip's exact
/// wording; `gzip -c /dev/null` compresses through (exit 0).
#[test]
fn special_file_notices_match_gzips_word_and_stdout_gate() {
    let dir = tempdir();
    let out = run(dir.path(), &["/dev/null"]);
    assert_eq!(out.status.code(), Some(2), "WARNING class (char device)");
    assert_eq!(
        stderr_of(&out),
        "gzippy: /dev/null is not a directory or a regular file - ignored\n"
    );

    let out = run(dir.path(), &["-c", "/dev/null"]);
    assert!(
        out.status.success(),
        "-c reads through the device (probed e0)"
    );
}

// ---------------------------------------------------------------------------
// 6. Directory skip + the multi-file exit ledger
// ---------------------------------------------------------------------------

/// gzip (probed): `is a directory -- ignored` → exit 2 (WARNING), remaining
/// files keep processing. The multi-file LEDGER: ERROR (1) always wins over
/// WARNING (2) regardless of argument order — gzip sets exit_code = ERROR
/// unconditionally while warnings only escalate from OK (gzip.h:263-264) —
/// and the run still processes every good file.
#[test]
fn exit_ledger_error_over_warning_in_both_orders() {
    let dir = tempdir();
    let payload_bytes = payload();
    fs::write(dir.path().join("good.txt"), &payload_bytes).expect("good.txt");
    fs::create_dir(dir.path().join("somedir")).expect("somedir");

    // Warning-only invocation → 2.
    let out = run(dir.path(), &["somedir"]);
    assert_eq!(out.status.code(), Some(2), "warning-only (probed)");

    // Decompress arm, both orders: missing-file ERROR + dir WARNING → 1.
    for args in [
        ["-dc", "nofile.gz", "somedir"],
        ["-dc", "somedir", "nofile.gz"],
    ] {
        let out = run(dir.path(), &args);
        assert_eq!(out.status.code(), Some(1), "{args:?}: ERROR wins (probed)");
        let stderr = stderr_of(&out);
        assert!(
            stderr.contains("No such file or directory"),
            "{args:?}: {stderr:?}"
        );
        assert!(stderr.contains("is a directory -- ignored"), "{args:?}");
    }

    // Compress arm with warning FIRST, error second, good file last: every
    // file is processed and the code lands on 1.
    let out = run(dir.path(), &["somedir", "nofile.txt", "good.txt"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "dir-warning first still loses to the error"
    );
    assert!(
        dir.path().join("good.txt.gz").exists(),
        "good file processed despite earlier failures"
    );
}

/// Inside -r (compress_directory) the same escalation rules apply: an ELOOP
/// refusal deep in the walk is an ERROR class result, not a warning.
#[test]
fn recursive_walk_uses_the_same_ledger_escalation() {
    let dir = tempdir();
    fs::create_dir(dir.path().join("walk")).expect("walk dir");
    fs::write(dir.path().join("walk/w1.txt"), payload()).expect("w1.txt");
    symlink("w1.txt", dir.path().join("walk/link.txt")).expect("link inside walk");

    // gzippy -r walk/: walkdir yields w1.txt and link.txt; the symlink fails
    // the ELOOP refusal → ERROR → exit 1 (was a naive overwrite).
    let out = run(dir.path(), &["-r", "walk"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "ELOOP is an ERROR class inside -r"
    );
    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("Too many levels of symbolic links"),
        "{stderr:?}"
    );
    assert!(
        dir.path().join("walk/w1.txt.gz").exists(),
        "regular file still processed"
    );
}

// ---------------------------------------------------------------------------
// 7. Usage errors: GNU getopt framing
// ---------------------------------------------------------------------------

/// gzip (probed): `gzip -Q` → "invalid option -- 'Q'" + "Try `gzip --help'
/// for more information." exit 1; `gzip --bar` → "unrecognized option
/// '--bar'" + Try (same tail); missing values: "option requires an argument
/// -- 'S'" and "option '--suffix' requires an argument". getopt renders the
/// first line, try_help (gzip.c:315-319) the second; do_exit(ERROR) exits 1.
#[test]
fn usage_errors_render_getopt_lines_with_try_help_and_exit_1() {
    let dir = tempdir();

    let out = run(dir.path(), &["-Q"]);
    assert_eq!(out.status.code(), Some(1), "ERROR class (do_exit(ERROR))");
    assert_eq!(
        stderr_of(&out),
        "gzippy: invalid option -- 'Q'\nTry `gzippy --help' for more information.\n"
    );

    let out = run(dir.path(), &["--bar"]);
    assert_eq!(
        stderr_of(&out),
        "gzippy: unrecognized option '--bar'\nTry `gzippy --help' for more information.\n"
    );

    let out = run(dir.path(), &["-S"]);
    assert_eq!(
        stderr_of(&out),
        "gzippy: option requires an argument -- 'S'\nTry `gzippy --help' for more information.\n"
    );

    let out = run(dir.path(), &["--suffix"]);
    assert_eq!(
        stderr_of(&out),
        "gzippy: option '--suffix' requires an argument\nTry `gzippy --help' for more information.\n"
    );
}

/// -h and -V precede any file processing and exit 0 (gzip.c:464/494:
/// help()/version() then do_exit(OK) during option parsing). Ours parses
/// the whole argv first, then acts on the flags — the observable precedence
/// matches (probed `gzip nofile -V` → version, exit 0).
#[test]
fn version_and_help_precede_files_and_exit_zero() {
    let dir = tempdir();
    let out = run(dir.path(), &["--version", "nofile"]);
    assert!(out.status.success(), "-V precedes files");
    assert!(String::from_utf8_lossy(&out.stdout).contains("gzippy"));

    let out = run(dir.path(), &["-h", "nofile"]);
    assert!(out.status.success(), "-h precedes files");
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage"));
}

// ---------------------------------------------------------------------------
// 8. Empty invocation on a TTY: gzip's two-line refusal
// ---------------------------------------------------------------------------

/// gzip (probed via util-linux `script` pty): the no-argument invocation on
/// a terminal prints TWO lines — "compressed data not written to a
/// terminal. Use -f to force compression." and "For help, type: gzip -h" —
/// and exits 1 (treat_stdin → do_exit(ERROR), gzip.c:610-621). Skips
/// cleanly where script(1) is unavailable.
#[test]
fn empty_invocation_on_a_tty_carries_the_for_help_line() {
    let script_ok = Command::new("script")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !script_ok {
        eprintln!("SKIP: script(1) unavailable (util-linux)");
        return;
    }

    let out = Command::new("script")
        .args(["-qec", gzippy(), "/dev/null"])
        .stdin(Stdio::null())
        .output()
        .expect("run gzippy under a pty");
    // script(1) forwards the pty traffic (gzippy's stderr) on its STDOUT;
    // the typescript goes to /dev/null, so the capture lives in stdout.
    let seen = String::from_utf8_lossy(&out.stdout);
    assert!(
        seen.contains("compressed data not written to a terminal. Use -f to force compression."),
        "{seen:?}"
    );
    assert!(
        seen.contains("For help, type: gzippy -h"),
        "gzip's second line (treat_stdin, gzip.c:615-617): {seen:?}"
    );
}

// ---------------------------------------------------------------------------
// 9. Mode-bit ladder: setuid/setgid/sticky/hardlink (gzip.c:737-768)
// ---------------------------------------------------------------------------

/// gzip (probed 1.12): `gzip hg.txt` where hg.txt has TWO links → "gzip:
/// hg.txt has 1 other link -- file ignored", exit 2 (WARNING class: WARN
/// escalates only from OK, gzip.h:263-264); `gzip -f hg.txt` compresses
/// (exit 0). NO colon after the name, singular "link" from nlink==2, plural
/// "links" from nlink==3 (treat_file:772-777: `link%s` with ""/n "s").
/// Released gzip (1.12/1.14) prints "-- file ignored" — the vendored 1.6
/// copy's "-- unchanged" wording predates the change; same class either way.
#[test]
fn hardlink_refusal_matches_released_gzips_text_and_yield() {
    let dir = tempdir();
    let payload_bytes = payload();

    fs::write(dir.path().join("hg.txt"), &payload_bytes).expect("hg.txt");
    fs::hard_link(dir.path().join("hg.txt"), dir.path().join("hg2.txt")).expect("second hardlink");

    // Compress arm: the exact single-other-link line.
    let out = run(dir.path(), &["hg.txt"]);
    assert_eq!(out.status.code(), Some(2), "hardlink refusal = WARNING");
    assert_eq!(
        stderr_of(&out),
        "gzippy: hg.txt has 1 other link -- file ignored\n"
    );
    assert!(
        !dir.path().join("hg.txt.gz").exists(),
        "input must be untouched"
    );

    // -f yields for the hardlink refusal (gzip.c:751 `if (! force)`) and
    // compresses in place: hg.txt is consumed, replaced by hg.txt.gz.
    let out = run(dir.path(), &["-f", "hg.txt"]);
    assert!(out.status.success(), "-f compresses a linked file");
    assert!(
        !dir.path().join("hg.txt").exists(),
        "compressed in place (input file consumed)"
    );
    assert!(
        dir.path().join("hg2.txt").exists(),
        "the other hardlink untouched"
    );

    // Decompress arm: treat_file's ladder is direction-agnostic (probed:
    // `gzip -d g.txt.gz` with a second link refuses identically), and -t —
    // internally to_stdout — reads THROUGH (probed `gzip -t h.gz` succeeds).
    fs::write(
        dir.path().join("hl.bin.gz"),
        build_gzip_fixture(&payload_bytes),
    )
    .expect("archive");
    fs::hard_link(dir.path().join("hl.bin.gz"), dir.path().join("hl2.bin.gz"))
        .expect("second link on the archive");

    let out = run(dir.path(), &["-d", "hl.bin.gz"]);
    assert_eq!(out.status.code(), Some(2), "-d refuses a linked archive");
    assert_eq!(
        stderr_of(&out),
        "gzippy: hl.bin.gz has 1 other link -- file ignored\n"
    );

    let out = run(dir.path(), &["-t", "hl.bin.gz"]);
    assert!(
        out.status.success(),
        "-t reads through (to_stdout ladder gate)"
    );

    // -c also reads through: the whole ladder is under `if (! to_stdout)`.
    let out = run(dir.path(), &["-c", "hg2.txt"]);
    assert!(
        out.status.success(),
        "-c ignores the mode-bit ladder: {out:?}"
    );

    // Three links → plural.
    fs::write(dir.path().join("tr.txt"), &payload_bytes).expect("tr.txt");
    fs::hard_link(dir.path().join("tr.txt"), dir.path().join("tr2.txt")).expect("link 2");
    fs::hard_link(dir.path().join("tr.txt"), dir.path().join("tr3.txt")).expect("link 3");
    let out = run(dir.path(), &["tr.txt"]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        stderr_of(&out),
        "gzippy: tr.txt has 2 other links -- file ignored\n"
    );
}

/// gzip (probed 1.12): set-user-ID and set-group-ID inputs are refused with
/// "is set-user-ID on execution - ignored" / "is set-group-ID on execution
/// - ignored", exit 2, REGARDLESS of -f (gzip.c:745-759 — the checks sit
/// outside the force gate); sticky ("has the sticky bit set - file ignored")
/// and the hardlink refusals yield to -f. Order: the regular-file check
/// beats the mode-bit checks (a setuid FIFO reports the FIFO wording).
#[test]
fn setuid_setgid_sticky_refusals_match_gzips_ladder() {
    let dir = tempdir();
    let payload_bytes = payload();

    let write_mode = |name: &str, mode: u32| {
        let p = dir.path().join(name);
        fs::write(&p, &payload_bytes).expect(name);
        fs::set_permissions(&p, fs::Permissions::from_mode(mode)).expect("chmod");
        p
    };

    // setuid: refused in compress with the exact line (exit 2).
    let su = write_mode("su.txt", 0o4755);
    let out = run(dir.path(), &["su.txt"]);
    assert_eq!(out.status.code(), Some(2), "setuid refusal");
    assert_eq!(
        stderr_of(&out),
        "gzippy: su.txt is set-user-ID on execution - ignored\n"
    );

    // ... and still refused with -f (unconditional refusal).
    let out = run(dir.path(), &["-f", "su.txt"]);
    assert_eq!(out.status.code(), Some(2), "no -f escape for setuid");
    assert_eq!(
        stderr_of(&out),
        "gzippy: su.txt is set-user-ID on execution - ignored\n"
    );
    // -c passes (whole ladder is !to_stdout).
    let out = run(dir.path(), &["-c", "su.txt"]);
    assert!(out.status.success(), "-c reads through setuid: {out:?}");
    let _ = su;

    // setgid: different wording, same unconditional class.
    let sg = write_mode("sg.txt", 0o2755);
    let out = run(dir.path(), &["sg.txt"]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        stderr_of(&out),
        "gzippy: sg.txt is set-group-ID on execution - ignored\n"
    );
    let _ = sg;

    // sticky: -f escapable (probed: `gzip c.bin` refuses, `-f` proceeds).
    let st = write_mode("st.txt", 0o1755);
    let out = run(dir.path(), &["st.txt"]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        stderr_of(&out),
        "gzippy: st.txt has the sticky bit set - file ignored\n"
    );
    let out = run(dir.path(), &["-f", "st.txt"]);
    assert!(out.status.success(), "-f overrides sticky");
    let _ = st;

    // Coming through decompress too (treat_file is direction-agnostic):
    // a setuid-named ARCHIVE is refused like the file.
    fs::write(
        dir.path().join("sa.bin.gz"),
        build_gzip_fixture(&payload_bytes),
    )
    .expect("archive");
    fs::hard_link(dir.path().join("sa.bin.gz"), dir.path().join("sa2.bin.gz"))
        .expect("needs nlink so unrelated paths cannot short-circuit");
    fs::set_permissions(
        dir.path().join("sa.bin.gz"),
        fs::Permissions::from_mode(0o4755),
    )
    .expect("chmod archive");
    let out = run(dir.path(), &["-d", "sa.bin.gz"]);
    assert_eq!(out.status.code(), Some(2), "-d refuses setuid archive");
    assert_eq!(
        stderr_of(&out),
        "gzippy: sa.bin.gz is set-user-ID on execution - ignored\n"
    );
}

/// gzip (probed): MIXED invocation of a missing file and a symlink in EITHER
/// argument order settles at exit 1: BOTH failures are progerror-grade
/// (missing → ENOENT; refused link → ELOOP; gzip.c:1153-1155 sets
/// exit_code = ERROR unconditionally), never the WARNING class the old
/// "skipped" symlink report produced. Both files report; the run completes.
#[test]
fn mixed_missing_and_symlink_args_settle_at_error_class() {
    let dir = tempdir();
    fs::write(dir.path().join("payload.txt"), payload()).expect("payload");
    symlink("payload.txt", dir.path().join("lnk.txt")).expect("link");
    fs::write(
        dir.path().join("payload.gz"),
        build_gzip_fixture(&payload()),
    )
    .expect("archive");
    symlink("payload.gz", dir.path().join("lnk.gz")).expect("archive link");

    // Compress arm first, decompress arm second; both orders.
    for args in [["lnk.txt", "ghost.txt"], ["ghost.txt", "lnk.txt"]] {
        let out = run(dir.path(), &args);
        assert_eq!(
            out.status.code(),
            Some(1),
            "{args:?}: ELOOP + ENOENT are both ERROR class (probed)"
        );
        let stderr = stderr_of(&out);
        assert!(
            stderr.contains("Too many levels of symbolic links"),
            "{args:?}: {stderr:?}"
        );
        assert!(
            stderr.contains("No such file or directory"),
            "{args:?}: {stderr:?}"
        );
    }
    for args in [["-d", "lnk.gz", "ghost.gz"], ["-d", "ghost.gz", "lnk.gz"]] {
        let out = run(dir.path(), &args);
        assert_eq!(out.status.code(), Some(1), "{args:?}: ERROR class");
        let stderr = stderr_of(&out);
        assert!(
            stderr.contains("Too many levels of symbolic links"),
            "{args:?}: {stderr:?}"
        );
        assert!(
            stderr.contains("No such file or directory"),
            "{args:?}: {stderr:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 10. -l container classes for truncated archives
// ---------------------------------------------------------------------------

/// gzip (probed): `gzip -l` of a 0/2/9/20-byte .gz = "\ngzip: NAME:
/// unexpected end of file", exit 1 (read_error, util.c:394-402 — the header
/// scan runs out of bytes), while garbage bytes keep get_method's
/// "\ngzip: NAME: not in gzip format" (gzip.c:1473-1475). gzippy used to
/// lump every <18-byte case into "not in gzip format" (and would slice
/// ISIZE out of bounds on sub-8-byte input).
#[test]
fn list_of_truncated_archive_reports_unexpected_eof() {
    let dir = tempdir();
    let good = build_gzip_fixture(&payload());
    fs::write(dir.path().join("tiny0.gz"), b"").expect("tiny0");
    fs::write(dir.path().join("tiny2.gz"), &good[..2]).expect("tiny2");
    fs::write(dir.path().join("tiny9.gz"), &good[..9]).expect("tiny9");

    for name in ["tiny0.gz", "tiny2.gz", "tiny9.gz"] {
        let out = run(dir.path(), &["-l", name]);
        assert_eq!(out.status.code(), Some(1), "{name}: ERROR class");
        let expected = format!("\ngzippy: {name}: unexpected end of file\n");
        assert_eq!(
            stderr_of(&out),
            expected,
            "{name}: read_error's class (probed)"
        );
    }

    // Bad magic keeps get_method's wording.
    fs::write(dir.path().join("junk9.gz"), b"junkjunkjunk").expect("junk9");
    let out = run(dir.path(), &["-l", "junk9.gz"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(stderr_of(&out), "\ngzippy: junk9.gz: not in gzip format\n");
}

// ---------------------------------------------------------------------------
// 11. l1-tune feature-off build: --tune is a normal unknown option
// ---------------------------------------------------------------------------

/// A build WITHOUT the dev-only l1-tune feature must reject `--tune=x`
/// exactly like any other unknown long option: getopt_long prints
/// "unrecognized option '--tune=x'" and try_help adds its line (gzip.c:315-
/// 319), exit 1. With the feature built in, the flag exists and this pin is
/// compiled out.
#[cfg(not(feature = "l1-tune"))]
#[test]
fn tune_without_the_dev_feature_is_an_unrecognized_option() {
    let dir = tempdir();

    let out = run(dir.path(), &["--tune=coarse", "-c"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr_of(&out),
        "gzippy: unrecognized option '--tune=coarse'\nTry `gzippy --help' for more information.\n"
    );

    // Bare `--tune` (no =value) takes the same catch-all path.
    let out = run(dir.path(), &["--tune"]);
    assert_eq!(
        stderr_of(&out),
        "gzippy: unrecognized option '--tune'\nTry `gzippy --help' for more information.\n"
    );
}
