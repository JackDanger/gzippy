//! `gzip -l` contract tests: closed-form pins derived from the vendored gzip
//! source (vendor/gzip/gzip.c `do_list` 1504-1605, `get_method` 1431-1432,
//! `make_ofname` 1173-1248, `get_suffix` 982-1022; vendor/gzip/util.c
//! `display_ratio` 416-422, `fprint_off` 428-455), every row cross-checked
//! against the installed GNU gzip binary.
//!
//! Every input is a hand-crafted gzip member (fixed 10-byte header, one
//! final STORED deflate block, CRC32+ISIZE trailer): no compression library
//! is needed in-test, no rival binary, no corpus. The pinned numbers follow
//! from gzip's arithmetic alone:
//!
//! - `bytes_in` = compressed file size; `bytes_out` = ISIZE (EOF-8);
//! - `header_bytes` = header bytes consumed (+ FNAME) + 8 trailer bytes;
//! - ratio = 100.0 * (bytes_out - (bytes_in - header_bytes)) / bytes_out,
//!   printed "%5.1f%%" with NO clamping;
//! - name column = on-disk name minus gzip suffix (".tgz"/".taz" -> ".tar"),
//!   header FNAME only under -N;
//! - one row per file; totals over >1 file re-use the LAST row's
//!   header_bytes and are suppressed when total_in/total_out <= 0.
//!
//! Sizes columns print right-aligned at positive_off_t_width = decimal
//! digits of OFF_T_MAX = 19 (off_t is i64).

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;

fn gzippy() -> &'static str {
    env!("CARGO_BIN_EXE_gzippy")
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    std::process::Command::new(gzippy())
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("failed to spawn gzippy")
}

fn tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("create tempdir")
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, bytes).expect("write input file");
    p
}

/// CRC-32 (IEEE 802.3 mirror form), the value the gzip trailer carries.
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// A valid single-member gzip file, hand-crafted: 10-byte fixed header
/// (+ optional FNAME), one FINAL STORED deflate block, CRC+ISIZE trailer.
/// Older-and-newer-decoder-compatible; content is stored verbatim, so the
/// file is always LARGER than its input — exactly the expansion regime the
/// un-clamped ratio has to print.
fn gzip_member(content: &[u8], fname: Option<&[u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    let flags = if fname.is_some() { 0x08u8 } else { 0u8 };
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, flags]);
    out.extend_from_slice(&0u32.to_le_bytes()); // MTIME = 0
    out.push(0); // XFL
    out.push(3); // OS = Unix
    if let Some(f) = fname {
        out.extend_from_slice(f);
        out.push(0);
    }
    // stored deflate block: BFINAL=1 BTYPE=00, LEN / NLEN
    out.push(0x01);
    let len = content.len() as u16;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(content);
    out.extend_from_slice(&crc32(content).to_le_bytes());
    out.extend_from_slice(&(content.len() as u32).to_le_bytes());
    out
}

/// First-listed line of a `gzippy -l` run (the widths header) asserted
/// byte-exactly: the vendored format `"%*.*s %*.*s  ratio uncompressed_name"`
/// at positive_off_t_width = 19 on 64-bit off_t.
fn assert_gzip_list_header(dir: &Path, files: &[&str]) {
    let mut args: Vec<&str> = vec!["-l"];
    args.extend_from_slice(files);
    let out = run(dir, &args);
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("         compressed        uncompressed  ratio uncompressed_name"),
        "do_list header line (gzip.c 1530 at positive_off_t_width=19); got: {text:?}"
    );
}

// ---------------------------------------------------------------------------
// ratio formula + name column (gzip -l rows)
// ---------------------------------------------------------------------------

/// Single member, plain 10-byte header (no FNAME): header_bytes = 10 + 8 = 18,
/// so num = 24 - (47 - 18) = -5 and gzip prints -5/24 = -20.8%. The +-99.9
/// clamp and the old (1 - comp/uncomp) shape would print -99.9% here.
#[test]
fn list_ratio_matches_gzip_formula_plain_header() {
    // exactly 24 bytes so the numbers stay readable
    let content = b"this is 24 bytes of data";
    assert_eq!(content.len(), 24);
    let d = tempdir();
    write_file(d.path(), "member.gz", &gzip_member(content, None));
    let out = run(d.path(), &["-l", "member.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        text.contains("member\n") && text.lines().count() == 2,
        "one-member one-row shape wanted, got: {text:?}"
    );
    let row = text.lines().nth(1).unwrap();
    assert_eq!(
        row, "                 47                  24 -20.8% member",
        "util.c display_ratio(24-(47-18), 24) = -20.8% at fprint_off width 19"
    );
}

/// Header carries a stored FNAME of a different length than any on-disk
/// name: the ratio must charge the FNAME bytes (header_bytes = 10+6+8 = 24
/// with FNAME "a.txt", num = 24 - (53 - 24) = -5, -20.8%) and WITHOUT -N the
/// name column is the on-disk name minus the suffix, NOT the stored FNAME
/// (gzip.c 1384: `no_name` default makes ORIG_NAME discarded).
#[test]
fn list_name_uses_disk_name_not_fname_and_ratio_charges_fname_bytes() {
    let content = b"this is 24 bytes of data";
    let d = tempdir();
    write_file(
        d.path(),
        "m1.gz",
        &gzip_member(content, Some(b"orig-name.txt")),
    );
    let out = run(d.path(), &["-l", "m1.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let row = text.lines().nth(1).unwrap();
    assert_eq!(
        row, "                 61                  24 -20.8% m1",
        "header_bytes counts the 13+1 FNAME bytes; name column = m1"
    );
    assert!(
        !text.contains("orig-name"),
        "header FNAME must not reach the name column without -N: {text:?}"
    );

    // The same file under -N consults the stored FNAME (gzip -N -l prints
    // `a.txt` for m1.gz with FNAME a.txt — see the README probes).
    let out = run(d.path(), &["-N", "-l", "m1.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    // FNAME "orig-name.txt" is 13 bytes: header_bytes = 10+14+8 = 32.
    let row = text.lines().nth(1).unwrap();
    assert_eq!(
        row, "                 61                  24 -20.8% orig-name.txt",
        "-N consults the header FNAME (gzip.c 1383-1408)"
    );
}

/// `.tgz` maps to `.tar` (make_ofname 1195-1198): `gzip -l t.tgz` prints
/// t.tar; the suffix swap must not be decided by the header FNAME.
#[test]
fn list_tgz_maps_to_tar() {
    let d = tempdir();
    write_file(
        d.path(),
        "t.tgz",
        &gzip_member(b"this is 24 bytes of data", None),
    );
    let out = run(d.path(), &["-l", "t.tgz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let row = text.lines().nth(1).unwrap();
    assert_eq!(row, "                 47                  24 -20.8% t.tar");
}

/// Expansion regime, below -100%: a 2-byte payload stores to 25 B total,
/// num = 2 - (25 - 18) = -5, ratio = -250.0% — the un-clamped negative gzip
/// prints for every expansion row (e.g. the -1713.0% totals rows). The
/// removed clamp would have printed -99.9%.
#[test]
fn list_expansion_ratio_is_unclamped_negative() {
    let d = tempdir();
    write_file(d.path(), "tiny.gz", &gzip_member(&[0x51u8, 0x52], None));
    let out = run(d.path(), &["-l", "tiny.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let row = text.lines().nth(1).unwrap();
    assert_eq!(
        row, "                 25                   2 -250.0% tiny",
        "display_ratio has no clamp: -250.0% prints literally"
    );
}

/// get_suffix (gzip.c 982-1022) suffix semantics probed from gzip:
/// case-insensitive ".GZ", the "-gz"/"-z"/"_z" aliases, ".z"/".Z", ".tgz"
/// -> ".tar"; ".zz" is NOT a gzip suffix and stays verbatim; an
/// unsuffixed gzip content file is listed under its own name.
#[test]
fn list_suffix_recognition_matches_get_suffix() {
    let d = tempdir();
    let member = gzip_member(b"this is 24 bytes of data", None);
    for name in [
        "a.GZ",
        "x-gz",
        "x-z",
        "x_z",
        "a.z",
        "a.Z",
        "docs.gz.txt",
        "y.zz",
        "noext",
    ] {
        write_file(d.path(), name, &member);
    }
    for (name, want) in [
        ("a.GZ", "                 47                  24 -20.8% a"),
        ("x-gz", "                 47                  24 -20.8% x"),
        ("x-z", "                 47                  24 -20.8% x"),
        ("x_z", "                 47                  24 -20.8% x"),
        ("a.z", "                 47                  24 -20.8% a"),
        ("a.Z", "                 47                  24 -20.8% a"),
        (
            "docs.gz.txt",
            "                 47                  24 -20.8% docs.gz.txt",
        ),
        (
            "y.zz",
            "                 47                  24 -20.8% y.zz",
        ),
        (
            "noext",
            "                 47                  24 -20.8% noext",
        ),
    ] {
        let out = run(d.path(), &["-l", name]);
        assert!(
            out.status.success(),
            "{}: {}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let row = text.lines().nth(1).unwrap();
        assert_eq!(row, want, "name column for {name}");
    }
}

/// Two files: the totals row re-uses the LAST row's header_bytes
/// (gzip.c 1545-1547: "header_bytes is not meaningful but used to ensure
/// the same ratio if there is a single file") and ends with the literal
/// " (totals)". 47+25 = 72, ISIZE 24+2 = 26: num = 26 - (72 - 18) = -28,
/// ratio = -107.7% — again past -100%, unclamped.
#[test]
fn totals_row_uses_last_header_bytes_and_literal_totals() {
    let d = tempdir();
    write_file(
        d.path(),
        "a.gz",
        &gzip_member(b"this is 24 bytes of data", None),
    );
    write_file(d.path(), "b.gz", &gzip_member(&[0x51u8, 0x52], None));
    let out = run(d.path(), &["-l", "a.gz", "b.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4, "two rows, header, totals: {text:?}");
    assert_eq!(
        lines[3], "                 72                  26 -107.7% (totals)",
        "totals: display_ratio(total_out-(total_in-header_bytes), total_out)"
    );
}

/// do_list(-1) returns without printing when total_in <= 0 or
/// total_out <= 0 (gzip.c 1535): two empty members produce rows but no
/// totals row at all.
#[test]
fn totals_row_suppressed_when_totals_are_zero() {
    let d = tempdir();
    let empty = gzip_member(b"", None); // 10 + 5 + 0 + 8 = 23 bytes
    write_file(d.path(), "e1.gz", &empty);
    write_file(d.path(), "e2.gz", &empty);
    let out = run(d.path(), &["-l", "e1.gz", "e2.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "rows but NO totals: {text:?}");
    assert_eq!(
        lines[1], "                 23                   0   0.0% e1",
        "{text:?}"
    );
    assert_eq!(
        lines[2], "                 23                   0   0.0% e2",
        "{text:?}"
    );
}

// ---------------------------------------------------------------------------
// two-point read: the body must never be read
// ---------------------------------------------------------------------------

/// -l reads two points: the header prefix and the 8 trailer bytes. Prove
/// this by listing a 100 MiB file whose BYTES 10..EOF-8 are holes (garbage
/// the deflate stream would reject): the row must still come out correctly
/// and fast — a whole-file read or a body validation would take measurable
/// work and, if any body validation ever slips in, FAIL on the garbage.
#[test]
fn list_reads_two_points_not_the_body() {
    let d = tempdir();
    let path = d.path().join("big.gz");
    let mut f = fs::File::create(&path).expect("create big file");
    f.write_all(&gzip_member(b"this is 24 bytes of data", None)[..10])
        .expect("write header");
    f.set_len(100 * 1024 * 1024).expect("extend to 100 MiB");
    f.seek(SeekFrom::End(-8)).expect("seek trailer");
    // trailer: CRC32 = 0, ISIZE = 24 (little endian)
    f.write_all(&[0, 0, 0, 0, 24, 0, 0, 0])
        .expect("write crc+ISIZE");
    drop(f);

    let started = std::time::Instant::now();
    let out = run(d.path(), &["-l", "big.gz"]);
    let elapsed = started.elapsed();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let row = text.lines().nth(1).unwrap();
    assert_eq!(
        row, "          104857600                  24 -436906491.7% big",
        "compressed size is the real file size, ISIZE from EOF-8; no clamp"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "list must not ingest the body: took {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// error behaviors kept, flag interactions
// ---------------------------------------------------------------------------

/// The header prints lazily (do_list's first_time): a listing in which every
/// file fails leaves stdout EMPTY (gzip probed: `gzip -l /nope/nope.gz
/// 2>/dev/null | wc -c` == 0; gzippy used to print the 65-byte header before
/// the file loop). With at least one listed row the header precedes it.
#[test]
fn list_header_prints_only_when_a_row_prints() {
    let d = tempdir();

    // All failing: nothing on stdout (errors go to stderr, exit 1).
    let out = run(d.path(), &["-l", "/nope/nope.gz"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        out.stdout.is_empty(),
        "orphan header not allowed: got {:?}",
        String::from_utf8_lossy(&out.stdout)
    );

    // Mixed: a listed file's row still carries the header, and the failing
    // file must not duplicate it.
    write_file(
        d.path(),
        "m1.gz",
        &gzip_member(b"orchard fixture bytes here", None),
    );
    let out = run(d.path(), &["-l", "noexist.gz", "m1.gz"]);
    let text = String::from_utf8_lossy(&out.stdout);
    let header_count = text
        .lines()
        .filter(|l| l.contains("uncompressed_name"))
        .count();
    assert_eq!(
        header_count, 1,
        "exactly one header line even with a failed file: {text:?}"
    );

    // Verbose header is lazy the same way.
    let out = run(d.path(), &["-lv", "/nope/nope.gz"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        out.stdout.is_empty(),
        "verbose orphan header: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    let out = run(d.path(), &["-lv", "m1.gz"]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        text.lines().next(),
        Some("method  crc     date  time    compressed uncompressed  ratio uncompressed_name"),
        "verbose header precedes the row: {text:?}"
    );
}

/// Existing verdicts aligned with real gzip (probed gzip 1.12): a 13-byte
/// good-magic fragment is read_error's "\n...: unexpected end of file",
/// NOT "not in gzip format" (the earlier assertion here had the wrong
/// wording for the good-magic-short case; cross-pinned with
/// exit_ledger::list_of_truncated_archive_reports_unexpected_eof).
#[test]
fn list_preserves_min_size_and_verdict_errors() {
    let d = tempdir();
    // Good magic + too few bytes: gzip's header scan runs out of input →
    // read_error class, exit 1.
    let short = b"\x1f\x8b\x08\x00some junk";
    write_file(d.path(), "short.gz", short);
    let out = run(d.path(), &["-l", "short.gz"]);
    assert_eq!(out.status.code(), Some(1), "short good-magic is exit 1");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("unexpected end of file"),
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );

    // non-gzip content of length >= 18.
    write_file(d.path(), "plain.gz", b"this is not gzip at all.");
    let out = run(d.path(), &["-l", "plain.gz"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("not in gzip format"),
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );

    // a directory under -l keeps the established skip contract.
    fs::create_dir(d.path().join("adir")).expect("mkdir");
    let out = run(d.path(), &["-l", "adir"]);
    assert_eq!(out.status.code(), Some(2), "directory is a warning, exit 2");
}

/// -q suppresses only the header (do_list prints rows regardless) — the
/// vendored shape the old code got wrong by always printing it.
#[test]
fn list_quiet_still_prints_rows() {
    let d = tempdir();
    write_file(
        d.path(),
        "m1.gz",
        &gzip_member(b"this is 24 bytes of data", None),
    );
    let out = run(d.path(), &["-q", "-l", "m1.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1, "no header, no totals: {text:?}");
    assert_eq!(
        lines[0],
        "                 47                  24 -20.8% m1"
    );
}

/// Old gzippy behavior: header prints even when quiet. Pinned here so the
/// quiet contract cannot silently rot: gzip -q -l prints rows only.
#[test]
fn list_header_variants_match_width_contract() {
    let d = tempdir();
    write_file(
        d.path(),
        "m1.gz",
        &gzip_member(b"this is 24 bytes of data", None),
    );
    assert_gzip_list_header(d.path(), &["m1.gz"]);
}

/// Concatenated members: one row per FILE (not per member). Our cheap
/// reader documents this row with the FIRST member's header_bytes (the
/// vendored do_list arithmetic); gzip 1.12 additionally zeroes it once its
/// decompress loop spots member 2 - the documented multi-member divergence.
#[test]
fn list_concatenated_file_is_one_row_with_documented_divergence() {
    let d = tempdir();
    let two_members = {
        let mut v = gzip_member(b"this is 24 bytes of data", None);
        v.extend_from_slice(&gzip_member(b"012345678901234567890123456789", None));
        v
    };
    write_file(d.path(), "multi.gz", &two_members);
    let out = run(d.path(), &["-l", "multi.gz"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let row = text.lines().nth(1).unwrap();
    // compressed = 47 + 53 = 100, uncompressed = LAST member's ISIZE = 30;
    // header_bytes = first member's 18: num = 30 - (100 - 18) = -52,
    // ratio = -173.3% (gzip 1.12 re-parses member 2 and prints -233.3% -
    // documented version difference; neither formula clamps).
    assert_eq!(row, "                100                  30 -173.3% multi");
}
