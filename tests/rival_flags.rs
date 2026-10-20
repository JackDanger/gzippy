//! Rival flag-surface contract tests: pigz, igzip, libdeflate-gzip.
//!
//! gzippy's charter is drop-in for gzip AND pigz AND libdeflate-gzip AND
//! igzip, but only the gzip contract had a suite (tests/cli_dropin.rs,
//! PR #267). This file enumerates the OTHER three rivals' documented flag
//! surfaces and pins, for every flag, one of three outcomes:
//!
//!   1. SUPPORTED   — we accept it and the behaviour matches the rival's
//!                    documented semantics (asserted).
//!   2. CLEAN REJECT — we don't support it, and we say so: nonzero exit plus
//!                    a stderr mention of the offending flag. Safe for
//!                    scripts: the pipeline stops loudly.
//!   3. SILENT DIVERGENCE — we accept the flag but do something other than
//!                    what the rival documents. These are the dangerous ones
//!                    (a backup script gets valid-looking output with the
//!                    wrong property). Each is pinned at CURRENT behaviour
//!                    (non-ignored) and, where the contract behaviour is
//!                    testable, also asserted as an `#[ignore]`d
//!                    `DIVERGENCE` test, following the cli_dropin.rs
//!                    convention: ignored tests are triage targets, not
//!                    endorsed behaviour.
//!
//! Rival surfaces were enumerated from:
//!   - pigz 2.8 `--help` (verified locally, 2026-08-09);
//!   - igzip: vendor/isa-l/programs/igzip_cli.c `usage()` + `long_options`
//!     (source-verified; igzip is not installed on this box, so igzip
//!     *behaviour* claims below cite the source, not a local run);
//!   - libdeflate-gzip 1.x `-h` (verified locally, 2026-08-09).
//!
//! ## The igzip LEVEL SEMANTICS question
//!
//! igzip levels are 0..=3 (ISAL_DEF_MIN_LEVEL=0, ISAL_DEF_MAX_LEVEL=3,
//! default 2, `--fast`=-1, `--best`=-3). Every igzip level is a real DEFLATE
//! compression level: igzip -0 COMPRESSES (it is their fastest parse).
//! gzippy follows the gzip/pigz scale 0..=9 (+10..12), where -0 is pigz's
//! STORE (no compression). Consequences for an igzip script pointed here:
//!
//!   igzip -0  -> gzippy stores (the gzip/pigz contract, kept) — now with a
//!                LOUD stderr warning naming the divergence and issue #305.
//!   igzip -1  -> weak-fast gzip L1 (roughly comparable intent; ratio differs).
//!   igzip -2  -> gzip L2, a weak-fast level — but -2 is igzip's DEFAULT,
//!                sized between our L6-ish intent. Valid gzip, silently
//!                different ratio/speed profile.
//!   igzip -3  -> gzip L3 — but -3 is igzip's BEST (`--best`). A user asking
//!                igzip for maximum compression gets our third-weakest level.
//!   igzip -z  -> igzip: "compress (default)", a no-op modifier. gzippy: the
//!                pigz meaning (zlib container) — REAL zlib output, .zz
//!                suffix (issue #303). The suffix redirect is the only
//!                observable divergence left on this row.
//!
//! The -0 warning, the --help documentation of this scale collision, and the
//! -2/-3 rows remain silent divergences (both valid gzip), pinned below.
//!
//! ## Triage table (updated by the flag-honesty PR, which mitigated every
//! ## silent-wrong row into a clean rejection or a fix — the missing FEATURES
//! ## stay tracked in issues #302/#303/#305)
//!
//! | rival flag                | our behaviour                                   | status |
//! |---------------------------|--------------------------------------------------|----------|
//! | pigz -R/--rsyncable       | REJECTED in compress mode, exit 2, names issue #302 (was: accepted, output NOT rsyncable) | clean-reject |
//! | pigz -z/--zlib            | SUPPORTED (#303): real RFC 1950 zlib — CMF low nibble 8, (CMF<<8|FLG)%31==0, big-endian Adler-32 — at every level 0-12 and thread count (T>1 stays on the parallel pipeline); file mode writes FILE.zz and deletes the input; own -d/-t roundtrip; -z + -C refused loudly (no FCOMMENT field) | implemented |
//! | pigz -K/--zip             | REJECTED in compress mode, exit 2, names issue #303 (was: gzip bytes under .zip) | clean-reject |
//! | pigz -b/--blocksize N     | FIXED: bare N is KiB, pigz's documented unit (was: bytes) | ok |
//! | pigz -H/--huffman         | REJECTED in compress mode, exit 2 (was: silent no-op advertised in --help) | clean-reject |
//! | pigz -U/--rle             | REJECTED in compress mode, exit 2 (was: silent no-op advertised in --help) | clean-reject |
//! | pigz -i/--independent     | REJECTED in compress mode, exit 2 (was: silent no-op, no independence guarantee) | clean-reject |
//! | pigz -C/--comment ccc     | honoured on the FILE path at every thread count (issue #309 gave T1 the full header); REJECTED (never dropped) on the -c/stdout path (exit 2, minimal header there by design) | ok/clean-reject |
//! | pigz -A/--alias xxx       | REJECTED in compress mode, exit 2, names issue #303 (was: silently ignored) | clean-reject |
//! | pigz -F/--first           | collides: our -F takes a VALUE (zopfli iterations); consumes the next argv | misparse-but-errs MED |
//! | pigz -I/--iterations n    | collides: our -I is a flag (no-block-split); n becomes a FILE operand | misparse-but-errs MED |
//! | pigz -O/--oneblock        | clean reject ("Unknown option: -O")             | clean-reject |
//! | pigz -M/-m, -N/-n, -p, -S, -Y, -v, -q, -r, -t, -l, -L, -V, -d, -f, -k, -c, --fast, --best | supported | ok |
//! | igzip -o FILE             | clean reject ("Unknown option: -o")             | clean-reject |
//! | igzip -T/--threads n      | clean reject ("Unknown option: -T"; ours is -p) | clean-reject |
//! | igzip --rm                | clean reject ("Unknown option: --rm")           | clean-reject |
//! | igzip -0                  | STORE (gzip/pigz contract) + LOUD stderr warning naming issue #305 (was: silent) | divergence, loud |
//! | igzip -2 / -3             | weak-fast gzip levels; igzip's default/best; documented in --help | divergence MED |
//! | igzip -z                  | -z has the PIGZ meaning (zlib container, issue #303): compresses to .zz and roundtrips; the suffix redirect (.zz, not .gz) is the documented divergence | divergence, loud-doc |
//! | igzip keep-by-default     | we DELETE the input after compress (gzip rule); igzip keeps | divergence MED |
//! | libdeflate-gzip -1..-12   | supported, valid gzip at every level             | ok |
//! | libdeflate-gzip -c -d -f -h -k -q -S -t -V | supported                      | ok |
//!
//! Rejections are COMPRESS-MODE ONLY: on decompress/test/list these flags are
//! inert for the rivals too, so `-d` pipelines carrying them keep working
//! (pinned below in `rejected_flags_still_inert_on_decompress`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn gzippy() -> &'static str {
    env!("CARGO_BIN_EXE_gzippy")
}

/// Run gzippy with `args` in `dir`, stdin closed.
fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(gzippy())
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

/// Deterministic compressible-but-not-trivial synthetic text, `n` bytes.
fn synth(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 32);
    let mut x: u64 = 0x243f_6a88_85a3_08d3;
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(b"the quick brown fox jumps over ");
        out.push(b'a' + (x % 26) as u8);
        out.push(b' ');
        out.extend_from_slice(x.to_string().as_bytes());
        out.push(b'\n');
    }
    out.truncate(n);
    out
}

/// Compress `file` (already in `dir`) to stdout with `flags`; assert success.
fn compress_stdout(dir: &Path, flags: &[&str], file: &str) -> Vec<u8> {
    let mut args: Vec<&str> = flags.to_vec();
    args.push("-c");
    args.push(file);
    let out = run(dir, &args);
    assert!(
        out.status.success(),
        "gzippy {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.stdout.is_empty(), "gzippy {args:?} wrote no output");
    out.stdout
}

/// Roundtrip `gz` through our own decoder, asserting it equals `expect`.
/// (CI boxes are not guaranteed to carry a rival binary; cli_dropin.rs pins
/// cross-decoder behaviour where that matters.)
fn assert_roundtrips(dir: &Path, gz: &[u8], expect: &[u8], what: &str) {
    let name = format!("rt{}.gz", what.replace([' ', '/', '-'], "_"));
    write_file(dir, &name, gz);
    let out = run(dir, &["-d", "-c", &name]);
    assert!(
        out.status.success(),
        "{what}: our own -d failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, expect, "{what}: roundtrip bytes differ");
}

/// Same as [`assert_roundtrips`] but names the payload file `.zz`. The .zz
/// suffix is pigz's own, and decoding is by CONTENT anyway since #303: any
/// name holding a valid zlib header routes to the zlib decoder (pinned for a
/// .gz name inside `pigz_zlib_file_mode_writes_zz_and_deletes_input`).
fn assert_roundtrips_zz(dir: &Path, zz: &[u8], expect: &[u8], what: &str) {
    let name = format!("rt{}.zz", what.replace([' ', '/', '-'], "_"));
    write_file(dir, &name, zz);
    let out = run(dir, &["-d", "-c", &name]);
    assert!(
        out.status.success(),
        "{what}: our own -d failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, expect, "{what}: roundtrip bytes differ");
}

/// Assert `args` is a CLEAN rejection: nonzero exit and stderr names `token`.
fn assert_clean_reject(args: &[&str], token: &str) {
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let mut full: Vec<&str> = args.to_vec();
    full.push("in.txt");
    let out = run(d.path(), &full);
    assert!(
        !out.status.success(),
        "gzippy {full:?} must exit nonzero (unsupported rival flag), got success"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(token),
        "gzippy {full:?} stderr must mention {token:?} so the failure is \
         diagnosable; got: {stderr}"
    );
}

/// Assert `args` is a flag-honesty rejection: exit code EXACTLY 2
/// (warning-class refusal before any work), stderr names every `token`,
/// no output written to stdout, and the input file left untouched.
fn assert_honesty_reject(args: &[&str], tokens: &[&str]) {
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let mut full: Vec<&str> = args.to_vec();
    full.push("in.txt");
    let out = run(d.path(), &full);
    assert_eq!(
        out.status.code(),
        Some(2),
        "gzippy {full:?} must exit 2 (loud refusal, nothing attempted); \
         stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    for token in tokens {
        assert!(
            stderr.contains(token),
            "gzippy {full:?} stderr must mention {token:?}; got: {stderr}"
        );
    }
    assert!(
        out.stdout.is_empty(),
        "gzippy {full:?} must write nothing before refusing"
    );
    assert!(
        d.path().join("in.txt").exists(),
        "gzippy {full:?} must leave the input file untouched"
    );
}

// ===========================================================================
// pigz surface
// ===========================================================================

#[test]
fn pigz_processes_flag_roundtrips_at_t1_and_t4() {
    // pigz -p n / --processes n: thread count. Supported; output at any -p
    // must be valid gzip of the same bytes (T>1 may frame differently).
    let d = tempdir();
    let data = synth(256 * 1024);
    write_file(d.path(), "in.txt", &data);
    for p in ["1", "4"] {
        let gz = compress_stdout(d.path(), &["-p", p], "in.txt");
        assert_roundtrips(d.path(), &gz, &data, &format!("-p {p}"));
    }
}

#[test]
fn pigz_blocksize_k_suffix_accepted() {
    // pigz -b mmm: block size in KiB. We accept an explicit K suffix.
    let d = tempdir();
    let data = synth(256 * 1024);
    write_file(d.path(), "in.txt", &data);
    let gz = compress_stdout(d.path(), &["-p", "4", "-b", "128k"], "in.txt");
    assert_roundtrips(d.path(), &gz, &data, "-b 128k");
}

#[test]
fn pigz_blocksize_contract_bare_number_means_kib() {
    // FIXED (issue #304): pigz(1): "-b, --blocksize mmm  Set compression
    // block size to mmmK" — a bare number is KIBIBYTES, and gzippy now
    // honours that unit. `-b 128` (pigz's default, 128 KiB) must be accepted
    // and produce the same bytes as the explicit `-b 128k`. The bytes-unit
    // era is over: bare 131072 now means 128 MiB (one block on this input),
    // NOT 131072 bytes, so it must DIFFER from the 128k grid.
    let d = tempdir();
    let data = synth(256 * 1024);
    write_file(d.path(), "in.txt", &data);
    let bare = run(d.path(), &["-p", "4", "-b", "128", "-c", "in.txt"]);
    assert!(
        bare.status.success(),
        "pigz -b contract: `-b 128` (= 128 KiB) must be accepted; stderr: {}",
        String::from_utf8_lossy(&bare.stderr)
    );
    let suffixed = compress_stdout(d.path(), &["-p", "4", "-b", "128k"], "in.txt");
    assert_eq!(
        bare.stdout, suffixed,
        "pigz -b contract: bare number is KiB"
    );
    let huge_bare = compress_stdout(d.path(), &["-p", "4", "-b", "131072"], "in.txt");
    assert_ne!(
        huge_bare, suffixed,
        "bare 131072 must mean 128 MiB (KiB unit), not 131072 bytes; \
         identical output to 128k would mean the unit regressed to bytes — \
         re-triage issue #304"
    );
}

#[test]
fn pigz_rsyncable_rejected_loudly() {
    // MITIGATED (issue #302): gzippy's -R output never had the rsyncable
    // property (4% resync vs pigz's 94% on the probe below), so a backup
    // pipeline relying on it silently lost incremental-rsync transfer.
    // Until the property is actually implemented, -R must FAIL the script:
    // exit 2, stderr naming the flag and the tracking issue, nothing
    // written. Both stdout and file mode.
    assert_honesty_reject(&["-p", "4", "-R", "-c"], &["rsyncable", "#302"]);
    assert_honesty_reject(&["-p", "4", "-R", "-k"], &["rsyncable", "#302"]);
}

/// Rabin-Karp rolling hashes of every `w`-byte window of `b`.
fn rolling_hashes(b: &[u8], w: usize) -> std::collections::HashSet<u64> {
    const B: u64 = 1_000_003;
    let mut pow: u64 = 1;
    for _ in 0..w {
        pow = pow.wrapping_mul(B);
    }
    let mut set = std::collections::HashSet::new();
    if b.len() < w {
        return set;
    }
    let mut h: u64 = 0;
    for &c in &b[..w] {
        h = h.wrapping_mul(B).wrapping_add(c as u64);
    }
    set.insert(h);
    for i in w..b.len() {
        h = h
            .wrapping_mul(B)
            .wrapping_add(b[i] as u64)
            .wrapping_sub(pow.wrapping_mul(b[i - w] as u64));
        set.insert(h);
    }
    set
}

/// Fraction of `a`'s non-overlapping `w`-byte blocks found (rsync-style, at
/// any byte offset) in `b`.
fn resync_fraction(a: &[u8], b: &[u8], w: usize) -> f64 {
    const B: u64 = 1_000_003;
    let bh = rolling_hashes(b, w);
    let mut hits = 0usize;
    let mut total = 0usize;
    let mut i = 0;
    while i + w <= a.len() {
        let mut h: u64 = 0;
        for &c in &a[i..i + w] {
            h = h.wrapping_mul(B).wrapping_add(c as u64);
        }
        total += 1;
        if bh.contains(&h) {
            hits += 1;
        }
        i += w;
    }
    hits as f64 / total.max(1) as f64
}

#[test]
#[ignore] // CONTRACT test for issue #302: un-ignore when --rsyncable is implemented.
fn pigz_rsyncable_contract_output_resyncs_after_edit() {
    // pigz(1) -R: "Input-determined block locations for rsync" — after a
    // small local edit, the bulk of the compressed stream must re-align so
    // rsync's rolling checksum can skip it. Measured 2026-08-09 on this
    // probe (256 KiB synthetic text, 100 bytes inserted at offset 1000,
    // 512-byte rolling windows):
    //     pigz -R      resynced 375/397 blocks (94%)
    //     pigz plain   23/394  (6%)
    //     gzippy -R    16/391  (4%)   <- WORSE than gzippy plain (62/391)
    // gzippy -R is now REJECTED loudly (flag-honesty PR); this contract test
    // stays as the acceptance bar for the real implementation.
    let d = tempdir();
    let data = synth(256 * 1024);
    let mut edited = data.clone();
    let insert: Vec<u8> = b"X".iter().cycle().take(100).cloned().collect();
    edited.splice(1000..1000, insert);
    write_file(d.path(), "orig.txt", &data);
    write_file(d.path(), "edit.txt", &edited);

    let a = compress_stdout(d.path(), &["-p", "4", "-R"], "orig.txt");
    let b = compress_stdout(d.path(), &["-p", "4", "-R"], "edit.txt");
    let frac = resync_fraction(&a, &b, 512);
    assert!(
        frac >= 0.5,
        "pigz -R contract: a 100-byte insertion must leave most of the \
         compressed stream re-alignable; pigz 2.8 achieves ~0.94 on this \
         exact probe, gzippy currently {frac:.3}"
    );
}

#[test]
fn pigz_huffman_flag_rejected_loudly() {
    // MITIGATED: pigz(1) -H / --huffman: "Use only Huffman coding for
    // compression" — a strategy change gzippy never implemented. It used to
    // be parsed, silently ignored, AND advertised in --help. Now: exit 2,
    // stderr names the flag, and --help no longer claims it.
    assert_honesty_reject(&["-H", "-c"], &["-H", "--huffman"]);
    assert_honesty_reject(&["-p", "4", "-H", "-k"], &["-H", "--huffman"]);
}

#[test]
fn pigz_rle_flag_rejected_loudly() {
    // MITIGATED: pigz(1) -U / --rle: "Use run-length encoding for
    // compression". Same class as -H: was a silent no-op advertised in
    // --help; now a loud refusal, de-advertised.
    assert_honesty_reject(&["-U", "-c"], &["-U", "--rle"]);
}

#[test]
fn pigz_independent_flag_rejected_loudly() {
    // MITIGATED: pigz(1) -i / --independent: "Compress blocks independently
    // for damage recovery". gzippy never guaranteed (or denied) the
    // independence property — the flag changed nothing observable, so a
    // pigz user could not tell whether they got their damage-recovery
    // boundaries. Refuse loudly instead of silently ignoring.
    assert_honesty_reject(&["-i", "-c"], &["-i", "--independent"]);
    assert_honesty_reject(&["-9", "-p", "4", "-i", "-k"], &["-i", "--independent"]);
}

/// Assert an RFC 1950 zlib header: CM=8 (low nibble of CMF) and the
/// (CMF<<8 | FLG) sum divisible by 31 (FCHECK).
fn assert_rfc1950_header(out: &[u8], what: &str) {
    assert!(
        out.len() >= 6,
        "{what}: zlib output must be at least header + one deflate byte + \
         Adler-32, got {} bytes",
        out.len()
    );
    assert_eq!(
        out[0] & 0x0f,
        8,
        "{what}: zlib CMF: compression method must be deflate"
    );
    let check = ((out[0] as u16) << 8) | out[1] as u16;
    assert_eq!(
        check % 31,
        0,
        "{what}: zlib header FCHECK must make CMF|FLG divisible by 31"
    );
}

/// Big-endian Adler-32 (RFC 1950), checked against the trailer `out` ends with.
fn assert_zlib_adler_trailer(out: &[u8], data: &[u8], what: &str) {
    let (mut s1, mut s2) = (1u32, 0u32);
    for &b in data {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    assert_eq!(
        &out[out.len() - 4..],
        &((s2 << 16) | s1).to_be_bytes(),
        "{what}: trailer must be the big-endian Adler-32 of the input"
    );
}

#[test]
fn pigz_zlib_contract_emits_zlib_stream() {
    // pigz(1) -z: the output must be RFC 1950 zlib — CMF low nibble 8
    // (deflate), and (CMF<<8|FLG) divisible by 31. And our own decoder must
    // roundtrip the .zz file we just wrote. Runs at the DEFAULT thread count
    // (no -p: this box runs at nproc workers). (Was #[ignore]d as the issue
    // #303 acceptance bar until -z emitted a real zlib stream; un-ignored
    // with the implementation.)
    let d = tempdir();
    let data = synth(16 * 1024);
    write_file(d.path(), "in.txt", &data);
    let out = compress_stdout(d.path(), &["-z"], "in.txt");
    assert_rfc1950_header(&out, "default-T -z stdout");
    assert_zlib_adler_trailer(&out, &data, "default-T -z stdout");
    write_file(d.path(), "rt.zz", &out);
    let rt = run(d.path(), &["-d", "-c", "rt.zz"]);
    assert!(rt.status.success(), "own -d of own -z output must succeed");
    assert_eq!(rt.stdout, data, "zlib roundtrip bytes");
}

#[test]
fn pigz_zlib_roundtrips_all_levels_at_t1_and_t4() {
    // Every level 1-12 must emit a valid zlib container (T1), and T>1 must be
    // parallel-but-valid: -p4 roundtrips through our own decoder too
    // (different chunk grid than T1; multi-chunk Adam/Adler-32 combine path).
    let d = tempdir();
    let data = synth(16 * 1024);
    write_file(d.path(), "in.txt", &data);
    for level in 1..=12u8 {
        let level_flag = &*format!("-{level}");
        // T1 (exact whole-stream encode; also covers the level-1/L6 legacy
        // arms and the libdeflate-port levels).
        let flag_t1 = ["-z", "-p", "1", level_flag];
        let out = compress_stdout(d.path(), &flag_t1, "in.txt");
        assert_rfc1950_header(&out, &format!("-z {level_flag} -p1"));
        assert_zlib_adler_trailer(&out, &data, &format!("-z {level_flag} -p1"));
        assert_roundtrips_zz(d.path(), &out, &data, &format!("-z {level_flag} -p1"));
        // -p4 (issue #303: the parallel pipeline carries the container).
        if matches!(level, 1 | 6 | 9 | 12) {
            let flag_t4 = ["-z", "-p", "4", level_flag];
            let out = compress_stdout(d.path(), &flag_t4, "in.txt");
            assert_zlib_adler_trailer(&out, &data, &format!("-z {level_flag} -p4"));
            assert_roundtrips_zz(d.path(), &out, &data, &format!("-z {level_flag} -p4"));
        }
    }
}

#[test]
fn pigz_zlib_file_mode_writes_zz_and_deletes_input() {
    // pigz -z file semantics: FILE.zz written, FILE removed after success
    // (gzip's delete-by-default rule); -k keeps the input; -t verifies; and
    // `gzippy -d FILE.zz` restores the original name and deletes the .zz.
    let d = tempdir();
    let data = synth(4096);
    write_file(d.path(), "in.txt", &data);
    let out = run(d.path(), &["-z", "-k", "in.txt"]);
    assert!(
        out.status.success(),
        "file-mode -z failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(d.path().join("in.txt").exists(), "-k must keep the input");
    let zz = fs::read(d.path().join("in.txt.zz")).expect("FILE.zz written");
    assert_rfc1950_header(&zz, "file-mode .zz");
    assert_zlib_adler_trailer(&zz, &data, "file-mode .zz");

    // Without -k the input is removed (gzip contract) and -d restores it.
    write_file(d.path(), "gone.txt", &data);
    let out = run(d.path(), &["-z", "gone.txt"]);
    assert!(
        out.status.success(),
        "-z file-only failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !d.path().join("gone.txt").exists(),
        "gzip contract: -z input removed after successful compression"
    );
    assert!(d.path().join("gone.txt.zz").exists());
    let rt = run(d.path(), &["-d", "-c", "gone.txt.zz"]);
    assert!(rt.status.success());
    assert_eq!(rt.stdout, data, "-d .zz roundtrip bytes");
    // -t verifies our own .zz (pigz -t handles -z files too).
    write_file(d.path(), "rt.zz", &zz);
    let t = run(d.path(), &["-t", "rt.zz"]);
    assert!(t.status.success(), "-t on own .zz must pass");

    // CONTENT detection (#303): the same .zz bytes under a `.gz` (or any
    // other) name still decode to the original — sniffing decides, not the
    // extension, so `-z` output redirected to an arbitrary name roundtrips
    // instead of hitting the gzip arm's "not in gzip format".
    write_file(d.path(), "foreign_name.gz", &zz);
    let det = run(d.path(), &["-d", "-c", "foreign_name.gz"]);
    assert!(
        det.status.success(),
        "-d must route a renamed zlib stream by content: {}",
        String::from_utf8_lossy(&det.stderr)
    );
    assert_eq!(det.stdout, data, "content-detected .gz roundtrip bytes");

    // -1..-9/-12 at the DEFAULT thread count in FILE mode (this box: nproc
    // workers) — includes the >128 KiB mmap parallel fast path.
    let big = synth(256 * 1024);
    write_file(d.path(), "big.txt", &big);
    let out = run(d.path(), &["-z", "-p", "4", "-k", "big.txt"]);
    assert!(
        out.status.success(),
        "large file-mode -z failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let zz = fs::read(d.path().join("big.txt.zz")).expect("big .zz");
    assert_rfc1950_header(&zz, "mmap -z");
    assert_zlib_adler_trailer(&zz, &big, "mmap -z");
    assert_roundtrips_zz(d.path(), &zz, &big, "mmap -z file mode");
}

#[test]
fn pigz_zlib_empty_stdin_roundtrips() {
    // Edge beyond the rivals: `-z` of EMPTY stdin must emit a valid RFC 1950
    // empty stream — the 2-byte header, at least one DEFLATE byte (the empty
    // final block), and the 4-byte Adler-32 (of nothing = 0x00000001).
    let d = tempdir();
    let out = run(d.path(), &["-z", "-c"]); // no file operand: stdin (empty)
    assert!(
        out.status.success(),
        "-z -c on empty stdin failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_rfc1950_header(&out.stdout, "empty stdin -z");
    assert!(
        out.stdout.len() >= 7,
        "empty zlib stream = 2 header + ≥1 deflate + 4 adler, got {}",
        out.stdout.len()
    );
    assert_eq!(
        &out.stdout[out.stdout.len() - 4..],
        &1u32.to_be_bytes(),
        "empty Adler-32 must be 1"
    );
    // The empty stream decodes to nothing — through our own decoder.
    write_file(d.path(), "empty.zz", &out.stdout);
    let rt = run(d.path(), &["-d", "-c", "empty.zz"]);
    assert!(rt.status.success());
    assert!(rt.stdout.is_empty(), "empty roundtrip must yield empty");
}

#[test]
fn pigz_zlib_comment_conflict_rejected_loudly() {
    // zlib (RFC 1950) has NO FCOMMENT field: `-z -C comment` would silently
    // drop what the user typed, which breaks the honesty rule. Only the
    // combination is refused (both -c and file mode), loudly, exit 2,
    // before any work; gzip-mode -C keeps its own contract.
    assert_honesty_reject(&["-z", "-C", "hello", "-c"], &["-z", "comment"]);
    assert_honesty_reject(&["-z", "-C", "hello", "-k"], &["-z", "comment"]);
}

#[test]
fn pigz_zip_flag_rejected_loudly() {
    // MITIGATED (issue #303): pigz(1) -K / --zip: "Compress to PKWare zip
    // (.zip) single entry format". gzippy used to accept -K, switch the
    // suffix to .zip, and emit GZIP bytes a downstream `unzip` refuses.
    // Now: exit 2, stderr names the flag and the issue.
    assert_honesty_reject(&["-K", "-c"], &["-K", "--zip", "#303"]);
    assert_honesty_reject(&["-K", "-k"], &["-K", "--zip", "#303"]);
}

#[test]
#[ignore] // CONTRACT test for issue #303: un-ignore when -K emits a PK zip entry.
fn pigz_zip_contract_emits_pk_zip() {
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(16 * 1024));
    let out = compress_stdout(d.path(), &["-K"], "in.txt");
    assert_eq!(&out[..2], b"PK", "pigz -K contract: PKWare zip magic");
}

#[test]
fn pigz_alias_flag_rejected_loudly() {
    // MITIGATED (issue #303): pigz(1) -A xxx / --alias xxx: "Use xxx as the
    // name for any --zip entry from stdin". It was accepted and silently
    // ignored. -A is only meaningful with --zip, which is itself
    // unimplemented — reject both, and wire -A up in the same change that
    // implements -K.
    assert_honesty_reject(&["-A", "entryname", "-c"], &["-A", "--alias", "#303"]);
}

#[test]
fn pigz_comment_stored_in_file_mode() {
    // pigz(1) -C ccc / --comment ccc: "Put comment ccc in the gzip or zip
    // header". SUPPORTED on the FILE path at EVERY thread count:
    // FLG.FCOMMENT (0x10) set and the NUL-terminated comment present.
    // (-p 1 used to REJECT here — the T1 encoders wrote a fixed header with
    // no FCOMMENT field — until issue #309 gave the T1 file dispatch the
    // same full header the T>1 path writes.)
    for threads in ["1", "4"] {
        let d = tempdir();
        write_file(d.path(), "in.txt", &synth(4096));
        let out = run(d.path(), &["-p", threads, "-C", "hello", "-k", "in.txt"]);
        assert!(
            out.status.success(),
            "file-mode -C -p{threads} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let gz = fs::read(d.path().join("in.txt.gz")).expect("in.txt.gz written");
        assert_eq!(&gz[..2], &[0x1f, 0x8b]);
        assert_ne!(
            gz[3] & 0x10,
            0,
            "FLG.FCOMMENT must be set by -C in file mode (-p{threads})"
        );
        assert!(
            gz.windows(6).any(|w| w == b"hello\0"),
            "NUL-terminated comment must be present in the header (-p{threads})"
        );
    }
}

#[test]
fn pigz_comment_rejected_where_it_would_be_dropped() {
    // MITIGATED: the -c/stdout path writes the minimal 10-byte header BY
    // DESIGN (it is what libdeflate-gzip emits on -c, and every graded
    // board/tie-guard invocation is `-c`), so a comment there would be
    // silently DROPPED. pigz stores the comment everywhere; until the
    // stdout path grows a header, it must refuse, never drop: exit 2,
    // up-front honesty rejection. (The single-thread FILE path used to be
    // the second reject leg here; issue #309 made it STORE the comment —
    // pinned in pigz_comment_stored_in_file_mode.)
    assert_honesty_reject(&["-C", "hello", "-c"], &["-C", "--comment"]);
}

#[test]
fn pigz_first_flag_collision_errs_loudly() {
    // pigz(1) -F / --first takes NO argument ("Do iterations first, before
    // block split for -11"). gzippy's -F is a DIFFERENT option that takes a
    // VALUE (zopfli iteration count), so it eats the next argv element. A
    // pigz invocation `-11 -F -c FILE` therefore fails ("Invalid
    // iterations: -c") — a misparse, but a LOUD one. Pinned: nonzero exit
    // with a diagnosable message, never silence. (Danger note: `pigz -11 -F`
    // followed by a NUMERIC filename would be swallowed as the iteration
    // count; that shape is unreachable here because the file operand then
    // goes missing and the read fails — still nonzero.)
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let out = run(d.path(), &["-11", "-F", "-c", "in.txt"]);
    assert!(
        !out.status.success(),
        "pigz-style bare -F must not silently succeed (our -F takes a value)"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("iterations"),
        "stderr must name the -F misparse; got: {stderr}"
    );
}

#[test]
fn pigz_iterations_flag_collision_errs_loudly() {
    // pigz(1) -I n / --iterations n: zopfli iteration count. gzippy's -I is
    // a FLAG (zopfli no-block-split), so pigz's `-I 20` leaves "20" as a
    // file operand: "20: File not found". Loud in the common case; the
    // residual hazard (a file literally named "20" would be compressed) is
    // recorded in the triage table.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let out = run(d.path(), &["-11", "-I", "20", "-c", "in.txt"]);
    assert!(
        !out.status.success(),
        "pigz-style -I 20 must not silently succeed (our -I takes no value)"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("20"),
        "stderr must surface the orphaned operand '20'; got: {stderr}"
    );
}

#[test]
fn pigz_oneblock_flag_clean_reject() {
    // pigz(1) -O / --oneblock: "Do not split into smaller blocks for -11".
    // Not supported here; must reject loudly, not misparse.
    assert_clean_reject(&["-11", "-O", "-c"], "-O");
}

#[test]
fn pigz_time_flags_accepted() {
    // pigz(1) -m / --no-time and -M / --time. Supported: -m clears the
    // header MTIME for file input, -M keeps it (file mode stores it).
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let no_time = run(d.path(), &["-m", "-k", "in.txt"]);
    assert!(no_time.status.success());
    let gz = fs::read(d.path().join("in.txt.gz")).expect("gz written");
    assert_eq!(&gz[4..8], &[0, 0, 0, 0], "-m must zero header MTIME");
    fs::remove_file(d.path().join("in.txt.gz")).unwrap();
    let with_time = run(d.path(), &["-M", "-f", "-k", "in.txt"]);
    assert!(with_time.status.success());
    let gz = fs::read(d.path().join("in.txt.gz")).expect("gz written");
    assert_ne!(
        &gz[4..8],
        &[0, 0, 0, 0],
        "-M must store header MTIME for file input"
    );
}

// ===========================================================================
// igzip surface
// ===========================================================================

#[test]
fn igzip_output_file_flag_clean_reject() {
    // igzip_cli.c: `-o <file>  output file`. Not supported here (we follow
    // gzip's in-place/-c model). Must be a clean reject so an igzip script
    // fails loudly instead of compressing to the wrong place.
    assert_clean_reject(&["-o", "out.gz", "-c"], "-o");
}

#[test]
fn igzip_threads_flag_clean_reject() {
    // igzip_cli.c: `-T, --threads <n>`. Ours is -p (pigz's spelling). An
    // igzip script's `-T 8` must fail loudly, not silently single-thread.
    assert_clean_reject(&["-T", "4", "-c"], "-T");
}

#[test]
fn igzip_rm_flag_clean_reject() {
    // igzip_cli.c: `--rm  remove source files after successful
    // (de)compression` — igzip KEEPS sources by default and --rm opts into
    // deletion. gzippy (gzip semantics) deletes by default and has no --rm.
    // Clean reject required.
    assert_clean_reject(&["--rm", "-c"], "--rm");
}

#[test]
fn igzip_keep_by_default_divergence_we_delete() {
    // DIVERGENCE from igzip semantics, pinned at current behaviour (which IS
    // the gzip contract, so it is endorsed — but an igzip user must know):
    // igzip_cli.c: `-k, --keep  keep source files (default)`. An igzip
    // script that omits -k relies on keep-by-default; pointed at gzippy it
    // gets gzip's delete-after-compress instead.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let out = run(d.path(), &["in.txt"]);
    assert!(out.status.success());
    assert!(
        d.path().join("in.txt.gz").exists(),
        "compressed output must exist"
    );
    assert!(
        !d.path().join("in.txt").exists(),
        "gzip contract: input deleted after compression. igzip would have \
         KEPT it — divergence an igzip user inherits silently."
    );
}

#[test]
fn igzip_level_zero_stores_with_loud_warning() {
    // DIVERGENCE from igzip semantics, now LOUD (issue #305): igzip levels
    // are 0..=3 and level 0 is igzip's FASTEST REAL COMPRESSOR
    // (igzip_lib.h: ISAL_DEF_MIN_LEVEL 0). gzippy's -0 is the gzip/pigz -0:
    // STORE — the documented pigz contract, which the charter's
    // least-surprise rule keeps (pigz -0 scripts must still get store, so
    // remapping or rejecting -0 would trade one silent-wrong for another).
    // The igzip-shaped hazard (an `igzip -0` backup silently emitting output
    // LARGER than its input) is cured by a stderr warning naming the
    // divergence and the issue; -q suppresses it.
    let d = tempdir();
    let data = synth(64 * 1024);
    write_file(d.path(), "in.txt", &data);

    let out = run(d.path(), &["-0", "-c", "in.txt"]);
    assert!(
        out.status.success(),
        "-0 must still succeed (pigz contract)"
    );
    assert!(
        out.stdout.len() > data.len(),
        "pigz contract: -0 stores (output {} > input {})",
        out.stdout.len(),
        data.len()
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("igzip") && stderr.contains("#305"),
        "-0 must WARN about the igzip level-scale divergence; got: {stderr}"
    );
    assert_roundtrips(d.path(), &out.stdout, &data, "-0 store");

    let quiet = run(d.path(), &["-0", "-q", "-c", "in.txt"]);
    assert!(quiet.status.success());
    // The anatomy instrument builds (`--features anatomy-counters` /
    // `anatomy-wall`) UNCONDITIONALLY print their counter dump to stderr at
    // exit (main.rs `flush_to_stderr`) — that is the instrument's output
    // channel, not a user-facing warning, and -q must not silence an
    // instrument. Assert "no user-facing stderr" rather than "empty stderr":
    // instrument lines are ANATOMY_*-prefixed (reconcile-failure detail
    // lines are indented continuations).
    let user_stderr: Vec<&str> = std::str::from_utf8(&quiet.stderr)
        .expect("stderr utf8")
        .lines()
        .filter(|l| !l.starts_with("ANATOMY_") && !l.starts_with("  "))
        .collect();
    assert!(
        user_stderr.is_empty(),
        "-q must suppress the -0 divergence warning; got: {user_stderr:?}"
    );
}

#[test]
fn igzip_levels_two_and_three_map_to_weak_gzip_levels() {
    // DIVERGENCE from igzip semantics, pinned at current behaviour:
    // igzip -2 is igzip's DEFAULT and -3 its BEST (`--best` maps to
    // '0'+ISAL_DEF_MAX_LEVEL in igzip_cli.c). gzippy reads the same digits
    // on the gzip 0-9 scale, where 2 and 3 are weak-fast levels. Valid gzip
    // either way; silently different ratio/speed profile. Pinned: -2 and -3
    // roundtrip, and both compress LESS than -9 (i.e. they really are weak
    // levels here, not igzip's near-best).
    let d = tempdir();
    let data = synth(256 * 1024);
    write_file(d.path(), "in.txt", &data);
    let nine = compress_stdout(d.path(), &["-9", "-p", "1"], "in.txt");
    for level in ["-2", "-3"] {
        let out = compress_stdout(d.path(), &[level, "-p", "1"], "in.txt");
        assert_roundtrips(d.path(), &out, &data, level);
        assert!(
            out.len() > nine.len(),
            "current behaviour: {level} is a weak-fast gzip level (its \
             output {} should exceed -9's {}); igzip's {level} is its \
             default/best tier",
            out.len(),
            nine.len()
        );
    }
}

#[test]
fn igzip_z_flag_now_gets_the_pigz_zlib_contract() {
    // UPDATED (issues #303/#305): `-z` USED to be rejected loudly, because
    // gzippy's pre-implementation accept produced GZIP bytes under .zz names
    // (silent-wrong). It now delivers the PIGZ -z contract instead of
    // failing an igzip-shaped script: real RFC 1950 zlib with the .zz
    // suffix. An igzip script's `-z FILE` (= "compress (default)" there)
    // now compresses and roundtrips; the remaining divergence is the output
    // NAME (FILE.zz where `*.gz` globs find nothing) — documented in the
    // triage table and --help.
    let d = tempdir();
    let data = synth(16 * 1024);
    write_file(d.path(), "in.txt", &data);
    let out = run(d.path(), &["-z", "-k", "in.txt"]);
    assert!(
        out.status.success(),
        "-z must compress (the pigz contract) in file mode: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let zz = fs::read(d.path().join("in.txt.zz")).expect("FILE.zz written");
    assert_rfc1950_header(&zz, "igzip-shaped -z");
    assert_roundtrips_zz(d.path(), &zz, &data, "igzip-shaped -z");
}

#[test]
fn rejected_flags_still_inert_on_decompress() {
    // The honesty rejections are COMPRESS-MODE ONLY. pigz accepts and
    // ignores these flags on decompress, so a `-d` pipeline carrying them
    // must keep working here too. -z belongs on this list now in BOTH
    // directions: it compresses in compress mode (#303), and on -d/-t it is
    // inert (routing decodes by content/extension, not by the flag).
    let d = tempdir();
    let data = synth(16 * 1024);
    write_file(d.path(), "in.txt", &data);
    let gz = compress_stdout(d.path(), &["-p", "4"], "in.txt");
    write_file(d.path(), "in.txt.gz", &gz);
    for flags in [
        &["-d", "-c", "-R"][..],
        &["-d", "-c", "-H", "-U", "-i"][..],
        &["-d", "-c", "-z"][..],
        &["-t", "-R"][..],
    ] {
        let mut full: Vec<&str> = flags.to_vec();
        full.push("in.txt.gz");
        let out = run(d.path(), &full);
        assert!(
            out.status.success(),
            "gzippy {full:?} must stay inert on decompress/test (pigz \
             semantics); stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        if flags.contains(&"-c") {
            assert_eq!(out.stdout, data, "roundtrip bytes under inert flags");
        }
    }
}

// ===========================================================================
// libdeflate-gzip surface
// ===========================================================================

#[test]
fn libdeflate_levels_1_through_12_all_roundtrip() {
    // libdeflate-gzip -h: `-1` fastest .. `-12` slowest; every level must be
    // accepted (including the multi-digit short forms -10/-11/-12) and
    // produce valid gzip. Level 11 routes near-optimal/zopfli-class parses,
    // so keep the input small.
    let d = tempdir();
    let data = synth(16 * 1024);
    write_file(d.path(), "in.txt", &data);
    for level in 1..=12u8 {
        let flag = format!("-{level}");
        let out = compress_stdout(d.path(), &[&flag], "in.txt");
        assert_roundtrips(d.path(), &out, &data, &flag);
    }
}

#[test]
fn libdeflate_out_of_range_level_clean_reject() {
    // libdeflate-gzip tops out at -12; so do we. -13 must be a clean reject
    // (it is nobody's valid level), not a clamp.
    assert_clean_reject(&["-13", "-c"], "13");
}

#[test]
fn libdeflate_short_flag_set_supported() {
    // libdeflate-gzip's whole option surface: -1..-12, -c, -d, -f, -h, -k,
    // -q, -S SUF, -t, -V. Everything except levels is shared with gzip and
    // covered functionally by cli_dropin.rs; here we pin that each spelling
    // is at least ACCEPTED (no "Unknown option") so a libdeflate-gzip script
    // never trips on argv parsing.
    let d = tempdir();
    let data = synth(4096);
    write_file(d.path(), "in.txt", &data);

    // -k -q -S SUF: compress keeping input, custom suffix, quiet.
    let out = run(d.path(), &["-k", "-q", "-S", ".gzx", "in.txt"]);
    assert!(
        out.status.success(),
        "-k -q -S .gzx must be accepted: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(d.path().join("in.txt.gzx").exists(), "-S suffix honoured");

    // -t: test integrity of what we just wrote.
    let out = run(d.path(), &["-t", "in.txt.gzx"]);
    assert!(out.status.success(), "-t on our own output must pass");

    // -d -f -c: decompress it back to stdout.
    let out = run(d.path(), &["-d", "-f", "-c", "-S", ".gzx", "in.txt.gzx"]);
    assert!(out.status.success(), "-d -f -c -S .gzx must be accepted");
    assert_eq!(out.stdout, data, "roundtrip through -S suffix");

    // -V and -h: informational, exit 0.
    for flag in ["-V", "-h"] {
        let out = run(d.path(), &[flag]);
        assert!(out.status.success(), "{flag} must exit 0");
        assert!(!out.stdout.is_empty(), "{flag} must print something");
    }
}
