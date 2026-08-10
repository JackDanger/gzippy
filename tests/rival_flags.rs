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
//!   igzip -0  -> gzippy stores: output LARGER than input. Silent, extreme.
//!   igzip -1  -> weak-fast gzip L1 (roughly comparable intent; ratio differs).
//!   igzip -2  -> gzip L2, a weak-fast level — but -2 is igzip's DEFAULT,
//!                sized between our L6-ish intent. Valid gzip, silently
//!                different ratio/speed profile.
//!   igzip -3  -> gzip L3 — but -3 is igzip's BEST (`--best`). A user asking
//!                igzip for maximum compression gets our third-weakest level.
//!   igzip -z  -> igzip: "compress (default)", a no-op modifier. gzippy: the
//!                pigz meaning, SWITCH CONTAINER TO ZLIB — writes FILE.zz.
//!
//! There is no error in any of these cases; each is pinned below.
//!
//! ## Triage table (matches the PR body)
//!
//! | rival flag                | our behaviour                                   | severity |
//! |---------------------------|--------------------------------------------------|----------|
//! | pigz -R/--rsyncable       | accepted, routed, output NOT rsyncable (4% resync vs pigz 94%) | silent-wrong HIGH |
//! | pigz -z/--zlib            | accepted; emits GZIP bytes with .zz suffix; own -d then fails on stdin path | silent-wrong HIGH |
//! | pigz -K/--zip             | accepted; emits GZIP bytes with .zip suffix (no PK entry) | silent-wrong HIGH |
//! | pigz -b/--blocksize N     | bare N is BYTES; pigz N means N KiB (128 -> clean error; 4096 -> silently 1024x smaller blocks) | silent-wrong HIGH |
//! | pigz -H/--huffman         | accepted, silently a no-op (help text still claims it) | silent-wrong MED |
//! | pigz -U/--rle             | accepted, silently a no-op (help text still claims it) | silent-wrong MED |
//! | pigz -i/--independent     | accepted, output byte-identical to without (no independence guarantee asserted) | silent-noop MED |
//! | pigz -C/--comment ccc     | honoured in file mode; silently DROPPED on the -c/stdout path | silent-wrong MED |
//! | pigz -A/--alias xxx       | accepted, silently ignored (only meaningful with --zip, which is itself broken) | silent-noop LOW |
//! | pigz -F/--first           | collides: our -F takes a VALUE (zopfli iterations); consumes the next argv | misparse-but-errs MED |
//! | pigz -I/--iterations n    | collides: our -I is a flag (no-block-split); n becomes a FILE operand | misparse-but-errs MED |
//! | pigz -O/--oneblock        | clean reject ("Unknown option: -O")             | clean-reject |
//! | pigz -M/-m, -N/-n, -p, -S, -Y, -v, -q, -r, -t, -l, -L, -V, -d, -f, -k, -c, --fast, --best | supported | ok |
//! | igzip -o FILE             | clean reject ("Unknown option: -o")             | clean-reject |
//! | igzip -T/--threads n      | clean reject ("Unknown option: -T"; ours is -p) | clean-reject |
//! | igzip --rm                | clean reject ("Unknown option: --rm")           | clean-reject |
//! | igzip -0                  | STORE (larger than input); igzip -0 compresses  | silent-wrong HIGH |
//! | igzip -2 / -3             | weak-fast gzip levels; igzip's default/best      | silent-wrong MED |
//! | igzip -z                  | zlib-container collision (see pigz -z row)       | silent-wrong HIGH |
//! | igzip keep-by-default     | we DELETE the input after compress (gzip rule); igzip keeps | divergence MED |
//! | libdeflate-gzip -1..-12   | supported, valid gzip at every level             | ok |
//! | libdeflate-gzip -c -d -f -h -k -q -S -t -V | supported                      | ok |

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
fn pigz_blocksize_bare_number_is_bytes_not_kib() {
    // DIVERGENCE from pigz contract, pinned at current behaviour:
    // pigz(1): "-b, --blocksize mmm  Set compression block size to mmmK" —
    // a bare number is KIBIBYTES. gzippy parses a bare number as BYTES:
    //   * `-b 128` (pigz: 128 KiB, its default) is a clean error here
    //     ("Block size must be at least 1K") — loud, safe;
    //   * `-b 4096` (pigz: 4 MiB) is silently accepted as 4096 BYTES —
    //     1024x smaller blocks, silently different framing and ratio.
    // The identity below proves the bare-number unit is bytes: 131072 bare
    // must equal 128k. Under pigz semantics they would differ (128 MiB vs
    // 128 KiB grids on a 256 KiB input).
    let d = tempdir();
    let data = synth(256 * 1024);
    write_file(d.path(), "in.txt", &data);

    let small = run(d.path(), &["-b", "128", "-c", "in.txt"]);
    assert!(
        !small.status.success(),
        "current behaviour: bare -b 128 (bytes) is below the 1K floor and \
         must error; if this now succeeds, the -b unit may have changed — \
         re-triage the pigz -b row"
    );

    let bare = compress_stdout(d.path(), &["-p", "4", "-b", "131072"], "in.txt");
    let suffixed = compress_stdout(d.path(), &["-p", "4", "-b", "128k"], "in.txt");
    assert_eq!(
        bare, suffixed,
        "current behaviour: bare 131072 == 128k, i.e. the bare -b unit is \
         BYTES. pigz's unit is KiB — see the divergence test below."
    );
}

#[test]
#[ignore] // DIVERGENCE from pigz contract: bare -b N means N KiB in pigz.
fn pigz_blocksize_contract_bare_number_means_kib() {
    // pigz(1): `-b 128` is pigz's default block size (128 KiB) and must be
    // accepted; a pigz-compatible gzippy would treat `-b 128` == `-b 128k`.
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
}

#[test]
fn pigz_rsyncable_accepted_and_roundtrips() {
    // pigz -R / --rsyncable: accepted, exit 0, routed to a dedicated path
    // (output differs from the plain path), valid gzip. Whether the OUTPUT
    // actually has the rsyncable property is the ignored divergence test
    // below — measured 2026-08-09 it does NOT.
    let d = tempdir();
    let data = synth(256 * 1024);
    write_file(d.path(), "in.txt", &data);
    let rsync = compress_stdout(d.path(), &["-p", "4", "-R"], "in.txt");
    assert_roundtrips(d.path(), &rsync, &data, "-R");
    let plain = compress_stdout(d.path(), &["-p", "4"], "in.txt");
    assert_ne!(
        rsync, plain,
        "-R is expected to route to the dedicated rsyncable path (its bytes \
         have always differed from the plain path); identical output would \
         mean the flag became a full no-op — re-triage"
    );
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
#[ignore] // DIVERGENCE from pigz contract: -R output is not actually rsyncable.
fn pigz_rsyncable_contract_output_resyncs_after_edit() {
    // pigz(1) -R: "Input-determined block locations for rsync" — after a
    // small local edit, the bulk of the compressed stream must re-align so
    // rsync's rolling checksum can skip it. Measured 2026-08-09 on this
    // probe (256 KiB synthetic text, 100 bytes inserted at offset 1000,
    // 512-byte rolling windows):
    //     pigz -R      resynced 375/397 blocks (94%)
    //     pigz plain   23/394  (6%)
    //     gzippy -R    16/391  (4%)   <- WORSE than gzippy plain (62/391)
    // gzippy -R exits 0 and emits valid gzip WITHOUT the property the flag
    // exists to provide — the silent-wrong case for backup pipelines.
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
fn pigz_huffman_flag_is_silent_noop() {
    // DIVERGENCE from pigz contract, pinned at current behaviour:
    // pigz(1) -H / --huffman: "Use only Huffman coding for compression" — a
    // strategy change. gzippy parses the flag (cli.rs sets args.huffman) but
    // NOTHING consumes it: output is byte-identical with and without. Worse,
    // `gzippy --help` still advertises "-H, --huffman  Huffman-only
    // compression". Silent no-op of an advertised flag.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(64 * 1024));
    for p in ["1", "4"] {
        let with = compress_stdout(d.path(), &["-p", p, "-H"], "in.txt");
        let without = compress_stdout(d.path(), &["-p", p], "in.txt");
        assert_eq!(
            with, without,
            "current behaviour at -p {p}: -H is a byte-for-byte no-op. If \
             this fails, -H gained behaviour — update the triage table and \
             the pigz -H issue."
        );
    }
}

#[test]
fn pigz_rle_flag_is_silent_noop() {
    // DIVERGENCE from pigz contract, pinned at current behaviour:
    // pigz(1) -U / --rle: "Use run-length encoding for compression". Parsed
    // (cli.rs sets args.rle), never consumed, advertised in --help. Same
    // class as -H.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(64 * 1024));
    for p in ["1", "4"] {
        let with = compress_stdout(d.path(), &["-p", p, "-U"], "in.txt");
        let without = compress_stdout(d.path(), &["-p", p], "in.txt");
        assert_eq!(
            with, without,
            "current behaviour at -p {p}: -U is a byte-for-byte no-op"
        );
    }
}

#[test]
fn pigz_independent_flag_is_silent_noop_on_stdout_path() {
    // pigz(1) -i / --independent: "Compress blocks independently for damage
    // recovery". Pinned current behaviour: output is byte-identical with and
    // without -i at L6 and L9, -p1 and -p4 (measured 2026-08-09). Note
    // src/compress/io.rs:203 consults args.independent only for levels 7-9,
    // and even at L9 the stdout-path bytes do not change. No independence
    // property is asserted or denied here — only that the flag changes
    // nothing observable, so a pigz user cannot tell whether they got it.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(64 * 1024));
    for level in ["-6", "-9"] {
        for p in ["1", "4"] {
            let with = compress_stdout(d.path(), &[level, "-p", p, "-i"], "in.txt");
            let without = compress_stdout(d.path(), &[level, "-p", p], "in.txt");
            assert_eq!(
                with, without,
                "current behaviour at {level} -p {p}: -i is a byte-for-byte no-op"
            );
        }
    }
}

#[test]
fn pigz_zlib_flag_currently_emits_gzip_magic() {
    // DIVERGENCE from pigz contract, pinned at current behaviour:
    // pigz(1) -z / --zlib: "Compress to zlib (.zz) instead of gzip format".
    // gzippy accepts -z, switches the SUFFIX to .zz — and then emits GZIP
    // bytes (magic 1f 8b, not a zlib 0x78 CMF). File mode therefore writes
    // FILE.zz containing a gzip stream, and gzippy's own `-d -c FILE.zz`
    // rejects it ("zlib decompression failed"), because decompression routes
    // by the suffix the compressor just lied with.
    let d = tempdir();
    let data = synth(16 * 1024);
    write_file(d.path(), "in.txt", &data);

    let out = compress_stdout(d.path(), &["-z"], "in.txt");
    assert_eq!(
        &out[..2],
        &[0x1f, 0x8b],
        "current behaviour: -z output starts with the GZIP magic; pigz emits \
         a zlib stream (first byte 0x78). If this fails, -z gained a real \
         zlib container — update the triage table, the issue, and the \
         ignored contract test."
    );

    // File mode: the suffix claims zlib while the bytes are gzip.
    let file_mode = run(d.path(), &["-z", "-k", "in.txt"]);
    assert!(
        file_mode.status.success(),
        "current behaviour: file-mode -z exits 0; stderr: {}",
        String::from_utf8_lossy(&file_mode.stderr)
    );
    let zz = fs::read(d.path().join("in.txt.zz")).expect("current behaviour: -z writes FILE.zz");
    assert_eq!(
        &zz[..2],
        &[0x1f, 0x8b],
        "current behaviour: FILE.zz contains gzip bytes"
    );
}

#[test]
#[ignore] // DIVERGENCE from pigz contract: -z must emit a zlib stream.
fn pigz_zlib_contract_emits_zlib_stream() {
    // pigz(1) -z: the output must be RFC 1950 zlib — CMF low nibble 8
    // (deflate), and (CMF<<8|FLG) divisible by 31. And our own decoder must
    // roundtrip the .zz file we just wrote.
    let d = tempdir();
    let data = synth(16 * 1024);
    write_file(d.path(), "in.txt", &data);
    let out = compress_stdout(d.path(), &["-z"], "in.txt");
    assert_eq!(
        out[0] & 0x0f,
        8,
        "zlib CMF: compression method must be deflate"
    );
    let check = ((out[0] as u16) << 8) | out[1] as u16;
    assert_eq!(
        check % 31,
        0,
        "zlib header FCHECK must make CMF|FLG divisible by 31"
    );
    write_file(d.path(), "rt.zz", &out);
    let rt = run(d.path(), &["-d", "-c", "rt.zz"]);
    assert!(rt.status.success(), "own -d of own -z output must succeed");
    assert_eq!(rt.stdout, data, "zlib roundtrip bytes");
}

#[test]
fn pigz_zip_flag_currently_emits_gzip_magic() {
    // DIVERGENCE from pigz contract, pinned at current behaviour:
    // pigz(1) -K / --zip: "Compress to PKWare zip (.zip) single entry
    // format". gzippy accepts -K, switches the suffix to .zip, and emits
    // GZIP bytes (no PK\x03\x04 local-file header). A downstream `unzip`
    // will refuse the file a script just named FILE.zip.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(16 * 1024));
    let out = compress_stdout(d.path(), &["-K"], "in.txt");
    assert_eq!(
        &out[..2],
        &[0x1f, 0x8b],
        "current behaviour: -K output starts with the GZIP magic; pigz emits \
         PK zip. If this fails, -K gained a real zip container — update the \
         triage table and issue."
    );
}

#[test]
#[ignore] // DIVERGENCE from pigz contract: -K must emit a PK zip entry.
fn pigz_zip_contract_emits_pk_zip() {
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(16 * 1024));
    let out = compress_stdout(d.path(), &["-K"], "in.txt");
    assert_eq!(&out[..2], b"PK", "pigz -K contract: PKWare zip magic");
}

#[test]
fn pigz_alias_flag_is_silently_ignored() {
    // pigz(1) -A xxx / --alias xxx: "Use xxx as the name for any --zip entry
    // from stdin". Pinned current behaviour: accepted (exit 0) and ignored —
    // args.alias has no consumer. Low severity only because --zip itself
    // does not produce zip (see above); if -K is ever fixed, -A must be
    // wired up in the same change or this becomes a real silent-wrong.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let with = compress_stdout(d.path(), &["-A", "entryname"], "in.txt");
    let without = compress_stdout(d.path(), &[], "in.txt");
    assert_eq!(
        with, without,
        "current behaviour: -A is a byte-for-byte no-op"
    );
}

#[test]
fn pigz_comment_stored_in_file_mode() {
    // pigz(1) -C ccc / --comment ccc: "Put comment ccc in the gzip or zip
    // header". SUPPORTED in file mode: FLG.FCOMMENT (0x10) set and the
    // NUL-terminated comment present.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let out = run(d.path(), &["-C", "hello", "-k", "in.txt"]);
    assert!(
        out.status.success(),
        "file-mode -C failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let gz = fs::read(d.path().join("in.txt.gz")).expect("in.txt.gz written");
    assert_eq!(&gz[..2], &[0x1f, 0x8b]);
    assert_ne!(
        gz[3] & 0x10,
        0,
        "FLG.FCOMMENT must be set by -C in file mode"
    );
    assert!(
        gz.windows(6).any(|w| w == b"hello\0"),
        "NUL-terminated comment must be present in the header"
    );
}

#[test]
fn pigz_comment_dropped_on_stdout_path() {
    // DIVERGENCE from pigz contract, pinned at current behaviour: the SAME
    // -C flag that works in file mode is silently dropped on the -c/stdout
    // path — FLG.FCOMMENT stays clear (measured 2026-08-09). pigz stores the
    // comment on both paths.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let out = compress_stdout(d.path(), &["-C", "hello"], "in.txt");
    assert_eq!(
        out[3] & 0x10,
        0,
        "current behaviour: stdout path drops the -C comment (FCOMMENT \
         clear). If this fails, the divergence was fixed — update the triage \
         table and drop this pin in favour of the file-mode assertion."
    );
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
fn igzip_level_zero_maps_to_store_not_fastest_compression() {
    // DIVERGENCE from igzip semantics, pinned at current behaviour:
    // igzip levels are 0..=3 and level 0 is igzip's FASTEST REAL COMPRESSOR
    // (igzip_lib.h: ISAL_DEF_MIN_LEVEL 0). gzippy's -0 is pigz's -0: STORE.
    // An `igzip -0` pipeline pointed here silently emits output LARGER than
    // its input. Both behaviours are valid gzip; the divergence is the
    // silent loss of all compression.
    let d = tempdir();
    let data = synth(64 * 1024);
    write_file(d.path(), "in.txt", &data);
    let out = compress_stdout(d.path(), &["-0"], "in.txt");
    assert!(
        out.len() > data.len(),
        "current behaviour: -0 stores (output {} > input {}). igzip -0 \
         compresses. If this fails, the -0 mapping changed — re-triage the \
         igzip level-mapping issue.",
        out.len(),
        data.len()
    );
    assert_roundtrips(d.path(), &out, &data, "-0 store");
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
fn igzip_compress_flag_z_collides_with_zlib_container() {
    // DIVERGENCE from igzip semantics, pinned at current behaviour:
    // igzip_cli.c: `-z, --compress  compress file (default)` — a no-op
    // modifier. gzippy gives -z the PIGZ meaning: switch container to zlib,
    // which today means "write FILE.zz with gzip bytes inside" (see the
    // pigz -z pins). So `igzip -z FILE` pointed here produces FILE.zz
    // instead of FILE.gz — downstream globs for *.gz find nothing.
    let d = tempdir();
    write_file(d.path(), "in.txt", &synth(4096));
    let out = run(d.path(), &["-z", "-k", "in.txt"]);
    assert!(
        out.status.success(),
        "current behaviour: -z file mode exits 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        d.path().join("in.txt.zz").exists() && !d.path().join("in.txt.gz").exists(),
        "current behaviour: -z writes FILE.zz, not FILE.gz. igzip's -z is a \
         plain 'compress' no-op and would write FILE.gz."
    );
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
