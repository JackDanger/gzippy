//! Failure-path and metadata contract tests, plus an ecosystem-decoder leg.
//!
//! gzip's contract is not only "same bytes out on success" — it includes what
//! happens to the INPUT and the OUTPUT when things go wrong (permission
//! denied, disk full, SIGINT) and which metadata travels with the file (mode,
//! mtime). None of that was tested before this suite, and our outputs'
//! decodability was proven only against C decoders + flate2 — never against
//! an independent zlib-lineage implementation.
//!
//! Every pinned behaviour below was probed against GNU gzip 1.14
//! (Homebrew, macOS, 2026-08-09) before the pin was written:
//!
//!   * metadata: `gzip f` gives f.gz the input's mode (0640 stayed 0640) and
//!     mtime (1577963045 stayed 1577963045); `gzip -d` restores both onto the
//!     recreated file (mtime from the header MTIME field).
//!   * read-only dir: `gzip rodir/h.txt` -> "Permission denied", exit 1,
//!     input intact, no output.
//!   * output path occupied (by a directory): "already exists; not
//!     overwritten", exit 2, input intact.
//!   * SIGINT mid-compression: partial .gz existed at signal time; gzip
//!     removes it, leaves the input, and dies by re-raised SIGINT
//!     (wait status = signaled, signal 2).
//!   * ENOSPC (2 MB HFS ram disk): "No space left on device", exit 1,
//!     partial output removed, input intact.
//!
//! gzippy matched every one of those probes at both default threads and
//! `-p 1` (see the PR that introduced this file for the probe transcript).
//! The only observed divergence is cosmetic: at default (multi-thread)
//! routing the ENOSPC message is "IO error: write failed" rather than naming
//! ENOSPC the way gzip and our own `-p 1` path do. Severity: low
//! (message quality only; exit code and file-state behaviour match).
//!
//! Runtime budget: the whole suite stays under ~60 s; the SIGINT test is
//! bounded by a poll-then-kill loop and the disk-full test by a 2 MB device.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant, SystemTime};

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

/// Compressible-but-not-trivial synthetic text.
fn text_bytes(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 32);
    let mut x: u64 = 0x243f_6a88_85a3_08d3;
    while out.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(b"the quick brown fox jumps over ");
        out.push(b'a' + (x % 26) as u8);
        out.push(b'\n');
    }
    out.truncate(len);
    out
}

/// Poorly-compressible xorshift bytes (keeps compressed size close to input
/// size — used to overflow the tiny disk-full volume).
fn noise_bytes(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 8);
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    while out.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// CSV-ish structured bytes — a third content flavour for the ecosystem leg.
fn csv_bytes(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 64);
    let mut x: u64 = 0x1234_5678_9abc_def0;
    let mut row = 0u64;
    while out.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(
            format!(
                "{row},widget-{},{},{:.2},EUR\n",
                x % 997,
                x % 100_000,
                (x % 9999) as f64 / 100.0
            )
            .as_bytes(),
        );
        row += 1;
    }
    out.truncate(len);
    out
}

fn mode_of(p: &Path) -> u32 {
    fs::metadata(p).expect("stat").permissions().mode() & 0o7777
}

fn mtime_secs(p: &Path) -> u64 {
    fs::metadata(p)
        .expect("stat")
        .modified()
        .expect("mtime")
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("mtime after epoch")
        .as_secs()
}

/// Set a file's mode and mtime (seconds precision is all gzip's MTIME field
/// carries, so seconds is what the pins compare).
fn set_mode_and_mtime(p: &Path, mode: u32, mtime_unix_secs: u64) {
    fs::set_permissions(p, fs::Permissions::from_mode(mode)).expect("chmod");
    let f = fs::File::options()
        .write(true)
        .open(p)
        .expect("open for utimes");
    f.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(mtime_unix_secs))
        .expect("set mtime");
}

// Distinctive fixture metadata: mode 0640, mtime 2020-01-02T03:04:05Z.
const FIXTURE_MODE: u32 = 0o640;
const FIXTURE_MTIME: u64 = 1_577_934_245; // any old, distinctive value; compared exactly

// ---------------------------------------------------------------------------
// 1. Metadata preservation
// ---------------------------------------------------------------------------

/// gzip contract (probed, GNU gzip 1.14): `gzip f` gives f.gz the input's
/// permission bits and mtime.
#[test]
fn compress_preserves_mode_and_mtime() {
    let dir = tempdir();
    let input = write_file(dir.path(), "meta.txt", &text_bytes(64 * 1024));
    set_mode_and_mtime(&input, FIXTURE_MODE, FIXTURE_MTIME);

    let out = run(dir.path(), &["meta.txt"]);
    assert!(out.status.success(), "compress failed: {out:?}");

    let gz = dir.path().join("meta.txt.gz");
    assert!(gz.exists(), "no .gz produced");
    assert_eq!(mode_of(&gz), FIXTURE_MODE, "mode not carried onto .gz");
    assert_eq!(mtime_secs(&gz), FIXTURE_MTIME, "mtime not carried onto .gz");
}

/// gzip contract (probed): `gzip -d f.gz` restores the original mode and
/// mtime onto the recreated file (mtime travels in the gzip header).
#[test]
fn decompress_restores_mode_and_mtime() {
    let dir = tempdir();
    let payload = text_bytes(64 * 1024);
    let input = write_file(dir.path(), "meta.txt", &payload);
    set_mode_and_mtime(&input, FIXTURE_MODE, FIXTURE_MTIME);

    let out = run(dir.path(), &["meta.txt"]);
    assert!(out.status.success(), "compress failed: {out:?}");
    assert!(
        !dir.path().join("meta.txt").exists(),
        "input should be removed"
    );

    let out = run(dir.path(), &["-d", "meta.txt.gz"]);
    assert!(out.status.success(), "decompress failed: {out:?}");

    let restored = dir.path().join("meta.txt");
    assert!(restored.exists(), "decompress did not recreate the file");
    assert_eq!(
        fs::read(&restored).expect("read restored"),
        payload,
        "payload mismatch"
    );
    assert_eq!(
        mode_of(&restored),
        FIXTURE_MODE,
        "mode not restored on decompress"
    );
    assert_eq!(
        mtime_secs(&restored),
        FIXTURE_MTIME,
        "mtime not restored on decompress (header MTIME)"
    );
}

/// gzip contract (probed, GNU gzip 1.14, 2026-08-12): compressing a named
/// FILE stores FNAME and MTIME in the gzip header — `gzip f` and `gzip -c f`
/// both emit FLG=0x08 with the file's mtime; only stdin input omits FNAME.
/// Issue #309: the -p1 file routes (mmap >128 KiB and streaming below it)
/// wrote FLG=0x00/MTIME=0, so `gzip -l`/`gzip -dN` could not restore
/// name/time from a -p1 archive while a -p4 archive worked. Pinned here on
/// BOTH -p1 routes and the -p4 route by parsing the header bytes directly —
/// no decoder fallback (the .gz file's own fs metadata) can fake a pass.
#[test]
fn file_output_header_stores_fname_and_mtime_on_p1_and_p4() {
    // 64 KiB rides the T1 streaming route, 256 KiB the T1 mmap route
    // (128 KiB threshold, pinned in t1_mmap_route.rs); 256 KiB at -p4 rides
    // the parallel pipelined route.
    for (threads, len) in [("1", 64 * 1024), ("1", 256 * 1024), ("4", 256 * 1024)] {
        let what = format!("-p{threads}, {len} B");
        let dir = tempdir();
        let input = write_file(dir.path(), "meta.txt", &text_bytes(len));
        set_mode_and_mtime(&input, FIXTURE_MODE, FIXTURE_MTIME);

        let out = run(dir.path(), &["-p", threads, "-k", "meta.txt"]);
        assert!(out.status.success(), "compress failed ({what}): {out:?}");
        let gz = fs::read(dir.path().join("meta.txt.gz")).expect("read .gz");

        assert_eq!(&gz[..3], &[0x1f, 0x8b, 0x08], "gzip magic/CM ({what})");
        assert_eq!(
            gz[3] & 0x08,
            0x08,
            "FLG.FNAME must be set on file output ({what})"
        );
        let hdr_mtime = u32::from_le_bytes([gz[4], gz[5], gz[6], gz[7]]);
        assert_eq!(
            u64::from(hdr_mtime),
            FIXTURE_MTIME,
            "header MTIME must be the input's mtime ({what})"
        );
        // No FEXTRA is ever set, so FNAME starts at byte 10.
        let fname_end = 10
            + gz[10..]
                .iter()
                .position(|&b| b == 0)
                .expect("NUL-terminated FNAME");
        assert_eq!(
            &gz[10..fname_end],
            b"meta.txt",
            "FNAME must be the input basename ({what})"
        );
    }
}

/// The restore leg of the pin above: rename the archive AND scrub its fs
/// mtime, so `-d -N` can recover the original name and time ONLY from the
/// header fields. (The older decompress_restores_mode_and_mtime test cannot
/// distinguish header MTIME from the fs-mtime fallback that
/// preserve_metadata copies onto the .gz.) Runs both -p1 and -p4 archives.
#[test]
fn decompress_dash_n_restores_name_and_mtime_from_header_alone() {
    for threads in ["1", "4"] {
        let dir = tempdir();
        let payload = text_bytes(256 * 1024);
        let input = write_file(dir.path(), "meta.txt", &payload);
        set_mode_and_mtime(&input, FIXTURE_MODE, FIXTURE_MTIME);

        let out = run(dir.path(), &["-p", threads, "meta.txt"]);
        assert!(
            out.status.success(),
            "compress failed (-p{threads}): {out:?}"
        );

        // Break both filesystem fallbacks: the archive name no longer hints
        // the original, and its fs mtime is wrong on purpose.
        let moved = dir.path().join("opaque-blob.gz");
        fs::rename(dir.path().join("meta.txt.gz"), &moved).expect("rename archive");
        set_mode_and_mtime(&moved, 0o644, FIXTURE_MTIME + 86_400);

        let out = run(dir.path(), &["-d", "-N", "opaque-blob.gz"]);
        assert!(
            out.status.success(),
            "decompress failed (-p{threads}): {out:?}"
        );

        let restored = dir.path().join("meta.txt");
        assert!(
            restored.exists(),
            "-dN must restore the header FNAME (-p{threads})"
        );
        assert_eq!(
            fs::read(&restored).expect("read restored"),
            payload,
            "payload mismatch (-p{threads})"
        );
        assert_eq!(
            mtime_secs(&restored),
            FIXTURE_MTIME,
            "-dN must restore the header MTIME, not the archive's fs mtime (-p{threads})"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Input removal ordering
// ---------------------------------------------------------------------------

/// Observable proxy for "remove input only after output is written and
/// closed": when the output cannot be created at all (read-only directory),
/// the input must survive and the exit must be nonzero. Probed: GNU gzip
/// prints "Permission denied", exits 1, leaves the input.
#[test]
fn failed_compress_in_readonly_dir_leaves_input_intact() {
    let dir = tempdir();
    let payload = text_bytes(32 * 1024);
    let input = write_file(dir.path(), "keepme.txt", &payload);

    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).expect("chmod dir ro");
    let out = run(dir.path(), &["keepme.txt"]);
    // Restore before asserting so tempdir cleanup works even on failure.
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).expect("chmod dir rw");

    assert!(
        !out.status.success(),
        "must fail when output dir is unwritable"
    );
    assert!(input.exists(), "input was lost on failed compression");
    assert_eq!(
        fs::read(&input).expect("read input"),
        payload,
        "input content changed"
    );
    assert!(
        !dir.path().join("keepme.txt.gz").exists(),
        "no output should exist"
    );
}

/// Same ordering proxy via a different failure: the output path is occupied
/// by a directory. Here the input's directory IS writable, so an
/// unlink-before-write bug would destroy the input — this test would catch
/// it. Probed: GNU gzip says "already exists; not overwritten", exits 2,
/// input intact. gzippy matches including the exit code.
#[test]
fn failed_compress_output_path_occupied_leaves_input_intact() {
    let dir = tempdir();
    let payload = text_bytes(32 * 1024);
    let input = write_file(dir.path(), "keepme.txt", &payload);
    fs::create_dir(dir.path().join("keepme.txt.gz")).expect("mkdir output blocker");

    let out = run(dir.path(), &["keepme.txt"]);

    assert!(
        !out.status.success(),
        "must fail when output path is a directory"
    );
    assert_eq!(
        out.status.code(),
        Some(2),
        "gzip exits 2 for 'already exists' (probed)"
    );
    assert!(input.exists(), "input was lost on failed compression");
    assert_eq!(
        fs::read(&input).expect("read input"),
        payload,
        "input content changed"
    );
}

// ---------------------------------------------------------------------------
// 3. SIGINT cleanup
// ---------------------------------------------------------------------------

/// gzip contract (probed): on SIGINT mid-compression gzip removes the
/// partial output, leaves the input, and dies by re-raised SIGINT (wait
/// status shows signal 2). gzippy matched at default threads and -p 1;
/// this pin runs default threads (the T>1 path — more processes to
/// coordinate, so the more failure-prone side).
#[test]
fn sigint_mid_compression_removes_partial_output_and_keeps_input() {
    let dir = tempdir();
    // ~64 MB of text at -9 gives a multi-second window (measured ~3.9 s wall
    // for 147 MB at default threads on an M-series laptop); the kill lands as
    // soon as the partial output appears, so typically well under 1 s in.
    let payload = text_bytes(64 * 1024 * 1024);
    let input = write_file(dir.path(), "slow.txt", &payload);
    let gz = dir.path().join("slow.txt.gz");

    let mut child = Command::new(gzippy())
        .args(["-9", "slow.txt"])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn gzippy");

    // Wait for the partial output to appear, then interrupt.
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut saw_partial = false;
    while Instant::now() < deadline {
        if gz.exists() {
            saw_partial = true;
            break;
        }
        if child.try_wait().expect("try_wait").is_some() {
            break; // exited before ever creating output — handled below
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        saw_partial,
        "gzippy never created an output file within 20s (or exited first: {:?})",
        child.try_wait()
    );

    // SIGINT via /bin/kill — no libc dev-dependency needed.
    let kill = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("run kill");
    assert!(kill.success(), "kill -INT failed");

    let status = child.wait().expect("wait gzippy");
    assert!(
        !status.success(),
        "gzippy must not report success after SIGINT (did it finish before the signal? \
         enlarge the input if this machine is that fast)"
    );
    // Probed: both gzip and gzippy die BY the signal (re-raise), not exit(1).
    assert_eq!(
        status.signal(),
        Some(2),
        "expected death by re-raised SIGINT"
    );

    // Give any cleanup-at-exit a moment, then pin the file state.
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !gz.exists(),
        "partial .gz left behind after SIGINT (gzip removes it; probed)"
    );
    assert!(
        input.exists(),
        "input must survive an interrupted compression"
    );
    assert_eq!(
        fs::read(&input).expect("read input").len(),
        payload.len(),
        "input content changed"
    );
}

// ---------------------------------------------------------------------------
// 4. Disk full
// ---------------------------------------------------------------------------

/// Detach guard so the ram disk is ejected even if an assertion panics.
struct RamDisk {
    dev: String,
    mount: PathBuf,
}

impl Drop for RamDisk {
    fn drop(&mut self) {
        let _ = Command::new("umount").arg(&self.mount).status();
        let _ = Command::new("hdiutil")
            .args(["detach", &self.dev, "-force"])
            .status();
        let _ = fs::remove_dir(&self.mount);
    }
}

/// Create a 2 MB HFS+ ram disk (macOS). Returns None (skip) if any tool is
/// missing or refuses — e.g. a locked-down CI runner.
fn tiny_ramdisk(mount: &Path) -> Option<RamDisk> {
    let attach = Command::new("hdiutil")
        .args(["attach", "-nomount", "ram://4096"]) // 4096 * 512 B = 2 MB
        .output()
        .ok()?;
    if !attach.status.success() {
        return None;
    }
    let dev = String::from_utf8_lossy(&attach.stdout)
        .split_whitespace()
        .next()?
        .to_string();
    let guard = RamDisk {
        dev: dev.clone(),
        mount: mount.to_path_buf(),
    };
    let mkfs = Command::new("newfs_hfs")
        .args(["-v", "gzippytiny", &dev])
        .status()
        .ok()?;
    if !mkfs.success() {
        return None; // guard detaches
    }
    fs::create_dir_all(mount).ok()?;
    let mnt = Command::new("mount")
        .args(["-t", "hfs", &dev, mount.to_str()?])
        .status()
        .ok()?;
    if !mnt.success() {
        return None; // guard detaches
    }
    Some(guard)
}

/// gzip contract (probed on a 2 MB HFS ram disk): ENOSPC mid-write =>
/// "No space left on device", exit 1, partial output REMOVED, input intact.
/// gzippy matches on exit code and file state at both default threads and
/// -p 1. Divergence (cosmetic, low severity): at default threads gzippy's
/// message is "IO error: write failed" and only the -p 1 path names ENOSPC.
///
/// macOS-only mechanics (hdiutil ram disk; needs no root here). On Linux a
/// loop device needs root, so this test skips cleanly there — a CI-linux
/// tmpfs variant would be the portable version if this ever proves flaky.
#[test]
fn disk_full_fails_cleanly_input_intact_no_partial_output() {
    if !cfg!(target_os = "macos") {
        eprintln!("SKIP: disk_full test uses an hdiutil ram disk (macOS only)");
        return;
    }
    let scratch = tempdir();
    let mount = scratch.path().join("tinyfs");
    let Some(_disk) = tiny_ramdisk(&mount) else {
        eprintln!("SKIP: could not set up 2 MB ram disk (locked-down environment?)");
        return;
    };

    // 1.2 MB of incompressible noise on a volume with ~1.9 MB free: the
    // output cannot fit next to the input.
    let payload = noise_bytes(1_200_000);
    let input = mount.join("rand.bin");
    fs::write(&input, &payload).expect("write input to ram disk");

    for threads in [&["-p", "1"][..], &[][..]] {
        let mut args = threads.to_vec();
        args.push("rand.bin");
        let out = run(&mount, &args);

        assert!(
            !out.status.success(),
            "ENOSPC must be an error (args {args:?}): {out:?}"
        );
        assert_eq!(
            out.status.code(),
            Some(1),
            "gzip exits 1 on ENOSPC (probed)"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.trim().is_empty(),
            "ENOSPC must be reported on stderr"
        );
        assert!(input.exists(), "input lost on ENOSPC (args {args:?})");
        assert_eq!(
            fs::read(&input).expect("read input"),
            payload,
            "input content changed on ENOSPC"
        );
        assert!(
            !mount.join("rand.bin.gz").exists(),
            "partial/zero-byte .gz left behind on ENOSPC (gzip removes it; probed); args {args:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Ecosystem decoder leg — Python's gzip module
// ---------------------------------------------------------------------------

/// Decode `gz` with Python's gzip module and return the payload bytes.
/// Python is a genuinely independent zlib-lineage decoder (CPython links the
/// real zlib but drives it through its own framing/member loop — the layer
/// where T>1 output differs most from single-stream output).
fn python_gunzip(gz: &Path) -> Vec<u8> {
    let out = Command::new("python3")
        .args([
            "-c",
            "import gzip,sys\nsys.stdout.buffer.write(gzip.decompress(open(sys.argv[1],'rb').read()))",
            gz.to_str().expect("utf8 path"),
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn python3");
    assert!(
        out.status.success(),
        "python3 gzip failed to decode {}: {}",
        gz.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Every (flavour x level {1,6,9} x threads {1,4}) output must decode
/// byte-exactly under Python's gzip module. Multi-member streams (the T>1
/// path emits them) are the interesting case: Python iterates members
/// itself rather than delegating to a C gzip reader.
#[test]
fn python_gzip_module_decodes_all_levels_and_thread_counts() {
    if !python3_available() {
        eprintln!("SKIP: python3 not on PATH");
        return;
    }
    let dir = tempdir();
    let flavours: [(&str, Vec<u8>); 3] = [
        ("text", text_bytes(4 * 1024 * 1024)),
        ("noise", noise_bytes(1024 * 1024)),
        ("csv", csv_bytes(2 * 1024 * 1024)),
    ];

    for (name, payload) in &flavours {
        for level in ["-1", "-6", "-9"] {
            for threads in ["1", "4"] {
                let fname = format!("{name}_{}_t{threads}.bin", &level[1..]);
                write_file(dir.path(), &fname, payload);
                let out = run(dir.path(), &[level, "-p", threads, &fname]);
                assert!(
                    out.status.success(),
                    "compress failed ({fname}): {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let gz = dir.path().join(format!("{fname}.gz"));
                let decoded = python_gunzip(&gz);
                assert_eq!(
                    &decoded, payload,
                    "python gzip decode mismatch: flavour={name} level={level} threads={threads}"
                );
                fs::remove_file(&gz).expect("cleanup .gz");
            }
        }
    }
}
