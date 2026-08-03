//! Roundtrip matrix for the T>1 bit-splice writer (the "delete the seam"
//! lever): every frozen fixture x levels {1,2,6,9} x -p{1,2,4,8}, decoded
//! byte-exact through OUR OWN decoder (`gzippy -dc`) and through system
//! `gzip -dc`. The bit-splicer rewrites every T>1 chunk boundary — verbatim
//! writes, bit-shifted fragments, and the stored-block alignment-seam
//! fallback (the `noise`/`binary` fixtures exercise it constantly) — so the
//! whole grid must hold VALID GZIP, the only T>1 correctness bar.
//!
//! Everything runs against the REAL binary (CARGO_BIN_EXE_gzippy), like the
//! fingerprint suite: the shipped quantity, not a library shortcut.

use gzippy::fixtures;
use std::process::Command;

fn run(bin: &str, args: &[&str]) -> Vec<u8> {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin}: {e}"));
    assert!(
        out.status.success(),
        "{bin} {args:?} exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
fn fixtures_roundtrip_across_thread_counts() {
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();
    let has_gzip = Command::new("gzip").arg("--version").output().is_ok();
    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let input = dir.path().join(name);
        std::fs::write(&input, &data).unwrap();
        for level in [1u32, 2, 6, 9] {
            for threads in [1u32, 2, 4, 8] {
                let gz = run(
                    bin,
                    &[
                        &format!("-{level}"),
                        "-p",
                        &threads.to_string(),
                        "-c",
                        input.to_str().unwrap(),
                    ],
                );
                let gz_path = dir.path().join(format!("{name}-L{level}-p{threads}.gz"));
                std::fs::write(&gz_path, &gz).unwrap();
                let ctx = format!("{name} L{level} p{threads}");

                // Our own decoder.
                let ours = run(bin, &["-dc", gz_path.to_str().unwrap()]);
                assert_eq!(ours, data, "gzippy -dc mismatch: {ctx}");

                // An independent decoder.
                if has_gzip {
                    let sys = run("gzip", &["-dc", gz_path.to_str().unwrap()]);
                    assert_eq!(sys, data, "gzip -dc mismatch: {ctx}");
                }
            }
        }
    }
}
