//! Thread-count structure parity for the production encoder.
//!
//! Thread count is a throughput request, not permission to change the
//! whole-stream gzip structure; the parallel entry point must therefore emit
//! the same member as T1 on real file inputs.

use gzippy::fixtures;
use std::process::Command;

const LEVELS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

fn compress(bin: &str, path: &std::path::Path, level: u32, threads: u32) -> Vec<u8> {
    let output = Command::new(bin)
        .args([
            &format!("-{level}"),
            "-p",
            &threads.to_string(),
            "-c",
            path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "gzippy failed on {} L{level} T{threads}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn file_compression_is_byte_identical_across_thread_counts() {
    let bin = env!("CARGO_BIN_EXE_gzippy");
    let dir = tempfile::tempdir().unwrap();

    for &name in fixtures::NAMES {
        let data = fixtures::generate(name);
        let path = dir.path().join(name);
        std::fs::write(&path, data).unwrap();

        for &level in LEVELS {
            let t1 = compress(bin, &path, level, 1);
            for threads in [2, 4, 8, 16] {
                assert_eq!(
                    compress(bin, &path, level, threads),
                    t1,
                    "{name} L{level} T{threads}: thread count changed libdeflate structure"
                );
            }
        }
    }
}
