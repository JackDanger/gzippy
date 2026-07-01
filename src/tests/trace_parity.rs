//! Trace-parity correctness gate — all gzip-family archive formats.
//!
//! Every production route must yield byte-identical output at T=1,2,4,8
//! (and T=16 for large single-member on x86_64). Run via:
//!   `cargo test --release trace_parity`
//! or `scripts/trace_parity_check.sh`.

#[cfg(test)]
mod tests {
    use crate::compress::parallel::{compress_single_member, GzipHeaderInfo, ParallelGzEncoder};
    use std::io::Write;
    use std::process::Command;

    const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

    fn make_payload(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeefcafebabe;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 32) as u8);
        }
        data
    }

    fn decompress_via_router(gz: &[u8], threads: usize) -> Vec<u8> {
        let mut out = Vec::new();
        crate::decompress::decompress_gzip_libdeflate(gz, &mut out, threads).unwrap();
        out
    }

    struct Fixtures {
        original: Vec<u8>,
        single_flate2: Vec<u8>,
        single_gzip_cli: Option<Vec<u8>>,
        multi: Vec<u8>,
        bgzf: Vec<u8>,
        gz_subfield: Vec<u8>,
    }

    impl Fixtures {
        fn new(payload_size: usize) -> Self {
            let original = make_payload(payload_size);

            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(&original).unwrap();
            let single_flate2 = enc.finish().unwrap();

            let single_gzip_cli = compress_via_gzip_cli(&original);

            let mut multi = Vec::new();
            let parts = 8usize;
            let chunk_sz = original.len().div_ceil(parts).max(1);
            for chunk in original.chunks(chunk_sz) {
                let mut e =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                e.write_all(chunk).unwrap();
                multi.extend_from_slice(&e.finish().unwrap());
            }

            let mut bgzf = Vec::new();
            let header = GzipHeaderInfo::default();
            compress_single_member(&mut bgzf, &original, 1, &header).unwrap();

            let mut gz_subfield = Vec::new();
            let par = ParallelGzEncoder::new(1, 4);
            par.compress_buffer(&original, &mut gz_subfield).unwrap();

            Self {
                original,
                single_flate2,
                single_gzip_cli,
                multi,
                bgzf,
                gz_subfield,
            }
        }
    }

    fn compress_via_gzip_cli(raw: &[u8]) -> Option<Vec<u8>> {
        let input =
            std::env::temp_dir().join(format!("gzippy_trace_parity_{}.in", std::process::id()));
        std::fs::write(&input, raw).ok()?;
        let status = Command::new("gzip")
            .args(["-9", "-c", "-n"])
            .arg(&input)
            .output()
            .ok()?;
        let _ = std::fs::remove_file(&input);
        if !status.status.success() {
            return None;
        }
        Some(status.stdout)
    }

    fn assert_thread_independent(label: &str, gz: &[u8], expected: &[u8], threads: &[usize]) {
        let ref_out = decompress_via_router(gz, 1);
        assert_eq!(ref_out, expected, "{label}: T=1 mismatch");
        let ref_hash = stable_digest(&ref_out);
        for &t in threads {
            if t == 1 {
                continue;
            }
            let out = decompress_via_router(gz, t);
            assert_eq!(
                stable_digest(&out),
                ref_hash,
                "{label}: T={t} digest differs from T=1 (output len {} vs {})",
                out.len(),
                ref_out.len()
            );
            assert_eq!(out, expected, "{label}: T={t} byte mismatch");
        }
    }

    fn stable_digest(data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        data.hash(&mut h);
        h.finish()
    }

    #[test]
    fn trace_parity_bgzf_all_threads() {
        let fx = Fixtures::new(2 * 1024 * 1024);
        assert_thread_independent("bgzf", &fx.bgzf, &fx.original, THREAD_COUNTS);
    }

    #[test]
    fn trace_parity_multi_member_all_threads() {
        let fx = Fixtures::new(2 * 1024 * 1024);
        assert_thread_independent("multi-member", &fx.multi, &fx.original, THREAD_COUNTS);
    }

    #[test]
    fn trace_parity_gz_subfield_all_threads() {
        let fx = Fixtures::new(2 * 1024 * 1024);
        assert_thread_independent("gz-subfield", &fx.gz_subfield, &fx.original, THREAD_COUNTS);
    }

    #[test]
    fn trace_parity_single_member_flate2_all_threads() {
        let fx = Fixtures::new(2 * 1024 * 1024);
        assert_thread_independent(
            "single-member-flate2",
            &fx.single_flate2,
            &fx.original,
            THREAD_COUNTS,
        );
    }

    /// Large single-member input clears the 10 MiB parallel gate on x86_64 + ISA-L.
    #[test]
    fn trace_parity_single_member_large_all_threads() {
        let original = make_payload(24 * 1024 * 1024);
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&original).unwrap();
        let gz = enc.finish().unwrap();
        assert!(
            gz.len() > 10 * 1024 * 1024,
            "fixture must exceed parallel gate (got {} bytes)",
            gz.len()
        );
        let threads = &[1usize, 2, 4, 8, 16];
        assert_thread_independent("single-member-large", &gz, &original, threads);
    }

    /// Real `gzip(1)` layout (FNAME, pigz-like blocks). Skipped if gzip missing.
    #[test]
    fn trace_parity_single_member_gzip_cli_all_threads() {
        let fx = Fixtures::new(24 * 1024 * 1024);
        let Some(gz) = fx.single_gzip_cli else {
            eprintln!("trace_parity: gzip(1) not on PATH — skip gzip-cli fixture");
            return;
        };
        assert!(
            gz.len() > 10 * 1024 * 1024,
            "gzip-cli fixture must exceed parallel gate"
        );
        let threads = &[1usize, 2, 4, 8, 16];
        assert_thread_independent("single-member-gzip-cli", &gz, &fx.original, threads);
    }
}
