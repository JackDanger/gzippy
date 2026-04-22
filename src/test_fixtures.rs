//! Canonical in-memory test fixtures for regression tests.
//!
//! Generated deterministically from a seeded PRNG — same seed = same bytes
//! across machines and runs. Memoized via OnceLock so generation runs once
//! per test process, not once per test.
//!
//! Each fixture set provides:
//! - `plain`            — raw uncompressed bytes
//! - `single_member_gz` — flate2 default compression (single gzip member)
//! - `multi_member_gz`  — 256KB chunks, each a separate gzip member (pigz-style)
//! - `bgzf_gz`          — gzippy parallel format ("GZ" FEXTRA subfield)

#[cfg(test)]
pub use fixtures::*;

#[cfg(test)]
mod fixtures {
    use crate::parallel_compress::{compress_single_member, GzipHeaderInfo};
    use std::io::Write;
    use std::sync::OnceLock;

    pub struct Fixtures {
        pub plain: Vec<u8>,
        pub single_member_gz: Vec<u8>,
        pub multi_member_gz: Vec<u8>,
        pub bgzf_gz: Vec<u8>,
    }

    /// 1MB of mixed compressible text (seed 12345).
    pub fn text_1mb() -> &'static Fixtures {
        static CELL: OnceLock<Fixtures> = OnceLock::new();
        CELL.get_or_init(|| build(1024 * 1024, 12345))
    }

    /// 10MB of mixed compressible text (seed 67890).
    pub fn text_10mb() -> &'static Fixtures {
        static CELL: OnceLock<Fixtures> = OnceLock::new();
        CELL.get_or_init(|| build(10 * 1024 * 1024, 67890))
    }

    /// 1MB of near-random binary (seed 99999) — tests incompressible path.
    pub fn binary_1mb() -> &'static Fixtures {
        static CELL: OnceLock<Fixtures> = OnceLock::new();
        CELL.get_or_init(|| build_random(1024 * 1024, 99999))
    }

    // ── generators ───────────────────────────────────────────────────────────

    /// Mixed compressible text — 70% English-like phrases, 30% random bytes.
    fn make_mixed(size: usize, seed: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng = seed;
        let phrases: &[&[u8]] = &[
            b"the quick brown fox jumps over the lazy dog. ",
            b"pack my box with five dozen liquor jugs! ",
            b"0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOP\n",
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
            b"fn decompress(data: &[u8]) -> Result<Vec<u8>, Error> {\n",
        ];
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if (rng >> 32) % 10 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let phrase = phrases[((rng >> 24) as usize) % phrases.len()];
                let remaining = size - data.len();
                data.extend_from_slice(&phrase[..remaining.min(phrase.len())]);
            }
        }
        data.truncate(size);
        data
    }

    /// Near-random binary — exercises incompressible path.
    fn make_random(size: usize, seed: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng = seed;
        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 32) as u8);
        }
        data
    }

    // ── compressors ──────────────────────────────────────────────────────────

    fn compress_single_member_gz(plain: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(plain).unwrap();
        enc.finish().unwrap()
    }

    fn compress_multi_member_gz(plain: &[u8]) -> Vec<u8> {
        let chunk = 256 * 1024;
        let mut out = Vec::new();
        for c in plain.chunks(chunk) {
            let mut enc =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(c).unwrap();
            out.extend_from_slice(&enc.finish().unwrap());
        }
        out
    }

    fn compress_bgzf_gz(plain: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let header = GzipHeaderInfo::default();
        compress_single_member(&mut out, plain, 1, &header).unwrap();
        out
    }

    // ── builders ─────────────────────────────────────────────────────────────

    fn build(size: usize, seed: u64) -> Fixtures {
        let plain = make_mixed(size, seed);
        let single_member_gz = compress_single_member_gz(&plain);
        let multi_member_gz = compress_multi_member_gz(&plain);
        let bgzf_gz = compress_bgzf_gz(&plain);
        Fixtures { plain, single_member_gz, multi_member_gz, bgzf_gz }
    }

    fn build_random(size: usize, seed: u64) -> Fixtures {
        let plain = make_random(size, seed);
        let single_member_gz = compress_single_member_gz(&plain);
        let multi_member_gz = compress_multi_member_gz(&plain);
        let bgzf_gz = compress_bgzf_gz(&plain);
        Fixtures { plain, single_member_gz, multi_member_gz, bgzf_gz }
    }

    // ── self-test ────────────────────────────────────────────────────────────

    /// Verify fixtures are deterministic — catches if flate2 version changes output.
    #[test]
    fn fixture_determinism() {
        let a = text_1mb();
        let b = build(1024 * 1024, 12345); // rebuild independently
        assert_eq!(a.plain, b.plain, "plain data not deterministic");
        assert_eq!(
            a.single_member_gz, b.single_member_gz,
            "single_member_gz not deterministic — flate2 version may have changed"
        );
    }

    /// Fixtures decompress to the original plain bytes.
    #[test]
    fn fixture_roundtrip() {
        use flate2::read::MultiGzDecoder;
        use std::io::Read;
        let f = text_1mb();
        let mut dec = MultiGzDecoder::new(f.single_member_gz.as_slice());
        let mut recovered = Vec::new();
        dec.read_to_end(&mut recovered).unwrap();
        assert_eq!(recovered, f.plain, "single_member_gz roundtrip failed");
    }
}
