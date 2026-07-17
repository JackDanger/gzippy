//! Deterministic seeded fuzz loop over the pure-Rust inflate entry points,
//! differential against independent oracles.
//!
//! WHY: cargo-fuzz / libFuzzer is not wired into this repo (no `fuzz/` crate,
//! and the bench host blocks the sanitizer runtime), so this is the in-tree
//! substitute: a self-contained, seeded fuzz loop that generates many random
//! payloads, compresses them through flate2 at random levels, and asserts
//! gzippy's decode matches a libdeflate oracle byte-for-byte. It is the standing
//! wide-input net for the inflate path.
//!
//! Determinism: the RNG is seeded from `GZIPPY_FUZZ_SEED` (default fixed), so a
//! failure reproduces exactly; the failing seed + iteration is printed. The
//! iteration budget is `GZIPPY_FUZZ_ITERS` (default tuned for CI-time).
//!
//! Gating: `#[ignore]` so the default `cargo test` stays fast. Run explicitly:
//!   `cargo test --release --features pure-rust-inflate fuzz_loop -- --ignored --nocapture`
//! CI runs it with a bounded `GZIPPY_FUZZ_ITERS`.

#[cfg(test)]
#[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
mod tests {
    use std::io::Write;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64* — deterministic, fast, no deps.
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            if n == 0 {
                0
            } else {
                (self.next() % n as u64) as usize
            }
        }
    }

    fn oracle_libdeflate(gz: &[u8], exact: usize) -> Vec<u8> {
        let mut out = vec![0u8; exact.max(1)];
        let mut decoder = crate::backends::libdeflate::DecompressorEx::new();
        let r = decoder
            .gzip_decompress_ex(gz, &mut out)
            .expect("libdeflate oracle");
        out.truncate(r.output_size);
        out
    }

    fn gzippy_st(gz: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        crate::decompress::decompress_bytes(gz, &mut out, 1).expect("gzippy single-thread decode");
        out
    }

    fn gz_at(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// Generate one random payload from `rng` with a shape chosen at random:
    /// random literals, low-alphabet repeats, motif-at-distance, or a mix.
    fn gen_payload(rng: &mut Rng) -> Vec<u8> {
        let target = rng.below(256 * 1024);
        let mut out = Vec::with_capacity(target);
        while out.len() < target {
            match rng.below(5) {
                0 => {
                    // Random literal burst.
                    let n = 1 + rng.below(4096);
                    for _ in 0..n {
                        out.push((rng.next() >> 24) as u8);
                    }
                }
                1 => {
                    // Single-byte run (long matches, distance 1).
                    let b = (rng.next() >> 24) as u8;
                    let n = 1 + rng.below(2048);
                    out.extend(std::iter::repeat_n(b, n));
                }
                2 => {
                    // Small motif repeated at its own period (varied distance).
                    let mlen = 1 + rng.below(4096);
                    let motif: Vec<u8> = (0..mlen).map(|_| (rng.next() >> 24) as u8).collect();
                    let reps = 1 + rng.below(16);
                    for _ in 0..reps {
                        out.extend_from_slice(&motif);
                    }
                }
                3 => {
                    // Near-max-distance: a ~32 K base repeated.
                    let period = 28_000 + rng.below(5000);
                    let base: Vec<u8> = (0..period).map(|_| (rng.next() >> 24) as u8).collect();
                    let reps = 1 + rng.below(4);
                    for _ in 0..reps {
                        out.extend_from_slice(&base);
                    }
                }
                _ => {
                    // Repeat a slice of what we already have (self-referential).
                    if !out.is_empty() {
                        let start = rng.below(out.len());
                        let len = 1 + rng.below(out.len() - start);
                        let slice = out[start..start + len].to_vec();
                        out.extend_from_slice(&slice);
                    } else {
                        out.push((rng.next() >> 24) as u8);
                    }
                }
            }
        }
        out.truncate(target);
        out
    }

    fn run_fuzz(seed: u64, iters: usize) {
        let mut rng = Rng(seed | 1);
        for i in 0..iters {
            let payload = gen_payload(&mut rng);
            let level = (rng.below(10)) as u32;
            let gz = gz_at(&payload, level);

            // Oracle must agree with the original first (fixture sanity).
            let oracle = oracle_libdeflate(&gz, payload.len());
            if oracle != payload {
                panic!(
                    "FUZZ FIXTURE BUG (seed={seed} iter={i} level={level} len={}): \
                     libdeflate oracle disagrees with original payload — recompress mismatch",
                    payload.len()
                );
            }

            let got = gzippy_st(&gz);
            if got != oracle {
                let d = got
                    .iter()
                    .zip(oracle.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(oracle.len()));
                panic!(
                    "FUZZ DIVERGENCE (seed={seed} iter={i} level={level} \
                     payload_len={} gz_len={}): gzippy != oracle at byte {d} \
                     (got=0x{:02x} oracle=0x{:02x}); reproduce with \
                     GZIPPY_FUZZ_SEED={seed} GZIPPY_FUZZ_ITERS={}",
                    payload.len(),
                    gz.len(),
                    got.get(d).copied().unwrap_or(0),
                    oracle.get(d).copied().unwrap_or(0),
                    i + 1
                );
            }
        }
        eprintln!("[fuzz] seed={seed} completed {iters} iterations, all byte-exact");
    }

    /// Default fuzz run. `#[ignore]` so the default suite stays fast.
    /// Env: GZIPPY_FUZZ_SEED (default 0x9E3779B97F4A7C15), GZIPPY_FUZZ_ITERS
    /// (default 300 — a few seconds; CI raises it).
    #[test]
    #[ignore = "fuzz loop — run explicitly: cargo test ... fuzz_loop -- --ignored"]
    fn fuzz_loop_differential() {
        let seed = std::env::var("GZIPPY_FUZZ_SEED")
            .ok()
            .and_then(|s| {
                s.strip_prefix("0x")
                    .map(|h| u64::from_str_radix(h, 16))
                    .unwrap_or_else(|| s.parse::<u64>())
                    .ok()
            })
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        let iters = std::env::var("GZIPPY_FUZZ_ITERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(300);
        run_fuzz(seed, iters);
    }

    /// A small, NON-ignored smoke fuzz (fixed seed, few iterations) so the
    /// fuzz generator + differential path is exercised in the default suite —
    /// guards against the fuzz harness silently bit-rotting.
    #[test]
    fn fuzz_smoke_default_suite() {
        run_fuzz(0xD1CE_F00D_1234_5678, 24);
    }
}
