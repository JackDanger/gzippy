//! Property-based differential testing of the pure-Rust inflate path.
//!
//! WHY: proptest generates structurally-diverse DEFLATE streams —
//! fixed + dynamic Huffman, stored blocks, varied back-ref distances/lengths,
//! varied BFINAL placement, level-induced block-type mixes — far beyond what a
//! handful of fixed fixtures cover. For every generated input we assert gzippy's
//! decode is byte-identical to TWO independent oracles (flate2/zlib-ng and
//! libdeflate). A subtle break in the inner-Huffman decode (BMI2/multi-literal/
//! packed-LUT) or the back-ref resolution that escapes the fixed fixtures lands
//! RED here, with proptest shrinking to a minimal repro.
//!
//! Surface driven here: `decompress_bytes(.., 1)` — the single-threaded
//! production inflate primitive (gzip-wrapped input) — against the libdeflate
//! oracle (which is itself cross-checked against the original payload). The
//! window-absent marker `Block`/flip engine the copy-free refactor rewrites is
//! covered end-to-end by `seam_crossing` (the real production fold path through
//! the 32 KiB flip) and `diff_multi_oracle`; driving the raw `Block` from
//! offset 0 here would NOT exercise the flip faithfully (the standalone
//! Vec<u16> sink is not the production drain), so it is intentionally omitted.
//!
//! Inputs stay small (<= ~256 KiB) so each case is fast and shrinking is cheap;
//! the parallel-SM (>10 MiB) path is covered by the end-to-end nets
//! (`diff_multi_oracle`, `pure_rust_inflate_corpus`) and the seam net
//! (`seam_crossing`). Case count is bounded so the default suite stays quick.
//!
//! Run: `cargo test --release --features pure-rust-inflate inflate_proptest`.

#[cfg(test)]
#[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
mod tests {
    use std::io::Write;

    use proptest::prelude::*;

    // ── Oracles ──────────────────────────────────────────────────────────────

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

    // ── Strategies ────────────────────────────────────────────────────────────

    /// High-entropy bytes (forces stored / mostly-literal blocks).
    fn random_bytes() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(any::<u8>(), 0..200_000)
    }

    /// Structured payload: alternating literal runs and repeated motifs at
    /// varied distances/lengths → fixed + dynamic blocks with diverse back-refs.
    fn structured_payload() -> impl Strategy<Value = Vec<u8>> {
        let segment = prop_oneof![
            // Literal run of random bytes.
            prop::collection::vec(any::<u8>(), 1..4096),
            // A small alphabet repeated (long matches, short distances).
            (any::<u8>(), 1usize..2048usize).prop_map(|(b, n)| vec![b; n]),
            // A motif repeated at a chosen period (varied distances up to ~32 K).
            (prop::collection::vec(any::<u8>(), 1..4096), 1usize..40usize).prop_map(
                |(motif, reps)| {
                    let mut v = Vec::new();
                    for _ in 0..reps {
                        v.extend_from_slice(&motif);
                    }
                    v
                }
            ),
        ];
        prop::collection::vec(segment, 0..40).prop_map(|segs| {
            let mut out = Vec::new();
            for s in segs {
                out.extend_from_slice(&s);
            }
            out
        })
    }

    /// Long-period repeats → near-max-distance back-refs (32 K window edge).
    fn near_max_distance_payload() -> impl Strategy<Value = Vec<u8>> {
        (28_000usize..33_000usize, 2usize..12usize, any::<u64>()).prop_map(
            |(period, reps, seed)| {
                let mut base = vec![0u8; period];
                let mut rng = seed | 1;
                for x in &mut base {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *x = (rng >> 24) as u8;
                }
                let mut out = Vec::with_capacity(period * reps);
                for _ in 0..reps {
                    out.extend_from_slice(&base);
                }
                out
            },
        )
    }

    fn level() -> impl Strategy<Value = u32> {
        0u32..=9u32
    }

    fn assert_all_agree(payload: &[u8], level: u32) -> Result<(), TestCaseError> {
        let gz = gz_at(payload, level);
        let lib = oracle_libdeflate(&gz, payload.len());
        prop_assert_eq!(&lib, payload, "libdeflate oracle != original payload");
        let gp = gzippy_st(&gz);
        prop_assert_eq!(
            &gp,
            &lib,
            "gzippy single-thread decode != libdeflate oracle"
        );
        Ok(())
    }

    proptest! {
        // Bounded so the default suite stays quick; raise via PROPTEST_CASES.
        #![proptest_config(ProptestConfig {
            cases: 96,
            max_shrink_iters: 2_000,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_random_bytes_roundtrip(payload in random_bytes(), level in level()) {
            assert_all_agree(&payload, level)?;
        }

        #[test]
        fn prop_structured_roundtrip(payload in structured_payload(), level in level()) {
            assert_all_agree(&payload, level)?;
        }

        #[test]
        fn prop_near_max_distance_roundtrip(payload in near_max_distance_payload(), level in level()) {
            assert_all_agree(&payload, level)?;
        }
    }

    // ── Regression anchors (deterministic, always run) ───────────────────────
    // Minimal hand-picked shapes that proptest classes generalize; these run
    // even when proptest is configured to few cases, so the surfaces are always
    // touched in CI.

    #[test]
    fn anchor_empty_and_tiny() {
        for p in [vec![], vec![0u8], vec![0xff; 3], b"ab".to_vec()] {
            for lvl in [0u32, 1, 6, 9] {
                let gz = gz_at(&p, lvl);
                assert_eq!(oracle_libdeflate(&gz, p.len()), p);
                assert_eq!(gzippy_st(&gz), p);
            }
        }
    }

    #[test]
    fn anchor_all_block_types() {
        // Stored-only (L0), fixed-likely (tiny L1), dynamic (large L9).
        let stored = vec![0xa5u8; 100];
        let tiny = b"the the the the the the".to_vec();
        let mut big = Vec::new();
        let mut rng = 0x1234u64;
        for _ in 0..50_000 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            big.push((rng >> 24) as u8);
        }
        for (p, lvl) in [(stored, 0u32), (tiny, 1), (big, 9)] {
            let gz = gz_at(&p, lvl);
            assert_eq!(gzippy_st(&gz), oracle_libdeflate(&gz, p.len()));
        }
    }
}
