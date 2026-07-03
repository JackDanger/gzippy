//! Selector byte-transparency differential + proptest.
//!
//! WHY (campaign): `effective_parallel_threads`
//! (`src/decompress/parallel/single_member.rs`) is a PURE thread-count selector
//! with FIVE decision dimensions —
//!   1. the hard ratio cap (`GZIPPY_PARALLEL_RATIO_MAX`, default 8),
//!   2. the serial-clean crossover margin (`GZIPPY_PARALLEL_CROSSOVER_MARGIN`),
//!   3. the small-output serial FLOOR (`GZIPPY_PARALLEL_MIN_OUTPUT_BYTES`, 8 MiB),
//!   4. the large-output notch bonus (`GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES/NOTCH`),
//!   5. the small-file chunk-size adjustment (min-work-per-thread,
//!      `adjusted_chunk_size_bytes`).
//! Each is BYTE-TRANSPARENT by construction — it may only change the thread
//! count / chunk granularity, never the decoded bytes. But the existing
//! multi-oracle net (`diff_multi_oracle`) only decodes at a FIXED T=4 on
//! >=60 MiB fixtures, so the <8 MiB FLOOR path — the exact case where the
//! selector CHANGES its answer (parallel→serial) — has ZERO oracle coverage.
//!
//! This net closes that gap. For gzip streams spanning the selector's decision
//! BOUNDARIES (ISIZE/deflate ratio × output size, including exact boundary
//! ISIZE values), it decodes each through THREE configurations —
//!   (a) the production selector (default env),
//!   (b) forced-parallel (all caps disabled),
//!   (c) forced-serial (crossover forced above every T) —
//! and asserts ALL THREE are byte-identical to TWO independent oracles
//! (flate2/zlib-ng + libdeflate). If any selector branch corrupted a byte
//! (e.g. a serial/parallel seam divergence that only manifests under a
//! particular thread count) it lands RED here.
//!
//! Gate-0 non-inertness: the boundary suite PROVES it exercises the <8 MiB
//! floor path (asserts `SMALL_OUTPUT_SERIAL_FLOOR_APPLIED` increments across a
//! real production decode) plus the ratio-cap, crossover, notch, and
//! chunk-adjust branches (each via its dedicated deletion-trap counter).
//!
//! Run: `cargo test --release --features pure-rust-inflate selector_byte_transparency`.

#[cfg(test)]
#[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
mod tests {
    use std::io::{Read, Write};
    use std::sync::atomic::Ordering;

    use proptest::prelude::*;

    use crate::decompress::parallel::single_member::{
        effective_parallel_threads, skip_gzip_header, ADJUSTED_CHUNK_SIZE_APPLIED,
        COMPRESSIBILITY_THREAD_CAP_APPLIED, MARKER_PIPELINE_RUNS, MARKER_PIPELINE_TEST_LOCK,
        SERIAL_CLEAN_FLOOR_APPLIED, SMALL_OUTPUT_SERIAL_FLOOR_APPLIED,
    };

    // ── Selector env knobs (the five dimensions) ──────────────────────────────
    const E_RATIO_MAX: &str = "GZIPPY_PARALLEL_RATIO_MAX";
    const E_MARGIN: &str = "GZIPPY_PARALLEL_CROSSOVER_MARGIN";
    const E_MIN_OUT: &str = "GZIPPY_PARALLEL_MIN_OUTPUT_BYTES";
    const E_LARGE_OUT: &str = "GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES";
    const E_NOTCH: &str = "GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH";

    const ALL_SELECTOR_ENV: &[&str] = &[E_RATIO_MAX, E_MARGIN, E_MIN_OUT, E_LARGE_OUT, E_NOTCH];

    fn clear_selector_env() {
        for k in ALL_SELECTOR_ENV {
            std::env::remove_var(k);
        }
    }

    // ── Oracles (two independent codebases; mirror of diff_multi_oracle) ───────

    /// Oracle 1: flate2 (zlib-ng). MultiGzDecoder tolerates single-member.
    fn oracle_flate2(gz: &[u8]) -> Vec<u8> {
        let mut decoder = flate2::read::MultiGzDecoder::new(gz);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).expect("flate2 oracle decode");
        out
    }

    /// Oracle 2: libdeflate one-shot FFI. Independent author/codebase.
    fn oracle_libdeflate(gz: &[u8], exact_size: usize) -> Vec<u8> {
        let mut out = vec![0u8; exact_size.max(1)];
        let mut decoder = crate::backends::libdeflate::DecompressorEx::new();
        let r = decoder
            .gzip_decompress_ex(gz, &mut out)
            .expect("libdeflate oracle decode");
        out.truncate(r.output_size);
        out
    }

    // ── Deterministic corpus generator with controllable ratio + exact size ───

    /// Build a payload of EXACTLY `len` bytes whose gzip ISIZE/deflate ratio
    /// lands near `target_ratio`. Interleaves incompressible PRNG blocks (ratio
    /// ~1) with highly-compressible motif blocks (ratio ~1000). Block 0 is
    /// always compressible so the FIRST deflate block is Huffman (not stored) —
    /// this pins routing to ParallelSM (the path that runs the selector) rather
    /// than the stored-block splitter. `len` becomes the gzip ISIZE exactly
    /// (payload < 4 GiB), so exact-boundary ISIZE values are reachable.
    fn gen_payload(len: usize, target_ratio: f64, seed: u64) -> Vec<u8> {
        const BLK: usize = 4096;
        let blocks = len.div_ceil(BLK).max(1);
        // compressed ≈ (#prng blocks)·BLK ⇒ ratio ≈ len / (n_prng·BLK).
        let want_prng = if target_ratio <= 1.0 {
            blocks.saturating_sub(1)
        } else {
            ((len as f64) / (target_ratio * BLK as f64)).round() as usize
        };
        // Reserve block 0 as compressible ⇒ at most blocks-1 PRNG blocks.
        let n_prng = want_prng.min(blocks.saturating_sub(1));

        // Deterministically choose which blocks (1..blocks) are PRNG. Spread
        // them by a fixed stride so ratio is stable across sizes.
        let mut is_prng = vec![false; blocks];
        if n_prng > 0 && blocks > 1 {
            let mut placed = 0usize;
            let mut idx = 1usize;
            // Fractional stride placement (deterministic, evenly spread).
            let mut acc = 0f64;
            let rate = n_prng as f64 / (blocks - 1) as f64;
            while idx < blocks && placed < n_prng {
                acc += rate;
                if acc >= 1.0 {
                    is_prng[idx] = true;
                    placed += 1;
                    acc -= 1.0;
                }
                idx += 1;
            }
            // Top up any remainder from rounding.
            let mut j = 1usize;
            while placed < n_prng && j < blocks {
                if !is_prng[j] {
                    is_prng[j] = true;
                    placed += 1;
                }
                j += 1;
            }
        }

        let mut out = Vec::with_capacity(len);
        let mut rng: u64 = seed | 1;
        for (bi, prng) in is_prng.iter().enumerate() {
            if out.len() >= len {
                break;
            }
            if *prng {
                for _ in 0..BLK {
                    if out.len() >= len {
                        break;
                    }
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    out.push((rng >> 24) as u8);
                }
            } else {
                // Per-block 16-byte motif repeated to fill the block: compresses
                // hard, and differs per block so the stream is genuinely dynamic.
                let mut motif = [0u8; 16];
                for (j, b) in motif.iter_mut().enumerate() {
                    *b = ((bi as u64).wrapping_mul(131).wrapping_add(j as u64) & 0xff) as u8;
                }
                for k in 0..BLK {
                    if out.len() >= len {
                        break;
                    }
                    out.push(motif[k % motif.len()]);
                }
            }
        }
        // Pad/truncate to EXACT len.
        while out.len() < len {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            out.push((rng >> 24) as u8);
        }
        out.truncate(len);
        out
    }

    fn gz_at(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// deflate-payload length as the selector sees it: gz minus header minus the
    /// 8-byte trailer (mirror of `decompress_parallel`'s `_deflate_data_len`).
    fn deflate_len(gz: &[u8]) -> usize {
        let hdr = skip_gzip_header(gz).expect("valid gzip header");
        gz.len().saturating_sub(hdr + 8)
    }

    fn isize_of(gz: &[u8]) -> u64 {
        let n = gz.len();
        u32::from_le_bytes([gz[n - 4], gz[n - 3], gz[n - 2], gz[n - 1]]) as u64
    }

    // ── The core differential: three configs, two oracles, byte-exact ─────────

    struct Cfg {
        label: &'static str,
        env: &'static [(&'static str, &'static str)],
    }

    /// (a) production, (b) forced-parallel (all caps off), (c) forced-serial
    /// (crossover forced above every thread count so the selector returns 1
    /// even at T=16). All three MUST decode byte-identically.
    const CONFIGS: &[Cfg] = &[
        Cfg {
            label: "production",
            env: &[],
        },
        Cfg {
            label: "forced-parallel",
            env: &[(E_RATIO_MAX, "0"), (E_MARGIN, "0"), (E_MIN_OUT, "0")],
        },
        Cfg {
            label: "forced-serial",
            // Huge crossover margin ⇒ crossover = ceil(ratio · 1e6) > any T ⇒
            // serial even for near-incompressible streams; ratio_max=1 and a huge
            // floor are belt-and-suspenders.
            env: &[
                (E_MARGIN, "1000000"),
                (E_RATIO_MAX, "1"),
                (E_MIN_OUT, "17179869184"),
            ],
        },
    ];

    /// Decode `gz` at thread count `t` under `env`, returning the bytes.
    /// Caller holds `MARKER_PIPELINE_TEST_LOCK` (serializes the process-global
    /// env + the shared MARKER_PIPELINE_RUNS snapshot).
    fn decode_under(gz: &[u8], t: usize, env: &[(&str, &str)], reserve: usize) -> Vec<u8> {
        clear_selector_env();
        for (k, v) in env {
            std::env::set_var(k, v);
        }
        let mut out = Vec::with_capacity(reserve);
        let r = crate::decompress::decompress_single_member(gz, &mut out, t);
        clear_selector_env();
        r.unwrap_or_else(|e| panic!("decode failed (t={t}): {e}"));
        out
    }

    /// Assert byte-transparency of ALL configs at all `threads`, vs both oracles.
    /// Returns nothing; panics LOUDLY on any divergence (correctness bug).
    fn assert_transparent(gz: &[u8], threads: &[usize], label: &str) {
        let exact = isize_of(gz) as usize;
        let ref_flate2 = oracle_flate2(gz);
        let ref_libdeflate = oracle_libdeflate(gz, exact);
        assert_eq!(
            ref_flate2, ref_libdeflate,
            "{label}: ORACLE DISAGREEMENT flate2 vs libdeflate — fixture suspect"
        );
        assert_eq!(
            ref_flate2.len(),
            exact,
            "{label}: ISIZE trailer ({exact}) != decoded len ({})",
            ref_flate2.len()
        );
        let reference = ref_flate2;

        let _lock = MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        for &t in threads {
            for cfg in CONFIGS {
                let before = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
                let got = decode_under(gz, t, cfg.env, reference.len());
                let after = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
                assert!(
                    after > before,
                    "{label}[{},t={t}]: ParallelSM did not run (MARKER_PIPELINE_RUNS \
                     {before}->{after}) — routing fell through, selector NOT exercised",
                    cfg.label
                );
                assert_eq!(
                    got.len(),
                    reference.len(),
                    "{label}[{},t={t}]: length {} != reference {}",
                    cfg.label,
                    got.len(),
                    reference.len()
                );
                if got != reference {
                    let d = got
                        .iter()
                        .zip(reference.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(got.len().min(reference.len()));
                    panic!(
                        "CORRECTNESS BUG {label}[{},t={t}]: BYTE DIVERGENCE at offset {d} \
                         (got=0x{:02x} ref=0x{:02x}) — selector is NOT byte-transparent",
                        cfg.label,
                        got.get(d).copied().unwrap_or(0),
                        reference.get(d).copied().unwrap_or(0)
                    );
                }
            }
        }
    }

    const MIB: usize = 1024 * 1024;

    // ── FIXED BOUNDARY TABLE ──────────────────────────────────────────────────
    //
    // Spans the selector's decision boundaries. Output size × ISIZE ratio,
    // INCLUDING exact boundary ISIZE values (8 MiB floor: below / exact / above).
    // Each fixture is decoded at T ∈ {1,4,8,16} × 3 configs × 2 oracles.
    #[test]
    fn boundary_table_byte_transparent() {
        let threads = [1usize, 4, 8, 16];
        // (label, output_len, target_ratio, gzip_level)
        let table: &[(&str, usize, f64, u32)] = &[
            // --- 8 MiB small-output FLOOR boundary (exact) ---
            ("floor_below", 8 * MIB - 1, 3.0, 6), // isize < 8 MiB ⇒ floor fires
            ("floor_exact", 8 * MIB, 3.0, 6),     // isize == 8 MiB ⇒ floor does NOT fire
            ("floor_above", 8 * MIB + 4096, 3.0, 6), // isize  > 8 MiB ⇒ floor does NOT fire
            // --- small outputs below the floor ---
            ("small_r3", 2 * MIB, 3.0, 6),
            ("small_r5", 5 * MIB, 5.0, 6),
            ("small_lowratio", MIB, 1.2, 6), // ratio < 1.5 ⇒ floor must NOT fire
            // --- mid outputs above floor, in the crossover zone ---
            ("mid_r3", 12 * MIB, 3.0, 6),
            ("mid_lowratio", 12 * MIB, 1.2, 6),
            // --- ratio-cap boundary (default cap = 8×), output > floor ---
            ("cap_below", 12 * MIB, 7.4, 6), // ratio just below 8 ⇒ cap must NOT fire
            ("cap_above", 12 * MIB, 8.6, 6), // ratio just above 8 ⇒ cap fires
            ("ratio_15", 12 * MIB, 15.0, 6),
            ("ratio_25", 12 * MIB, 25.0, 6),
            // --- a larger output ---
            ("large_r3", 20 * MIB, 3.0, 6),
        ];

        for (label, len, ratio, level) in table {
            let p = gen_payload(*len, *ratio, 0xC0FFEE ^ (*len as u64));
            let gz = gz_at(&p, *level);
            let measured = isize_of(&gz) as f64 / deflate_len(&gz).max(1) as f64;
            eprintln!(
                "boundary {label}: out={} deflate={} measured_ratio={:.2} (target {:.1})",
                p.len(),
                deflate_len(&gz),
                measured,
                ratio
            );
            assert_transparent(&gz, &threads, label);
        }
    }

    /// 64-byte blob whose last 4 bytes are the gzip-trailer ISIZE the selector
    /// reads. `effective_parallel_threads` only reads those 4 bytes plus the
    /// passed `deflate_len`, so this drives the DIRECT selector DECISION at
    /// exact (isize, deflate) points — independent of any real compression
    /// outcome, hence deterministic and arch-independent.
    fn blob_with_isize(isize_val: u32) -> Vec<u8> {
        let mut v = vec![0u8; 64];
        v[60..64].copy_from_slice(&isize_val.to_le_bytes());
        v
    }

    // ── COVERAGE / NON-INERT PROOF ────────────────────────────────────────────
    //
    // Deterministically proves each selector branch is EXERCISED (its dedicated
    // deletion-trap counter increments / its decision flips), AND that the <8 MiB
    // floor + chunk-adjust paths are hit through REAL production decodes. The
    // branch DECISIONS use synthetic exact-(isize,deflate) blobs so they are
    // deterministic on x86 AND aarch64 (env overrides pin the arch-dispatched
    // defaults); the non-inert proofs use real compressible streams whose
    // floor/chunk-adjust preconditions hold with wide margin regardless of the
    // exact realized ratio.
    #[test]
    fn selector_branches_are_exercised() {
        let _lock = MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        // ============================================================
        // 1. <8 MiB SMALL-OUTPUT FLOOR — direct decision + REAL decode.
        // ============================================================
        clear_selector_env(); // default env: the floor is arch-INDEPENDENT
        let before_floor = SMALL_OUTPUT_SERIAL_FLOOR_APPLIED.load(Ordering::Relaxed);

        // Direct: isize 6 MiB, deflate 2 MiB (ratio 3 ≥ 1.5, both < 8 MiB) ⇒ floor
        // fires ⇒ serial at every T>1.
        let floor_blob = blob_with_isize(6 * MIB as u32);
        assert_eq!(
            effective_parallel_threads(&floor_blob, 2 * MIB, 8),
            1,
            "floor: <8 MiB compressible must cap to serial (default env)"
        );

        // REAL production decode of a 6 MiB compressible stream at T=8 must ALSO
        // trip the floor counter — this is the proof the test is NON-INERT on the
        // <8 MiB floor path (the exact path diff_multi_oracle never reaches).
        let floor_p = gen_payload(6 * MIB, 3.0, 0xF100_0001);
        let floor_gz = gz_at(&floor_p, 6);
        assert!(
            isize_of(&floor_gz) < 8 * MIB as u64
                && (deflate_len(&floor_gz) as u64) < 8 * MIB as u64,
            "floor fixture must sit below the 8 MiB floor (isize={} deflate={})",
            isize_of(&floor_gz),
            deflate_len(&floor_gz)
        );
        let mut sink = Vec::with_capacity(floor_p.len());
        crate::decompress::decompress_single_member(&floor_gz, &mut sink, 8).expect("floor decode");
        let after_floor = SMALL_OUTPUT_SERIAL_FLOOR_APPLIED.load(Ordering::Relaxed);
        assert!(
            after_floor >= before_floor + 2,
            "FLOOR PATH NOT HIT: SMALL_OUTPUT_SERIAL_FLOOR_APPLIED {before_floor}->{after_floor} \
             (expected +2: one direct call + one real production decode) — test is INERT on the floor"
        );
        assert_eq!(
            sink,
            oracle_flate2(&floor_gz),
            "floor real-decode bytes wrong"
        );

        // ============================================================
        // 2. RATIO CAP (>= 8×) — direct decision (floor disabled to isolate).
        // ============================================================
        clear_selector_env();
        std::env::set_var(E_MIN_OUT, "0"); // disable floor
        std::env::set_var(E_MARGIN, "1.0"); // arch-independent pin
        let before_cap = COMPRESSIBILITY_THREAD_CAP_APPLIED.load(Ordering::Relaxed);
        // isize 10 MiB, deflate 1 MiB ⇒ ratio 10 ≥ 8 ⇒ cap fires.
        let cap_blob = blob_with_isize(10 * MIB as u32);
        assert_eq!(
            effective_parallel_threads(&cap_blob, MIB, 8),
            1,
            "cap: ratio 10 >= 8 must cap to serial"
        );
        // Just below the cap (ratio ~7.9) must NOT cap.
        let uncap_blob = blob_with_isize((79 * MIB / 10) as u32);
        assert_eq!(
            effective_parallel_threads(&uncap_blob, MIB, 8),
            8,
            "cap: ratio 7.9 < 8 must keep threads"
        );
        let after_cap = COMPRESSIBILITY_THREAD_CAP_APPLIED.load(Ordering::Relaxed);
        assert!(
            after_cap > before_cap,
            "RATIO-CAP PATH NOT HIT: COMPRESSIBILITY_THREAD_CAP_APPLIED {before_cap}->{after_cap}"
        );

        // ============================================================
        // 3. CROSSOVER (serial-clean) — ratio 3 (< cap 8), margin 1.0 ⇒ xover 3.
        // ============================================================
        clear_selector_env();
        std::env::set_var(E_MIN_OUT, "0"); // disable floor
        std::env::set_var(E_MARGIN, "1.0");
        let xo_blob = blob_with_isize(3 * MIB as u32); // ratio 3 vs deflate 1 MiB
        let before_xo = SERIAL_CLEAN_FLOOR_APPLIED.load(Ordering::Relaxed);
        assert_eq!(
            effective_parallel_threads(&xo_blob, MIB, 2),
            1,
            "crossover: T2 < crossover(3) must route serial"
        );
        assert_eq!(
            effective_parallel_threads(&xo_blob, MIB, 8),
            8,
            "crossover: T8 >= crossover(3) must stay parallel"
        );
        let after_xo = SERIAL_CLEAN_FLOOR_APPLIED.load(Ordering::Relaxed);
        assert!(
            after_xo > before_xo,
            "CROSSOVER PATH NOT HIT: SERIAL_CLEAN_FLOOR_APPLIED {before_xo}->{after_xo}"
        );

        // ============================================================
        // 4. LARGE-OUTPUT NOTCH — 1-notch bonus drops crossover 3→2 at T=2.
        // ============================================================
        clear_selector_env();
        std::env::set_var(E_MIN_OUT, "0");
        std::env::set_var(E_MARGIN, "1.0"); // crossover 3
        std::env::set_var(E_LARGE_OUT, "1"); // 1-byte threshold ⇒ everything is "large"
        std::env::set_var(E_NOTCH, "1");
        assert_eq!(
            effective_parallel_threads(&xo_blob, MIB, 2),
            2,
            "notch: 1-notch bonus must lower crossover 3->2 so T2 stays parallel"
        );
        std::env::set_var(E_LARGE_OUT, "0"); // notch off
        assert_eq!(
            effective_parallel_threads(&xo_blob, MIB, 2),
            1,
            "notch off: T2 must revert to serial (crossover 3)"
        );
        clear_selector_env();

        // ============================================================
        // 5. NOTCH is BYTE-TRANSPARENT on a REAL decode (threshold lowered so a
        //    12 MiB output is "large"). Bytes must match the oracle regardless
        //    of whether the notch flipped the realized thread count.
        // ============================================================
        let notch_p = gen_payload(12 * MIB, 3.0, 0x0F5A);
        let notch_gz = gz_at(&notch_p, 6);
        let notch_ref = oracle_flate2(&notch_gz);
        let notch_env: &[(&str, &str)] = &[
            (E_MIN_OUT, "0"),
            (E_MARGIN, "1.0"),
            (E_LARGE_OUT, "1048576"),
            (E_NOTCH, "1"),
        ];
        let notch_got = decode_under(&notch_gz, 2, notch_env, notch_ref.len());
        assert_eq!(
            notch_got, notch_ref,
            "CORRECTNESS BUG: large-output notch is NOT byte-transparent"
        );

        // ============================================================
        // 6. CHUNK-ADJUST (min-work-per-thread) — REAL decode at high T on a
        //    small file shrinks the chunk below the 4 MiB default.
        // ============================================================
        clear_selector_env();
        std::env::set_var(E_MARGIN, "0"); // keep it parallel (skip the crossover)
        std::env::set_var(E_RATIO_MAX, "0"); // and the ratio cap does not pre-empt
        let before_adj = ADJUSTED_CHUNK_SIZE_APPLIED.load(Ordering::Relaxed);
        // 12 MiB out at T=16: 4 MiB default × 2 × 16 ≫ fileSize ⇒ shrink fires.
        let mut sink2 = Vec::with_capacity(notch_p.len());
        crate::decompress::decompress_single_member(&notch_gz, &mut sink2, 16)
            .expect("chunk-adjust decode");
        let after_adj = ADJUSTED_CHUNK_SIZE_APPLIED.load(Ordering::Relaxed);
        clear_selector_env();
        assert_eq!(sink2, notch_ref, "chunk-adjust decode bytes wrong");
        assert!(
            after_adj > before_adj,
            "CHUNK-ADJUST PATH NOT HIT: ADJUSTED_CHUNK_SIZE_APPLIED {before_adj}->{after_adj}"
        );
    }

    // ── SEEDED PROPTEST (>=256 cases) ─────────────────────────────────────────
    //
    // Randomizes output size (mostly below the 8 MiB floor so the floor path is
    // hit constantly), ISIZE ratio, gzip level, and thread count. Every case
    // asserts all three configs == both oracles, byte-for-byte. Shrinks to a
    // minimal repro on any divergence.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_selector_byte_transparent(
            // 96 KiB .. ~7.9 MiB — straddles the floor, cheap to decode.
            len in 96usize * 1024 ..= (8 * MIB - 4096),
            ratio_milli in 1100u32 ..= 22000, // ratio 1.1 .. 22.0
            level in prop::sample::select(vec![1u32, 6, 9]),
            t in prop::sample::select(vec![1usize, 4, 8, 16]),
            seed in any::<u64>(),
        ) {
            let ratio = ratio_milli as f64 / 1000.0;
            let payload = gen_payload(len, ratio, seed);
            let gz = gz_at(&payload, level);

            let exact = isize_of(&gz) as usize;
            let reference = oracle_flate2(&gz);
            let ref_ld = oracle_libdeflate(&gz, exact);
            prop_assert_eq!(&reference, &ref_ld, "oracle disagreement (len={}, ratio={})", len, ratio);
            prop_assert_eq!(reference.len(), exact);

            let _lock = MARKER_PIPELINE_TEST_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());

            for cfg in CONFIGS {
                let before = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
                let got = decode_under(&gz, t, cfg.env, reference.len());
                let after = MARKER_PIPELINE_RUNS.load(Ordering::Relaxed);
                prop_assert!(
                    after > before,
                    "ParallelSM did not run [{},t={}] len={} ratio={}",
                    cfg.label, t, len, ratio
                );
                prop_assert_eq!(
                    got, reference.clone(),
                    "BYTE DIVERGENCE [{},t={}] len={} ratio={} level={} seed={}",
                    cfg.label, t, len, ratio, level, seed
                );
            }
        }
    }
}
