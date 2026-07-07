//! Selector byte-transparency differential + proptest.
//!
//! WHY (campaign): `effective_parallel_threads`
//! (`src/decompress/parallel/single_member.rs`) is a PURE thread-count selector
//! with FOUR decision dimensions —
//!   1. the serial-clean crossover margin (also the sole owner of high-ratio
//!      routing since the T-blind hard ratio cap env knob
//!      was deleted, 2026-07-05),
//!   2. the small-output serial FLOOR (8 MiB),
//!   3. the large-output notch bonus,
//!   4. the small-file chunk-size adjustment (min-work-per-thread,
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
//!   (a) the production selector (frozen defaults, `effective_parallel_threads`),
//!   (b) forced-parallel (selector bypassed — decode with exactly the
//!       requested thread count),
//!   (c) forced-serial (selector bypassed — decode with exactly 1 thread) —
//! and asserts ALL THREE are byte-identical to TWO independent oracles
//! (flate2/zlib-ng + libdeflate). If any selector branch corrupted a byte
//! (e.g. a serial/parallel seam divergence that only manifests under a
//! particular thread count) it lands RED here.
//!
//! (2026-07: the five parallel-selector env
//! knobs that used to drive configs (b)/(c) were frozen to their campaign-
//! measured defaults and the env reads deleted — see
//! `single_member::effective_parallel_threads_with`, the now-pure
//! parameterized core. Configs (b)/(c) are reproduced here by calling
//! `sm_driver::read_parallel_sm_capturing` directly with an explicit
//! `parallelization`, bypassing the selector rather than overriding it via
//! env — a strictly stronger check, since it covers EVERY thread count, not
//! just the ones a particular env override happened to force.)
//!
//! Gate-0 non-inertness: the boundary suite PROVES it exercises the <8 MiB
//! floor path (asserts `SMALL_OUTPUT_SERIAL_FLOOR_APPLIED` increments across a
//! real production decode) plus the crossover, notch, and chunk-adjust
//! branches (each via its dedicated deletion-trap counter).
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
        MARKER_PIPELINE_TEST_LOCK, SERIAL_CLEAN_FLOOR_APPLIED, SMALL_OUTPUT_SERIAL_FLOOR_APPLIED,
    };

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

    /// How the requested thread count `t` is turned into the ACTUAL
    /// `parallelization` handed to the driver.
    #[derive(Clone, Copy)]
    enum Mode {
        /// The frozen production selector (`effective_parallel_threads`).
        Production,
        /// Selector bypassed — decode with exactly the requested `t`.
        ForcedParallel,
        /// Selector bypassed — decode with exactly 1 thread.
        ForcedSerial,
    }

    struct Cfg {
        label: &'static str,
        mode: Mode,
    }

    /// (a) production (frozen selector), (b) forced-parallel (selector
    /// bypassed, exact requested `t`), (c) forced-serial (selector bypassed,
    /// pinned to 1 thread). All three drive the SAME underlying `sm_driver`
    /// engine, so this still exercises the parallel-vs-serial seam. All three
    /// MUST decode byte-identically.
    const CONFIGS: &[Cfg] = &[
        Cfg {
            label: "production",
            mode: Mode::Production,
        },
        Cfg {
            label: "forced-parallel",
            mode: Mode::ForcedParallel,
        },
        Cfg {
            label: "forced-serial",
            mode: Mode::ForcedSerial,
        },
    ];

    /// Decode `gz` requesting thread count `t`, resolving to an actual
    /// `parallelization` per `cfg.mode`, then driving `sm_driver` directly.
    /// This bypasses `single_member::decompress_parallel`'s classifier
    /// wrapper — there is no other decode engine to silently fall back to
    /// (NO FALLBACKS, per project rules), so a successful, length-matching
    /// return already proves the real chunk_fetcher/marker pipeline ran.
    fn decode_under(gz: &[u8], t: usize, cfg: &Cfg, reserve: usize) -> Vec<u8> {
        let dlen = deflate_len(gz);
        let parallelization = match cfg.mode {
            Mode::Production => effective_parallel_threads(gz, dlen, t),
            Mode::ForcedParallel => t,
            Mode::ForcedSerial => 1,
        };
        let mut out = Vec::with_capacity(reserve);
        let mut bytes_written = 0usize;
        let r = crate::decompress::parallel::sm_driver::read_parallel_sm_capturing(
            gz,
            &mut out,
            None,
            parallelization.max(1),
            4 * 1024 * 1024,
            &mut bytes_written,
            false,
        );
        r.unwrap_or_else(|e| panic!("decode failed (t={t}, cfg={}): {e:?}", cfg.label));
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
                let got = decode_under(gz, t, cfg, reference.len());
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
            // --- higher-ratio crossover cells (the deleted T-blind cap used to
            //     sit at ratio 8; the T-aware crossover now owns these) ---
            ("cap_below", 12 * MIB, 7.4, 6),
            ("cap_above", 12 * MIB, 8.6, 6),
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
    // branch DECISIONS use synthetic exact-(isize,deflate) blobs through
    // `single_member::effective_parallel_threads_with` (explicit params pin the
    // margin/floor/notch dimensions being exercised, deterministic on x86 AND
    // aarch64 — no env, no arch dispatch); the non-inert proofs use real
    // compressible streams whose floor/chunk-adjust preconditions hold with wide
    // margin regardless of the exact realized ratio.
    #[test]
    fn selector_branches_are_exercised() {
        use crate::decompress::parallel::single_member::effective_parallel_threads_with;

        let _lock = MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        // ============================================================
        // 1. <8 MiB SMALL-OUTPUT FLOOR — direct decision (frozen production
        //    default, arch-independent) + REAL decode.
        // ============================================================
        let before_floor = SMALL_OUTPUT_SERIAL_FLOOR_APPLIED.load(Ordering::Relaxed);

        // Direct: isize 6 MiB, deflate 2 MiB (ratio 3 ≥ 1.5, both < 8 MiB) ⇒ floor
        // fires ⇒ serial at every T>1.
        let floor_blob = blob_with_isize(6 * MIB as u32);
        assert_eq!(
            effective_parallel_threads(&floor_blob, 2 * MIB, 8),
            1,
            "floor: <8 MiB compressible must cap to serial (frozen default)"
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
        crate::decompress::decompress_single_member(&floor_gz, &mut sink, 8, false)
            .expect("floor decode");
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
        // 2. HIGH-RATIO ROUTING (blind cap deleted on x86_64, 2026-07-05).
        //    x86_64: routing is the T-aware crossover's decision alone — ratio 10
        //    at T8 (below its crossover 10) routes serial VIA THE SELECTOR
        //    (counter proves it); the same stream at T16 (>= crossover)
        //    parallelizes — impossible under the old blind cap.
        //    aarch64: the prestack cap remains (selector disabled there; cap
        //    removal regresses M1 2.1-2.6× — see single_member.rs) → serial at
        //    both T8 and T16.
        //    margin pinned to 1.0 (vendor-independent) and the floor disabled,
        //    isolating the crossover/prestack-cap decision under test.
        // ============================================================
        let hr_blob = blob_with_isize(10 * MIB as u32);
        let hr_call = |t: usize| {
            effective_parallel_threads_with(
                &hr_blob,
                MIB,
                t,
                0,   // floor disabled
                1.0, // margin pinned
                crate::decompress::parallel::single_member::PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
                crate::decompress::parallel::single_member::arch_large_output_notch_default(),
                0, // work-per-thread cap disabled
                crate::decompress::parallel::single_member::MIN_THREADS_FLOOR_DEFAULT,
            )
        };
        assert_eq!(
            hr_call(8),
            1,
            "ratio 10 at T8 (below crossover 10) routes serial"
        );
        #[cfg(not(target_arch = "aarch64"))]
        {
            let before_xover = SERIAL_CLEAN_FLOOR_APPLIED.load(Ordering::Relaxed);
            assert_eq!(
                hr_call(8),
                1,
                "x86: selector (not a cap) serializes below crossover"
            );
            assert_eq!(
                hr_call(16),
                16,
                "x86: ratio 10 at T16 (>= crossover 10) must parallelize — no blind cap"
            );
            let after_xover = SERIAL_CLEAN_FLOOR_APPLIED.load(Ordering::Relaxed);
            assert!(
                after_xover > before_xover,
                "SELECTOR PATH NOT HIT: SERIAL_CLEAN_FLOOR_APPLIED {before_xover}->{after_xover}"
            );
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            hr_call(16),
            1,
            "aarch64: prestack cap serializes ratio >= 8 at every T"
        );

        // ============================================================
        // 3. CROSSOVER (serial-clean) — ratio 3 (< cap 8), margin 1.0 ⇒ xover 3.
        // ============================================================
        let xo_blob = blob_with_isize(3 * MIB as u32); // ratio 3 vs deflate 1 MiB
        let xo_call = |t: usize| {
            effective_parallel_threads_with(
                &xo_blob,
                MIB,
                t,
                0,
                1.0,
                crate::decompress::parallel::single_member::PARALLEL_LARGE_OUTPUT_BYTES_DEFAULT,
                crate::decompress::parallel::single_member::arch_large_output_notch_default(),
                0,
                crate::decompress::parallel::single_member::MIN_THREADS_FLOOR_DEFAULT,
            )
        };
        let before_xo = SERIAL_CLEAN_FLOOR_APPLIED.load(Ordering::Relaxed);
        assert_eq!(
            xo_call(2),
            1,
            "crossover: T2 < crossover(3) must route serial"
        );
        assert_eq!(
            xo_call(8),
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
        let notch_call = |t: usize, large_output_bytes: u64, notch: u64| {
            effective_parallel_threads_with(
                &xo_blob,
                MIB,
                t,
                0,
                1.0, // crossover 3
                large_output_bytes,
                notch,
                0,
                crate::decompress::parallel::single_member::MIN_THREADS_FLOOR_DEFAULT,
            )
        };
        assert_eq!(
            notch_call(2, 1, 1), // 1-byte threshold ⇒ everything is "large"
            2,
            "notch: 1-notch bonus must lower crossover 3->2 so T2 stays parallel"
        );
        assert_eq!(
            notch_call(2, 0, 1), // notch off (threshold 0 disables the bonus)
            1,
            "notch off: T2 must revert to serial (crossover 3)"
        );

        // ============================================================
        // 5. NOTCH is BYTE-TRANSPARENT on a REAL decode — decode the SAME 12 MiB
        //    fixture at several EXPLICIT parallelization values (bypassing the
        //    selector via the `sm_driver` seam), asserting bytes match the
        //    oracle regardless of thread count. This subsumes "does the notch
        //    flipping the thread count corrupt bytes" — arithmetic #4 above
        //    already proves the notch's DECISION; this proves every reachable
        //    thread count DECODES correctly.
        // ============================================================
        let notch_p = gen_payload(12 * MIB, 3.0, 0x0F5A);
        let notch_gz = gz_at(&notch_p, 6);
        let notch_ref = oracle_flate2(&notch_gz);
        for &t in &[1usize, 2, 3, 4] {
            let mut got = Vec::with_capacity(notch_ref.len());
            let mut bytes_written = 0usize;
            crate::decompress::parallel::sm_driver::read_parallel_sm_capturing(
                &notch_gz,
                &mut got,
                None,
                t,
                4 * 1024 * 1024,
                &mut bytes_written,
                false,
            )
            .unwrap_or_else(|e| panic!("notch real-decode t={t} failed: {e:?}"));
            assert_eq!(
                got, notch_ref,
                "CORRECTNESS BUG: t={t} real decode is NOT byte-transparent"
            );
        }

        // ============================================================
        // 6. CHUNK-ADJUST (min-work-per-thread) — REAL decode at high T on a
        //    small file shrinks the chunk below the 4 MiB default. Frozen
        //    production defaults keep this fixture (ratio 3, 12 MiB out) parallel
        //    at T16 on every arch (Intel crossover 3, AMD crossover 5, aarch64
        //    selector off), so no override is needed to reach the chunk-adjust
        //    branch.
        // ============================================================
        let before_adj = ADJUSTED_CHUNK_SIZE_APPLIED.load(Ordering::Relaxed);
        // 12 MiB out at T=16: 4 MiB default × 2 × 16 ≫ fileSize ⇒ shrink fires.
        let mut sink2 = Vec::with_capacity(notch_p.len());
        crate::decompress::decompress_single_member(&notch_gz, &mut sink2, 16, false)
            .expect("chunk-adjust decode");
        let after_adj = ADJUSTED_CHUNK_SIZE_APPLIED.load(Ordering::Relaxed);
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
                let got = decode_under(&gz, t, cfg, reference.len());
                prop_assert_eq!(
                    got, reference.clone(),
                    "BYTE DIVERGENCE [{},t={}] len={} ratio={} level={} seed={}",
                    cfg.label, t, len, ratio, level, seed
                );
            }
        }
    }
}
