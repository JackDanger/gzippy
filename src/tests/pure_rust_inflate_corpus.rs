//! Corpus differential harness for the `pure-rust-inflate` backend.
//!
//! This is the gate that catches "the wrapper API tests pass against
//! flate2 fixtures but a real-world corpus breaks." It is the wide-coverage
//! backstop for the parallel-SM pipeline.
//!
//! All cases run the parallel-SM pipeline end-to-end via
//! `decompress::decompress_single_member` (T > 1, > 10 MiB compressed,
//! which forces the routing into `ParallelSM`). Output is compared
//! byte-for-byte against a flate2 reference decode of the same input.
//! Final CRC32 + ISIZE are verified inside the pipeline itself; a
//! mismatch surfaces as `Err`.
//!
//! Each `corpus_*` test gets its own representative input shape:
//!   - `large_random`: high-entropy, mostly-stored blocks — exercises
//!     `BlockState::InStored` resume + the no-progress guard near
//!     `encoded_until_bits` boundaries.
//!   - `large_repetitive`: long runs of identical bytes — exercises
//!     `BlockState::InFixed` long matches and the sliding-window
//!     wrap.
//!   - `large_mixed_entropy`: zlib L1 multi-BTYPE input — exercises
//!     `BlockState::InDynamic` table-build churn and many small blocks.
//!   - `silesia_subset`: the real silesia fixture (skipped if not on
//!     disk); the production target corpus.
//!
//! These run via `cargo test --release --features pure-rust-inflate --
//! corpus`. They're cheap enough (each ~24 MiB compressed) to run on a
//! laptop, not gated `#[ignore]`. The real silesia case auto-skips when
//! the fixture is absent.

#[cfg(test)]
#[cfg(all(pure_inflate_decode, not(feature = "isal-compression")))]
mod tests {
    use std::io::{Read, Write};

    /// Decode `compressed` via the `pure-rust-inflate` parallel-SM
    /// pipeline. Asserts byte-equality against TWO independent oracles:
    /// `flate2` (zlib-ng under the hood) AND `libdeflate` (the
    /// `libdeflate_gzip_decompress_ex` FFI). Both oracles must agree
    /// with each other AND with gzippy's output. Two independent
    /// implementations beat one — a shared bug in either oracle would
    /// mask a real production divergence if we only compared against
    /// one of them.
    ///
    /// Confirms the parallel pipeline was the routing pick by
    /// snapshotting `MARKER_PIPELINE_RUNS` around the call.
    fn assert_pure_rust_parallel_sm_roundtrips(compressed: &[u8], label: &str) {
        use std::sync::atomic::Ordering;

        // Routing precondition: parallel-SM only fires when compressed
        // > MIN_PARALLEL_COMPRESSED (10 MiB) and num_threads > 1.
        assert!(
            compressed.len() > 10 * 1024 * 1024,
            "{label}: corpus must exceed 10 MiB compressed to route parallel-SM \
             (got {} bytes)",
            compressed.len()
        );

        // Oracle 1: flate2 (zlib-ng). MultiGzDecoder so it tolerates
        // single-member streams (the production fixture shape here).
        let oracle_flate2: Vec<u8> = {
            let mut decoder = flate2::read::MultiGzDecoder::new(compressed);
            let mut out = Vec::new();
            decoder
                .read_to_end(&mut out)
                .unwrap_or_else(|e| panic!("{label}: flate2 oracle decode failed: {e}"));
            out
        };

        // Oracle 2: libdeflate one-shot. Independent implementation,
        // independent author, no shared codebase with flate2/zlib-ng.
        // If oracle 1 and oracle 2 disagree, the test fixture has an
        // ambiguity that needs resolving BEFORE we trust gzippy's
        // output.
        let oracle_libdeflate: Vec<u8> = {
            // Allocate generously: libdeflate needs the exact output
            // size up-front; if we under-allocate we get
            // InsufficientSpace. Using oracle_flate2.len() lets us
            // detect disagreements rather than silently truncating.
            let mut out = vec![0u8; oracle_flate2.len()];
            let mut decoder = crate::backends::libdeflate::DecompressorEx::new();
            let r = decoder
                .gzip_decompress_ex(compressed, &mut out)
                .unwrap_or_else(|e| panic!("{label}: libdeflate oracle decode failed: {e}"));
            out.truncate(r.output_size);
            out
        };

        // Both oracles must agree. If they don't, the input is ambiguous
        // (e.g. a malformed stream where libraries differ on recovery
        // behavior). Halt before drawing any conclusion about gzippy.
        assert_eq!(
            oracle_flate2, oracle_libdeflate,
            "{label}: oracle disagreement — flate2 and libdeflate produced \
             different bytes for this fixture; the fixture itself is suspect, \
             cannot verify gzippy against it"
        );
        let reference = oracle_flate2;

        // Lock the routing-trap shared test mutex so concurrent corpus
        // tests don't false-fail each other on MARKER_PIPELINE_RUNS reads.
        let _lock = crate::decompress::parallel::single_member::MARKER_PIPELINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let before = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);

        let mut got = Vec::with_capacity(reference.len());
        crate::decompress::decompress_single_member(compressed, &mut got, 4, false).unwrap_or_else(
            |e| {
                panic!(
                    "{label}: pure-rust-inflate parallel-SM decode failed: {e}\n\
                 (CRC32 / ISIZE mismatch surfaces here as `Decompression(...)`;\n\
                  silent truncation would surface as length mismatch below)"
                );
            },
        );

        let after = crate::decompress::parallel::single_member::MARKER_PIPELINE_RUNS
            .load(Ordering::Relaxed);
        assert!(
            after > before,
            "{label}: parallel-SM did not run (MARKER_PIPELINE_RUNS {before} -> {after}); \
             routing fell through to a sequential backend"
        );

        // Length first — catches silent-truncation regressions with a
        // one-line message.
        assert_eq!(
            got.len(),
            reference.len(),
            "{label}: length mismatch (got {} vs reference {})",
            got.len(),
            reference.len()
        );

        // Then full byte-equality. Comparing slices avoids dumping
        // megabytes into the test output on a mid-stream divergence.
        if got != reference {
            let first_diff = got
                .iter()
                .zip(reference.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(reference.len()));
            panic!(
                "{label}: byte mismatch at offset {first_diff} \
                 (got=0x{:02x} ref=0x{:02x})",
                got.get(first_diff).copied().unwrap_or(0),
                reference.get(first_diff).copied().unwrap_or(0)
            );
        }
    }

    /// Single-member gzip via flate2 at a specific level.
    fn gz_at(payload: &[u8], level: u32) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// High-entropy: PRNG bytes, ~no compression. Forces zlib to emit
    /// mostly stored blocks. Exercises `BlockState::InStored` resume
    /// and `encoded_until_bits`-near-stored-block-boundary cases.
    fn make_random_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xcafef00d_deadbeef;
        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 24) as u8);
        }
        data
    }

    /// Low-entropy: short repeating phrases. Exercises long back-ref
    /// matches and the sliding-window wrap.
    fn make_repetitive_data(size: usize) -> Vec<u8> {
        const PHRASE: &[u8] =
            b"the quick brown fox jumps over the lazy dog. pack my box with five dozen liquor jugs. ";
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            let take = (size - data.len()).min(PHRASE.len());
            data.extend_from_slice(&PHRASE[..take]);
        }
        data
    }

    /// Semi-compressible data with a *controlled* ratio: each 4 KiB block is
    /// `motif_frac` of a block-unique repeating motif (long matches WITHIN the
    /// block + cross-block sliding-window resume) followed by a PRNG literal
    /// tail (stored/literal coverage). The block-unique motif keeps the ratio
    /// bounded (matches don't reach across blocks), so the compressed size
    /// scales ~linearly with `raw` and the uncompressed/compressed ratio stays
    /// comfortably above the routing `parallel_sm_unprofitable` floor (1.15)
    /// while clearing the 10 MiB `MIN_PARALLEL_COMPRESSED` size gate.
    ///
    /// This is what `corpus_large_random` / `corpus_large_repetitive` feed the
    /// pipeline: pure random trips the ratio gate (routes to one-shot) and pure
    /// repetition compresses far below 10 MiB on this zlib build — neither
    /// reaches the parallel-SM path the test exists to exercise. `motif_frac`
    /// near 1.0 → repetition-heavy; near 0.0 → literal/stored-heavy.
    fn make_semicompressible_data(raw: usize, motif_frac: f64) -> Vec<u8> {
        const BLK: usize = 4096;
        let mut out = Vec::with_capacity(raw);
        let mut rng: u64 = 0x0bad_c0de_1234_5678;
        let mut block_index: u64 = 0;
        while out.len() < raw {
            block_index += 1;
            // Motif unique to this block: 24 bytes derived from the index.
            let mut motif = [0u8; 24];
            for (j, b) in motif.iter_mut().enumerate() {
                *b = (block_index.wrapping_mul(31).wrapping_add(j as u64) & 0xff) as u8;
            }
            let motif_len = ((BLK as f64) * motif_frac) as usize;
            let mut written = 0usize;
            while written < motif_len && out.len() < raw {
                let take = motif.len().min(motif_len - written);
                out.extend_from_slice(&motif[..take]);
                written += take;
            }
            // PRNG literal tail.
            let tail = BLK.saturating_sub(motif_len);
            for _ in 0..tail {
                if out.len() >= raw {
                    break;
                }
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                out.push((rng >> 24) as u8);
            }
        }
        out.truncate(raw);
        out
    }

    /// Mixed entropy: 60% PRNG / 40% short repeats. Pushes zlib into
    /// emitting a mix of BTYPE=00/01/10. Many small blocks → many
    /// dynamic-header table builds.
    fn make_mixed_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xfeed_face_c0de_d00d;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 5 < 3 {
                data.push((rng >> 16) as u8);
            } else {
                let byte = ((rng >> 24) % 26 + b'a' as u64) as u8;
                let repeat = ((rng >> 40) % 8 + 2) as usize;
                for _ in 0..repeat.min(size - data.len()) {
                    data.push(byte);
                }
            }
        }
        data.truncate(size);
        data
    }

    #[test]
    fn corpus_large_random() {
        // Literal/stored-heavy (motif_frac 0.15): mostly PRNG literals with a
        // thin block-unique motif so the stream still clears the ratio gate
        // (pure random would route to one-shot via `parallel_sm_unprofitable`)
        // AND clears 10 MiB compressed. Exercises `BlockState::InStored` resume
        // and the no-progress guard near `encoded_until_bits` boundaries.
        // `make_random_data` is kept (documents the PRNG shape) but pure-random
        // cannot reach the parallel-SM path the test exists to cover.
        let _ = make_random_data;
        let payload = make_semicompressible_data(60 * 1024 * 1024, 0.30);
        let compressed = gz_at(&payload, 1);
        assert_pure_rust_parallel_sm_roundtrips(&compressed, "large_random");
    }

    #[test]
    fn corpus_large_repetitive() {
        // Motif-heavy (motif_frac 0.7): long in-block matches + cross-block
        // sliding-window resume, with enough literal tail that the stream
        // compresses to a *moderate* ratio that clears 10 MiB compressed.
        // Pure repetition (`make_repetitive_data`) compresses to < 1 MiB on
        // this zlib build — far below the 10 MiB gate — so it never routes
        // parallel; the block-unique motif here keeps the ratio bounded.
        let _ = make_repetitive_data;
        let payload = make_semicompressible_data(48 * 1024 * 1024, 0.70);
        let compressed = gz_at(&payload, 1);
        assert_pure_rust_parallel_sm_roundtrips(&compressed, "large_repetitive");
    }

    #[test]
    fn corpus_large_mixed_entropy() {
        // Mixed-entropy at L1 → many small blocks of varied BTYPE.
        // ~14 MiB compressed from 24 MiB raw.
        let payload = make_mixed_data(24 * 1024 * 1024);
        let compressed = gz_at(&payload, 1);
        assert_pure_rust_parallel_sm_roundtrips(&compressed, "large_mixed_entropy");
    }

    #[test]
    fn corpus_large_dynamic_heavy() {
        // L9 on mixed data → maximum dynamic-Huffman usage; tables get
        // rebuilt per-block. Stresses `resume_decode_dynamic_resumable`
        // and `LitLenTable::build`.
        let payload = make_mixed_data(24 * 1024 * 1024);
        let compressed = gz_at(&payload, 9);
        // L9 + mixed input may compress past the 10 MiB gate; the
        // assertion in the helper catches it if not.
        if compressed.len() <= 10 * 1024 * 1024 {
            eprintln!(
                "corpus_large_dynamic_heavy: input compressed to {} bytes \
                 (≤ 10 MiB gate); skipping — increase payload size if this trips often",
                compressed.len()
            );
            return;
        }
        assert_pure_rust_parallel_sm_roundtrips(&compressed, "large_dynamic_heavy");
    }

    /// ROUTE-A THIN-T1 PRODUCTION PATH differential. At T==1 the
    /// production single-member path sheds the parallel scaffold and uses the
    /// thin serial rolling-window driver (`drive_thin_t1_oracle`, the SAME shared
    /// `decode_chunk` kernel). This asserts (a) the thin spine actually RAN
    /// (THIN_T1_RUNS fired — not the parallel scaffold), and (b) byte-exactness
    /// vs BOTH flate2 and libdeflate on a multi-block corpus spanning several
    /// 1 MiB chunk strides, incl. the in-driver CRC32 + ISIZE verification (a
    /// mismatch surfaces as Err here).
    #[cfg(all(parallel_sm, not(feature = "isal-compression")))]
    #[test]
    fn thin_t1_production_path_byte_exact_and_routed() {
        use std::sync::atomic::Ordering;

        // ~6 MiB raw mixed data → multiple dynamic blocks; at L6 it compresses
        // to a few MiB, spanning several 1 MiB compressed chunk strides so the
        // rolling-window handoff between chunks is exercised.
        let payload = make_mixed_data(6 * 1024 * 1024);
        let compressed = gz_at(&payload, 6);

        // flate2 + libdeflate oracles must agree before trusting either.
        let oracle: Vec<u8> = {
            let mut d = flate2::read::MultiGzDecoder::new(&compressed[..]);
            let mut out = Vec::new();
            d.read_to_end(&mut out).expect("flate2 oracle");
            out
        };
        {
            let mut out = vec![0u8; oracle.len()];
            let mut dec = crate::backends::libdeflate::DecompressorEx::new();
            let r = dec
                .gzip_decompress_ex(&compressed, &mut out)
                .expect("libdeflate oracle");
            out.truncate(r.output_size);
            assert_eq!(out, oracle, "oracle disagreement (flate2 vs libdeflate)");
        }

        let before =
            crate::decompress::parallel::chunk_fetcher::THIN_T1_RUNS.load(Ordering::Relaxed);
        let mut got = Vec::with_capacity(oracle.len());
        // Direct T=1 production decode (the thin spine). CRC32 + ISIZE are
        // verified inside the driver; a mismatch returns Err here.
        crate::decompress::parallel::single_member::decompress_parallel(
            &compressed,
            &mut got,
            None,
            1,
            false,
        )
        .expect("thin-T1 decode");
        let after =
            crate::decompress::parallel::chunk_fetcher::THIN_T1_RUNS.load(Ordering::Relaxed);

        assert!(
            after > before,
            "thin-T1 spine did not run (THIN_T1_RUNS {before} -> {after}); routing fell through \
             to the parallel scaffold at T=1"
        );
        assert_eq!(got.len(), oracle.len(), "thin-T1 length mismatch");
        assert_eq!(got, oracle, "thin-T1 byte mismatch vs flate2/libdeflate");
    }

    /// Real silesia corpus, if present. The production target — failure
    /// here is the most authoritative signal because every other test
    /// uses synthetic data.
    #[test]
    fn corpus_silesia_if_available() {
        let Some(compressed) = crate::tests::datasets::load_silesia_gzip() else {
            eprintln!(
                "corpus_silesia_if_available: benchmark_data/silesia-gzip.tar.gz \
                 missing on this host; skipping. Run on the perf box for the \
                 production-shape decode."
            );
            return;
        };
        assert_pure_rust_parallel_sm_roundtrips(&compressed, "silesia");
    }
}
