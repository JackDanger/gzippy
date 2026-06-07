//! Engine isolation microbench — CLAUDE.md rule-3 "isolation oracle".
//!
//! Decodes ONE known-window CLEAN silesia deflate chunk three ways, byte-exact
//! each, and reports the per-variant CLEAN decode RATE (MB/s). This BOUNDS the
//! engine speed-up ceiling: it removes the parallel-SM scheduler/publish/marker
//! machinery and measures only the single-thread inner clean-decode rate.
//!
//! THREE VARIANTS (same input slice, same start_bit, same 32 KiB window, same N):
//!  (i)   VAR_I  scalar_u16 — gzippy's CURRENT clean inner loop:
//!        `marker_inflate::Block` with the window pre-primed via
//!        `set_initial_window` so `contains_marker_bytes == false` from the first
//!        block (a genuine clean decode). Output accumulates as `Vec<u16>` (one
//!        u16 per decoded byte through the u16 ring) and is narrowed u16->u8 ONCE
//!        at the end. This is the SCALAR u16 baseline.
//!  (ii)  VAR_II E1-partial — same `Block` clean inner loop, but the decode sink
//!        is u8-direct (`U8Sink`): the post-flip drain calls `push_clean_u8`,
//!        which here writes bytes STRAIGHT into a `Vec<u8>` with NO u16
//!        accumulation and NO final narrow pass. This halves the OUTPUT write
//!        traffic (u8 vs u16) and removes the narrow. NOTE: the inner ring itself
//!        is still u16 (a full E1 would make the ring u8 too) — so this bounds the
//!        OUTPUT-traffic component of E1, reported honestly as "E1-partial".
//!  (iii) VAR_III isal — `isal_decompress::decompress_deflate_from_bit`, the FFI
//!        ISA-L oracle (upper bound; FFI is a MEASUREMENT oracle only).
//!
//! Byte-exactness is the ABSOLUTE gate: all three outputs must be identical over
//! the first N bytes or the rate numbers are VOID.
//!
//! Self-test (RECALIBRATED round-2): on a clean single-thread chunk pure ISA-L
//! is ~3x gzippy's current scalar-u16 inner loop. The round-1 band [1.7x,2.6x]
//! was MIS-CALIBRATED — it was lifted from the 2.1-2.38x system-vs-system wall
//! ratio, but THIS bench's (iii) is a PURE ISA-L clean decode (no marker
//! machinery, no CRC), a purer/faster denominator that yields a LARGER honest
//! ratio (advisor-confirmed iii/ii ~= 3.10x, iii/i ~= 3.29x). PASS band
//! (iii)/(i) in [2.5x, 3.6x] (guest ratio; under Rosetta the absolute MB/s
//! differ but the ratio should still hold — note if it does not; the guest run
//! is authoritative).

#[cfg(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    feature = "pure-rust-inflate"
))]
mod bench {
    use gzippy::decompress::inflate::consume_first_decode::Bits;
    use gzippy::decompress::parallel::marker_inflate::{Block, MarkerSink, MAX_WINDOW_SIZE};
    use gzippy::isal_decompress_oracle::decompress_deflate_from_bit;
    use std::time::Instant;

    const SEED_PATH: &str = "/tmp/engine.seed";
    const CORPUS: &str = "benchmark_data/silesia-gzip.tar.gz";
    const REQUESTED_N: usize = 4 * 1024 * 1024;
    const ITERS: usize = 11; // best-of-N, N >= 9

    // ── GZSEEDW2 seed-file parse (mirror of seed_windows.rs:163-224) ──────────
    struct SeedEntry {
        start_bit: usize,
        window: Vec<u8>,
    }

    fn load_seed() -> Vec<SeedEntry> {
        let buf = std::fs::read(SEED_PATH)
            .unwrap_or_else(|e| panic!("cannot read seed {SEED_PATH}: {e} (run the capture step)"));
        assert!(
            buf.len() >= 16 && &buf[0..8] == b"GZSEEDW2",
            "bad seed magic"
        );
        let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let mut p = 16usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            assert!(p + 16 <= buf.len(), "truncated seed entry header");
            let off = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(buf[p + 8..p + 16].try_into().unwrap()) as usize;
            p += 16;
            assert!(p + len <= buf.len(), "truncated seed entry body");
            out.push(SeedEntry {
                start_bit: off,
                window: buf[p..p + len].to_vec(),
            });
            p += len;
        }
        out.sort_by_key(|e| e.start_bit);
        out
    }

    /// Load the raw deflate slice (header stripped, 8-byte trailer dropped) —
    /// IDENTICAL base to what `sm_driver::read_parallel_sm` passes to
    /// `chunk_fetcher::drive`, i.e. the base that seed start_bits are relative to.
    fn load_deflate() -> Vec<u8> {
        let data = std::fs::read(CORPUS).unwrap_or_else(|e| panic!("cannot read {CORPUS}: {e}"));
        let (_h, header) = gzippy::decompress::parallel::gzip_format::read_header(&data)
            .expect("gzip header parse");
        data[header..data.len().saturating_sub(8)].to_vec()
    }

    // ── u8-direct sink for variant (ii) ───────────────────────────────────────
    // The clean-primed Block drains exclusively via `push_clean_u8` (drain's
    // contains_marker_bytes==false branch), so this sink only ever sees u8 bytes
    // and stores them directly — no u16 accumulation, no final narrow.
    struct U8Sink {
        data: Vec<u8>,
    }
    impl U8Sink {
        fn with_capacity(c: usize) -> Self {
            Self {
                data: Vec::with_capacity(c),
            }
        }
    }
    impl MarkerSink for U8Sink {
        #[inline]
        fn push_slice(&mut self, values: &[u16]) {
            // Defensive: a clean decode never hits this (drain uses push_clean_u8).
            for &v in values {
                debug_assert!((v as usize) < 256, "marker value {v:#x} on clean path");
                self.data.push(v as u8);
            }
        }
        #[inline]
        fn sink_len(&self) -> usize {
            self.data.len()
        }
        #[inline]
        fn as_slice(&self) -> &[u16] {
            &[]
        }
        #[inline]
        fn push_clean_u8(&mut self, bytes: &[u8]) {
            self.data.extend_from_slice(bytes);
        }
    }

    // ── Variant (i): scalar u16 clean decode → Vec<u16>, narrow once at end ────
    fn decode_var_i(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        let mut block = Block::new();
        let mut dummy: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut dummy, window)
            .expect("prime window (i)");
        debug_assert!(
            !block.contains_marker_bytes(),
            "(i) window-primed block must be clean"
        );
        let mut sink: Vec<u16> = Vec::with_capacity(target_n + 4096);
        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        loop {
            block.read_header(&mut bits, false).expect("(i) header");
            while !block.eob() {
                block
                    .read(&mut bits, &mut sink, usize::MAX)
                    .expect("(i) body");
            }
            if block.is_last_block() || sink.len() >= target_n {
                break;
            }
        }
        // Narrow u16 -> u8 (the variant-(i) final pass).
        sink.iter().map(|&v| v as u8).collect()
    }

    // ── Variant (ii): E1-partial u8-direct sink (no u16 accumulation/narrow) ───
    fn decode_var_ii(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        let mut block = Block::new();
        let mut dummy: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut dummy, window)
            .expect("prime window (ii)");
        debug_assert!(
            !block.contains_marker_bytes(),
            "(ii) window-primed block must be clean"
        );
        let mut sink = U8Sink::with_capacity(target_n + 4096);
        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        loop {
            block.read_header(&mut bits, false).expect("(ii) header");
            while !block.eob() {
                block
                    .read(&mut bits, &mut sink, usize::MAX)
                    .expect("(ii) body");
            }
            if block.is_last_block() || sink.sink_len() >= target_n {
                break;
            }
        }
        sink.data
    }

    // ── Variant (iii): ISA-L FFI oracle ───────────────────────────────────────
    fn decode_var_iii(deflate: &[u8], start_bit: usize, window: &[u8], target_n: usize) -> Vec<u8> {
        decompress_deflate_from_bit(deflate, start_bit, window, target_n)
            .expect("(iii) isal decode")
    }

    // ── Variant (iv): clean E2/E3/E4 engine via read_clean_e234 ────────────────
    // Drives the new `Block::read_clean_e234` clean-only sibling (const-generic
    // E2/E3/E4 flags) and drains via `drain_clean_u8` into a Vec<u8> directly —
    // same u8-direct sink as variant (ii), so the E-deltas isolate the inner
    // technique, not the output-traffic component already in (ii).
    fn decode_var_iv<const E2: bool, const E3: bool, const E4: bool>(
        deflate: &[u8],
        start_bit: usize,
        window: &[u8],
        target_n: usize,
    ) -> Vec<u8> {
        let mut block = Block::new();
        let mut dummy: Vec<u16> = Vec::new();
        block
            .set_initial_window(&mut dummy, window)
            .expect("prime window (iv)");
        debug_assert!(
            !block.contains_marker_bytes(),
            "(iv) window-primed block must be clean"
        );
        let mut sink: Vec<u8> = Vec::with_capacity(target_n + 4096);
        let mut bits = Bits::at_bit_offset(deflate, start_bit);
        loop {
            if block.read_header(&mut bits, false).is_err() {
                break;
            }
            while !block.eob() {
                if block
                    .read_clean_e234::<E2, E3, E4>(&mut bits, usize::MAX)
                    .is_err()
                {
                    return sink;
                }
                block.drain_clean_u8(&mut sink);
            }
            if block.is_last_block() || sink.len() >= target_n {
                break;
            }
        }
        sink
    }

    fn crc32(b: &[u8]) -> u32 {
        let mut h = crc32fast::Hasher::new();
        h.update(b);
        h.finalize()
    }

    fn stats(times: &[f64]) -> (f64, f64, f64) {
        // returns (min, median, sigma%)
        let mut s = times.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = s[0];
        let median = s[s.len() / 2];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let var = s.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / s.len() as f64;
        let sigma = var.sqrt();
        (min, median, 100.0 * sigma / mean.max(1e-12))
    }

    // Decode-variant table. Every entry has the SAME signature, so (i)/(ii)/
    // (iii) and the const-generic (iv) stacks live in one array and share the
    // interleaved timing + byte-exact gate. Order matters: index 0 = scalar
    // reference, index 2 = ISA-L oracle (both used as byte-exact denominators).
    type DecodeFn = fn(&[u8], usize, &[u8], usize) -> Vec<u8>;
    const VARIANTS: [(&str, DecodeFn); 7] = [
        ("VAR_I_scalar_u16", decode_var_i),
        ("VAR_II_E1u8_part", decode_var_ii),
        ("VAR_III_isal", decode_var_iii),
        // VAR_IV_E000 is the engine WITH NO TECHNIQUE ON (E2=E3=E4=false). It
        // is the byte-exactness anchor required by the round-2 charter: it must
        // be SHA-identical to (i) scalar AND (iii) ISA-L, proving the new
        // read_clean_e234 entry decodes byte-for-byte like the production
        // <false> path before any technique can be trusted.
        ("VAR_IV_E000", decode_var_iv::<false, false, false>),
        ("VAR_IV_E2", decode_var_iv::<true, false, false>),
        ("VAR_IV_E23", decode_var_iv::<true, true, false>),
        ("VAR_IV_E234", decode_var_iv::<true, true, true>),
    ];

    /// Per-chunk result: median MB/s per variant (index-aligned with VARIANTS)
    /// and whether every variant was byte-exact vs scalar AND scalar vs ISA-L.
    struct ChunkResult {
        med_mbps: [f64; 7],
        exact: [bool; 7],
        all_equal: bool,
        r_iii_i: f64,
    }

    /// Run the full byte-exact gate + interleaved timing for one seed entry.
    /// Returns None when the chunk is unusable (wrong window size, not mid-
    /// stream, or decodes too little).
    fn run_chunk(deflate: &[u8], entry: &SeedEntry) -> Option<ChunkResult> {
        if entry.window.len() != MAX_WINDOW_SIZE {
            return None;
        }
        let start_bit = entry.start_bit;
        let window = &entry.window[..];
        if !(start_bit > 64 && start_bit / 8 < deflate.len()) {
            return None;
        }

        // N_actual from the scalar reference (clamps to BFINAL if early).
        let probe = decode_var_i(deflate, start_bit, window, REQUESTED_N);
        let n_actual = probe.len().min(REQUESTED_N);
        if n_actual < 64 * 1024 {
            return None;
        }

        // Decode every variant once for the byte-exact gate.
        let outs: Vec<Vec<u8>> = VARIANTS
            .iter()
            .map(|(_, f)| f(deflate, start_bit, window, n_actual))
            .collect();
        let scalar = &outs[0][..n_actual];
        let isal = &outs[2][..n_actual];
        let scalar_eq_isal = scalar == isal;
        // Length-safe exact check: variant must be >= n_actual long AND match
        // scalar over [0, n_actual) AND scalar must match ISA-L.
        let mut exact = [false; 7];
        for (k, o) in outs.iter().enumerate() {
            exact[k] = o.len() >= n_actual && &o[..n_actual] == scalar && scalar_eq_isal;
        }
        let all_equal = exact.iter().all(|&b| b);

        if !all_equal {
            eprintln!("BYTE-EXACT FAILURE chunk start_bit={start_bit}:");
            for (k, (label, _)) in VARIANTS.iter().enumerate() {
                if !exact[k] {
                    let common = outs[k].len().min(n_actual);
                    let fd = outs[k][..common]
                        .iter()
                        .zip(&scalar[..common])
                        .position(|(p, q)| p != q);
                    eprintln!(
                        "  {label} VOID len={} (n_actual={n_actual}) first_diff={:?} crc={:#010x} (scalar={:#010x})",
                        outs[k].len(),
                        fd,
                        crc32(&outs[k][..common]),
                        crc32(&scalar[..common])
                    );
                    if let Some(d) = fd {
                        let lo = d.saturating_sub(6);
                        let hi = (d + 10).min(common);
                        eprintln!("    scalar[{lo}..{hi}] = {:02x?}", &scalar[lo..hi]);
                        eprintln!("    {label}[{lo}..{hi}] = {:02x?}", &outs[k][lo..hi]);
                    }
                }
            }
        }

        // Warm-up (discarded) then interleaved best-of-N.
        for (_, f) in VARIANTS.iter() {
            let _ = f(deflate, start_bit, window, n_actual);
        }
        let mut times: [Vec<f64>; 7] = Default::default();
        for _ in 0..ITERS {
            for (k, (_, f)) in VARIANTS.iter().enumerate() {
                let s = Instant::now();
                let r = f(deflate, start_bit, window, n_actual);
                times[k].push(s.elapsed().as_secs_f64());
                std::hint::black_box(&r);
            }
        }

        let mbps = |secs: f64| (n_actual as f64) / secs / 1e6;
        let mut med_mbps = [0.0f64; 7];
        for k in 0..7 {
            let (_min, med, _sig) = stats(&times[k]);
            med_mbps[k] = mbps(med);
        }
        let r_iii_i = med_mbps[2] / med_mbps[0];

        // Per-chunk report.
        println!(
            "CHUNK start_bit={start_bit} N_bytes={n_actual} SHA_ALL_EQUAL={}",
            if all_equal { "yes" } else { "no" }
        );
        for (k, (label, _)) in VARIANTS.iter().enumerate() {
            if exact[k] {
                println!(
                    "  {:<17} MBps_med={:>6.0}  vs_i={:.3} vs_iii={:.3}",
                    label,
                    med_mbps[k],
                    med_mbps[k] / med_mbps[0],
                    med_mbps[k] / med_mbps[2]
                );
            } else {
                println!("  {:<17} VOID (byte-exact gate failed)", label);
            }
        }

        Some(ChunkResult {
            med_mbps,
            exact,
            all_equal,
            r_iii_i,
        })
    }

    pub fn run() {
        // Note the AVX2 status: under Rosetta x86-64-v2 this is false, so E2's
        // scalar word-copy fallback runs and the byte-exact gate validates IT;
        // the AVX2 path itself is only exercised (and measured) on the guest.
        eprintln!("avx2_detected={}", std::is_x86_feature_detected!("avx2"));
        let seed = load_seed();
        assert!(
            seed.len() >= 3,
            "need >=3 seed entries to sweep, got {}",
            seed.len()
        );
        let deflate = load_deflate();

        // Sweep chunks at 10/30/50/70/90% of the sorted-by-start_bit seed list.
        // run_chunk() skips entries without a 32 KiB window or too-short decode,
        // so we over-pick and keep the usable ones.
        let pct = [10usize, 30, 50, 70, 90];
        let mut results: Vec<ChunkResult> = Vec::new();
        let mut median_chunk_idx: Option<usize> = None;
        for (j, &p) in pct.iter().enumerate() {
            let idx = (seed.len().saturating_sub(1) * p) / 100;
            if let Some(r) = run_chunk(&deflate, &seed[idx]) {
                if j == 2 {
                    median_chunk_idx = Some(results.len());
                }
                results.push(r);
            }
        }
        assert!(
            !results.is_empty(),
            "no usable chunks in the sweep (all skipped)"
        );

        // Aggregate: median-of-per-chunk-medians + min/max spread per variant.
        let med_of = |vals: &mut Vec<f64>| -> f64 {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals[vals.len() / 2]
        };
        println!("\nAGGREGATE over {} chunk(s):", results.len());
        for (k, (label, _)) in VARIANTS.iter().enumerate() {
            // Only aggregate chunks where this variant passed the gate.
            let mut vals: Vec<f64> = results
                .iter()
                .filter(|r| r.exact[k])
                .map(|r| r.med_mbps[k])
                .collect();
            if vals.is_empty() {
                println!("  {:<17} VOID (no byte-exact chunk)", label);
                continue;
            }
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let med = med_of(&mut vals);
            println!(
                "  {:<17} MBps_med_of_med={:>6.0}  min={:>6.0} max={:>6.0}",
                label, med, min, max
            );
        }

        // Self-test on the MEDIAN chunk (the 50% pick), preserved from round-2:
        // (iii)/(i) should land in [2.5x, 3.6x] on the guest. Under Rosetta the
        // absolute MB/s are garbage so the ratio can drift — the guest run is
        // authoritative; we only HARD-gate byte-exactness here.
        let all_chunks_exact = results.iter().all(|r| r.all_equal);
        let sha_all = if all_chunks_exact { "yes" } else { "no" };
        let r_iii_i = median_chunk_idx
            .map(|i| results[i].r_iii_i)
            .unwrap_or(results[0].r_iii_i);
        let selftest = r_iii_i >= 2.5 && r_iii_i <= 3.6;
        println!(
            "\nSHA_ALL_EQUAL={}  SELFTEST={}  (median-chunk iii/i={:.3})",
            sha_all,
            if selftest { "PASS" } else { "FAIL" },
            r_iii_i
        );
        if !selftest {
            eprintln!(
                "SELFTEST note: (iii)/(i)={:.3} outside [2.5,3.6]. Under Rosetta the \
                 ratio can drift; the guest run is authoritative.",
                r_iii_i
            );
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    feature = "pure-rust-inflate"
))]
fn main() {
    bench::run();
}

#[cfg(not(all(
    target_arch = "x86_64",
    feature = "isal-compression",
    feature = "pure-rust-inflate"
)))]
fn main() {
    eprintln!("engine_isolation: requires x86_64 + isal-compression + pure-rust-inflate");
}
