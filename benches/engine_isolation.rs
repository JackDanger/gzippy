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

    pub fn run() {
        let seed = load_seed();
        assert!(
            seed.len() >= 3,
            "need >=3 seed entries to pick a mid-stream chunk, got {}",
            seed.len()
        );
        let deflate = load_deflate();

        // Pick a genuine MID-STREAM clean chunk: median entry by start_bit, with a
        // full 32 KiB window (so set_initial_window flips clean).
        let mid = &seed[seed.len() / 2];
        assert_eq!(
            mid.window.len(),
            MAX_WINDOW_SIZE,
            "chosen seed window is not 32 KiB ({})",
            mid.window.len()
        );
        let start_bit = mid.start_bit;
        let window = &mid.window[..];
        assert!(
            start_bit > 64 && start_bit / 8 < deflate.len(),
            "start_bit {start_bit} not mid-stream in {}-byte deflate",
            deflate.len()
        );

        // Determine N_actual from variant (i) (clamps to BFINAL if it lands early).
        let probe_i = decode_var_i(&deflate, start_bit, window, REQUESTED_N);
        let n_actual = probe_i.len().min(REQUESTED_N);
        assert!(
            n_actual >= 64 * 1024,
            "decoded only {n_actual} bytes — chunk too short, pick a deeper entry"
        );

        // Byte-exactness gate (the ABSOLUTE requirement).
        let out_i = decode_var_i(&deflate, start_bit, window, n_actual);
        let out_ii = decode_var_ii(&deflate, start_bit, window, n_actual);
        let out_iii = decode_var_iii(&deflate, start_bit, window, n_actual);
        let a = &out_i[..n_actual];
        let b = &out_ii[..n_actual];
        let c = &out_iii[..n_actual];

        let eq_i_ii = a == b;
        let eq_i_iii = a == c;
        let all_equal = eq_i_ii && eq_i_iii;

        if !all_equal {
            eprintln!("BYTE-EXACT FAILURE — outputs diverge:");
            eprintln!(
                "  crc i={:#010x} ii={:#010x} iii={:#010x}",
                crc32(a),
                crc32(b),
                crc32(c)
            );
            let first_diff = |x: &[u8], y: &[u8]| x.iter().zip(y).position(|(p, q)| p != q);
            if !eq_i_ii {
                eprintln!("  (i)vs(ii) first diff at {:?}", first_diff(a, b));
            }
            if !eq_i_iii {
                eprintln!("  (i)vs(iii) first diff at {:?}", first_diff(a, c));
            }
        }

        // Warm up once (discarded).
        let _ = decode_var_i(&deflate, start_bit, window, n_actual);
        let _ = decode_var_ii(&deflate, start_bit, window, n_actual);
        let _ = decode_var_iii(&deflate, start_bit, window, n_actual);

        // Interleaved best-of-N: i, ii, iii, i, ii, iii, ...
        let mut t_i = Vec::with_capacity(ITERS);
        let mut t_ii = Vec::with_capacity(ITERS);
        let mut t_iii = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let s = Instant::now();
            let r = decode_var_i(&deflate, start_bit, window, n_actual);
            t_i.push(s.elapsed().as_secs_f64());
            std::hint::black_box(&r);

            let s = Instant::now();
            let r = decode_var_ii(&deflate, start_bit, window, n_actual);
            t_ii.push(s.elapsed().as_secs_f64());
            std::hint::black_box(&r);

            let s = Instant::now();
            let r = decode_var_iii(&deflate, start_bit, window, n_actual);
            t_iii.push(s.elapsed().as_secs_f64());
            std::hint::black_box(&r);
        }

        let (min_i, med_i, sig_i) = stats(&t_i);
        let (min_ii, med_ii, sig_ii) = stats(&t_ii);
        let (min_iii, med_iii, sig_iii) = stats(&t_iii);

        let mbps = |secs: f64| (n_actual as f64) / secs / 1e6;
        let mb_i = mbps(med_i);
        let mb_ii = mbps(med_ii);
        let mb_iii = mbps(med_iii);

        // Ratios on median MB/s.
        let r_ii_i = mb_ii / mb_i;
        let r_iii_i = mb_iii / mb_i;
        let r_ii_iii = mb_ii / mb_iii;

        // RECALIBRATED round-2: [2.5x,3.6x] (gzippy-scalar-u16 vs pure-ISA-L is
        // a purer denominator than the 2.1-2.38x system wall ratio).
        let selftest = r_iii_i >= 2.5 && r_iii_i <= 3.6;

        println!("ENGINE_BENCH start_bit={start_bit} N_bytes={n_actual}");
        println!(
            "VAR_I   scalar_u16  min={:.6} med={:.6} sigma={:.1}%  MBps_med={:.0}",
            min_i, med_i, sig_i, mb_i
        );
        println!(
            "VAR_II  E1_u8(part) min={:.6} med={:.6} sigma={:.1}%  MBps_med={:.0}",
            min_ii, med_ii, sig_ii, mb_ii
        );
        println!(
            "VAR_III isal        min={:.6} med={:.6} sigma={:.1}%  MBps_med={:.0}",
            min_iii, med_iii, sig_iii, mb_iii
        );
        println!(
            "RATIO ii/i={:.3} iii/i={:.3} ii/iii={:.3}",
            r_ii_i, r_iii_i, r_ii_iii
        );
        println!(
            "SHA_ALL_EQUAL={}  SELFTEST={}",
            if all_equal { "yes" } else { "no" },
            if selftest { "PASS" } else { "FAIL" }
        );
        println!(
            "CRC i={:#010x} ii={:#010x} iii={:#010x}",
            crc32(a),
            crc32(b),
            crc32(c)
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
