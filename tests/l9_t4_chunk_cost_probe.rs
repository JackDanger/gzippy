//! Probe 1 (final-lap lever #364 investigation, agent-16): the L9/T>1
//! config-cost matrix.
//!
//! Times the EXACT production chunk entry (`encode_deflate_splice_chunk_to_sink`)
//! over one 1.8 MB chunk + 32 KiB prefix dictionary, for the variants that
//! discriminate the pigz-L9/T4-loss hypotheses:
//!
//! 1. production: `parallel=true, level=9`  → `params_parallel(9)` ≡
//!    `params_parallel(11)` with the trunk's L9 retune: near-optimal, depth
//!    400-alias knobs, **passes 2** (level.rs `max_optim_passes = 2`).
//! 2. T1 engine:  `parallel=false, level=9` → `params(9)` (Lazy2, depth 600,
//!    nice 258 — the ldx-shaped parse the T1 cells ship).
//! 3. control:    `parallel=true, level=11` (L11 default knobs, **passes 4** —
//!    since the retune this NO LONGER aliases variant 1: wall(1)/wall(3)
//!    now prices the passes 2 vs 4 delta).
//! 4. control:    `parallel=false, level=11`.
//!
//! READS: best-of-k wall via std::time::Instant + output bytes (the size
//! ledger side; the near-opt upgrade was *paid for* in size).
//!
//! DISCRIMINATION (docs/board/sprint-2026-09-25.md):
//!   wall(1)/wall(2) >= 2.5  → the near-opt upgrade is the multiplier
//!                             (agent-16 predicted 3.5-4.5x from 61f0f01d)
//!   wall(1) ~ wall(2)       → REFUTED; escalate to scheduler phase accounting
//!   wall(1)/wall(3)         → the passes-2 retune's own chunk-level price.
//!
//! Measurement only (`#[ignore]`), never a regression gate.
//!
//! The `parallel_flush_probes` module below (feature `near-opt-parallel-flush`,
//! lever ledger row 3) adds the flush-dispatch variants to this matrix:
//! serial-vs-parallel flush timing on the probe corpus, and the
//! whole-chunk byte-identity gate serial-vs-pool across the probe corpus and
//! the four campaign corpora at the L11-T1 and L9-T4 shapes.

use std::time::Instant;

const CHUNK: usize = 1_800_000;
const DICT: usize = 32 * 1024;

fn build_corpus() -> Vec<u8> {
    if let Ok(path) = std::env::var("PROBE_CORPUS") {
        // real-corporus mode: the file must be at least DICT + CHUNK + 4096
        if let Ok(bytes) = std::fs::read(&path) {
            return bytes;
        }
    }

    // non-periodic deterministic corpus: interleave the four frozen fixtures
    let mut base = Vec::new();
    for name in gzippy::fixtures::NAMES {
        base.extend(gzippy::fixtures::generate(name));
    }
    let mut cursor = [0usize; 4];
    let target = CHUNK + DICT + 4096;
    while base.len() < target {
        for (i, name) in gzippy::fixtures::NAMES.iter().enumerate() {
            let piece = gzippy::fixtures::generate(name);
            let start = cursor[i].min(piece.len());
            let take = (piece.len() / 8)
                .max(4096)
                .min(piece.len().saturating_sub(start));
            base.extend_from_slice(&piece[start..start + take]);
            cursor[i] = (start + take) % piece.len().max(1);
        }
    }
    base
}

fn run(
    label: &str,
    body: &[u8],
    dict: &[u8],
    level: u8,
    parallel: bool,
    input_total_len: usize,
    runs: usize,
) -> (f64, usize) {
    let mut times = Vec::with_capacity(runs);
    let mut bytes = 0usize;
    for _ in 0..2 {
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let _ = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            level.into(),
            true,
            &mut out,
            parallel,
            input_total_len,
        );
    }
    for _ in 0..runs {
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let t = Instant::now();
        let _ = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            level.into(),
            true,
            &mut out,
            parallel,
            input_total_len,
        );
        times.push(t.elapsed().as_secs_f64());
        bytes = out.len();
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times[0];
    println!("{label}: best {best:.4}s bytes {bytes}");
    (best, bytes)
}

#[ignore]
fn matrix_for(level: u8, parallel: bool, label: &'static str, runs: usize) {
    let data = build_corpus();
    let body = &data[DICT..DICT + CHUNK];
    let dict = &data[..DICT];
    run(label, body, dict, level, parallel, 200 * 1024 * 1024, runs);
}

#[test]
#[ignore]
fn prod_l9t4_depth400() {
    matrix_for(9, true, "1 production (near-opt@L11, depth400)", 5);
}

#[test]
#[ignore]
fn t1_engine_lazy2_depth600() {
    matrix_for(9, false, "2 T1 engine (Lazy2 depth600)", 5);
}

#[test]
#[ignore]
fn l11_alias_check() {
    matrix_for(11, true, "3 control (L11 parallel=true alias)", 5);
}

#[test]
#[ignore]
fn depth_split_d100() {
    matrix_for(9, true, "4 nearoptimal:100:150 (depth100 ladder)", 3);
}

#[test]
#[ignore]
fn depth_split_d400_p1() {
    matrix_for(9, true, "5 nearoptimal:400:150:1 (depth400, passes1)", 3);
}

#[test]
#[ignore]
fn depth_split_d400_p4() {
    matrix_for(
        9,
        true,
        "6 nearoptimal:400:150:4 (L11 default, passes 4)",
        3,
    );
}

// ─────────────────────────────────────────────────────────────────────────
// LEVER #3 (feature `near-opt-parallel-flush`, DEFAULT OFF): the block-
// parallel optimize_and_flush variants. Without the feature this module
// compiles to nothing and the matrix above is the whole file.
// ─────────────────────────────────────────────────────────────────────────
#[cfg(feature = "near-opt-parallel-flush")]
mod parallel_flush_probes {
    use super::*;
    use gzippy::compress::deflate::parse::near_opt_flush_probe;

    /// Deterministic incompressible noise (own LCG, mirroring the bitstream
    /// tests' fixture style). Appended as a small tail to one identity
    /// corpus so the CHUNK's final partial block exercises the STORED
    /// replay path (`BitWriter::append_fragment`'s BTYPE=00 arm) inside the
    /// real pipeline, not just the unit test.
    fn noise(n: usize) -> Vec<u8> {
        let mut s: u32 = 0x5EED_1234;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            out.push((s >> 24) as u8);
        }
        out
    }

    /// The L9 production chunk shape (the exact `matrix_for` entry): chunk
    /// + 32 KiB dictionary, `parallel=true`. `pool` toggles the
    /// near-opt flush pool — with it off this IS the serial reference.
    fn encode_l9t4(
        body: &[u8],
        dict: &[u8],
        input_total_len: usize,
        pool: bool,
    ) -> (Vec<u8>, u8, bool) {
        near_opt_flush_probe::set_force_serial_flush(!pool);
        let mut out = Vec::with_capacity(body.len() / 2 + 1024);
        let meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
            body,
            dict,
            9,
            true,
            &mut out,
            true,
            input_total_len,
        );
        (out, meta.pad_bits, meta.needs_alignment)
    }

    /// The L11-T1 whole-buffer shape (the T1 route through the same
    /// near-optimal parser).
    fn encode_l11t1(data: &[u8], pool: bool) -> Vec<u8> {
        near_opt_flush_probe::set_force_serial_flush(!pool);
        gzippy::compress::deflate::encode_deflate_bytes_to_vec(data, 11)
    }

    /// Variant 7 of the matrix: the SAME production chunk through the SAME
    /// entry, serial flush vs block-parallel flush pool, best-of-5 each,
    /// plus the byte-identity verdict between the two arms.
    #[test]
    #[ignore]
    fn flush_serial_vs_parallel() {
        let data = build_corpus();
        let body = &data[DICT..DICT + CHUNK];
        let dict = &data[..DICT];
        // Two warmups so the pool spawn and the flush workers' first-touch
        // allocations land outside the measured window (the same shape the
        // `run` helper below uses).
        let (ref_bytes, ref_pad, ref_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
        let (par_bytes, par_pad, par_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
        let identical = ref_bytes == par_bytes && ref_pad == par_pad && ref_align == par_align;

        let mut serial_times = Vec::new();
        let mut parallel_times = Vec::new();
        let runs = 5;
        for _ in 0..runs {
            let t = Instant::now();
            let _ = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
            serial_times.push(t.elapsed().as_secs_f64());
        }
        for _ in 0..runs {
            let t = Instant::now();
            let _ = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
            parallel_times.push(t.elapsed().as_secs_f64());
        }
        serial_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        parallel_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (serial, parallel) = (serial_times[0], parallel_times[0]);
        println!(
            "7 parallel-flush serial   : best {serial:.4}s bytes {}",
            ref_bytes.len()
        );
        println!(
            "7 parallel-flush parallel : best {parallel:.4}s bytes {} (ratio {:.3}x)",
            par_bytes.len(),
            parallel / serial
        );
        println!(
            "7 parallel-flush identity : {}",
            if identical {
                "BYTE-IDENTICAL (whole chunk)"
            } else {
                "DIVERGED"
            }
        );
        assert!(
            identical,
            "flush pool must not move a single bit of the chunk stream"
        );
    }

    /// Whole-stream byte identity, serial reference vs flush pool, at the
    /// L11-T1 and L9-T4 shapes, on the probe corpus plus the four mission
    /// corpora (the campaign files fall back to a note when the local
    /// checkout does not carry them — the synthetic probe corpus is always
    /// present). Also proves, per parallel arm, that the pool actually
    /// engaged (`parallel_flush_writes` moves) and the stale-flag guard
    /// never fired, then exercises the guard's fallback half by latching
    /// it and re-checking serial bytes.
    #[test]
    fn chunks_are_byte_identical_to_the_serial_stream() {
        // The four campaign corpora, sliced like the production chunk probe
        // (32 KiB dict + chunk body). silesia.tar / software.archive are
        // campaign-local fixtures and may be absent on other checkouts;
        // text-1MB / alice are tracked.
        let mut corpora: Vec<(String, Vec<u8>)> = Vec::new();
        {
            let data = build_corpus();
            let mut noisy = data.clone();
            // The final partial block of a chunk near the end is small;
            // appending incompressible noise makes its emission price
            // STORED, so the write-back crosses
            // `BitWriter::append_fragment`'s stored replay in the real
            // pipeline.
            noisy.extend_from_slice(&noise(600));
            corpora.push(("probe-corpus+noise-tail".to_string(), noisy));
        }
        for (name, path) in [
            ("test_data/text-1MB.txt", "test_data/text-1MB.txt"),
            ("test_data/alice.txt", "test_data/alice.txt"),
            (
                "benchmark_data/software.archive (2 MiB slice)",
                "benchmark_data/software.archive",
            ),
            (
                "benchmark_data/silesia.tar (2 MiB slice)",
                "benchmark_data/silesia.tar",
            ),
        ] {
            match std::fs::read(path) {
                Ok(bytes) => {
                    let take = bytes.len().min(2 * 1024 * 1024);
                    corpora.push((name.to_string(), bytes[..take].to_vec()));
                }
                Err(_) => println!("corpus unavailable, skipping: {path}"),
            }
        }

        for (name, data) in &corpora {
            // L9-T4 shape: production chunk splice, dict = first 32 KiB (the
            // exact shape `matrix_for` drives; corpora are all >= 32 KiB +
            // 4 KiB of body, so the split is total).
            let (dict, body) = data.split_at(DICT);
            // serial reference
            let (ser, ser_pad, ser_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
            let ser_l11 = encode_l11t1(data, false);
            // parallel arms, and PROVE the pool ran them
            let writes_before = near_opt_flush_probe::parallel_flush_writes();
            let (par, par_pad, par_align) = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
            let par_l11 = encode_l11t1(data, true);
            let writes_delta = near_opt_flush_probe::parallel_flush_writes() - writes_before;
            assert_eq!(
                writes_delta, 2,
                "{name}: the pool must have engaged for BOTH shapes"
            );
            assert_eq!(ser_pad, par_pad, "{name}: L9-T4 trailing pad_bits diverged");
            assert_eq!(
                ser_align, par_align,
                "{name}: L9-T4 needs_alignment diverged (stored-block detection)"
            );
            if name == "probe-corpus+noise-tail" {
                // The reviewer's positivity fix (agent-29 minor 3): the noise
                // tail exists to price a STORED block into the final
                // fragment; assert the witness rather than trusting
                // equality with itself.
                assert!(
                    ser_align,
                    "{name}: the noise tail must still price a stored block \
                     (else the stored-replay coverage silently stops firing)"
                );
            }
            assert_eq!(
                ser, par,
                "{name}: L9-T4 chunk stream diverged from the serial reference"
            );
            assert_eq!(
                ser_l11, par_l11,
                "{name}: L11-T1 whole-buffer stream diverged"
            );
            assert!(
                !near_opt_flush_probe::stale_flag_fired(),
                "{name}: the stale-flag guard fired on a campaign corpus — \
                 the zero-flip finding no longer holds here"
            );
            println!("identity ok: {name} ({} B)", data.len());
        }
        // The interior-chunk shape (`is_last = false`): the caller appends the
        // sync-flush stored marker onto the SAME writer after `run` returns,
        // exercising the packed stream's trailing pending state that the
        // final-chunk arm above does not.
        {
            let data = &corpora[0].1;
            let (dict, body) = data.split_at(DICT);
            near_opt_flush_probe::set_force_serial_flush(true);
            let mut ser_out = Vec::with_capacity(body.len() / 2 + 1024);
            let ser_meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
                body,
                dict,
                9,
                false,
                &mut ser_out,
                true,
                200 * 1024 * 1024,
            );
            near_opt_flush_probe::set_force_serial_flush(false);
            let writes_before = near_opt_flush_probe::parallel_flush_writes();
            let mut par_out = Vec::with_capacity(body.len() / 2 + 1024);
            let par_meta = gzippy::compress::deflate::encode_deflate_splice_chunk_to_sink(
                body,
                dict,
                9,
                false,
                &mut par_out,
                true,
                200 * 1024 * 1024,
            );
            assert_eq!(
                near_opt_flush_probe::parallel_flush_writes() - writes_before,
                1,
                "the pool must have engaged for the interior-chunk shape"
            );
            assert_eq!(
                ser_out, par_out,
                "interior chunk (sync-flush seam follows) diverged from serial"
            );
            assert_eq!(
                (ser_meta.pad_bits, ser_meta.needs_alignment),
                (par_meta.pad_bits, par_meta.needs_alignment),
                "interior chunk ChunkMeta diverged"
            );
        }
        // The guard's fallback half: latch the stale flag and prove the next
        // chunk's bytes are the serial ones even with the pool addressable.
        near_opt_flush_probe::set_stale_flag_for_tests(true);
        {
            let data = &corpora[0].1;
            let (dict, body) = data.split_at(DICT);
            let (ser, ..) = encode_l9t4(body, dict, 200 * 1024 * 1024, false);
            let (par, ..) = encode_l9t4(body, dict, 200 * 1024 * 1024, true);
            assert_eq!(ser, par, "vetoed runs must produce the serial bytes");
        }
        near_opt_flush_probe::set_stale_flag_for_tests(false);
    }
}
