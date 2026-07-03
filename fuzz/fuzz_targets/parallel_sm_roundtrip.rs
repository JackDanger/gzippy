#![no_main]
//! cargo-fuzz target for the pure-Rust parallel single-member (ParallelSM)
//! decode path.
//!
//! Strategy (two arms per input, both exercising `parallel_sm`):
//!
//!   ARM 1 — round-trip correctness. Compress the fuzz bytes with flate2
//!   (zlib-ng), an INDEPENDENT gzip encoder, then decompress with gzippy at
//!   T=8 and assert byte-exact recovery of the original input. This is the
//!   real correctness oracle: a mismatch or a decode error on a well-formed
//!   gzip stream is a LOUD bug.
//!
//!   ARM 2 — malformed-input robustness. Feed the raw fuzz bytes straight to
//!   the decoder (most are rejected as non-gzip). We only require it does not
//!   PANIC — an `Err` is fine, a panic/abort is a bug.
//!
//! The three `GZIPPY_PARALLEL_*` caps are disabled once at process start so
//! the small fuzz inputs still route through the parallel pipeline instead of
//! the small-output serial floor — i.e. we fuzz the parallel machinery
//! (chunk_decode / marker_inflate / apply_window / replace_markers), not the
//! T1 inline path.

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use std::sync::Once;

static INIT: Once = Once::new();

fn force_parallel_env() {
    INIT.call_once(|| {
        // Disable the compressibility cap, the crossover selector, and the
        // small-output serial floor so ParallelSM runs at the requested T.
        std::env::set_var("GZIPPY_PARALLEL_RATIO_MAX", "0");
        std::env::set_var("GZIPPY_PARALLEL_CROSSOVER_MARGIN", "0");
        std::env::set_var("GZIPPY_PARALLEL_MIN_OUTPUT_BYTES", "0");
    });
}

fn gzip_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fuzz_target!(|data: &[u8]| {
    force_parallel_env();

    // Cap decoded work: derive the compression level from the first byte and
    // keep the plaintext bounded so the corpus stays fast to replay.
    if data.len() > 8 * 1024 * 1024 {
        return;
    }
    let level = if data.is_empty() {
        6
    } else {
        (data[0] % 10) as u32
    };
    let payload = if data.is_empty() { &[][..] } else { &data[1..] };

    // ARM 1: independent-encoder round-trip, decoded on the parallel path.
    let gz = gzip_compress(payload, level);
    match gzippy::decompress_with_threads(&gz, 8) {
        Ok(out) => {
            assert!(
                out == payload,
                "ROUND-TRIP MISMATCH: len_in={} len_out={} level={}",
                payload.len(),
                out.len(),
                level
            );
        }
        Err(e) => {
            panic!(
                "decode of a well-formed gzip stream FAILED: {e:?} (len_in={}, level={})",
                payload.len(),
                level
            );
        }
    }

    // ARM 2: raw fuzz bytes as (mostly malformed) gzip — must not panic.
    // Restrict to inputs the router sends to the parallel single-member path
    // (ParallelSM / StoredParallel) so this target fuzzes ITS path, not the
    // multi-member sequential decoder (whose growth-loop OOM on crafted input
    // is a separate, documented finding outside this target's scope).
    match gzippy::classify(data, 8) {
        gzippy::DecodePath::ParallelSM | gzippy::DecodePath::StoredParallel => {
            let _ = gzippy::decompress_with_threads(data, 8);
        }
        _ => {}
    }
});
