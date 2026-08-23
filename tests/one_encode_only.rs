//! ONE ENCODE PER INPUT — the architectural guard.
//!
//! ⭐ OWNER, 2026-08-23:
//!
//!   "Why do we even have pick-min? Isn't that the approach that drove to parallel
//!    implementations which caused us to lose so much wall clock time? I told you to
//!    start with the perfect port of the vendor we're competing against and then to
//!    make optimizations that you could surpass in all cases. ... This project is
//!    named after its speed. Compression can't get worse, but that is strictly
//!    secondary."
//!
//! Whole-buffer pick-min encoded every input TWICE and kept the smaller result. It
//! cost 2.1-2.3x CLI wall at every level to defend 0.002-1.95% of size, and it is why
//! the shipped binary ran 1.6-4.1x its own libdeflate port. It has been deleted.
//!
//! This file exists so it cannot come back quietly. It COUNTS encoder entries rather
//! than asserting a predicate, because a predicate has lied about exactly this three
//! times in this campaign (`level_streams` depends on the pick-min predicate, so
//! disabling pick-min REROUTED to a duplicate dispatch instead of removing it, and a
//! whole session published "L2/L4 pick-min is free" off the back of it).
//!
//! The counters are NOT feature-gated, so this runs in the default `cargo test` suite
//! and in every CI job — a guard that only runs under a feature flag is one that gets
//! missed.

use gzippy::compress::deflate::{encode_census, encode_gzip_bytes_to_vec};
use std::sync::{Mutex, MutexGuard};

/// The census is process-global, and `cargo test` runs test fns on separate threads.
/// Without this every test here would see every other test's encodes. Any NEW test in
/// this file must take the lock too.
static CENSUS: Mutex<()> = Mutex::new(());
fn census_lock() -> MutexGuard<'static, ()> {
    CENSUS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Inputs chosen to exercise both the short-circuit paths and the real parsers:
/// below `max_passthrough_size`, across a block boundary, and comfortably multi-block.
fn inputs() -> Vec<(&'static str, Vec<u8>)> {
    let text = b"the quick brown fox jumps over the lazy dog. ".repeat(4000);
    let mut lcg: u32 = 0x1234_5678;
    let noise: Vec<u8> = (0..300_000)
        .map(|_| {
            lcg = lcg.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (lcg >> 16) as u8
        })
        .collect();
    vec![
        ("tiny", b"abcd".to_vec()),
        ("short", b"hello world, hello world".to_vec()),
        ("text", text),
        ("noise", noise),
    ]
}

/// EVERY level, at T1, must reach the encoder EXACTLY ONCE.
///
/// A second entry means a second parse of the same bytes — pick-min, a retry, or a
/// "just try it both ways" arm. That is the thing this project removed.
#[test]
fn every_level_encodes_each_input_exactly_once() {
    let _guard = census_lock();
    let mut failures = Vec::new();
    for (name, data) in inputs() {
        for level in 0..=9u32 {
            encode_census::reset();
            let out = encode_gzip_bytes_to_vec(&data, level);
            let (port, legacy) = encode_census::snapshot();
            let total = port + legacy;
            if total != 1 {
                failures.push(format!(
                    "{name} L{level}: {total} encoder entries (port={port} legacy={legacy}), \
                     expected exactly 1 — a second encode of the same input is pick-min \
                     by another name. Output was {} B.",
                    out.len()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\nMORE THAN ONE ENCODE PER INPUT — the pick-min architecture is back:\n  {}\n\n\
         This is not a tuning regression. Encoding twice and keeping the smaller output \
         costs ~2x wall on every cell to buy a fraction of a percent of size, and it is \
         what this project deleted on 2026-08-23. Fix the routing, do not relax this test.\n",
        failures.join("\n  ")
    );
}

/// The libdeflate port is the production encoder for every level it implements.
///
/// Not "may be used", not "is used at some levels" — the baseline we optimise FROM.
/// If a level silently falls back to the legacy encoder, its wall and size stop being
/// comparable to the vendor and every ours-vs-libdeflate number becomes meaningless.
#[test]
fn the_port_is_the_production_encoder_for_levels_0_through_9() {
    let _guard = census_lock();
    let data = b"the quick brown fox jumps over the lazy dog. ".repeat(4000);
    let mut failures = Vec::new();
    for level in 0..=9u32 {
        encode_census::reset();
        let _ = encode_gzip_bytes_to_vec(&data, level);
        let (port, legacy) = encode_census::snapshot();
        if !(port == 1 && legacy == 0) {
            failures.push(format!(
                "L{level}: port={port} legacy={legacy}, expected port=1 legacy=0"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "\nA level 0-9 did NOT route to the libdeflate port:\n  {}\n\n\
         The port is the baseline this project optimises from (owner, 2026-08-23). A \
         level that falls back to the legacy encoder is a level where we cannot say \
         whether we beat libdeflate.\n",
        failures.join("\n  ")
    );
}
