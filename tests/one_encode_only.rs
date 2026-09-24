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
use gzippy::compress::pipelined::PipelinedGzEncoder;
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
///
/// ⭐ WITH FOUR MEASURED EXCEPTIONS (L1, L3, L6, L7), each with a named gate and a
/// path back to the port — see `level_uses_ldx` in `src/compress/deflate/mod.rs`
/// for the full measurement record. Routing all of 0-9 to the port (`b28e96f3`)
/// went red on the per-commit ledger immediately and stayed red for 45 commits:
/// L1 loses to pigz on the `fast_l1_ratio_multi_corpus` cell, and L6 regresses
/// FOUR `won_cells_stay_won` cells (binary vs gzip +1,614 B / vs pigz +887 B;
/// text vs gzip +12,610 B / vs pigz +12,090 B). L3 is a different class: the port
/// has no len-3/sparse machinery, so its L3 is 1-7% LARGER than the legacy L3
/// (the campaign-winning L3 guards) and lost 11 T1 wall cells in the 2026-09-01
/// solvency try. The port L6/L7 collapse the moment it learns the `good_match`
/// knob (PR #363, verified byte-identical 11/11); the port L3 collapses the
/// moment it learns the len-3 machinery. Until then these levels stay on the
/// measured-best legacy config.
const PORT_EXCEPTIONS: &[u32] = &[1, 3, 6, 7];

#[test]
fn the_port_is_the_production_encoder_for_levels_0_through_9() {
    let _guard = census_lock();
    let data = b"the quick brown fox jumps over the lazy dog. ".repeat(4000);
    let mut failures = Vec::new();
    for level in 0..=9u32 {
        encode_census::reset();
        let _ = encode_gzip_bytes_to_vec(&data, level);
        let (port, legacy) = encode_census::snapshot();
        let (exp_port, exp_legacy) = if PORT_EXCEPTIONS.contains(&level) {
            (0, 1)
        } else {
            (1, 0)
        };
        if !(port == exp_port && legacy == exp_legacy) {
            failures.push(format!(
                "L{level}: port={port} legacy={legacy}, expected port={exp_port} legacy={exp_legacy}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "\nA level 0-9 did NOT route to its expected encoder:\n  {}\n\n\
         The port is the baseline this project optimises from (owner, 2026-08-23), \
         with exactly the measured exceptions named in PORT_EXCEPTIONS above. A level \
         that routes somewhere unexpected is a routing regression; a new exception \
         needs a measured receipt in `level_uses_ldx` before it is added here.\n",
        failures.join("\n  ")
    );
}

/// The LIBRARY T1 route must emit the same bytes the CLI production encoder
/// does, at every level 0-9. Codex's review of the structure slice named the
/// leak: `PipelinedGzEncoder::new(level, 1).compress_exact_to_writer` called
/// `ldx::compress_into` at every level without consulting `level_uses_ldx`,
/// so a library caller at L6 got port bytes (322,110 B on the text fixture)
/// where the ledger-won production bytes are 304,252 B — the pick-min win
/// silently forfeited one lib route at a time.
///
/// This compares against `encode_gzip_bytes_to_vec`'s ten-byte header +
/// trailer too: the production T1 bytes include the flat-XFL contract
/// (`16de4ed6`), while the port levels carry the vendor XFL form — which is
/// what makes the two bodies identical-but-headers-different at some levels.
/// Compare DEFLATE BODIES, not the whole stream.
#[test]
fn the_1_processor_emits_production_bytes_at_every_level() {
    let _guard = census_lock();
    // 1 MiB so the grid is multi-block at every level; fixtures::generate is
    // too small to exercise pick-free routing past the passthrough cap.
    let data = gzippy::fixtures::generate_sized("text", 1 << 20);
    for level in 0..=9u32 {
        let expected = encode_gzip_bytes_to_vec(&data, level);
        let mut actual = Vec::new();
        let mut encoder = PipelinedGzEncoder::new(level, 1);
        encoder.set_minimal_gzip_header(true);
        encoder
            .compress_buffer_pure(&data, &mut actual)
            .expect("pipeline T1 encode");
        // The pipeline's minimal header is byte-identical to the production
        // T1 route's header on this branch (both flat-XFL); if they go
        // non-identical, this assert names the level and both lengths so the
        // next reader does not binary-search routing.
        if !(expected[10..expected.len() - 8] == actual[10..actual.len() - 8]
            && expected[..10] == actual[..10])
        {
            panic!(
                "L{level}: library T1 route emitted different bytes from the production \
                 T1 route: production {} B vs pipeline {} B",
                expected.len(),
                actual.len()
            );
        }
    }
}
