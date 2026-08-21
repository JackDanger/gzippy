//! ONE GZIP ENTRY POINT — `encode_gzip_bytes_to_vec` must equal the shipped path.
//!
//! WHY THIS EXISTS. `encode_gzip_bytes_to_vec` used to be a SECOND ENGINE. It
//! carried its own pick-min dispatch covering only the mmap levels (1/2/4) with
//! no zlib branch, so at L5-L7 the zlib pick-min silently never ran — by the
//! time the shared dispatcher was reached, the 10-byte gzip header made its
//! `bw.byte_len() == 0` guard false. Measured 2026-08-21 before the fix:
//! dickens L5 4,582,861 here vs 4,544,452 shipped (+38,409 B); access.log L6
//! +43,787 B; never smaller.
//!
//! Nothing caught it, because the whole pin lattice is levels {1,2,6,9} and the
//! only cross-entry-point check (`mod.rs` inplace_tests) uses inputs <= 4 KiB,
//! where both pick-min arms tie. FIVE suites call this function —
//! `size_invariants` (which walks L0-L9 and owns ladder monotonicity),
//! `anatomy_pins`, `perf_shape`, `startup_cost`, `anatomy_wall` — and
//! `anatomy_pins` called it "the production T1 entry point", which it was not.
//! They agreed with production only by the accident of their fixtures.
//!
//! `encode_gzip_bytes_to_vec` now delegates, so the agreement is structural.
//! This test is what keeps it that way: it uses a LARGE, multi-block, mixed
//! input (the small-input case cannot see the divergence) and EVERY level,
//! including the L5-L7 band where the accident lived.

use gzippy::compress::deflate::{encode_gzip_bytes_to_vec, encode_gzip_slack_padded_to_vec};

/// Deterministic, >1 MiB, several block boundaries, and mixed enough that the
/// pick-min arms genuinely disagree. A uniform buffer would tie and prove nothing.
fn mixed_input(min_len: usize) -> Vec<u8> {
    let phrases: [&[u8]; 4] = [
        b"the quick brown fox jumps over the lazy dog ",
        b"\x00\x01\x02\x03\x04binary\xff\xfe\xfd payload",
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        b"{\"key\":\"value\",\"n\":1234567890}",
    ];
    let mut data = Vec::with_capacity(min_len + 64);
    let mut i = 0usize;
    while data.len() < min_len {
        data.extend_from_slice(phrases[i % phrases.len()]);
        i += 1;
    }
    data
}

fn slack_padded(data: &[u8], level: u32) -> Vec<u8> {
    // 16 = parse::BUF_PAD / INPLACE_TAIL_PAD, the speculative-load slack the
    // padded entry point requires of its caller.
    let mut buf = data.to_vec();
    buf.resize(data.len() + 16, 0);
    encode_gzip_slack_padded_to_vec(&buf, data.len(), level)
}

#[test]
fn gzip_entry_points_agree_at_every_level() {
    for (label, data) in [
        ("large-mixed", mixed_input(1_500_000)),
        ("small", mixed_input(512)),
        ("empty", Vec::new()),
    ] {
        for level in 0..=12u32 {
            let via_bytes = encode_gzip_bytes_to_vec(&data, level);
            let via_padded = slack_padded(&data, level);
            assert_eq!(
                via_bytes,
                via_padded,
                "ENTRY POINTS DIVERGED on {label} at L{level}: \
                 encode_gzip_bytes_to_vec emitted {} B, the shipped padded path emitted {} B. \
                 These must be byte-identical — bytes_to_vec is a thin wrapper over the padded \
                 path, and five test suites depend on it measuring what ships.",
                via_bytes.len(),
                via_padded.len()
            );
        }
    }
}
