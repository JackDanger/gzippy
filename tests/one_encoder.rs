//! ONE ENCODER: the libdeflate port owns levels 0-9.
//!
//! The repo carried TWO implementations of the same algorithm — `src/compress/ldx`
//! (7,687 lines, a per-decision transliteration of
//! `vendor/libdeflate/lib/deflate_compress.c`, byte-identical to the C) and
//! `src/compress/deflate` (27,266 lines, hand-grown). The exact one was used ONLY as a
//! test oracle while the hand-grown one shipped and ran 1.6-4.1x slower.
//!
//! The hand-grown encoder's advantage turned out to be FOUR PARAMETERS, not 27k lines:
//! L4 routed to lazy instead of greedy, and zlib's chain depths (32/128/256) at L5/L6/L7
//! instead of libdeflate's (16/35/100). Those now live on the port.
//!
//! This test is the SEVERANCE. The legacy path is still compiled because levels 10-12
//! have no libdeflate counterpart, but nothing at 0-9 may reach it, and CI fails if
//! anything does.
#![cfg(feature = "anatomy-counters")]

use gzippy::compress::deflate::anatomy_counters::COUNTERS;

#[test]
fn legacy_encoder_is_unreachable_below_level_10() {
    let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut inputs: Vec<Vec<u8>> = vec![
        b"".to_vec(),
        b"a".to_vec(),
        vec![0u8; 100_000],
        (0..200_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect(),
    ];
    if let Ok(rd) = std::fs::read_dir(&corpus) {
        for e in rd.flatten().take(6) {
            if let Ok(d) = std::fs::read(e.path()) {
                if !d.is_empty() {
                    inputs.push(d);
                }
            }
        }
    }

    for level in 0..=9u32 {
        for input in &inputs {
            let before = COUNTERS.snapshot().legacy_encoder_entries;
            let _ = gzippy::compress::deflate::encode_gzip_bytes_to_vec(input, level);
            let after = COUNTERS.snapshot().legacy_encoder_entries;
            assert_eq!(
                before,
                after,
                "LEGACY ENCODER REACHED at level {level} on a {}-byte input.\n\
                 Levels 0-9 belong to the libdeflate port (`compress::ldx`). If a level \n\
                 genuinely needs the legacy path again, that is a deliberate un-collapse: \n\
                 say so in the commit and narrow `level_uses_ldx`, do not delete this test.",
                input.len()
            );
        }
    }
}
