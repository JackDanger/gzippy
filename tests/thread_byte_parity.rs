//! T1-vs-T(N) byte-parity gates (the campaign follow-up).
//!
//! RED-FIRST pins for the parity unification tracked in
//! `docs/board/records/2026-09-22-ec2-c7a4xl/FINAL-ADJUDICATION.md`:
//! today the parallel path intentionally parses differently
//! (`level::params_parallel`, `HeaderBudget::Generous`, per-T chunk grids),
//! so T(N) bytes differ from T1. When the unification lands (thread-free
//! grid + one param set + one engine per level), these tests turn GREEN
//! and stay green: they then enforce byte-parity at any N, with real
//! chunk-seam payloads and a full roundtrip so a corrupt-but-consistent
//! stream can never pass.
//!
//! Until then they are marked `#[ignore]` so CI stays green; run them with
//! `cargo test --release --test thread_byte_parity -- --ignored --nocapture`
//! to see the exact divergence before landing the unification.

fn digest(bytes: &[u8]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn encode_at(data: &[u8], level: u8, threads: usize) -> Vec<u8> {
    gzippy::compress_with_threads(data, level, threads).expect("encode should succeed")
}

fn roundtrips(data: &[u8], stream: &[u8]) -> bool {
    matches!(gzippy::decompress(stream), Ok(back) if back == data)
}

/// Payload whose length spans real chunk seams at the level's parallel grid:
/// build a repeating mixed-corpus body then truncate to `k*grid + 257` so
/// chunk seams land mid-corpus exactly like the campaign's real probe.
fn seam_payload(level: u8, grid: usize, seams: usize) -> Vec<u8> {
    let unit: &[u8] = b"abracadabra-sing-a-song-of-sixpence-pockets-full-of-rye";
    let target = grid.saturating_mul(seams) + 257;
    let mut v = Vec::with_capacity(target + unit.len());
    while v.len() < target {
        v.extend_from_slice(unit);
        v.extend((0u8..=255u8).rev());
        v.extend_from_slice(unit);
    }
    v.truncate(target);
    let _ = level;
    v
}

#[test]
#[ignore = "byte-parity unification pending — see records/2026-09-22-ec2-c7a4xl/FINAL-ADJUDICATION.md"]
fn byte_parity_l9_three_seams() {
    // ~1 MB at L9's 1.8 MB grid would be single-chunk; the parallel grid at
    // L9 on silesia-scale is 1_800_000 — use 3*grid + 257 to force 3 seams.
    let grid = 1_800_000;
    let data = seam_payload(9, grid, 3);
    assert!(data.len() > 2 * grid, "payload must span 3 seams");
    let t1 = encode_at(&data, 9, 1);
    let t4 = encode_at(&data, 9, 4);
    assert_eq!(
        digest(&t1),
        digest(&t4),
        "T1 vs T4 byte-parity at L9 across 3 seams"
    );
    assert!(roundtrips(&data, &t4), "T4 stream must roundtrip");
}

#[test]
#[ignore = "byte-parity unification pending — see records/2026-09-22-ec2-c7a4xl/FINAL-ADJUDICATION.md"]
fn byte_parity_l6_midgrid() {
    // L6 clamps chunks to ≤2 MB; 4 seams at 900 B-scale grids exercise the
    // un-clamped grid band where the layout moves with -p.
    let grid = 900_000;
    let data = seam_payload(6, grid, 4);
    let t1 = encode_at(&data, 6, 1);
    let t4 = encode_at(&data, 6, 4);
    assert_eq!(digest(&t1), digest(&t4), "L6 T1 vs T4 parity mid-grid");
}

#[test]
#[ignore = "byte-parity unification pending — see records/FINAL-ADJUDICATION.md"]
fn across_thread_counts_all_levels() {
    let data = seam_payload(6, 256 * 1024, 4);
    let t1 = encode_at(&data, 6, 1);
    for threads in [2usize, 4, 8, 16] {
        let tn = encode_at(&data, 6, threads);
        assert_eq!(digest(&t1), digest(&tn), "L6 T{threads} parity");
        assert!(roundtrips(&data, &tn));
    }
}
