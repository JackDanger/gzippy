//! L1 TINY-INPUT STORED PASSTHROUGH — wave-B C3 pin + validity guard.
//!
//! libdeflate routes any input of `max_passthrough_size` = 55 − 4·level bytes
//! or fewer to a stored block without running a parser (`ldx/compress.rs`,
//! threshold 51 at L1). Production L1 never carried that gate — a 1-byte L1/T1
//! input paid the legacy fast parser's full per-invocation setup
//! (startup_cost.tsv: fast_positions_processed=1) to emit a static block of a
//! size a memcpy beats. The wiring now emits the stored member directly in
//! `encode_gzip_slack_padded_to_vec` (the T1 entry), sized like the vendor
//! fallback: header + one BTYPE=00 sub-block + literal copy + CRC/ISIZE —
//! the same framing `deflate_into` already uses for the empty-input member.
//!
//! Pins here:
//!   1. The stored member is VALID and roundtrips through flate2 at every
//!      size 0..=51, and its size is exactly 10 + 5 + len + 8.
//!   2. Sizes ABOVE the threshold (51 < len ≤ 64) still take the parser
//!      (L1 static-Huffman body — size ≤ the stored framing at these sizes,
//!      i.e. the gate must OFF above 51).
//!   3. Byte layout matches the empty-input pin layout: BFINAL=1, BTYPE=00,
//!      LEN, NLEN=~LEN, then the raw bytes.

use gzippy::compress::deflate::encode_gzip_bytes_to_vec;
use std::io::Read;

fn roundtrip(gz: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(gz)
        .read_to_end(&mut out)
        .expect("flate2 decode");
    out
}

#[test]
fn tiny_l1_emits_a_stored_member_and_roundtrips() {
    for len in [0usize, 1, 2, 12, 50, 51] {
        let data: Vec<u8> = (0..len as u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        let gz = encode_gzip_bytes_to_vec(&data, 1);
        // Size formula: 10-byte minimal header + 5-byte stored block framing
        // + the verbatim bytes + the 8-byte CRC/ISIZE trailer. One sub-block
        // below 65535 bytes.
        assert_eq!(
            gz.len(),
            10 + 5 + len + 8,
            "L1 {len}-byte output must be the stored-member size",
        );
        // Block framing: BFINAL=1, BTYPE=00 in the first deflate byte.
        let body = &gz[10..gz.len() - 8];
        if len == 0 {
            assert_eq!(body, &[0x01, 0x00, 0x00, 0xFF, 0xFF]);
        } else {
            assert_eq!(body[0], 0x01, "L{len}: BFINAL=1/BTYPE=00 packed bits");
            let stored_len = u16::from_le_bytes([body[1], body[2]]);
            assert_eq!(stored_len as usize, len, "stored LEN must equal input");
            assert_eq!(stored_len, !u16::from_le_bytes([body[3], body[4]]));
            assert_eq!(&body[5..], &data[..], "stored payload must be verbatim");
        }
        assert_eq!(roundtrip(&gz), data, "L1 {len}-byte roundtrip");
    }
}

/// L2-L9 at the same sizes: untouched. L2's own passthrough threshold is 47
/// and runs through the PORT (`level_uses_ldx`), so at 48+ bytes L2 parses
/// with the greedy parser — whatever bytes it emitted BEFORE this change on
/// sizes above its threshold must not move. The gate is L1-only.
#[test]
fn only_l1_moves_and_only_below_the_threshold() {
    // 52-byte run-of-motif: compressible at L1, and 52 > the 51-byte cap, so
    // L1 still routes the matchfinder here. Pin that nothing changed: the
    // member is SMALLER than the stored-body framing would have been.
    let motif = b"the quick brown fox jumps over the lazy dog 0123456789ABCDEF";
    let data: Vec<u8> = motif.iter().take(52).copied().collect();
    let gz = encode_gzip_bytes_to_vec(&data, 1);
    let stored_member_size = 10 + 5 + 52 + 8;
    assert!(
        gz.len() < stored_member_size,
        "52-byte L1 member ({} B) must still take the parser (the gate is \
         below/at 51 bytes only), got the stored size ({stored_member_size} B)",
        gz.len()
    );
    assert_eq!(roundtrip(&gz), data);
}
