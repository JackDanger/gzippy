//! Integration tests for gzippy's raw DEFLATE API (`deflate_encode` / `deflate_decode`).
//!
//! These tests verify the named entry-points used by 7zippy's Deflate coder and
//! any other consumer that embeds raw DEFLATE streams without a gzip envelope.

#[test]
fn round_trip_raw_deflate() {
    let data = b"hello, raw deflate!".repeat(1000);
    for level in [1, 6, 9] {
        let compressed = gzippy::deflate_encode(&data, level).unwrap();
        let decompressed = gzippy::deflate_decode(&compressed).unwrap();
        assert_eq!(decompressed, data, "level {level}");
    }
}
