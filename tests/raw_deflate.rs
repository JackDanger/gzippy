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

#[test]
fn raw_deflate_no_gzip_header() {
    let data = b"check no framing";
    let compressed = gzippy::deflate_encode(data, 6).unwrap();
    // gzip header magic is 0x1F 0x8B; raw deflate must NOT start with it
    assert_ne!(&compressed[0..2], &[0x1F, 0x8B][..]);
}
