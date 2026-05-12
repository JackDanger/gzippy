#[test]
fn smoke_round_trip() {
    let data = b"hello, gzippy library API!".repeat(100);
    let compressed = gzippy::deflate::encode(&data, 6).unwrap();
    let decompressed = gzippy::deflate::decode(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn smoke_empty() {
    let compressed = gzippy::deflate::encode(b"", 6).unwrap();
    let decompressed = gzippy::deflate::decode(&compressed).unwrap();
    assert!(decompressed.is_empty());
}
