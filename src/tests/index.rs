//! Tests for indexed/random-access decompression.

use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;

#[test]
fn test_build_and_seek_random_data() {
    // Create test data with better compression characteristics
    let data: Vec<u8> = (0..200 * 1024)
        .map(|i| ((i ^ (i >> 8) ^ (i >> 16)) % 256) as u8)
        .collect();

    // Compress it
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index (note: checkpoints are at deflate block boundaries, not fixed intervals)
    let index = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index");

    // Index might have 0 or more checkpoints depending on deflate block boundaries
    assert_eq!(
        index.total_uncompressed_size,
        data.len() as u64,
        "Index total size mismatch"
    );

    // Verify checkpoints are monotonic
    for i in 1..index.points.len() {
        assert!(
            index.points[i].uncompressed_offset > index.points[i - 1].uncompressed_offset,
            "Checkpoints should be monotonically increasing"
        );
    }
}

#[test]
fn test_seek_at_first_checkpoint() {
    // Create test data: 100KB
    let data: Vec<u8> = (0..100 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index");

    if let Some(first_checkpoint) = index.points.first() {
        let offset = first_checkpoint.uncompressed_offset;
        let mut output = Vec::new();
        crate::decompress::index::seek_decompress(
            &compressed,
            &index,
            offset,
            u64::MAX,
            &mut output,
        )
        .expect("Seek decompress failed");

        let expected = &data[offset as usize..];
        assert_eq!(
            &output[..],
            expected,
            "Seek to first checkpoint should match"
        );
    }
}

#[test]
fn test_seek_at_second_checkpoint() {
    // Create test data: 200KB
    let data: Vec<u8> = (0..200 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 30 * 1024)
        .expect("Failed to build index");

    if index.points.len() >= 2 {
        let offset = index.points[1].uncompressed_offset;
        let mut output = Vec::new();
        crate::decompress::index::seek_decompress(
            &compressed,
            &index,
            offset,
            u64::MAX,
            &mut output,
        )
        .expect("Seek decompress failed");

        let expected = &data[offset as usize..];
        assert_eq!(
            &output[..],
            expected,
            "Seek to second checkpoint should match"
        );
    }
}

#[test]
fn test_index_serialize_roundtrip() {
    // Create test data: 50KB
    let data: Vec<u8> = (0..50 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index1 = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index");

    // Serialize
    let mut serialized = Vec::new();
    crate::decompress::index::serialize_index(&index1, &mut serialized)
        .expect("Failed to serialize index");

    // Deserialize
    let index2 = crate::decompress::index::load_index(&serialized).expect("Failed to load index");

    // Verify they match
    assert_eq!(
        index1.total_uncompressed_size, index2.total_uncompressed_size,
        "Total size mismatch"
    );
    assert_eq!(
        index1.deflate_offset, index2.deflate_offset,
        "Deflate offset mismatch"
    );
    assert_eq!(
        index1.points.len(),
        index2.points.len(),
        "Point count mismatch"
    );

    for (i, (p1, p2)) in index1.points.iter().zip(index2.points.iter()).enumerate() {
        assert_eq!(
            p1.compressed_bit_offset, p2.compressed_bit_offset,
            "Point {} compressed offset mismatch",
            i
        );
        assert_eq!(
            p1.uncompressed_offset, p2.uncompressed_offset,
            "Point {} uncompressed offset mismatch",
            i
        );
        assert_eq!(
            &p1.window[..],
            &p2.window[..],
            "Point {} window mismatch",
            i
        );
    }
}

#[test]
fn test_build_index_multi_member() {
    // Create two separate gzip members
    let part1: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();
    let part2: Vec<u8> = (0..50_000).map(|i| ((i + 50) % 256) as u8).collect();

    let mut enc1 = GzEncoder::new(Vec::new(), Compression::default());
    enc1.write_all(&part1).unwrap();
    let compressed1 = enc1.finish().unwrap();

    let mut enc2 = GzEncoder::new(Vec::new(), Compression::default());
    enc2.write_all(&part2).unwrap();
    let compressed2 = enc2.finish().unwrap();

    let mut multi = compressed1;
    multi.extend_from_slice(&compressed2);

    // Should reject multi-member
    let result = crate::decompress::index::build_index(&multi, 10 * 1024);
    assert!(result.is_err(), "Should reject multi-member gzip files");
}

#[test]
fn test_seek_with_max_bytes_limit() {
    // Create test data: 100KB
    let data: Vec<u8> = (0..100 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index");

    if let Some(checkpoint) = index.points.first() {
        let seek_offset = checkpoint.uncompressed_offset;
        let max_bytes = 10 * 1024;
        let mut output = Vec::new();
        let bytes_read = crate::decompress::index::seek_decompress(
            &compressed,
            &index,
            seek_offset,
            max_bytes as u64,
            &mut output,
        )
        .expect("Seek decompress failed");

        assert_eq!(
            bytes_read, max_bytes as u64,
            "Should read exactly max_bytes"
        );
        assert_eq!(
            output.len(),
            max_bytes,
            "Output should be max_bytes in size"
        );

        let expected_offset = seek_offset as usize;
        let expected = &data[expected_offset..expected_offset + max_bytes];
        assert_eq!(&output[..], expected, "Seek with limit output mismatch");
    }
}

#[test]
fn test_seek_offset_equals_checkpoint() {
    // Create test data: 150KB
    let data: Vec<u8> = (0..150 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 20 * 1024)
        .expect("Failed to build index");

    // For each checkpoint, verify seeking to its exact offset works
    for checkpoint in index.points.iter() {
        let mut output = Vec::new();
        crate::decompress::index::seek_decompress(
            &compressed,
            &index,
            checkpoint.uncompressed_offset,
            u64::MAX,
            &mut output,
        )
        .expect("Seek decompress failed");

        let expected_offset = checkpoint.uncompressed_offset as usize;
        let expected = &data[expected_offset..];
        assert_eq!(
            &output[..],
            expected,
            "Seek to checkpoint {} should match",
            checkpoint.uncompressed_offset
        );
    }
}

#[test]
fn test_seek_from_first_checkpoint_to_end() {
    // Create test data
    let data: Vec<u8> = (0..50 * 1024)
        .map(|i| ((i ^ (i >> 8)) % 256) as u8)
        .collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index");

    if let Some(first_checkpoint) = index.points.first() {
        // Seek from first checkpoint
        let seek_offset = first_checkpoint.uncompressed_offset;
        let mut output = Vec::new();
        crate::decompress::index::seek_decompress(
            &compressed,
            &index,
            seek_offset,
            u64::MAX,
            &mut output,
        )
        .expect("Seek to first checkpoint failed");

        // Should match data from that offset to the end
        let expected = &data[seek_offset as usize..];
        assert_eq!(
            &output[..],
            expected,
            "Seeking to first checkpoint should decompress rest of file"
        );
    }
}
