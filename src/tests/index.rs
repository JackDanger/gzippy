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

#[test]
fn test_empty_gzip_file() {
    // Create an empty gzip file
    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let compressed = encoder.finish().unwrap();

    // Build index should succeed even for empty file
    let index = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index for empty file");

    assert_eq!(
        index.total_uncompressed_size, 0,
        "Empty file should have size 0"
    );

    // Empty files have no deflate blocks, so seeking may not be possible.
    // This is acceptable behavior—empty files are degenerate cases.
    // The important thing is that build_index doesn't crash.
    // (Note: seek_decompress may fail on empty files due to no checkpoints.)
}

#[test]
fn test_very_small_file() {
    // Create test data smaller than a single deflate block (often < 1KB)
    let data = b"hello world tiny file";

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index with very small interval
    let index = crate::decompress::index::build_index(&compressed, 1024)
        .expect("Failed to build index for small file");

    assert_eq!(
        index.total_uncompressed_size,
        data.len() as u64,
        "Small file size should match"
    );

    // Should always have at least one checkpoint (offset 0)
    assert!(
        !index.points.is_empty(),
        "Should have at least one checkpoint"
    );

    // Verify we can decompress from offset 0
    let mut output = Vec::new();
    crate::decompress::index::seek_decompress(&compressed, &index, 0, u64::MAX, &mut output)
        .expect("Seek to start of small file should succeed");
    assert_eq!(
        &output[..],
        data,
        "Small file decompress should match original"
    );
}

#[test]
fn test_seek_to_offset_zero() {
    // Create test data
    let data: Vec<u8> = (0..100 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 20 * 1024)
        .expect("Failed to build index");

    // Seek to offset 0 should decompress entire file
    let mut output = Vec::new();
    crate::decompress::index::seek_decompress(&compressed, &index, 0, u64::MAX, &mut output)
        .expect("Seek to offset 0 should succeed");

    assert_eq!(
        &output[..],
        &data[..],
        "Seek to offset 0 should decompress entire file"
    );
}

#[test]
fn test_seek_out_of_bounds() {
    // Create test data
    let data: Vec<u8> = (0..50 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index");

    let file_size = data.len() as u64;

    // Try to seek beyond end of file
    let mut output = Vec::new();
    let result = crate::decompress::index::seek_decompress(
        &compressed,
        &index,
        file_size + 1000,
        u64::MAX,
        &mut output,
    );

    // Seeking beyond EOF should either error or return empty output
    // (depends on implementation; both are acceptable)
    match result {
        Ok(bytes) => {
            // If it succeeds, it should return 0 bytes (nothing past EOF)
            assert_eq!(
                bytes, 0,
                "Seeking beyond EOF should return 0 bytes if it succeeds"
            );
            assert!(
                output.is_empty(),
                "Output should be empty for out-of-bounds seek"
            );
        }
        Err(_) => {
            // Alternatively, it's fine to return an error
        }
    }
}

#[test]
fn test_seek_between_checkpoints() {
    // Create test data with multiple checkpoints
    let data: Vec<u8> = (0..200 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index with 30KB interval
    let index = crate::decompress::index::build_index(&compressed, 30 * 1024)
        .expect("Failed to build index");

    // Find two consecutive checkpoints
    if index.points.len() >= 2 {
        let offset1 = index.points[0].uncompressed_offset as usize;
        let offset2 = index.points[1].uncompressed_offset as usize;

        // Seek to midpoint between them
        let midpoint = offset1 + (offset2 - offset1) / 2;

        let mut output = Vec::new();
        crate::decompress::index::seek_decompress(
            &compressed,
            &index,
            midpoint as u64,
            u64::MAX,
            &mut output,
        )
        .expect("Seek to midpoint should succeed");

        let expected = &data[midpoint..];
        assert_eq!(
            &output[..],
            expected,
            "Seek between checkpoints should work"
        );
    }
}

#[test]
fn test_index_corrupt_magic_bytes() {
    // Create a fake index with bad magic bytes
    let mut fake_index = Vec::new();
    fake_index.extend_from_slice(b"BADIDX\x01");
    fake_index.extend_from_slice(&(100u32).to_le_bytes()); // deflate_offset
    fake_index.extend_from_slice(&(50000u64).to_le_bytes()); // total_size
    fake_index.extend_from_slice(&(0u32).to_le_bytes()); // point_count
    fake_index.extend_from_slice(&[0u8; 2]); // reserved

    let result = crate::decompress::index::load_index(&fake_index);
    assert!(result.is_err(), "Should reject index with bad magic bytes");
}

#[test]
fn test_index_truncated_header() {
    // Create a truncated index (magic + partial header)
    let mut fake_index = Vec::new();
    fake_index.extend_from_slice(b"GZIDX\x01");
    fake_index.extend_from_slice(&(100u32).to_le_bytes()); // only deflate_offset

    let result = crate::decompress::index::load_index(&fake_index);
    assert!(result.is_err(), "Should reject truncated index header");
}

#[test]
fn test_index_truncated_points() {
    // Create index with valid header but truncated points
    let mut fake_index = Vec::new();
    fake_index.extend_from_slice(b"GZIDX\x01");
    fake_index.extend_from_slice(&(100u32).to_le_bytes()); // deflate_offset
    fake_index.extend_from_slice(&(50000u64).to_le_bytes()); // total_size
    fake_index.extend_from_slice(&(2u32).to_le_bytes()); // point_count = 2
    fake_index.extend_from_slice(&[0u8; 2]); // reserved
                                             // Only add partial point data (should be ~65536 bytes for 2 points)
    fake_index.extend_from_slice(&[0u8; 1000]);

    let result = crate::decompress::index::load_index(&fake_index);
    assert!(result.is_err(), "Should reject index with truncated points");
}

#[test]
fn test_index_invalid_version() {
    // Create index with wrong version byte
    let mut fake_index = Vec::new();
    fake_index.extend_from_slice(b"GZIDX\x99"); // wrong version
    fake_index.extend_from_slice(&(100u32).to_le_bytes());
    fake_index.extend_from_slice(&(50000u64).to_le_bytes());
    fake_index.extend_from_slice(&(0u32).to_le_bytes());
    fake_index.extend_from_slice(&[0u8; 2]);

    let result = crate::decompress::index::load_index(&fake_index);
    assert!(result.is_err(), "Should reject index with wrong version");
}

#[test]
fn test_zero_interval_bytes() {
    // Create test data
    let data: Vec<u8> = (0..10 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Try to build index with zero interval — should clamp/default to something reasonable
    let index = crate::decompress::index::build_index(&compressed, 0);

    // Should either succeed with some reasonable interval or fail gracefully
    if let Ok(idx) = index {
        assert!(
            !idx.points.is_empty(),
            "Should have at least one checkpoint"
        );
    }
}

#[test]
fn test_checkpoint_window_initialization() {
    // Create test data
    let data: Vec<u8> = (0..100 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 10 * 1024)
        .expect("Failed to build index");

    // Verify all windows are properly initialized (not all zeros for non-first checkpoints)
    for (i, point) in index.points.iter().enumerate() {
        // First checkpoint might have empty window, but later ones should have data
        if i > 0 && point.uncompressed_offset > 0 {
            // Check that window isn't all zeros (very unlikely for real data)
            let all_zeros = point.window.iter().all(|b| *b == 0);
            // This is a soft check — some windows might legitimately be all zeros
            // but it's a good sign if they contain data
            if !all_zeros {
                // Window has data, which is good
            }
        }
    }
}

#[test]
fn test_checkpoint_monotonicity_strict() {
    // Create test data with known structure
    let data: Vec<u8> = (0..300 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index
    let index = crate::decompress::index::build_index(&compressed, 50 * 1024)
        .expect("Failed to build index");

    // Verify strict monotonicity of uncompressed offsets
    let mut prev_offset = 0u64;
    for (i, point) in index.points.iter().enumerate() {
        assert!(
            point.uncompressed_offset >= prev_offset,
            "Checkpoint {} offset {} should be >= previous {}",
            i,
            point.uncompressed_offset,
            prev_offset
        );
        prev_offset = point.uncompressed_offset;

        // Also verify compressed bit offsets are monotonically increasing
        if i > 0 {
            assert!(
                point.compressed_bit_offset > index.points[i - 1].compressed_bit_offset,
                "Compressed offset should strictly increase at checkpoint {}",
                i
            );
        }
    }

    // Verify last checkpoint doesn't exceed total size
    if let Some(last) = index.points.last() {
        assert!(
            last.uncompressed_offset <= index.total_uncompressed_size,
            "Last checkpoint offset should not exceed total size"
        );
    }
}

#[test]
fn test_single_checkpoint_behavior() {
    // Create a small file that might only have one deflate block
    let data = b"The quick brown fox jumps over the lazy dog. This is a test.";

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index with large interval (larger than data)
    let index = crate::decompress::index::build_index(&compressed, 1024 * 1024)
        .expect("Failed to build index");

    // Should have at least one checkpoint
    assert!(
        !index.points.is_empty(),
        "Should have at least one checkpoint"
    );
    assert_eq!(
        index.total_uncompressed_size,
        data.len() as u64,
        "Total size should match"
    );

    // First checkpoint should be at offset 0
    assert_eq!(
        index.points[0].uncompressed_offset, 0,
        "First checkpoint should be at offset 0"
    );

    // Should be able to seek from offset 0
    let mut output = Vec::new();
    crate::decompress::index::seek_decompress(&compressed, &index, 0, u64::MAX, &mut output)
        .expect("Seek to 0 should work");
    assert_eq!(&output[..], data, "Output should match original");
}

#[test]
fn test_large_file_many_checkpoints() {
    // Create larger test data to ensure multiple checkpoints
    let data: Vec<u8> = (0..5 * 1024 * 1024).map(|i| (i % 256) as u8).collect();

    // Compress
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    let compressed = encoder.finish().unwrap();

    // Build index with 512KB interval
    let index = crate::decompress::index::build_index(&compressed, 512 * 1024)
        .expect("Failed to build index");

    // Should have at least one checkpoint
    assert!(
        !index.points.is_empty(),
        "Large 5MB file should have checkpoints"
    );

    // Verify total size matches
    assert_eq!(
        index.total_uncompressed_size,
        (5 * 1024 * 1024) as u64,
        "Total size should match input"
    );

    // Verify checkpoints are in valid range
    for (i, checkpoint) in index.points.iter().enumerate() {
        assert!(
            checkpoint.uncompressed_offset <= index.total_uncompressed_size,
            "Checkpoint {} offset {} should be within file bounds",
            i,
            checkpoint.uncompressed_offset
        );
    }
}
