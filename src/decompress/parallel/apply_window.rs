//! Port of `rapidgzip::ChunkData::applyWindow` (ChunkData.hpp:302).
//!
//! Resolves marker-tagged values in `chunk.data_with_markers` against a
//! known 32-KiB window: every u16 ≥ MARKER_BASE is replaced with the
//! corresponding byte from the window. After this call, every value in
//! `data_with_markers` is < 256 (a literal byte).
//!
//! Updates `chunk.crc` so that the chunk's per-chunk CRC32 covers the
//! full ordered byte stream `(data_with_markers as u8) ++ data`.
//! `data` was CRC'd at append time; we prepend the CRC of the resolved
//! `data_with_markers` bytes via `crc32fast::Hasher::combine`.

use crate::decompress::parallel::chunk_data::{ChunkData, MARKER_BASE};
use crate::decompress::parallel::replace_markers::replace_markers;

#[allow(dead_code)]
pub fn apply_window(chunk: &mut ChunkData, window: &[u8]) {
    if chunk.data_with_markers.is_empty() {
        return;
    }
    debug_assert!(
        window.len() == 32768,
        "rapidgzip semantics require a 32 KiB window for applyWindow"
    );

    let t0 = std::time::Instant::now();

    // Resolve markers in place. After this call, every value in
    // `data_with_markers` is < 256 (a literal byte).
    replace_markers(&mut chunk.data_with_markers, window);

    // Verify in debug builds: any remaining marker would indicate a
    // resolver bug (out-of-range distance or wrong window size).
    debug_assert!(
        chunk.data_with_markers.iter().all(|v| *v < MARKER_BASE),
        "apply_window left unresolved markers in data_with_markers"
    );

    // CRC accounting: data_with_markers' resolved bytes precede `data`
    // in the output stream. data was CRC'd at append time. Per rapidgzip
    // (ChunkData.hpp:432: "data with markers ought not cross footer
    // boundaries"), markers belong entirely to the FIRST stream, so we
    // prepend the resolved-markers CRC into crc32s[0].
    if chunk.configuration.crc32_enabled && !chunk.crc32s.is_empty() {
        let mut resolved_crc = crc32fast::Hasher::new();
        let mut scratch = [0u8; 4096];
        let mut i = 0;
        while i < chunk.data_with_markers.len() {
            let n = scratch.len().min(chunk.data_with_markers.len() - i);
            for (k, v) in chunk.data_with_markers[i..i + n].iter().enumerate() {
                scratch[k] = *v as u8;
            }
            resolved_crc.update(&scratch[..n]);
            i += n;
        }
        resolved_crc.combine(&chunk.crc32s[0]);
        chunk.crc32s[0] = resolved_crc;
    }

    chunk.statistics.apply_window_duration_ns += t0.elapsed().as_nanos() as u64;
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::chunk_data::ChunkConfiguration;

    fn config() -> ChunkConfiguration {
        ChunkConfiguration {
            split_chunk_size: 100,
            max_decoded_chunk_size: 10_000,
            crc32_enabled: true,
        }
    }

    fn make_window(byte_at: impl Fn(usize) -> u8) -> [u8; 32768] {
        let mut w = [0u8; 32768];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = byte_at(i);
        }
        w
    }

    #[test]
    fn apply_window_resolves_back_references_to_window_bytes() {
        let mut chunk = ChunkData::new(0, config());
        // 5 literals (0..5) followed by 3 markers. MapMarkers semantics:
        // marker value (post-MARKER_BASE subtraction) is a direct index
        // into the 32-KiB window from the OLDEST byte.
        let markers: Vec<u16> = vec![0, 1, 2, 3, 4, MARKER_BASE, MARKER_BASE + 1, MARKER_BASE + 2];
        chunk.append_markered(&markers);

        // Window: bytes 0..32768 where byte i = (i % 251) as u8.
        let window = make_window(|i| (i % 251) as u8);
        apply_window(&mut chunk, &window);

        // First 5 bytes unchanged.
        for i in 0..5 {
            assert_eq!(chunk.data_with_markers[i], i as u16);
        }
        // Marker 0 → window[0], marker 1 → window[1], marker 2 → window[2].
        assert_eq!(chunk.data_with_markers[5] as u8, window[0]);
        assert_eq!(chunk.data_with_markers[6] as u8, window[1]);
        assert_eq!(chunk.data_with_markers[7] as u8, window[2]);
    }

    #[test]
    fn apply_window_is_noop_when_no_markers() {
        let mut chunk = ChunkData::new(0, config());
        chunk.append_clean(b"hello world");
        let original_crc = chunk.crc32s[0].clone().finalize();
        let window = make_window(|i| (i % 251) as u8);
        apply_window(&mut chunk, &window);
        // No mutation: data_with_markers is empty, data unchanged, CRC unchanged.
        assert!(chunk.data_with_markers.is_empty());
        assert_eq!(chunk.data, b"hello world");
        assert_eq!(chunk.crc32s[0].clone().finalize(), original_crc);
    }

    #[test]
    fn apply_window_crc_matches_concatenated_bytes() {
        let mut chunk = ChunkData::new(0, config());
        // Mix of markers (resolved against window) and clean bytes.
        chunk.append_markered(&[0u16, 1, 2, MARKER_BASE, MARKER_BASE + 5]);
        chunk.append_clean(b"trailing");

        let window = make_window(|i| (i % 251) as u8);
        apply_window(&mut chunk, &window);

        // Construct the expected ordered byte stream and CRC it directly.
        let mut expected_bytes = Vec::new();
        for v in &chunk.data_with_markers {
            expected_bytes.push(*v as u8);
        }
        expected_bytes.extend_from_slice(&chunk.data);
        let mut expected_crc = crc32fast::Hasher::new();
        expected_crc.update(&expected_bytes);
        assert_eq!(chunk.crc32s[0].clone().finalize(), expected_crc.finalize());
    }

    #[test]
    fn apply_window_skips_crc_when_disabled() {
        let cfg = ChunkConfiguration {
            split_chunk_size: 100,
            max_decoded_chunk_size: 10_000,
            crc32_enabled: false,
        };
        let mut chunk = ChunkData::new(0, cfg);
        chunk.append_markered(&[0u16, 1, MARKER_BASE]);
        let window = make_window(|_i| 0xAB);
        apply_window(&mut chunk, &window);
        // CRC stays at identity (never updated).
        assert_eq!(chunk.crc32s[0].clone().finalize(), 0);
        // But markers still resolved.
        assert_eq!(chunk.data_with_markers[2] as u8, 0xAB);
    }
}
