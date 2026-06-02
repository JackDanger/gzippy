#![cfg(parallel_sm)]

//! Port of `rapidgzip::ChunkData::applyWindow` (ChunkData.hpp:302).
//!
//! Resolves marker-tagged values in `chunk.data_with_markers` against a
//! known 32-KiB window: every u16 ≥ MARKER_BASE is replaced with the
//! corresponding byte from the window. After this call, every value in
//! `data_with_markers` is < 256 (a literal byte).
//!
//! CRC32 accounting is the caller's responsibility — see `apply_window`.

use crate::decompress::parallel::chunk_data::{ChunkData, MARKER_BASE};
use crate::decompress::parallel::replace_markers::replace_markers;

/// Resolve markers in place. CRC32 accounting is the CALLER's
/// responsibility: the consumer CRCs the resolved bytes once, over the
/// already-narrowed buffer it builds for the output write, so this
/// function does not pay a separate narrow + CRC pass.
pub fn apply_window(chunk: &mut ChunkData, window: &[u8]) {
    if chunk.data_with_markers.is_empty() {
        return;
    }
    debug_assert!(
        window.len() == 32768,
        "rapidgzip semantics require a 32 KiB window for applyWindow"
    );
    // A1 invariant: marker resolution only writes into
    // `chunk.data_with_markers`, but downstream the consumer reads
    // `chunk.data` AFTER apply_window. A leftover window-image prefix
    // would be emitted as decoded output. Trim before apply_window.
    debug_assert_eq!(
        chunk.data_prefix_len, 0,
        "trim_window_prefix before apply_window"
    );

    // Resolve markers in place. After this call, every value in
    // `data_with_markers` is < 256 (a literal byte).
    replace_markers(&mut chunk.data_with_markers, window);

    // Verify in debug builds: any remaining marker would indicate a
    // resolver bug (out-of-range distance or wrong window size).
    debug_assert!(
        chunk.data_with_markers.iter().all(|v| *v < MARKER_BASE),
        "apply_window left unresolved markers in data_with_markers"
    );
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
        let original_crc = chunk.crc32s[0].crc32();
        let window = make_window(|i| (i % 251) as u8);
        apply_window(&mut chunk, &window);
        // No mutation: data_with_markers is empty, data unchanged, CRC unchanged.
        assert!(chunk.data_with_markers.is_empty());
        assert_eq!(chunk.data.to_contiguous(), b"hello world");
        assert_eq!(chunk.crc32s[0].crc32(), original_crc);
    }

    #[test]
    fn apply_window_resolves_markers_for_concatenated_byte_stream() {
        let mut chunk = ChunkData::new(0, config());
        // Mix of markers (resolved against window) and clean bytes.
        chunk.append_markered(&[0u16, 1, 2, MARKER_BASE, MARKER_BASE + 5]);
        chunk.append_clean(b"trailing");

        let window = make_window(|i| (i % 251) as u8);
        apply_window(&mut chunk, &window);

        // After resolution every data_with_markers entry is < 256.
        assert!(chunk.data_with_markers.iter().all(|v| *v < MARKER_BASE));
        // Markers map to window bytes 0 and 5.
        assert_eq!(chunk.data_with_markers[3] as u8, window[0]);
        assert_eq!(chunk.data_with_markers[4] as u8, window[5]);
        // (CRC accounting moved to the consumer — verified by the
        // routing-level round-trip tests in tests/routing.rs and the
        // full bench's `output size mismatch` / md5 check.)
    }

    #[test]
    fn apply_window_does_not_touch_crc() {
        // Apply_window no longer mutates crc32s — the consumer's
        // post-narrow CRC pass owns that responsibility.
        let mut chunk = ChunkData::new(0, config());
        chunk.append_markered(&[0u16, 1, MARKER_BASE]);
        let crc_before = chunk.crc32s[0].crc32();
        let window = make_window(|_i| 0xAB);
        apply_window(&mut chunk, &window);
        assert_eq!(chunk.crc32s[0].crc32(), crc_before);
        // Markers still resolved.
        assert_eq!(chunk.data_with_markers[2] as u8, 0xAB);
    }
}
