//! Port of `rapidgzip::ChunkData` (ChunkData.hpp, especially lines 80-400)
//! plus its nested `Subchunk` and `Statistics`. Exactly the shape rapidgzip
//! uses; gzippy-specific extensions get added in later commits, not here.
//!
//! Two segment layout matches rapidgzip's `DecodedData`:
//!   - `data_with_markers`: prefix where some u16 values are markers
//!     (>= MARKER_BASE) standing in for cross-chunk back-references. CRC32
//!     of these bytes is computed AFTER `apply_window` resolves them.
//!   - `data`: suffix of clean bytes already CRC32'd at append time.
//!
//! Subchunks track every deflate block boundary the decoder crossed,
//! enabling per-subchunk independent post-processing once their window is
//! available — rapidgzip's pattern.
//
// dead_code is allowed module-wide: these types are part of the
// rapidgzip-port-design.md step 2 (types-only commit). The next modules
// (apply_window, gzip_chunk, chunk_fetcher) consume them; until those
// land the items have no production callers but are exercised by the
// unit tests below.
#![allow(dead_code)]

use std::sync::Arc;

pub use crate::decompress::parallel::replace_markers::MARKER_BASE;

/// One deflate-block-aligned slice of a chunk's decoded output.
/// Port of `rapidgzip::ChunkData::Subchunk` (ChunkData.hpp:138-145).
#[derive(Debug, Clone)]
pub struct Subchunk {
    pub encoded_offset_bits: usize,
    pub encoded_size_bits: usize,
    pub decoded_offset: usize,
    pub decoded_size: usize,
    /// The 32-KiB window the deflate decoder needed at this subchunk's
    /// start. Populated after sequential window propagation completes.
    pub window: Option<Arc<[u8; 32768]>>,
}

/// Per-chunk timing + counter statistics. Subset of rapidgzip's
/// `ChunkData::Statistics` (ChunkData.hpp:147-180); we surface only the
/// counters used by bench-sm telemetry.
#[derive(Debug, Default, Clone, Copy)]
pub struct ChunkStatistics {
    pub decode_duration_ns: u64,
    pub apply_window_duration_ns: u64,
    pub compute_checksum_duration_ns: u64,
    pub append_duration_ns: u64,
    pub marker_count: u64,
    pub non_marker_count: u64,
}

/// Per-chunk configuration. Port of `rapidgzip::ChunkData::Configuration`
/// (ChunkData.hpp:99-).
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfiguration {
    /// Subchunk emission threshold: when a chunk's accumulated
    /// `decoded_size` reaches this, the next block boundary starts a new
    /// subchunk. Mirror of `rapidgzip splitChunkSize`.
    /// Default = 4 MiB per `GzipChunkFetcher.hpp:706`.
    pub split_chunk_size: usize,
    /// Preemptive-stop threshold: the inexact decoder stops once
    /// `decoded_size` exceeds this, leaving more compressed input to be
    /// consumed by the next chunk. Mirror of `rapidgzip
    /// maxDecompressedChunkSize`. Default = 20× split_chunk_size per
    /// `ParallelGzipReader.hpp:292`.
    pub max_decoded_chunk_size: usize,
    pub crc32_enabled: bool,
}

impl Default for ChunkConfiguration {
    fn default() -> Self {
        let split = 4 * 1024 * 1024;
        Self {
            split_chunk_size: split,
            max_decoded_chunk_size: 20 * split,
            crc32_enabled: true,
        }
    }
}

/// Port of `rapidgzip::ChunkData` (ChunkData.hpp:80-400) + the
/// `DecodedData` base it inherits from. The two-segment layout
/// (`data_with_markers` then `data`) matches rapidgzip's
/// `DecodedData::dataWithMarkers` + `DecodedData::data`.
#[derive(Debug)]
pub struct ChunkData {
    pub encoded_offset_bits: usize,
    /// Upper bound on the encoded offset this chunk could "match" if
    /// the consumer asks for a slightly different start. Equals
    /// `encoded_offset_bits` for a chunk that decoded from an exact
    /// position; can be larger if the chunk's worker is willing to
    /// stand in for any expected start in [encoded_offset_bits,
    /// max_encoded_offset_bits]. Mirror of rapidgzip's
    /// `ChunkData::maxEncodedOffsetInBits` (ChunkData.hpp:546-549).
    pub max_encoded_offset_bits: usize,
    pub encoded_size_bits: usize,
    /// Marker-tagged prefix. Each u16 < MARKER_BASE is a literal byte
    /// (`v as u8`); values ≥ MARKER_BASE encode a cross-chunk back-
    /// reference of distance `v - MARKER_BASE` into the predecessor's
    /// last 32 KiB window. `apply_window` (next module) resolves these
    /// in place against a known window.
    pub data_with_markers: Vec<u16>,
    /// Clean byte suffix. All bytes here were decoded with a known
    /// window (set via IsalInflateWrapper::set_window) so no markers
    /// were emitted. CRC32'd at append time.
    pub data: Vec<u8>,
    /// Real deflate block boundaries the decoder crossed during decode.
    /// Indexed into the combined `(data_with_markers ++ data)` stream.
    /// Per rapidgzip's pattern, an entry is appended whenever the
    /// decoder hits a boundary AND `decoded_size >= split_chunk_size`
    /// since the last subchunk.
    pub subchunks: Vec<Subchunk>,
    /// Per-chunk CRC. Combined into the gzip-trailer CRC32 in chunk
    /// order at end of decode.
    pub crc: crc32fast::Hasher,
    /// True iff the inexact decoder hit `max_decoded_chunk_size`
    /// before reaching the requested `until_bits`. Tells the
    /// dispatcher this chunk needs a successor starting at
    /// `encoded_offset_bits + encoded_size_bits`.
    pub stopped_preemptively: bool,
    pub statistics: ChunkStatistics,
    pub configuration: ChunkConfiguration,
    /// Running output byte position since the last subchunk start.
    /// Used by `append_block_boundary` to decide whether to emit a new
    /// subchunk. Internal; not part of rapidgzip's public surface but
    /// the equivalent of `decodedSize - subchunks.back().decodedOffset`.
    next_subchunk_start_decoded_offset: usize,
}

impl ChunkData {
    /// Construct an empty chunk anchored at the given compressed-stream
    /// bit offset. The first subchunk is pre-allocated at offset 0 with
    /// zero size, mirroring rapidgzip's `startNewSubchunk` pattern
    /// (GzipChunk.hpp:47-58 init).
    pub fn new(encoded_offset_bits: usize, configuration: ChunkConfiguration) -> Self {
        let first_subchunk = Subchunk {
            encoded_offset_bits,
            encoded_size_bits: 0,
            decoded_offset: 0,
            decoded_size: 0,
            window: None,
        };
        Self {
            encoded_offset_bits,
            max_encoded_offset_bits: encoded_offset_bits,
            encoded_size_bits: 0,
            data_with_markers: Vec::new(),
            data: Vec::new(),
            subchunks: vec![first_subchunk],
            crc: crc32fast::Hasher::new(),
            stopped_preemptively: false,
            statistics: ChunkStatistics::default(),
            configuration,
            next_subchunk_start_decoded_offset: 0,
        }
    }

    /// Whether `expected_start` falls within this chunk's acceptable
    /// start range. Mirror of `rapidgzip::ChunkData::matchesEncodedOffset`
    /// (ChunkData.hpp:396-403). Used by the consumer to accept
    /// speculative chunks whose start is in tolerance rather than
    /// exactly equal to the predecessor's actual end.
    pub fn matches_encoded_offset(&self, expected_start: usize) -> bool {
        self.encoded_offset_bits <= expected_start && expected_start <= self.max_encoded_offset_bits
    }

    /// Extract the trailing 32 KiB of decoded output as a window
    /// suitable for seeding the next chunk's decoder. Returns None if
    /// the trailing 32 KiB contains any markers (i.e. would corrupt
    /// the successor's dict). For fast-path chunks (data_with_markers
    /// is empty) and chunks whose ISA-L bulk segment is at least
    /// 32 KiB, this always returns Some.
    pub fn last_32kib_window(&self) -> Option<[u8; 32768]> {
        const W: usize = 32768;
        let total = self.decoded_size();
        if total == 0 {
            return None;
        }
        let mut out = [0u8; W];
        if self.data.len() >= W {
            out.copy_from_slice(&self.data[self.data.len() - W..]);
            Some(out)
        } else if total >= W {
            // Tail straddles markers + clean. Markers in the trailing
            // W bytes mean we can't build a clean window without
            // apply_window resolving them first; caller has to wait.
            let from_data = self.data.len();
            let from_markers = W - from_data;
            let m_start = self.data_with_markers.len() - from_markers;
            for v in &self.data_with_markers[m_start..] {
                if *v >= crate::decompress::parallel::replace_markers::MARKER_BASE {
                    return None;
                }
            }
            for (i, v) in self.data_with_markers[m_start..].iter().enumerate() {
                out[i] = *v as u8;
            }
            out[from_markers..].copy_from_slice(&self.data);
            Some(out)
        } else {
            // Less than W bytes total. We don't try to combine with the
            // predecessor's window here; the consumer can do that if it
            // needs to. Return None to signal "no clean tail-window
            // available standalone."
            None
        }
    }

    #[inline]
    pub fn decoded_size(&self) -> usize {
        self.data_with_markers.len() + self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data_with_markers.is_empty() && self.data.is_empty()
    }

    /// Append marker-tagged output (u16 with `MARKER_BASE` bit set on
    /// back-references). Mirror of `ChunkData::append(DecodedVector&&)`
    /// for the markered branch. CRC32 is deferred to `apply_window`
    /// because markers don't have a final byte value yet.
    pub fn append_markered(&mut self, values: &[u16]) {
        for v in values {
            if *v >= MARKER_BASE {
                self.statistics.marker_count += 1;
            } else {
                self.statistics.non_marker_count += 1;
            }
        }
        self.data_with_markers.extend_from_slice(values);
        if let Some(last) = self.subchunks.last_mut() {
            last.decoded_size += values.len();
        }
    }

    /// Append clean (already-resolved) output bytes. Mirror of
    /// `ChunkData::append(DecodedDataView)` for the non-marker branch.
    /// CRC32'd immediately if enabled.
    pub fn append_clean(&mut self, bytes: &[u8]) {
        if self.configuration.crc32_enabled {
            self.crc.update(bytes);
        }
        self.statistics.non_marker_count += bytes.len() as u64;
        self.data.extend_from_slice(bytes);
        if let Some(last) = self.subchunks.last_mut() {
            last.decoded_size += bytes.len();
        }
    }

    /// Called by the decoder when it hits a real deflate block boundary.
    /// Port of `ChunkData::appendDeflateBlockBoundary` (ChunkData.hpp:159-180):
    /// emits a new subchunk if the previous one has grown past
    /// `split_chunk_size`. Otherwise the current subchunk just absorbs
    /// the next block.
    pub fn append_block_boundary(&mut self, encoded_offset_bits: usize) {
        let current_decoded = self.decoded_size();
        let bytes_since_subchunk_start =
            current_decoded.saturating_sub(self.next_subchunk_start_decoded_offset);
        if bytes_since_subchunk_start < self.configuration.split_chunk_size {
            return;
        }
        // Finalize the trailing subchunk: set its encoded_size_bits to
        // span from its encoded_offset to here.
        if let Some(last) = self.subchunks.last_mut() {
            debug_assert!(encoded_offset_bits >= last.encoded_offset_bits);
            last.encoded_size_bits = encoded_offset_bits - last.encoded_offset_bits;
        }
        // Start the next subchunk at this boundary.
        self.next_subchunk_start_decoded_offset = current_decoded;
        self.subchunks.push(Subchunk {
            encoded_offset_bits,
            encoded_size_bits: 0,
            decoded_offset: current_decoded,
            decoded_size: 0,
            window: None,
        });
    }

    /// Finalize at end of decode. Mirror of `ChunkData::finalizeChunk`
    /// (ChunkData.hpp:136-159): set the parent chunk's
    /// `encoded_size_bits`; close out the trailing subchunk's
    /// `encoded_size_bits`. Sub-chunk merging on undersize trailing
    /// pieces follows rapidgzip's pattern.
    pub fn finalize(&mut self, end_encoded_offset_bits: usize) {
        debug_assert!(end_encoded_offset_bits >= self.encoded_offset_bits);
        self.encoded_size_bits = end_encoded_offset_bits - self.encoded_offset_bits;
        if let Some(last) = self.subchunks.last_mut() {
            debug_assert!(end_encoded_offset_bits >= last.encoded_offset_bits);
            last.encoded_size_bits = end_encoded_offset_bits - last.encoded_offset_bits;
        }
        // Merge an undersized trailing subchunk back into its predecessor.
        // Rapidgzip does this to avoid fragmentation when the final block
        // group is smaller than split_chunk_size.
        if self.subchunks.len() >= 2 {
            let tail_size = self.subchunks.last().unwrap().decoded_size;
            if tail_size < self.configuration.split_chunk_size {
                let tail = self.subchunks.pop().unwrap();
                let prev = self.subchunks.last_mut().unwrap();
                prev.decoded_size += tail.decoded_size;
                prev.encoded_size_bits += tail.encoded_size_bits;
            }
        }
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ChunkConfiguration {
        ChunkConfiguration {
            split_chunk_size: 100,
            max_decoded_chunk_size: 10_000,
            crc32_enabled: true,
        }
    }

    #[test]
    fn new_chunk_has_one_zero_size_subchunk_anchored_at_encoded_offset() {
        let chunk = ChunkData::new(12_345, small_config());
        assert_eq!(chunk.encoded_offset_bits, 12_345);
        assert_eq!(chunk.encoded_size_bits, 0);
        assert!(chunk.is_empty());
        assert_eq!(chunk.decoded_size(), 0);
        assert_eq!(chunk.subchunks.len(), 1);
        let sc = &chunk.subchunks[0];
        assert_eq!(sc.encoded_offset_bits, 12_345);
        assert_eq!(sc.encoded_size_bits, 0);
        assert_eq!(sc.decoded_offset, 0);
        assert_eq!(sc.decoded_size, 0);
        assert!(sc.window.is_none());
    }

    #[test]
    fn append_markered_grows_data_with_markers_and_subchunk_size() {
        let mut chunk = ChunkData::new(0, small_config());
        let values: Vec<u16> = (0..50u16).collect(); // 50 literals
        chunk.append_markered(&values);
        assert_eq!(chunk.data_with_markers.len(), 50);
        assert_eq!(chunk.data.len(), 0);
        assert_eq!(chunk.decoded_size(), 50);
        assert_eq!(chunk.subchunks[0].decoded_size, 50);
        assert_eq!(chunk.statistics.marker_count, 0);
        assert_eq!(chunk.statistics.non_marker_count, 50);
    }

    #[test]
    fn append_markered_counts_marker_values() {
        let mut chunk = ChunkData::new(0, small_config());
        let values: Vec<u16> = vec![0u16, 1, 2, MARKER_BASE, MARKER_BASE + 5, 6];
        chunk.append_markered(&values);
        assert_eq!(chunk.statistics.marker_count, 2);
        assert_eq!(chunk.statistics.non_marker_count, 4);
    }

    #[test]
    fn append_clean_grows_data_and_updates_crc() {
        let mut chunk = ChunkData::new(0, small_config());
        chunk.append_clean(b"hello");
        chunk.append_clean(b" world");
        assert_eq!(chunk.data_with_markers.len(), 0);
        assert_eq!(chunk.data.len(), 11);
        assert_eq!(chunk.data, b"hello world");
        // CRC of "hello world" — non-zero, deterministic.
        let mut expected = crc32fast::Hasher::new();
        expected.update(b"hello world");
        assert_eq!(chunk.crc.clone().finalize(), expected.finalize());
        assert_eq!(chunk.subchunks[0].decoded_size, 11);
    }

    #[test]
    fn append_block_boundary_is_noop_under_split_threshold() {
        let mut chunk = ChunkData::new(0, small_config()); // split = 100
        chunk.append_clean(&[0u8; 50][..]); // 50 bytes
        chunk.append_block_boundary(400); // bit offset 400, 50 < 100 split
        assert_eq!(chunk.subchunks.len(), 1, "no new subchunk under threshold");
    }

    #[test]
    fn append_block_boundary_emits_subchunk_when_split_threshold_reached() {
        let mut chunk = ChunkData::new(0, small_config()); // split = 100
        chunk.append_clean(&[0u8; 100][..]); // exactly 100 bytes
        chunk.append_block_boundary(800);
        assert_eq!(chunk.subchunks.len(), 2);
        // Old subchunk now has finalized encoded_size_bits = 800 - 0 = 800.
        assert_eq!(chunk.subchunks[0].encoded_size_bits, 800);
        assert_eq!(chunk.subchunks[0].decoded_size, 100);
        // New subchunk starts at 800/100.
        assert_eq!(chunk.subchunks[1].encoded_offset_bits, 800);
        assert_eq!(chunk.subchunks[1].decoded_offset, 100);
        assert_eq!(chunk.subchunks[1].decoded_size, 0);
    }

    #[test]
    fn multiple_block_boundaries_emit_multiple_subchunks() {
        let mut chunk = ChunkData::new(0, small_config()); // split = 100
        for round in 0..5 {
            chunk.append_clean(&[0u8; 120][..]); // exceed threshold each time
            chunk.append_block_boundary(1000 * (round + 1));
        }
        // Expected: 6 subchunks (initial + 5 emissions, no merging yet).
        assert_eq!(chunk.subchunks.len(), 6);
        // Decoded sizes sum to 5*120 = 600.
        let total: usize = chunk.subchunks.iter().map(|s| s.decoded_size).sum();
        assert_eq!(total, 600);
    }

    #[test]
    fn finalize_sets_chunk_encoded_size_and_closes_tail_subchunk() {
        let mut chunk = ChunkData::new(100, small_config());
        chunk.append_clean(&[0u8; 50][..]);
        chunk.finalize(900); // total bits = 900 - 100 = 800
        assert_eq!(chunk.encoded_size_bits, 800);
        // Single subchunk (no boundary emitted because 50 < 100 split);
        // its encoded_size_bits should equal the parent's.
        assert_eq!(chunk.subchunks.len(), 1);
        assert_eq!(chunk.subchunks[0].encoded_size_bits, 800);
    }

    #[test]
    fn finalize_merges_undersize_trailing_subchunk() {
        let mut chunk = ChunkData::new(0, small_config()); // split = 100
        chunk.append_clean(&[0u8; 200][..]); // 200 > 100 split
        chunk.append_block_boundary(2000); // emits subchunk #2
        chunk.append_clean(&[0u8; 30][..]); // tail of 30 < 100 split
        chunk.finalize(2500);
        // Tail (30 bytes) is < split_chunk_size → merged back into #1.
        assert_eq!(chunk.subchunks.len(), 1);
        assert_eq!(chunk.subchunks[0].decoded_size, 230);
        assert_eq!(chunk.subchunks[0].encoded_size_bits, 2500);
    }

    #[test]
    fn append_clean_skips_crc_when_disabled() {
        let cfg = ChunkConfiguration {
            split_chunk_size: 100,
            max_decoded_chunk_size: 10_000,
            crc32_enabled: false,
        };
        let mut chunk = ChunkData::new(0, cfg);
        chunk.append_clean(b"hello world");
        // CRC unchanged from identity.
        assert_eq!(chunk.crc.clone().finalize(), 0);
    }
}
