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

/// Per-stream gzip footer (the 8 bytes after the final deflate block
/// of a gzip stream). Port of `rapidgzip::gzip::Footer`
/// (vendor/rapidgzip/.../gzip/gzip.hpp:151-155) wrapped with rapidgzip's
/// outer `rapidgzip::Footer` (gzip.hpp:420-448) which records the
/// associated block boundary. We only record the gzip CRC32 + ISIZE +
/// the boundary at footer-end (post-footer file offset).
#[derive(Debug, Clone, Copy, Default)]
pub struct Footer {
    /// Gzip footer CRC32 (last 8 bytes, little-endian: CRC32 then ISIZE).
    pub crc32: u32,
    /// Gzip footer ISIZE field, low 32 bits of uncompressed stream size.
    pub uncompressed_size: u32,
    /// Bit offset in the compressed stream where this footer's stream ends
    /// (footer-end). For a multi-stream input, the next gzip header starts
    /// at this position.
    pub end_bit_offset: usize,
    /// Decoded byte offset within the chunk where this stream's bytes
    /// end. Used to delimit which CRC32 in `crc32s` covers which bytes.
    pub decoded_end_offset: usize,
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
    /// (`v as u8`); values ≥ MARKER_BASE are direct indices into the
    /// predecessor's 32 KiB window from the OLDEST byte (mirrors
    /// rapidgzip MapMarkers: window[v - MARKER_BASE]). Cross-chunk
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
    /// One CRC32 hasher per gzip stream this chunk spans. For a
    /// single-stream chunk (the common case) this is a single-element
    /// vector. Rapidgzip allocates a new entry when an `END_OF_STREAM`
    /// stopping point fires mid-chunk (`ChunkData.hpp:228-243`). The
    /// first entry covers `data_with_markers ++ data[0..footers[0].decoded_end_offset]`
    /// (after `apply_window`); subsequent entries cover the bytes between
    /// consecutive footers.
    pub crc32s: Vec<crc32fast::Hasher>,
    /// Footers detected in this chunk, one per gzip stream that ended
    /// inside the chunk's compressed range. Mirror of rapidgzip's
    /// `ChunkData::footers` (ChunkData.hpp:472-489 `appendFooter`).
    pub footers: Vec<Footer>,
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
            crc32s: vec![crc32fast::Hasher::new()],
            footers: Vec::new(),
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
    /// CRC32'd immediately if enabled. The CRC always feeds the most
    /// recent stream's hasher (`crc32s.last_mut()`); cross-stream byte
    /// runs are split by `append_footer`.
    pub fn append_clean(&mut self, bytes: &[u8]) {
        if self.configuration.crc32_enabled {
            if let Some(last_crc) = self.crc32s.last_mut() {
                last_crc.update(bytes);
            }
        }
        self.statistics.non_marker_count += bytes.len() as u64;
        self.data.extend_from_slice(bytes);
        if let Some(last) = self.subchunks.last_mut() {
            last.decoded_size += bytes.len();
        }
    }

    /// Literal port of `ChunkData::appendFooter`
    /// (vendor/rapidgzip/.../ChunkData.hpp:472-489). Records that a gzip
    /// stream ended at `end_bit_offset` (= byte position immediately
    /// after the 8-byte footer) AND the chunk's current decoded-byte
    /// position is the boundary. Starts a fresh CRC hasher for the next
    /// stream's bytes (mirrors rapidgzip allocating a new
    /// `CRC32Calculator` in its `crc32s` vector).
    pub fn append_footer(&mut self, crc32: u32, uncompressed_size: u32, end_bit_offset: usize) {
        self.footers.push(Footer {
            crc32,
            uncompressed_size,
            end_bit_offset,
            decoded_end_offset: self.decoded_size(),
        });
        // Open a fresh hasher for the next stream's bytes. If no more
        // bytes follow, the trailing empty hasher's CRC == 0 and the
        // consumer's combine treats it as a no-op.
        self.crc32s.push(crc32fast::Hasher::new());
    }

    /// Called by the decoder when it hits a real deflate block boundary.
    /// Always emits a new subchunk (unlike rapidgzip's
    /// `ChunkData::appendDeflateBlockBoundary` which gates on
    /// split_chunk_size). This enables the consumer to trim a chunk
    /// whose start doesn't exactly match the expected_start: scan
    /// subchunks for one with encoded_offset_bits == expected_start,
    /// drop everything before that subchunk's decoded_offset.
    ///
    /// Trade-off: per-chunk memory grows by ~50 bytes per block
    /// boundary crossed. A 12 MiB chunk with ~50 blocks costs ~2.5 KiB
    /// of subchunk records, negligible. The win: speculation hit rate
    /// jumps from ~3% to ~100% on Silesia gzip -9 because the consumer
    /// no longer demands exact start matching.
    pub fn append_block_boundary(&mut self, encoded_offset_bits: usize) {
        // Don't push duplicate subchunks if the decoder calls us at
        // the same bit position twice (defensive).
        if let Some(last) = self.subchunks.last() {
            if last.encoded_offset_bits == encoded_offset_bits {
                return;
            }
        }
        let current_decoded = self.decoded_size();
        if let Some(last) = self.subchunks.last_mut() {
            debug_assert!(encoded_offset_bits >= last.encoded_offset_bits);
            last.encoded_size_bits = encoded_offset_bits - last.encoded_offset_bits;
        }
        self.next_subchunk_start_decoded_offset = current_decoded;
        self.subchunks.push(Subchunk {
            encoded_offset_bits,
            encoded_size_bits: 0,
            decoded_offset: current_decoded,
            decoded_size: 0,
            window: None,
        });
    }

    /// Finalize at end of decode. Sets `encoded_size_bits` for the
    /// chunk and its trailing subchunk; updates `max_encoded_offset_bits`
    /// so the consumer's `matches_encoded_offset` accepts any expected
    /// start in [encoded_offset_bits, end_encoded_offset_bits]. With
    /// per-boundary subchunks, that range typically holds a subchunk
    /// at the exact expected_start, so the consumer can trim.
    pub fn finalize(&mut self, end_encoded_offset_bits: usize) {
        debug_assert!(end_encoded_offset_bits >= self.encoded_offset_bits);
        self.encoded_size_bits = end_encoded_offset_bits - self.encoded_offset_bits;
        self.max_encoded_offset_bits = end_encoded_offset_bits;
        if let Some(last) = self.subchunks.last_mut() {
            debug_assert!(end_encoded_offset_bits >= last.encoded_offset_bits);
            last.encoded_size_bits = end_encoded_offset_bits - last.encoded_offset_bits;
        }
    }

    /// Literal port of `ChunkData::setEncodedOffset`
    /// (vendor/rapidgzip/.../ChunkData.hpp:601-629). After a chunk
    /// completes, the consumer calls this with the REAL start offset
    /// (predecessor's actual end) to re-anchor the chunk from its
    /// speculative-seed start. Collapses the [encoded, max] match range
    /// to a single point.
    ///
    /// Panics if `offset` is not in `[encoded_offset_bits, max_encoded_offset_bits]`
    /// — rapidgzip throws `std::invalid_argument` in the same condition.
    /// In our codepath this method is currently exercised only on hits
    /// where `encoded == max == offset` (a no-op re-anchor), but keeping
    /// the full range semantics enables future-stored-block range
    /// candidate hits per rapidgzip's `Uncompressed.hpp` finder pair.
    pub fn set_encoded_offset(&mut self, offset: usize) {
        debug_assert!(
            self.matches_encoded_offset(offset),
            "set_encoded_offset called with offset {} outside range [{}, {}]",
            offset,
            self.encoded_offset_bits,
            self.max_encoded_offset_bits,
        );
        let end_offset = self.encoded_offset_bits + self.encoded_size_bits;
        debug_assert!(
            end_offset >= offset,
            "chunk end {} is before requested start {}",
            end_offset,
            offset,
        );
        self.encoded_size_bits = end_offset - offset;
        self.encoded_offset_bits = offset;
        self.max_encoded_offset_bits = offset;
        // Adjust first subchunk: its encoded_offset becomes the new
        // chunk start, and its encoded_size spans to the next subchunk
        // (or to the chunk end if there's only one subchunk).
        if !self.subchunks.is_empty() {
            let next_offset = if self.subchunks.len() >= 2 {
                self.subchunks[1].encoded_offset_bits
            } else {
                end_offset
            };
            let first = &mut self.subchunks[0];
            first.encoded_offset_bits = offset;
            first.encoded_size_bits = next_offset.saturating_sub(offset);
        }
    }

    /// Find the subchunk whose `encoded_offset_bits` exactly matches
    /// `expected_start`, returning its `decoded_offset` (i.e. how many
    /// leading bytes to skip in this chunk's output). Returns None if
    /// no subchunk matches; the consumer then falls back to
    /// authoritative re-dispatch.
    pub fn decoded_offset_for(&self, expected_start: usize) -> Option<usize> {
        if expected_start == self.encoded_offset_bits {
            return Some(0);
        }
        self.subchunks
            .iter()
            .find(|s| s.encoded_offset_bits == expected_start)
            .map(|s| s.decoded_offset)
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
    fn set_encoded_offset_re_anchors_chunk_and_first_subchunk() {
        // Build a chunk with two subchunks: first at bit 100, second at bit 200,
        // chunk end at bit 300. max = 150 (range [100,150]).
        let mut chunk = ChunkData::new(100, small_config());
        chunk.append_block_boundary(200);
        chunk.finalize(300);
        // Pretend the decoder set max via tryToDecode iteration; rapidgzip
        // chunks created from a stored-block range have max > encoded.
        chunk.max_encoded_offset_bits = 150;

        // Re-anchor to 130 (within [100, 150]).
        chunk.set_encoded_offset(130);

        assert_eq!(chunk.encoded_offset_bits, 130);
        assert_eq!(chunk.max_encoded_offset_bits, 130);
        assert_eq!(chunk.encoded_size_bits, 300 - 130);
        assert_eq!(chunk.subchunks[0].encoded_offset_bits, 130);
        assert_eq!(chunk.subchunks[0].encoded_size_bits, 200 - 130);
        // Second subchunk unchanged.
        assert_eq!(chunk.subchunks[1].encoded_offset_bits, 200);
    }

    #[test]
    fn set_encoded_offset_single_subchunk_extends_to_chunk_end() {
        let mut chunk = ChunkData::new(0, small_config());
        chunk.finalize(500);
        chunk.max_encoded_offset_bits = 100;

        chunk.set_encoded_offset(50);

        assert_eq!(chunk.encoded_offset_bits, 50);
        assert_eq!(chunk.subchunks[0].encoded_offset_bits, 50);
        assert_eq!(chunk.subchunks[0].encoded_size_bits, 500 - 50);
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
        assert_eq!(chunk.crc32s[0].clone().finalize(), expected.finalize());
        assert_eq!(chunk.subchunks[0].decoded_size, 11);
    }

    #[test]
    fn append_block_boundary_always_emits_subchunk() {
        // Semantics changed: every call to append_block_boundary emits
        // a new subchunk (replaces the split_chunk_size gate). This
        // enables consumer-side per-block-boundary trimming.
        let mut chunk = ChunkData::new(0, small_config()); // split = 100
        chunk.append_clean(&[0u8; 50][..]); // 50 bytes
        chunk.append_block_boundary(400);
        assert_eq!(
            chunk.subchunks.len(),
            2,
            "subchunk emitted at every boundary"
        );
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
    fn finalize_keeps_per_boundary_subchunks() {
        // Semantics changed: finalize no longer merges undersize tails.
        // Every block boundary keeps its subchunk so the consumer can
        // trim at any boundary.
        let mut chunk = ChunkData::new(0, small_config());
        chunk.append_clean(&[0u8; 200][..]);
        chunk.append_block_boundary(2000); // subchunk #2 at bit 2000
        chunk.append_clean(&[0u8; 30][..]);
        chunk.finalize(2500);
        assert_eq!(chunk.subchunks.len(), 2);
        assert_eq!(chunk.subchunks[0].decoded_size, 200);
        assert_eq!(chunk.subchunks[1].decoded_size, 30);
        assert_eq!(chunk.subchunks[1].encoded_size_bits, 500);
        assert_eq!(chunk.max_encoded_offset_bits, 2500);
        assert_eq!(chunk.decoded_offset_for(0), Some(0));
        assert_eq!(chunk.decoded_offset_for(2000), Some(200));
        assert_eq!(chunk.decoded_offset_for(1234), None);
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
        assert_eq!(chunk.crc32s[0].clone().finalize(), 0);
    }
}
