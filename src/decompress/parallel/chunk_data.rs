#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]

//! Port of `rapidgzip::ChunkData` (ChunkData.hpp, especially lines 80-400)
//! plus its nested `Subchunk` and `Statistics`.
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
// Module-wide dead_code allowance: this is a types-heavy port; some
// accessors are exercised only by the unit tests below or by
// configuration-gated callers.

use std::sync::Arc;

use crate::decompress::parallel::crc32::CRC32Calculator;
pub use crate::decompress::parallel::replace_markers::MARKER_BASE;
use crate::decompress::parallel::rpmalloc_alloc::types::{self, U16, U8};

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
#[allow(dead_code)] // timing fields populated for future diagnostics
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
#[allow(dead_code)] // seekable-index / multi-stream footer metadata
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
///
/// `Clone` is derived to mirror rapidgzip's `std::shared_ptr<ChunkData>`
/// semantics: when the consumer pulls a cached chunk and wants to
/// mutate it (apply_window, set_encoded_offset, populate subchunk
/// windows), it can clone the cached entry. In rapidgzip this is
/// implicit via shared_ptr (mutation through the pointer aliases the
/// cached copy); in Rust we make it explicit by cloning so the cache
/// entry and the consumer's working copy don't alias mutably. The
/// underlying `crc32fast::Hasher` is itself `Clone` (it serializes its
/// rolling state), so this is structurally cheap.
#[derive(Debug, Clone)]
pub struct ChunkData {
    pub encoded_offset_bits: usize,
    /// Highest **start** offset this chunk can satisfy via
    /// `matches_encoded_offset` / `set_encoded_offset` — NOT the
    /// chunk's compressed end. Speculative prefetches set this to the
    /// real block boundary (`offset.second` in vendor) while
    /// `encoded_offset_bits` stays at the partition seed
    /// (`offset.first`). Decoded bytes begin at `max_acceptable_start_bit`.
    /// Mirror of `ChunkData::maxEncodedOffsetInBits` (ChunkData.hpp:546).
    pub max_acceptable_start_bit: usize,
    pub encoded_size_bits: usize,
    /// Marker-tagged prefix. Each u16 < MARKER_BASE is a literal byte
    /// (`v as u8`); values ≥ MARKER_BASE are direct indices into the
    /// predecessor's 32 KiB window from the OLDEST byte (mirrors
    /// rapidgzip MapMarkers: window[v - MARKER_BASE]). Cross-chunk
    /// last 32 KiB window. `apply_window` (next module) resolves these
    /// in place against a known window.
    pub data_with_markers: U16,
    /// Clean byte suffix. All bytes here were decoded with a known
    /// window (set via IsalInflateWrapper::set_window) so no markers
    /// were emitted. CRC32'd at append time.
    pub data: U8,
    /// `data_with_markers` narrowed to u8 — populated by the
    /// post-process worker (`run_post_process_task`) right after
    /// `apply_window` resolves markers. Consumer thread just streams
    /// this Vec to the output writer instead of allocating + scalar-
    /// narrowing per chunk (the prior pattern serialized 24 ms/chunk
    /// on the single consumer thread and dominated wall time on
    /// real silesia — see `plans/pure-rust-perf.md` consumer-narrow
    /// finding).
    ///
    /// Empty when `data_with_markers` is empty or `apply_window` hasn't
    /// run yet. Pool-recycled via the U8 pool alongside `data`.
    pub narrowed: U8,
    /// True if `apply_window` has already been run on this chunk —
    /// either on the worker thread before send (when the predecessor's
    /// window was published in time) or by an earlier consumer pass.
    /// Consumer skips its own `apply_window` call when set, saving
    /// the redundant scan of `data_with_markers`. Mirror of vendor's
    /// `m_markersResolved` flag pattern at ChunkData.hpp.
    #[allow(dead_code)] // set by apply_window; read in future seekable path
    pub markers_resolved: bool,
    /// Real deflate block boundaries the decoder crossed during decode.
    /// Indexed into the combined `(data_with_markers ++ data)` stream.
    /// Per rapidgzip's pattern, an entry is appended whenever the
    /// decoder hits a boundary AND `decoded_size >= split_chunk_size`
    /// since the last subchunk.
    pub subchunks: Vec<Subchunk>,
    /// One CRC32 calculator per gzip stream this chunk spans. For a
    /// single-stream chunk (the common case) this is a single-element
    /// vector. Rapidgzip allocates a new entry when an `END_OF_STREAM`
    /// stopping point fires mid-chunk (`ChunkData.hpp:228-243`). The
    /// first entry covers `data_with_markers ++ data[0..footers[0].decoded_end_offset]`
    /// (after `apply_window`); subsequent entries cover the bytes between
    /// consecutive footers.
    ///
    /// Mirror of `std::vector<CRC32Calculator> crc32s` at
    /// vendor/rapidgzip/librapidarchive/src/rapidgzip/ChunkData.hpp:561.
    /// Routed through the ported `CRC32Calculator` so the polynomial
    /// combine (`append` / `prepend`) goes through the ported
    /// `combine_crc32` (gzip/crc32.hpp:214-258) — the same code path
    /// the consumer's `total_crc.append(&written_crc)` uses
    /// (GzipChunkFetcher.hpp:340).
    pub crc32s: Vec<CRC32Calculator>,
    /// Footers detected in this chunk, one per gzip stream that ended
    /// inside the chunk's compressed range. Mirror of rapidgzip's
    /// `ChunkData::footers` (ChunkData.hpp:472-489 `appendFooter`).
    pub footers: Vec<Footer>,
    /// True iff the inexact decoder hit `max_decoded_chunk_size`
    /// before reaching the requested `stop_hint_bits`. Tells the
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
    /// Worker pool index that allocated `data` / `data_with_markers`.
    /// Buffers return to this worker's arena on drop.
    pool_worker_index: usize,
}

impl ChunkData {
    /// Construct an empty chunk anchored at the given compressed-stream
    /// bit offset. The first subchunk is pre-allocated at offset 0 with
    /// zero size, mirroring rapidgzip's `startNewSubchunk` pattern
    /// (GzipChunk.hpp:47-58 init).
    ///
    /// Pulls the underlying Vecs from `chunk_buffer_pool` (mirror of
    /// vendor's per-thread rpmalloc arena recycling
    /// `FasterVector<uint8_t>` allocations — `core/FasterVector.hpp:
    /// 120-128`). On pool miss, `chunk_buffer_pool::take_*` falls
    /// back to a fresh `Vec::with_capacity(cap)`. On chunk drop, the
    /// `Drop` impl returns the Vecs to the pool.
    pub fn new(encoded_offset_bits: usize, configuration: ChunkConfiguration) -> Self {
        use crate::decompress::parallel::chunk_buffer_pool;
        let cap = configuration.max_decoded_chunk_size;
        // `data_with_markers` is allocated lazily — the fast path
        // (window known) never emits markers, so paying for the
        // capacity reservation up front is wasted address space AND
        // wasted page commits if the worker writes anywhere into it
        // (e.g. `Vec::truncate` does not, but a later `extend` would).
        // Slow-path workers call `append_markered` which grows the
        // Vec from zero on demand; the pool's `take_u16` returns
        // recycled Vecs at that point so growth is amortized.
        //
        // Vendor parity: `ChunkData::data_with_markers` is a
        // FasterVector<uint16_t> that's default-constructed (empty)
        // and only grown when the marker pipeline emits markers
        // (ChunkData.hpp uses `subchunks.back().data.push_back(...)`).
        // Allocating max-size up front was a gzippy-specific
        // pre-emptive sizing that defied the vendor pattern.
        Self::new_with_buffers(
            encoded_offset_bits,
            configuration,
            chunk_buffer_pool::take_u16(0),
            chunk_buffer_pool::take_u8(cap),
            chunk_buffer_pool::current_worker_pool_index().unwrap_or(0),
        )
    }

    /// Like [`Self::new`] but consumes caller-provided Vecs. Workers
    /// pool these across chunks via the parallel-drive recycle path —
    /// reuses capacity AND already-committed pages so the per-chunk
    /// page-zero cost drops to one-time-per-worker rather than one-time-
    /// per-chunk. Caller must pass Vecs with `len() == 0` (the
    /// caller-side recycler does this by calling `.clear()`).
    pub fn new_with_buffers(
        encoded_offset_bits: usize,
        configuration: ChunkConfiguration,
        data_with_markers: U16,
        data: U8,
        pool_worker_index: usize,
    ) -> Self {
        debug_assert!(data_with_markers.is_empty());
        debug_assert!(data.is_empty());
        let first_subchunk = Subchunk {
            encoded_offset_bits,
            encoded_size_bits: 0,
            decoded_offset: 0,
            decoded_size: 0,
            window: None,
        };
        Self {
            encoded_offset_bits,
            max_acceptable_start_bit: encoded_offset_bits,
            encoded_size_bits: 0,
            data_with_markers,
            data,
            narrowed: types::u8_with_capacity(0),
            markers_resolved: false,
            subchunks: vec![first_subchunk],
            // Mirror of ChunkData.hpp:561 default-init:
            // `std::vector<CRC32Calculator>( 1 )`.
            crc32s: vec![CRC32Calculator::new()],
            footers: Vec::new(),
            stopped_preemptively: false,
            statistics: ChunkStatistics::default(),
            configuration,
            next_subchunk_start_decoded_offset: 0,
            pool_worker_index,
        }
    }

    /// Whether `expected_start` falls within this chunk's acceptable
    /// start range. Mirror of `rapidgzip::ChunkData::matchesEncodedOffset`
    /// (ChunkData.hpp:396-403). Used by the consumer to accept
    /// speculative chunks whose start is in tolerance rather than
    /// exactly equal to the predecessor's actual end.
    pub fn matches_encoded_offset(&self, expected_start: usize) -> bool {
        self.encoded_offset_bits <= expected_start
            && expected_start <= self.max_acceptable_start_bit
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
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn is_empty(&self) -> bool {
        self.data_with_markers.is_empty() && self.data.is_empty()
    }

    /// Append marker-tagged output (u16 with `MARKER_BASE` bit set on
    /// back-references). Mirror of `ChunkData::append(DecodedVector&&)`
    /// for the markered branch. CRC32 is deferred to `apply_window`
    /// because markers don't have a final byte value yet.
    ///
    /// Statistics note: the per-element marker/non-marker classification
    /// was a separate O(n) scan before `extend_from_slice`. On large
    /// bootstrap outputs (3 MB+ per chunk on silesia), that scan added
    /// measurable overhead. Deferred to `apply_window` time (where every
    /// value is touched anyway); the counters are best-effort and now
    /// report total values in `non_marker_count` only.
    pub fn append_markered(&mut self, values: &[u16]) {
        self.statistics.non_marker_count += values.len() as u64;
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

    /// In-place sibling of `append_clean` for callers that wrote bytes
    /// DIRECTLY into `self.data` (e.g. ISA-L's `next_out` pointed at
    /// `data.as_mut_ptr().add(prev_len)` and produced `written` bytes
    /// there). Saves the `extend_from_slice` memcpy that
    /// `decode_chunk_isal`'s tight loop was paying on every
    /// `isal_inflate` call — measured at ~10% of decode wall time.
    ///
    /// The caller is responsible for:
    /// - Reserving capacity in `self.data` BEFORE writing.
    /// - Calling `self.data.set_len(prev_len + written)` after writing
    ///   so subsequent CRC reads see the bytes (this method does it for
    ///   you if `extend_len` is true).
    ///
    /// Safety contract: bytes in `self.data[prev_len..prev_len+written]`
    /// must be fully initialized at the moment of this call.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn note_clean_bytes_written_in_place(
        &mut self,
        prev_len: usize,
        written: usize,
        extend_len: bool,
    ) {
        if written == 0 {
            return;
        }
        if extend_len {
            unsafe { self.data.set_len(prev_len + written) };
        }
        if self.configuration.crc32_enabled {
            if let Some(last_crc) = self.crc32s.last_mut() {
                last_crc.update(&self.data[prev_len..prev_len + written]);
            }
        }
        self.statistics.non_marker_count += written as u64;
        if let Some(last) = self.subchunks.last_mut() {
            last.decoded_size += written;
        }
    }

    /// Move a whole owned decoded-byte buffer into `data`. Literal port
    /// of `ChunkData::append(deflate::DecodedVector&&)`
    /// (vendor/.../ChunkData.hpp:209-224). CRC32s the whole slice once
    /// and updates statistics — does NOT touch subchunks (rapidgzip
    /// updates `subchunks.back().decodedSize` inside the inner
    /// `readStream` loop per call, before this method runs; see
    /// GzipChunk.hpp:321 vs 379). Callers using the outer/inner pattern
    /// must call `note_inner_decoded_bytes` per inner inflate call and
    /// then `append_owned_buffer` once per outer iteration.
    ///
    /// If `self.data` is empty we use `Vec::append` so the buffer's
    /// allocation is moved in rather than copied — matching C++'s
    /// `std::move` semantics. When `data` already holds bytes we fall
    /// back to `extend_from_slice` (a single contiguous Vec is required
    /// by downstream consumers).
    pub fn append_owned_buffer(&mut self, mut buffer: U8) {
        if self.configuration.crc32_enabled {
            if let Some(last_crc) = self.crc32s.last_mut() {
                last_crc.update(&buffer);
            }
        }
        self.statistics.non_marker_count += buffer.len() as u64;
        if self.data.is_empty() {
            // Zero-copy move of the owned allocation, mirroring
            // BaseType::append(std::move(toAppend)).
            self.data = std::mem::replace(&mut buffer, types::u8_empty());
        } else {
            self.data.extend_from_slice(&buffer);
        }
    }

    /// Mirror of `subchunks.back().decodedSize += nBytesReadPerCall`
    /// (vendor/.../chunkdecoding/GzipChunk.hpp:321). Used by the
    /// outer/inner buffer pattern: each inner inflate call writes some
    /// bytes into the outer buffer; we credit them to the current
    /// subchunk immediately so block-boundary subchunks emitted mid-
    /// buffer carry accurate sizes, even though the buffer's bytes
    /// aren't yet appended to `data` (that happens once per outer iter
    /// via `append_owned_buffer`).
    pub fn note_inner_decoded_bytes(&mut self, n_bytes: usize) {
        if let Some(last) = self.subchunks.last_mut() {
            last.decoded_size += n_bytes;
        }
    }

    /// Variant of `append_block_boundary` that takes an explicit
    /// `decoded_offset` rather than deriving it from
    /// `self.decoded_size()`. Used in the outer/inner buffer pattern
    /// where the running decoded total (alreadyDecoded + nBytesRead in
    /// rapidgzip terms) is more accurate than `data.len()` mid-buffer.
    ///
    /// Mirror of `chunk.appendDeflateBlockBoundary(encodedOffset, decodedOffset)`
    /// at vendor/.../ChunkData.hpp:455-467 — the C++ signature takes
    /// decoded_offset explicitly.
    ///
    /// Returns `true` if a new boundary was appended, `false` if it was
    /// a duplicate of the last (matches the C++ bool return). Dedup
    /// criterion mirrors rapidgzip: the (encodedOffset, decodedOffset)
    /// pair must differ from the last boundary, NOT just the encoded
    /// offset — a stored-block plus dynamic-block pair can land at the
    /// same encoded bit position with different decoded offsets (and
    /// vice versa for empty dynamic blocks at the same decoded pos), so
    /// dedupping on encoded-only loses information.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn append_block_boundary_at(
        &mut self,
        encoded_offset_bits: usize,
        decoded_offset: usize,
    ) -> bool {
        // Rapidgzip's appendDeflateBlockBoundary at ChunkData.hpp:459-461:
        //   blockBoundaries.empty()
        //   || back().encodedOffset != encodedOffset
        //   || back().decodedOffset != decodedOffset
        // We translate that against the LAST subchunk (gzippy's subchunks
        // play the role of rapidgzip's blockBoundaries — see B2).
        if let Some(last) = self.subchunks.last() {
            if last.encoded_offset_bits == encoded_offset_bits
                && last.decoded_offset == decoded_offset
            {
                return false;
            }
        }
        // Vendor's on-the-fly chunk splitting at
        // `GzipChunk.hpp::appendDeflateBlockBoundary` (lines 177-182):
        // only start a new subchunk when the trailing subchunk's
        // decoded byte count crosses `splitChunkSize`. Without this
        // gate, gzippy was pushing a new subchunk for EVERY deflate
        // EOB boundary — on silesia gzip-9, that's ~17 subchunks per
        // chunk vs vendor's ~3, polluting the BlockFinder index space
        // with intra-chunk block offsets (Total Existing: 458 vs
        // vendor 85). The prefetcher then dispatched prefetches at
        // these intra-chunk offsets (5-50 KiB-sized "chunks"), causing
        // 27% useless prefetches and a 3x decode-block-time inflation.
        //
        // The dedup check (`encoded_offset_bits == last`) above still
        // returns `false` for duplicates — i.e., we still "accept" the
        // boundary; we just don't materialize a new Subchunk entry
        // until the spacing threshold fires. Callers that need
        // per-EOB awareness use the dedup return value, not the
        // post-call subchunk count.
        let split_threshold = self.configuration.split_chunk_size;
        let current_subchunk_decoded_size = self
            .subchunks
            .last()
            .map(|last| {
                // For the trailing (open) subchunk, the live size lives
                // on `last.decoded_size` (updated by
                // `note_inner_decoded_bytes`); for already-closed
                // entries it's already the final size. Either way it
                // reflects bytes since the subchunk's `decoded_offset`.
                last.decoded_size
            })
            .unwrap_or(0);
        if current_subchunk_decoded_size < split_threshold {
            // Vendor `GzipChunk.hpp:178` — no split yet. Return `true`
            // to acknowledge a distinct boundary (so the caller's
            // dedup logic flows correctly), but don't push a new
            // Subchunk entry.
            return true;
        }
        if let Some(last) = self.subchunks.last_mut() {
            debug_assert!(encoded_offset_bits >= last.encoded_offset_bits);
            last.encoded_size_bits = encoded_offset_bits - last.encoded_offset_bits;
        }
        self.next_subchunk_start_decoded_offset = decoded_offset;
        self.subchunks.push(Subchunk {
            encoded_offset_bits,
            encoded_size_bits: 0,
            decoded_offset,
            decoded_size: 0,
            window: None,
        });
        true
    }

    /// Literal port of `ChunkData::split(spacing)`
    /// (vendor/.../ChunkData.hpp:632-741). Partitions decoded data into
    /// approximately-equal-sized subchunks of `spacing` bytes each by
    /// choosing the recorded block boundary closest to each ideal
    /// split point.
    ///
    /// Returns a fresh `Vec<Subchunk>` (does NOT mutate `self.subchunks`).
    /// Caller can swap them in via `setSubchunks` (rapidgzip's
    /// ChunkData.hpp:408-412 — represented here as direct field write
    /// `chunk.subchunks = chunk.split(spacing)`).
    ///
    /// Panics if `spacing == 0` (mirrors rapidgzip's
    /// `std::invalid_argument`).
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn split(&self, spacing: usize) -> Vec<Subchunk> {
        if spacing == 0 {
            panic!("ChunkData::split: spacing must be > 0");
        }
        let encoded_end_offset_bits = self.encoded_offset_bits + self.encoded_size_bits;
        let decoded_size = self.decoded_size();
        if self.encoded_size_bits == 0 && decoded_size == 0 {
            return Vec::new();
        }

        let n_blocks = (decoded_size as f64 / spacing as f64).round() as usize;
        let whole_chunk = Subchunk {
            encoded_offset_bits: self.encoded_offset_bits,
            decoded_offset: 0,
            encoded_size_bits: self.encoded_size_bits,
            decoded_size,
            window: None,
        };
        if n_blocks <= 1 || self.subchunks.is_empty() {
            return vec![whole_chunk];
        }

        let perfect_spacing = decoded_size as f64 / n_blocks as f64;
        let mut result: Vec<Subchunk> = Vec::with_capacity(n_blocks + 1);
        let mut last_boundary_enc = self.encoded_offset_bits;
        let mut last_boundary_dec: usize = 0;

        // Treat each `subchunk`'s (encoded_offset_bits, decoded_offset)
        // pair as a candidate block boundary. The first subchunk's
        // boundary is the chunk start (already represented by
        // `last_boundary_*`); subsequent subchunks are the real
        // candidates.
        let boundaries: Vec<(usize, usize)> = self
            .subchunks
            .iter()
            .map(|s| (s.encoded_offset_bits, s.decoded_offset))
            .collect();

        for i_subchunk in 1..n_blocks {
            let perfect_decompressed_offset = (i_subchunk as f64 * perfect_spacing) as usize;
            let mut closest_idx = 0usize;
            let mut closest_diff = usize::MAX;
            for (idx, &(_, dec)) in boundaries.iter().enumerate() {
                let diff = dec.abs_diff(perfect_decompressed_offset);
                if diff < closest_diff {
                    closest_diff = diff;
                    closest_idx = idx;
                }
            }
            // Skip over duplicate decoded_offsets (pigz empty blocks),
            // taking the LAST boundary with the same decoded_offset.
            while closest_idx + 1 < boundaries.len()
                && boundaries[closest_idx].1 == boundaries[closest_idx + 1].1
            {
                closest_idx += 1;
            }
            let (closest_enc, closest_dec) = boundaries[closest_idx];
            if closest_dec <= last_boundary_dec {
                continue;
            }
            if closest_enc <= last_boundary_enc {
                panic!(
                    "ChunkData::split: encoded offset must strictly increase with decoded offset"
                );
            }
            let decoded_offset = result
                .last()
                .map(|s| s.decoded_offset + s.decoded_size)
                .unwrap_or(0);
            result.push(Subchunk {
                encoded_offset_bits: last_boundary_enc,
                decoded_offset,
                encoded_size_bits: closest_enc - last_boundary_enc,
                decoded_size: closest_dec - last_boundary_dec,
                window: None,
            });
            last_boundary_enc = closest_enc;
            last_boundary_dec = closest_dec;
        }

        // Tail subchunk from last boundary to chunk end.
        if last_boundary_dec < decoded_size || result.is_empty() {
            let decoded_offset = result
                .last()
                .map(|s| s.decoded_offset + s.decoded_size)
                .unwrap_or(0);
            result.push(Subchunk {
                encoded_offset_bits: last_boundary_enc,
                decoded_offset,
                encoded_size_bits: encoded_end_offset_bits - last_boundary_enc,
                decoded_size: decoded_size - last_boundary_dec,
                window: None,
            });
        } else if last_boundary_dec == decoded_size {
            if let Some(last) = result.last_mut() {
                last.encoded_size_bits = encoded_end_offset_bits - last.encoded_offset_bits;
            }
        }

        result
    }

    /// Migrate the marker-free trailing portion of `data_with_markers`
    /// into the front of `data`. Mirror of rapidgzip's
    /// `DecodedData::cleanUnmarkedData` (vendor/.../DecodedData.hpp:492-516).
    ///
    /// After this call, `data_with_markers` ends with an actual marker
    /// (or is empty), so `apply_window` does the minimum work. Bytes
    /// migrated to `data` are CRC'd now (since they weren't CRC'd at
    /// `append_markered` time) and the result is combined with the
    /// existing `crc32s[0]` so the running CRC still covers the
    /// in-order output.
    pub fn clean_unmarked_data(&mut self) {
        let split_at = match self
            .data_with_markers
            .iter()
            .rposition(|&v| v >= MARKER_BASE)
        {
            Some(last_marker_pos) => last_marker_pos + 1,
            None => 0,
        };
        if split_at >= self.data_with_markers.len() {
            return;
        }
        let prefix_len = self.data_with_markers.len() - split_at;
        let existing_len = self.data.len();
        let new_len = existing_len + prefix_len;

        // Reuse self.data's pre-faulted allocation (worker scratch
        // from parallel_drive::drive_two_pass; cap ==
        // max_decoded_chunk_size). `reserve` is a no-op when cap
        // already covers — the typical case. Avoids the page-fault
        // churn the prior implementation caused on its fresh
        // `Vec<u8>` allocation (perf flamegraph 2026-05-18: this
        // function drove ~8% of total CPU via `clear_page_erms`).
        self.data.reserve(prefix_len);
        // SAFETY: the bytes in [0, new_len) are fully initialized
        // before any read by (a) `copy_within` reading from the
        // old self.data range [0, existing_len) which contains
        // ISA-L-written bytes, and (b) the narrowing loop writing
        // u8s into [0, prefix_len). No code reads uninit.
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.data.set_len(new_len);
        }
        // Shift the ISA-L bulk right by prefix_len so the front
        // prefix_len slots are free for the narrowed clean tail.
        // `copy_within` is a single memmove — well-vectorized in
        // libc on x86_64.
        if existing_len > 0 {
            self.data.copy_within(0..existing_len, prefix_len);
        }
        // Narrow u16 -> u8 directly into the front of self.data.
        // All values in [split_at..] are < 256 by the rposition
        // search above (last >= MARKER_BASE element ends at split_at).
        for (i, &v) in self.data_with_markers[split_at..].iter().enumerate() {
            self.data[i] = v as u8;
        }

        // CRC the migrated bytes (they were NOT CRC'd at
        // append_markered time). Result must reflect the in-order
        // output. crc32s[0] currently covers original_data; after
        // this it covers (clean_tail | original_data). Mirror of
        // vendor's `crc32s.front().prepend( crc32 )` after
        // `cleanUnmarkedData` in `ChunkData::finalize`
        // (vendor/rapidgzip/.../ChunkData.hpp:426-435). Source slice
        // is the just-narrowed bytes already at self.data[..prefix_len]
        // — no second narrow pass, no scratch Vec.
        if self.configuration.crc32_enabled && !self.crc32s.is_empty() {
            let mut migrated_crc = CRC32Calculator::new();
            migrated_crc.update(&self.data[..prefix_len]);
            self.crc32s[0].prepend(&migrated_crc);
        }

        self.data_with_markers.truncate(split_at);
    }

    /// Resolve the trailing 32 KiB of this chunk's decoded output —
    /// the "tail window" required to seed the successor chunk's
    /// decoder. Mirrors `rapidgzip::deflate::DecodedData::getLastWindow`
    /// (vendor/.../DecodedData.hpp:394-398), which forwards to
    /// `getWindowAt(previousWindow, size())` (DecodedData.hpp:401-490).
    ///
    /// Unlike `apply_window` + `populate_subchunk_windows`, this only
    /// touches the LAST 32 KiB of the concatenated
    /// `(previousWindow | dataWithMarkers | data)` view, then maps any
    /// markers in that tail through `MapMarkers` (MarkerReplacement.hpp:15-46).
    /// Cost is O(W) regardless of chunk size — designed to run
    /// synchronously on the consumer thread so the tail window can be
    /// published to the `WindowMap` BEFORE the (much slower) full
    /// `applyWindow` task is queued. This mirrors the consumer's
    /// critical-path split at
    /// `GzipChunkFetcher::queueChunkForPostProcessing`
    /// (vendor/.../GzipChunkFetcher.hpp:553-583), where line 572 calls
    /// `chunkData->getLastWindow( *previousWindow )` on the consumer
    /// thread and line 579 enqueues the actual `applyWindow` to the
    /// thread pool via `submitTaskWithHighPriority`.
    ///
    /// `predecessor_window` must be exactly 32 KiB. (The vendor allows
    /// shorter previousWindow at start-of-stream — represented in
    /// gzippy by chunk 0's all-zero `[u8; 32768]` seed in
    /// `chunk_fetcher::drive` — so we keep the same width invariant
    /// here and let the caller pad with zeros.)
    pub fn get_last_window(&self, predecessor_window: &[u8]) -> [u8; 32768] {
        const W: usize = 32768;
        debug_assert_eq!(
            predecessor_window.len(),
            W,
            "get_last_window requires a full 32 KiB predecessor window \
             (vendor DecodedData.hpp:402: `DecodedVector window( MAX_WINDOW_SIZE )`)"
        );

        // Direct port of `getWindowAt(previousWindow, skipBytes = size())`
        // (vendor/.../DecodedData.hpp:401-490). We want the last W bytes
        // of the concatenated view (predecessor_window | dataWithMarkers | data),
        // so `skip_bytes == decoded_size()`. The vendor's two-arm copy
        // (prefilled-from-previousWindow then copyFromDataWithMarkers
        // then data) collapses for `skip_bytes == size()` into:
        //   - if decoded_size < W: take the trailing (W - decoded_size)
        //     bytes of predecessor_window into the head of the window,
        //     then ALL of dataWithMarkers (mapped) and data into the tail.
        //   - else: take only the last W bytes of (dataWithMarkers | data),
        //     mapping any marker we land on through MapMarkers.
        let dwm_len = self.data_with_markers.len();
        let data_len = self.data.len();
        let total = dwm_len + data_len;

        let mut window = [0u8; W];

        if total >= W {
            // Last W bytes are entirely inside (dataWithMarkers | data).
            // Compute absolute start `s` in [0, total) such that s + W = total.
            // Then translate to per-segment offsets.
            //
            // Vendor parity: DecodedData.hpp:439:
            //   `offset = skipBytes - remainingBytes;`
            // where `skipBytes == size()` and `remainingBytes == W` here,
            // so `offset = total - W`. The subsequent loops walk
            // dataWithMarkers then data with that initial offset
            // (vendor lines 445-485).
            let mut offset = total - W; // start within (dwm | data)
            let mut written: usize = 0;

            // Segment 1: from dataWithMarkers, mapping markers.
            // Vendor DecodedData.hpp:445-469 (`copyFromDataWithMarkers`).
            // The C++ has separate FULL_WINDOW true/false specializations
            // (lines 465-469) for whether previousWindow itself is full
            // 32 KiB — we always pass a full window so the FULL_WINDOW=true
            // branch applies (vendor MarkerReplacement.hpp:15-46 dispatch
            // skips the bounds-check on `value - MAX_WINDOW_SIZE`).
            if offset < dwm_len {
                let take = (dwm_len - offset).min(W - written);
                for i in 0..take {
                    let v = self.data_with_markers[offset + i];
                    // MapMarkers semantics (vendor MarkerReplacement.hpp:24-42):
                    //   value <= 0xFF → literal byte
                    //   value >= MAX_WINDOW_SIZE → predecessor_window[v - MAX_WINDOW_SIZE]
                    //   else (0x100..MAX_WINDOW_SIZE) → invalid
                    window[written + i] = if v >= MARKER_BASE {
                        predecessor_window[(v - MARKER_BASE) as usize]
                    } else {
                        v as u8
                    };
                }
                written += take;
                offset = 0;
            } else {
                offset -= dwm_len;
            }

            // Segment 2: from data (already clean bytes).
            // Vendor DecodedData.hpp:471-485.
            if written < W && offset < data_len {
                let take = (data_len - offset).min(W - written);
                window[written..written + take].copy_from_slice(&self.data[offset..offset + take]);
                written += take;
            }
            debug_assert_eq!(written, W, "get_last_window underran the tail buffer");
        } else {
            // total < W: window head comes from predecessor_window's tail.
            // Vendor DecodedData.hpp:411-434 (`if ( skipBytes < MAX_WINDOW_SIZE )`).
            let from_prev = W - total;
            window[..from_prev].copy_from_slice(&predecessor_window[total..]);
            let mut written = from_prev;

            // Then ALL of dataWithMarkers, mapped through MapMarkers.
            for v in &self.data_with_markers {
                window[written] = if *v >= MARKER_BASE {
                    predecessor_window[(*v - MARKER_BASE) as usize]
                } else {
                    *v as u8
                };
                written += 1;
            }
            // Then ALL of data.
            window[written..written + data_len].copy_from_slice(&self.data);
        }

        window
    }

    /// Populate the `window` field of every subchunk with the 32 KiB
    /// window required to resume decode at that subchunk's start.
    /// Must be called AFTER `apply_window` resolves markers — the
    /// per-subchunk windows are derived from the chunk's own resolved
    /// output prefixed by the predecessor's tail window.
    ///
    /// `dwm_bytes` must equal the u8 narrow of `data_with_markers`
    /// post-`apply_window` (i.e., the buffer the consumer will later
    /// write to the output). Callers narrow once and reuse the buffer
    /// here so this routine does NOT re-narrow `data_with_markers`.
    ///
    /// Literal port of the window-emplacement step in rapidgzip's
    /// `appendSubchunksToIndexes` cascade
    /// (vendor/.../GzipChunkFetcher.hpp:560-580): for each subchunk's
    /// `decodedOffset`, the window is the 32 KiB immediately preceding
    /// that offset in the concatenated `predecessor_window ++ data_with_markers ++ data`.
    pub fn populate_subchunk_windows(&mut self, predecessor_window: &[u8], dwm_bytes: &[u8]) {
        const W: usize = 32768;
        debug_assert_eq!(predecessor_window.len(), W);
        debug_assert!(
            self.data_with_markers.iter().all(|v| *v < MARKER_BASE),
            "populate_subchunk_windows requires apply_window already ran"
        );
        debug_assert_eq!(
            dwm_bytes.len(),
            self.data_with_markers.len(),
            "populate_subchunk_windows requires narrowed dwm_bytes matching data_with_markers length"
        );

        let dwm_len = dwm_bytes.len();

        for sc in self.subchunks.iter_mut() {
            // Build window for offset `sc.decoded_offset`. Source bytes
            // come from (predecessor_window | dwm_bytes | data), taking
            // the last 32 KiB before `sc.decoded_offset` — i.e., from
            // absolute index `sc.decoded_offset` (in the concatenated
            // source after `predecessor_window`) over `W` bytes.
            //
            // Vendor parity: `DecodedData::getWindowAt`
            // (vendor/.../DecodedData.hpp:401-490) walks the same
            // (previousWindow | dataWithMarkers chunks | data chunks)
            // concatenation. Vendor's C++ loops over per-chunk inner
            // segments because DecodedData stores both `data` and
            // `dataWithMarkers` as `std::vector<FasterVector<...>>`
            // (lists of contiguous buffers). gzippy's single-Vec layout
            // for both fields lets us replace vendor's element loops
            // with three `copy_from_slice` calls — same total work as
            // the C++, just expressed as bulk memcpys the compiler can
            // forward to SSE/AVX `mov`s instead of the previous
            // per-byte `match abs { … }` chain.
            let mut window = [0u8; W];
            let needed_start = sc.decoded_offset; // see ascii diagram above
            let mut written: usize = 0;

            // Segment 1: bytes from `predecessor_window[needed_start..W]`.
            if needed_start < W {
                let take = W - needed_start;
                let take = take.min(W - written);
                window[written..written + take]
                    .copy_from_slice(&predecessor_window[needed_start..needed_start + take]);
                written += take;
            }

            // Segment 2: bytes from `dwm_bytes`.
            if written < W {
                let abs = needed_start + written;
                let dwm_offset = abs.saturating_sub(W);
                if dwm_offset < dwm_len {
                    let take = (dwm_len - dwm_offset).min(W - written);
                    window[written..written + take]
                        .copy_from_slice(&dwm_bytes[dwm_offset..dwm_offset + take]);
                    written += take;
                }
            }

            // Segment 3: bytes from `data`.
            if written < W {
                let abs = needed_start + written;
                let data_offset = abs.saturating_sub(W + dwm_len);
                if data_offset < self.data.len() {
                    let take = (self.data.len() - data_offset).min(W - written);
                    window[written..written + take]
                        .copy_from_slice(&self.data[data_offset..data_offset + take]);
                    written += take;
                }
            }
            // Trailing bytes (if subchunk extends past `data.len()`) stay
            // zero — matches the previous `unwrap_or(0)` fallback.
            let _ = written;

            sc.window = Some(Arc::new(window));
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
        // Open a fresh calculator for the next stream's bytes. If no
        // more bytes follow, the trailing empty calculator's CRC == 0
        // and the consumer's combine treats it as a no-op. Mirror of
        // vendor's `crc32s.emplace_back()` at ChunkData.hpp:478.
        self.crc32s.push(CRC32Calculator::new());
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
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn append_block_boundary(&mut self, encoded_offset_bits: usize) -> bool {
        // Derive decoded_offset from current data size (caller wasn't
        // forced to supply it). Forward to append_block_boundary_at,
        // which carries the (encoded, decoded) dedup semantic from
        // rapidgzip's appendDeflateBlockBoundary (ChunkData.hpp:455-467).
        let current_decoded = self.decoded_size();
        self.append_block_boundary_at(encoded_offset_bits, current_decoded)
    }

    /// Finalize at end of decode. Sets `encoded_size_bits` for the
    /// chunk and its trailing subchunk. **Does NOT touch
    /// `max_acceptable_start_bit`** — vendor's pattern is that the
    /// worker sets max at decode time (`offset.second` per
    /// `GzipChunk.hpp:722`: where the boundary was actually found),
    /// and finalize leaves it alone. For fast-path decodes with a
    /// known predecessor window, `offset.first == offset.second ==
    /// encoded_offset_bits`, so max stays at encoded_offset_bits
    /// (exact-match semantics in `matches_encoded_offset`). Setting
    /// max to `end_encoded_offset_bits` here was the bug that made
    /// `try_take_prefetched(partition_offset)` return a chunk whose
    /// range INCLUDED the next chunk's start, causing the consumer
    /// to re-use the predecessor chunk and `set_encoded_offset` to
    /// zero out `encoded_size_bits` → premature EOF break.
    pub fn finalize(&mut self, end_encoded_offset_bits: usize) {
        debug_assert!(end_encoded_offset_bits >= self.encoded_offset_bits);
        self.encoded_size_bits = end_encoded_offset_bits - self.encoded_offset_bits;
        if let Some(last) = self.subchunks.last_mut() {
            debug_assert!(end_encoded_offset_bits >= last.encoded_offset_bits);
            last.encoded_size_bits = end_encoded_offset_bits - last.encoded_offset_bits;
        }
        // append_block_boundary may have emitted a trailing subchunk at the
        // chunk-end position (e.g. multi-stream END_OF_STREAM right at chunk
        // end). Such a subchunk has zero encoded and decoded size and would
        // duplicate the next chunk's first subchunk in the BlockMap.
        while self.subchunks.len() > 1 {
            let last = self.subchunks.last().unwrap();
            if last.encoded_size_bits == 0 && last.decoded_size == 0 {
                self.subchunks.pop();
            } else {
                break;
            }
        }
        // Mirrors rapidgzip's ChunkData::finalize calling cleanUnmarkedData
        // (vendor/.../ChunkData.hpp ~422). Cuts down apply_window's work
        // by moving any marker-free trailing region into `data`.
        self.clean_unmarked_data();
    }

    /// Literal port of `ChunkData::setEncodedOffset`
    /// (vendor/rapidgzip/.../ChunkData.hpp:601-629). After a chunk
    /// completes, the consumer calls this with the REAL start offset
    /// (predecessor's actual end) to re-anchor the chunk from its
    /// speculative-seed start. Collapses the [encoded, max] match range
    /// to a single point.
    ///
    /// Panics if `offset` is not in `[encoded_offset_bits, max_acceptable_start_bit]`
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
            self.max_acceptable_start_bit,
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
        self.max_acceptable_start_bit = offset;
        // Adjust first subchunk: its encoded_offset becomes the new
        // chunk start, and its encoded_size spans to the next subchunk
        // (or to the chunk end if there's only one subchunk). If the new
        // offset coincides with subchunks[1].encoded_offset_bits, the
        // collapsed first subchunk has zero encoded_size and would
        // duplicate subchunks[1]'s key in the BlockMap — drop it instead.
        if !self.subchunks.is_empty() {
            let next_offset = if self.subchunks.len() >= 2 {
                self.subchunks[1].encoded_offset_bits
            } else {
                end_offset
            };
            if self.subchunks.len() >= 2 && offset == next_offset {
                self.subchunks.remove(0);
            } else {
                let first = &mut self.subchunks[0];
                first.encoded_offset_bits = offset;
                first.encoded_size_bits = next_offset.saturating_sub(offset);
            }
        }
    }

    /// Find the subchunk whose `encoded_offset_bits` exactly matches
    /// `expected_start`, returning its `decoded_offset` (i.e. how many
    /// leading bytes to skip in this chunk's output). Returns None if
    /// no subchunk matches; the consumer then falls back to
    /// authoritative re-dispatch.
    #[allow(dead_code)] // vendor parity or unit-test surface
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

/// Return `data` and `data_with_markers` to the shared
/// `chunk_buffer_pool` so the next worker on any thread can reuse
/// the already-faulted pages. Mirror of vendor's rpmalloc
/// auto-recycle for `FasterVector<uint8_t>` allocations
/// (`core/FasterVector.hpp:120-128`).
///
/// `Clone` is derived: clones produce independent Vecs, each
/// returning to the pool on its own drop. Cross-thread safe: pool
/// is a `static Mutex<Vec<...>>`.
impl Drop for ChunkData {
    fn drop(&mut self) {
        use crate::decompress::parallel::chunk_buffer_pool;
        let data = std::mem::replace(&mut self.data, types::u8_empty());
        let dwm = std::mem::replace(&mut self.data_with_markers, types::u16_empty());
        let narrowed = std::mem::replace(&mut self.narrowed, types::u8_empty());
        chunk_buffer_pool::return_u8_to_worker(self.pool_worker_index, data);
        chunk_buffer_pool::return_u8_to_worker(self.pool_worker_index, narrowed);
        chunk_buffer_pool::return_u16_to_worker(self.pool_worker_index, dwm);
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
    fn split_returns_single_subchunk_when_spacing_exceeds_decoded() {
        let mut chunk = ChunkData::new(0, small_config());
        chunk.append_clean(&[0u8; 100]);
        chunk.finalize(400);
        // spacing = 1000, decoded = 100, n_blocks = round(0.1) = 0 → 0
        let parts = chunk.split(1000);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].decoded_size, 100);
    }

    #[test]
    fn split_partitions_into_approximately_equal_subchunks() {
        let mut chunk = ChunkData::new(0, small_config());
        // Emit a chunk with multiple block boundaries at known decoded offsets.
        chunk.append_clean(&[0u8; 100]);
        chunk.append_block_boundary(800);
        chunk.append_clean(&[0u8; 100]);
        chunk.append_block_boundary(1600);
        chunk.append_clean(&[0u8; 100]);
        chunk.append_block_boundary(2400);
        chunk.append_clean(&[0u8; 100]);
        chunk.finalize(3200);
        // Total decoded = 400 bytes, spacing = 100 → n_blocks = 4
        let parts = chunk.split(100);
        assert!(!parts.is_empty());
        let total_decoded: usize = parts.iter().map(|s| s.decoded_size).sum();
        assert_eq!(total_decoded, 400);
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
        chunk.max_acceptable_start_bit = 150;

        // Re-anchor to 130 (within [100, 150]).
        chunk.set_encoded_offset(130);

        assert_eq!(chunk.encoded_offset_bits, 130);
        assert_eq!(chunk.max_acceptable_start_bit, 130);
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
        chunk.max_acceptable_start_bit = 100;

        chunk.set_encoded_offset(50);

        assert_eq!(chunk.encoded_offset_bits, 50);
        assert_eq!(chunk.subchunks[0].encoded_offset_bits, 50);
        assert_eq!(chunk.subchunks[0].encoded_size_bits, 500 - 50);
    }

    #[test]
    fn set_encoded_offset_drops_collapsed_first_subchunk() {
        // Speculative chunk decoded with seed=100, found block boundary at
        // bit 400 → subchunks[0]=(enc=100, size=300), subchunks[1]=(enc=400, ...).
        // If the consumer re-anchors to offset=400 (matching the second
        // boundary exactly), the first subchunk would collapse to zero
        // encoded_size and duplicate subchunks[1]'s key in the BlockMap.
        let mut chunk = ChunkData::new(100, small_config());
        chunk.append_clean(&[0u8; 50]);
        chunk.append_block_boundary(400);
        chunk.append_clean(&[0u8; 50]);
        chunk.finalize(700);
        // Stretch the speculative match range so set_encoded_offset(400) is valid.
        chunk.max_acceptable_start_bit = 500;
        assert_eq!(chunk.subchunks.len(), 2);

        chunk.set_encoded_offset(400);

        assert_eq!(chunk.encoded_offset_bits, 400);
        // The collapsed first subchunk must be gone — otherwise the
        // BlockMap push will panic on duplicate offset.
        assert_eq!(chunk.subchunks.len(), 1);
        assert_eq!(chunk.subchunks[0].encoded_offset_bits, 400);
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
        // Counters now lumped into non_marker_count (per-value
        // classification was an O(n) overhead removed for bootstrap perf;
        // see chunk_data.rs append_markered doc).
        assert_eq!(
            chunk.statistics.marker_count + chunk.statistics.non_marker_count,
            50
        );
    }

    #[test]
    fn append_markered_counts_marker_values() {
        let mut chunk = ChunkData::new(0, small_config());
        let values: Vec<u16> = vec![0u16, 1, 2, MARKER_BASE, MARKER_BASE + 5, 6];
        chunk.append_markered(&values);
        // Total count preserved; per-class split moved to apply_window.
        assert_eq!(
            chunk.statistics.marker_count + chunk.statistics.non_marker_count,
            6
        );
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
        assert_eq!(chunk.crc32s[0].crc32(), expected.finalize());
        assert_eq!(chunk.subchunks[0].decoded_size, 11);
    }

    #[test]
    fn append_block_boundary_below_threshold_does_not_emit_subchunk() {
        // Vendor parity (GzipChunk.hpp::appendDeflateBlockBoundary
        // lines 177-182): only emit a new subchunk when the trailing
        // subchunk's decoded byte count crosses split_chunk_size. A
        // boundary below threshold is acknowledged (return value
        // signals it was distinct) but the subchunk list does NOT
        // grow. This is the gating fix for the BlockFinder index-
        // space pollution that drove silesia parallel SM speculation
        // overhead from ~16% to ~3x the necessary chunks.
        let mut chunk = ChunkData::new(0, small_config()); // split = 100
        chunk.append_clean(&[0u8; 50][..]); // 50 bytes < threshold
        chunk.append_block_boundary(400);
        assert_eq!(
            chunk.subchunks.len(),
            1,
            "no new subchunk below split_chunk_size threshold"
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
        // max_acceptable_start_bit stays at construction-time value (0)
        // — vendor's pattern: max is set by the worker (offset.second
        // per GzipChunk.hpp:722), not by finalize. For fast-path
        // chunks max == encoded_offset_bits (exact-match semantics).
        assert_eq!(chunk.max_acceptable_start_bit, 0);
        assert_eq!(chunk.decoded_offset_for(0), Some(0));
        assert_eq!(chunk.decoded_offset_for(2000), Some(200));
        assert_eq!(chunk.decoded_offset_for(1234), None);
    }

    #[test]
    fn get_last_window_matches_apply_window_tail_when_total_exceeds_w() {
        // Sanity: get_last_window must produce the same trailing 32 KiB
        // as running apply_window + reading the last 32 KiB of the
        // resolved (dwm | data). This is the property
        // populate_subchunk_windows + apply_window jointly rely on; if
        // get_last_window disagreed, the successor chunk's dict would
        // diverge from rapidgzip.
        const W: usize = 32768;
        let mut prev = [0u8; W];
        for (i, b) in prev.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }

        let mut chunk_a = ChunkData::new(0, small_config());
        // ~40 KiB total so trailing 32 KiB sits inside (dwm | data).
        // 8 KiB of markers (some literal, some back-refs into prev),
        // then 32 KiB of clean bytes.
        let mut markers: Vec<u16> = Vec::with_capacity(8192);
        for i in 0..8192 {
            if i % 3 == 0 {
                markers.push(MARKER_BASE + (i as u16 % 1024));
            } else {
                markers.push((i & 0xff) as u16);
            }
        }
        chunk_a.append_markered(&markers);
        let mut clean = vec![0u8; W];
        for (i, b) in clean.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(31)) & 0xff) as u8;
        }
        chunk_a.append_clean(&clean);

        // Reference: apply_window + take last 32 KiB.
        let mut chunk_b = chunk_a.clone();
        crate::decompress::parallel::apply_window::apply_window(&mut chunk_b, &prev);
        let mut reference = [0u8; W];
        // total > W so reference is the last W bytes of (dwm | data).
        let dwm_b_len = chunk_b.data_with_markers.len();
        let total_b = dwm_b_len + chunk_b.data.len();
        let start = total_b - W;
        let mut written = 0;
        if start < dwm_b_len {
            let take = (dwm_b_len - start).min(W);
            for (i, slot) in reference.iter_mut().enumerate().take(take) {
                *slot = chunk_b.data_with_markers[start + i] as u8;
            }
            written += take;
            reference[written..].copy_from_slice(&chunk_b.data[..W - written]);
        } else {
            let off = start - dwm_b_len;
            reference.copy_from_slice(&chunk_b.data[off..off + W]);
        }

        let tail = chunk_a.get_last_window(&prev);
        assert_eq!(&tail[..], &reference[..]);
    }

    #[test]
    fn get_last_window_handles_total_smaller_than_w() {
        // total < W → window head is `predecessor_window`'s tail.
        // Mirror of vendor DecodedData.hpp:411-434.
        const W: usize = 32768;
        let mut prev = [0u8; W];
        for (i, b) in prev.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }

        let mut chunk = ChunkData::new(0, small_config());
        // 100 bytes of clean output → total = 100 < W.
        let clean = vec![0xABu8; 100];
        chunk.append_clean(&clean);

        let tail = chunk.get_last_window(&prev);
        // Head: prev[100..] should occupy window[..W-100].
        assert_eq!(&tail[..W - 100], &prev[100..]);
        // Tail: all 0xAB.
        assert!(tail[W - 100..].iter().all(|b| *b == 0xAB));
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
        assert_eq!(chunk.crc32s[0].crc32(), 0);
    }
}
