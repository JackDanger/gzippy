#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

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

use crate::decompress::parallel::compressed_vector::{CompressedVector, CompressionType};
use crate::decompress::parallel::crc32::CRC32Calculator;
use crate::decompress::parallel::rpmalloc_alloc::types::U8;
use crate::decompress::parallel::segmented_buffer::SegmentedU8;
use crate::decompress::parallel::segmented_markers::SegmentedU16;
use crate::decompress::parallel::window_map::Window;

/// One deflate-block-aligned slice of a chunk's decoded output.
/// Port of `rapidgzip::ChunkData::Subchunk` (ChunkData.hpp:138-145).
#[derive(Debug, Clone)]
pub struct Subchunk {
    pub encoded_offset_bits: usize,
    pub encoded_size_bits: usize,
    pub decoded_offset: usize,
    pub decoded_size: usize,
    /// Compressed 32 KiB window at this subchunk's end (vendor
    /// `Subchunk::window` = `SharedWindow`, ChunkData.hpp:143).
    pub window: Option<Window>,
    /// Vendor `Subchunk::newlineCount` (ChunkData.hpp:142).
    pub newline_count: Option<usize>,
    /// Vendor `Subchunk::usedWindowSymbols` (ChunkData.hpp:144). Populated at
    /// subchunk split/finalize via `getUsedWindowSymbols`; cleared in
    /// `populate_subchunk_windows` after sparsity is applied.
    pub used_window_symbols: Vec<bool>,
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
    /// Vendor `Configuration::windowSparsity` (ChunkData.hpp:104).
    pub window_sparsity: bool,
    /// Vendor `Configuration::windowCompressionType` (ChunkData.hpp:103).
    pub window_compression_type: Option<CompressionType>,
    /// Member-level output/compressed expansion ceiling: `ceil((ISIZE /
    /// compressed_len) × 1.25)` as an integer factor, minimum 2. Set once per
    /// gzip stream from the footer ISIZE and total compressed length in
    /// `sm_driver`; 0 = unknown → `finish_decode_chunk_isal_oracle` falls back
    /// to the historical 8× factor.
    ///
    /// Purpose: size the ISA-L clean-tail upfront output reserve proportionally
    /// to the actual data ratio instead of a fixed 8×. On near-incompressible
    /// corpora (model: ~1.3×) the 8× reserve over-allocates ~6× per chunk;
    /// with O(T) concurrent workers the excess page-fault/dTLB pressure
    /// collapses per-worker ISA-L throughput under concurrency. Sizing from the
    /// known ratio eliminates the over-reservation while preserving a 1.25×
    /// headroom margin for intra-member chunk variation.
    ///
    /// ISIZE is mod 2^32: for raw output > 4 GiB the ratio may under-estimate
    /// (ISIZE wraps). That is safe — the initial just under-sizes and the
    /// GROW_BYTES loop fills the gap incrementally with no correctness risk.
    pub expansion_ratio_ceil: u16,
}

impl Default for ChunkConfiguration {
    fn default() -> Self {
        let split = 4 * 1024 * 1024;
        Self {
            split_chunk_size: split,
            max_decoded_chunk_size: 20 * split,
            crc32_enabled: true,
            window_sparsity: true,
            window_compression_type: None,
            expansion_ratio_ceil: 0, // unknown — falls back to 8× in finish_decode_chunk_isal_oracle
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
    /// (Design B) The encoded bit offset the WORKER actually decoded byte 0
    /// from — i.e. `decode_start` (== `max_acceptable_start_bit` for a
    /// range-speculative chunk). For non-speculative chunks this equals
    /// `encoded_offset_bits`. Records WHERE the chunk's decoded bytes
    /// truly begin, independent of the speculative claimed-start range
    /// `[encoded_offset_bits, max_acceptable_start_bit]`. Audit-only; the
    /// speculative-window KEY is derived from `encoded_offset_bits +
    /// encoded_size_bits` (see `chunk_fetcher::run_decode_task`), which is
    /// the value the consumer recomputes after `set_encoded_offset`.
    pub decode_origin_bit: usize,
    pub encoded_size_bits: usize,
    /// Marker-tagged prefix. Each u16 < MARKER_BASE is a literal byte
    /// (`v as u8`); values ≥ MARKER_BASE are direct indices into the
    /// predecessor's 32 KiB window from the OLDEST byte (mirrors
    /// rapidgzip MapMarkers: window[v - MARKER_BASE]). Cross-chunk
    /// last 32 KiB window. `apply_window` (next module) resolves these
    /// in place against a known window.
    // std `Vec<u16>` (not the rpmalloc `U16`): the bootstrap decodes into this
    // buffer directly (zero-copy merge), and std-Vec decode is ~3% faster
    // per-byte than rpmalloc here (measured). Warm-cycled via the dedicated
    // `chunk_buffer_pool::{take,return}_std_u16` retained pool.
    pub data_with_markers: SegmentedU16,
    /// Clean byte suffix. All bytes here were decoded with a known
    /// window (set via StreamingInflateWrapper::set_window) so no markers
    /// were emitted. CRC32'd at append time.
    ///
    /// CONTIGUOUS (2026-06-04): the clean decoded bytes live in ONE
    /// contiguous allocation (see [`SegmentedU8`], now contiguous-backed),
    /// grown by amortized reserve. This is the FAITHFUL port of vendor's
    /// CLEAN-data storage: `DecodedData::append` (`DecodedData.hpp:278-289`)
    /// stores the clean `data` in single contiguous per-append buffers and
    /// the comment there states forcing them to 128 KiB "makes no sense".
    /// The prior FOOTPRINT-ALIGN revision wrongly applied vendor's *marker*
    /// 128 KiB-segment pattern to the clean buffer; that measured 3.26×
    /// DTLB-walks / 1.42× cycles vs rapidgzip at equal instruction count —
    /// the memory-bound gap. Reverted to contiguous; the RSS regression is
    /// accepted for speed parity (the prime directive). The u16 MARKER
    /// buffer (`data_with_markers`) stays 128 KiB-segmented — that one
    /// matches vendor's `dataWithMarkers` and is correct.
    pub data: SegmentedU8,
    /// Length of the resolved-marker prefix, narrowed IN PLACE inside
    /// `data_with_markers`'s own u16 backing store (low byte of each u16
    /// element). FOOTPRINT-ALIGN: replaces the former separate `narrowed`
    /// u8 buffer — a buffer rapidgzip does NOT have (it resolves markers
    /// in place and reinterprets the u16 buffer as u8,
    /// DecodedData.hpp:325-388). After `narrow_markers_in_place`, the first
    /// `narrowed_len` BYTES of `data_with_markers`'s allocation are the
    /// resolved u8 output; `narrowed_bytes()` exposes them as `&[u8]`.
    /// 0 when there were no markers.
    pub narrowed_len: usize,
    /// CRC32 of `narrowed`, computed on the post-process worker (parallel)
    /// instead of on the consumer (serial). Vendor parity: `ChunkData::
    /// applyWindow` (vendor/.../ChunkData.hpp:313-328) fuses the CRC32
    /// pass into apply_window on the worker; gzippy previously deferred
    /// this to `drain_one_pending` and paid pclmulqdq cycles on the
    /// single-threaded consumer. Move to the worker so it parallelizes
    /// across the 16-thread pool — for silesia (~400 MB marker bytes /
    /// pclmulqdq ~15 GB/s) saves ~25 ms of consumer-serial wall.
    ///
    /// `default()` is the empty-stream CRC sentinel; valid for chunks
    /// with no narrowed bytes (where the consumer skips the append).
    pub narrowed_crc: CRC32Calculator,
    /// True if `apply_window` has already been run on this chunk —
    /// either on the worker thread before send (when the predecessor's
    /// window was published in time) or by an earlier consumer pass.
    /// Consumer skips its own `apply_window` call when set, saving
    /// the redundant scan of `data_with_markers`. Mirror of vendor's
    /// `m_markersResolved` flag pattern at ChunkData.hpp.
    #[allow(dead_code)] // set by apply_window; read in future seekable path
    pub markers_resolved: bool,
    /// Window-map key used when `apply_window` last ran (`None` if never).
    /// Consumer reuses eager/cache-resolved chunks only when this equals the
    /// handoff predecessor key (vendor `hasBeenPostProcessed` byte-identity).
    pub resolved_pred_key: Option<usize>,
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
    /// Scaffolding for Option A (window pre-fill). When > 0, the first
    /// `data_prefix_len` bytes of `self.data` are the predecessor's
    /// 32 KiB sliding-window image written by `prefill_window_prefix`,
    /// NOT decoded output. `decoded_size()` subtracts this; the
    /// scaffolding-time invariant is that callers must call
    /// `trim_window_prefix()` before the chunk is finalized so that
    /// `self.data` once again contains only decoded bytes.
    ///
    /// Segment-native A3 (production): `prefill_window_prefix` writes the
    /// predecessor 32 KiB window image into segment 0; the consumer skips
    /// it via `write_payload_skipping_prefix` / `append_payload_iovecs`.
    pub data_prefix_len: usize,
    /// STEP-1 U16-preserving ceiling oracle (GZIPPY_MARKER_CEILING_U16): when a
    /// window-absent chunk is decoded through the seeded-clean path, this records
    /// the decoded byte count so the CONSUMER thread can inject the phantom u16
    /// write + resolve traffic serially (the pessimistic resolve-location arm).
    /// 0 in production / for all other arms. Byte-transparent (only read by the
    /// ceiling instrument).
    pub phantom_ceiling_len: usize,
    /// DoS/OOM guard: hard upper bound on this chunk's decoded output bytes,
    /// derived from the compressed input length (`input_len ×
    /// MAX_DEFLATE_EXPANSION + slack`). Malformed input can drive the deflate
    /// decoder to fabricate phantom blocks from zero-padding past end-of-input
    /// (both bit readers zero-pad on underflow), producing UNBOUNDED output from
    /// a tiny input and OOMing the process. A valid DEFLATE stream expands by at
    /// most ~1032:1, so this ceiling never trips on legitimate data but caps the
    /// malformed runaway. `usize::MAX` (no bound) until a decoder sets it via
    /// [`Self::set_output_ceiling_for_input`]. See `ensure_within_output_ceiling`.
    pub output_ceiling: usize,
}

/// Maximum bytes a single DEFLATE byte can expand to. The theoretical worst
/// case is 1032:1 (a length-258 back-reference encoded in as few as ~2 Huffman
/// bits). We never need a tighter bound — the goal is only to stop a malformed
/// stream from allocating without limit, not to predict the exact output size.
pub const MAX_DEFLATE_EXPANSION: usize = 1032;

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
        // FOOTPRINT-ALIGN (2026-06-02) — REVERTED after measurement.
        // Right-sizing `data`'s initial capacity from max_decoded_chunk_size
        // (80 MiB) down to split_chunk_size (4 MiB) cut allocator TRAFFIC 55%
        // (memlife T8: data alloc 8.64 GB→1.10 GB, rpmalloc-huge 3.27 GB→159
        // MB) but on the frozen interleaved A/B it REGRESSED the wall
        // (T8-file 1.297→1.510 vs rapidgzip; T16 1.677→1.922), RAISED minflt
        // 232k→363k, and barely moved peak maxrss (1025→1010 MB). MECHANISM:
        // gzippy stores decoded bytes in ONE contiguous growing `data` Vec.
        // A small initial cap forces the `reserve(ALLOCATION_CHUNK_SIZE)`
        // growth loop (chunk_decode.rs:318) to realloc+memcpy+RE-FAULT as the
        // chunk grows past 4 MiB — the extra faults are the regression. This
        // is the OPPOSITE of vendor: rapidgzip's `DecodedData::data` is a
        // std::vector<DecodedVector> — a LIST of fixed 128 KiB chunks that is
        // NEVER reallocated/copied (DecodedData.hpp:231-289), so it has no
        // grow-realloc cost regardless of total size. The real fix (now
        // implemented) is the segmented-buffer rewrite: `data` is a
        // `SegmentedU8` (list of 128 KiB segments, never realloc'd), so no
        // initial reservation is needed at all — segments are allocated on
        // demand from the worker's recycled segment pool as the decoder
        // fills them, exactly like vendor's `DecodedData::append`.
        let _ = configuration.max_decoded_chunk_size;
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
        // `data` segments come from the worker's recycled segment pool
        // (warm 128 KiB pages) via `SegmentedU8::writable_tail`; start empty.
        Self::new_with_buffers(
            encoded_offset_bits,
            configuration,
            SegmentedU16::default(),
            SegmentedU8::default(),
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
        data_with_markers: SegmentedU16,
        data: SegmentedU8,
        pool_worker_index: usize,
    ) -> Self {
        debug_assert!(data_with_markers.is_empty());
        debug_assert!(data.is_empty());
        // Instrumentation (Step A, 2026-05-28): track the peak number of
        // simultaneously-live ChunkData. This is the in-flight buffer
        // working set that drives page-fault churn — each live chunk holds
        // a ~12 MiB u8 + (slow path) up-to-24 MiB u16 Vec, both > rpmalloc's
        // ~3.94 MiB huge-alloc threshold so they munmap-on-free / re-fault
        // on next take. MAX_LIVE_CHUNKS sizes any bounded-pool / ring fix
        // and tests the "prefetch depth, not topology" churn thesis.
        // Clones=0 in production (Arc::try_unwrap 36/0) so this stays
        // balanced against the Drop decrement below.
        let live = LIVE_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let prev_max = MAX_LIVE_CHUNKS.fetch_max(live, std::sync::atomic::Ordering::Relaxed);
        // LIFECYCLE-SPLIT: peak live is always at a construct — snapshot the
        // per-holder gauges whenever this construct sets a new max.
        if live > prev_max {
            lc_record_peak(live);
        }
        let first_subchunk = Subchunk {
            encoded_offset_bits,
            encoded_size_bits: 0,
            decoded_offset: 0,
            decoded_size: 0,
            window: None,
            newline_count: None,
            used_window_symbols: Vec::new(),
        };
        Self {
            encoded_offset_bits,
            max_acceptable_start_bit: encoded_offset_bits,
            decode_origin_bit: encoded_offset_bits,
            encoded_size_bits: 0,
            data_with_markers,
            data,
            narrowed_len: 0,
            narrowed_crc: CRC32Calculator::new(),
            markers_resolved: false,
            resolved_pred_key: None,
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
            data_prefix_len: 0,
            phantom_ceiling_len: 0,
            output_ceiling: usize::MAX,
        }
    }

    /// Reconstruct a `ChunkData` from precomputed segments + metadata,
    /// for the decode-BYPASS experiment (`decode_bypass.rs`). NOT a
    /// production path — gated to the bypass harness, used to replay a
    /// captured decode with ~zero inner-Huffman CPU so the wall reveals
    /// coordination cost. The caller supplies the already-CRC'd `crc0`
    /// covering all of `data` (mirrors the post-finalize invariant).
    #[allow(clippy::too_many_arguments)]
    pub fn from_bypass_parts(
        encoded_offset_bits: usize,
        max_acceptable_start_bit: usize,
        decode_origin_bit: usize,
        encoded_size_bits: usize,
        stopped_preemptively: bool,
        data_prefix_len: usize,
        data_with_markers: Vec<u16>,
        data: U8,
        crc0: CRC32Calculator,
        subchunks: Vec<Subchunk>,
        footers: Vec<Footer>,
        configuration: ChunkConfiguration,
    ) -> Self {
        let live = LIVE_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        MAX_LIVE_CHUNKS.fetch_max(live, std::sync::atomic::Ordering::Relaxed);
        let subchunks = if subchunks.is_empty() {
            vec![Subchunk {
                encoded_offset_bits,
                encoded_size_bits,
                decoded_offset: 0,
                decoded_size: 0,
                window: None,
                newline_count: None,
                used_window_symbols: Vec::new(),
            }]
        } else {
            subchunks
        };
        // FOOTPRINT-ALIGN: the bypass harness still produces a contiguous
        // `data: U8`; copy it into the segmented buffer.
        let mut seg = SegmentedU8::default();
        seg.extend_from_slice(&data);
        let mut dwm = SegmentedU16::default();
        dwm.push_slice(&data_with_markers);
        Self {
            encoded_offset_bits,
            max_acceptable_start_bit,
            decode_origin_bit,
            encoded_size_bits,
            data_with_markers: dwm,
            data: seg,
            narrowed_len: 0,
            narrowed_crc: CRC32Calculator::new(),
            markers_resolved: false,
            resolved_pred_key: None,
            subchunks,
            crc32s: vec![crc0],
            footers,
            stopped_preemptively,
            statistics: ChunkStatistics::default(),
            configuration,
            next_subchunk_start_decoded_offset: 0,
            pool_worker_index: 0,
            data_prefix_len,
            phantom_ceiling_len: 0,
            output_ceiling: usize::MAX,
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
        // `data` is the chunk's clean decoded suffix (segmented).
        let data_skip = self.data_prefix_len;
        let decoded_data_len = self.data.len().saturating_sub(data_skip);
        let mut out = [0u8; W];
        if decoded_data_len >= W {
            self.data.copy_last_32k(&mut out);
            Some(out)
        } else if total >= W {
            // Tail straddles markers + clean. Markers in the trailing
            // W bytes mean we can't build a clean window without
            // apply_window resolving them first; caller has to wait.
            let from_data = decoded_data_len;
            let from_markers = W - from_data;
            let m_start = self.data_with_markers.len() - from_markers;
            for i in 0..from_markers {
                let v = self.data_with_markers.get(m_start + i)?;
                if v >= crate::decompress::parallel::replace_markers::MARKER_BASE {
                    return None;
                }
            }
            let window_stub = [0u8; 32768];
            self.data_with_markers.resolve_range_into_buf(
                m_start,
                from_markers,
                &window_stub,
                &mut out[..from_markers],
            );
            self.data.copy_last_into(&mut out[from_markers..]);
            Some(out)
        } else {
            // Less than W bytes total. We don't try to combine with the
            // predecessor's window here; the consumer can do that if it
            // needs to. Return None to signal "no clean tail-window
            // available standalone."
            None
        }
    }

    /// Vec-producing twin of `last_32kib_window`. Builds the clean tail
    /// window DIRECTLY into a freshly-allocated 32 KiB `Vec<u8>` (no
    /// stack-array intermediate), so the consumer's publish path can hand
    /// ownership straight into the `WindowMap` via `insert_owned_none` —
    /// exactly one heap allocation, zero redundant memcpy. Byte-identical
    /// to `last_32kib_window().map(|a| a.to_vec())`.
    pub fn last_32kib_window_vec(&self) -> Option<Vec<u8>> {
        const W: usize = 32768;
        let total = self.decoded_size();
        if total == 0 {
            return None;
        }
        let data_skip = self.data_prefix_len;
        let decoded_data_len = self.data.len().saturating_sub(data_skip);
        if decoded_data_len >= W {
            return Some(self.data.copy_last_32k_vec());
        }
        if total >= W {
            // Tail straddles markers + clean. Markers in the trailing W
            // bytes mean we can't build a clean window standalone.
            let from_data = decoded_data_len;
            let from_markers = W - from_data;
            let m_start = self.data_with_markers.len() - from_markers;
            for i in 0..from_markers {
                let v = self.data_with_markers.get(m_start + i)?;
                if v >= crate::decompress::parallel::replace_markers::MARKER_BASE {
                    return None;
                }
            }
            let mut out = vec![0u8; W];
            // All literals in range — one segment walk, not per-byte get().
            let window_stub = [0u8; 32768];
            self.data_with_markers.resolve_range_into_buf(
                m_start,
                from_markers,
                &window_stub,
                &mut out[..from_markers],
            );
            self.data.copy_last_into(&mut out[from_markers..]);
            debug_assert_eq!(out.len(), W);
            Some(out)
        } else {
            None
        }
    }

    /// DoS/OOM guard: arm the decoded-output ceiling from the compressed input
    /// length this chunk may consume. A valid DEFLATE stream expands by at most
    /// [`MAX_DEFLATE_EXPANSION`]:1, so any decode producing more than
    /// `input_len × MAX_DEFLATE_EXPANSION + slack` is malformed (the decoder is
    /// fabricating phantom blocks from zero-padding past end-of-input). The
    /// slack absorbs the tiny-input edge (one window + headroom) so small valid
    /// members never trip. Callers pass the whole deflate-data length, which
    /// bounds the whole member's output.
    #[inline]
    pub fn set_output_ceiling_for_input(&mut self, input_len: usize) {
        const SLACK: usize = 64 * 1024; // one 32 KiB window + headroom for tiny inputs
        self.output_ceiling = input_len
            .saturating_mul(MAX_DEFLATE_EXPANSION)
            .saturating_add(SLACK);
    }

    /// DoS/OOM guard: error if decoded output has exceeded the armed ceiling.
    /// Called once per decode-loop iteration; cheap (one comparison). Returns
    /// the implausible produced size so the caller can surface a terminal
    /// corruption error instead of OOMing.
    #[inline]
    pub fn ensure_within_output_ceiling(&self) -> Result<(), usize> {
        let produced = self.decoded_size();
        if produced > self.output_ceiling {
            Err(produced)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn decoded_size(&self) -> usize {
        let marker_bytes = if self.narrowed_len > 0 {
            self.narrowed_len
        } else {
            self.data_with_markers.len()
        };
        marker_bytes + self.data.len().saturating_sub(self.data_prefix_len)
    }

    /// Vendor `ChunkData::containsMarkers` — any unresolved marker symbol in
    /// `data_with_markers` (values ≥ `MARKER_BASE`).
    ///
    /// `narrowed_len > 0` means `apply_window` already resolved+narrowed the
    /// markers in place (u8 written into the low bytes of the u16 backing) under
    /// the view-based emit model — the buffer is NOT cleared (its narrowed bytes
    /// are the output views, recycled on writev). In that state `all_resolved()`
    /// would misread the stale u16 high bytes, so treat narrowed == resolved
    /// (vendor swaps the resolved buffer out, so its `containsMarkers` is false
    /// post-`applyWindow` too).
    #[inline]
    pub fn contains_markers(&self) -> bool {
        self.narrowed_len == 0
            && !self.data_with_markers.is_empty()
            && !self.data_with_markers.all_resolved()
    }

    /// Vendor `ChunkData::hasBeenPostProcessed` (ChunkData.hpp:496-501).
    /// `require_newline_count` mirrors `configuration.newlineCharacter.has_value()`.
    #[inline]
    pub fn has_been_post_processed(&self, require_newline_count: bool) -> bool {
        !self.subchunks.is_empty()
            && !self.contains_markers()
            && self.subchunks.iter().all(|sc| {
                sc.window.is_some()
                    && sc.used_window_symbols.is_empty()
                    && (sc.newline_count.is_some() || !require_newline_count)
            })
    }

    #[inline]
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn is_empty(&self) -> bool {
        self.data_with_markers.is_empty() && self.data.len() == self.data_prefix_len
    }

    /// Pre-fill `self.data[0..window.len()]` with the predecessor's
    /// 32 KiB sliding-window image and record `data_prefix_len`.
    /// Scaffolding for Option A — see `git history (campaign plan, removed)` §6
    /// "copy_match_windowed slow-path elimination."
    ///
    /// Caller contract:
    /// - Must be called BEFORE any decoded byte is appended to
    ///   `self.data` (the prefill writes at offset 0).
    /// - `window.len()` must be ≤ 32 KiB (the DEFLATE max distance).
    /// - The chunk's `data` buffer must have capacity ≥
    ///   `window.len() + max_decoded_chunk_size` — pool allocation
    ///   already grows to `max_decoded_chunk_size`, so today this
    ///   is satisfied for `window.len() ≤ MAX_DISTANCE` (32 KiB <<
    ///   typical chunk cap of 4 MiB).
    /// - Caller must call `trim_window_prefix()` before the chunk is
    ///   finalized (consumed by `apply_window`/`clean_unmarked_data`
    ///   etc.) so that downstream code sees only decoded bytes in
    ///   `self.data`.
    ///
    /// Segment-native A3: prefill segment 0 with the predecessor window image.
    pub fn prefill_window_prefix(&mut self, window: &[u8]) {
        debug_assert!(
            self.data.is_empty(),
            "prefill_window_prefix must run before any decoded bytes are appended"
        );
        debug_assert!(
            self.data_prefix_len == 0,
            "prefill_window_prefix called twice on the same chunk"
        );
        debug_assert!(
            window.len() <= 32 * 1024,
            "window prefix exceeds DEFLATE max distance of 32 KiB"
        );
        if window.is_empty() {
            return;
        }
        self.data.prefill_window_prefix(window);
        self.data_prefix_len = window.len();
    }

    /// No-op: the window prefix stays in `data` through finalize; the consumer
    /// skips `data_prefix_len` bytes when writing (A4).
    pub fn trim_window_prefix(&mut self) {
        let _ = self.data_prefix_len;
    }

    /// Copy the A3 predecessor window prefix (32 KiB) when window-present
    /// decode seeded `data[0..32K]`. Used to publish `WindowMap` entries at
    /// confirmed handoff keys (vendor `get(blockOffset)` parity).
    pub fn window_prefix_vec(&self) -> Option<Vec<u8>> {
        use crate::decompress::parallel::marker_inflate::MAX_WINDOW_SIZE;
        if self.data_prefix_len < MAX_WINDOW_SIZE {
            return None;
        }
        let mut buf = vec![0u8; MAX_WINDOW_SIZE];
        self.data.copy_range_into(0, &mut buf);
        Some(buf)
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
    #[allow(dead_code)]
    pub fn append_markered(&mut self, values: &[u16]) {
        // `worker.append_markered` span — wraps the memcpy of the
        // bootstrap's u16 marker output into the chunk's
        // `data_with_markers`. For bootstraps that filled the whole
        // chunk (no clean-window handoff), values can be several
        // million u16 = 10+ MB memcpy per call.
        let _tv2 = crate::decompress::parallel::trace_v2::SpanGuard::begin_with(
            "worker.append_markered",
            &format!(r#""len":{}"#, values.len()),
        );
        self.statistics.non_marker_count += values.len() as u64;
        // `allocator_api2::vec::Vec::extend_from_slice` does NOT
        // specialize for `Copy` source types — it falls back to
        // the generic `extend → Cloned::next → Option::cloned →
        // Clone::clone` iterator chain. For `Vec<u16>` that's a
        // per-element u16 load through 4 inlined adapter frames
        // instead of a memcpy. Replace with the explicit
        // `reserve` + `copy_nonoverlapping` + `set_len` shape that
        // `std::vec::Vec`'s `SpecExtend` specialization would
        // have produced.
        self.data_with_markers.push_slice(values);
        if let Some(last) = self.subchunks.last_mut() {
            last.decoded_size += values.len();
        }
    }

    /// Append clean (already-resolved) output bytes. Mirror of
    /// `ChunkData::append(DecodedDataView)` for the non-marker branch.
    /// CRC32'd immediately if enabled. The CRC always feeds the most
    /// recent stream's hasher (`crc32s.last_mut()`); cross-stream byte
    /// runs are split by `append_footer`.
    #[allow(dead_code)]
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

    /// Reserve `n` bytes of CONTIGUOUS spare capacity in the clean `data`
    /// buffer up-front (one allocation), so subsequent [`Self::append_clean`]
    /// runs land in one region without per-run amortized regrow. Used by the
    /// gzippy-native FOLD path (`decode_chunk_unified_marker`) to give the
    /// post-flip clean tail a single contiguous output region (vendor decodes the
    /// clean tail into one `DecodedData` buffer, DecodedData.hpp:278-289).
    /// Byte-transparent — only affects allocation cadence, never the decoded
    /// bytes; an under-reserve is a safe amortized regrow, never corruption.
    pub fn reserve_clean(&mut self, n: usize) {
        self.data.reserve(n);
    }

    /// Append clean output that arrives as u16 ring values from a
    /// post-flip `Block` drain. After the flip (`!contains_marker_bytes()`) every value is
    /// guaranteed `< 256`, so narrowing to u8 is lossless — this is the
    /// unified-decoder analogue of [`append_clean`] for the merged clean
    /// tail (the same `crc32s.last_mut()` + `subchunks.last` accounting,
    /// no second decode engine and no 32 KiB window copy). Mirrors the
    /// `append`-into-`data` branch of vendor `ChunkData::append`.
    ///
    /// `debug_assert`s the all-clean invariant; in release a stray marker
    /// would narrow to its low byte (a corruption the CRC check catches).
    pub fn append_clean_narrowed(&mut self, values: &[u16]) {
        thread_local! {
            static NARROW_SCRATCH: std::cell::RefCell<Vec<u8>> =
                std::cell::RefCell::new(Vec::with_capacity(64 * 1024));
        }
        NARROW_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            scratch.reserve(values.len());
            debug_assert!(
                values.iter().all(|&v| v < 256),
                "append_clean_narrowed received a marker value (decoder did not flip clean)"
            );
            scratch.extend(values.iter().map(|&v| v as u8));
            if self.configuration.crc32_enabled {
                if let Some(last_crc) = self.crc32s.last_mut() {
                    last_crc.update(&scratch);
                }
            }
            self.statistics.non_marker_count += scratch.len() as u64;
            self.data.extend_from_slice(&scratch);
            if let Some(last) = self.subchunks.last_mut() {
                last.decoded_size += scratch.len();
            }
        });
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
    #[allow(dead_code)]
    pub fn append_owned_buffer(&mut self, buffer: U8) {
        // A1 invariant: the `self.data.is_empty()` fast path below
        // would `mem::replace` away an in-place window prefix. Caller
        // must `trim_window_prefix` before handing ownership of a new
        // bulk buffer to the chunk. (Or A3 can pre-fill the prefix
        // into `buffer` itself BEFORE this call so no swap is needed.)
        debug_assert_eq!(
            self.data_prefix_len, 0,
            "trim_window_prefix before append_owned_buffer (zero-copy path would discard prefix)"
        );
        if self.configuration.crc32_enabled {
            if let Some(last_crc) = self.crc32s.last_mut() {
                last_crc.update(&buffer);
            }
        }
        self.statistics.non_marker_count += buffer.len() as u64;
        // Segmented `data`: copy the owned buffer's bytes in (segment-
        // distributed). The former zero-copy `mem::replace` of a single
        // contiguous Vec no longer applies — segments are the unit now.
        self.data.extend_from_slice(&buffer);
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
    /// Vendor `minimumSplitChunkSize` (ChunkData.hpp:516-518).
    #[inline]
    pub fn minimum_split_chunk_size(&self) -> usize {
        self.configuration.split_chunk_size / 4
    }

    /// Vendor `ChunkData::windowCompressionType` (ChunkData.hpp:197-207).
    pub fn window_compression_type(&self) -> CompressionType {
        if let Some(ct) = self.configuration.window_compression_type {
            return ct;
        }
        if self.configuration.window_sparsity
            || self.decoded_size().saturating_mul(8) > self.encoded_size_bits.saturating_mul(2)
        {
            CompressionType::Zlib
        } else {
            CompressionType::None
        }
    }

    /// Vendor `GzipChunk::determineUsedWindowSymbolsForLastSubchunk`
    /// (chunkdecoding/GzipChunk.hpp:61-97).
    fn determine_used_window_symbols_for_last_subchunk(&mut self, deflate_data: &[u8]) {
        if !self.configuration.window_sparsity {
            return;
        }
        let Some(last) = self.subchunks.last_mut() else {
            return;
        };
        if last.encoded_size_bits == 0 {
            return;
        }
        // Past all early exits — we are about to run the 32 KiB marker-engine scan.
        // Count each such execution so the kill-switch effect is observable in VERBOSE.
        SPARSITY_DECODE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let start_bit = last
            .encoded_offset_bits
            .saturating_add(last.encoded_size_bits);
        last.used_window_symbols =
            crate::decompress::parallel::used_window_symbols::get_used_window_symbols(
                deflate_data,
                start_bit,
            );
        if last.used_window_symbols.iter().all(|&used| !used) {
            last.used_window_symbols.clear();
        }
    }

    /// Vendor `GzipChunk::finalizeWindowForLastSubchunk` (GzipChunk.hpp:100-133).
    fn finalize_window_for_last_subchunk(&mut self, deflate_data: Option<&[u8]>) {
        if let Some(data) = deflate_data {
            self.determine_used_window_symbols_for_last_subchunk(data);
        }
    }

    /// Vendor `GzipChunk::finalizeChunk` small-subchunk merge (GzipChunk.hpp:141-153).
    fn merge_small_trailing_subchunk_if_needed(&mut self) {
        if self.subchunks.len() < 2 {
            return;
        }
        let min_size = self.minimum_split_chunk_size();
        let Some(last) = self.subchunks.last() else {
            return;
        };
        if last.decoded_size >= min_size {
            return;
        }
        let last = self.subchunks.pop().expect("len >= 2");
        if let Some(prev) = self.subchunks.last_mut() {
            prev.encoded_size_bits += last.encoded_size_bits;
            prev.decoded_size += last.decoded_size;
            prev.used_window_symbols.clear();
            prev.window = None;
        }
    }

    pub fn append_block_boundary_at(
        &mut self,
        encoded_offset_bits: usize,
        decoded_offset: usize,
        deflate_data: Option<&[u8]>,
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
        self.finalize_window_for_last_subchunk(deflate_data);
        self.next_subchunk_start_decoded_offset = decoded_offset;
        self.subchunks.push(Subchunk {
            encoded_offset_bits,
            encoded_size_bits: 0,
            decoded_offset,
            decoded_size: 0,
            window: None,
            newline_count: None,
            used_window_symbols: Vec::new(),
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
            newline_count: None,
            used_window_symbols: Vec::new(),
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
                newline_count: None,
                used_window_symbols: Vec::new(),
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
                newline_count: None,
                used_window_symbols: Vec::new(),
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
        let split_at = match self.data_with_markers.rposition_last_marker() {
            Some(last_marker_pos) => last_marker_pos + 1,
            None => 0,
        };
        if split_at >= self.data_with_markers.len() {
            return;
        }
        let prefix_len = self.data_with_markers.len() - split_at;

        if self.data_prefix_len == 0 {
            // View-list convergence (vendor cleanUnmarkedData, DecodedData.hpp:
            // 502-505): narrow the marker-free clean tail u16→u8 DIRECTLY into a
            // fresh FRONT segment of `data` and O(1)-insert it at the logical
            // front — ONE copy, no temp `Vec`, no bulk memmove. (The old path
            // narrowed into a temp `Vec` then `prepend_bytes`-copied it again,
            // and on flip-to-clean chunks also memmoved the whole existing
            // bulk; that double-copy/memmove was the ~10% finalize migration
            // tax vs rapidgzip's 0.11% `ChunkData::finalize`.)
            self.data
                .prepend_narrowed_clean_tail(&self.data_with_markers, split_at, prefix_len);

            // CRC the migrated bytes (NOT CRC'd at append_markered time) and
            // prepend into crc32s[0] so it covers (clean_tail | original_data).
            // Vendor: `crc32s.front().prepend(crc32)` (ChunkData.hpp:426-435).
            // The narrowed bytes are read straight out of their destination
            // front segment — no re-copy.
            if self.configuration.crc32_enabled && !self.crc32s.is_empty() {
                let mut migrated_crc = CRC32Calculator::new();
                migrated_crc.update(self.data.first_front_bytes());
                self.crc32s[0].prepend(&migrated_crc);
            }
        } else {
            // Window-present (`data_prefix_len > 0`) path: the clean tail must be
            // inserted after the window-image prefix in the contiguous bulk.
            // This path carries no front prepends; narrow into a temp buffer and
            // insert. (Markers + a real window rarely co-occur in production —
            // a seeded window makes decode clean — so this is the cold path.)
            let mut narrowed_prefix: Vec<u8> = Vec::with_capacity(prefix_len);
            let mut skip = split_at;
            for seg in self.data_with_markers.segments() {
                if skip >= seg.len() {
                    skip -= seg.len();
                    continue;
                }
                narrowed_prefix.extend(seg[skip..].iter().map(|&v| v as u8));
                skip = 0;
            }
            if self.configuration.crc32_enabled && !self.crc32s.is_empty() {
                let mut migrated_crc = CRC32Calculator::new();
                migrated_crc.update(&narrowed_prefix);
                self.crc32s[0].prepend(&migrated_crc);
            }
            self.data
                .insert_logical_at(self.data_prefix_len, &narrowed_prefix);
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
        let mut window = [0u8; 32768];
        self.fill_last_window_into(predecessor_window, &mut window);
        window
    }

    /// Vec-producing twin of `get_last_window`. One 32 KiB heap allocation,
    /// zero stack-array → Vec memcpy.
    pub fn get_last_window_vec(&self, predecessor_window: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; 32768];
        self.fill_last_window_into(predecessor_window, &mut v);
        v
    }

    /// Core window builder shared by [`Self::get_last_window`] and
    /// [`Self::get_last_window_vec`]. Direct port of
    /// `getWindowAt(previousWindow, skipBytes = size())`
    /// (vendor/.../DecodedData.hpp:401-490).
    fn fill_last_window_into(&self, predecessor_window: &[u8], window: &mut [u8]) {
        const W: usize = 32768;
        debug_assert_eq!(
            predecessor_window.len(),
            W,
            "get_last_window requires a full 32 KiB predecessor window \
             (vendor DecodedData.hpp:402: `DecodedVector window( MAX_WINDOW_SIZE )`)"
        );
        debug_assert_eq!(window.len(), W);
        self.copy_window_at_chunk_offset(predecessor_window, self.decoded_size(), window);
    }

    /// Port of `DecodedData::getWindowAt(previousWindow, skipBytes)`.
    /// `chunk_offset` is a byte index into this chunk's decoded output
    /// (0 = first emitted byte). Returns the 32 KiB window immediately
    /// preceding that offset in `predecessor_window ‖ markers ‖ data`.
    fn copy_window_at_chunk_offset(
        &self,
        predecessor_window: &[u8],
        chunk_offset: usize,
        out: &mut [u8],
    ) {
        const W: usize = 32768;
        debug_assert_eq!(out.len(), W);

        let marker_len = if self.narrowed_len > 0 {
            self.narrowed_len
        } else {
            self.data_with_markers.len()
        };
        let data_skip = self.data_prefix_len;
        let payload_len = self.data.len().saturating_sub(data_skip);

        // Fast path: clean-only payload with enough bytes — one tail memcpy.
        if marker_len == 0 && payload_len >= W && chunk_offset >= payload_len {
            self.data.copy_last_into(out);
            return;
        }

        let needed_start = chunk_offset;
        let mut written: usize = 0;

        if needed_start < W {
            let take = (W - needed_start).min(W - written);
            out[written..written + take]
                .copy_from_slice(&predecessor_window[needed_start..needed_start + take]);
            written += take;
        }

        if written < W && marker_len > 0 {
            let abs = needed_start + written;
            let m_off = abs.saturating_sub(W);
            if m_off < marker_len {
                let take = (marker_len - m_off).min(W - written);
                if self.narrowed_len > 0 {
                    self.data_with_markers.copy_narrowed_u8_range_into(
                        m_off,
                        take,
                        &mut out[written..written + take],
                    );
                } else {
                    self.data_with_markers.resolve_range_into_buf(
                        m_off,
                        take,
                        predecessor_window,
                        &mut out[written..written + take],
                    );
                }
                written += take;
            }
        }

        if written < W {
            let abs = needed_start + written;
            let data_off = abs.saturating_sub(W + marker_len);
            if data_off < payload_len {
                let take = (payload_len - data_off).min(W - written);
                self.data
                    .copy_range_into(data_skip + data_off, &mut out[written..written + take]);
                written += take;
            }
        }
        debug_assert_eq!(written, W, "copy_window_at_chunk_offset underran");
    }

    /// Populate the `window` field of every subchunk with the 32 KiB
    /// window required to resume decode at that subchunk's start.
    ///
    /// Vendor `applyWindow` is `narrow (DecodedData.hpp:325-363) → swap+views
    /// (:365-388)`; there is NO output-size copy. gzippy mirrors that: the
    /// narrowed marker bytes stay in `data_with_markers` (the u8 view of the
    /// u16 backing) and are emitted directly via `append_narrowed_iovecs`.
    /// So this step may run with `narrowed_len > 0` (un-merged state) —
    /// `copy_window_at_chunk_offset` reads the 3-part view
    /// `predecessor ‖ markers ‖ data` and branches on `narrowed_len > 0`,
    /// so `getWindowAt` is correct either merged or un-merged.
    ///
    /// Literal port of the window-emplacement step in rapidgzip's
    /// `appendSubchunksToIndexes` cascade
    /// (vendor/.../GzipChunkFetcher.hpp:560-580).
    pub fn populate_subchunk_windows(&mut self, predecessor_window: &[u8]) {
        const W: usize = 32768;
        debug_assert_eq!(predecessor_window.len(), W);
        let compression = self.window_compression_type();
        let offsets: Vec<usize> = self.subchunks.iter().map(|sc| sc.decoded_offset).collect();

        for (i, &decoded_offset) in offsets.iter().enumerate() {
            let mut window = [0u8; W];
            self.copy_window_at_chunk_offset(predecessor_window, decoded_offset, &mut window);

            // Vendor `ChunkData::applyWindow` sparsity (ChunkData.hpp:341-345).
            let sc = &mut self.subchunks[i];
            if sc.used_window_symbols.len() == W {
                for (j, used) in sc.used_window_symbols.iter().enumerate() {
                    if !used {
                        window[j] = 0;
                    }
                }
            }
            sc.used_window_symbols.clear();
            sc.window = Some(Arc::new(CompressedVector::from_bytes(&window, compression)));
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
        self.append_block_boundary_at(encoded_offset_bits, current_decoded, None)
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
        self.finalize_with_deflate(end_encoded_offset_bits, None);
    }

    /// Like [`Self::finalize`] but runs vendor subchunk sparsity when
    /// `deflate_data` is provided (GzipChunk.hpp:136-158).
    pub fn finalize_with_deflate(
        &mut self,
        end_encoded_offset_bits: usize,
        deflate_data: Option<&[u8]>,
    ) {
        // A4: a window-image prefix is allowed to remain in self.data
        // through finalize and into the consumer. Downstream methods
        // (`get_last_window`, `populate_subchunk_windows`,
        // `last_32kib_window`) handle `data_prefix_len > 0` via
        // explicit prefix-skipping, and the consumer write site reads
        // `&chunk.data[chunk.data_prefix_len..]`.
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
        self.merge_small_trailing_subchunk_if_needed();
        self.finalize_window_for_last_subchunk(deflate_data);
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
/// Live / peak simultaneously-live ChunkData (Step A instrumentation).
/// Read via GZIPPY_VERBOSE. MAX_LIVE_CHUNKS = peak in-flight depth, which
/// determines the page-fault working set and sizes a bounded-pool fix.
pub static LIVE_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MAX_LIVE_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// RSS-SPLIT ISOLATION INSTRUMENT (env GZIPPY_RSS_SPLIT=1; byte-transparent,
/// no effect on output). Accounts, per chunk at consumer-write time, the
/// payload bytes that flowed through the u16 `data_with_markers` 2× buffer
/// (`narrowed_len`) vs the u8 `data` 1× buffer (`decoded_data_len`). The u16
/// marker buffer is held at 2× its narrowed byte count for the chunk's life
/// (vendor DecodedData.hpp:374-379 keeps the 2× too). With a small file where
/// MAX_LIVE_CHUNKS == total chunks (all chunks cached/live at peak), the peak
/// resident u16 marker bytes ≈ 2 × NARROWED_BYTES_TOTAL and the freeable upper
/// half ≈ NARROWED_BYTES_TOTAL. CONSERVATION: NARROWED+CLEAN == total decoded.
pub static NARROWED_BYTES_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static CLEAN_DATA_BYTES_TOTAL: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Number of chunks accounted (non-inert proof: must be > 0 and == chunk count).
pub static RSS_SPLIT_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// True iff the RSS-split isolation instrument is armed.
#[inline]
pub fn rss_split_enabled() -> bool {
    std::env::var_os("GZIPPY_RSS_SPLIT").is_some()
}

/// Account one chunk's payload split (consumer write path).
#[inline]
pub fn rss_split_account(narrowed_len: usize, clean_data_len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    NARROWED_BYTES_TOTAL.fetch_add(narrowed_len as u64, Relaxed);
    CLEAN_DATA_BYTES_TOTAL.fetch_add(clean_data_len as u64, Relaxed);
    RSS_SPLIT_CHUNKS.fetch_add(1, Relaxed);
}

/// Emit the RSS-split isolation report (Gate-0 self-tests: non-inert +
/// conservation). Returns false if the instrument was inert.
pub fn rss_split_report(total_decoded: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let narrowed = NARROWED_BYTES_TOTAL.load(Relaxed);
    let clean = CLEAN_DATA_BYTES_TOTAL.load(Relaxed);
    let chunks = RSS_SPLIT_CHUNKS.load(Relaxed);
    let max_live = MAX_LIVE_CHUNKS.load(Relaxed);
    let sum = narrowed + clean;
    let non_inert = chunks > 0 && sum > 0;
    let conserves = sum == total_decoded;
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[rss_split] chunks={chunks} max_live_chunks={max_live} \
         narrowed(u16 2x src)={:.2}MiB clean(u8 1x)={:.2}MiB sum={:.2}MiB \
         total_decoded={:.2}MiB",
        mib(narrowed),
        mib(clean),
        mib(sum),
        mib(total_decoded),
    );
    eprintln!(
        "[rss_split] peak_resident_u16_markers≈2x_narrowed={:.2}MiB \
         freeable_upper_half≈narrowed={:.2}MiB | GATE0 non_inert={non_inert} \
         conservation(sum==total)={conserves}",
        mib(narrowed * 2),
        mib(narrowed),
    );
    if !non_inert {
        eprintln!("[rss_split] WARNING: instrument INERT (no chunks accounted)");
    }
    if !conserves {
        eprintln!(
            "[rss_split] WARNING: conservation FAILED (sum {} != total_decoded {})",
            sum, total_decoded
        );
    }
    non_inert && conserves
}

/// LIFECYCLE-SPLIT ISOLATION INSTRUMENT (env GZIPPY_LIFECYCLE_SPLIT=1;
/// byte-transparent, OFF == identity). Decomposes the simultaneously-live
/// ChunkData set (MAX_LIVE_CHUNKS) into lifecycle states, to answer WHY all
/// chunks of a small file are resident at peak RSS:
///   AHEAD     — decoded, sitting in BlockFetcher prefetch_cache / main cache
///               awaiting the in-order consumer (= prefetch depth; decoded
///               ahead for overlap).
///   PENDING   — consumer holds it (post-process in flight / awaiting write).
///   RETAINED  — written to the output sink, still held in `recycle_deferral`
///               before its buffers are recycled/freed (FREEABLE after write).
///   DECODING  — derived = peak_live − (AHEAD+PENDING+RETAINED): workers
///               actively building it (genuinely in-flight, needed).
/// Per-holder current-size GAUGES are kept live; snapshotted at the construct
/// that sets a new MAX_LIVE_CHUNKS (peak live is ALWAYS at a construct, since
/// LIVE_CHUNKS only rises there). Gate-0: non-inert (snapshot fired) +
/// conservation (AHEAD+PENDING+RETAINED ≤ peak_live; DECODING ≥ 1 — the
/// just-constructed chunk).
pub static LC_G_PREFETCH: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_G_MAIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_G_PREFETCHING: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_G_PENDING: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_G_RECYCLE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

pub static LC_S_LIVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LC_S_PREFETCH: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_S_MAIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_S_PREFETCHING: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_S_PENDING: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_S_RECYCLE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static LC_S_FIRED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// True iff the lifecycle-split isolation instrument is armed (cached).
#[inline]
pub fn lifecycle_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| std::env::var_os("GZIPPY_LIFECYCLE_SPLIT").is_some())
}

/// Set a holder gauge to its current size (called at the mutation site while the
/// holder's lock is held). No-op unless armed.
#[inline]
pub fn lc_set(gauge: &std::sync::atomic::AtomicI64, n: usize) {
    if lifecycle_enabled() {
        gauge.store(n as i64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Snapshot all holder gauges. Called from `new_with_buffers` when LIVE hits a
/// new peak. No-op unless armed.
#[inline]
pub fn lc_record_peak(live: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if !lifecycle_enabled() {
        return;
    }
    LC_S_LIVE.store(live, Relaxed);
    LC_S_PREFETCH.store(LC_G_PREFETCH.load(Relaxed), Relaxed);
    LC_S_MAIN.store(LC_G_MAIN.load(Relaxed), Relaxed);
    LC_S_PREFETCHING.store(LC_G_PREFETCHING.load(Relaxed), Relaxed);
    LC_S_PENDING.store(LC_G_PENDING.load(Relaxed), Relaxed);
    LC_S_RECYCLE.store(LC_G_RECYCLE.load(Relaxed), Relaxed);
    LC_S_FIRED.fetch_add(1, Relaxed);
}

/// Emit the lifecycle-split report. Returns false if inert. `per_chunk_mib` is
/// the average resident MiB per live chunk (peak_rss_mib / max_live) supplied by
/// the caller so the state COUNTS can be priced into bytes vs the rg gap.
pub fn lifecycle_report() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if !lifecycle_enabled() {
        return true;
    }
    let fired = LC_S_FIRED.load(Relaxed);
    let live = LC_S_LIVE.load(Relaxed) as i64;
    let prefetch = LC_S_PREFETCH.load(Relaxed);
    let main = LC_S_MAIN.load(Relaxed);
    let prefetching = LC_S_PREFETCHING.load(Relaxed);
    let pending = LC_S_PENDING.load(Relaxed);
    let recycle = LC_S_RECYCLE.load(Relaxed);
    let ahead = prefetch + main;
    let accounted = ahead + pending + recycle;
    // DECODING = live − accounted (includes the just-constructed chunk, ≥1).
    let decoding = live - accounted;
    let non_inert = fired > 0;
    let max_live = MAX_LIVE_CHUNKS.load(Relaxed) as i64;
    eprintln!(
        "[lifecycle] peak_live={live} max_live={max_live} fired={fired} | \
         AHEAD(prefetch={prefetch}+main={main})={ahead} PENDING={pending} \
         RETAINED(recycle)={recycle} DECODING(derived)={decoding} \
         in_flight_decode_jobs={prefetching}"
    );
    let pct = |n: i64| {
        if live > 0 {
            100.0 * n as f64 / live as f64
        } else {
            0.0
        }
    };
    eprintln!(
        "[lifecycle] state share of peak_live: AHEAD={:.0}% PENDING={:.0}% \
         RETAINED={:.0}% DECODING={:.0}% | GATE0 non_inert={non_inert} \
         conservation(accounted<=live & decoding>=1)={}",
        pct(ahead),
        pct(pending),
        pct(recycle),
        pct(decoding),
        accounted <= live && decoding >= 1,
    );
    if !non_inert {
        eprintln!("[lifecycle] WARNING: instrument INERT (snapshot never fired)");
    }
    non_inert
}

/// Effect counter: number of times `determine_used_window_symbols_for_last_subchunk`
/// actually ran the 32 KiB marker-engine scan (window_sparsity=true path, not
/// early-returned). With window_sparsity=false (default, keepIndex=false faithful port)
/// this stays 0. With GZIPPY_WINDOW_SPARSITY=1 kill-switch it tracks sparsity work done.
/// Read via GZIPPY_VERBOSE.
pub static SPARSITY_DECODE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl ChunkData {
    /// Return the (now fully-resolved) `data_with_markers` buffer to the owner
    /// worker's u16 pool EAGERLY — right after the post-process worker narrows
    /// it into `narrowed`, instead of waiting for `Drop` (which under
    /// pipelining returns it only after the consumer has already drained the
    /// chunk, by which point the next chunk's bootstrap has taken a cold
    /// buffer). This is what makes the zero-copy bootstrap merge (decode
    /// straight into `data_with_markers`, no `append_markered` copy) a clean
    /// no-copy path rather than a ~3% regression: the warm buffer cycles
    /// take → decode → resolve → return in time for the next take. Idempotent:
    /// leaves an empty buffer, so the `Drop` return becomes a no-op.
    ///
    /// NOTE: no longer called on the production post-process path. Under the
    /// view-based applyWindow model (vendor swap+views, DecodedData.hpp:365-388)
    /// the narrowed marker bytes stay in `data_with_markers` and ARE the output
    /// views, so they must outlive the consumer's writev; recycling is deferred
    /// behind the write via `defer_chunk_recycle` → `recycle_decoded_buffers`.
    /// Kept for the legacy merge-then-recycle path + as vendor-parity surface.
    #[allow(dead_code)]
    pub(crate) fn recycle_markers_after_resolution(&mut self) {
        // After `merge_resolved_markers_into_data`, marker segments are empty
        // and payload bytes live in `data` — safe to return u16 buffers to the
        // worker pool immediately (vendor clears `reusedDataBuffers` post-swap).
        use crate::decompress::parallel::chunk_buffer_pool;
        let dwm = self.data_with_markers.take_segments();
        if !dwm.is_empty() {
            chunk_buffer_pool::return_marker_segments_to_worker(self.pool_worker_index, dwm);
        }
    }

    /// FOOTPRINT-ALIGN: narrow the marker-resolved `data_with_markers` (u16,
    /// every element already < 256 after `apply_window`) into u8 IN PLACE,
    /// writing byte `i` = `data_with_markers[i] as u8` into the buffer's own
    /// backing memory reinterpreted as `*mut u8`. Records `narrowed_len`.
    /// Faithful port of rapidgzip's `applyWindow` tail (DecodedData.hpp:325-388)
    /// which writes `target[i] = …` via `reinterpret_cast<uint8_t*>(chunk.data())`
    /// and then exposes the buffer as u8 — NO separate buffer, NO copy.
    ///
    /// SAFETY/CORRECTNESS: the u8 write at byte offset `i` never clobbers a
    /// u16 element at index `j >= i` before it is read, because that element
    /// lives at byte offset `2*j >= 2*i > i`. The narrowed bytes occupy the
    /// first `len` bytes of the same allocation; the upper bytes (stale u16
    /// high bytes) are simply ignored. Returns the byte count.
    pub fn narrow_markers_in_place(&mut self) -> usize {
        let len = self.data_with_markers.len();
        debug_assert!(
            self.data_with_markers.all_resolved(),
            "narrow_markers_in_place before markers resolved"
        );
        if len > 0 {
            self.data_with_markers.narrow_markers_to_u8_in_place();
        }
        self.narrowed_len = len;
        len
    }

    /// FUSED resolve + narrow: replaces the two-pass
    /// `apply_window` (resolve u16→u16) + `narrow_markers_in_place` (u16→u8)
    /// with a single pass over `data_with_markers` (64 KiB u8 LUT). After this
    /// the low `narrowed_len` bytes of `data_with_markers` hold the resolved
    /// u8 output, exactly as the two-pass path produced.
    pub fn resolve_and_narrow_markers_in_place(&mut self, window: &[u8]) -> usize {
        debug_assert_eq!(
            self.data_prefix_len, 0,
            "trim_window_prefix before resolve_and_narrow_markers_in_place"
        );
        // Double-resolution tripwire. Under the view-based applyWindow model the
        // narrowed bytes stay in `data_with_markers` (NOT cleared by a merge), so
        // `all_resolved()` can no longer catch a second resolve (it would read the
        // stale u16 high bytes left by the first narrow and feed them through the
        // LUT → silent corruption). The merge used to empty the buffer and act as
        // that guard; now correctness rests on `markers_resolved` / `narrowed_len`,
        // so assert them here. (Advisor: merge-removal-advisor-verdict.md.)
        debug_assert!(
            self.narrowed_len == 0 && !self.markers_resolved,
            "resolve_and_narrow_markers_in_place: already resolved (double-resolve)"
        );
        let len = self.data_with_markers.len();
        if len > 0 {
            self.data_with_markers.resolve_and_narrow_in_place(window);
        }
        self.narrowed_len = len;
        len
    }

    /// CRC the in-place-narrowed marker bytes (may span multiple segments).
    pub fn update_narrowed_crc(&mut self) {
        if self.narrowed_len == 0 {
            return;
        }
        let mut left = self.narrowed_len;
        for seg in self.data_with_markers.segments() {
            if left == 0 {
                break;
            }
            let n = left.min(seg.len());
            let sl = unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, n) };
            self.narrowed_crc.update(sl);
            left -= n;
        }
    }

    /// Vendor `DecodedData::applyWindow` buffer swap (DecodedData.hpp:365-388):
    /// prepend the in-place-narrowed marker bytes into `data`, then drop
    /// `data_with_markers` so later `getWindowAt` / Iterator walks unified
    /// `data` only.
    pub fn merge_resolved_markers_into_data(&mut self) {
        let n = self.narrowed_len;
        if n == 0 {
            return;
        }
        if self.data_prefix_len == 0 {
            self.data
                .prepend_narrowed_from_markers(&self.data_with_markers, n);
        } else {
            let mut merged = vec![0u8; n];
            self.data_with_markers
                .copy_narrowed_u8_range_into(0, n, &mut merged);
            self.data.insert_logical_at(self.data_prefix_len, &merged);
        }
        self.narrowed_len = 0;
        self.data_with_markers.clear();
    }

    /// Collect output iovecs for the consumer write path. After
    /// `merge_resolved_markers_into_data`, payload lives only in `data`.
    pub fn append_output_iovecs<'a>(&'a self, out: &mut Vec<&'a [u8]>) {
        if self.narrowed_len > 0 {
            self.data_with_markers
                .append_narrowed_iovecs(self.narrowed_len, out);
        }
        self.data.append_payload_iovecs(self.data_prefix_len, out);
    }

    /// Return decoded payload buffers to the per-worker pool immediately
    /// after the consumer has finished writing them (vendor FasterVector
    /// auto-recycle on `writeAll` completion). Idempotent — safe to call
    /// again from `Drop` on an already-empty shell.
    pub(crate) fn recycle_decoded_buffers(&mut self) {
        use crate::decompress::parallel::chunk_buffer_pool;
        let segments = self.data.take_segments();
        for seg in segments {
            chunk_buffer_pool::return_u8_to_worker(self.pool_worker_index, seg);
        }
        let dwm = self.data_with_markers.take_segments();
        chunk_buffer_pool::return_marker_segments_to_worker(self.pool_worker_index, dwm);
        self.narrowed_len = 0;
    }
}

impl Drop for ChunkData {
    fn drop(&mut self) {
        LIVE_CHUNKS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        self.recycle_decoded_buffers();
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::parallel::replace_markers::MARKER_BASE;

    fn small_config() -> ChunkConfiguration {
        ChunkConfiguration {
            split_chunk_size: 100,
            max_decoded_chunk_size: 10_000,
            crc32_enabled: true,
            ..Default::default()
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
        // Need enough decoded bytes in the first subchunk (> split_chunk_size = 100)
        // for append_block_boundary_at to materialize a NEW subchunk per the
        // vendor-parity gate at append_block_boundary_at:541. Without enough
        // bytes the gate returns `true` for dedup but does not push a Subchunk.
        chunk.append_clean(&[0u8; 150]);
        chunk.append_block_boundary(200);
        // Enough bytes in the trailing subchunk to avoid vendor
        // `finalizeChunk` small-subchunk merge (minimumSplitChunkSize = 25).
        chunk.append_clean(&[0u8; 50]);
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
        // First subchunk needs > split_chunk_size = 100 decoded bytes for the
        // gate at append_block_boundary_at:541 to materialize a new Subchunk.
        chunk.append_clean(&[0u8; 150]);
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
        assert_eq!(chunk.data.to_contiguous(), b"hello world");
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
                *slot = chunk_b.data_with_markers.get(start + i).unwrap_or(0) as u8;
            }
            written += take;
            let n = W - written;
            chunk_b
                .data
                .copy_range_into(0, &mut reference[written..written + n]);
        } else {
            let off = start - dwm_b_len;
            chunk_b.data.copy_range_into(off, &mut reference[..]);
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
    fn get_last_window_vec_matches_array_path() {
        const W: usize = 32768;
        let mut prev = [0u8; W];
        for (i, b) in prev.iter_mut().enumerate() {
            *b = ((i * 13) & 0xff) as u8;
        }

        let mut chunk = ChunkData::new(0, small_config());
        chunk.append_markered(&[1, 2, 3, MARKER_BASE + 7, MARKER_BASE + 9]);
        let clean = vec![0x5Au8; W + 257];
        chunk.append_clean(&clean);

        assert_eq!(
            chunk.get_last_window_vec(&prev),
            chunk.get_last_window(&prev).to_vec()
        );
    }

    #[test]
    fn last_32kib_window_vec_matches_array_fast_path() {
        const W: usize = 32768;
        let mut chunk = ChunkData::new(0, small_config());
        let clean: Vec<u8> = (0..(W + 1234)).map(|i| (i & 0xff) as u8).collect();
        chunk.append_clean(&clean);

        assert_eq!(
            chunk.last_32kib_window_vec().unwrap(),
            chunk.last_32kib_window().unwrap().to_vec()
        );
    }

    #[test]
    fn populate_subchunk_windows_after_merge_sets_post_processed_state() {
        const W: usize = 32768;
        let mut prev = [0u8; W];
        for (i, b) in prev.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }

        let mut chunk = ChunkData::new(0, small_config());
        chunk.append_markered(&[0x11, MARKER_BASE + 5, 0x22]);
        chunk.append_clean(&[0xAAu8; W + 512]);
        chunk.finalize(10_000);

        crate::decompress::parallel::apply_window::apply_window(&mut chunk, &prev);
        chunk.narrow_markers_in_place();
        chunk.merge_resolved_markers_into_data();
        chunk.populate_subchunk_windows(&prev);

        assert!(chunk
            .subchunks
            .iter()
            .all(|sc| sc.window.is_some() && sc.used_window_symbols.is_empty()));
        assert!(chunk.has_been_post_processed(false));
    }

    #[test]
    fn merge_resolved_markers_into_data_unifies_output_buffer() {
        const W: usize = 32768;
        let mut prev = [0u8; W];
        for (i, b) in prev.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }

        let mut chunk = ChunkData::new(0, small_config());
        chunk.append_markered(&[0x11, MARKER_BASE + 5, 0x22]);
        chunk.append_clean(&[0xAA, 0xBB, 0xCC]);

        crate::decompress::parallel::apply_window::apply_window(&mut chunk, &prev);
        chunk.narrow_markers_in_place();
        chunk.update_narrowed_crc();
        let n = chunk.narrowed_len;
        let mut expected_marker_bytes = vec![0u8; n];
        chunk
            .data_with_markers
            .copy_narrowed_u8_range_into(0, n, &mut expected_marker_bytes);
        chunk.merge_resolved_markers_into_data();

        assert_eq!(chunk.narrowed_len, 0);
        assert!(chunk.data_with_markers.is_empty());
        assert_eq!(chunk.decoded_size(), expected_marker_bytes.len() + 3);
        let payload_len = chunk.data.len();
        let mut out = vec![0u8; payload_len];
        chunk.data.copy_range_into(0, &mut out);
        assert_eq!(
            &out[..expected_marker_bytes.len()],
            &expected_marker_bytes[..]
        );
        assert_eq!(&out[expected_marker_bytes.len()..], &[0xAA, 0xBB, 0xCC]);

        let mut iovecs = Vec::new();
        chunk.append_output_iovecs(&mut iovecs);
        assert_eq!(iovecs.iter().map(|s| s.len()).sum::<usize>(), payload_len);
    }

    /// View-based applyWindow (vendor swap+views, DecodedData.hpp:365-388):
    /// the production post-process now SKIPS `merge_resolved_markers_into_data`
    /// and leaves the narrowed marker bytes in `data_with_markers` with
    /// `narrowed_len > 0`. Lock that un-merged state: subchunk windows populate,
    /// `has_been_post_processed` is true, and the iovec emit yields the SAME
    /// bytes (markers ‖ data) as the merged path.
    #[test]
    fn populate_subchunk_windows_unmerged_view_based_apply_window() {
        const W: usize = 32768;
        let mut prev = [0u8; W];
        for (i, b) in prev.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }

        let mut chunk = ChunkData::new(0, small_config());
        chunk.append_markered(&[0x11, MARKER_BASE + 5, 0x22]);
        chunk.append_clean(&[0xAAu8; W + 512]);
        chunk.finalize(10_000);

        // Production order: fused resolve+narrow → CRC → subchunk windows.
        // NO merge, NO recycle (the swap+views model).
        chunk.resolve_and_narrow_markers_in_place(&prev);
        chunk.update_narrowed_crc();
        assert!(
            chunk.narrowed_len > 0,
            "narrowed bytes must remain in markers"
        );
        chunk.populate_subchunk_windows(&prev);

        // Resolved marker chunk reports done even un-merged.
        assert!(!chunk.contains_markers());
        assert!(chunk.has_been_post_processed(false));
        assert!(chunk
            .subchunks
            .iter()
            .all(|sc| sc.window.is_some() && sc.used_window_symbols.is_empty()));

        // The iovec emit reads markers ‖ data directly (no unified `data`).
        let mut iovecs = Vec::new();
        chunk.append_output_iovecs(&mut iovecs);
        let emitted: usize = iovecs.iter().map(|s| s.len()).sum();
        assert_eq!(emitted, chunk.decoded_size());
        // First narrowed byte is the resolved 0x11 literal; the marker
        // MARKER_BASE+5 resolves to prev[5].
        let flat: Vec<u8> = iovecs.iter().flat_map(|s| s.iter().copied()).collect();
        assert_eq!(flat[0], 0x11);
        assert_eq!(flat[1], prev[5]);
        assert_eq!(flat[2], 0x22);
        assert_eq!(flat[3], 0xAA);
    }

    #[test]
    fn append_clean_skips_crc_when_disabled() {
        let cfg = ChunkConfiguration {
            split_chunk_size: 100,
            max_decoded_chunk_size: 10_000,
            crc32_enabled: false,
            ..Default::default()
        };
        let mut chunk = ChunkData::new(0, cfg);
        chunk.append_clean(b"hello world");
        // CRC unchanged from identity.
        assert_eq!(chunk.crc32s[0].crc32(), 0);
    }
}
