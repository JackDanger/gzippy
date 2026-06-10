#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

//! Literal port of `rapidgzip::deflate::Block`
//! (vendor/.../gzip/deflate.hpp:513-1156): the deflate Block state
//! machine, header parser, and decode inner loops.

use std::io;

use crate::decompress::inflate::consume_first_decode::Bits;
// Vendor primitive: `nLowestBitsSet<T>(uint8_t)`
// (vendor/.../core/BitManipulation.hpp:60-73). Used here to construct
// extra-bits masks for length/distance/uncompressed-padding reads, so
// the production deflate header parser drives the ported primitive
// rather than an inline `(1 << n) - 1`.
use crate::decompress::parallel::bit_manipulation::n_lowest_bits_set;

/// Sink for decoded u16 marker/literal output. Lets the bootstrap decode
/// DIRECTLY into the chunk's pooled `U16` (rpmalloc-backed) buffer instead of
/// a separate `std::Vec<u16>` that is then copied in (`append_markered`). Both
/// the std `Vec<u16>` (tests / non-arena builds) and the arena-backed `U16`
/// implement it, so the same decode path serves both with zero copy.
pub trait MarkerSink {
    fn push_slice(&mut self, values: &[u16]);
    fn sink_len(&self) -> usize;
    fn as_slice(&self) -> &[u16];

    /// Record a deflate block boundary at `encoded_offset_bits` with the
    /// sink-relative `decoded_offset` (clean-tail bytes emitted so far).
    /// Default no-op: the marker-phase sinks (`Vec<u16>`/arena `U16`) do not
    /// split subchunks — only the merged clean-tail sink overrides this to
    /// drive `ChunkData::append_block_boundary_at` (the vendor on-the-fly
    /// `appendDeflateBlockBoundary` subchunk split, GzipChunk.hpp:177-182).
    fn note_block_boundary(&mut self, _encoded_offset_bits: usize, _decoded_offset: usize) {}

    /// Push already-narrowed clean u8 bytes from the post-flip u8-direct decode
    /// (vendor's `getWindow()` path). The clean-tail sink overrides this to write
    /// straight into `chunk.data` (no narrow pass). Default widens to u16 for the
    /// marker sinks — never hit on the production post-flip path (only the
    /// clean-tail sink is active there), but keeps the trait total.
    fn push_clean_u8(&mut self, bytes: &[u8]) {
        let mut buf = [0u16; 256];
        for chunk in bytes.chunks(256) {
            for (i, &b) in chunk.iter().enumerate() {
                buf[i] = b as u16;
            }
            self.push_slice(&buf[..chunk.len()]);
        }
    }

    /// Clean u8 bytes routed to `chunk.data` during the marker loop
    /// (vendor `cleanDataCount`). Default 0 for marker-only sinks.
    fn clean_appended_len(&self) -> usize {
        0
    }

    /// Trailing run of clean (`< MARKER_BASE`) values starting at logical
    /// index `from` (inclusive). Used by the bootstrap block loop.
    fn trailing_clean_since(&self, from: usize) -> usize {
        use crate::decompress::parallel::replace_markers::MARKER_BASE;
        self.as_slice()[from..]
            .iter()
            .rev()
            .take_while(|&&v| v < MARKER_BASE)
            .count()
    }

    /// Copy the last `n` sink elements as u8 into `out` if they are all clean.
    fn copy_last_n_clean_u8(&self, n: usize, out: &mut Vec<u8>) -> bool {
        use crate::decompress::parallel::replace_markers::MARKER_BASE;
        let len = self.sink_len();
        if n == 0 || n > len {
            return false;
        }
        let start = len - n;
        let slice = &self.as_slice()[start..];
        if slice.iter().any(|&v| v >= MARKER_BASE) {
            return false;
        }
        out.clear();
        out.extend(slice.iter().map(|&v| v as u8));
        true
    }

    /// True iff the last `n` sink elements are all clean (`< MARKER_BASE`).
    /// Cheap predicate (no copy) so the marker→clean handoff can decide
    /// Handoff-vs-Finished WITHOUT materializing a window buffer — the
    /// caller narrows the clean tail into a reused thread-local buffer
    /// only once the handoff is confirmed (kills the per-chunk 32 KiB
    /// `Vec<u8>` clean-window allocation).
    ///
    /// Routed through `trailing_clean_since` (NOT `as_slice`) so it is
    /// correct for SEGMENTED sinks: `SegmentedU16::as_slice()` returns `&[]`
    /// (segmented storage is not one contiguous slice) and overrides
    /// `trailing_clean_since` with a segment-aware scan. The `Vec<u16>`
    /// default `trailing_clean_since` uses `as_slice`, which is contiguous.
    fn is_last_n_clean(&self, n: usize) -> bool {
        let len = self.sink_len();
        if n == 0 || n > len {
            return false;
        }
        self.trailing_clean_since(len - n) >= n
    }
}

impl MarkerSink for Vec<u16> {
    #[inline]
    fn push_slice(&mut self, values: &[u16]) {
        self.extend_from_slice(values);
    }
    #[inline]
    fn sink_len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn as_slice(&self) -> &[u16] {
        self
    }
}

// When `arena-allocator` is on, `U16` is a DISTINCT type from `Vec<u16>`
// (allocator_api2 Vec), so it needs its own impl. Without the feature `U16 ==
// Vec<u16>` and the impl above already covers it (cfg avoids a duplicate).
#[cfg(feature = "arena-allocator")]
impl MarkerSink for crate::decompress::parallel::rpmalloc_alloc::types::U16 {
    #[inline]
    fn push_slice(&mut self, values: &[u16]) {
        self.extend_from_slice(values);
    }
    #[inline]
    fn sink_len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn as_slice(&self) -> &[u16] {
        self
    }
}

// ── Constants (from rapidgzip definitions.hpp) ──────────────────────────────

pub const MAX_LITERAL_OR_LENGTH_SYMBOLS: usize = 286;
pub const MAX_DISTANCE_SYMBOL_COUNT: usize = 30;
pub const MAX_PRECODE_LENGTH: u8 = 7;
pub const PRECODE_BITS: u8 = 3;
pub const MAX_PRECODE_COUNT: usize = 19;
pub const END_OF_BLOCK_SYMBOL: u16 = 256;
pub const MAX_WINDOW_SIZE: usize = 32768;

/// RFC 1951 precode alphabet order (matches deflate.hpp's PRECODE_ALPHABET).
pub const PRECODE_ALPHABET: [usize; MAX_PRECODE_COUNT] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

// ── Types ───────────────────────────────────────────────────────────────────

/// Compression type from the 3-bit deflate block header. Mirror of
/// `rapidgzip::deflate::CompressionType` (definitions.hpp).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    Uncompressed = 0,
    FixedHuffman = 1,
    DynamicHuffman = 2,
    Reserved = 3,
}

impl CompressionType {
    pub fn from_btype(btype: u8) -> Self {
        match btype {
            0 => CompressionType::Uncompressed,
            1 => CompressionType::FixedHuffman,
            2 => CompressionType::DynamicHuffman,
            _ => CompressionType::Reserved,
        }
    }
}

/// A back-reference discovered during decoding. Mirror of
/// `Block::Backreference` (deflate.hpp:520-523).
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)] // populated when track_backreferences is enabled
pub struct Backreference {
    pub distance: u16,
    pub length: u16,
}

/// Errors returned by `Block`'s methods. Subset of rapidgzip's
/// `Error` enum covering only the deflate-Block error paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    EndOfFile,
    UnexpectedLastBlock,
    NonZeroPadding,
    LengthChecksumMismatch,
    ExceededLiteralRange,
    ExceededDistanceRange,
    InvalidCompression,
    InvalidCodeLengths,
    InvalidHuffmanCode,
    ExceededWindowRange,
}

impl From<BlockError> for io::Error {
    fn from(e: BlockError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}"))
    }
}

// ── Block ───────────────────────────────────────────────────────────────────

/// Literal port of `rapidgzip::deflate::Block<ENABLE_STATISTICS>`
/// (deflate.hpp:513-961) — the deflate block state machine.
///
/// Holds the block state + header-derived metadata, with the decode
/// inner loops as methods.
///
/// **Huffman tables are held as fields and REUSED across blocks** —
/// rebuilt in-place by `read_internal_compressed` via
/// `IsalLitLenCode::rebuild_from` / `IsalDistCode::rebuild_from`. Mirror
/// of rapidgzip's `Block<>::m_literalHC` + `m_distanceHC` members
/// (deflate.hpp:920-925) which are persistent for the same reason: a
/// per-block 19 KiB ISA-L table allocation dominates the critical path
/// on small dynamic blocks. See `HuffmanCodingISAL.hpp:21-26` for the
/// reference table layout.
/// Ring buffer size in u16 elements. Mirror of vendor's
/// `PreDecodedBuffer = std::array<uint16_t, 2 * MAX_WINDOW_SIZE>`
/// (vendor/.../gzip/deflate.hpp:805). 2× the window so back-refs
/// can always reach `MAX_WINDOW_SIZE` bytes back without wraparound
/// concerns within a single read() call.
pub const RING_SIZE: usize = 2 * MAX_WINDOW_SIZE;

/// u8 view of the SAME `output_ring` backing store, post-flip. Vendor's
/// `getWindow()` (deflate.hpp:890-894) `reinterpret_cast`s the 65536-element
/// `m_window16` (128 KB) to a `uint8_t*` of `2 * MAX_WINDOW_SIZE * sizeof(u16)`
/// = 131072 u8 slots over the identical bytes (DecodedBuffer, deflate.hpp:806).
/// Post-flip (`m_containsMarkerBytes == false`) the clean tail decodes
/// u8-DIRECT into this view, indexed `% U8_RING_SIZE` — half the memory
/// traffic of the u16 ring, and the drain is a plain copy (no u16->u8 narrow).
/// The flip's `setInitialWindow` repack (deflate.hpp:1772-1782) value-downcasts
/// the surviving 32 KiB window into the upper half of this view.
pub const U8_RING_SIZE: usize = 2 * RING_SIZE;

/// Initialize the ring's marker zone — the upper half (slots
/// `MAX_WINDOW_SIZE..RING_SIZE`) holds pre-computed marker values
/// `MAX_WINDOW_SIZE..RING_SIZE`. Mirror of vendor's
/// `initializeMarkedWindowBuffer` at deflate.hpp:875-888.
///
/// When a cross-chunk back-ref at chunk position `p` with
/// distance `d > p+i` fires, it reads from
/// `ring[(p + i - d + RING_SIZE) % RING_SIZE]` which lands in the
/// marker zone (slot `RING_SIZE - (d - p - i)`). The pre-initialized
/// value at that slot equals the slot index, which by construction
/// IS the correct marker value:
///   marker_value = MARKER_BASE + (MAX_WINDOW_SIZE + p + i - d)
///                = 32768 + (32768 + p + i - d)
///                = 65536 + p + i - d
///                = ring slot index
/// So `emit_backref_ring`'s single `memcpy` produces correct
/// markers for the cross-chunk portion AND correct decoded bytes
/// for the in-chunk portion of a single back-ref — no explicit
/// `marker_count` loop required.
#[inline]
pub(crate) fn init_marker_zone(ring: &mut [u16; RING_SIZE]) {
    for i in 0..MAX_WINDOW_SIZE {
        ring[MAX_WINDOW_SIZE + i] = (MAX_WINDOW_SIZE + i) as u16;
    }
}

// ── M2b (DIV-5): vendor stored-block special cases — counters + kill-switch ──
//
// Proof-of-path counters for the three vendor stored-block special cases
// (deflate.hpp:1212-1256) plus the contig sibling of the clean bulk read.
// Relaxed; test + GZIPPY_DEBUG observability only — never decode logic.

/// Case 1: `uncompressedSize >= MAX_WINDOW_SIZE` ⇒ straight u8 read + flip
/// (deflate.hpp:1214-1219).
pub static STORED_FLIP_GE_WINDOW: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Case 2: markers + `distanceToLastMarkerByte + uncompressedSize >=
/// MAX_WINDOW_SIZE` ⇒ downcast surviving prefix + read + flip
/// (deflate.hpp:1220-1242).
pub static STORED_FLIP_CROSSING: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Case 3: already clean ⇒ bulk byte read into the u8 ring
/// (deflate.hpp:1243-1255).
pub static STORED_CLEAN_BULK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Contig sibling of case 3: bulk byte read straight into `chunk.data`
/// (`decode_clean_stored_into_contig`).
pub static STORED_CONTIG_BULK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Test-only override for the kill-switch: -1 = follow the env var,
/// 0 = force-enabled, 1 = force-disabled. Lets one test process exercise
/// BOTH arms (the env-var read is `OnceLock`-cached). One relaxed load per
/// stored BLOCK (not per byte) — negligible.
pub(crate) static STORED_FLIP_OVERRIDE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

/// M2b kill-switch: `GZIPPY_NO_STORED_FLIP=1` restores the exact pre-M2b
/// stored-block behavior (per-byte decode, generic arming only) in both the
/// ring path (`try_read_stored_special`) and the contig path
/// (`decode_clean_stored_into_contig`'s bulk read).
fn stored_flip_disabled() -> bool {
    let ov = STORED_FLIP_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if ov >= 0 {
        return ov == 1;
    }
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GZIPPY_NO_STORED_FLIP").is_ok_and(|v| v == "1"))
}

/// `GZIPPY_DEBUG=1`-gated one-line trace for the two flip cases (proves the
/// new path active/inactive on a shipped binary without counters plumbing).
fn stored_flip_debug_log(case: &str, n: usize) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok_and(|v| v == "1")) {
        eprintln!("gzippy: stored-flip {case} uncompressed_size={n}");
    }
}

/// Bulk byte read for stored-block payloads — the analog of vendor
/// `BitReader::read(char*, size)` used by all three special cases
/// (deflate.hpp:1219, :1241, :1253) and the same pattern as the existing
/// stored fast path in `consume_first_decode::decode_stored`: drain whole
/// already-buffered bytes from the bit buffer, then memcpy the remainder
/// straight from the input slice. Requires byte alignment (guaranteed after
/// `read_uncompressed_header`'s padding drain + 32-bit LEN/NLEN read).
/// Returns the number of bytes written (== `dst.len()` when the caller's
/// availability gate held).
fn read_stored_bytes_aligned(bits: &mut Bits, dst: &mut [u8]) -> usize {
    debug_assert_eq!(
        bits.available() % 8,
        0,
        "stored payload must be byte-aligned"
    );
    let mut i = 0usize;
    while i < dst.len() && bits.available() >= 8 {
        dst[i] = (bits.peek() & 0xFF) as u8;
        bits.consume(8);
        i += 1;
    }
    let rest = (dst.len() - i).min(bits.data.len() - bits.pos);
    if rest > 0 {
        dst[i..i + rest].copy_from_slice(&bits.data[bits.pos..bits.pos + rest]);
        bits.pos += rest;
        // Moving `pos` past bytes that never went through the bit buffer
        // invalidates the refill invariant (the fast refill keeps the next
        // byte's bits ABOVE `bitsleft` in `bitbuf` and relies on re-OR-ing
        // the SAME byte at `pos`); reset like `decode_stored` does
        // (consume_first_decode.rs:1462-1464). The drain loop above already
        // emptied every credited bit (aligned ⇒ available() hit 0), so
        // nothing valid is discarded.
        bits.bitbuf = 0;
        bits.bitsleft = 0;
    }
    i + rest
}

/// Literal port of rapidgzip `deflate::Block` — production bootstrap session
/// (vendor `decodeChunkWithRapidgzip`, GzipChunk.hpp:468-654). `MarkerRing`
/// remains available via `GZIPPY_MARKER_RING=1` for A/B only.
#[cfg(parallel_sm)]
pub struct Block {
    at_end_of_block: bool,
    at_end_of_file: bool,
    is_last_block: bool,
    compression_type: CompressionType,
    /// Stored-block padding bits (0..7) — zero in well-formed input.
    padding: u8,
    /// Stored-block declared payload length (only set when
    /// `compression_type == Uncompressed`).
    uncompressed_size: usize,
    /// Total decoded bytes across all blocks decoded by this Block
    /// instance (rapidgzip's `m_decodedBytes`, deflate.hpp:921).
    decoded_bytes: usize,
    /// `m_decodedBytes` snapshot at the start of the current block.
    decoded_bytes_at_block_start: usize,
    /// The ONE dual-width decode window (M2, plans/engine-u8-design.md §2/§5):
    /// vendor's `m_window16` + `getWindow()` u8 view + `m_windowPosition` +
    /// `m_distanceToLastMarkerByte` + `m_containsMarkerBytes` as a single
    /// `WidthRing` (see `width_ring.rs` for the per-field vendor citations).
    ///
    /// The optimized fast loops below pull `ring.pos` /
    /// `ring.distance_to_last_marker` into locals and write through raw
    /// pointers derived from `ring.window16` — identical codegen to the
    /// pre-M2 inline fields. Width dispatch (`ring.is_marker()`) replaces the
    /// old `contains_marker_bytes` bool (a 2-variant enum byte compare).
    ///
    /// `decoded_bytes` (vendor `m_decodedBytes`) stays a `Block` field —
    /// vendor keeps it on `Block`, not the window — and is passed to
    /// `ring.should_flip(..)` / `ring.flip_in_place(..)` explicitly.
    ring: super::width_ring::WidthRing,
    /// Code lengths for the precode alphabet (P), populated by
    /// `read_dynamic_huffman_coding`.
    pub precode_cl: [u8; MAX_PRECODE_COUNT],
    /// Code lengths for the combined literal/length + distance
    /// alphabets, populated by `read_dynamic_huffman_coding`. First
    /// `literal_code_count` entries are literal/length lengths;
    /// remainder are distance lengths.
    pub literal_cl: Vec<u8>,
    /// Number of literal/length codes from this block's header.
    pub literal_code_count: usize,
    /// Number of distance codes from this block's header.
    pub distance_code_count: usize,
    /// Tracked back-references (debug instrumentation).
    pub backreferences: Vec<Backreference>,
    track_backreferences: bool,
    /// Pure-Rust ISA-L LUT tables (byte-matched to igzip; same decode loop
    /// as the C `IsalLitLenCode` / `IsalDistCode` fields above).
    #[cfg(pure_inflate_decode)]
    lut_litlen: crate::decompress::parallel::lut_huffman::LutLitLenCode,
    /// Distance Huffman decoder. Vendor rapidgzip explicitly REJECTED ISA-L
    /// for distance and uses `HuffmanCodingReversedBitsCached` (gzip/deflate.hpp:336;
    /// ISA-L distance commented out :338) — with `Symbol = uint8_t`, a 64 KiB
    /// cache (deflate.hpp:668). gzippy uses the vendor's bounded-LUT sibling
    /// `HuffmanCodingShortBitsCached` (`LUT_BITS_COUNT = 12`, `Symbol = u8`):
    /// byte-identical (decode_long fallback for the rare >12-bit distance code)
    /// at an 8 KiB/thread cache instead of 128 KiB. Faithful transliteration of
    /// the distance-decode choice (mirrors the canonical fallback path :1514-1586).
    #[cfg(pure_inflate_decode)]
    dist_hc: crate::decompress::parallel::huffman_short_bits_cached::DistanceShortBitsCached<
        MAX_DISTANCE_SYMBOL_COUNT,
    >,
    /// True after `read_header` built ISA-L LUTs for this block (vendor
    /// `m_literalHC` / `m_distanceHC` at deflate.hpp:1137-1141). Cleared
    /// on each new header so `read()` in the `while !eob` loop reuses them.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    block_huffman_luts_ready: bool,
}

#[cfg(parallel_sm)]
impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Block")
            .field("at_end_of_block", &self.at_end_of_block)
            .field("at_end_of_file", &self.at_end_of_file)
            .field("is_last_block", &self.is_last_block)
            .field("compression_type", &self.compression_type)
            .field("padding", &self.padding)
            .field("uncompressed_size", &self.uncompressed_size)
            .field("decoded_bytes", &self.decoded_bytes)
            .field("literal_code_count", &self.literal_code_count)
            .field("distance_code_count", &self.distance_code_count)
            .finish_non_exhaustive()
    }
}

#[cfg(parallel_sm)]
impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(parallel_sm)]
impl Block {
    pub fn new() -> Self {
        // The ring (heap-boxed, marker zone pre-initialized so cross-chunk
        // back-refs produce correct markers via plain memcpy) is built by
        // `WidthRing::new` — see width_ring.rs for the vendor citations.
        Self {
            at_end_of_block: false,
            at_end_of_file: false,
            is_last_block: false,
            compression_type: CompressionType::DynamicHuffman,
            padding: 0,
            uncompressed_size: 0,
            decoded_bytes: 0,
            decoded_bytes_at_block_start: 0,
            precode_cl: [0u8; MAX_PRECODE_COUNT],
            literal_cl: vec![0u8; MAX_LITERAL_OR_LENGTH_SYMBOLS + MAX_DISTANCE_SYMBOL_COUNT],
            literal_code_count: 0,
            distance_code_count: 0,
            backreferences: Vec::new(),
            track_backreferences: false,
            ring: super::width_ring::WidthRing::new(),
            #[cfg(pure_inflate_decode)]
            lut_litlen: crate::decompress::parallel::lut_huffman::LutLitLenCode::new_empty(),
            #[cfg(pure_inflate_decode)]
            dist_hc:
                crate::decompress::parallel::huffman_short_bits_cached::DistanceShortBitsCached::new(
                ),
            #[cfg(any(
                all(
                    feature = "isal-compression",
                    not(feature = "pure-rust-inflate"),
                    target_arch = "x86_64"
                ),
                pure_inflate_decode
            ))]
            block_huffman_luts_ready: false,
        }
    }

    /// Resident PER-THREAD working-set byte breakdown for the gzippy-native
    /// cache-residency mandate (`plans/gzippy-native-design-mandate.md`). This
    /// is the persistent thread-local engine state (`BOOTSTRAP_BLOCK`,
    /// `gzip_chunk.rs`) — the real native working set after the flip-in-place
    /// fold removed Engine C. Counters only; never mutates decode state.
    ///
    /// Components (native `pure_inflate_decode` build):
    /// - `output_ring`: `Box<[u16; RING_SIZE]>` = 128 KiB.
    /// - `lut_litlen` (`LutLitLenCode`): the per-thread ISA-L-style lit/len LUT
    ///   (`InflateHuffCodeLarge` short+long lookup + code lists), rebuilt
    ///   in-place per block — persistent allocation, not shared.
    /// - `dist_hc` (`HuffmanCodingReversedBitsCached`): the distance
    ///   `code_cache` = `[(u8, u16); 1<<15]` = 128 KiB.
    /// - `literal_cl` / `backreferences` Vecs (heap), `precode_cl` is inline.
    ///
    /// Returns `(total, ring, litlen_lut, dist_cache, misc_vecs)`.
    #[cfg(parallel_sm)]
    pub fn heap_bytes(&self) -> super::super::inflate::mem_stats::BlockHeapBytes {
        use std::mem::size_of;
        let ring = size_of::<[u16; RING_SIZE]>();
        #[cfg(pure_inflate_decode)]
        let litlen_lut = self.lut_litlen.heap_bytes();
        #[cfg(not(pure_inflate_decode))]
        let litlen_lut = 0usize;
        #[cfg(pure_inflate_decode)]
        let dist_cache = self.dist_hc.heap_bytes();
        #[cfg(not(pure_inflate_decode))]
        let dist_cache = 0usize;
        let misc_vecs = self.literal_cl.capacity()
            + self.backreferences.capacity() * size_of::<Backreference>();
        super::super::inflate::mem_stats::BlockHeapBytes {
            total: ring + litlen_lut + dist_cache + misc_vecs,
            ring,
            litlen_lut,
            dist_cache,
            misc_vecs,
        }
    }

    // ── Accessors (deflate.hpp:526-561) ─────────────────────────────────────

    pub fn eob(&self) -> bool {
        self.at_end_of_block
    }
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn eos(&self) -> bool {
        self.at_end_of_block && self.is_last_block
    }
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn eof(&self) -> bool {
        self.at_end_of_file
    }
    pub fn is_last_block(&self) -> bool {
        self.is_last_block
    }
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn compression_type(&self) -> CompressionType {
        self.compression_type
    }
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn padding(&self) -> u8 {
        self.padding
    }
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn uncompressed_size(&self) -> usize {
        if self.compression_type == CompressionType::Uncompressed {
            self.uncompressed_size
        } else {
            0
        }
    }
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn set_track_backreferences(&mut self, enable: bool) {
        self.track_backreferences = enable;
    }
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn track_backreferences(&self) -> bool {
        self.track_backreferences
    }

    pub fn backreferences(&self) -> &[Backreference] {
        &self.backreferences
    }

    /// Vendor `resolveBackreference` tracking (`deflate.hpp:1354-1364`).
    fn record_backreference_for_sparsity(
        &mut self,
        distance: usize,
        length: usize,
        emitted: usize,
    ) {
        if !self.track_backreferences {
            return;
        }
        let decoded_in_block = self
            .decoded_bytes
            .saturating_sub(self.decoded_bytes_at_block_start)
            .saturating_add(emitted);
        if distance > decoded_in_block {
            let stored_dist = (distance - decoded_in_block) as u16;
            let stored_len = (length as u16).min(stored_dist);
            self.backreferences.push(Backreference {
                distance: stored_dist,
                length: stored_len,
            });
        }
    }
    /// True while back-refs may produce markers (pre mid-decode
    /// switch). Flipped to false by `pub fn read` once the chunk has
    /// accumulated 32 KiB of consecutive clean output AND the chunk
    /// hasn't emitted any markers yet, OR the entire ring is clean.
    /// Mirror of vendor's `m_containsMarkerBytes` accessor.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn contains_marker_bytes(&self) -> bool {
        self.ring.is_marker()
    }

    /// Reset to a fresh state (rapidgzip's `Block::reset`, deflate.hpp:670-697).
    /// Mirrors the C++ method's semantics: zeros out the state-machine
    /// counters AND optionally re-seeds with an initial 32 KiB window via
    /// `set_initial_window`. When no `initial_window` is provided the
    /// block remains in the "marker-emitting" mode (rapidgzip's
    /// `m_containsMarkerBytes = true` default at deflate.hpp:683); when
    /// one IS provided the next back-references resolve directly against
    /// the seeded window prefix in `output`.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn reset(&mut self, output: Option<&mut Vec<u16>>, initial_window: Option<&[u8]>) {
        self.at_end_of_block = false;
        self.at_end_of_file = false;
        self.is_last_block = false;
        self.compression_type = CompressionType::DynamicHuffman;
        self.padding = 0;
        self.uncompressed_size = 0;
        self.decoded_bytes = 0;
        self.decoded_bytes_at_block_start = 0;
        self.precode_cl = [0u8; MAX_PRECODE_COUNT];
        for v in self.literal_cl.iter_mut() {
            *v = 0;
        }
        self.literal_code_count = 0;
        self.distance_code_count = 0;
        self.backreferences.clear();
        // Reset ring tracking and re-prime the marker zone (WidthRing::reset).
        // The chunk we just finished may have overwritten the marker
        // zone (any decode crossing logical position 32768
        // wraps writes into ring[32768..65536]); a fresh chunk
        // needs the pre-init pattern restored so cross-chunk
        // back-refs in the new chunk's prefix produce correct
        // markers. Cost: 64 KiB write per chunk-recycle, well
        // amortized vs. allocating a fresh Block per chunk.
        self.ring.reset();
        #[cfg(any(
            all(
                feature = "isal-compression",
                not(feature = "pure-rust-inflate"),
                target_arch = "x86_64"
            ),
            pure_inflate_decode
        ))]
        {
            self.block_huffman_luts_ready = false;
        }

        // rapidgzip's `reset` ends with: `if (initialWindow) setInitialWindow(*initialWindow);`
        // (deflate.hpp:692-696). We mirror that contract — when both an
        // output buffer and a window are supplied, prime the chunk with
        // the window so the first block's back-references resolve to
        // literal bytes instead of MapMarkers.
        if let (Some(out), Some(window)) = (output, initial_window) {
            // Ignore the error path: this constructor is only called after
            // `new()` (which sets decoded_bytes = 0) so set_initial_window
            // can only fail when the output buffer is non-empty — a caller
            // bug we surface in the test below rather than via the reset
            // return value (matching rapidgzip's void return).
            let _ = Self::set_initial_window_impl(
                out,
                window,
                &mut self.decoded_bytes,
                &mut self.decoded_bytes_at_block_start,
                &mut self.ring,
            );
        }
    }

    /// Literal port of `Block::setInitialWindow`
    /// (vendor/.../gzip/deflate.hpp:1740-1785). Seeds a fresh `Block` with
    /// a 32 KiB sliding-window prefix so that subsequent back-references
    /// resolve to literal bytes rather than `MapMarkers`. This is the API
    /// rapidgzip uses for chunk N > 0 on its fast path
    /// (chunkdecoding/GzipChunk.hpp:190-268, `decodeChunkWithInflateWrapper`
    /// — the analog of our `decode_chunk_isal`).
    ///
    /// **Storage model.** Mirrors rapidgzip's: `Block` owns a 64 KiB
    /// internal ring (`output_ring`, the analog of vendor's
    /// `m_window16`); `setInitialWindow` writes the predecessor bytes
    /// into the ring's first slots and advances `ring_pos` /
    /// `ring_drained` / `decoded_bytes` so that subsequent back-refs
    /// resolve from the ring (via `emit_backref_ring`) rather than
    /// emitting markers. The caller's `output: Vec<u16>` receives
    /// ONLY decoded bytes from subsequent `read()` calls — the seed
    /// itself is not drained.
    ///
    /// When `out_pos >= 32768` and `distance <= 32768`,
    /// `marker_count = distance.saturating_sub(out_pos) = 0` —
    /// every back-reference resolves to a literal source byte in
    /// the ring at `ring_pos - distance`, identical to what
    /// rapidgzip's `getWindow()`-backed back-reference path produces.
    ///
    /// # Errors
    /// Returns `BlockError::ExceededWindowRange` if `initial_window.len() > MAX_WINDOW_SIZE`
    /// or if the block has already started decoding (mirrors rapidgzip's
    /// early-return at deflate.hpp:1751 — "before decoding has started").
    ///
    /// # Empty window
    /// Vendor-faithful (M3, deflate.hpp:1750-1759): an EMPTY `initial_window`
    /// still flips the ring to CLEAN mode (`m_containsMarkerBytes = false` at
    /// `:1757` is written OUTSIDE the `!initialWindow.empty()` arm). Used at
    /// stream starts where no history can be referenced — back-refs past the
    /// (zero-length) history then ERROR via the clean-mode range check, exactly
    /// like vendor's `distance > m_decodedBytes + nBytesRead` check
    /// (deflate.hpp:1652-1655). The historical pre-M3 no-op (stay in marker
    /// mode) was the recorded divergence resolved here.
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn set_initial_window(
        &mut self,
        output: &mut Vec<u16>,
        initial_window: &[u8],
    ) -> Result<(), BlockError> {
        Self::set_initial_window_impl(
            output,
            initial_window,
            &mut self.decoded_bytes,
            &mut self.decoded_bytes_at_block_start,
            &mut self.ring,
        )
    }

    /// Inner helper that drives the actual mutation. Split out so `reset`
    /// can re-use it without taking `&mut self` twice (the public method
    /// takes `&mut self` whereas `reset` holds fields-by-ref via
    /// destructuring inside `reset`).
    #[allow(clippy::ptr_arg)] // public API takes &mut Vec for symmetry with read()
    fn set_initial_window_impl(
        output: &mut Vec<u16>,
        initial_window: &[u8],
        decoded_bytes: &mut usize,
        decoded_bytes_at_block_start: &mut usize,
        ring: &mut super::width_ring::WidthRing,
    ) -> Result<(), BlockError> {
        // Rapidgzip's deflate.hpp:1751 guards on the `m_decodedBytes == 0 &&
        // m_windowPosition == 0` invariant — the API is only valid before
        // decoding starts. gzippy collapses both into `decoded_bytes == 0`.
        if *decoded_bytes != 0 || !output.is_empty() {
            return Err(BlockError::ExceededWindowRange);
        }
        if initial_window.len() > MAX_WINDOW_SIZE {
            return Err(BlockError::ExceededWindowRange);
        }
        // Empty window: vendor-faithful as of M3 — fall through to
        // `WidthRing::seed_window`, which flips the ring CLEAN even for a
        // zero-length seed (vendor deflate.hpp:1751-1758: the `:1757`
        // `m_containsMarkerBytes = false` write is OUTSIDE the
        // `!initialWindow.empty()` arm). The pre-M3 early-return no-op
        // (stay in marker mode) was the recorded divergence; it is resolved
        // here because M3 moves the seeded-chunk callers onto Block.
        //
        // Seed the u8 VIEW of the ring with the initial window
        // (`WidthRing::seed_window` — vendor's pre-decode prime path,
        // deflate.hpp:1748-1759: memcpy into the u8 view at [0, len),
        // cursor/drained = len, CLEAN mode from byte 0). Subsequent
        // back-refs resolve via the ring (emit_backref_ring_u8), so the
        // window must land there — not in the caller's output Vec; the seed
        // itself is NOT drained (it is predecessor output).
        ring.seed_window(initial_window)
            .map_err(|_| BlockError::ExceededWindowRange)?;
        // Mirror m_windowPosition = m_decodedBytes = initialWindow.size()
        // (deflate.hpp:1754-1755). `m_decodedBytes` stays a Block field.
        *decoded_bytes = initial_window.len();
        *decoded_bytes_at_block_start = initial_window.len();
        // Pre-M2 Block also primed the (dead-while-clean) marker counter to
        // the seed length; preserved verbatim for mechanical parity.
        ring.distance_to_last_marker = initial_window.len();
        Ok(())
    }

    /// Drain newly-written ring slots `[ring_drained .. ring_pos)` to
    /// the caller's output Vec via `extend_from_slice`. Handles the
    /// wrap-around case where the live region straddles the ring's
    /// physical end. Idempotent: a no-op when nothing new has been
    /// written since the last drain.
    ///
    /// Must be called before the gap `ring_pos - ring_drained` exceeds
    /// `RING_SIZE - MAX_WINDOW_SIZE` (otherwise the oldest undrained
    /// data gets overwritten by new writes; the lookback contract
    /// guarantees no read past `MAX_WINDOW_SIZE` bytes ago, so the
    /// preserved-history requirement is `MAX_WINDOW_SIZE` slots
    /// behind `ring_pos`).
    fn drain_to_output(&mut self, output: &mut impl MarkerSink) {
        let new_bytes = self.ring.pos - self.ring.drained;
        if new_bytes == 0 {
            return;
        }
        if self.ring.is_clean() {
            // POST-FLIP clean tail: `output_ring` is the u8 view, positions
            // are u8-LOGICAL (`% U8_RING_SIZE`). Plain u8 copy — no u16->u8
            // narrow. Vendor `result.data` u8 path (deflate.hpp:1285-1292).
            //
            // COPY-FREE DRAIN: push the (≤2) CONTIGUOUS u8 ring slices DIRECTLY
            // to the sink — no per-block `Vec::with_capacity` + byte-by-byte
            // fill (the symmetric of the marker branch's `push_slice` below).
            // The sink's `push_clean_u8` does ONE `extend_from_slice` memcpy.
            // Byte-identical to the prior `u8buf` materialization.
            // SAFETY: `ring8` valid for [0, U8_RING_SIZE); slices stay within it.
            let ring8 = self.ring.window16.as_ptr() as *const u8;
            let start = self.ring.drained % U8_RING_SIZE;
            unsafe {
                if start + new_bytes <= U8_RING_SIZE {
                    output.push_clean_u8(std::slice::from_raw_parts(ring8.add(start), new_bytes));
                } else {
                    let first = U8_RING_SIZE - start;
                    output.push_clean_u8(std::slice::from_raw_parts(ring8.add(start), first));
                    output.push_clean_u8(std::slice::from_raw_parts(ring8, new_bytes - first));
                }
            }
        } else {
            let start_idx = self.ring.drained % RING_SIZE;
            let end_idx_excl = (self.ring.drained + new_bytes) % RING_SIZE;
            if start_idx + new_bytes <= RING_SIZE {
                output.push_slice(&self.ring.window16[start_idx..start_idx + new_bytes]);
            } else {
                output.push_slice(&self.ring.window16[start_idx..]);
                output.push_slice(&self.ring.window16[..end_idx_excl]);
            }
        }
        self.ring.drained = self.ring.pos;
    }

    /// BENCH-ONLY clean-mode drain that mirrors `drain_to_output`'s
    /// `contains_marker_bytes == false` branch, but appends straight into a
    /// `Vec<u8>` (the engine-isolation bench's sink). Narrows each ring slot
    /// u16 -> u8 (clean bytes are guaranteed < 256) and advances
    /// `ring_drained`. NOT used by production routing.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    pub fn drain_clean_u8(&mut self, out: &mut Vec<u8>) {
        let new_bytes = self.ring.pos - self.ring.drained;
        if new_bytes == 0 {
            return;
        }
        debug_assert!(
            self.ring.is_clean(),
            "drain_clean_u8 requires clean (window-primed) mode"
        );
        out.reserve(new_bytes);
        for i in 0..new_bytes {
            let v = self.ring.window16[(self.ring.drained + i) % RING_SIZE];
            debug_assert!((v as usize) < 256, "clean drain emitted marker {v:#x}");
            out.push(v as u8);
        }
        self.ring.drained = self.ring.pos;
    }

    // ── Header parser (deflate.hpp:964-1156) ────────────────────────────────

    /// Literal port of `Block::readHeader<treatLastBlockAsError>`
    /// (deflate.hpp:964-1022).
    ///
    /// Reads the 3-bit BFINAL + BTYPE header, then dispatches into
    /// the per-type follow-up parsing (stored-block padding + LEN/NLEN,
    /// or dynamic-Huffman coding).
    ///
    /// On `treat_last_block_as_error == true`, returns
    /// `UnexpectedLastBlock` if BFINAL=1 — used by block-finder
    /// candidate validation (filters out final blocks).
    pub fn read_header(
        &mut self,
        bits: &mut Bits,
        treat_last_block_as_error: bool,
    ) -> Result<(), BlockError> {
        ensure_bits(bits, 3)?;
        let bfinal = (bits.peek() & 1) != 0;
        bits.consume(1);
        self.is_last_block = bfinal;
        #[cfg(any(
            all(
                feature = "isal-compression",
                not(feature = "pure-rust-inflate"),
                target_arch = "x86_64"
            ),
            pure_inflate_decode
        ))]
        {
            self.block_huffman_luts_ready = false;
        }
        if treat_last_block_as_error && bfinal {
            return Err(BlockError::UnexpectedLastBlock);
        }
        let btype = (bits.peek() & 0b11) as u8;
        bits.consume(2);
        self.compression_type = CompressionType::from_btype(btype);

        match self.compression_type {
            CompressionType::Uncompressed => {
                self.read_uncompressed_header(bits)?;
            }
            CompressionType::FixedHuffman => {
                // No further header parsing needed; the fixed coding
                // tables are static (RFC 1951 §3.2.6).
            }
            CompressionType::DynamicHuffman => {
                self.read_dynamic_huffman_coding(bits)?;
            }
            CompressionType::Reserved => {
                return Err(BlockError::InvalidCompression);
            }
        }

        self.at_end_of_block = false;
        self.decoded_bytes_at_block_start = self.decoded_bytes;
        self.backreferences.clear();
        #[cfg(any(
            all(
                feature = "isal-compression",
                not(feature = "pure-rust-inflate"),
                target_arch = "x86_64"
            ),
            pure_inflate_decode
        ))]
        {
            self.build_huffman_luts_for_block()?;
            self.block_huffman_luts_ready = true;
        }
        Ok(())
    }

    /// Build ISA-L lit/dist LUTs once per block header (vendor
    /// `readDynamicHuffmanCoding` tail at deflate.hpp:1137-1141).
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    fn build_huffman_luts_for_block(&mut self) -> Result<(), BlockError> {
        match self.compression_type {
            CompressionType::FixedHuffman => {
                if !self.lut_litlen_rebuild(&FIXED_LIT_LEN_LENGTHS[..]) {
                    return Err(BlockError::InvalidCodeLengths);
                }
                // Vendor parity (gzip/deflate.hpp:336): distance via the cached
                // reversed-bits decoder, not the ISA-L LUT. Mirror of the canonical
                // fallback build at :1516-1519.
                if self
                    .dist_hc
                    .initialize_from_lengths(&FIXED_DIST_LENGTHS[..], false)
                    != super::error::Error::None
                {
                    return Err(BlockError::InvalidCodeLengths);
                }
            }
            CompressionType::DynamicHuffman => {
                let split = self.literal_code_count;
                let end = split + self.distance_code_count;
                let mut lit_stack = [0u8; MAX_LITERAL_OR_LENGTH_SYMBOLS + 2];
                let mut dist_stack = [0u8; MAX_DISTANCE_SYMBOL_COUNT + 2];
                lit_stack[..split].copy_from_slice(&self.literal_cl[..split]);
                dist_stack[..end - split].copy_from_slice(&self.literal_cl[split..end]);
                if !self.lut_litlen_rebuild(&lit_stack[..split]) {
                    return Err(BlockError::InvalidCodeLengths);
                }
                // Vendor parity (gzip/deflate.hpp:336): distance via the cached
                // reversed-bits decoder, not the ISA-L LUT. Mirror of the canonical
                // fallback build at :1516-1519.
                if self
                    .dist_hc
                    .initialize_from_lengths(&dist_stack[..end - split], false)
                    != super::error::Error::None
                {
                    return Err(BlockError::InvalidCodeLengths);
                }
            }
            CompressionType::Uncompressed | CompressionType::Reserved => {}
        }
        Ok(())
    }

    fn read_uncompressed_header(&mut self, bits: &mut Bits) -> Result<(), BlockError> {
        // Pad to byte boundary; rapidgzip enforces padding == 0
        // (deflate.hpp:991-996).
        let bits_to_drain = bits.available() & 7;
        if bits_to_drain > 0 {
            // Need to consume bits up to the next byte boundary. Since
            // `bits` is bit-oriented, drain the low bits of the buffer.
            ensure_bits(bits, bits_to_drain)?;
            // Mirror of vendor `BitReader::read<N>()` masking
            // (BitReader.hpp:194-209) via the ported
            // `n_lowest_bits_set` (BitManipulation.hpp:60-73).
            let pad = (bits.peek() & n_lowest_bits_set(bits_to_drain as u8)) as u8;
            bits.consume(bits_to_drain);
            self.padding = pad;
            if pad != 0 {
                return Err(BlockError::NonZeroPadding);
            }
        }
        ensure_bits(bits, 32)?;
        let len = (bits.peek() & 0xFFFF) as u16;
        bits.consume(16);
        let nlen = (bits.peek() & 0xFFFF) as u16;
        bits.consume(16);
        if len != !nlen {
            return Err(BlockError::LengthChecksumMismatch);
        }
        self.uncompressed_size = len as usize;
        Ok(())
    }

    /// Literal port of `Block::readDynamicHuffmanCoding`
    /// (deflate.hpp:1025-1156). Reads HLIT/HDIST/HCLEN, decodes the
    /// precode lengths, builds the precode Huffman code, then decodes
    /// the literal/length + distance code lengths from it.
    ///
    /// On success, populates `self.precode_cl`, `self.literal_cl`,
    /// `self.literal_code_count`, and `self.distance_code_count`.
    pub fn read_dynamic_huffman_coding(&mut self, bits: &mut Bits) -> Result<(), BlockError> {
        ensure_bits(bits, 14)?;
        let literal_code_count = 257 + (bits.peek() & 0x1F) as usize;
        bits.consume(5);
        if literal_code_count > MAX_LITERAL_OR_LENGTH_SYMBOLS {
            return Err(BlockError::ExceededLiteralRange);
        }
        let distance_code_count = 1 + (bits.peek() & 0x1F) as usize;
        bits.consume(5);
        if distance_code_count > MAX_DISTANCE_SYMBOL_COUNT {
            return Err(BlockError::ExceededDistanceRange);
        }
        let code_length_count = 4 + (bits.peek() & 0xF) as usize;
        bits.consume(4);

        self.literal_code_count = literal_code_count;
        self.distance_code_count = distance_code_count;

        // Read the precode lengths in PRECODE_ALPHABET order.
        for v in self.precode_cl.iter_mut() {
            *v = 0;
        }
        for &slot in PRECODE_ALPHABET.iter().take(code_length_count) {
            ensure_bits(bits, PRECODE_BITS as u32)?;
            self.precode_cl[slot] = (bits.peek() & 0x7) as u8;
            bits.consume(PRECODE_BITS as u32);
        }

        // Decode HLIT + HDIST code lengths using the precode.
        let total = literal_code_count + distance_code_count;
        let lit_dist_lengths =
            read_literal_and_distance_code_lengths(bits, &self.precode_cl, total)?;
        self.literal_cl[..total].copy_from_slice(&lit_dist_lengths);

        // End-of-block symbol MUST have a non-zero code length.
        if self.literal_cl[END_OF_BLOCK_SYMBOL as usize] == 0 {
            return Err(BlockError::InvalidCodeLengths);
        }
        Ok(())
    }

    /// Public entry point — literal port of `Block::read`
    /// (deflate.hpp:1192-1300). Decodes up to `n_max_to_decode` bytes
    /// of the CURRENT block's payload, appending u16 values into
    /// `output`.
    pub fn read(
        &mut self,
        bits: &mut Bits,
        output: &mut impl MarkerSink,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        if self.eob() {
            return Ok(0);
        }
        let result = match self.compression_type {
            CompressionType::Reserved => Err(BlockError::InvalidCompression),
            CompressionType::Uncompressed => self.read_internal_uncompressed(bits, n_max_to_decode),
            CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {
                self.read_internal_compressed(bits, n_max_to_decode)
            }
        };
        // Mid-decode mode switch — mirror of vendor's check at
        // deflate.hpp:1282-1289. Once the decoded prefix is fully
        // clean (no markers anywhere reachable via back-ref), flip
        // `contains_marker_bytes` to false. Subsequent literals and
        // back-refs skip the marker-counter maintenance, matching
        // vendor's clean-decode performance path.
        //
        // Two trigger conditions:
        //   1. Counter reaches `RING_SIZE` — the entire ring has
        //      been overwritten by clean bytes since the last
        //      marker, including the original pre-init marker zone.
        //   2. Counter reaches `MAX_WINDOW_SIZE` AND equals
        //      `decoded_bytes` — the chunk has only emitted clean
        //      bytes so far, and 32 KiB of those are consecutive.
        //      The latter is the typical bootstrap-from-empty case
        //      (chunk has no marker emissions at all).
        let just_flipped = self.ring.should_flip(self.decoded_bytes);

        if just_flipped {
            // SEAM. `WidthRing::flip_in_place` is vendor's `setInitialWindow()`
            // at the flip site (deflate.hpp:1285): zero the not-yet-decoded
            // remainder, conflate the FULL u16 ring into the u8 view's tail,
            // re-base the cursor to u8-logical. This read's still-undrained
            // bytes are then emitted as a u8 view into the just-conflated
            // window (vendor `result.data = lastBuffers(window,
            // m_windowPosition, nBytesRead)`, deflate.hpp:1286) — no u16
            // re-read, no temp narrow (map DIV-4 closed). From the next
            // `read()` on, `<false>` decodes u8-DIRECT and the drain is a
            // plain u8 copy.
            let seam_len = self.ring.flip_in_place(self.decoded_bytes);
            if seam_len > 0 {
                output.push_clean_u8(self.ring.flipped_seam(seam_len));
            }
        } else {
            // Always drain — even on Err, any bytes already written should be
            // visible to the caller. `ring.drained` advances so subsequent calls
            // don't re-emit. The clean branch here is the post-flip u8 path.
            self.drain_to_output(output);
        }
        result
    }

    /// M2b (DIV-5): vendor's three stored-block special cases in `Block::read`
    /// (deflate.hpp:1212-1256), verified first-hand 2026-06-10:
    ///
    /// 1. `m_uncompressedSize >= MAX_WINDOW_SIZE` (:1214-1219), ANY width:
    ///    `m_windowPosition = m_uncompressedSize`, read the whole payload
    ///    straight into the u8 window at offset 0 — the >=32 KiB stored block
    ///    supersedes the entire back-reference window, so markers are dropped
    ///    (`m_containsMarkerBytes = false` at :1259) and the flip arms HERE,
    ///    not after 64 Ki clean symbols of generic arming.
    /// 2. markers present AND `m_distanceToLastMarkerByte +
    ///    m_uncompressedSize >= MAX_WINDOW_SIZE` (:1220-1242): downcast the last
    ///    `MAX_WINDOW_SIZE - uncompressedSize` clean u16 elements (vendor
    ///    throws `logic_error` if a marker survives — impossible because the
    ///    condition guarantees `remainingData.size() <=
    ///    m_distanceToLastMarkerByte`), place them at u8 offset 0, read the
    ///    payload after them, `m_windowPosition = MAX_WINDOW_SIZE`, flip.
    ///    Only the NEW stored bytes are emitted (`lastBuffers(window, 32768,
    ///    nBytesRead)` at :1263); the downcast prefix is prior output.
    ///    Applies to `uncompressedSize == 0` too (sync-flush empty stored
    ///    block after >=32 KiB of trailing clean ⇒ flip, zero bytes emitted).
    /// 3. already clean (:1243-1255): bulk `bitReader.read` into the u8 ring
    ///    via up-to-2 wraparound segments — vendor's "~400 MB/s → ~6 GB/s"
    ///    speedup over per-byte `appendToWindow`. No arming change (already
    ///    clean).
    ///
    /// All three consume the WHOLE payload (ignoring `nMaxToDecode`, exactly
    /// as vendor does — vendor's only finite-`nMaxToDecode` caller of stored
    /// blocks tolerates overshoot, and so does gzippy's
    /// `used_window_symbols`), set `m_atEndOfBlock = true`, and return only
    /// the new bytes. The fall-through (markers + `dist + size <
    /// MAX_WINDOW_SIZE`) stays the per-byte u16 path with generic arming.
    ///
    /// DEVIATION (documented): vendor reads a PARTIAL payload at EOF and
    /// still flips, returning `Error::EOF_UNCOMPRESSED` (:1269) — chunk-fatal
    /// in vendor's driver. gzippy instead takes the special cases only when
    /// the full payload is available and otherwise falls through to the
    /// per-byte path, preserving the existing commit-then-`Err` resumable
    /// truncation semantics byte-for-byte. Both arms reject the chunk on a
    /// truncated stored block, so output is unaffected.
    ///
    /// KILL-SWITCH: `GZIPPY_NO_STORED_FLIP=1` restores the exact pre-M2b
    /// behavior (per-byte path, generic arming only).
    ///
    /// Returns `Some(bytes_emitted)` when a special case ran.
    fn try_read_stored_special(&mut self, bits: &mut Bits) -> Option<usize> {
        use super::width_ring::RingWidth;

        if stored_flip_disabled() {
            return None;
        }
        let n = self.uncompressed_size;
        // Whole bytes immediately readable: aligned bytes still in the bit
        // buffer plus the unread remainder of the slice.
        let avail = (bits.available() / 8) as usize + (bits.data.len() - bits.pos);
        if avail < n {
            return None; // truncated payload: keep the per-byte error path
        }
        if n >= MAX_WINDOW_SIZE {
            // Case 1 (deflate.hpp:1214-1219). n <= 65535 < U8_RING_SIZE.
            // SAFETY: ring8 valid for [0, U8_RING_SIZE); n < U8_RING_SIZE;
            // `bits.data` and the ring never alias. Pointer derived
            // immediately before use (no intervening safe ring access).
            let ring8 = self.ring.window16.as_mut_ptr() as *mut u8;
            let dst = unsafe { std::slice::from_raw_parts_mut(ring8, n) };
            let got = read_stored_bytes_aligned(bits, dst);
            debug_assert_eq!(got, n, "avail gate guaranteed the full payload");
            self.ring.width = RingWidth::Clean;
            // Vendor `m_windowPosition = m_uncompressedSize` over the u8 view
            // (:1218); u8-logical re-base as in `WidthRing::flip_in_place`
            // (U8_RING_SIZE ≡ physical 0 so `pos - distance` cannot
            // underflow). Emit window = [drained, pos) = physical [0, n).
            self.ring.pos = U8_RING_SIZE + n;
            self.ring.drained = U8_RING_SIZE;
            STORED_FLIP_GE_WINDOW.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            stored_flip_debug_log("case1-ge-window", n);
        } else if self.ring.is_marker() && self.ring.distance_to_last_marker + n >= MAX_WINDOW_SIZE
        {
            // Case 2 (deflate.hpp:1220-1242).
            debug_assert!(self.ring.distance_to_last_marker <= self.decoded_bytes);
            let rem = MAX_WINDOW_SIZE - n; // > 0 here (n < MAX_WINDOW_SIZE)
                                           // Downcast the last `rem` u16 elements before the cursor through
                                           // a scratch buffer FIRST (the u8 destination [0, rem) physically
                                           // overlaps u16 source slots [0, rem/2)) — vendor's
                                           // `remainingData` vector (:1226-1236), stack here like the
                                           // flip's `conflatedBuffer`.
            let mut scratch = [0u8; MAX_WINDOW_SIZE];
            let pos_phys = self.ring.pos % RING_SIZE;
            for (k, out) in scratch.iter_mut().take(rem).enumerate() {
                let v = self.ring.window16[(pos_phys + RING_SIZE - rem + k) % RING_SIZE];
                // Vendor throws logic_error on a surviving marker (:1229-1231);
                // unreachable: rem <= distance_to_last_marker by the case
                // condition, so the last `rem` elements are clean.
                debug_assert!(v < 256, "marker {v:#x} in stored-flip prefix");
                *out = (v & 0xFF) as u8;
            }
            // SAFETY: as case 1; rem + n == MAX_WINDOW_SIZE < U8_RING_SIZE.
            // Pointer derived AFTER the safe `window16` reads above (stacked-
            // borrows: a safe access would invalidate an earlier raw borrow).
            let ring8 = self.ring.window16.as_mut_ptr() as *mut u8;
            unsafe { std::slice::from_raw_parts_mut(ring8, rem) }.copy_from_slice(&scratch[..rem]);
            let dst = unsafe { std::slice::from_raw_parts_mut(ring8.add(rem), n) };
            let got = read_stored_bytes_aligned(bits, dst);
            debug_assert_eq!(got, n, "avail gate guaranteed the full payload");
            self.ring.width = RingWidth::Clean;
            // Vendor `m_windowPosition = MAX_WINDOW_SIZE` (:1238); only the
            // new stored bytes drain: [drained, pos) = physical [rem, 32768).
            self.ring.pos = U8_RING_SIZE + MAX_WINDOW_SIZE;
            self.ring.drained = U8_RING_SIZE + rem;
            STORED_FLIP_CROSSING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            stored_flip_debug_log("case2-crossing", n);
        } else if self.ring.is_clean() {
            // Case 3 (deflate.hpp:1243-1255): bulk read into the u8 ring at
            // the cursor, <=2 wraparound segments (vendor `lastBuffers` over
            // the PRE-advanced position). Output drains via the standard
            // clean-mode `[drained, pos)` wrap-aware drain.
            let start = self.ring.pos % U8_RING_SIZE;
            let first = n.min(U8_RING_SIZE - start);
            // SAFETY: as case 1; [start, start+first) and [0, n-first) are
            // in-bounds; first <= n <= 65535 < U8_RING_SIZE.
            let ring8 = self.ring.window16.as_mut_ptr() as *mut u8;
            let dst = unsafe { std::slice::from_raw_parts_mut(ring8.add(start), first) };
            let mut got = read_stored_bytes_aligned(bits, dst);
            if first < n {
                let dst2 = unsafe { std::slice::from_raw_parts_mut(ring8, n - first) };
                got += read_stored_bytes_aligned(bits, dst2);
            }
            debug_assert_eq!(got, n, "avail gate guaranteed the full payload");
            self.ring.pos += n;
            STORED_CLEAN_BULK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            return None; // markers + dist + n < MAX_WINDOW_SIZE: per-byte path
        }
        // Common tail (vendor :1258-1261): whole payload consumed.
        self.uncompressed_size = 0;
        self.decoded_bytes += n;
        self.at_end_of_block = true;
        Some(n)
    }

    /// Literal port of `Block::readInternalUncompressed` semantics
    /// (deflate.hpp:1212-1278): consume `uncompressed_size` bytes from
    /// the bit stream (which are byte-aligned per the deflate spec)
    /// and emit them as literal u16 values into `output`. Caps at
    /// `n_max_to_decode`; sets `at_end_of_block` when the full payload
    /// is consumed.
    ///
    /// M2b: the vendor stored-block special cases (early flips + clean bulk
    /// read, deflate.hpp:1212-1256) are tried first — see
    /// [`Block::try_read_stored_special`]. `GZIPPY_NO_STORED_FLIP=1` disables
    /// them, restoring this per-byte path exactly.
    pub fn read_internal_uncompressed(
        &mut self,
        bits: &mut Bits,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        if let Some(n) = self.try_read_stored_special(bits) {
            return Ok(n);
        }
        // Stored blocks have no back-refs and no markers — every output
        // is a pure literal byte. We still write to the ring so the
        // public `read()` wrapper's drain emits these bytes in order
        // alongside compressed-block output.
        //
        // Uncompressed blocks are bounded at 65535 bytes (deflate
        // spec §3.2.4) which is one slot less than RING_SIZE; with at
        // most one drain at the end, no mid-call overflow check is
        // needed even at n_max_to_decode = usize::MAX.
        let to_read = self.uncompressed_size.min(n_max_to_decode);
        // Marker mode writes u16 (`% RING_SIZE`); post-flip clean mode writes
        // u8-DIRECT into the u8 view (`% U8_RING_SIZE`, u8-logical `pos`) — a
        // stored block can occur after the flip.
        let clean = self.ring.is_clean();
        let ring_ptr = self.ring.window16.as_mut_ptr();
        let ring8 = self.ring.window16.as_mut_ptr() as *mut u8;
        let mut pos = self.ring.pos;
        let mut read_count: usize = 0;
        // Pre-bind to keep error-path state-flush symmetric with the
        // compressed paths' commit! pattern (writes that survived
        // the per-iter ensure_bits get committed).
        for _ in 0..to_read {
            if let Err(e) = ensure_bits(bits, 8) {
                self.ring.pos = pos;
                self.uncompressed_size -= read_count;
                self.decoded_bytes += read_count;
                if self.ring.is_marker() {
                    self.ring.distance_to_last_marker += read_count;
                }
                return Err(e);
            }
            let byte = (bits.peek() & 0xFF) as u16;
            bits.consume(8);
            // SAFETY: ring_ptr valid for [0, RING_SIZE); ring8 valid for
            // [0, U8_RING_SIZE); index masks within each view's bound.
            unsafe {
                if clean {
                    ring8.add(pos % U8_RING_SIZE).write((byte & 0xFF) as u8);
                } else {
                    ring_ptr.add(pos % RING_SIZE).write(byte);
                }
            }
            pos += 1;
            read_count += 1;
        }
        self.ring.pos = pos;
        self.uncompressed_size -= read_count;
        self.decoded_bytes += read_count;
        // Stored blocks emit only literals — always increment the
        // marker-counter when we're still in marker mode. Mirror of
        // vendor's appendToWindow loop at deflate.hpp:1311-1322 for
        // the byte-write path.
        if self.ring.is_marker() {
            self.ring.distance_to_last_marker += read_count;
        }
        if self.uncompressed_size == 0 {
            self.at_end_of_block = true;
        }
        Ok(read_count)
    }

    /// Literal port of `Block::readInternalCompressed`
    /// (deflate.hpp:1510-1582). Decodes one Huffman-coded block body
    /// (Fixed or Dynamic) using the already-populated `literal_cl` /
    /// `distance_code_count` from `read_header`. Emits literals as u16
    /// values < 256; emits cross-chunk back-refs via the MapMarkers
    /// encoding from `replace_markers::MARKER_BASE`; in-chunk back-refs
    /// resolve immediately by copying from `output`.
    ///
    /// On hitting END_OF_BLOCK (symbol 256), sets `at_end_of_block`.
    ///
    /// **On x86_64 + isal-compression**, the lit/len + distance decode
    /// tables live as fields on `Block` (`isal_litlen`, `isal_dist`) and
    /// are rebuilt **in place** at the start of each block. This mirrors
    /// rapidgzip's `Block<>` members of type `HuffmanCodingISAL`
    /// (vendor/.../gzip/deflate.hpp:920-925) — the persistent-storage +
    /// in-place rebuild contract is what makes per-block bootstrap
    /// affordable. Earlier code allocated a 32 K-entry canonical-Huffman
    /// table per block (~50 MB/s); switching to the ISA-L LUT pushes
    /// header decode towards rapidgzip's ~340 MB/s/thread.
    ///
    /// Other platforms fall back to the canonical-Huffman implementation
    /// (only used by unit tests on those archs; the parallel-SM
    /// production path is x86_64+isal-only).
    /// Public entry — runtime-dispatches to the const-generic
    /// specialization based on `contains_marker_bytes`. Both
    /// specializations exist at compile time; the optimizer
    /// dead-strips marker-maintenance code in the `false` variant.
    /// Mirror of vendor's `if constexpr (containsMarkerBytes)`
    /// pattern at deflate.hpp:1311, 1379, 1652.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    pub fn read_internal_compressed(
        &mut self,
        bits: &mut Bits,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        if self.ring.is_marker() {
            self.read_internal_compressed_specialized::<true>(bits, n_max_to_decode)
        } else {
            self.read_internal_compressed_specialized::<false>(bits, n_max_to_decode)
        }
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[inline(always)]
    fn lut_litlen_rebuild(&mut self, litlen_lens: &[u8]) -> bool {
        #[cfg(pure_inflate_decode)]
        {
            self.lut_litlen.rebuild_from(litlen_lens)
        }
        #[cfg(not(any(
            all(
                feature = "isal-compression",
                not(feature = "pure-rust-inflate"),
                target_arch = "x86_64"
            ),
            pure_inflate_decode
        )))]
        {
            let _ = (self, litlen_lens);
            false
        }
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[inline(always)]
    fn lut_litlen_decode(&self, bits: &mut Bits) -> (u32, u32, u32) {
        #[cfg(pure_inflate_decode)]
        {
            let d = self.lut_litlen.decode(bits);
            (d.symbol, d.sym_count, d.bit_count)
        }
        #[cfg(not(any(
            all(
                feature = "isal-compression",
                not(feature = "pure-rust-inflate"),
                target_arch = "x86_64"
            ),
            pure_inflate_decode
        )))]
        {
            let _ = (self, bits);
            (0, 0, 0)
        }
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    fn read_internal_compressed_specialized<const CONTAINS_MARKERS: bool>(
        &mut self,
        bits: &mut Bits,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        match self.compression_type {
            CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {
                if !self.block_huffman_luts_ready {
                    self.build_huffman_luts_for_block()?;
                    self.block_huffman_luts_ready = true;
                }
            }
            _ => return Err(BlockError::InvalidCompression),
        }

        // Literal port of vendor's `readInternalCompressedMultiCached`
        // (vendor/.../gzip/deflate.hpp:1585-1666). Critical protocol notes
        // vs. the canonical-Huffman path:
        //
        // 1. **Multi-symbol packing.** ISA-L's `make_inflate_huff_code_lit_len`
        //    is called with `multisym = 0 = TRIPLE_SYM_FLAG`
        //    (vendor/.../external/isa-l/igzip/igzip_inflate.c:88, used at
        //    HuffmanCodingISAL.hpp:71). A single 12-bit lookup can return
        //    UP TO THREE literal symbols packed at bit offsets 0/8/16,
        //    with `sym_count` ∈ {1, 2, 3}. The decode loop must extract
        //    each via `symbol >>= 8` (vendor deflate.hpp:1612-1623).
        //    Treating sym_count as always-1 (the prior bug) drops the
        //    2nd/3rd packed bytes and desyncs the output.
        //
        // 2. **Length values are pre-expanded — no RFC extra-bit reads.**
        //    `set_and_expand_lit_len_huffcode` expands each length code
        //    (257..285) into 2^extra_bits separate table entries, each
        //    with the FULL length value baked in as `symbol = 254 + length`
        //    (igzip_inflate.c:359-372). The table's `bit_count` already
        //    consumes BOTH the Huffman code AND the length-extra bits.
        //    Vendor extracts length directly via `symbol - 254U`
        //    (deflate.hpp:1642). Calling `read_length_extra` on top of
        //    the LUT's bit consumption (the prior bug) reads garbage
        //    bits and desyncs the stream — the first cross-chunk back-
        //    reference then fires `InvalidHuffmanCode` on the next
        //    lit/len lookup.
        //
        // 3. **Length-code symbols can be up to MAX_LIT_LEN_SYM = 512**,
        //    not 285. The expansion pushes them into [257, ~512].
        //    Rejecting `sym > 285` (the prior bug) errors on legitimate
        //    high-length back-references.
        //
        // 4. **EOB check applies only to single-symbol entries**
        //    (sym_count == 1) — a packed pair/triple cannot end in EOB
        //    because pair/triple encoding skips when `sym >= 256`
        //    (igzip_inflate.c:473-476). Vendor expresses this as the
        //    `code <= 255 || sym_count > 1 → literal` branch at
        //    deflate.hpp:1615.
        const MAX_LIT_LEN_SYM: u32 = 512;
        const MAX_RUN_LENGTH: usize = 258;

        // Cap n_max_to_decode to ring capacity minus one max back-ref
        // length so a single call cannot overflow the ring. Mirror of
        // vendor's `nMaxToDecode = std::min(nMaxToDecode, window.size()
        // - MAX_RUN_LENGTH)` at gzip/deflate.hpp:1602. Without this,
        // chunks with high-marker-density blocks (typical for highly
        // compressible inputs in silesia at gzip -9) produce > 64 KiB
        // per call; writes wrap, overwrite undrained data, drain
        // double-counts in the wrap branch, and the decoded output
        // ends up both wrong-bytes and short of the gzip-trailer
        // ISIZE. Bootstrap's caller already loops on eob(), so the
        // returned `< n_max_to_decode` triggers another read() call.
        // Ring modulus depends on storage width: marker mode (`<true>`)
        // addresses the u16 ring (`RING_SIZE` slots); clean mode (`<false>`)
        // addresses the SAME backing as a u8 ring (`U8_RING_SIZE` = 2× slots,
        // vendor's `getWindow()` reinterpret — half the memory traffic). Both
        // modulus and `pos` units flip at the seam (handled in `read()`'s flip
        // hook before the first `<false>` call). Const-folded per instantiation.
        let ring_modulus: usize = if CONTAINS_MARKERS {
            RING_SIZE
        } else {
            U8_RING_SIZE
        };
        let n_max_to_decode = n_max_to_decode.min(ring_modulus - MAX_RUN_LENGTH);

        // All hot-path writes land in the ring buffer (vendor's
        // `m_window16` equivalent). `pos` is the LOGICAL write
        // position (never wraps); physical slot is `pos % ring_modulus`.
        //
        // SAFETY: both pointers are derived from `&mut self.ring.window16`
        // (a fixed `[u16; RING_SIZE]` = 128 KB = 131072-byte heap
        // allocation). `ring_ptr` (u16) is valid for [0, RING_SIZE);
        // `ring8` (u8, alignment 1 over the same bytes — vendor's
        // `reinterpret_cast`, deflate.hpp:893) is valid for
        // [0, U8_RING_SIZE). Marker mode (`CONTAINS_MARKERS`) writes only
        // through `ring_ptr % RING_SIZE`; clean mode writes only through
        // `ring8 % U8_RING_SIZE`. The two views are never live for the
        // same byte simultaneously (the seam flip is one-shot), and there
        // is no aliasing of either pointer elsewhere in this function.
        let ring_ptr = self.ring.window16.as_mut_ptr();
        let ring8 = self.ring.window16.as_mut_ptr() as *mut u8;
        let mut pos = self.ring.pos;
        let mut emitted: usize = 0;
        // Local copy of the marker counter — pulled into a register
        // for the hot loop. Written back to self on commit. With
        // `CONTAINS_MARKERS = false`, the increments and the
        // backward-scan branch are entirely dead code (compiler
        // const-folds the `if CONTAINS_MARKERS` checks). This is
        // the perf-meaningful effect of the const generic split:
        // marker-mode-only paperwork stops being paid at all on
        // chunks that have already switched to clean mode.
        let mut distance_marker = self.ring.distance_to_last_marker;

        macro_rules! commit {
            ($result:expr) => {{
                self.ring.pos = pos;
                self.decoded_bytes += emitted;
                self.ring.distance_to_last_marker = distance_marker;
                return $result;
            }};
        }

        // Causal-perturbation slow-injection knob. Snapshot the resolved
        // per-decode-event spin count + kind ONCE here, before the loop, so the
        // per-iteration cost when OFF is a single hoistable branch on a local
        // `== 0`. There are TWO independent knobs, selected by the const generic:
        //   * `<false>` clean path  → `GZIPPY_SLOW_MODE`        (spin_iters)
        //   * `<true>`  u16 marker  → `GZIPPY_SLOW_MARKER_MODE` (marker_spin_iters)
        // Each const-folds to 0 on the OTHER specialization, so the clean knob is
        // compiled away on the marker path and vice-versa — a clean-path slow-down
        // cannot leak into the u16-path measurement, and the u16-path knob fires at
        // the careful-loop inject site below (the `<true>` path always uses the
        // careful loop; the VAR_V fast loop is `!CONTAINS_MARKERS`-gated).
        // Byte-transparent (DUAL-SHA gate). See `slow_knob.rs`.
        let slow_spin: u64 = if CONTAINS_MARKERS {
            super::slow_knob::marker_spin_iters()
        } else {
            super::slow_knob::spin_iters()
        };
        // Gate the sleep-control kind on THIS specialization actually injecting
        // (`slow_spin != 0`). `GZIPPY_SLOW_KIND` is global, so without this gate a
        // MARKER+SLEEP run would set `slow_yield = true` on the `<false>` clean
        // instantiation too and knock it off the VAR_V fast loop (the `:1484`
        // gate is `... && !slow_yield`) even though the clean injection is zero —
        // contaminating the clean wall during a marker-only measurement (advisor
        // D2). With the gate, the control only affects the path it injects into.
        let slow_yield: bool = slow_spin != 0 && super::slow_knob::yield_kind();

        // ── VAR_V SPECULATIVE SOFTWARE-PIPELINED FAST LOOP (clean path only) ──
        // igzip trick #2 ported faithfully ONTO the production wrapping u8 ring
        // (NOT a flat buffer — the real `% U8_RING_SIZE` modulus, wrap-straddle
        // handling, resumable `n_max_to_decode` cap, drain, and CRC are ALL kept;
        // the bench's flat-buffer optimism is NOT reproduced). The fast loop runs
        // only while the physical write region (and the back-ref source region)
        // are far enough from the ring wrap that the speculative 8-byte packed
        // store + the word-copy back-ref can over-write without straddling, and
        // while input slop permits unchecked refills. When ANY guard fails it
        // breaks and the existing per-symbol careful loop below takes over — it
        // owns the wrap-straddle, the resumable boundary, and the block tail.
        //
        // Const-folded away entirely on the marker path (`CONTAINS_MARKERS`):
        // the speculative packed store is unsound when bytes may be marker u16s,
        // and `emit_backref_ring_u8` is u8-only, so the whole region is dead code
        // there (the `if !CONTAINS_MARKERS` gate const-folds to false).
        //
        // Output headroom the fast loop reserves so it can over-write without a
        // per-symbol bounds check: 8 speculative literal bytes + a 258-byte
        // max-length back-ref + a 16-byte word-copy overshoot (igzip asm:511).
        const FAST_OUT_SLOP: usize = 8 + MAX_RUN_LENGTH + 16;
        // Input slop so a refill can always read an 8-byte word (igzip
        // IN_BUFFER_SLOP, asm:48).
        const FAST_IN_SLOP: usize = 8;
        if !CONTAINS_MARKERS && slow_spin == 0 && !slow_yield {
            let ring8_fast = ring8;
            let in_end = bits.data.len();
            // Preload (igzip pipeline): decode the first symbol before entering
            // the loop so the back-ref branch resolves against an already-fetched
            // next symbol. `pre` is ALWAYS a fresh, un-consumed decode at the top
            // of each iteration (we consume its bits inside, then preload again),
            // so on ANY break the bit cursor sits exactly before `pre`'s bits and
            // the careful loop re-decodes from the same position — never desyncs.
            bits.refill();
            let mut pre = self.lut_litlen.decode(bits);
            'fast: loop {
                // Resumable cap + wrap headroom + input slop — all REAL overheads.
                let dst_phys = pos % U8_RING_SIZE;
                let out_ok = emitted + FAST_OUT_SLOP < n_max_to_decode
                    && dst_phys + FAST_OUT_SLOP <= U8_RING_SIZE;
                let in_ok = bits.pos + FAST_IN_SLOP < in_end;
                if !(out_ok && in_ok) {
                    break 'fast;
                }
                let sym0 = pre.symbol;
                let sym_count0 = pre.sym_count;
                let bit_count0 = pre.bit_count;
                if bit_count0 == 0 {
                    // Invalid code — let the careful loop produce the error with
                    // the cursor at `pre`'s start (bits not yet consumed).
                    break 'fast;
                }
                bits.consume(bit_count0);

                // SPECULATIVE 8-byte store of the packed bytes (igzip asm:518):
                // write all up-to-3 packed literal bytes unconditionally at the
                // non-wrapping dst, then advance by the count of LEADING LITERALS
                // only. Wrong trailing bytes are overwritten by the next packet
                // (or by the back-ref below). dst_phys + 8 <= U8_RING_SIZE holds
                // (FAST_OUT_SLOP >= 8), so the 8-byte store never straddles.
                unsafe {
                    let packed = (sym0 & 0x00FF_FFFF) as u64;
                    (ring8_fast.add(dst_phys) as *mut u64).write_unaligned(packed);
                }
                let mut s = sym0;
                let mut remaining = sym_count0;
                let mut lit_prefix = 0usize;
                let mut trailing_code: u16 = 0;
                let mut have_trailing = false;
                while remaining > 0 {
                    let code = (s & 0xFFFF) as u16;
                    if code <= 255 || remaining > 1 {
                        if remaining == 1 && code > 255 {
                            trailing_code = code;
                            have_trailing = true;
                            break;
                        }
                        lit_prefix += 1;
                        remaining -= 1;
                        s >>= 8;
                        continue;
                    }
                    trailing_code = code;
                    have_trailing = true;
                    break;
                }
                pos += lit_prefix;
                emitted += lit_prefix;

                if have_trailing {
                    let code = trailing_code;
                    if code == END_OF_BLOCK_SYMBOL {
                        self.at_end_of_block = true;
                        self.ring.pos = pos;
                        self.decoded_bytes += emitted;
                        self.ring.distance_to_last_marker = distance_marker;
                        return Ok(emitted);
                    }
                    if (code as u32) > MAX_LIT_LEN_SYM {
                        self.ring.pos = pos;
                        self.decoded_bytes += emitted;
                        self.ring.distance_to_last_marker = distance_marker;
                        return Err(BlockError::InvalidHuffmanCode);
                    }
                    let length = (code as usize).wrapping_sub(254);
                    if length != 0 {
                        // Distance via production's cached reversed-bits decoder
                        // (self.dist_hc) — SAME as the careful loop, NOT the
                        // bench's LutDistCode. Keeps byte-exactness + faithfulness.
                        let dsym = match self.dist_hc.decode(bits) {
                            Some(d) => d,
                            None => {
                                self.ring.pos = pos;
                                self.decoded_bytes += emitted;
                                self.ring.distance_to_last_marker = distance_marker;
                                return Err(BlockError::InvalidHuffmanCode);
                            }
                        };
                        if dsym as usize >= DISTANCE_BASE.len() {
                            self.ring.pos = pos;
                            self.decoded_bytes += emitted;
                            self.ring.distance_to_last_marker = distance_marker;
                            return Err(BlockError::InvalidHuffmanCode);
                        }
                        let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                        let distance = if extra > 0 {
                            if bits.available() < extra {
                                bits.refill();
                                if bits.available() < extra {
                                    self.ring.pos = pos;
                                    self.decoded_bytes += emitted;
                                    self.ring.distance_to_last_marker = distance_marker;
                                    return Err(BlockError::EndOfFile);
                                }
                            }
                            let mask = (1u64 << extra) - 1;
                            let v = (bits.peek() & mask) as usize;
                            bits.consume(extra);
                            DISTANCE_BASE[dsym as usize] as usize + v
                        } else {
                            DISTANCE_BASE[dsym as usize] as usize
                        };
                        if distance == 0 || distance > MAX_WINDOW_SIZE {
                            self.ring.pos = pos;
                            self.decoded_bytes += emitted;
                            self.ring.distance_to_last_marker = distance_marker;
                            return Err(BlockError::ExceededWindowRange);
                        }
                        if distance > self.decoded_bytes + emitted {
                            self.ring.pos = pos;
                            self.decoded_bytes += emitted;
                            self.ring.distance_to_last_marker = distance_marker;
                            return Err(BlockError::ExceededWindowRange);
                        }
                        // Back-ref copy via the SAME production routine the
                        // careful loop uses — fully wrap-safe (its non-overlap /
                        // RLE / overlap arms each mask every index `% U8_RING_SIZE`
                        // and only take the fast word-copy when the rounded run
                        // fits without straddling). The dst word-copy headroom is
                        // guaranteed by the top guard (`dst_phys + FAST_OUT_SLOP
                        // <= U8_RING_SIZE`); the source straddle is handled inside.
                        unsafe {
                            emit_backref_ring_u8(ring8_fast, &mut pos, distance, length);
                        }
                        self.record_backreference_for_sparsity(distance, length, emitted);
                        emitted += length;
                    }
                }

                // Preload next symbol (pipeline). Refill first so the decode and
                // any subsequent dist-extra reads have headroom.
                bits.refill();
                pre = self.lut_litlen.decode(bits);
            }
            // FALL THROUGH: `pre` was decoded but NOT consumed (we always break
            // with a fresh un-consumed `pre`), so the bit cursor is positioned
            // exactly before `pre`'s bits. The careful loop below re-decodes from
            // here — no state carried, no desync. `pos`/`emitted`/`at_end_of_block`
            // are all live and consistent.
        }

        // ── MARKER (u16) SPECULATIVE SOFTWARE-PIPELINED FAST LOOP ──
        // FAITHFUL PORT of rg's `readInternalCompressedMultiCached` (vendor
        // deflate.hpp:1585-1666) for the u16 marker window. rg runs the SAME tight
        // multi-cached loop for u16 markers as for u8 clean (it is templated on
        // `Window`; there is no separate slow marker path — the only marker-vs-clean
        // deltas are cheap constexpr-gated bookkeeping). gzippy's clean path already
        // got this fast loop (above); before this port the MARKER path was stuck on
        // the careful per-symbol loop while rg's marker path was its fast multi-cached
        // loop — the structural source of the measured ~1.69× decodeBlock gap on the
        // window-absent (34.5%-marker) workload.
        //
        // This mirrors the clean fast loop above with three faithful u16 deltas:
        //   1. Literal store width is u16 (vendor `appendToWindow` stores
        //      `Window::value_type` = uint16_t in marker mode, deflate.hpp:1319). The
        //      speculative store widens the up-to-3 packed literal BYTES (each <256)
        //      into u16 slots — value-identical to the careful loop's per-literal
        //      `ring_ptr.add(..).write(code & 0xFF)`.
        //   2. The marker counter `distance_marker` increments per clean literal
        //      (vendor's `++m_distanceToLastMarkerByte`, deflate.hpp:1315; literals
        //      are always <256 ⇒ always clean) and is recomputed across back-refs
        //      inside `emit_backref_ring::<true>` (the backward marker scan).
        //   3. NO `distance > decoded_bytes + emitted` window-range check — vendor
        //      const-folds it away for marker windows (`!containsMarkerBytes`,
        //      deflate.hpp:1652-1655). Back-refs in marker mode may legitimately
        //      reach into the (not-yet-known) predecessor window = the marker bytes.
        //
        // On ANY break the bit cursor sits exactly before `pre`'s un-consumed bits
        // and the careful loop below re-decodes from there — no state carried, no
        // desync (identical contract to the clean fast loop). Const-folded away
        // entirely on the clean path (`!CONTAINS_MARKERS`).
        if CONTAINS_MARKERS && slow_spin == 0 && !slow_yield {
            let in_end = bits.data.len();
            bits.refill();
            let mut pre = self.lut_litlen.decode(bits);
            'mfast: loop {
                // Resumable cap + wrap headroom + input slop. Headroom is in u16
                // SLOTS (ring modulus is RING_SIZE = 65536 slots = 128 KiB).
                let dst_phys = pos % RING_SIZE;
                let out_ok = emitted + FAST_OUT_SLOP < n_max_to_decode
                    && dst_phys + FAST_OUT_SLOP <= RING_SIZE;
                let in_ok = bits.pos + FAST_IN_SLOP < in_end;
                if !(out_ok && in_ok) {
                    break 'mfast;
                }
                let sym0 = pre.symbol;
                let sym_count0 = pre.sym_count;
                let bit_count0 = pre.bit_count;
                if bit_count0 == 0 {
                    break 'mfast;
                }
                bits.consume(bit_count0);

                // SPECULATIVE wide store of the (up-to-3) packed literal bytes,
                // WIDENED to u16. Each packed byte b is stored as the u16 value b
                // (< 256). One 8-byte unaligned write covers 4 u16 slots (≥ the 3
                // possible literals); the trailing slot(s) past `lit_prefix` are
                // overwritten by the next packet or the back-ref below. `dst_phys +
                // FAST_OUT_SLOP <= RING_SIZE` (FAST_OUT_SLOP >= 4 u16 slots) ⇒ the
                // 8-byte store never straddles the ring wrap. Value-identical to the
                // careful loop's per-literal `write(code & 0xFF)`.
                unsafe {
                    let p = sym0 as u64;
                    let widened: u64 = (p & 0xFF) | ((p & 0xFF00) << 8) | ((p & 0xFF_0000) << 16);
                    (ring_ptr.add(dst_phys) as *mut u64).write_unaligned(widened);
                }
                let mut s = sym0;
                let mut remaining = sym_count0;
                let mut lit_prefix = 0usize;
                let mut trailing_code: u16 = 0;
                let mut have_trailing = false;
                while remaining > 0 {
                    let code = (s & 0xFFFF) as u16;
                    if code <= 255 || remaining > 1 {
                        if remaining == 1 && code > 255 {
                            trailing_code = code;
                            have_trailing = true;
                            break;
                        }
                        lit_prefix += 1;
                        remaining -= 1;
                        s >>= 8;
                        continue;
                    }
                    trailing_code = code;
                    have_trailing = true;
                    break;
                }
                pos += lit_prefix;
                emitted += lit_prefix;
                // Vendor: per clean literal `++m_distanceToLastMarkerByte`.
                distance_marker += lit_prefix;

                if have_trailing {
                    let code = trailing_code;
                    if code == END_OF_BLOCK_SYMBOL {
                        self.at_end_of_block = true;
                        self.ring.pos = pos;
                        self.decoded_bytes += emitted;
                        self.ring.distance_to_last_marker = distance_marker;
                        return Ok(emitted);
                    }
                    if (code as u32) > MAX_LIT_LEN_SYM {
                        self.ring.pos = pos;
                        self.decoded_bytes += emitted;
                        self.ring.distance_to_last_marker = distance_marker;
                        return Err(BlockError::InvalidHuffmanCode);
                    }
                    let length = (code as usize).wrapping_sub(254);
                    if length != 0 {
                        let dsym = match self.dist_hc.decode(bits) {
                            Some(d) => d,
                            None => {
                                self.ring.pos = pos;
                                self.decoded_bytes += emitted;
                                self.ring.distance_to_last_marker = distance_marker;
                                return Err(BlockError::InvalidHuffmanCode);
                            }
                        };
                        if dsym as usize >= DISTANCE_BASE.len() {
                            self.ring.pos = pos;
                            self.decoded_bytes += emitted;
                            self.ring.distance_to_last_marker = distance_marker;
                            return Err(BlockError::InvalidHuffmanCode);
                        }
                        let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                        let distance = if extra > 0 {
                            if bits.available() < extra {
                                bits.refill();
                                if bits.available() < extra {
                                    self.ring.pos = pos;
                                    self.decoded_bytes += emitted;
                                    self.ring.distance_to_last_marker = distance_marker;
                                    return Err(BlockError::EndOfFile);
                                }
                            }
                            let mask = (1u64 << extra) - 1;
                            let v = (bits.peek() & mask) as usize;
                            bits.consume(extra);
                            DISTANCE_BASE[dsym as usize] as usize + v
                        } else {
                            DISTANCE_BASE[dsym as usize] as usize
                        };
                        if distance == 0 || distance > MAX_WINDOW_SIZE {
                            self.ring.pos = pos;
                            self.decoded_bytes += emitted;
                            self.ring.distance_to_last_marker = distance_marker;
                            return Err(BlockError::ExceededWindowRange);
                        }
                        // NO clean-mode `distance > decoded+emitted` check here —
                        // vendor const-folds it out for marker windows.
                        // Back-ref via the SAME production routine the careful loop
                        // uses for markers; it maintains `distance_marker` (the fast
                        // `>= distance` skip + backward marker scan) and is wrap-safe.
                        unsafe {
                            emit_backref_ring::<CONTAINS_MARKERS>(
                                ring_ptr,
                                &mut pos,
                                distance,
                                length,
                                &mut distance_marker,
                            );
                        }
                        self.record_backreference_for_sparsity(distance, length, emitted);
                        emitted += length;
                    }
                }

                bits.refill();
                pre = self.lut_litlen.decode(bits);
            }
        }

        while emitted < n_max_to_decode {
            // One slow-knob injection per decode event (this outer iteration =
            // exactly one Huffman codeword decode). No-op when `slow_spin == 0`.
            super::slow_knob::inject(slow_spin, slow_yield);
            // Single refill at the top of the outer iteration. After
            // `bits.refill()` returns, `bits.available()` is in
            // [56, 63] (libdeflate-style refill rounds DOWN to a
            // multiple of 8 plus the residue) — well above the 48
            // bits a worst-case back-ref iter consumes (20 for
            // lit/len decode + 15 for dist decode + 13 for dist extra).
            // With this guarantee, the per-decode `< 32` checks
            // inside `IsalLitLenCode::decode`, `IsalDistCode::decode`,
            // and the dist-extra `ensure_bits` all become
            // predictably-false branches that never trigger the
            // expensive 8-byte unaligned load. Net: ~1-2 refills
            // saved per iter on back-ref-heavy chunks.
            //
            // Calling refill when already near-full is a no-op:
            // `refill` advances `pos` by 0 when `bitsleft >= 56`
            // (see consume_first_decode.rs:259).
            bits.refill();
            let (symbol, sym_count, bit_count) = self.lut_litlen_decode(bits);
            if bit_count == 0 {
                // Inside `IsalLitLenCode::decode`, `symbol` is set to
                // `INVALID_SYMBOL` (0x1FFF) exactly when
                // `bit_count == 0`, so the prior
                // `|| symbol == INVALID_SYMBOL` half of the check was
                // redundant. Drop it: one cmp + branch saved per
                // outer iter.
                commit!(Err(BlockError::InvalidHuffmanCode));
            }
            // Consume the LUT-reported bit count (covers Huffman code +
            // any baked-in length-extra bits). Vendor's `seekAfterPeek`
            // at HuffmanCodingISAL.hpp:143 is equivalent.
            bits.consume(bit_count);
            let mut sym = symbol;
            let mut sym_count = sym_count;

            // Multi-symbol unpack loop — vendor deflate.hpp:1612-1661.
            loop {
                let code = (sym & 0xFFFF) as u16;
                if code <= 255 || sym_count > 1 {
                    // SAFETY: see ring_ptr / ring8 SAFETY note above. Marker
                    // mode stores u16 (`code & 0xFF` < 256) into the u16 ring;
                    // clean mode stores the byte u8-DIRECT into the u8 view
                    // (vendor `appendToWindow`, deflate.hpp:1319, one u8 store).
                    unsafe {
                        if CONTAINS_MARKERS {
                            ring_ptr.add(pos % RING_SIZE).write(code & 0xFF);
                        } else {
                            ring8.add(pos % U8_RING_SIZE).write((code & 0xFF) as u8);
                        }
                    }
                    pos += 1;
                    emitted += 1;
                    // Vendor's appendToWindow updates m_distanceToLastMarkerByte
                    // per literal (deflate.hpp:1311-1322): increment on
                    // clean bytes (literals always satisfy < 256), reset
                    // on marker writes. Markers in our path arrive
                    // exclusively via emit_backref_ring; literal writes
                    // here always increment. Const-folded out entirely
                    // when CONTAINS_MARKERS = false.
                    if CONTAINS_MARKERS {
                        distance_marker += 1;
                    }
                    sym_count -= 1;
                    if sym_count == 0 {
                        break;
                    }
                    sym >>= 8;
                    continue;
                }
                // sym_count == 1 here. Either EOB, length code, or
                // (defensively) an out-of-range symbol.
                if code == END_OF_BLOCK_SYMBOL {
                    self.at_end_of_block = true;
                    commit!(Ok(emitted));
                }
                if (code as u32) > MAX_LIT_LEN_SYM {
                    commit!(Err(BlockError::InvalidHuffmanCode));
                }
                let length = (code as usize).wrapping_sub(254);
                if length == 0 {
                    break;
                }
                // Vendor parity (gzip/deflate.hpp:336): distance via the cached
                // reversed-bits decoder. `decode` consumes the code bits
                // internally and returns the raw distance symbol — extra bits
                // are read below, exactly as the canonical fallback at :1580-1590.
                let dsym = match self.dist_hc.decode(bits) {
                    Some(d) => d,
                    None => commit!(Err(BlockError::InvalidHuffmanCode)),
                };
                if dsym as usize >= DISTANCE_BASE.len() {
                    commit!(Err(BlockError::InvalidHuffmanCode));
                }
                // Inlined `read_distance_extra` — eliminates the
                // function call, the double `if extra > 0` checks,
                // the `ensure_bits` function call (re-check + branch),
                // and the `n_lowest_bits_set` runtime-branch helper.
                // Outer-loop refill guarantees `bits.available() >=
                // 13 - dbit - decoded.bit_count` (worst case 13 bits
                // of extra for symbol 29), so the inner availability
                // check is also a predictably-false branch.
                let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                let distance = if extra > 0 {
                    if bits.available() < extra {
                        bits.refill();
                        if bits.available() < extra {
                            commit!(Err(BlockError::EndOfFile));
                        }
                    }
                    // `extra` is bounded by DISTANCE_EXTRA[29] = 13,
                    // so `1u64 << extra` is always well-defined.
                    let mask = (1u64 << extra) - 1;
                    let v = (bits.peek() & mask) as usize;
                    bits.consume(extra);
                    DISTANCE_BASE[dsym as usize] as usize + v
                } else {
                    DISTANCE_BASE[dsym as usize] as usize
                };
                if distance == 0 || distance > MAX_WINDOW_SIZE {
                    commit!(Err(BlockError::ExceededWindowRange));
                }
                // Clean-mode distance check — mirror of vendor's
                // `!containsMarkerBytes && distance > m_decodedBytes
                // + nBytesRead` at deflate.hpp:1652-1655. Whole
                // branch const-folded away when CONTAINS_MARKERS=true.
                if !CONTAINS_MARKERS && distance > self.decoded_bytes + emitted {
                    commit!(Err(BlockError::ExceededWindowRange));
                }
                unsafe {
                    if CONTAINS_MARKERS {
                        emit_backref_ring::<CONTAINS_MARKERS>(
                            ring_ptr,
                            &mut pos,
                            distance,
                            length,
                            &mut distance_marker,
                        );
                    } else {
                        // Clean tail: u8-direct copy (half the traffic, no
                        // marker scan). Vendor readInternal<false> back-ref.
                        emit_backref_ring_u8(ring8, &mut pos, distance, length);
                    }
                }
                self.record_backreference_for_sparsity(distance, length, emitted);
                emitted += length;
                break;
            }
        }
        self.ring.pos = pos;
        self.decoded_bytes += emitted;
        self.ring.distance_to_last_marker = distance_marker;
        Ok(emitted)
    }

    /// COPY-FREE-TO-FINAL (Stage 1) clean-phase decode into a CONTIGUOUS buffer.
    ///
    /// The faithful vendor `setInitialWindow` model: the post-flip clean
    /// (`<false>`) DEFLATE body is decoded straight into one contiguous output
    /// buffer (`chunk.data`'s reserved tail) with the 32 KiB predecessor window
    /// installed as a DICTIONARY PREFIX at `base[0..window_len)` — so back-refs
    /// of distance ≤ 32768 at the first clean byte resolve contiguously into that
    /// prefix (DecodedData.hpp:278-289 + deflate.hpp:1778), with NO u8 ring and
    /// NO ring→data drain memcpy. This is the byte-exact reference for the wired
    /// path (Stage 2); for Stage 1 it is exercised by the ring-vs-contiguous
    /// differential test (`contig_clean_matches_ring_clean_on_*`) to retire the
    /// addressing-retarget + contiguous-back-ref correctness risk.
    ///
    /// Semantics mirror `read_internal_compressed_specialized::<false>`'s CAREFUL
    /// loop exactly (the byte-exact reference; the VAR_V fast loop is a perf
    /// optimization that is byte-identical to it). Self-contained — it does NOT
    /// touch `self.ring.window16`/`ring_pos`/`ring_drained`; it advances
    /// `self.decoded_bytes` and `self.at_end_of_block` so the EOB/last-block
    /// contract is unchanged, and tracks the contiguous write index in `*pos`.
    ///
    /// Caller contract (the Engine-C grow-between-calls discipline):
    ///   * `base` valid for `[0, cap)`; `base[0..*pos)` already populated
    ///     (window prefix + prior output of THIS chunk).
    ///   * `*pos` is the logical write index (starts at `window_len` on the
    ///     first call after the flip).
    ///   * The buffer must NOT be reallocated while a returned `*pos` is live;
    ///     grow + re-fetch `base`/`cap` only BETWEEN calls.
    /// Caps writes at `cap - MAX_RUN_LENGTH - 8` so a single max-length back-ref
    /// plus the 8-byte word-copy overshoot can never exceed `cap`.
    ///
    /// # Errors
    /// Same `BlockError` set as the ring path (invalid code, exceeded window
    /// range, EOF). On error `*pos`/`decoded_bytes` reflect the bytes written
    /// before the failing symbol.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    pub fn decode_clean_into_contig(
        &mut self,
        bits: &mut Bits,
        base: *mut u8,
        cap: usize,
        pos: &mut usize,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        debug_assert!(
            self.ring.is_clean(),
            "decode_clean_into_contig requires clean (window-primed) mode"
        );
        match self.compression_type {
            CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {
                if !self.block_huffman_luts_ready {
                    self.build_huffman_luts_for_block()?;
                    self.block_huffman_luts_ready = true;
                }
            }
            _ => return Err(BlockError::InvalidCompression),
        }

        const MAX_LIT_LEN_SYM: u32 = 512;
        const MAX_RUN_LENGTH: usize = 258;
        // Contiguous-buffer cap: leave room for one max back-ref + word overshoot.
        //
        // Headroom proof (advisor-required): one outer iteration writes EITHER
        // ≤3 packed literals OR exactly one back-ref of length ≤ MAX_RUN_LENGTH,
        // NEVER both — the ISA-L LUT's multi-symbol packing (sym_count ∈ {1,2,3})
        // stops at any symbol ≥ 256, so a pair/triple slot is all literals and a
        // back-ref appears only at sym_count == 1 (igzip_inflate.c:473-476, see
        // the multi-symbol notes on read_internal_compressed_specialized). The
        // outer guard keeps `*pos ≤ out_room - 1 = cap - (MAX_RUN_LENGTH+8) - 1`
        // at any back-ref, and `emit_backref_contig`'s word path touches up to
        // `*pos + ((MAX_RUN_LENGTH+7)&!7) = *pos + 264`, max end `= cap - 3 < cap`.
        // The literal-triple path ends at `≤ cap - 264`. So MAX_RUN_LENGTH+8 is
        // provably sufficient — but the proof DEPENDS on `MAX sym_count == 3` and
        // "multi-symbol ⇒ all literals"; a LUT change that packed a length code
        // into a multi-symbol slot would invalidate it.
        let out_room = cap.saturating_sub(MAX_RUN_LENGTH + 8);
        debug_assert!(*pos <= out_room, "decode_clean_into_contig: no spare");
        let local_cap = n_max_to_decode.min(out_room.saturating_sub(*pos));

        let mut emitted: usize = 0;

        // Causal-perturbation slow-injection (measurement-only, byte-transparent;
        // OFF==identity, DUAL-SHA gated). This is the PRODUCTION gzippy-native FOLD
        // clean loop (post-flip contig tail) — the prior `GZIPPY_SLOW_MODE` knob
        // lived only on the ring-based `read_internal_compressed_specialized::<false>`
        // path and did NOT perturb this contig loop, so a clean-path perturbation
        // could not test where the residual native_fold→ocl_cf gap actually is.
        // Wire the SAME clean knob (`GZIPPY_SLOW_MODE` / `GZIPPY_SLOW_KIND`) here so
        // slowing the contig symbol-decode by a known factor and watching the
        // interleaved T8 wall answers it causally. Snapshot once before the loop.
        let slow_spin: u64 = super::slow_knob::spin_iters();
        let slow_yield: bool = slow_spin != 0 && super::slow_knob::yield_kind();
        // DECODE-COMPUTE vs STORE-BANDWIDTH localization knobs (measurement-only,
        // byte-transparent; see slow_knob.rs). These do NOT force the careful loop
        // (the fast-loop gate stays `slow_spin == 0` only) so the perturbation
        // lands on the PRODUCTION fast path. Snapshot once before the loop.
        let dec_spin: u64 = super::slow_knob::decode_spin_iters();
        let dec_yield: bool = super::slow_knob::localize_yield_kind(dec_spin);
        let st_spin: u64 = super::slow_knob::store_spin_iters();
        let st_yield: bool = super::slow_knob::localize_yield_kind(st_spin);
        macro_rules! commit {
            ($result:expr) => {{
                self.decoded_bytes += emitted;
                return $result;
            }};
        }

        // ── SPECULATIVE SOFTWARE-PIPELINED FAST LOOP (contig clean) ──────────
        // Faithful port of the ring-path VAR_V fast loop (igzip asm:518) onto the
        // CONTIGUOUS copy-free-to-final tail. The contig case is strictly simpler
        // than the ring: no `% U8_RING_SIZE` modulus, no wrap-straddle — back-refs
        // resolve directly from `base[*pos - distance]` and the speculative 8-byte
        // store never straddles a ring boundary. Same pipeline contract as the
        // ring loop: `pre` is always a fresh UN-consumed decode at the top of each
        // iteration, so on ANY break the bit cursor sits exactly before `pre`'s
        // bits and the careful loop below re-decodes from the same position — never
        // desyncs, no state carried.
        //
        // Gated `slow_spin == 0 && !slow_yield` so the causal-perturbation knob
        // (GZIPPY_SLOW_MODE) forces the careful loop and its per-event inject still
        // fires — identical gating to the ring fast loop (:1501).
        //
        // Headroom: at any literal `*pos <= out_room - 1 = cap - (MAX_RUN_LENGTH+8)
        // - 1 = cap - 267`, so the speculative 8-byte store touches at most
        // `*pos + 8 <= cap - 259 < cap` — never OOB. The back-ref word-copy headroom
        // is the SAME MAX_RUN_LENGTH+8 the careful loop relies on. Input slop:
        // FAST_IN_SLOP bytes ahead of `bits.pos` so refills stay on the fast word
        // path; when input slop fails we break to the careful loop (which owns the
        // block tail / resumable boundary).
        const FAST_OUT_SLOP: usize = 8;
        const FAST_IN_SLOP: usize = 8;
        if slow_spin == 0 && !slow_yield {
            let in_end = bits.data.len();
            bits.refill();
            let mut pre = self.lut_litlen.decode(bits);
            super::slow_knob::inject_localize(dec_spin, dec_yield);
            'fast: loop {
                // Out headroom: keep `*pos + FAST_OUT_SLOP` within the reserved
                // word-copy region (out_room already reserves MAX_RUN_LENGTH+8 so
                // both the 8-byte literal store and a max back-ref fit). In slop:
                // enough bytes ahead for the unchecked fast refill.
                let out_ok = emitted + FAST_OUT_SLOP < local_cap;
                let in_ok = bits.pos + FAST_IN_SLOP < in_end;
                if !(out_ok && in_ok) {
                    break 'fast;
                }
                let sym0 = pre.symbol;
                let sym_count0 = pre.sym_count;
                let bit_count0 = pre.bit_count;
                if bit_count0 == 0 {
                    // Invalid code — let the careful loop produce the error with
                    // the cursor at `pre`'s start (bits not yet consumed).
                    break 'fast;
                }
                bits.consume(bit_count0);

                // Resolve leading-literal count + any trailing non-literal.
                // Fast-path the DOMINANT single-symbol packet (sym_count==1): a
                // lone literal writes ONE byte (no 8-byte store / no inner
                // bookkeeping — the wide store would only WASTE bandwidth on a
                // 1-byte packet, advisor item Q3); a lone non-literal goes
                // straight to the trailing handler. Only `sym_count > 1` (a real
                // multi-literal pack) takes the speculative 8-byte packed store
                // the technique exists to exploit (igzip asm:518): write up to 3
                // packed literal bytes (symbol bits 0..24) at the contiguous dst,
                // advance by the LEADING-LITERAL count; a wrong trailing byte is
                // overwritten by the next store or the back-ref.
                // SAFETY (both arms): `*pos + 8 <= cap` (headroom proof above);
                // `base` valid for `[0, cap)`.
                let mut trailing_code: u16 = 0;
                let mut have_trailing = false;
                if sym_count0 == 1 {
                    let code = (sym0 & 0xFFFF) as u16;
                    if code <= 255 {
                        unsafe {
                            base.add(*pos).write(code as u8);
                        }
                        *pos += 1;
                        emitted += 1;
                        super::slow_knob::inject_localize(st_spin, st_yield);
                    } else {
                        trailing_code = code;
                        have_trailing = true;
                    }
                } else {
                    unsafe {
                        let packed = (sym0 & 0x00FF_FFFF) as u64;
                        (base.add(*pos) as *mut u64).write_unaligned(packed);
                    }
                    let mut s = sym0;
                    let mut remaining = sym_count0;
                    let mut lit_prefix = 0usize;
                    while remaining > 0 {
                        let code = (s & 0xFFFF) as u16;
                        if code <= 255 || remaining > 1 {
                            if remaining == 1 && code > 255 {
                                trailing_code = code;
                                have_trailing = true;
                                break;
                            }
                            lit_prefix += 1;
                            remaining -= 1;
                            s >>= 8;
                            continue;
                        }
                        trailing_code = code;
                        have_trailing = true;
                        break;
                    }
                    *pos += lit_prefix;
                    emitted += lit_prefix;
                    super::slow_knob::inject_localize(st_spin, st_yield);
                }

                if have_trailing {
                    let code = trailing_code;
                    if code == END_OF_BLOCK_SYMBOL {
                        self.at_end_of_block = true;
                        commit!(Ok(emitted));
                    }
                    if (code as u32) > MAX_LIT_LEN_SYM {
                        commit!(Err(BlockError::InvalidHuffmanCode));
                    }
                    let length = (code as usize).wrapping_sub(254);
                    if length != 0 {
                        let dsym = match self.dist_hc.decode(bits) {
                            Some(d) => d,
                            None => commit!(Err(BlockError::InvalidHuffmanCode)),
                        };
                        super::slow_knob::inject_localize(dec_spin, dec_yield);
                        if dsym as usize >= DISTANCE_BASE.len() {
                            commit!(Err(BlockError::InvalidHuffmanCode));
                        }
                        let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                        let distance = if extra > 0 {
                            if bits.available() < extra {
                                bits.refill();
                                if bits.available() < extra {
                                    commit!(Err(BlockError::EndOfFile));
                                }
                            }
                            let mask = (1u64 << extra) - 1;
                            let v = (bits.peek() & mask) as usize;
                            bits.consume(extra);
                            DISTANCE_BASE[dsym as usize] as usize + v
                        } else {
                            DISTANCE_BASE[dsym as usize] as usize
                        };
                        if distance == 0 || distance > MAX_WINDOW_SIZE {
                            commit!(Err(BlockError::ExceededWindowRange));
                        }
                        // Clean-mode range check — mirror of the careful loop
                        // (deflate.hpp:1652-1655): valid iff `distance <= *pos`.
                        if distance > *pos {
                            commit!(Err(BlockError::ExceededWindowRange));
                        }
                        // SAFETY: `distance <= *pos`; `*pos + ((length+7)&!7) <= cap`
                        // (out_room reserved MAX_RUN_LENGTH + 8 headroom).
                        unsafe {
                            emit_backref_contig(base, pos, distance, length);
                        }
                        super::slow_knob::inject_localize(st_spin, st_yield);
                        self.record_backreference_for_sparsity(distance, length, emitted);
                        emitted += length;
                    }
                }

                // Preload next symbol (pipeline). Refill first so the decode and
                // any subsequent dist-extra reads have headroom.
                bits.refill();
                pre = self.lut_litlen.decode(bits);
                super::slow_knob::inject_localize(dec_spin, dec_yield);
            }
            // FALL THROUGH: `pre` was decoded but NOT consumed (every break leaves
            // a fresh un-consumed `pre`), so the bit cursor sits exactly before
            // `pre`'s bits. The careful loop below re-decodes from here — no state
            // carried, no desync. `*pos`/`emitted`/`at_end_of_block` are all live.
        }

        while emitted < local_cap {
            super::slow_knob::inject(slow_spin, slow_yield);
            bits.refill();
            let (symbol, sym_count, bit_count) = self.lut_litlen_decode(bits);
            super::slow_knob::inject_localize(dec_spin, dec_yield);
            if bit_count == 0 {
                commit!(Err(BlockError::InvalidHuffmanCode));
            }
            bits.consume(bit_count);
            let mut sym = symbol;
            let mut sym_count = sym_count;

            loop {
                let code = (sym & 0xFFFF) as u16;
                if code <= 255 || sym_count > 1 {
                    // SAFETY: `*pos < out_room <= cap`; `base` valid for [0, cap).
                    unsafe {
                        base.add(*pos).write((code & 0xFF) as u8);
                    }
                    *pos += 1;
                    emitted += 1;
                    super::slow_knob::inject_localize(st_spin, st_yield);
                    sym_count -= 1;
                    if sym_count == 0 {
                        break;
                    }
                    sym >>= 8;
                    continue;
                }
                // sym_count == 1: EOB, length code, or out-of-range.
                if code == END_OF_BLOCK_SYMBOL {
                    self.at_end_of_block = true;
                    commit!(Ok(emitted));
                }
                if (code as u32) > MAX_LIT_LEN_SYM {
                    commit!(Err(BlockError::InvalidHuffmanCode));
                }
                let length = (code as usize).wrapping_sub(254);
                if length == 0 {
                    break;
                }
                let dsym = match self.dist_hc.decode(bits) {
                    Some(d) => d,
                    None => commit!(Err(BlockError::InvalidHuffmanCode)),
                };
                super::slow_knob::inject_localize(dec_spin, dec_yield);
                if dsym as usize >= DISTANCE_BASE.len() {
                    commit!(Err(BlockError::InvalidHuffmanCode));
                }
                let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                let distance = if extra > 0 {
                    if bits.available() < extra {
                        bits.refill();
                        if bits.available() < extra {
                            commit!(Err(BlockError::EndOfFile));
                        }
                    }
                    let mask = (1u64 << extra) - 1;
                    let v = (bits.peek() & mask) as usize;
                    bits.consume(extra);
                    DISTANCE_BASE[dsym as usize] as usize + v
                } else {
                    DISTANCE_BASE[dsym as usize] as usize
                };
                if distance == 0 || distance > MAX_WINDOW_SIZE {
                    commit!(Err(BlockError::ExceededWindowRange));
                }
                // Clean-mode range check — mirror of vendor deflate.hpp:1652-1655.
                // The window prefix occupies `base[0..window_len)`, so a distance
                // reaching into it is valid iff `distance <= *pos` (the contiguous
                // index). `self.decoded_bytes + emitted` equals `*pos - window_len`
                // by construction; the prefix adds the extra `window_len` reach,
                // which is exactly what `distance <= *pos` permits.
                if distance > *pos {
                    commit!(Err(BlockError::ExceededWindowRange));
                }
                // SAFETY: `distance <= *pos`; `*pos + ((length+7)&!7) <= cap`
                // (out_room reserved MAX_RUN_LENGTH + 8 headroom).
                unsafe {
                    emit_backref_contig(base, pos, distance, length);
                }
                super::slow_knob::inject_localize(st_spin, st_yield);
                self.record_backreference_for_sparsity(distance, length, emitted);
                emitted += length;
                break;
            }
        }
        self.decoded_bytes += emitted;
        Ok(emitted)
    }

    /// Copy-free-to-final STORED (uncompressed) clean block: drain the
    /// (already-headered) uncompressed payload byte-for-byte straight into the
    /// caller's contiguous buffer at `base[*pos..]`. Stored blocks have no
    /// back-refs and no markers (deflate spec §3.2.4), so this is a pure byte
    /// copy — the contig sibling of `read_internal_uncompressed`'s clean branch
    /// (marker_inflate.rs:1195). Sets `at_end_of_block` when the full payload is
    /// consumed; advances `*pos`/`decoded_bytes`. Caps at `n_max_to_decode` AND
    /// the spare in `[*pos, cap)`. Requires clean mode (post-flip).
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    pub fn decode_clean_stored_into_contig(
        &mut self,
        bits: &mut Bits,
        base: *mut u8,
        cap: usize,
        pos: &mut usize,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        debug_assert!(
            self.ring.is_clean(),
            "decode_clean_stored_into_contig requires clean mode"
        );
        debug_assert!(self.compression_type == CompressionType::Uncompressed);
        let spare = cap.saturating_sub(*pos);
        let to_read = self.uncompressed_size.min(n_max_to_decode).min(spare);
        // M2b (DIV-5 case 3, contig sibling): bulk byte read straight into the
        // contiguous destination — byte-identical to the per-byte loop below
        // (same `to_read` bytes, same cursor advance), just one memcpy instead
        // of per-byte bit-buffer pulls. Vendor's clean stored read is the same
        // bulk `bitReader.read` (deflate.hpp:1243-1255); the destination here
        // is gzippy's kept DIV-2 contig deviation. Same kill-switch + full-
        // availability gate as the ring path (short payloads keep the
        // per-byte commit-then-`Err` truncation semantics).
        if !stored_flip_disabled() {
            let avail = (bits.available() / 8) as usize + (bits.data.len() - bits.pos);
            if avail >= to_read {
                // SAFETY: `base` valid for [0, cap); `to_read <= spare = cap -
                // *pos`; `bits.data` and the chunk buffer never alias.
                let dst = unsafe { std::slice::from_raw_parts_mut(base.add(*pos), to_read) };
                let got = read_stored_bytes_aligned(bits, dst);
                debug_assert_eq!(got, to_read, "avail gate guaranteed the payload");
                *pos += to_read;
                self.uncompressed_size -= to_read;
                self.decoded_bytes += to_read;
                if self.uncompressed_size == 0 {
                    self.at_end_of_block = true;
                }
                STORED_CONTIG_BULK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(to_read);
            }
        }
        let mut read_count: usize = 0;
        for _ in 0..to_read {
            if let Err(e) = ensure_bits(bits, 8) {
                self.uncompressed_size -= read_count;
                self.decoded_bytes += read_count;
                return Err(e);
            }
            let byte = (bits.peek() & 0xFF) as u8;
            bits.consume(8);
            // SAFETY: `*pos < cap` (bounded by `spare`); `base` valid for [0, cap).
            unsafe {
                base.add(*pos).write(byte);
            }
            *pos += 1;
            read_count += 1;
        }
        self.uncompressed_size -= read_count;
        self.decoded_bytes += read_count;
        if self.uncompressed_size == 0 {
            self.at_end_of_block = true;
        }
        Ok(read_count)
    }

    /// BENCH-ONLY clean-mode sibling of
    /// `read_internal_compressed_specialized::<false>` with the round-2
    /// inner-Huffman techniques layered in behind const-generic flags so the
    /// engine-isolation bench can measure each stack independently:
    ///
    ///   * **E2** — AVX2 32-byte (16-u16) wide back-ref copy in the non-overlap
    ///     arm, runtime-gated via `is_x86_feature_detected!("avx2")`. Under
    ///     x86-64-v2 (Rosetta) the scalar word-copy fallback runs and is
    ///     byte-identical to `emit_backref_ring::<false>`.
    ///   * **E3** — packed multi-literal store: a TRIPLE_SYM pair/triple
    ///     (`sym_count >= 2`, all guaranteed < 256) is written with ONE wide
    ///     u32/u64 store instead of the per-lane shift loop.
    ///   * **E4** — refill amortized over multiple symbols: the per-iteration
    ///     `refill()` is elided when ≥ 48 bits are already buffered (a
    ///     worst-case back-ref). The decode primitives keep their own < 32-bit
    ///     safety refill, so this stays byte-exact.
    ///
    /// With `E2 = E3 = E4 = false` this decodes byte-for-byte identically to
    /// `read_internal_compressed_specialized::<false>`. Requires CLEAN mode
    /// (window-primed). NOT wired into production `read()` — bench-only.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        all(pure_inflate_decode, target_arch = "x86_64")
    ))]
    pub fn read_clean_e234<const E2: bool, const E3: bool, const E4: bool>(
        &mut self,
        bits: &mut Bits,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        debug_assert!(
            self.ring.is_clean(),
            "read_clean_e234 requires clean (window-primed) mode"
        );
        match self.compression_type {
            CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {
                if !self.block_huffman_luts_ready {
                    self.build_huffman_luts_for_block()?;
                    self.block_huffman_luts_ready = true;
                }
            }
            _ => return Err(BlockError::InvalidCompression),
        }

        const MAX_LIT_LEN_SYM: u32 = 512;
        const MAX_RUN_LENGTH: usize = 258;
        let n_max_to_decode = n_max_to_decode.min(RING_SIZE - MAX_RUN_LENGTH);

        let ring_ptr = self.ring.window16.as_mut_ptr();
        let mut pos = self.ring.pos;
        let mut emitted: usize = 0;

        // E2 AVX2 dispatch resolved ONCE before the loop. `E2` is const, so
        // when false the whole product folds to a compile-time `false` and the
        // AVX2 arm in `emit_backref_ring_clean` is dead-stripped. Under Rosetta
        // (x86-64-v2, no AVX2) this is false at runtime => scalar fallback.
        let use_avx2 = E2 && std::is_x86_feature_detected!("avx2");

        macro_rules! commit {
            ($result:expr) => {{
                self.ring.pos = pos;
                self.decoded_bytes += emitted;
                return $result;
            }};
        }

        'outer: while emitted < n_max_to_decode {
            // E4: amortize the refill across symbols. After a refill bitsleft is
            // in [56, 63]; a worst-case back-ref consumes ~48 bits, so when ≥ 48
            // are already buffered the refill is provably a no-op and is skipped.
            // Without E4 we refill unconditionally (production behaviour).
            if E4 {
                if bits.available() < 48 {
                    bits.refill();
                }
            } else {
                bits.refill();
            }
            let (symbol, sym_count, bit_count) = self.lut_litlen_decode(bits);
            if bit_count == 0 {
                commit!(Err(BlockError::InvalidHuffmanCode));
            }
            bits.consume(bit_count);
            let mut sym = symbol;
            let mut sym_count = sym_count;

            // E3: packed multi-literal store. In a TRIPLE_SYM entry only the
            // first `sym_count - 1` symbols are GUARANTEED literals; the LAST
            // one (when count reaches 1) may be a length/EOB code (verified on
            // silesia: e.g. `sym=0x10273 cnt=2` = literal 0x73 + length-258).
            // So this fast path applies ONLY when the last symbol is ALSO a
            // literal (`last_code <= 255`) — then all lanes are bytes and we
            // store them with one wide write, byte-identical to the per-lane
            // `code & 0xFF` loop. Otherwise fall through to the faithful loop,
            // which decodes the trailing length code (and its distance bits).
            if E3 && (sym_count == 2 || sym_count == 3) {
                let last_code = ((sym >> (8 * (sym_count - 1))) & 0xFFFF) as u16;
                if last_code <= 255 {
                    let dst_phys = pos % RING_SIZE;
                    let b0 = (sym & 0xFF) as u64;
                    let b1 = ((sym >> 8) & 0xFF) as u64;
                    if sym_count == 2 && dst_phys + 2 <= RING_SIZE {
                        // ONE u32 store = two u16 lanes [b0, b1].
                        unsafe {
                            (ring_ptr.add(dst_phys) as *mut u32)
                                .write_unaligned((b0 | (b1 << 16)) as u32);
                        }
                        pos += 2;
                        emitted += 2;
                        continue 'outer;
                    } else if sym_count == 3 && dst_phys + 4 <= RING_SIZE {
                        // ONE u64 store = three u16 lanes [b0, b1, b2] (+ a zero
                        // overshoot lane at dst+3, overwritten before drain).
                        let b2 = ((sym >> 16) & 0xFF) as u64;
                        unsafe {
                            (ring_ptr.add(dst_phys) as *mut u64)
                                .write_unaligned(b0 | (b1 << 16) | (b2 << 32));
                        }
                        pos += 3;
                        emitted += 3;
                        continue 'outer;
                    }
                    // else: near a ring wrap → fall through to the scalar loop.
                }
            }

            // Faithful multi-symbol unpack — EXACT mirror of
            // `read_internal_compressed_specialized::<false>` (deflate.hpp:1612).
            // The `code <= 255 || sym_count > 1` test forces literal
            // interpretation for every symbol except the last; the last is then
            // a literal, EOB, or length code.
            loop {
                let code = (sym & 0xFFFF) as u16;
                if code <= 255 || sym_count > 1 {
                    unsafe {
                        ring_ptr.add(pos % RING_SIZE).write(code & 0xFF);
                    }
                    pos += 1;
                    emitted += 1;
                    sym_count -= 1;
                    if sym_count == 0 {
                        break;
                    }
                    sym >>= 8;
                    continue;
                }
                if code == END_OF_BLOCK_SYMBOL {
                    self.at_end_of_block = true;
                    commit!(Ok(emitted));
                }
                if (code as u32) > MAX_LIT_LEN_SYM {
                    commit!(Err(BlockError::InvalidHuffmanCode));
                }
                let length = (code as usize).wrapping_sub(254);
                if length == 0 {
                    break;
                }
                let dsym = match self.dist_hc.decode(bits) {
                    Some(d) => d,
                    None => commit!(Err(BlockError::InvalidHuffmanCode)),
                };
                if dsym as usize >= DISTANCE_BASE.len() {
                    commit!(Err(BlockError::InvalidHuffmanCode));
                }
                let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                let distance = if extra > 0 {
                    if bits.available() < extra {
                        bits.refill();
                        if bits.available() < extra {
                            commit!(Err(BlockError::EndOfFile));
                        }
                    }
                    let mask = (1u64 << extra) - 1;
                    let v = (bits.peek() & mask) as usize;
                    bits.consume(extra);
                    DISTANCE_BASE[dsym as usize] as usize + v
                } else {
                    DISTANCE_BASE[dsym as usize] as usize
                };
                if distance == 0 || distance > MAX_WINDOW_SIZE {
                    commit!(Err(BlockError::ExceededWindowRange));
                }
                // Clean-mode distance check (vendor deflate.hpp:1652-1655).
                if distance > self.decoded_bytes + emitted {
                    commit!(Err(BlockError::ExceededWindowRange));
                }
                unsafe {
                    emit_backref_ring_clean(ring_ptr, &mut pos, distance, length, use_avx2);
                }
                self.record_backreference_for_sparsity(distance, length, emitted);
                emitted += length;
                break;
            }
        }
        self.ring.pos = pos;
        self.decoded_bytes += emitted;
        Ok(emitted)
    }

    /// Canonical-Huffman decode path. On non-x86_64 / no-ISA-L builds
    /// this is the only decoder. On x86_64+ISA-L it is invoked from
    /// `read_internal_compressed` for FIXED Huffman blocks (vendor uses
    /// `HuffmanCodingReversedBitsCached` there — see the dispatch
    /// comment in the ISA-L branch) — dynamic blocks take the
    /// `IsalLitLenCode` multi-symbol fast path.
    ///
    /// Decodes via the ported `HuffmanCodingSymbolsPerLength` (vendor
    /// `vendor/.../huffman/HuffmanCodingSymbolsPerLength.hpp:97-124` —
    /// `decode<BitReader>`), driven by the `LsbBitReader` adapter that
    /// `huffman_base.rs` provides for `Bits`.
    /// Public entry — runtime-dispatches to const-generic specialization
    /// (see ISA-L sibling).
    #[allow(dead_code)] // vendor parity or unit-test surface
    pub fn read_internal_compressed_canonical(
        &mut self,
        bits: &mut Bits,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        if self.ring.is_marker() {
            self.read_internal_compressed_canonical_specialized::<true>(bits, n_max_to_decode)
        } else {
            self.read_internal_compressed_canonical_specialized::<false>(bits, n_max_to_decode)
        }
    }

    fn read_internal_compressed_canonical_specialized<const CONTAINS_MARKERS: bool>(
        &mut self,
        bits: &mut Bits,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        use crate::decompress::parallel::huffman_reversed_bits_cached::HuffmanCodingReversedBitsCached;

        let (litlen_lens, dist_lens) = match self.compression_type {
            CompressionType::DynamicHuffman => {
                let lit = self.literal_cl[..self.literal_code_count].to_vec();
                let dist = self.literal_cl
                    [self.literal_code_count..self.literal_code_count + self.distance_code_count]
                    .to_vec();
                (lit, dist)
            }
            CompressionType::FixedHuffman => fixed_huffman_code_lengths(),
            _ => return Err(BlockError::InvalidCompression),
        };

        const LITLEN_CAP: usize = MAX_LITERAL_OR_LENGTH_SYMBOLS + 2;
        // Vendor parity (deflate.hpp:336): distance Huffman uses the cached
        // variant, not the canonical bit-by-bit decoder. perf record on
        // PGO build showed 8.85% of cycles in the canonical fallback via
        // get_distance_dynamic — swapping cuts that.
        let mut dist_hc: HuffmanCodingReversedBitsCached<MAX_DISTANCE_SYMBOL_COUNT> =
            HuffmanCodingReversedBitsCached::new();
        let err = dist_hc.initialize_from_lengths(&dist_lens, false);
        if err != super::error::Error::None {
            return Err(BlockError::InvalidCodeLengths);
        }

        macro_rules! run_canonical_loop {
            ($decode_litlen:expr) => {{
                // Cap n_max_to_decode to ring capacity minus one max back-ref
                // length — see ISA-L path for full rationale (mirror of vendor
                // gzip/deflate.hpp:1602).
                const MAX_RUN_LENGTH: usize = 258;
                let n_max_to_decode = n_max_to_decode.min(RING_SIZE - MAX_RUN_LENGTH);

                let ring_ptr = self.ring.window16.as_mut_ptr();
                let mut pos = self.ring.pos;
                // Step C SIMD-bootstrap experiment (per Opus advisor 2026-05-25):
                // maintain `phys` as a u16 physical-ring-index alongside the
                // monotonic logical `pos`. Per-literal write becomes a pure
                // pointer bump (no `% RING_SIZE` AND on the write-address
                // dependency chain). Back-refs resync via `pos & (RING_SIZE-1)`
                // after emit_backref_ring runs.
                let mut phys: u16 = (pos & (RING_SIZE - 1)) as u16;
                let mut emitted: usize = 0;
                let mut distance_marker = self.ring.distance_to_last_marker;

                macro_rules! commit {
                    ($result:expr) => {{
                        self.ring.pos = pos;
                        self.decoded_bytes += emitted;
                        self.ring.distance_to_last_marker = distance_marker;
                        return $result;
                    }};
                }

                // Causal-perturbation slow-injection knob — see the ISA-L
                // sibling for the two-knob contract. Snapshot once before the
                // loop; each knob const-folds to 0 on the other specialization.
                let slow_spin: u64 = if CONTAINS_MARKERS {
                    super::slow_knob::marker_spin_iters()
                } else {
                    super::slow_knob::spin_iters()
                };
                let slow_yield: bool = slow_spin != 0 && super::slow_knob::yield_kind();

                while emitted < n_max_to_decode {
                    // One injection per decode event. No-op when slow_spin == 0.
                    super::slow_knob::inject(slow_spin, slow_yield);
                    let sym = match $decode_litlen(bits) {
                        Some(s) => s,
                        None => commit!(Err(BlockError::InvalidHuffmanCode)),
                    };

                    if sym < 256 {
                        unsafe {
                            ring_ptr.add(phys as usize).write(sym);
                        }
                        phys = phys.wrapping_add(1);
                        pos += 1;
                        emitted += 1;
                        if CONTAINS_MARKERS {
                            distance_marker += 1;
                        }
                        continue;
                    }
                    if sym == END_OF_BLOCK_SYMBOL {
                        self.at_end_of_block = true;
                        commit!(Ok(emitted));
                    }
                    if sym > 285 {
                        commit!(Err(BlockError::InvalidHuffmanCode));
                    }
                    let lidx = (sym - 257) as usize;
                    let length = match read_length_extra(bits, lidx) {
                        Ok(l) => l,
                        Err(e) => commit!(Err(e)),
                    };
                    let dsym = match dist_hc.decode(bits) {
                        Some(d) => d,
                        None => commit!(Err(BlockError::InvalidHuffmanCode)),
                    };
                    if (dsym as usize) >= DISTANCE_BASE.len() {
                        commit!(Err(BlockError::InvalidHuffmanCode));
                    }
                    let distance = match read_distance_extra(bits, dsym as usize) {
                        Ok(d) => d,
                        Err(e) => commit!(Err(e)),
                    };
                    if distance == 0 || distance > MAX_WINDOW_SIZE {
                        commit!(Err(BlockError::ExceededWindowRange));
                    }
                    if !CONTAINS_MARKERS && distance > self.decoded_bytes + emitted {
                        commit!(Err(BlockError::ExceededWindowRange));
                    }
                    unsafe {
                        emit_backref_ring::<CONTAINS_MARKERS>(
                            ring_ptr,
                            &mut pos,
                            distance,
                            length,
                            &mut distance_marker,
                        );
                    }
                    phys = (pos & (RING_SIZE - 1)) as u16;
                    self.record_backreference_for_sparsity(distance, length, emitted);
                    emitted += length;
                }
                self.ring.pos = pos;
                self.decoded_bytes += emitted;
                self.ring.distance_to_last_marker = distance_marker;
                Ok(emitted)
            }};
        }

        match self.compression_type {
            CompressionType::FixedHuffman => {
                let mut litlen_hc = HuffmanCodingReversedBitsCached::<LITLEN_CAP>::new();
                let err = litlen_hc.initialize_from_lengths(&litlen_lens, true);
                if err != super::error::Error::None {
                    return Err(BlockError::InvalidCodeLengths);
                }
                run_canonical_loop!(|bits: &mut Bits| litlen_hc.decode(bits))
            }
            CompressionType::DynamicHuffman => {
                #[cfg(any(
                    all(
                        feature = "isal-compression",
                        not(feature = "pure-rust-inflate"),
                        target_arch = "x86_64"
                    ),
                    pure_inflate_decode
                ))]
                {
                    Err(BlockError::InvalidCompression)
                }

                #[cfg(not(any(
                    all(
                        feature = "isal-compression",
                        not(feature = "pure-rust-inflate"),
                        target_arch = "x86_64"
                    ),
                    pure_inflate_decode
                )))]
                {
                    use crate::decompress::inflate::libdeflate_entry::{
                        DistTable, LitLenTable, HUFFDEC_END_OF_BLOCK, HUFFDEC_EXCEPTIONAL,
                    };

                    // libdeflate canonical tables when ISA-L LUT decode is off.
                    let litlen = match LitLenTable::build(&litlen_lens) {
                        Some(t) => t,
                        None => return Err(BlockError::InvalidCodeLengths),
                    };
                    let dist = match DistTable::build(&dist_lens) {
                        Some(t) => t,
                        None => return Err(BlockError::InvalidCodeLengths),
                    };

                    macro_rules! run_multi_cached_loop {
                        () => {{
                            const MAX_RUN_LENGTH: usize = 258;
                            // FASTLOOP_INPUT_TAIL (mirror of resumable.rs:1110):
                            // a conservative 32-byte tail kept off-limits to the
                            // branchless refill so its 8-byte unaligned load never
                            // reads past `bits.data.len()` — even on the LAST /
                            // BFINAL block of the WHOLE input. A single FASTLOOP
                            // iteration issues at most one branchless refill, which
                            // advances `bits.pos` by ≤7, so the load touches at most
                            // `(in_fastloop_end - 1) + 7 + 8 = data.len() - 18`.
                            // 32 (vs the strictly-needed 8) keeps a comfortable
                            // margin and matches the reference decoder. This is the
                            // #1 correctness risk per the design note — the SAFE
                            // tier (existing branchy decode/refill) handles the
                            // final 32 bytes + EOB / sub-byte-EOF padding.
                            const FASTLOOP_INPUT_TAIL: usize = 32;
                            // Refill threshold mirrors resumable.rs:959. After a
                            // refill bitsleft ∈ [56, 63]; a single packet's worst
                            // case is litlen codeword (≤12) + length extras (≤5) +
                            // dist codeword (≤12 main + ≤6 sub) + dist extras
                            // (≤13) = ≤48 bits, so refilling whenever bitsleft < 48
                            // guarantees every consume in a packet reads valid bits.
                            const REFILL_THRESHOLD: u8 = 48;
                            let n_max_to_decode = n_max_to_decode.min(RING_SIZE - MAX_RUN_LENGTH);

                            // Conservative FASTLOOP input bound: never let the
                            // branchless 8-byte load reach past `data.len()`.
                            // There is no separate output-margin bound — the u16
                            // ring is pre-sized to RING_SIZE and `emitted` is
                            // capped at `n_max_to_decode` (= RING_SIZE -
                            // MAX_RUN_LENGTH), so a single packet's ≤258-byte
                            // back-ref or ≤3 literals can never overrun it; that
                            // check stays on the SAME `emitted < n_max_to_decode`
                            // tier guard both loops share.
                            let in_fastloop_end: usize =
                                bits.data.len().saturating_sub(FASTLOOP_INPUT_TAIL);

                            let ring_ptr = self.ring.window16.as_mut_ptr();
                            let mut pos = self.ring.pos;
                            // Step C SIMD-bootstrap experiment: hoist ring-modulo
                            // out of the per-literal hot path. `phys` is the
                            // physical u16 ring index, kept in sync with pos.
                            let mut phys: u16 = (pos & (RING_SIZE - 1)) as u16;
                            let mut emitted: usize = 0;
                            let mut distance_marker = self.ring.distance_to_last_marker;

                            macro_rules! commit {
                                ($result:expr) => {{
                                    self.ring.pos = pos;
                                    self.decoded_bytes += emitted;
                                    self.ring.distance_to_last_marker = distance_marker;
                                    return $result;
                                }};
                            }

                            // Emit a single decoded literal byte into the ring,
                            // advancing the phys/pos/emitted/distance_marker
                            // bookkeeping (mirror of the prior path's per-literal
                            // store; `distance_marker += 1` is the clean-byte run
                            // extension, dead-stripped when !CONTAINS_MARKERS).
                            macro_rules! emit_literal {
                                ($value:expr) => {{
                                    unsafe {
                                        ring_ptr.add(phys as usize).write($value as u16);
                                    }
                                    phys = phys.wrapping_add(1);
                                    pos += 1;
                                    emitted += 1;
                                    if CONTAINS_MARKERS {
                                        distance_marker += 1;
                                    }
                                }};
                            }

                            // Branchless refill for the FASTLOOP body ONLY (mirror
                            // of resumable.rs:1052 `refill_fast!`). Identical
                            // arithmetic to `Bits::refill`'s fast arm but WITHOUT
                            // the per-refill `pos + 8 <= data.len()` bounds branch
                            // — the FASTLOOP guard `bits.pos < in_fastloop_end`
                            // already proves the 8-byte load is in bounds. The
                            // `bits_u8 > 64` underflow guard is RETAINED so this is
                            // byte-for-byte equivalent to `Bits::refill` on every
                            // input the FASTLOOP admits; the ONLY elided work is the
                            // unpredictable bounds branch.
                            macro_rules! refill_fast {
                                () => {{
                                    let mut bits_u8 = bits.bitsleft as u8;
                                    if bits_u8 > 64 {
                                        bits.bitbuf = 0;
                                        bits_u8 = 0;
                                    }
                                    let word = unsafe {
                                        (bits.data.as_ptr().add(bits.pos) as *const u64)
                                            .read_unaligned()
                                    };
                                    let word = u64::from_le(word);
                                    bits.bitbuf |= word << bits_u8;
                                    bits.pos += (7 - ((bits_u8 >> 3) & 7)) as usize;
                                    bits.bitsleft = (bits_u8 as u32) | 56;
                                }};
                            }

                            // Bounds-checked refill (mirror of resumable.rs:1007
                            // `refill_local!`). Delegates to `Bits::refill`, which
                            // is the SAME branchless-when-room / byte-by-byte-at-EOF
                            // arithmetic with the underflow guard.
                            macro_rules! refill_local {
                                () => {{
                                    bits.refill();
                                }};
                            }

                            // PRELOAD the first entry before entering the loop
                            // (mirror of resumable.rs:1091-1094). Declared BEFORE
                            // `decode_one_symbol!` so the macro's `entry` refers to
                            // this loop-persistent local. `Bits` is refilled on
                            // construction, but a partially-consumed reader may be
                            // below threshold; refill defensively so the first
                            // lookup sees valid bits.
                            if (bits.bitsleft as u8) < REFILL_THRESHOLD {
                                refill_local!();
                            }
                            let mut entry = litlen.lookup(bits.bitbuf);

                            // Per-packet symbol-decode body, shared verbatim by the
                            // FASTLOOP and SAFE tiers so output is bit-identical
                            // (mirror of resumable.rs:1125-1361
                            // `decode_one_symbol!`). `$refill` selects the refill
                            // macro: FASTLOOP passes `refill_fast` (branchless,
                            // bounds elided — its guard makes it sound), SAFE passes
                            // `refill_local` (bounds-checked, valid to EOF). The
                            // decode arithmetic is otherwise identical, so the two
                            // tiers produce bit-identical output and hand off the
                            // exact same `bits` state. `entry` is the
                            // loop-persistent preloaded `LitLenEntry` corresponding
                            // to the current `bits.bitbuf`. On EOB the macro
                            // `commit!`s and returns; on a literal/length it carries
                            // the next iteration's `entry` forward (the speculative
                            // lookup that avoids a wasted re-lookup).
                            macro_rules! decode_one_symbol {
                                ($refill:ident) => {{
                                    let mut saved_bitbuf = bits.bitbuf;
                                    let mut raw = entry.raw();

                                    bits.bitbuf >>= raw as u8;
                                    bits.bitsleft = bits.bitsleft.wrapping_sub(raw & 0x1F);

                                    // LITERAL FAST PATH (bit 31).
                                    if (raw as i32) < 0 {
                                        emit_literal!(entry.literal_value());

                                        // Multi-literal chain (2-extra cap, mirror
                                        // of resumable.rs:1215-1272). Speculatively
                                        // look up the next entry: if literal, emit
                                        // inline (chain up to 3 total); whether or
                                        // not literals chain, carry the final
                                        // lookup forward as the next iter's `entry`.
                                        if (bits.bitsleft as u8) >= REFILL_THRESHOLD {
                                            let e1 = litlen.lookup(bits.bitbuf);
                                            let r1 = e1.raw();
                                            if (r1 as i32) < 0 {
                                                // 2nd literal
                                                bits.bitbuf >>= r1 as u8;
                                                bits.bitsleft =
                                                    bits.bitsleft.wrapping_sub(r1 & 0x1F);
                                                emit_literal!(e1.literal_value());

                                                if (bits.bitsleft as u8) >= 24 {
                                                    let e2 = litlen.lookup(bits.bitbuf);
                                                    let r2 = e2.raw();
                                                    if (r2 as i32) < 0 {
                                                        // 3rd literal (the last)
                                                        bits.bitbuf >>= r2 as u8;
                                                        bits.bitsleft =
                                                            bits.bitsleft.wrapping_sub(r2 & 0x1F);
                                                        emit_literal!(e2.literal_value());
                                                        if (bits.bitsleft as u8) < REFILL_THRESHOLD
                                                        {
                                                            $refill!();
                                                        }
                                                        entry = litlen.lookup(bits.bitbuf);
                                                        continue;
                                                    }
                                                    // 3rd was non-literal → carry e2.
                                                    if (bits.bitsleft as u8) < REFILL_THRESHOLD {
                                                        $refill!();
                                                    }
                                                    entry = e2;
                                                    continue;
                                                }
                                                // bitsleft too low for 3rd lookup.
                                                if (bits.bitsleft as u8) < REFILL_THRESHOLD {
                                                    $refill!();
                                                }
                                                entry = litlen.lookup(bits.bitbuf);
                                                continue;
                                            }
                                            // 2nd was non-literal → carry e1.
                                            if (bits.bitsleft as u8) < REFILL_THRESHOLD {
                                                $refill!();
                                            }
                                            entry = e1;
                                            continue;
                                        }

                                        // bitsleft too low even for 2nd lookup.
                                        if (bits.bitsleft as u8) < REFILL_THRESHOLD {
                                            $refill!();
                                        }
                                        entry = litlen.lookup(bits.bitbuf);
                                        continue;
                                    }

                                    // EXCEPTIONAL PATH (bit 15: subtable_ptr OR
                                    // end_of_block). Mirror of resumable.rs:1277.
                                    if (raw & HUFFDEC_EXCEPTIONAL) != 0 {
                                        if (raw & HUFFDEC_END_OF_BLOCK) != 0 {
                                            self.at_end_of_block = true;
                                            commit!(Ok(emitted));
                                        }
                                        // SUBTABLE
                                        entry = litlen.lookup_subtable_direct(entry, bits.bitbuf);
                                        saved_bitbuf = bits.bitbuf;
                                        raw = entry.raw();
                                        bits.bitbuf >>= raw as u8;
                                        bits.bitsleft = bits.bitsleft.wrapping_sub(raw & 0x1F);

                                        if (raw as i32) < 0 {
                                            emit_literal!(entry.literal_value());
                                            if (bits.bitsleft as u8) < REFILL_THRESHOLD {
                                                $refill!();
                                            }
                                            entry = litlen.lookup(bits.bitbuf);
                                            continue;
                                        }
                                        if (raw & HUFFDEC_END_OF_BLOCK) != 0 {
                                            self.at_end_of_block = true;
                                            commit!(Ok(emitted));
                                        }
                                        // Fall through to LENGTH path.
                                    }

                                    // LENGTH+DISTANCE path (mirror of
                                    // resumable.rs:1309-1354).
                                    let length = entry.decode_length(saved_bitbuf) as usize;
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                                        let idx = pos.wrapping_sub(32) & (RING_SIZE - 1);
                                        _mm_prefetch(ring_ptr.add(idx) as *const i8, _MM_HINT_T0);
                                    }
                                    let dist_saved = bits.bitbuf;
                                    let mut dist_entry = dist.lookup(dist_saved);
                                    if dist_entry.is_subtable_ptr() {
                                        bits.bitbuf >>= DistTable::TABLE_BITS;
                                        bits.bitsleft = bits
                                            .bitsleft
                                            .wrapping_sub(DistTable::TABLE_BITS as u32);
                                        dist_entry =
                                            dist.lookup_subtable_direct(dist_entry, bits.bitbuf);
                                    }
                                    let dist_extra_saved = bits.bitbuf;
                                    let dist_raw = dist_entry.raw();
                                    bits.bitbuf >>= dist_raw as u8;
                                    bits.bitsleft = bits.bitsleft.wrapping_sub(dist_raw & 0x1F);
                                    let distance =
                                        dist_entry.decode_distance(dist_extra_saved) as usize;

                                    if distance == 0 || distance > MAX_WINDOW_SIZE {
                                        commit!(Err(BlockError::ExceededWindowRange));
                                    }
                                    if !CONTAINS_MARKERS && distance > self.decoded_bytes + emitted
                                    {
                                        commit!(Err(BlockError::ExceededWindowRange));
                                    }
                                    unsafe {
                                        emit_backref_ring::<CONTAINS_MARKERS>(
                                            ring_ptr,
                                            &mut pos,
                                            distance,
                                            length,
                                            &mut distance_marker,
                                        );
                                    }
                                    phys = (pos & (RING_SIZE - 1)) as u16;
                                    self.record_backreference_for_sparsity(
                                        distance, length, emitted,
                                    );
                                    emitted += length;

                                    // Refill + preload next entry.
                                    if (bits.bitsleft as u8) < REFILL_THRESHOLD {
                                        $refill!();
                                    }
                                    entry = litlen.lookup(bits.bitbuf);
                                }};
                            }

                            // Two-tier decode (mirror of resumable.rs:1364-1417 /
                            // the prior ISA-L two-tier shape). FASTLOOP: while there
                            // is input headroom AND output budget, decode with the
                            // branchless refill — the litlen/dist lookups pay no
                            // bounds-checked refill. SAFE tier: one symbol at a time
                            // with the bounds-checked refill, which handles EOB,
                            // sub-byte-EOF padding, and the final FASTLOOP_INPUT_TAIL
                            // bytes correctly. EOB exits both tiers via `commit!`
                            // inside `decode_one_symbol!`. The carried `entry`
                            // crosses the FASTLOOP↔SAFE boundary unchanged, so the
                            // handoff is bit-identical.
                            while emitted < n_max_to_decode {
                                if bits.pos < in_fastloop_end {
                                    while bits.pos < in_fastloop_end && emitted < n_max_to_decode {
                                        decode_one_symbol!(refill_fast);
                                    }
                                } else {
                                    decode_one_symbol!(refill_local);
                                }
                            }
                            self.ring.pos = pos;
                            self.decoded_bytes += emitted;
                            self.ring.distance_to_last_marker = distance_marker;
                            Ok(emitted)
                        }};
                    }
                    run_multi_cached_loop!()
                }
            }
            _ => Err(BlockError::InvalidCompression),
        }
    }

    /// Canonical fallback when ISA-L LUT decode is not compiled in.
    #[cfg(not(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    )))]
    pub fn read_internal_compressed(
        &mut self,
        bits: &mut Bits,
        n_max_to_decode: usize,
    ) -> Result<usize, BlockError> {
        self.read_internal_compressed_canonical(bits, n_max_to_decode)
    }
}

// ── Length / distance extra-bits tables (RFC 1951 §3.2.5) ──────────────────

pub const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

pub const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

pub const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

pub const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

#[cfg(parallel_sm)]
fn read_length_extra(bits: &mut Bits, lidx: usize) -> Result<usize, BlockError> {
    let extra = LENGTH_EXTRA[lidx] as u32;
    if extra > 0 {
        ensure_bits(bits, extra)?;
    }
    let extra_val = if extra > 0 {
        // Mask via ported `n_lowest_bits_set` (vendor
        // BitManipulation.hpp:60-73) rather than an inline shift.
        let v = (bits.peek() & n_lowest_bits_set(extra as u8)) as u16;
        bits.consume(extra);
        v
    } else {
        0
    };
    Ok((LENGTH_BASE[lidx] + extra_val) as usize)
}

fn read_distance_extra(bits: &mut Bits, dsym: usize) -> Result<usize, BlockError> {
    let extra = DISTANCE_EXTRA[dsym] as u32;
    if extra > 0 {
        ensure_bits(bits, extra)?;
    }
    let extra_val = if extra > 0 {
        // Mask via ported `n_lowest_bits_set` (vendor
        // BitManipulation.hpp:60-73) rather than an inline shift.
        let v = (bits.peek() & n_lowest_bits_set(extra as u8)) as u32;
        bits.consume(extra);
        v
    } else {
        0
    };
    Ok(DISTANCE_BASE[dsym] as usize + extra_val as usize)
}

/// RFC 1951 §3.2.6 fixed Huffman — 288 lit/len symbols (286–287 participate
/// in Kraft construction). Distances: 32 codes × length 5.
#[cfg(parallel_sm)]
const FIXED_LIT_LEN_LENGTHS: [u8; MAX_LITERAL_OR_LENGTH_SYMBOLS + 2] = {
    let mut t = [0u8; MAX_LITERAL_OR_LENGTH_SYMBOLS + 2];
    let mut i = 0usize;
    while i < 144 {
        t[i] = 8;
        i += 1;
    }
    while i < 256 {
        t[i] = 9;
        i += 1;
    }
    while i < 280 {
        t[i] = 7;
        i += 1;
    }
    while i < t.len() {
        t[i] = 8;
        i += 1;
    }
    t
};
#[cfg(parallel_sm)]
const FIXED_DIST_LENGTHS: [u8; MAX_DISTANCE_SYMBOL_COUNT] = [5u8; MAX_DISTANCE_SYMBOL_COUNT];

/// Returns owned copies of the RFC 1951 fixed-Huffman tables.
#[cfg(parallel_sm)]
fn fixed_huffman_code_lengths() -> (Vec<u8>, Vec<u8>) {
    (FIXED_LIT_LEN_LENGTHS.to_vec(), FIXED_DIST_LENGTHS.to_vec())
}

/// Ring-buffer back-reference emit. Mirror of vendor's
/// `resolveBackreference` (vendor/.../gzip/deflate.hpp:1349-1410):
/// non-overlap is a single `memcpy`, RLE (distance=1) is a tight
/// repeat, general overlap is sequential element copy.
///
/// **No explicit marker emission.** The ring's upper half is
/// pre-initialized to marker values (see `init_marker_zone`), so a
/// back-ref reading from a slot in `[MAX_WINDOW_SIZE..RING_SIZE)`
/// retrieves the correct `MARKER_BASE + idx` value automatically.
/// One `memcpy` handles BOTH the cross-chunk marker portion AND
/// the in-chunk literal portion of a single back-ref — exactly
/// vendor's design.
///
/// When `CONTAINS_MARKERS` is true, the function additionally scans
/// the just-written `length` bytes backward to update
/// `*distance_marker` per vendor's pattern at
/// deflate.hpp:1379-1389: if a marker is found at offset `k` from
/// the end, `distance_marker` = k; otherwise `distance_marker += length`.
/// When `CONTAINS_MARKERS` is false (const generic), the scan is
/// dead-stripped by the compiler (mirror of vendor's
/// `if constexpr ( containsMarkerBytes )` at deflate.hpp:1379).
///
/// # Safety
/// Caller must ensure:
///   * `ring_ptr` is a valid `*mut u16` pointing at a fixed
///     `[u16; RING_SIZE]` allocation; valid for reads and writes at
///     every index in `[0, RING_SIZE)`.
///   * `*pos % RING_SIZE` is a valid physical slot.
///   * The slot at `(*pos - distance) % RING_SIZE` for `distance <=
///     MAX_WINDOW_SIZE` still holds the byte (or pre-init marker)
///     logically written at position `*pos - distance` (i.e. the
///     lookback range has not been overwritten by wrap — guaranteed
///     by per-call drain semantics in `Block::read` and the
///     `n_max_to_decode <= RING_SIZE - MAX_RUN_LENGTH` cap).
///   * No aliasing other `*mut` derived from the same ring exists.
///
/// On return `*pos` is incremented by exactly `length`, and (when
/// `CONTAINS_MARKERS`) `*distance_marker` reflects the post-copy
/// distance to the nearest marker in the ring.
#[inline]
pub(crate) unsafe fn emit_backref_ring<const CONTAINS_MARKERS: bool>(
    ring_ptr: *mut u16,
    pos: &mut usize,
    distance: usize,
    length: usize,
    distance_marker: &mut usize,
) {
    // Physical source / destination slots. Source range may straddle
    // the ring's physical end in the wrap case; we handle that
    // explicitly below.
    let src_phys = (*pos + RING_SIZE - distance) % RING_SIZE;
    let dst_phys = *pos % RING_SIZE;

    if distance >= length {
        // Non-overlap path. Vendor: `std::memcpy(&window[m_windowPosition],
        // &window[offset], length * 2)` at deflate.hpp:1376.
        //
        // MEASURED DISTRIBUTION (silesia, flate2 L6, window-absent path):
        // 99.94% of back-refs are non-overlap and ~98.6% have length < 16
        // (77% are length 4-7, mean ≈ 6.3 u16). The prior 8-u16 (16-byte)
        // chunk loop NEVER fired for those — its `i + 8 <= length` guard is
        // false for length < 8 — so every common match fell to the per-element
        // scalar tail: 4-7 dependent single-u16 stores on the critical path.
        // That tail, not wide copies, is the window-absent path's output cost.
        //
        // Mirror `copy_match_fast`'s `dist >= 8` (bytes) arm
        // (consume_first_decode.rs:479-503) for u16: copy in 8-byte (4-u16)
        // unaligned WORD writes, UNCONDITIONALLY rounding the run up to a
        // multiple of 4 u16. The overshoot (≤ 3 u16) is sound because:
        //   * `distance >= 4` u16 ⇒ the 8-byte word stride never aliases
        //     (src and dst are ≥ 8 bytes apart), so each word read sees the
        //     original bytes — exactly the `dist >= WORDBYTES` invariant
        //     copy_match_fast relies on.
        //   * The rounded copy stays inside the physical buffer (the
        //     `*_round_fits` guards below), and the overcopied tail lies
        //     AHEAD of the advanced `*pos`; no back-ref ever reads ahead of
        //     `*pos`, and subsequent literals/copies overwrite those slots,
        //     so the extra bytes are invisible. The marker backward-scan
        //     below reads only the `length` bytes behind the new `*pos`.
        // `distance < 4` non-overlap (only length-3/distance-3) is rare and
        // would alias the word stride, so it keeps the exact element copy.
        let rounded = (length + 3) & !3;
        let src_round_fits = src_phys + rounded <= RING_SIZE;
        let dst_round_fits = dst_phys + rounded <= RING_SIZE;
        if distance >= 4 && src_round_fits && dst_round_fits {
            // One 8-byte word covers length ≤ 4; a second covers 5–8; the
            // loop handles the rare long tail (≥ 9 u16).
            let s = ring_ptr.add(src_phys) as *const u8;
            let d = ring_ptr.add(dst_phys) as *mut u8;
            (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
            if length > 4 {
                if length <= 8 {
                    (d.add(8) as *mut u64)
                        .write_unaligned((s.add(8) as *const u64).read_unaligned());
                } else {
                    let mut s = s.add(8);
                    let mut d = d.add(8);
                    let dend = (ring_ptr.add(dst_phys + rounded)) as *mut u8;
                    while d < dend {
                        (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
                        s = s.add(8);
                        d = d.add(8);
                    }
                }
            }
        } else if src_phys + length <= RING_SIZE && dst_phys + length <= RING_SIZE {
            // Non-wrap but distance < 4 (would alias a word stride): exact
            // element copy.
            let src = ring_ptr.add(src_phys);
            let dst = ring_ptr.add(dst_phys);
            for i in 0..length {
                dst.add(i).write(*src.add(i));
            }
        } else {
            // Wrap-straddle non-overlap fallback (rare boundary case).
            for i in 0..length {
                let v = *ring_ptr.add((src_phys + i) % RING_SIZE);
                ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
            }
        }
        *pos += length;
    } else if distance == 1 {
        // RLE: repeat the last element. Vendor's clean-mode path
        // (`!containsMarkerBytes && nToCopyPerRepeat == 1`) at
        // deflate.hpp:1393-1398 uses `std::memset` — a SIMD broadcast.
        //   * dst fits without wrap → `slice::fill` (Rust's idiomatic
        //     memset, lowers to `vpbroadcastw` + aligned-store on AVX2).
        //   * wrap-straddle → element-by-element loop with `% RING_SIZE`.
        // The fill is now used in MARKER MODE TOO (was gated on
        // `!CONTAINS_MARKERS`): it writes the identical `v` `length` times
        // either way — byte-for-byte the same ring contents — and the shared
        // backward marker-scan below still recomputes `distance_marker` from
        // those bytes, so the counter is unchanged. This was the largest
        // single avoidable slowdown on the window-absent path (RLE runs are
        // common; the slow element loop was paid only because the chunk hadn't
        // armed a clean window yet). [2026-05-31, fulcrum head-to-head lever]
        let v = *ring_ptr.add((*pos + RING_SIZE - 1) % RING_SIZE);
        if dst_phys + length <= RING_SIZE {
            let dst = std::slice::from_raw_parts_mut(ring_ptr.add(dst_phys), length);
            dst.fill(v);
        } else {
            for i in 0..length {
                ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
            }
        }
        *pos += length;
    } else {
        // General overlap (1 < distance < length). Sequential
        // element copy — earlier writes feed later reads, matching
        // deflate's run-length semantics. Wrap-safe via the per-
        // element modulo.
        for i in 0..length {
            let v = *ring_ptr.add((src_phys + i) % RING_SIZE);
            ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
        }
        *pos += length;
    }

    // Counter recompute — vendor's pattern at deflate.hpp:1379-1389.
    // Compile-time-eliminated when CONTAINS_MARKERS = false (post-
    // switch clean path); the entire scan + counter update is dead
    // code the compiler strips.
    if CONTAINS_MARKERS {
        // Fast path: if the clean run already covers the back-ref distance
        // (`distance_marker >= distance`), the ENTIRE copy source lies inside
        // the clean zone (the last `distance_marker` bytes have no marker, and
        // a back-ref reads only the last `distance <= distance_marker` bytes —
        // or, for the overlap/RLE case, repeats them), so every copied byte is
        // clean and the run simply extends. This skips the O(length) backward
        // scan that is the marker-mode path's main overhead over the clean
        // (CONTAINS_MARKERS=false) path — and it holds for the common case once
        // a chunk has decoded a window's worth of clean output. Byte-identical
        // to the scan (which would also find no marker and do `+= length`).
        // [2026-05-31, fulcrum-vs lever: window-absent decode is ~1.7x rg]
        if *distance_marker >= distance {
            *distance_marker += length;
            return;
        }
        // Scan backward through the just-written `length` u16 slots.
        const MARKER_U16: u16 = MAX_WINDOW_SIZE as u16;
        if dst_phys + length <= RING_SIZE {
            let slice = std::slice::from_raw_parts(ring_ptr.add(dst_phys), length);
            for (k, &v) in slice.iter().rev().enumerate() {
                if v >= MARKER_U16 {
                    *distance_marker = k;
                    return;
                }
            }
        } else {
            let mut k = 0usize;
            while k < length {
                let v = *ring_ptr.add((*pos - 1 - k) % RING_SIZE);
                if v >= MARKER_U16 {
                    *distance_marker = k;
                    return;
                }
                k += 1;
            }
        }
        *distance_marker += length;
    }
}

/// CONTIGUOUS (non-wrapping) clean-mode back-ref copy — the copy-free-to-final
/// (Stage 1) sibling of [`emit_backref_ring_u8`]. The output is a single
/// CONTIGUOUS buffer (`chunk.data`'s reserved tail with the 32 KiB predecessor
/// window prepended at `[0, window_len)`), so positions NEVER wrap: `*pos` is a
/// plain index into `base`, `src = base + (*pos - distance)`, `dst = base +
/// *pos`. This is the vendor `setInitialWindow` model — clean bulk decoded
/// straight into one contiguous buffer (DecodedData.hpp:278-289), window
/// installed as a dictionary prefix (deflate.hpp:1778). Eliminating the ring's
/// `% U8_RING_SIZE` modulus collapses `emit_backref_ring_u8`'s three wrap-aware
/// arms (src/dst round-fit + wrap-straddle fallback) to: non-overlap word copy,
/// `distance == 1` RLE fill, general overlap sequential — fewer branches, half
/// the index arithmetic (the secondary win on top of the dropped drain memcpy).
///
/// The caller GUARANTEES, before each call, that `*pos + length + 7 <= cap`
/// (the per-`read()` cap to contiguous spare, the Engine-C contract: never grow
/// mid-block, only between calls). So the ≤7-byte word-copy overshoot is always
/// in-bounds. Back-refs of `distance <= 32768` at the first clean byte
/// (`*pos == window_len`) resolve into the prepended window prefix; the caller
/// enforces `distance <= *pos` (== vendor's `distance <= decodedBytes`,
/// deflate.hpp:1652-1655) so `*pos - distance` never underflows.
///
/// SAFETY: `base` must be valid for `[0, cap)`; `*pos < cap`; `distance <= *pos`;
/// `*pos + ((length + 7) & !7) <= cap`. `*pos` is the logical write index.
#[cfg(any(
    all(
        feature = "isal-compression",
        not(feature = "pure-rust-inflate"),
        target_arch = "x86_64"
    ),
    pure_inflate_decode
))]
#[inline]
pub(crate) unsafe fn emit_backref_contig(
    base: *mut u8,
    pos: &mut usize,
    distance: usize,
    length: usize,
) {
    let dst = base.add(*pos);
    let src = base.add(*pos - distance);
    if distance >= length {
        // Non-overlap. 8-byte unaligned word copy; run rounded up to 8. The
        // ≤7-byte overshoot is ahead of the advanced `*pos`, never sourced by a
        // later back-ref, and overwritten — invisible. Caller guarantees the
        // rounded run fits `cap`. The `distance >= 8` guard keeps the 8-byte
        // stride non-aliasing (same invariant as the ring path).
        if distance >= 8 {
            let rounded = (length + 7) & !7;
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            if length > 8 {
                let mut s = src.add(8);
                let mut d = dst.add(8);
                let dend = dst.add(rounded);
                while d < dend {
                    (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
                    s = s.add(8);
                    d = d.add(8);
                }
            }
        } else {
            // distance < 8 (would alias the word stride): exact byte copy.
            for i in 0..length {
                dst.add(i).write(*src.add(i));
            }
        }
        *pos += length;
    } else if distance == 1 {
        // RLE memset (deflate.hpp:1393-1398).
        let v = *src;
        std::slice::from_raw_parts_mut(dst, length).fill(v);
        *pos += length;
    } else {
        // General overlap (1 < distance < length): sequential per-byte copy.
        for i in 0..length {
            dst.add(i).write(*src.add(i));
        }
        *pos += length;
    }
}

/// PRODUCTION post-flip clean-mode back-ref copy — **u8-direct** into the u8
/// view of `output_ring` (vendor `getWindow()` reinterpret, deflate.hpp:893).
/// Faithful port of `readInternal<false>`'s back-ref (deflate.hpp:1367-1399)
/// with `sizeof(window.front()) == 1` (deflate.hpp:1376): the memcpy is
/// `length` BYTES, not `length * 2`. No marker-counter maintenance and no
/// backward marker scan — `containsMarkerBytes == false` const-folds them out
/// in vendor (deflate.hpp:1367) and they are simply absent here.
///
/// Mirrors `emit_backref_ring::<false>`'s three arms at u8 width:
///   * non-overlap (`distance >= length`): 8-BYTE unaligned word copy, run
///     rounded up to a multiple of 8 bytes. The `distance >= 8` (BYTES) guard
///     keeps the 8-byte stride non-aliasing (same invariant as the u16 path's
///     `distance >= 4` u16 = 8 bytes). The ≤7-byte overshoot lies ahead of the
///     advanced `*pos`, is never sourced by a later back-ref, and gets
///     overwritten — invisible. Bounds-checked against `U8_RING_SIZE`.
///   * RLE (`distance == 1`): `slice::fill` = vendor `memset` (deflate.hpp:1395).
///   * general overlap (`1 < distance < length`): sequential per-byte copy.
///
/// SAFETY: `ring8` must be valid for all physical indices `[0, U8_RING_SIZE)`
/// (it is `output_ring.as_mut_ptr() as *mut u8` — the 128 KB allocation is
/// 131072 valid bytes, u8 alignment 1). `*pos` is the logical u8 write position.
#[cfg(any(
    all(
        feature = "isal-compression",
        not(feature = "pure-rust-inflate"),
        target_arch = "x86_64"
    ),
    pure_inflate_decode
))]
#[inline]
pub(crate) unsafe fn emit_backref_ring_u8(
    ring8: *mut u8,
    pos: &mut usize,
    distance: usize,
    length: usize,
) {
    let src_phys = (*pos + U8_RING_SIZE - distance) % U8_RING_SIZE;
    let dst_phys = *pos % U8_RING_SIZE;

    if distance >= length {
        // Non-overlap. Vendor: `memcpy(&window[wp], &window[off], length)`
        // (deflate.hpp:1376, sizeof==1). Word = 8 u8; round run up to 8.
        let rounded = (length + 7) & !7;
        let src_round_fits = src_phys + rounded <= U8_RING_SIZE;
        let dst_round_fits = dst_phys + rounded <= U8_RING_SIZE;
        if distance >= 8 && src_round_fits && dst_round_fits {
            let s = ring8.add(src_phys);
            let d = ring8.add(dst_phys);
            (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
            if length > 8 {
                let mut s = s.add(8);
                let mut d = d.add(8);
                let dend = ring8.add(dst_phys + rounded);
                while d < dend {
                    (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
                    s = s.add(8);
                    d = d.add(8);
                }
            }
        } else if src_phys + length <= U8_RING_SIZE && dst_phys + length <= U8_RING_SIZE {
            // Non-wrap, distance < 8 (would alias the word stride): exact copy.
            let src = ring8.add(src_phys);
            let dst = ring8.add(dst_phys);
            for i in 0..length {
                dst.add(i).write(*src.add(i));
            }
        } else {
            // Wrap-straddle non-overlap fallback.
            for i in 0..length {
                let v = *ring8.add((src_phys + i) % U8_RING_SIZE);
                ring8.add((dst_phys + i) % U8_RING_SIZE).write(v);
            }
        }
        *pos += length;
    } else if distance == 1 {
        // RLE memset (deflate.hpp:1393-1398).
        let v = *ring8.add((*pos + U8_RING_SIZE - 1) % U8_RING_SIZE);
        if dst_phys + length <= U8_RING_SIZE {
            let dst = std::slice::from_raw_parts_mut(ring8.add(dst_phys), length);
            dst.fill(v);
        } else {
            for i in 0..length {
                ring8.add((dst_phys + i) % U8_RING_SIZE).write(v);
            }
        }
        *pos += length;
    } else {
        // General overlap (1 < distance < length): sequential, wrap-safe.
        for i in 0..length {
            let v = *ring8.add((src_phys + i) % U8_RING_SIZE);
            ring8.add((dst_phys + i) % U8_RING_SIZE).write(v);
        }
        *pos += length;
    }
}

/// BENCH-ONLY clean-mode back-ref copy with the E2 AVX2 32-byte wide path.
/// No marker-counter maintenance (clean path only). `use_avx2` is resolved by
/// the caller via `is_x86_feature_detected!("avx2")`; when false (e.g. Rosetta
/// x86-64-v2) the scalar word-copy fallback runs and is byte-identical to
/// `emit_backref_ring::<false>`'s non-overlap / RLE / overlap arms.
///
/// SAFETY: `ring_ptr` must be valid for all physical indices `[0, RING_SIZE)`;
/// `*pos` is the logical write position. Same contract as `emit_backref_ring`.
#[cfg(any(
    all(
        feature = "isal-compression",
        not(feature = "pure-rust-inflate"),
        target_arch = "x86_64"
    ),
    all(pure_inflate_decode, target_arch = "x86_64")
))]
#[inline]
unsafe fn emit_backref_ring_clean(
    ring_ptr: *mut u16,
    pos: &mut usize,
    distance: usize,
    length: usize,
    use_avx2: bool,
) {
    let src_phys = (*pos + RING_SIZE - distance) % RING_SIZE;
    let dst_phys = *pos % RING_SIZE;

    if distance >= length {
        // E2: AVX2 32-byte (16-u16) wide copy. Safe only when `distance >= 16`
        // u16 (32 bytes) — the 32-byte stride then never aliases the just-
        // written dest for the REAL output bytes (each real byte at src+i with
        // i < length*2 <= distance*2 is read before any write reaches it; the
        // rounded-up overshoot lies AHEAD of the advanced `*pos`, is overwritten
        // before drain, and is never read as a back-ref source) — and when the
        // rounded run fits the ring without wrapping.
        let rounded16 = (length + 15) & !15;
        let did_avx2 = if use_avx2
            && distance >= 16
            && src_phys + rounded16 <= RING_SIZE
            && dst_phys + rounded16 <= RING_SIZE
        {
            emit_avx2_copy_u16(
                ring_ptr.add(src_phys) as *const u8,
                ring_ptr.add(dst_phys) as *mut u8,
                rounded16,
            );
            true
        } else {
            false
        };

        if !did_avx2 {
            // Scalar word-copy fallback — byte-identical to `emit_backref_ring`
            // (see the SAFETY / overshoot reasoning there).
            let rounded = (length + 3) & !3;
            let src_round_fits = src_phys + rounded <= RING_SIZE;
            let dst_round_fits = dst_phys + rounded <= RING_SIZE;
            if distance >= 4 && src_round_fits && dst_round_fits {
                let s = ring_ptr.add(src_phys) as *const u8;
                let d = ring_ptr.add(dst_phys) as *mut u8;
                (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
                if length > 4 {
                    if length <= 8 {
                        (d.add(8) as *mut u64)
                            .write_unaligned((s.add(8) as *const u64).read_unaligned());
                    } else {
                        let mut s = s.add(8);
                        let mut d = d.add(8);
                        let dend = (ring_ptr.add(dst_phys + rounded)) as *mut u8;
                        while d < dend {
                            (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
                            s = s.add(8);
                            d = d.add(8);
                        }
                    }
                }
            } else if src_phys + length <= RING_SIZE && dst_phys + length <= RING_SIZE {
                let src = ring_ptr.add(src_phys);
                let dst = ring_ptr.add(dst_phys);
                for i in 0..length {
                    dst.add(i).write(*src.add(i));
                }
            } else {
                for i in 0..length {
                    let v = *ring_ptr.add((src_phys + i) % RING_SIZE);
                    ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
                }
            }
        }
        *pos += length;
    } else if distance == 1 {
        let v = *ring_ptr.add((*pos + RING_SIZE - 1) % RING_SIZE);
        if dst_phys + length <= RING_SIZE {
            let dst = std::slice::from_raw_parts_mut(ring_ptr.add(dst_phys), length);
            dst.fill(v);
        } else {
            for i in 0..length {
                ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
            }
        }
        *pos += length;
    } else {
        for i in 0..length {
            let v = *ring_ptr.add((src_phys + i) % RING_SIZE);
            ring_ptr.add((dst_phys + i) % RING_SIZE).write(v);
        }
        *pos += length;
    }
}

/// E2 AVX2 copy helper: copies `n_u16` u16 slots (a multiple of 16, i.e. a
/// multiple of 32 bytes) forward in 32-byte `_mm256` load/store steps. Only
/// called when the caller has verified `is_x86_feature_detected!("avx2")` and
/// `distance >= 16` u16 (so the real output bytes never alias an earlier
/// write). Compiles under x86-64-v2 (the `target_feature` attribute makes the
/// AVX2 intrinsics available) but is never CALLED there.
#[cfg(any(
    all(
        feature = "isal-compression",
        not(feature = "pure-rust-inflate"),
        target_arch = "x86_64"
    ),
    all(pure_inflate_decode, target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
unsafe fn emit_avx2_copy_u16(mut src: *const u8, mut dst: *mut u8, n_u16: usize) {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};
    let end = dst.add(n_u16 * 2);
    while dst < end {
        let v = _mm256_loadu_si256(src as *const __m256i);
        _mm256_storeu_si256(dst as *mut __m256i, v);
        src = src.add(32);
        dst = dst.add(32);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn ensure_bits(bits: &mut Bits, n: u32) -> Result<(), BlockError> {
    if bits.available() < n {
        bits.refill();
    }
    if bits.available() < n {
        return Err(BlockError::EndOfFile);
    }
    Ok(())
}

/// Decode the literal + distance code lengths from the precode-encoded
/// stream. Mirror of rapidgzip's `readDistanceAndLiteralCodeLengths`
/// (deflate.hpp's literalCL reader, around lines 750-900).
///
/// Uses our existing `block_finder` Huffman decoder primitives via a
/// small ad-hoc canonical decoder — adequate for header parsing
/// (called once per dynamic block); the hot lit/len + distance decode
/// uses `IsalLitLenCode` + `IsalDistCode` from `isal_huffman.rs`.
fn read_literal_and_distance_code_lengths(
    bits: &mut Bits,
    precode_cl: &[u8; MAX_PRECODE_COUNT],
    total: usize,
) -> Result<Vec<u8>, BlockError> {
    // Build a simple canonical-Huffman decoder for the precode.
    // (Precode max length = 7 → a 128-entry decode table.)
    let table = build_precode_table(precode_cl)?;

    let mut out = vec![0u8; total];
    let mut i = 0;
    while i < total {
        ensure_bits(bits, 7)?;
        let peek7 = (bits.peek() & 0x7F) as usize;
        let entry = table[peek7];
        let length = (entry & 0xF) as u32;
        let symbol = entry >> 4;
        if length == 0 {
            return Err(BlockError::InvalidHuffmanCode);
        }
        bits.consume(length);
        match symbol {
            0..=15 => {
                out[i] = symbol as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(BlockError::InvalidCodeLengths);
                }
                ensure_bits(bits, 2)?;
                let repeat = (bits.peek() & 0b11) as usize + 3;
                bits.consume(2);
                let prev = out[i - 1];
                if i + repeat > total {
                    return Err(BlockError::InvalidCodeLengths);
                }
                for _ in 0..repeat {
                    out[i] = prev;
                    i += 1;
                }
            }
            17 => {
                ensure_bits(bits, 3)?;
                let repeat = (bits.peek() & 0b111) as usize + 3;
                bits.consume(3);
                if i + repeat > total {
                    return Err(BlockError::InvalidCodeLengths);
                }
                i += repeat;
            }
            18 => {
                ensure_bits(bits, 7)?;
                let repeat = (bits.peek() & 0x7F) as usize + 11;
                bits.consume(7);
                if i + repeat > total {
                    return Err(BlockError::InvalidCodeLengths);
                }
                i += repeat;
            }
            _ => return Err(BlockError::InvalidHuffmanCode),
        }
    }
    Ok(out)
}

/// Build a 128-entry decode table for the precode. Each entry packs
/// (symbol << 4) | length, with length = 0 indicating no valid code.
fn build_precode_table(precode_cl: &[u8; MAX_PRECODE_COUNT]) -> Result<[u16; 128], BlockError> {
    // Standard canonical-Huffman construction.
    let mut bl_count = [0u16; 16];
    for &len in precode_cl.iter() {
        if len > MAX_PRECODE_LENGTH {
            return Err(BlockError::InvalidCodeLengths);
        }
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }
    // Kraft check.
    let mut code = 0u32;
    let mut next_code = [0u32; 16];
    for bits in 1..=MAX_PRECODE_LENGTH as usize {
        code = (code + bl_count[bits - 1] as u32) << 1;
        if code > (1u32 << bits) {
            return Err(BlockError::InvalidCodeLengths);
        }
        next_code[bits] = code;
    }
    let mut table = [0u16; 128];
    for (sym, &len) in precode_cl.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let canonical_code = next_code[len as usize];
        next_code[len as usize] += 1;
        // Reverse bits (LSB-first stream).
        let reversed = reverse_bits(canonical_code as u16, len);
        let entry = ((sym as u16) << 4) | (len as u16);
        // Fill all table positions whose low `len` bits match `reversed`.
        let step = 1usize << len;
        let mut idx = reversed as usize;
        while idx < 128 {
            table[idx] = entry;
            idx += step;
        }
    }
    Ok(table)
}

fn reverse_bits(mut v: u16, n: u8) -> u16 {
    let mut r = 0u16;
    for _ in 0..n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// Test helper: decode the deflate stream in `data` from bit 0,
/// returning every block-start bit position observed. Each position is
/// a valid starting bit for resuming decode at the start of that
/// block's header.
///
/// Used by `backends::isal_decompress` invariant tests to oracle-check
/// ISA-L's `end_bit` values against an independently-derived set of
/// real block boundaries. Implemented by driving `Block::read_header`
/// + `Block::read` block-by-block until BFINAL.
#[cfg(test)]
pub fn record_block_starts(data: &[u8]) -> std::io::Result<Vec<usize>> {
    use crate::decompress::inflate::consume_first_decode::Bits;
    let mut bits = Bits::new(data);
    let mut output: Vec<u16> = Vec::with_capacity(data.len().saturating_mul(4));
    let mut block = Block::new();
    let mut starts = Vec::new();
    loop {
        // Snapshot bit position at the start of this block's header.
        let consumed_bytes = bits.pos;
        let bits_in_buf = bits.available();
        let abs_bit = consumed_bytes
            .saturating_mul(8)
            .saturating_sub(bits_in_buf as usize);
        starts.push(abs_bit);

        block
            .read_header(&mut bits, false)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}")))?;
        while !block.eob() {
            block
                .read(&mut bits, &mut output, usize::MAX)
                .map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}"))
                })?;
        }
        if block.is_last_block() {
            return Ok(starts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bits(data: &[u8]) -> Bits<'static> {
        // Leak a static copy so the Bits borrow is 'static within the test.
        let boxed: &'static [u8] = Box::leak(data.to_vec().into_boxed_slice());
        Bits::new(boxed)
    }

    #[test]
    fn read_header_stored_zero_length() {
        // Build a minimal stored block: BFINAL=0, BTYPE=00, byte-aligned padding=0,
        // LEN=0x0005, NLEN=0xFFFA.
        let mut bytes = vec![0b0000_0000u8]; // BFINAL=0, BTYPE=00, pad=00000
        bytes.extend_from_slice(&[0x05, 0x00, 0xFA, 0xFF]);
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        b.read_header(&mut bits, false).expect("header");
        assert_eq!(b.compression_type(), CompressionType::Uncompressed);
        assert!(!b.is_last_block());
        assert_eq!(b.uncompressed_size(), 5);
    }

    #[test]
    fn read_header_stored_bad_length_check() {
        let mut bytes = vec![0b0000_0000u8]; // stored
        bytes.extend_from_slice(&[0x05, 0x00, 0x00, 0x00]); // LEN != ~NLEN
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        assert_eq!(
            b.read_header(&mut bits, false),
            Err(BlockError::LengthChecksumMismatch)
        );
    }

    #[test]
    fn read_header_rejects_last_block_when_requested() {
        let bytes = vec![0b0000_0001u8]; // BFINAL=1
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        assert_eq!(
            b.read_header(&mut bits, true),
            Err(BlockError::UnexpectedLastBlock)
        );
    }

    #[test]
    fn read_header_fixed_huffman_no_followup() {
        let bytes = vec![0b0000_0010u8]; // BFINAL=0, BTYPE=01 (fixed)
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        b.read_header(&mut bits, false).expect("header");
        assert_eq!(b.compression_type(), CompressionType::FixedHuffman);
    }

    #[test]
    fn read_header_reserved_btype_rejected() {
        let bytes = vec![0b0000_0110u8]; // BFINAL=0, BTYPE=11
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        assert_eq!(
            b.read_header(&mut bits, false),
            Err(BlockError::InvalidCompression)
        );
    }

    #[test]
    fn reverse_bits_works() {
        assert_eq!(reverse_bits(0b1011, 4), 0b1101);
        assert_eq!(reverse_bits(0b1, 1), 0b1);
        assert_eq!(reverse_bits(0b0001, 4), 0b1000);
    }

    #[test]
    fn read_uncompressed_payload_emits_literal_bytes() {
        // BFINAL=0, BTYPE=00, pad=0, LEN=4, NLEN=0xFFFB, then "test".
        let mut bytes = vec![0b0000_0000u8];
        bytes.extend_from_slice(&[0x04, 0x00, 0xFB, 0xFF]);
        bytes.extend_from_slice(b"test");
        let mut bits = make_bits(&bytes);
        let mut b = Block::new();
        b.read_header(&mut bits, false).unwrap();
        let mut output: Vec<u16> = Vec::new();
        let n = b.read(&mut bits, &mut output, 1024).unwrap();
        assert_eq!(n, 4);
        assert_eq!(
            output,
            vec![b't' as u16, b'e' as u16, b's' as u16, b't' as u16]
        );
        assert!(b.eob());
    }

    #[test]
    fn read_round_trips_a_compressed_block() {
        // Both Fixed and Dynamic Huffman bodies should round-trip
        // byte-identical. flate2 picks the encoding based on payload
        // entropy; we test both via large + small payloads.
        for payload in &[
            b"a".repeat(2048),
            b"the quick brown fox jumps over the lazy dog. ".repeat(40),
        ] {
            use flate2::write::DeflateEncoder;
            use flate2::Compression;
            use std::io::Write;
            let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
            enc.write_all(payload).unwrap();
            let deflate_bytes = enc.finish().unwrap();
            let mut bits = make_bits(&deflate_bytes);
            let mut b = Block::new();
            b.read_header(&mut bits, false).unwrap();
            assert!(
                matches!(
                    b.compression_type(),
                    CompressionType::FixedHuffman | CompressionType::DynamicHuffman
                ),
                "expected compressed block, got {:?}",
                b.compression_type()
            );
            let mut output: Vec<u16> = Vec::new();
            let r = b.read(&mut bits, &mut output, payload.len() * 2);
            assert!(
                b.eob(),
                "decoder should reach end-of-block; read returned {:?}, output.len()={}, payload.len()={}",
                r,
                output.len(),
                payload.len(),
            );
            // For single-block flate2 output every back-ref is in-block,
            // so no markers expected.
            let resolved: Vec<u8> = output
                .iter()
                .map(|&v| {
                    assert!(v < 256, "in-block back-refs only; v={v:#x}");
                    v as u8
                })
                .collect();
            assert_eq!(resolved, *payload);
        }
    }

    // ── Copy-free-to-final (Stage 1) ring-vs-contiguous differential ──
    //
    // Decode the SAME window-seeded clean (`<false>`) DEFLATE body two ways and
    // assert byte-equal output:
    //   * RING path: `set_initial_window` + `read()` into a `Vec<u16>` (the
    //     production FOLD clean engine — back-refs resolve in `output_ring`,
    //     output drained to the Vec).
    //   * CONTIG path: `decode_clean_into_contig` — window prepended into a
    //     contiguous buffer at `[0, window_len)`, clean body decoded straight
    //     into the tail, back-refs resolving from that contiguous buffer (the
    //     vendor `setInitialWindow` model the wired Stage-2 path will use).
    // This retires the addressing-retarget + contiguous-back-ref correctness
    // risk independently of wiring (the advisor-required differential).
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    fn deflate_with_dictionary(payload: &[u8], dict: &[u8]) -> Vec<u8> {
        // Raw DEFLATE with a preset dictionary, so the encoder emits back-refs
        // reaching into the dictionary window (distance up to dict.len()).
        use flate2::{Compress, Compression, FlushCompress};
        let mut c = Compress::new(Compression::default(), false);
        c.set_dictionary(dict).expect("set_dictionary");
        let mut out = vec![0u8; payload.len() + 1024];
        let status = c
            .compress(payload, &mut out, FlushCompress::Finish)
            .expect("compress");
        assert_eq!(status, flate2::Status::StreamEnd, "deflate did not finish");
        let n = c.total_out() as usize;
        out.truncate(n);
        out
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    fn decode_ring_clean(deflate_bytes: &[u8], window: &[u8], n_max: usize) -> Vec<u8> {
        let boxed: &'static [u8] = Box::leak(deflate_bytes.to_vec().into_boxed_slice());
        let mut bits = Bits::new(boxed);
        let mut b = Block::new();
        let mut output: Vec<u16> = Vec::new();
        b.set_initial_window(&mut output, window).unwrap();
        b.read_header(&mut bits, false).unwrap();
        // Loop read() until end-of-block (resumable cap may split the body).
        while !b.eob() {
            let n = b.read(&mut bits, &mut output, n_max).unwrap();
            if n == 0 && !b.eob() {
                panic!("ring decode stalled before EOB");
            }
        }
        output
            .iter()
            .map(|&v| {
                assert!(v < 256, "window-seeded clean decode must not emit markers");
                v as u8
            })
            .collect()
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    fn decode_contig_clean(
        deflate_bytes: &[u8],
        window: &[u8],
        cap: usize,
        per_call: usize,
    ) -> Vec<u8> {
        let boxed: &'static [u8] = Box::leak(deflate_bytes.to_vec().into_boxed_slice());
        let mut bits = Bits::new(boxed);
        let mut b = Block::new();
        // Prime clean (window-primed) mode WITHOUT touching the ring: the contig
        // path owns its output. Mirror set_initial_window's clean-mode state.
        let mut sink: Vec<u16> = Vec::new();
        b.set_initial_window(&mut sink, window).unwrap();
        // Contiguous buffer: [window prefix][clean tail].
        let mut buf = vec![0u8; cap];
        buf[..window.len()].copy_from_slice(window);
        let base = buf.as_mut_ptr();
        let mut pos = window.len();
        b.read_header(&mut bits, false).unwrap();
        while !b.eob() {
            let before = pos;
            let n = b
                .decode_clean_into_contig(&mut bits, base, cap, &mut pos, per_call)
                .unwrap();
            assert_eq!(n, pos - before, "contig: emitted != pos delta");
            if n == 0 && !b.eob() {
                panic!("contig decode stalled before EOB (cap too small?)");
            }
        }
        buf[window.len()..pos].to_vec()
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[test]
    fn contig_clean_matches_ring_clean_backref_into_window() {
        // A window whose tail the payload references (back-refs into the prefix).
        let window: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .repeat(800) // ~36 KiB, clamped to 32 KiB by the dictionary path
            .into_iter()
            .collect();
        let window = &window[window.len() - 32768..]; // exactly 32 KiB
                                                      // Payload that repeats window content (forces back-refs into the dict)
                                                      // plus fresh content (forces in-block back-refs + literals).
        let mut payload = Vec::new();
        payload.extend_from_slice(&window[..4096]); // copy of the window head
        payload.extend_from_slice(&b"NEW DATA ".repeat(2000)); // fresh + RLE-ish
        payload.extend_from_slice(&window[16384..16384 + 8192]); // mid-window ref

        let deflate_bytes = deflate_with_dictionary(&payload, window);
        let n_max = payload.len() + 16;
        let cap = window.len() + payload.len() + 512;

        let ring = decode_ring_clean(&deflate_bytes, window, n_max);
        let contig = decode_contig_clean(&deflate_bytes, window, cap, n_max);

        assert_eq!(ring.len(), payload.len(), "ring decoded wrong length");
        assert_eq!(ring, payload, "ring decode != payload");
        assert_eq!(
            contig,
            ring,
            "CONTIG decode diverged from RING decode (len contig={}, ring={})",
            contig.len(),
            ring.len()
        );
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[test]
    fn contig_clean_matches_ring_clean_multi_call_resumable() {
        // Force the resumable cap to split the body across MANY contig calls
        // (per_call small) — exercises the grow-between-calls boundary without
        // a regrow (cap is large enough; the per-call cap is the only split).
        let window: Vec<u8> = (0u32..32768)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        let mut payload = Vec::new();
        payload.extend_from_slice(&window[..20000]); // long dict back-ref
        payload.extend_from_slice(&b"abcdefghijklmnop".repeat(4000));
        payload.extend_from_slice(&window[1000..1000 + 12000]);

        let deflate_bytes = deflate_with_dictionary(&payload, &window);
        let cap = window.len() + payload.len() + 512;
        let ring = decode_ring_clean(&deflate_bytes, &window, payload.len() + 16);
        // Small per-call cap (4 KiB) splits the body into ~10 calls.
        let contig = decode_contig_clean(&deflate_bytes, &window, cap, 4096);
        assert_eq!(ring, payload, "ring decode != payload");
        assert_eq!(contig, ring, "multi-call contig decode diverged from ring");
    }

    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[test]
    fn contig_clean_matches_ring_clean_rle_and_short_distance() {
        // distance==1 RLE + short distances (<8) exercise emit_backref_contig's
        // RLE-fill and exact-byte-copy arms.
        let window: Vec<u8> = b"PREFIX-".repeat(4682).into_iter().collect(); // ~32 KiB
        let window = &window[..32768.min(window.len())];
        let mut payload = Vec::new();
        payload.extend_from_slice(&b"X".repeat(5000)); // long distance==1 RLE
        payload.extend_from_slice(b"abab"); // distance==2 overlap
        payload.extend_from_slice(&b"ababab".repeat(500));
        payload.extend_from_slice(&window[..1000]);

        let deflate_bytes = deflate_with_dictionary(&payload, window);
        let cap = window.len() + payload.len() + 512;
        let ring = decode_ring_clean(&deflate_bytes, window, payload.len() + 16);
        let contig = decode_contig_clean(&deflate_bytes, window, cap, payload.len() + 16);
        assert_eq!(ring, payload, "ring decode != payload");
        assert_eq!(contig, ring, "rle/short-distance contig decode diverged");
    }

    #[test]
    fn build_precode_table_rejects_overlong_code() {
        let mut cl = [0u8; MAX_PRECODE_COUNT];
        cl[0] = 8; // > MAX_PRECODE_LENGTH
        assert_eq!(
            build_precode_table(&cl),
            Err(BlockError::InvalidCodeLengths)
        );
    }

    #[test]
    fn set_initial_window_empty_arms_clean() {
        // Vendor deflate.hpp:1750-1759 (M3 adoption): an EMPTY initial window
        // still flips to CLEAN mode (`m_containsMarkerBytes = false` at :1757
        // is outside the `!initialWindow.empty()` arm). The pre-M3 no-op
        // (stay in marker mode) was the recorded divergence.
        let mut b = Block::new();
        let mut output: Vec<u16> = Vec::new();
        assert!(b.contains_marker_bytes(), "fresh Block starts marker-mode");
        assert!(b.set_initial_window(&mut output, &[]).is_ok());
        assert!(output.is_empty());
        assert!(
            !b.contains_marker_bytes(),
            "empty seed must arm CLEAN mode (vendor :1757)"
        );
    }

    #[test]
    fn set_initial_window_empty_decodes_no_backref_stream_byte_exact() {
        // Gauntlet net: a zero-length window seed arms Clean and a stream with
        // NO back-refs (incompressible random bytes are stored/literal-coded)
        // decodes byte-exact through the seeded-clean engine.
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Deterministic pseudo-random payload — no repeats long enough for
        // back-refs at this scale, and even if the encoder emits one its
        // distance stays within already-decoded output (still valid clean).
        let mut payload = Vec::with_capacity(4096);
        let mut x: u32 = 0x9E37_79B9;
        for _ in 0..4096 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            payload.push((x >> 24) as u8);
        }
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&payload).unwrap();
        let deflate_bytes = enc.finish().unwrap();

        let boxed: &'static [u8] = Box::leak(deflate_bytes.into_boxed_slice());
        let mut bits = Bits::new(boxed);
        let mut b = Block::new();
        let mut output: Vec<u16> = Vec::new();
        b.set_initial_window(&mut output, &[]).unwrap();
        assert!(!b.contains_marker_bytes());
        loop {
            b.read_header(&mut bits, false).unwrap();
            while !b.eob() {
                let n = b.read(&mut bits, &mut output, usize::MAX).unwrap();
                if n == 0 && !b.eob() {
                    panic!("empty-seed clean decode stalled before EOB");
                }
            }
            if b.is_last_block() {
                break;
            }
        }
        let decoded: Vec<u8> = output
            .iter()
            .map(|&v| {
                assert!(v < 256, "clean decode must not emit markers");
                v as u8
            })
            .collect();
        assert_eq!(decoded, payload, "empty-seed clean decode != payload");
    }

    #[test]
    fn set_initial_window_rejects_oversize() {
        let mut b = Block::new();
        let mut output: Vec<u16> = Vec::new();
        let too_big = vec![0u8; MAX_WINDOW_SIZE + 1];
        assert_eq!(
            b.set_initial_window(&mut output, &too_big),
            Err(BlockError::ExceededWindowRange)
        );
        assert!(output.is_empty());
    }

    #[test]
    fn set_initial_window_rejects_after_decode_started() {
        let mut b = Block::new();
        let mut output: Vec<u16> = vec![0xAA];
        assert_eq!(
            b.set_initial_window(&mut output, b"hello world"),
            Err(BlockError::ExceededWindowRange)
        );
        // Output is preserved on error.
        assert_eq!(output, vec![0xAA]);
    }

    #[test]
    fn set_initial_window_seeds_output_and_resolves_backref_to_literal() {
        // Build a deflate block that references the FIRST half of the
        // pre-seeded window — without `set_initial_window` this would
        // emit markers; with it, it should resolve to literal bytes.
        //
        // Plan: compress "abcdefghij" + "<32750 bytes of filler>" + back-ref-only-payload
        // becomes too complex. Simpler: directly test the math by hand —
        // after set_initial_window with N bytes, decoded_bytes_at_block_start
        // = N and the FIRST back-reference at chunk-relative position 0
        // with distance D must source `output[N - D]` and emit NO
        // markers (since marker_count = D - N + 0 saturates at 0).
        //
        // We test this end-to-end by feeding a tiny synthetic dynamic
        // block whose only symbol is a back-reference of distance=10,
        // length=5 — verifying it copies 5 bytes from the window prefix
        // verbatim, never emitting MARKER_BASE+something.
        //
        // Easier route: round-trip a payload that EXACTLY matches the
        // window's first byte pattern. We compress payload P, decompress
        // with set_initial_window(P[:32768] equivalent) and verify
        // back-refs landing in the window prefix decode to the literal
        // bytes from the window. We use a flate2-produced block where
        // the encoder happens to emit back-refs of small distance — the
        // round-trip already tests in-block back-refs at line 1273-1281;
        // here we verify the SEEDED case still works for those (no
        // regression).
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        let payload = b"the quick brown fox jumps over the lazy dog. ".repeat(40);
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&payload).unwrap();
        let deflate_bytes = enc.finish().unwrap();

        // Seed with a 100-byte synthetic window. All subsequent back-refs
        // in the block are SHORT (in-block back-refs) so they shouldn't
        // touch the seeded prefix — but the seeded prefix MUST still
        // appear at output[0..100] for the test to be meaningful.
        let mut b = Block::new();
        let mut output: Vec<u16> = Vec::new();
        let window: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        b.set_initial_window(&mut output, &window).unwrap();
        // Vendor-equivalent semantics: set_initial_window seeds the
        // Block's internal RING (mirror of m_window16). The caller's
        // output Vec receives ONLY decoded bytes from subsequent
        // read() calls — the window prefix lives in the ring and is
        // never drained.
        assert_eq!(output.len(), 0);

        let mut bits = make_bits(&deflate_bytes);
        b.read_header(&mut bits, false).unwrap();
        let _ = b.read(&mut bits, &mut output, payload.len() * 2);
        assert!(b.eob());

        // output[..] = the decoded payload, with NO markers (since
        // distances <= window prefix + decoded < length saturate
        // marker_count to 0 in emit_backref_ring).
        assert_eq!(output.len(), payload.len());
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v < 256,
                "seeded chunk must not emit markers; v={v:#x} at offset {i}"
            );
            assert_eq!(v as u8, payload[i]);
        }
    }

    /// Reproduces the production bug: bootstrapping `Block` at REAL
    /// deflate block boundaries past bit 0 (chunk N > 0 in production)
    /// must succeed. The marker-decoder bootstrap was failing with
    /// `InvalidHuffmanCode` on real silesia chunks, killing parallel
    /// throughput (every prefetch failed → effective T=2 → 0.24× rapidgzip).
    ///
    /// This test compiles on ALL archs (uses pure-Rust canonical-Huffman
    /// fallback on non-x86_64) so the bug is caught locally during dev,
    /// not just on the bench server. If the canonical and ISA-L LUT paths
    /// share the bug, both fail here. If only ISA-L diverges, the test
    /// passes on arm64 and the bug shows on x86_64 only.
    #[test]
    fn block_decode_succeeds_at_every_real_block_boundary() {
        use crate::decompress::inflate::consume_first_decode::Bits;
        use flate2::{write::DeflateEncoder, Compression};
        use std::io::Write;

        // Varied data that flate2 splits into multiple dynamic blocks.
        let mut payload = Vec::with_capacity(2 * 1024 * 1024);
        for i in 0u32..(2 * 1024 * 1024 / 4) {
            payload.extend_from_slice(&i.to_le_bytes());
        }
        payload.extend(
            b"the quick brown fox jumps over the lazy dog. "
                .repeat(2000)
                .iter(),
        );

        let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(6));
        enc.write_all(&payload).unwrap();
        let deflate = enc.finish().unwrap();

        let starts =
            super::record_block_starts(&deflate).expect("oracle should walk the stream cleanly");
        let non_zero_starts: Vec<usize> = starts.into_iter().filter(|&b| b > 0).collect();
        assert!(
            non_zero_starts.len() >= 3,
            "fixture should produce ≥4 deflate blocks, got {}",
            non_zero_starts.len()
        );

        let mut failures = Vec::new();
        for &start_bit in &non_zero_starts {
            // Reproduce bootstrap_with_deflate_block's setup verbatim:
            // construct Bits at the start-byte offset, consume the
            // bit-in-byte, then run Block::read_header + Block::read
            // until EOB or BFINAL or 32 KiB clean tail.
            let byte_offset = start_bit / 8;
            let bit_in_byte = (start_bit % 8) as u32;
            let mut bits = Bits::new(&deflate[byte_offset..]);
            if bit_in_byte > 0 {
                bits.consume(bit_in_byte);
            }
            let mut output: Vec<u16> = Vec::with_capacity(64 * 1024);
            let mut block = Block::new();
            // Decode at most ONE block from this boundary: that's enough
            // to prove header + body decode at this offset.
            let header_res = block.read_header(&mut bits, false);
            if let Err(e) = header_res {
                failures.push((start_bit, format!("header: {e:?}")));
                continue;
            }
            let mut body_err = None;
            while !block.eob() {
                match block.read(&mut bits, &mut output, usize::MAX) {
                    Ok(_) => {}
                    Err(e) => {
                        body_err = Some(format!("body: {e:?}"));
                        break;
                    }
                }
            }
            if let Some(e) = body_err {
                failures.push((start_bit, e));
            }
        }

        assert!(
            failures.is_empty(),
            "Block decode failed at {}/{} real block boundaries — this is THE \
             parallel-SM throughput blocker. First 3: {:?}",
            failures.len(),
            non_zero_starts.len(),
            &failures[..failures.len().min(3)],
        );
    }

    // ── Ring-buffer correctness tests (advisor-flagged scenarios) ────────────
    //
    // These tests pin down the ring buffer + marker pre-init contract
    // explicitly, so any future regression is caught by `cargo test`
    // rather than requiring perf-level inspection of decoded bytes.

    /// The marker zone (upper half of the ring) holds pre-computed
    /// values that EQUAL their own slot index. A back-ref reading
    /// from a slot in this zone therefore pulls a value that, by
    /// construction, IS the correct cross-chunk marker for the
    /// (chunk_position, distance) pair that produced the source slot.
    ///
    /// Verifies vendor's `initializeMarkedWindowBuffer` pre-init
    /// trick at deflate.hpp:875-888 is correctly implemented.
    #[test]
    fn ring_marker_zone_pre_init_values_match_slot_indices() {
        let block = Block::new();
        for slot in MAX_WINDOW_SIZE..RING_SIZE {
            assert_eq!(
                block.ring.window16[slot] as usize, slot,
                "pre-init: ring[{slot}] should equal {slot}, got {}",
                block.ring.window16[slot]
            );
        }
        // Lower half is zero (no pre-init needed; decoder writes here
        // from the start).
        for slot in 0..MAX_WINDOW_SIZE {
            assert_eq!(
                block.ring.window16[slot], 0,
                "pre-init: ring[{slot}] (lower half) should be 0"
            );
        }
    }

    /// For EVERY cross-chunk back-ref scenario `(out_pos, distance)`
    /// where `distance > out_pos` (back-ref reaches before chunk
    /// start), the source slot `(out_pos - distance + RING_SIZE) %
    /// RING_SIZE` holds a pre-init value that EQUALS the marker
    /// value our convention expects. A single `memcpy` from this
    /// slot therefore produces the right marker — no explicit
    /// `marker_count` loop required.
    ///
    /// Marker convention: byte `i` of a back-ref at chunk position
    /// `p` with distance `d`, where `d > p+i`, has expected value
    /// `MARKER_BASE + (MAX_WINDOW_SIZE + p + i - d)`.
    #[test]
    fn pre_init_values_match_marker_convention_for_all_cross_chunk_backrefs() {
        use crate::decompress::parallel::replace_markers::MARKER_BASE;
        let block = Block::new();
        // Sample p and d across the full range, including the
        // boundary cases (d == MAX_WINDOW_SIZE, p == 0,
        // p == MAX_WINDOW_SIZE - 1).
        for &p in &[0, 1, 100, 1000, 16384, MAX_WINDOW_SIZE - 1] {
            for &d in &[p + 1, p + 100, MAX_WINDOW_SIZE / 2, MAX_WINDOW_SIZE] {
                if d <= p {
                    continue; // not a cross-chunk back-ref
                }
                // First byte of the back-ref (i = 0).
                let src_slot = (p + RING_SIZE - d) % RING_SIZE;
                let expected_marker = MARKER_BASE as usize + (MAX_WINDOW_SIZE + p) - d;
                assert_eq!(
                    block.ring.window16[src_slot] as usize, expected_marker,
                    "p={p} d={d}: ring[{src_slot}] = {} but marker convention says {expected_marker}",
                    block.ring.window16[src_slot]
                );
            }
        }
    }

    /// This is the SCENARIO THE ADVISOR FLAGGED.
    ///
    /// State: `ring_pos = MAX_WINDOW_SIZE` (32768 pure literals
    /// decoded), the mid-decode switch has just fired
    /// (`contains_marker_bytes = false`), and the next back-ref
    /// arrives with `distance = MAX_WINDOW_SIZE` (= 32768).
    ///
    /// Advisor's claim: source slot lands at index 32768 (the start
    /// of the marker zone), so the back-ref's `copy_nonoverlapping`
    /// pulls marker value `32768` into the output where a clean
    /// literal is expected.
    ///
    /// Truth (verified by this test): source slot is
    /// `(32768 + RING_SIZE - 32768) % RING_SIZE = 65536 % 65536 = 0`
    /// — the FIRST decoded literal (lower half of ring). Output is
    /// the original literal, not a marker. The pre-init zone is
    /// UNREACHABLE once `ring_pos >= MAX_WINDOW_SIZE` because every
    /// back-ref source `(ring_pos - distance) % RING_SIZE` for
    /// `distance ≤ MAX_WINDOW_SIZE` lies in `[0, ring_pos) ⊂ [0,
    /// RING_SIZE)` and was overwritten by literal decode.
    #[test]
    fn post_switch_max_distance_backref_reads_clean_lower_half_not_marker_zone() {
        let mut block = Block::new();
        // Simulate the state described in the advisor's concern:
        // 32 KiB of distinct literal bytes have been decoded into
        // the lower half of the ring, and the mid-decode switch
        // condition is satisfied.
        for i in 0..MAX_WINDOW_SIZE {
            block.ring.window16[i] = (i & 0xFF) as u16; // distinct (mod 256) literal
        }
        block.ring.pos = MAX_WINDOW_SIZE;
        block.ring.drained = MAX_WINDOW_SIZE;
        block.decoded_bytes = MAX_WINDOW_SIZE;
        block.ring.distance_to_last_marker = MAX_WINDOW_SIZE;
        block.ring.width = super::super::width_ring::RingWidth::Clean;

        // Now invoke emit_backref_ring directly with distance =
        // MAX_WINDOW_SIZE, length = 100. This mirrors the back-ref
        // the advisor said would corrupt output.
        let ring_ptr = block.ring.window16.as_mut_ptr();
        let mut pos = block.ring.pos;
        let mut distance_marker = block.ring.distance_to_last_marker;
        unsafe {
            emit_backref_ring::<false>(
                ring_ptr,
                &mut pos,
                MAX_WINDOW_SIZE, // distance
                100,             // length
                &mut distance_marker,
            );
        }

        // After the call: positions [32768, 32868) in the ring
        // hold the copied bytes. They should equal ring[0..100]
        // (the original literals), NOT marker values from the
        // pre-init zone.
        for i in 0..100 {
            let written = block.ring.window16[MAX_WINDOW_SIZE + i];
            let expected = (i & 0xFF) as u16; // matches the literal we wrote at ring[i]
            assert_eq!(
                written, expected,
                "back-ref at distance=MAX_WINDOW_SIZE wrote {written:#x} at ring[{}] but expected literal {expected:#x} — \
                 this would mean the advisor's bug claim is real",
                MAX_WINDOW_SIZE + i
            );
            // And critically: the written value MUST NOT be a
            // marker value (>= MARKER_BASE = MAX_WINDOW_SIZE).
            assert!(
                (written as usize) < MAX_WINDOW_SIZE,
                "back-ref pulled MARKER VALUE {written:#x} into clean-mode output at ring[{}]",
                MAX_WINDOW_SIZE + i
            );
        }
        assert_eq!(
            pos,
            MAX_WINDOW_SIZE + 100,
            "pos should have advanced by length=100"
        );
    }

    /// Companion to the test above: BEFORE the switch fires (chunk
    /// start, no prior decode), a back-ref with `distance > out_pos`
    /// MUST produce correct marker values via the pre-init pull.
    ///
    /// This verifies the pre-init scheme works as intended for the
    /// case it IS designed to serve.
    #[test]
    fn pre_switch_cross_chunk_backref_produces_correct_markers() {
        use crate::decompress::parallel::replace_markers::MARKER_BASE;
        let mut block = Block::new();
        // Initial state: chunk just started, nothing decoded yet.
        // contains_marker_bytes = true (default from new()).
        let ring_ptr = block.ring.window16.as_mut_ptr();
        let mut pos: usize = 0;
        let mut distance_marker: usize = 0;

        // Back-ref: distance = 100, length = 50, at chunk position
        // 0. Since distance > out_pos (= 0), ALL 50 bytes should be
        // cross-chunk markers.
        unsafe {
            emit_backref_ring::<true>(
                ring_ptr,
                &mut pos,
                100, // distance
                50,  // length
                &mut distance_marker,
            );
        }

        // Check each emitted byte is the correct marker per our
        // convention: MARKER_BASE + (MAX_WINDOW_SIZE + 0 + i - 100).
        for i in 0..50 {
            let written = block.ring.window16[i];
            let expected = MARKER_BASE as usize + MAX_WINDOW_SIZE + i - 100;
            assert_eq!(
                written as usize, expected,
                "back-ref byte {i}: ring[{i}] = {written:#x} but marker convention says {expected:#x}"
            );
        }
        // Counter: we emitted 50 markers in a row → distance to
        // last marker is 0 (the last byte written was a marker).
        assert_eq!(
            distance_marker, 0,
            "every byte was a marker; counter should be 0"
        );
    }

    // ── M2b (DIV-5): vendor stored-block special cases ───────────────────
    //
    // Each test runs BOTH kill-switch arms (via STORED_FLIP_OVERRIDE, since
    // the env read is OnceLock-cached) and asserts byte-identical resolved
    // output plus the per-case proof counters. Serialized by a mutex: while
    // the override forces DISABLED, no thread in the process can increment
    // the counters (every increment is behind `!stored_flip_disabled()`), so
    // exact-zero deltas are safe there; ENABLED-arm deltas use `>=` because
    // unrelated concurrent tests may also decode stored blocks.
    mod stored_flip {
        use super::*;
        use crate::decompress::parallel::replace_markers::MARKER_BASE;
        use std::sync::atomic::Ordering::Relaxed;
        use std::sync::Mutex;

        static LOCK: Mutex<()> = Mutex::new(());

        struct OverrideGuard;
        impl Drop for OverrideGuard {
            fn drop(&mut self) {
                STORED_FLIP_OVERRIDE.store(-1, Relaxed);
            }
        }

        fn with_stored_flip<T>(disabled: bool, f: impl FnOnce() -> T) -> T {
            let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
            STORED_FLIP_OVERRIDE.store(if disabled { 1 } else { 0 }, Relaxed);
            let _restore = OverrideGuard;
            f()
        }

        fn counters() -> (u64, u64, u64, u64) {
            (
                STORED_FLIP_GE_WINDOW.load(Relaxed),
                STORED_FLIP_CROSSING.load(Relaxed),
                STORED_CLEAN_BULK.load(Relaxed),
                STORED_CONTIG_BULK.load(Relaxed),
            )
        }

        /// Hand-built stored block. Valid only at a byte-aligned position
        /// (stream start, after another stored block, or after a sync flush).
        fn stored_block(payload: &[u8], bfinal: bool) -> Vec<u8> {
            assert!(payload.len() <= 65535);
            let mut v = Vec::with_capacity(payload.len() + 5);
            v.push(u8::from(bfinal)); // BFINAL bit, BTYPE=00, zero padding
            let len = payload.len() as u16;
            v.extend_from_slice(&len.to_le_bytes());
            v.extend_from_slice(&(!len).to_le_bytes());
            v.extend_from_slice(payload);
            v
        }

        fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 33) as u8
                })
                .collect()
        }

        fn test_dict() -> Vec<u8> {
            lcg_bytes(7, MAX_WINDOW_SIZE)
        }

        /// Raw DEFLATE of `payload` with preset `dict`, ending in a SYNC
        /// flush (compressed block(s) + the empty stored block, byte-aligned
        /// end) so hand-built stored blocks can be appended.
        fn deflate_with_dict_sync(payload: &[u8], dict: &[u8]) -> Vec<u8> {
            use flate2::{Compress, Compression, FlushCompress};
            let mut c = Compress::new(Compression::default(), false);
            c.set_dictionary(dict).expect("set_dictionary");
            let mut out = vec![0u8; payload.len() + 4096];
            c.compress(payload, &mut out, FlushCompress::Sync)
                .expect("compress+sync");
            assert_eq!(c.total_in() as usize, payload.len(), "sync consumed all");
            out.truncate(c.total_out() as usize);
            out
        }

        /// Ground truth: flate2 raw-inflate with dictionary.
        fn inflate_with_dict(stream: &[u8], dict: &[u8], expect_len: usize) -> Vec<u8> {
            use flate2::{Decompress, FlushDecompress};
            let mut d = Decompress::new(false);
            d.set_dictionary(dict).expect("set_dictionary");
            let mut out = vec![0u8; expect_len + 4096];
            d.decompress(stream, &mut out, FlushDecompress::Finish)
                .expect("ground-truth inflate");
            out.truncate(d.total_out() as usize);
            out
        }

        /// Decode a full raw-deflate stream windowless through the production
        /// `Block::read` ring path. Returns (sink, block-at-end).
        fn decode_windowless(stream: &[u8]) -> (Vec<u16>, Block) {
            let mut bits = make_bits(stream);
            let mut b = Block::new();
            let mut out: Vec<u16> = Vec::new();
            loop {
                b.read_header(&mut bits, false).expect("header");
                let mut guard = 0usize;
                while !b.eob() {
                    let before = out.len();
                    b.read(&mut bits, &mut out, usize::MAX).expect("read");
                    if out.len() == before {
                        guard += 1;
                        assert!(guard < 4, "no decode progress");
                    }
                }
                if b.is_last_block() {
                    break;
                }
            }
            (out, b)
        }

        fn resolve(sink: &[u16], dict: &[u8]) -> Vec<u8> {
            sink.iter()
                .map(|&v| {
                    if v < 256 {
                        v as u8
                    } else {
                        assert!(v >= MARKER_BASE, "invalid 2B code {v:#x}");
                        dict[(v - MARKER_BASE) as usize]
                    }
                })
                .collect()
        }

        /// Case 1, pure-stored stream (gzip-on-incompressible shape):
        /// 40000-byte stored blocks fire the >=window early flip; both arms
        /// produce identical bytes (the disabled arm flips at the same read()
        /// boundary via generic arming cond 2 — all-clean chunk).
        #[test]
        fn case1_pure_stored_stream_both_arms() {
            let payload = lcg_bytes(11, 80_020);
            let mut stream = Vec::new();
            stream.extend_from_slice(&stored_block(&payload[..40_000], false));
            stream.extend_from_slice(&stored_block(&payload[40_000..80_000], false));
            stream.extend_from_slice(&stored_block(&payload[80_000..], true));

            let (enabled, dis) = with_stored_flip(false, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                let c1 = counters();
                assert!(!b.contains_marker_bytes(), "must be clean at end");
                assert!(
                    c1.0 - c0.0 >= 1,
                    "case 1 must fire on a 40000-byte stored block"
                );
                (sink, ())
            });
            let _ = dis;
            let disabled = with_stored_flip(true, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                let c1 = counters();
                assert_eq!(c0, c1, "kill-switch arm must not touch any special case");
                assert!(
                    !b.contains_marker_bytes(),
                    "pure-clean stream still flips via generic arming"
                );
                sink
            });
            let e: Vec<u8> = enabled.iter().map(|&v| v as u8).collect();
            let d: Vec<u8> = disabled.iter().map(|&v| v as u8).collect();
            assert_eq!(e, payload, "enabled arm bytes");
            assert_eq!(d, payload, "disabled arm bytes");
        }

        /// Case 1 after MARKERS — the arming-divergence shape: vendor flips
        /// at the >=32 KiB stored block; the pre-M2b path stays in marker
        /// mode (needs 64 Ki consecutive clean after a marker). Resolved
        /// bytes must be identical; the width at end differs.
        #[test]
        fn case1_after_markers_flips_early_same_bytes() {
            let dict = test_dict();
            // Dict-referencing prefix => markers when decoded windowless.
            let mut p1 = Vec::new();
            for k in 0..8 {
                let s = (k * 1013) % (MAX_WINDOW_SIZE - 400);
                p1.extend_from_slice(&dict[s..s + 300]);
            }
            let prefix = deflate_with_dict_sync(&p1, &dict);
            let stored_a = lcg_bytes(13, 40_000);
            let stored_b = lcg_bytes(17, 10);
            let mut stream = prefix.clone();
            stream.extend_from_slice(&stored_block(&stored_a, false));
            stream.extend_from_slice(&stored_block(&stored_b, true));

            let mut truth = p1.clone();
            truth.extend_from_slice(&stored_a);
            truth.extend_from_slice(&stored_b);
            assert_eq!(
                inflate_with_dict(&stream, &dict, truth.len()),
                truth,
                "stream self-check vs flate2"
            );

            let enabled = with_stored_flip(false, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                let c1 = counters();
                assert!(
                    c1.0 - c0.0 >= 1,
                    "case 1 must fire (40000 >= MAX_WINDOW_SIZE)"
                );
                assert!(
                    !b.contains_marker_bytes(),
                    "M2b: >=window stored block must flip even after markers"
                );
                sink
            });
            let disabled = with_stored_flip(true, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                assert_eq!(c0, counters(), "kill-switch arm inert");
                assert!(
                    b.contains_marker_bytes(),
                    "pre-M2b arming (64 Ki clean after marker) must NOT have flipped"
                );
                sink
            });
            assert!(
                enabled.len() == disabled.len() && enabled.len() == truth.len(),
                "lengths: enabled {} disabled {} truth {}",
                enabled.len(),
                disabled.len(),
                truth.len()
            );
            assert_eq!(resolve(&enabled, &dict), truth, "enabled arm resolved");
            assert_eq!(resolve(&disabled, &dict), truth, "disabled arm resolved");
        }

        /// Case 2 at the EXACT boundary: markers, then stored blocks sized at
        /// runtime so `distance_to_last_marker + uncompressed_size` hits
        /// 32767 (no fire) then 32768 (fires). Covers the window-crossing
        /// downcast + the tiny-stored-run fall-through, plus a trailing
        /// clean-bulk (case 3) block.
        #[test]
        fn case2_crossing_at_exact_boundary() {
            let dict = test_dict();
            let mut p1 = Vec::new();
            for k in 0..4 {
                let s = (k * 911) % (MAX_WINDOW_SIZE - 400);
                p1.extend_from_slice(&dict[s..s + 250]);
            }
            let prefix = deflate_with_dict_sync(&p1, &dict);

            // Measure the post-prefix marker distance d0 on a throwaway decode.
            let d0 = with_stored_flip(true, || {
                let (_, b) = decode_windowless(&{
                    let mut s = prefix.clone();
                    s.extend_from_slice(&stored_block(&[], true));
                    s
                });
                assert!(b.contains_marker_bytes(), "prefix must leave markers");
                b.ring.distance_to_last_marker
            });
            assert!(d0 < 16_000, "prefix trailing-clean too large: {d0}");

            // Block A: brings dist to exactly 32767 (falls through, per-byte).
            let a = lcg_bytes(19, MAX_WINDOW_SIZE - 1 - d0);
            // Block B: 1 byte => dist 32767 + 1 == 32768 fires case 2.
            let b_payload = lcg_bytes(23, 1);
            // Block C: clean bulk (case 3) after the flip.
            let c_payload = lcg_bytes(29, 5_000);
            let mut stream = prefix.clone();
            stream.extend_from_slice(&stored_block(&a, false));
            stream.extend_from_slice(&stored_block(&b_payload, false));
            stream.extend_from_slice(&stored_block(&c_payload, true));

            let mut truth = p1.clone();
            truth.extend_from_slice(&a);
            truth.extend_from_slice(&b_payload);
            truth.extend_from_slice(&c_payload);
            assert_eq!(
                inflate_with_dict(&stream, &dict, truth.len()),
                truth,
                "stream self-check vs flate2"
            );

            let enabled = with_stored_flip(false, || {
                // Step through manually to pin WHICH block fires.
                let mut bits = make_bits(&stream);
                let mut blk = Block::new();
                let mut out: Vec<u16> = Vec::new();
                // Prefix blocks + sync stored + block A: no case-2 fire.
                loop {
                    blk.read_header(&mut bits, false).expect("header");
                    while !blk.eob() {
                        blk.read(&mut bits, &mut out, usize::MAX).expect("read");
                    }
                    if out.len() >= p1.len() + a.len() {
                        break; // block A consumed
                    }
                }
                assert!(blk.contains_marker_bytes(), "dist 32767: no flip yet");
                assert_eq!(blk.ring.distance_to_last_marker, MAX_WINDOW_SIZE - 1);
                let c0 = counters();
                // Block B (1 byte): crossing fires.
                blk.read_header(&mut bits, false).expect("header B");
                while !blk.eob() {
                    blk.read(&mut bits, &mut out, usize::MAX).expect("read B");
                }
                let c1 = counters();
                assert!(c1.1 - c0.1 >= 1, "case 2 must fire at dist+n == 32768");
                assert!(!blk.contains_marker_bytes(), "flipped by case 2");
                // Block C: clean bulk.
                blk.read_header(&mut bits, false).expect("header C");
                while !blk.eob() {
                    blk.read(&mut bits, &mut out, usize::MAX).expect("read C");
                }
                let c2 = counters();
                assert!(c2.2 - c1.2 >= 1, "case 3 must take the clean bulk read");
                assert!(blk.is_last_block());
                out
            });
            let disabled = with_stored_flip(true, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                assert_eq!(c0, counters(), "kill-switch arm inert");
                assert!(b.contains_marker_bytes(), "pre-M2b stays in marker mode");
                sink
            });
            assert_eq!(resolve(&enabled, &dict), truth, "enabled arm resolved");
            assert_eq!(resolve(&disabled, &dict), truth, "disabled arm resolved");
        }

        /// Case 2 with `uncompressed_size == 0`: the sync-flush empty stored
        /// block flips once >=32 KiB of trailing clean exists (markers
        /// earlier), emitting ZERO bytes.
        #[test]
        fn case2_empty_stored_block_flips_after_32k_clean() {
            let dict = test_dict();
            let mut p1 = Vec::new();
            for k in 0..4 {
                let s = (k * 700) % (MAX_WINDOW_SIZE - 300);
                p1.extend_from_slice(&dict[s..s + 200]);
            }
            let prefix = deflate_with_dict_sync(&p1, &dict);
            let d0 = with_stored_flip(true, || {
                let (_, b) = decode_windowless(&{
                    let mut s = prefix.clone();
                    s.extend_from_slice(&stored_block(&[], true));
                    s
                });
                b.ring.distance_to_last_marker
            });
            // Clean run to dist exactly 32768 via two fall-through stored
            // blocks (each keeps dist + n < 32768 BEFORE it completes —
            // sizes chosen so neither fires case 2)…
            assert!(d0 < 10_000);
            let a = lcg_bytes(31, 20_000 - d0); // dist -> 20000
            let b_pay = lcg_bytes(37, 12_767); // dist+n = 32767: no fire
            let mut stream = prefix.clone();
            stream.extend_from_slice(&stored_block(&a, false));
            stream.extend_from_slice(&stored_block(&b_pay, false));
            // dist is now 32767. An EMPTY stored block here is a fall-through
            // (32767 + 0 < 32768). The next 1-byte block hits dist + n ==
            // 32768 exactly ⇒ case 2 fires with n == 1 (any clean stored byte
            // that completes the window fires first — the n == 0 crossing is
            // only reachable after compressed clean output, covered by
            // case2_crossing_after_compressed_clean). The empty block AFTER
            // the flip then exercises the clean n == 0 path (case 3, zero
            // bytes emitted).
            stream.extend_from_slice(&stored_block(&[], false)); // fall-through at 32767
            let one = lcg_bytes(41, 1);
            stream.extend_from_slice(&stored_block(&one, false)); // case 2 fires (n=1)
            stream.extend_from_slice(&stored_block(&[], false)); // clean n=0 (case 3)
            let tail = lcg_bytes(43, 64);
            stream.extend_from_slice(&stored_block(&tail, true));

            let mut truth = p1.clone();
            truth.extend_from_slice(&a);
            truth.extend_from_slice(&b_pay);
            truth.extend_from_slice(&one);
            truth.extend_from_slice(&tail);
            assert_eq!(
                inflate_with_dict(&stream, &dict, truth.len()),
                truth,
                "stream self-check vs flate2"
            );

            let enabled = with_stored_flip(false, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                let c1 = counters();
                assert!(c1.1 - c0.1 >= 1, "case 2 fires at the 1-byte block");
                assert!(!b.contains_marker_bytes());
                sink
            });
            let disabled = with_stored_flip(true, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                assert_eq!(c0, counters(), "kill-switch arm inert");
                assert!(b.contains_marker_bytes());
                sink
            });
            assert_eq!(resolve(&enabled, &dict), truth);
            assert_eq!(resolve(&disabled, &dict), truth);
        }

        /// Case 2 fired by an EMPTY (n=0) stored block after >=32 KiB of
        /// clean output from COMPRESSED blocks (the sync-flush resync shape):
        /// markers first, then ~34 KiB of literal-ish data, then a sync
        /// flush whose empty stored block must flip with zero bytes emitted.
        #[test]
        fn case2_crossing_after_compressed_clean() {
            let dict = test_dict();
            let mut p1 = Vec::new();
            for k in 0..3 {
                let s = (k * 1100) % (MAX_WINDOW_SIZE - 400);
                p1.extend_from_slice(&dict[s..s + 200]);
            }
            // One continuing flate2 stream: P1 (dict-reaching) SYNC, then P2
            // (independent random — no dict reach) SYNC.
            use flate2::{Compress, Compression, FlushCompress};
            let p2 = lcg_bytes(53, 35_000);
            let mut c = Compress::new(Compression::default(), false);
            c.set_dictionary(&dict).expect("set_dictionary");
            let mut out = vec![0u8; p1.len() + p2.len() + 8192];
            c.compress(&p1, &mut out, FlushCompress::Sync).unwrap();
            let n1 = c.total_out() as usize;
            let mut out2 = vec![0u8; p2.len() + 8192];
            c.compress(&p2, &mut out2, FlushCompress::Sync).unwrap();
            let n2 = c.total_out() as usize - n1;
            let mut stream = out[..n1].to_vec();
            stream.extend_from_slice(&out2[..n2]);
            let tail = lcg_bytes(59, 32);
            stream.extend_from_slice(&stored_block(&tail, true));

            let mut truth = p1.clone();
            truth.extend_from_slice(&p2);
            truth.extend_from_slice(&tail);
            assert_eq!(
                inflate_with_dict(&stream, &dict, truth.len()),
                truth,
                "stream self-check vs flate2"
            );

            let enabled = with_stored_flip(false, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                let c1 = counters();
                assert!(!b.contains_marker_bytes(), "must flip by stream end");
                // The P2 sync's empty stored block (or a stored block zlib
                // chose for incompressible P2) fires the crossing case.
                assert!(
                    c1.1 - c0.1 >= 1,
                    "case 2 must fire after >=32 KiB compressed clean"
                );
                sink
            });
            let disabled = with_stored_flip(true, || {
                let c0 = counters();
                let (sink, _b) = decode_windowless(&stream);
                assert_eq!(c0, counters(), "kill-switch arm inert");
                sink
            });
            assert_eq!(resolve(&enabled, &dict), truth);
            assert_eq!(resolve(&disabled, &dict), truth);
        }

        /// Case 3: window-seeded (clean from byte 0) decode of stored blocks,
        /// including ring wrap-around (>131072 bytes total via sub-window
        /// blocks) and tiny stored runs. Also pins that a >=32 KiB stored
        /// block in CLEAN width takes case 1 (vendor's first branch is
        /// width-independent), not case 3.
        #[test]
        fn case3_seeded_bulk_wrap_and_tiny_runs() {
            let dict = test_dict();
            // 30000-byte blocks stay under MAX_WINDOW_SIZE ⇒ case 3 for every
            // block; total 245000 wraps the 131072-slot u8 ring twice.
            let payload = lcg_bytes(61, 245_000);
            let mut stream = Vec::new();
            let mut off = 0usize;
            while off < payload.len() {
                let n = (payload.len() - off).min(30_000);
                stream.extend_from_slice(&stored_block(
                    &payload[off..off + n],
                    off + n == payload.len(),
                ));
                off += n;
            }
            // Tiny stored runs appendix (1..64-byte blocks).
            let tiny = lcg_bytes(67, 300);
            let mut tiny_stream = Vec::new();
            let mut t = 0usize;
            let mut k = 0usize;
            while t < tiny.len() {
                let n = (tiny.len() - t).min(1 + (k % 64));
                tiny_stream.extend_from_slice(&stored_block(&tiny[t..t + n], t + n == tiny.len()));
                t += n;
                k += 1;
            }

            let run = |stream: &[u8], dict: &[u8]| -> Vec<u8> {
                let mut bits = make_bits(stream);
                let mut b = Block::new();
                let mut out: Vec<u16> = Vec::new();
                b.set_initial_window(&mut out, dict).expect("seed");
                loop {
                    b.read_header(&mut bits, false).expect("header");
                    while !b.eob() {
                        b.read(&mut bits, &mut out, usize::MAX).expect("read");
                    }
                    if b.is_last_block() {
                        break;
                    }
                }
                out.iter()
                    .map(|&v| {
                        assert!(v < 256, "seeded clean decode must stay clean");
                        v as u8
                    })
                    .collect()
            };

            // Clean-width >=window stored block: case 1, not case 3.
            let big_payload = lcg_bytes(79, 40_000 + 12);
            let mut big_stream = stored_block(&big_payload[..40_000], false).to_vec();
            big_stream.extend_from_slice(&stored_block(&big_payload[40_000..], true));

            let (e_big, e_tiny, e_case1) = with_stored_flip(false, || {
                let c0 = counters();
                let big = run(&stream, &dict);
                let c1 = counters();
                assert!(
                    c1.2 - c0.2 >= 8,
                    "case 3 must take every sub-window seeded stored block (got {})",
                    c1.2 - c0.2
                );
                let tiny_out = run(&tiny_stream, &dict);
                let c2 = counters();
                let case1_out = run(&big_stream, &dict);
                let c3 = counters();
                assert!(
                    c3.0 - c2.0 >= 1,
                    "clean-width >=window stored block must take case 1"
                );
                (big, tiny_out, case1_out)
            });
            let (d_big, d_tiny, d_case1) = with_stored_flip(true, || {
                let c0 = counters();
                let r = (
                    run(&stream, &dict),
                    run(&tiny_stream, &dict),
                    run(&big_stream, &dict),
                );
                assert_eq!(c0, counters(), "kill-switch arm inert");
                r
            });
            assert_eq!(e_big, payload, "enabled big");
            assert_eq!(d_big, payload, "disabled big");
            assert_eq!(e_tiny, tiny, "enabled tiny runs");
            assert_eq!(d_tiny, tiny, "disabled tiny runs");
            assert_eq!(e_case1, big_payload, "enabled clean-width case 1");
            assert_eq!(d_case1, big_payload, "disabled clean-width case 1");
        }

        /// Contig sibling (`decode_clean_stored_into_contig`): bulk read is
        /// byte-identical to the per-byte arm, respects `n_max_to_decode`
        /// partial reads, and bumps the contig counter.
        #[test]
        fn contig_stored_bulk_matches_per_byte() {
            let dict = test_dict();
            let payload = lcg_bytes(71, 50_000);
            let stream = stored_block(&payload, true);

            let run = |_label: &str| -> Vec<u8> {
                let mut b = Block::new();
                let mut sink: Vec<u16> = Vec::new();
                b.set_initial_window(&mut sink, &dict).expect("seed");
                let mut bits = make_bits(&stream);
                b.read_header(&mut bits, false).expect("header");
                let mut buf = vec![0u8; payload.len() + 512];
                let base = buf.as_mut_ptr();
                let cap = buf.len();
                let mut pos = 0usize;
                // Partial first call exercises the n_max cap.
                let n1 = b
                    .decode_clean_stored_into_contig(&mut bits, base, cap, &mut pos, 100)
                    .expect("partial");
                assert_eq!(n1, 100);
                assert!(!b.eob());
                let n2 = b
                    .decode_clean_stored_into_contig(&mut bits, base, cap, &mut pos, usize::MAX)
                    .expect("rest");
                assert_eq!(n1 + n2, payload.len());
                assert!(b.eob());
                buf.truncate(pos);
                buf
            };

            let enabled = with_stored_flip(false, || {
                let c0 = counters();
                let out = run("enabled");
                let c1 = counters();
                assert!(c1.3 - c0.3 >= 2, "contig bulk must fire on both calls");
                out
            });
            let disabled = with_stored_flip(true, || {
                let c0 = counters();
                let out = run("disabled");
                assert_eq!(c0, counters(), "kill-switch arm inert");
                out
            });
            assert_eq!(enabled, payload);
            assert_eq!(disabled, payload);
        }

        /// Truncated stored payload: the availability gate must keep the
        /// pre-M2b per-byte commit-then-Err semantics in BOTH arms (same
        /// partial output, same residual state, same error).
        #[test]
        fn truncated_stored_keeps_per_byte_error_path() {
            let payload = lcg_bytes(73, 10);
            let mut stream = vec![0u8]; // BFINAL=0 BTYPE=00
            stream.extend_from_slice(&1000u16.to_le_bytes());
            stream.extend_from_slice(&(!1000u16).to_le_bytes());
            stream.extend_from_slice(&payload); // only 10 of 1000 bytes

            let run = || {
                let mut bits = make_bits(&stream);
                let mut b = Block::new();
                b.read_header(&mut bits, false).expect("header");
                let mut out: Vec<u16> = Vec::new();
                let err = b.read(&mut bits, &mut out, usize::MAX).unwrap_err();
                (out, err, b.uncompressed_size(), b.eob())
            };
            let (e_out, e_err, e_left, e_eob) = with_stored_flip(false, run);
            let (d_out, d_err, d_left, d_eob) = with_stored_flip(true, run);
            assert_eq!(e_out, d_out, "partial output identical");
            assert_eq!(
                e_out,
                payload.iter().map(|&b| b as u16).collect::<Vec<u16>>()
            );
            assert_eq!(e_err, d_err, "same error");
            assert_eq!((e_left, e_eob), (d_left, d_eob));
            assert_eq!(e_left, 990, "residual size committed");
            assert!(!e_eob);
        }
    }
}
