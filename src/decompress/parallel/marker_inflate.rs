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
// THREAD-LOCAL, test-only proof-of-path counters for the four stored-block
// decode paths — the three vendor stored-block special cases (deflate.hpp:
// 1212-1256) plus the contig sibling of the clean bulk read. Thread-local for
// the SAME reason MARKER_DIST_LUT_HITS is (see its note below): the
// `stored_flip` differential asserts the kill-switch (disabled) arm records
// EXACTLY zero hits while the enabled arm records the expected counts, and a
// thread-local cell captures exactly this decode's hits — immune to a
// concurrent decode on another test thread in the same binary (which a
// process-wide counter would let contaminate the delta under full-suite
// parallelism). Incremented at the four call sites under `#[cfg(test)]` only;
// never present in a production build, so they cannot touch decode or codegen.
#[cfg(test)]
thread_local! {
    pub(crate) static STORED_FLIP_GE_WINDOW_TL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(crate) static STORED_FLIP_CROSSING_TL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(crate) static STORED_CLEAN_BULK_TL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(crate) static STORED_CONTIG_BULK_TL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test-only override for the kill-switch: -1 = follow the env var,
/// 0 = force-enabled, 1 = force-disabled. Lets one test process exercise
/// BOTH arms (the env-var read is `OnceLock`-cached). One relaxed load per
/// stored BLOCK (not per byte) — negligible.
pub(crate) static STORED_FLIP_OVERRIDE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

/// M2b: when forced off (test-only override) restores the exact pre-M2b
/// stored-block behavior (per-byte decode, generic arming only) in both the
/// ring path (`try_read_stored_special`) and the contig path
/// (`decode_clean_stored_into_contig`'s bulk read).
fn stored_flip_disabled() -> bool {
    let ov = STORED_FLIP_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if ov >= 0 {
        return ov == 1;
    }
    // Shipped default: stored-flip ENABLED (env kill-switch removed). The
    // test override above still exercises the disabled arm for the byte-equality
    // differential.
    false
}

/// Rung-(d) increment 1 test override (mirror of `STORED_FLIP_OVERRIDE`):
/// -1 = follow the env kill-switch; 0 = force the DistTable path ON;
/// 1 = force it OFF (the exact pre-change `dist_hc` chain). Tests flip both
/// arms on the same stream and assert u16 + cursor equality. The env read is
/// OnceLock-cached, so tests must use the override, not set_var.
#[cfg(pure_inflate_decode)]
pub(crate) static MARKER_DIST_LUT_OVERRIDE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

// Test-only liveness counter for the rung-(d) DistTable arm (compiled out of
// production builds): proves the differential's ON arm actually routed
// marker-fast-loop distance decodes through the DistTable path (and that the
// OFF arm routed zero through it).
//
// THREAD-LOCAL, not a process-wide atomic: the differential asserts the OFF
// arm's delta is exactly zero, but the `marker_dist_lut` enable flag is latched
// ONCE per `decode_marker_fast_loop` call (not re-checked per backref), so a
// CONCURRENT test that latched the LUT ON *before* this test's OFF arm sets the
// override would keep incrementing a shared counter inside the OFF window —
// contaminating the delta (a real data race observed on slower/wider-window
// targets, e.g. aarch64 debug CI). The test's own `decode_marker_u16` runs
// synchronously on the test thread, so a thread-local cell captures exactly
// this decode's hits and is immune to other test threads. Test-only; no
// production reader.
#[cfg(all(test, pure_inflate_decode))]
thread_local! {
    pub(crate) static MARKER_DIST_LUT_HITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// ENGINE-W INC-1 / N2 test override for the marker-fast-loop local-Bits
/// mirror (the env kill-switch that used to gate this was removed):
/// -1 = shipped default (localbits ON); 0 = force localbits ON;
/// 1 = force localbits OFF (exact pre-change struct-field path via `bits`).
/// Tests flip both arms on the same stream and assert byte + cursor equality.
pub(crate) static MFAST_LOCALBITS_OVERRIDE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

// Test-only routing counter for the N2 local-Bits ON arm: counts the number
// of 'mfast iterations that ran through the lb (stack-local) code path.
// Compiled out of production builds — engagement proof for the differential.
//
// THREAD-LOCAL for the same reason as [`MARKER_DIST_LUT_HITS`]: the localbits
// enable flag is latched once per call, so a process-wide atomic would let a
// concurrent test's in-flight ON decode leak increments into this test's OFF
// (kill-switch) delta window. The test decodes synchronously on its own thread.
#[cfg(test)]
thread_local! {
    pub(crate) static MFAST_LOCALBITS_ON_ITERS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Rung-(d) increment 1 (git history (campaign plan, removed) §5, F-d1):
/// DistTable distance decode is the shipped default (proven byte-exact
/// equivalent to the pre-change `dist_hc` → DISTANCE_EXTRA → refill-check →
/// DISTANCE_BASE dependent chain — see `marker_dist_lut_diff` below). The
/// `GZIPPY_MARKER_DIST_TABLE=0` env override was removed 2026-07-07 (batch
/// 4f); `MARKER_DIST_LUT_OVERRIDE` (test-only atomic, NOT env-backed) still
/// drives the same-binary causal A/B differential test.
#[cfg(pure_inflate_decode)]
fn marker_dist_lut_disabled() -> bool {
    MARKER_DIST_LUT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) == 1
}

/// ENGINE-W INC-1 / N2: when forced off (test-only override) restores
/// the exact pre-change bit-cursor path in the `'mfast` marker fast loop —
/// struct-field `bits.xxx` accesses instead of the stack-local `lb.xxx` copy.
/// The stack-local breaks the aliasing that forces `bits.bitbuf`/`bitsleft`/
/// `pos` to round-trip memory after each ring store (same finding as the P3.1
/// Lever-B1 on the contig clean loop). One OnceLock read per
/// `read_internal_compressed_specialized::<true>` call; zero-cost when OFF.
fn mfast_localbits_disabled() -> bool {
    let ov = MFAST_LOCALBITS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if ov >= 0 {
        return ov == 1;
    }
    // Shipped default: local-Bits register mirror ENABLED (env kill-switch
    // removed). The test override above still exercises the struct-field arm for
    // the byte-equality differential.
    false
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
/// (vendor `decodeChunkWithRapidgzip`, GzipChunk.hpp:468-654). This is the
/// ONE compiled engine (the legacy `MarkerRing` alternate was deleted in M5).
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
    /// The ONE dual-width decode window (M2, git history (campaign plan, removed) §2/§5):
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
    /// ELEMENT A (igzip single-state-base): the litlen LUT and the
    /// contig-loop dist table are CO-LOCATED INLINE in ONE boxed `AsmState`
    /// (asm_kernel.rs) so the asm addresses both off a single `ctx` base
    /// (igzip `[state+_lit_huff_code+...]`/`[state+_dist_huff_code+...]`).
    /// Boxed → stable address → tables built IN PLACE (zero-copy) and read by
    /// the asm via `[{ctx}+disp+idx*4]`. Was two separate fields
    /// (`lut_litlen: LutLitLenCode` + `dist_table: Option<DistTable>`).
    /// `self.asm.lut_litlen` / `self.asm.dist` everywhere they were used.
    #[cfg(pure_inflate_decode)]
    asm: Box<crate::decompress::parallel::asm_kernel::AsmState>,
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
    /// P3.1 (T1 recovery): libdeflate-style single-lookup distance table for
    /// the CONTIG CLEAN fast loop only. The contig-vs-wrapper cycle profile
    /// measured the back-ref iteration at 84.8 vs 61.5
    /// cyc — the dependent chain `dist_hc cache -> DISTANCE_EXTRA -> refill
    /// check -> DISTANCE_BASE` is the gap; one `DistEntry` lookup decodes
    /// code+extra from the already-peeked word (same technique as the
    /// wrapper's `DistTable`, resumable.rs:1394-1405). AUTHORIZED DEVIATION
    /// from the vendor distance-decode choice, scoped to the inner Huffman
    /// loop (CLAUDE.md "fastest possible raw Huffman decoder": LitLenTable/
    /// DistTable/Bits primitives are open territory). `dist_hc` stays the
    /// engine for the careful loop and every marker/ring path (vendor-
    /// faithful); byte-exactness: identical symbols => identical
    /// distance/bit-consumption, unassigned/invalid code => raw==0 entry =>
    /// `InvalidHuffmanCode`, exactly `dist_hc`'s `None`.
    ///
    /// ELEMENT A: the dist table now lives INLINE in `self.asm.dist` (always
    /// present); this latch replaces the old `Option::is_some()` "built &
    /// valid" signal used by the asm-dispatch gate and the contig loop.
    #[cfg(pure_inflate_decode)]
    dist_valid: bool,
    /// P3.4 item 1 (DistTable build amortization): the distance code lengths
    /// `dist_table` was last built from (`dist_table_nlens == 0` ⇒ never
    /// built). The table is a pure function of the lens, so when a new
    /// dynamic block's dist lens memcmp-equal the cached ones the table is
    /// REUSED verbatim instead of rebuilt — the per-block build cadence is
    /// what the P3.3b T16 triage measured (+8.6ms at 16 threads). Fixed-
    /// Huffman blocks never touch this cache (they use the process-wide
    /// static `fixed_dist_table()`).
    #[cfg(pure_inflate_decode)]
    dist_table_lens: [u8; MAX_DISTANCE_SYMBOL_COUNT],
    #[cfg(pure_inflate_decode)]
    dist_table_nlens: usize,
    /// Per-block "lens verified for the current block" latch — replaces the
    /// old `dist_table = None` reset in `read_header` so the table (and its
    /// allocation) survives across blocks.
    #[cfg(pure_inflate_decode)]
    dist_table_checked: bool,
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
    /// True after `dist_hc` (the reversed-bits distance decoder) was built for
    /// the current block. Split out of `block_huffman_luts_ready` so the clean
    /// CONTIG production path — which decodes distance via the libdeflate-shape
    /// `asm.dist` `DistTable`, NEVER `dist_hc` — skips the redundant per-block
    /// `dist_hc` build. This converges gzippy's per-block table build toward
    /// igzip, which builds a SINGLE distance table (`make_inflate_huff_code_dist`,
    /// igzip_inflate.c) folded into decode; gzippy was building two (the
    /// libdeflate `DistTable` AND `dist_hc`). Built lazily by `ensure_dist_hc`
    /// only on the ring/marker decode paths and the contig `!dist_valid`
    /// fallback — the sole readers of `dist_hc`. Reset per header.
    #[cfg(pure_inflate_decode)]
    dist_hc_built: bool,
    /// Flat libdeflate-style litlen table (engine A) for the CLEAN contig
    /// fastloop wire-in. Built per DynamicHuffman block from `literal_cl` and
    /// CACHED across blocks (rebuilt only when the lengths change — same scheme
    /// as `dist_table_lens`); FixedHuffman uses the process-wide static
    /// `get_fixed_tables().0` and leaves this `None`. Only present on the
    /// pure-Rust clean path that runs the flat fastloop.
    #[cfg(all(
        pure_inflate_decode,
        not(all(feature = "asm-kernel", target_arch = "x86_64"))
    ))]
    flat_litlen: Option<crate::decompress::inflate::libdeflate_entry::LitLenTable>,
    /// Litlen code lengths the cached `flat_litlen` was built from (cache key).
    #[cfg(all(
        pure_inflate_decode,
        not(all(feature = "asm-kernel", target_arch = "x86_64"))
    ))]
    flat_litlen_lens: Vec<u8>,
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
            asm: Box::new(crate::decompress::parallel::asm_kernel::AsmState {
                in_ptr: 0,
                in_lim: 0,
                out_lim: 0,
                out_base: 0,
                lut_litlen: crate::decompress::parallel::lut_huffman::LutLitLenCode::new_empty(),
                dist: crate::decompress::inflate::libdeflate_entry::DistTable::new_empty(),
            }),
            #[cfg(pure_inflate_decode)]
            dist_hc:
                crate::decompress::parallel::huffman_short_bits_cached::DistanceShortBitsCached::new(
                ),
            #[cfg(pure_inflate_decode)]
            dist_valid: false,
            #[cfg(pure_inflate_decode)]
            dist_table_lens: [0u8; MAX_DISTANCE_SYMBOL_COUNT],
            #[cfg(pure_inflate_decode)]
            dist_table_nlens: 0,
            #[cfg(pure_inflate_decode)]
            dist_table_checked: false,
            #[cfg(any(
                all(
                    feature = "isal-compression",
                    not(feature = "pure-rust-inflate"),
                    target_arch = "x86_64"
                ),
                pure_inflate_decode
            ))]
            block_huffman_luts_ready: false,
            #[cfg(pure_inflate_decode)]
            dist_hc_built: false,
            #[cfg(all(
                pure_inflate_decode,
                not(all(feature = "asm-kernel", target_arch = "x86_64"))
            ))]
            flat_litlen: None,
            #[cfg(all(
                pure_inflate_decode,
                not(all(feature = "asm-kernel", target_arch = "x86_64"))
            ))]
            flat_litlen_lens: Vec::new(),
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
    /// MEASUREMENT-ONLY (kernel-isolation A/B harness, examples/kernel_ab_aarch64.rs):
    /// reset ONLY the per-block-body decode accounting (`at_end_of_block`,
    /// `decoded_bytes`) so the SAME real DEFLATE block can be re-decoded by
    /// `decode_clean_into_contig` in a loop WITHOUT re-parsing the header or
    /// rebuilding the Huffman LUTs (`block_huffman_luts_ready` is left TRUE).
    /// Production never calls this — it exists solely to amortize table-build
    /// to ~0 in the isolated kernel-vs-kernel timing. Byte-transparent: the
    /// loop body each iteration decodes from the identical bit cursor + tables
    /// into the identical buffer, so every iteration's output is identical.
    #[allow(dead_code)] // measurement-harness surface only
    pub fn reset_block_body_for_isolation(&mut self) {
        self.at_end_of_block = false;
        self.decoded_bytes = 0;
        self.decoded_bytes_at_block_start = 0;
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
        #[cfg(pure_inflate_decode)]
        {
            self.dist_hc_built = false;
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
        // P3.4 item 1: drop the per-block lens-verified latch (was
        // `dist_table = None`, forcing a fresh alloc+build every block) so
        // decode_clean_into_contig re-validates lazily — marker-mode blocks
        // that never flip clean still skip the build entirely, and a dynamic
        // block whose dist lens match the cached ones reuses the table.
        #[cfg(pure_inflate_decode)]
        {
            self.dist_table_checked = false;
            self.dist_hc_built = false;
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
        // EAGER litlen-LUT build — kept ONLY where the clean-decode hot path READS
        // `lut_litlen`, i.e. when the BMI2 asm kernel owns the contig loop
        // (`run_contig` + `decode_prefilled`, the x86 +7.7% multisym-TRIPLE win) or
        // the legacy ISA-L x86 path. That condition is the exact inverse of the
        // `decode_clean_into_contig` engine-A cfg `not(all(asm-kernel, x86_64))`:
        // when engine A is compiled in (aarch64, or x86 asm-off) the clean path
        // decodes through `flat_litlen` (a `LitLenTable`) and NEVER touches
        // `lut_litlen`, so an eager per-block build here is built-then-discarded
        // (measured M1 silesia T1: builds=3286, reads=0). On those arches the
        // build is left to the lazy, latch-guarded sites that actually read the
        // table — the marker/ring path (`read_internal_compressed_specialized`)
        // and the engine-B fallback in `decode_clean_into_contig` — so the table
        // is still built (identically) on demand wherever it is read; only the
        // never-read clean-path build is retired. Byte-exact: the build is a
        // deterministic function of the code lengths, relocated/skipped, never
        // changed. The dynamic-header validity check is preserved on the clean
        // path by `LitLenTable::build` (engine A's `ensure_flat_litlen`) and on
        // the fallback paths by the lazy `build_huffman_luts_for_block()?`.
        #[cfg(any(
            all(
                feature = "isal-compression",
                not(feature = "pure-rust-inflate"),
                target_arch = "x86_64"
            ),
            all(pure_inflate_decode, feature = "asm-kernel", target_arch = "x86_64")
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
        // Builds ONLY the literal/length LUT (the clean contig hot loop's table
        // + the ISA-L multi-symbol packing). The distance decoder `dist_hc` is
        // built LAZILY by `ensure_dist_hc` only where it is actually read (the
        // ring/marker loops + the contig `!dist_valid` fallback), so the clean
        // contig production path — which decodes distance via `asm.dist`
        // (`ensure_dist_table`) — does not pay a redundant second distance
        // build. Converges toward igzip's single inline distance table
        // (`make_inflate_huff_code_dist`, igzip_inflate.c).
        match self.compression_type {
            CompressionType::FixedHuffman => {
                if !self.lut_litlen_rebuild(&FIXED_LIT_LEN_LENGTHS[..]) {
                    return Err(BlockError::InvalidCodeLengths);
                }
            }
            CompressionType::DynamicHuffman => {
                let split = self.literal_code_count;
                let mut lit_stack = [0u8; MAX_LITERAL_OR_LENGTH_SYMBOLS + 2];
                lit_stack[..split].copy_from_slice(&self.literal_cl[..split]);
                if !self.lut_litlen_rebuild(&lit_stack[..split]) {
                    return Err(BlockError::InvalidCodeLengths);
                }
            }
            CompressionType::Uncompressed | CompressionType::Reserved => {}
        }
        Ok(())
    }

    /// Build the reversed-bits distance decoder (`dist_hc`) for the current
    /// block, lazily and at most once (latched by `dist_hc_built`, reset per
    /// header). Split out of `build_huffman_luts_for_block` so the clean CONTIG
    /// production path — which decodes distance via the libdeflate-shape
    /// `asm.dist` `DistTable`, never `dist_hc` — does NOT pay this per-block
    /// build. Called by the ring/marker decode paths
    /// (`read_internal_compressed_specialized` and its `decode_*` loops) and the
    /// contig `!dist_valid` fallback, the only readers of `dist_hc`.
    ///
    /// Vendor parity (gzip/deflate.hpp:336): distance via the cached
    /// reversed-bits decoder. Mirror of the canonical fallback build at
    /// :1516-1519. Distance lengths come from the SAME source the eager build
    /// used (FixedHuffman → `FIXED_DIST_LENGTHS`; DynamicHuffman → the dist
    /// slice of `literal_cl`) ⇒ byte-identical to the prior eager build.
    #[cfg(pure_inflate_decode)]
    fn ensure_dist_hc(&mut self) -> Result<(), BlockError> {
        if self.dist_hc_built {
            return Ok(());
        }
        match self.compression_type {
            CompressionType::FixedHuffman => {
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
                let mut dist_stack = [0u8; MAX_DISTANCE_SYMBOL_COUNT + 2];
                dist_stack[..end - split].copy_from_slice(&self.literal_cl[split..end]);
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
        self.dist_hc_built = true;
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
    /// When forced off (test-only override) restores the exact pre-M2b
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
            #[cfg(test)]
            STORED_FLIP_GE_WINDOW_TL.with(|c| c.set(c.get() + 1));
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
            #[cfg(test)]
            STORED_FLIP_CROSSING_TL.with(|c| c.set(c.get() + 1));
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
            #[cfg(test)]
            STORED_CLEAN_BULK_TL.with(|c| c.set(c.get() + 1));
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
    /// [`Block::try_read_stored_special`]. The test-only override (forced off)
    /// disables them, restoring this per-byte path exactly.
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
            self.asm.lut_litlen.rebuild_from(litlen_lens)
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
            // WINDOW-ABSENT CONVERGE (Lever A): every caller (`decode_careful_tail`
            // and the `decode_clean_into_contig` Rust-fallback careful loop) calls
            // this IMMEDIATELY after a `bits.refill()`, so the `available() < 32`
            // backstop inside `decode` is provably dead (lut_huffman.rs:1088-1101).
            // Use `decode_prefilled` to drop the per-symbol load+branch — byte-exact,
            // matching the clean asm path's litlen decode.
            let d = self.asm.lut_litlen.decode_prefilled(bits);
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
                // The ring/marker decode loops dispatched below
                // (`decode_clean_fast_loop`, `decode_marker_fast_loop`,
                // `decode_careful_tail`) decode distance via `dist_hc`, so it
                // must be built here. The eager build in `read_header` no longer
                // does it (so the clean contig path can skip it); build it
                // lazily now, latched so the per-block cost is paid once.
                #[cfg(pure_inflate_decode)]
                self.ensure_dist_hc()?;
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
        //
        // The three decode loops this wrapper dispatches to —
        // `decode_clean_fast_loop` (Loop A), `decode_marker_fast_loop` (Loop B),
        // `decode_careful_tail` (Loop C) — are the split of the former monolith,
        // mirroring rg's `readInternalCompressedMultiCached` (fast loops) +
        // `readInternalCompressed` (careful per-symbol fallback) decomposition.
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
        // Drain frontier — constant for the duration of this call (the drain
        // runs in `read()` AFTER this function returns). Passed to the
        // back-ref copy routines so their word-copy rounding overshoot can
        // never wrap onto undrained output (P0 stored-marker-CRC fix).
        let drained = self.ring.drained;
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
        // Hoist the per-backref sparsity-tracking dispatch out of the per-symbol
        // loop. `track_backreferences` is CONSTANT for the duration of this
        // `read()` (only `set_track_backreferences` mutates it, never mid-decode),
        // and defaults to `false` in production — it is set `true` only by the
        // non-production `used_window_symbols` analysis path. When `false`,
        // `record_backreference_for_sparsity` is a no-op early-return, yet at the
        // base it stays an out-of-line CALL per back-reference (perf-annotate:
        // marker_inflate.rs:825 `if !self.track_backreferences` = 0.92% + the two
        // call sites = ~1.1%, pure prologue/epilogue + arg-spill rg never pays —
        // vendor has no sparsity tracking). Snapshotting it here lets the hot loop
        // guard the call with a single hoistable bool (predictably-false in
        // production), removing the per-backref dispatch entirely on the common
        // path. Byte-exact: `false` ⇒ the function did nothing anyway; `true` ⇒
        // the guarded call is identical to the unconditional one.
        let track_backref = self.track_backreferences;

        // ── DISPATCH: clean fast loop (Loop A) → marker fast loop (Loop B) →
        //              careful per-symbol tail (Loop C). ──────────────────────
        // The fast loops (`decode_clean_fast_loop` / `decode_marker_fast_loop`)
        // are the igzip-style speculative multi-cached loops; whichever one
        // matches this specialization runs first and, on `None`, falls through to
        // `decode_careful_tail` with `pos`/`emitted`/`distance_marker` live. Each
        // const-folds to the right branch per `CONTAINS_MARKERS`.
        if !CONTAINS_MARKERS {
            // Loop A: mirror of vendor's clean-window
            // `readInternalCompressedMultiCached<false>` (gzip/deflate.hpp:1589-
            // 1666). `Some(r)` ⇒ the whole decode finished (EOB / error) —
            // propagate; `None` ⇒ fast loop broke, fall through to the careful
            // tail with `pos`/`emitted` live.
            if let Some(r) = self.decode_clean_fast_loop(
                bits,
                ring8,
                &mut pos,
                &mut emitted,
                &mut distance_marker,
                n_max_to_decode,
                drained,
            ) {
                return r;
            }
        }

        // The detail of the marker fast loop (the three u16 deltas vs the clean
        // loop) lives in the `decode_marker_fast_loop` doc.
        if CONTAINS_MARKERS {
            // Loop B extracted into `decode_marker_fast_loop` (mirror of vendor's
            // marker-window `readInternalCompressedMultiCached<true>`,
            // gzip/deflate.hpp:1585-1666). Same `Some`/`None` contract as the
            // clean fast loop. `#[inline(always)]` keeps codegen identical.
            if let Some(r) = self.decode_marker_fast_loop(
                bits,
                ring_ptr,
                &mut pos,
                &mut emitted,
                &mut distance_marker,
                n_max_to_decode,
                drained,
                track_backref,
            ) {
                return r;
            }
        }

        // ── CAREFUL PER-SYMBOL TAIL (Loop C) ─────────────────────────────────
        // The universal per-symbol decode that owns the wrap-straddle, the
        // resumable boundary, and the block tail for BOTH specializations after
        // the fast loops above break. Mirrors vendor's canonical per-symbol
        // `readInternalCompressed` fallback (gzip/deflate.hpp:1612-1661, the
        // unrolled-cache miss path of `readInternalCompressedMultiCached`).
        // Extracted verbatim into `decode_careful_tail` so this monolith mirrors
        // rg's decomposition; `#[inline(always)]` keeps codegen identical.
        self.decode_careful_tail::<CONTAINS_MARKERS>(
            bits,
            ring_ptr,
            ring8,
            &mut pos,
            &mut emitted,
            &mut distance_marker,
            n_max_to_decode,
            drained,
            track_backref,
        )
    }

    /// MARKER-WINDOW SPECULATIVE FAST LOOP (Loop B) — mirror of vendor's
    /// marker (`containsMarkerBytes`) `readInternalCompressedMultiCached`
    /// (gzip/deflate.hpp:1585-1666): the SAME tight multi-cached loop rg runs
    /// for the u16 marker window. Factored out of
    /// `read_internal_compressed_specialized::<true>` for structural convergence
    /// with rapidgzip; `#[inline(always)]` keeps codegen identical (pure code
    /// motion ⇒ byte-identical output). Same `Some(result)` / `None` contract as
    /// [`Self::decode_clean_fast_loop`]: `Some` ⇒ decode terminated here (state
    /// written back to `self.ring.*`), `None` ⇒ fast loop broke, `pos`/`emitted`/
    /// `distance_marker` written back through the `&mut` params for the careful
    /// tail. This method only runs on the `<true>` specialization, so the back-
    /// ref copy is monomorphised at `emit_backref_ring::<true>`.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn decode_marker_fast_loop(
        &mut self,
        bits: &mut Bits,
        ring_ptr: *mut u16,
        pos_io: &mut usize,
        emitted_io: &mut usize,
        distance_marker_io: &mut usize,
        n_max_to_decode: usize,
        drained: usize,
        track_backref: bool,
    ) -> Option<Result<usize, BlockError>> {
        const MAX_RUN_LENGTH: usize = 258;
        const MAX_LIT_LEN_SYM: u32 = 512;
        const FAST_OUT_SLOP: usize = 8 + MAX_RUN_LENGTH + 16;
        const FAST_IN_SLOP: usize = 8;
        let mut pos = *pos_io;
        let mut emitted = *emitted_io;
        let mut distance_marker = *distance_marker_io;
        // ── Rung (d) increment 1 (git history (campaign plan, removed) §4/N1) ──────
        // The fast loop's distance decode goes through the libdeflate-shape
        // `DistTable` — ONE entry load + in-register `consume_entry` /
        // `decode_distance` — replacing the dist_hc → DISTANCE_EXTRA →
        // refill-check → DISTANCE_BASE dependent chain (the P3.1-measured
        // 84.8→61.5 cyc/backref mechanism; see the `dist_table` field doc).
        // Byte-exact by the same P3.1 equivalence already differentialed on
        // the contig loop: identical symbols ⇒ identical distance +
        // identical bit consumption; unassigned/invalid code ⇒ raw==0
        // entry ⇒ `InvalidHuffmanCode`, exactly dist_hc's `None`. The
        // careful loop keeps dist_hc verbatim (rare tail/edge path).
        //
        // Test-only override (`MARKER_DIST_LUT_OVERRIDE`, same-binary causal
        // A/B arm, F-d1): when forced OFF the table is neither built nor used
        // here — the else-arm below is the exact pre-change chain.
        #[cfg(pure_inflate_decode)]
        let marker_dist_lut: bool = !marker_dist_lut_disabled();
        #[cfg(pure_inflate_decode)]
        if marker_dist_lut {
            // P3.4-amortized: fixed blocks use the process-wide static
            // table (no per-block work); dynamic blocks memcmp-reuse.
            // Latched per block (`dist_table_checked`), shared with the
            // contig clean loop's call site.
            self.ensure_dist_table();
        }
        // ELEMENT A: fixed AND dynamic blocks now share the inline
        // `self.asm.dist` (built in `ensure_dist_table`). It is re-borrowed
        // per backref below (the field-path SHARED borrow ends before the
        // `&mut self` sparsity call, so it cannot be hoisted across the loop).
        let in_end = bits.data.len();
        bits.refill();
        // WINDOW-ABSENT CONVERGE (Lever A): backstop-free `decode_prefilled`,
        // matching the clean asm path's litlen decode (chunk_decode contig path
        // / `decode_prefilled` at the run_contig preload). This site sits
        // IMMEDIATELY after the `bits.refill()` above, so the `available() < 32`
        // backstop inside `decode` is a no-op (lut_huffman.rs:1088-1101 proof) —
        // byte-exact, drops the per-symbol load+branch from the marker hot loop.
        let mut pre = self.asm.lut_litlen.decode_prefilled(bits);
        // T3-ILP #2: speculative pre-refill litlen short entry (assigned at the
        // bottom of each iteration BEFORE the refill; declared here in the
        // caller scope so the macro-expanded loop body and the `$litlen_decode`
        // block that reads it share one variable — macro hygiene, same as `pre`).
        // The `= 0` init is dead (every read is preceded by the per-iteration
        // assignment at the loop bottom); it exists only to satisfy the
        // definite-init check for the cross-macro-hygiene shared binding.
        #[allow(unused_assignments)]
        let mut spec_litlen: u32 = 0;
        // T3-MARKER-ILP LEVER: the marker-fast-loop litlen preload is the shipped
        // default (the A/B kill-switch that reverted to `refill(); decode()` was
        // removed); the load-before-refill software pipeline always runs.
        // T3-ILP #3 (dist-preload): loop-invariant enable. The speculative
        // first-level dist lookup is hoisted to the top of the loop body ONLY
        // when the marker LUT dist path is the active decode arm (marker_dist_lut
        // && dist_valid ⇔ `marker_dt = Some(..)` below); otherwise the dist_hc
        // arm is taken and there is nothing to preload. `dist_valid` is set by
        // `ensure_dist_table` (called just above) and is loop-invariant (no table
        // rebuild inside the fast loop).
        //
        // ARCH-GATED (2026-07-02, gated cross-arch A/B on silesia, same-binary
        // dist-preload ON vs OFF, sha 028bd002...):
        //   aarch64 (Apple M1): WIN — T3 ON<OFF 23/25 (min -2.0% / median -2.4%),
        //     T4 21/25 (-1.6% / -1.7%). The wide OoO window + spare load ports
        //     absorb the per-iteration speculative load and hide the dist
        //     dependent-load latency.
        //   x86_64: NO WIN — AMD Zen2 TIE (T3 15/31, T4 19/31; Δmedian ≤0.6% ≪
        //     ~1.3% run spread; 16xT2 throughput unchanged), Intel slight
        //     LOSS-lean (T3 8/25, T8 10/25 favour OFF). Microcoded/narrower
        //     schedulers do not hide the extra load, so compile-gate it OFF on
        //     x86 — `dist_preload_on` folds to `false`, the top-of-loop hoist and
        //     its branch dead-code-eliminate, and the length arm reverts to the
        //     exact pre-lever in-arm `dt.lookup` (x86 codegen byte-identical).
        #[cfg(all(pure_inflate_decode, target_arch = "aarch64"))]
        let dist_preload_on = marker_dist_lut && self.dist_valid;
        #[cfg(not(all(pure_inflate_decode, target_arch = "aarch64")))]
        let dist_preload_on = false;
        // ── N2 (ENGINE-W INC-1): local-Bits register mirror ──────────────
        // Hoist bitbuf/bitsleft/pos into a stack-local `lb: Bits` for the
        // duration of `'mfast` and write back at every exit. The raw-pointer
        // ring stores (through `ring_ptr`) defeat LLVM's alias analysis on the
        // struct-field path: the compiler cannot prove `ring_ptr` never aliases
        // `bits.bitbuf` etc. and reloads them from memory after each store. A
        // non-escaping stack local has unambiguous provenance — LLVM keeps it
        // in registers. Byte-exact by construction (identical bit reads /
        // consumption / output). The test-only override (forced off)
        // restores the struct-field path; the same-binary two-variant macro
        // (`mfast_lb_run!`) keeps the loop body in one place.
        macro_rules! mfast_lb_run {
            ($cur:ident, $dist_hc_decode:expr, $litlen_decode:block, $sync_at_exit:block) => {
                'mfast: loop {
                    // Resumable cap + wrap headroom + input slop. Headroom is in u16
                    // SLOTS (ring modulus is RING_SIZE = 65536 slots = 128 KiB).
                    let dst_phys = pos % RING_SIZE;
                    let out_ok = emitted + FAST_OUT_SLOP < n_max_to_decode
                        && dst_phys + FAST_OUT_SLOP <= RING_SIZE;
                    let in_ok = $cur.pos + FAST_IN_SLOP < in_end;
                    if !(out_ok && in_ok) {
                        break 'mfast;
                    }
                    let sym0 = pre.symbol;
                    let sym_count0 = pre.sym_count;
                    let bit_count0 = pre.bit_count;
                    if bit_count0 == 0 {
                        break 'mfast;
                    }
                    $cur.consume(bit_count0);

                    // ── T3 MARKER-ILP LEVER #3: SPECULATIVE DIST-ENTRY PRELOAD ──
                    // Issue the FIRST-level distance table load NOW — right after
                    // the litlen `consume` shifts `$cur.bitbuf` so its low
                    // `DistTable::TABLE_BITS` bits index the (possible) next dist
                    // code — instead of inside the length arm after the branch
                    // resolves. The address depends only on `$cur.bitbuf & mask`,
                    // so the OoO core issues this ~3-4-cyc dependent load in
                    // PARALLEL with the independent work that follows (the wide
                    // literal store + branchless leading-literal count + the
                    // literal-vs-length branch), pulling it off the backref
                    // critical chain. This is the igzip asm `mov {dpre},0x1FF /
                    // and {dpre},{bitbuf} / mov {dpre},[dist_off]` speculation at
                    // asm_kernel.rs:625-627 ("all the speculation above ran before
                    // this single data-dependent branch resolves", asm 628-634),
                    // ported into the marker path.
                    //
                    // Byte-exact: `$cur.bitbuf` is NOT modified between here and
                    // the length arm's use (the store/count/EOB-/oversize-checks
                    // never touch it, and `self.asm.dist` is not rebuilt inside the
                    // loop), so `dist_spec` is bit-identical to the fresh in-arm
                    // `dt.lookup($cur.bitbuf)` it replaces — a debug_assert proves
                    // it at every backref. `lookup` masks to TABLE_BITS and indexes
                    // an always-allocated table (memory-safe even on the discarded
                    // literal iterations). Gated on `dist_preload_on`
                    // (marker_dist_lut && dist_valid ⇔ the length arm takes the
                    // `Some(dt)` path).
                    let dist_spec = if dist_preload_on {
                        self.asm.dist.lookup($cur.bitbuf)
                    } else {
                        // Dummy (never read): the OFF arm decodes via the in-arm
                        // `dt.lookup` exactly as before this lever.
                        crate::decompress::inflate::libdeflate_entry::DistEntry::distance(0, 0, 0)
                    };

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
                        let widened: u64 =
                            (p & 0xFF) | ((p & 0xFF00) << 8) | ((p & 0xFF_0000) << 16);
                        (ring_ptr.add(dst_phys) as *mut u64).write_unaligned(widened);
                    }

                    // ── T3 MARKER-ILP LEVER (branchless leading-literal count) ──
                    // Replaces the per-symbol `while remaining > 0` walk (a serial
                    // `s >>= 8` loop-carried chain + a data-dependent branch per
                    // packed byte) with a single read of the FINAL packed symbol.
                    //
                    // Byte-exact invariant (proven at the LUT packer): every
                    // NON-final packed symbol is a literal (<256). The pair/triple
                    // builders fast-forward past length-code buckets
                    // (`sym >= 256` skips at lut_huffman.rs:752/803/822), so only
                    // the LAST packed symbol (sym2 of a pair at bits 8.., sym3 of a
                    // triple at bits 16..) can be a length/EOB (>255). The scalar
                    // walk above therefore ALWAYS counted `sym_count-1` leading
                    // literals and only inspected the final symbol for `>255`
                    // (its `|| remaining > 1` arm forced every non-final symbol to
                    // be treated as a literal regardless of its bits). This computes
                    // the identical `lit_prefix` / `trailing_code` / `have_trailing`
                    // directly. The `symbol` field is masked to LARGE_SHORT_SYM_MASK
                    // (25 bits) so `(sym0 >> shift) & 0xFFFF` reads a clean window
                    // (zeros above the final symbol) — same value the walk saw at
                    // its final iteration. Removes the mispredictable loop-trip
                    // branch and the serial shift chain from the marker hot loop.
                    let count = sym_count0 as usize;
                    let last_code = ((sym0 >> ((count - 1) * 8)) & 0xFFFF) as u16;
                    let (lit_prefix, trailing_code, have_trailing) = if last_code > 255 {
                        (count - 1, last_code, true)
                    } else {
                        (count, 0u16, false)
                    };
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
                            $sync_at_exit;
                            return Some(Ok(emitted));
                        }
                        if (code as u32) > MAX_LIT_LEN_SYM {
                            self.ring.pos = pos;
                            self.decoded_bytes += emitted;
                            self.ring.distance_to_last_marker = distance_marker;
                            $sync_at_exit;
                            return Some(Err(BlockError::InvalidHuffmanCode));
                        }
                        let length = (code as usize).wrapping_sub(254);
                        if length != 0 {
                            // Rung (d) N1: per-backref table select (see the hoist
                            // comment above the loop). `None` ⇔ kill-switch OFF
                            // arm or a builder-declined block — both take the
                            // pre-change dist_hc chain below.
                            #[cfg(pure_inflate_decode)]
                            let marker_dt: Option<
                                &crate::decompress::inflate::libdeflate_entry::DistTable,
                            > = if marker_dist_lut && self.dist_valid {
                                Some(&self.asm.dist)
                            } else {
                                None
                            };
                            #[cfg(not(pure_inflate_decode))]
                            let marker_dt: Option<
                                &crate::decompress::inflate::libdeflate_entry::DistTable,
                            > = None;
                            let distance = if let Some(dt) = marker_dt {
                                // Single-lookup path — mirror of the contig fast
                                // loop's dist decode (incl. subtable + raw==0
                                // handling). Bit-budget proof: the bottom
                                // `$cur.refill()` is fast-form whenever the top
                                // `in_ok` guard holds (a slow refill ends with
                                // `pos == data.len()`, which fails `in_ok`), so
                                // iteration tops have >= 56 real bits; the litlen
                                // consume is <= 20 ⇒ >= 36 here >= the 28-bit
                                // worst case (15-bit dist code + 13 extra).
                                use crate::decompress::inflate::libdeflate_entry::DistTable;
                                #[cfg(all(test, pure_inflate_decode))]
                                MARKER_DIST_LUT_HITS.with(|c| c.set(c.get() + 1));
                                // T3-ILP #3: use the entry preloaded at the top of
                                // the body (its load overlapped the store/count/
                                // branch). Byte-exact: `$cur.bitbuf` is unchanged
                                // since the hoist, so `dist_spec` == a fresh lookup
                                // here (asserted in debug/test). OFF ⇒ verbatim
                                // in-arm lookup (pre-lever behaviour).
                                debug_assert!(
                                    !dist_preload_on
                                        || dist_spec.raw() == dt.lookup($cur.bitbuf).raw(),
                                    "dist preload diverged from fresh lookup: bitsleft={}",
                                    $cur.bitsleft
                                );
                                let mut dist_entry = if dist_preload_on {
                                    dist_spec
                                } else {
                                    dt.lookup($cur.bitbuf)
                                };
                                if dist_entry.is_subtable_ptr() {
                                    $cur.consume(DistTable::TABLE_BITS as u32);
                                    dist_entry = dt.lookup_subtable_direct(dist_entry, $cur.bitbuf);
                                }
                                let dist_raw = dist_entry.raw();
                                // raw == 0 is the unassigned/invalid-code marker
                                // (codes 30/31 and incomplete-code holes) — the
                                // exact set `dist_hc.decode` returns `None` for.
                                if dist_raw == 0 {
                                    self.ring.pos = pos;
                                    self.decoded_bytes += emitted;
                                    self.ring.distance_to_last_marker = distance_marker;
                                    $sync_at_exit;
                                    return Some(Err(BlockError::InvalidHuffmanCode));
                                }
                                let dist_extra_saved = $cur.bitbuf;
                                $cur.consume_entry(dist_raw);
                                dist_entry.decode_distance(dist_extra_saved) as usize
                            } else {
                                // Pre-change chain VERBATIM (N1 kill-switch arm).
                                let dsym = match $dist_hc_decode {
                                    Some(d) => d,
                                    None => {
                                        self.ring.pos = pos;
                                        self.decoded_bytes += emitted;
                                        self.ring.distance_to_last_marker = distance_marker;
                                        $sync_at_exit;
                                        return Some(Err(BlockError::InvalidHuffmanCode));
                                    }
                                };
                                if dsym as usize >= DISTANCE_BASE.len() {
                                    self.ring.pos = pos;
                                    self.decoded_bytes += emitted;
                                    self.ring.distance_to_last_marker = distance_marker;
                                    $sync_at_exit;
                                    return Some(Err(BlockError::InvalidHuffmanCode));
                                }
                                let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                                if extra > 0 {
                                    if $cur.available() < extra {
                                        $cur.refill();
                                        if $cur.available() < extra {
                                            self.ring.pos = pos;
                                            self.decoded_bytes += emitted;
                                            self.ring.distance_to_last_marker = distance_marker;
                                            $sync_at_exit;
                                            return Some(Err(BlockError::EndOfFile));
                                        }
                                    }
                                    let mask = (1u64 << extra) - 1;
                                    let v = ($cur.peek() & mask) as usize;
                                    $cur.consume(extra);
                                    DISTANCE_BASE[dsym as usize] as usize + v
                                } else {
                                    DISTANCE_BASE[dsym as usize] as usize
                                }
                            };
                            if distance == 0 || distance > MAX_WINDOW_SIZE {
                                self.ring.pos = pos;
                                self.decoded_bytes += emitted;
                                self.ring.distance_to_last_marker = distance_marker;
                                $sync_at_exit;
                                return Some(Err(BlockError::ExceededWindowRange));
                            }
                            // NO clean-mode `distance > decoded+emitted` check here —
                            // vendor const-folds it out for marker windows.
                            // Back-ref via the SAME production routine the careful loop
                            // uses for markers; it maintains `distance_marker` (the fast
                            // `>= distance` skip + backward marker scan) and is wrap-safe.
                            unsafe {
                                emit_backref_ring::<true>(
                                    ring_ptr,
                                    &mut pos,
                                    drained,
                                    distance,
                                    length,
                                    &mut distance_marker,
                                );
                            }
                            if track_backref {
                                self.record_backreference_for_sparsity(distance, length, emitted);
                            }
                            emitted += length;
                        }
                    }

                    // ── T3 MARKER-ILP LEVER #2: LOAD-BEFORE-REFILL PRELOAD ──────
                    // Speculatively load the NEXT litlen short entry from the
                    // CURRENT (pre-refill) bitbuf, THEN refill. The entry address
                    // depends only on `$cur.bitbuf`'s low 12 bits — NOT on the
                    // refill's word-load+shift+or — so the OoO core issues this
                    // dependent table load in PARALLEL with the refill instead of
                    // serialising its ~3-4-cyc latency behind it (the M1-preload
                    // pattern, consume_first_decode.rs:1832-1856 / igzip asm
                    // asm_kernel.rs:602-607). Shortens the loop-carried
                    // `refill → LUT load → bit_count → consume` recurrence that
                    // gates marker-loop IPC (T3 residual: 1.99 vs rg 2.06).
                    //
                    // Byte-exact contract: `bitsleft >= ISAL_DECODE_LONG_BITS(12)`
                    // here (worst case = one 15-bit long litlen + 28-bit dist off a
                    // ≥56-bit fill ⇒ ≥13 remaining), so the low-12 index is
                    // unchanged by the refill's high-bit OR; `decode_from_spec`
                    // trusts the entry ONLY for short codes and re-decodes long
                    // codes (>12 bits) from the post-refill reader.
                    spec_litlen = self.asm.lut_litlen.spec_short_entry($cur.bitbuf);
                    debug_assert!(
                        ($cur.bitsleft as u8)
                            >= crate::decompress::parallel::lut_huffman::ISAL_DECODE_LONG_BITS
                                as u8
                            || $cur.pos >= in_end,
                        "marker preload invariant violated: bitsleft={} < 12",
                        $cur.bitsleft as u8
                    );
                    $cur.refill();
                    $litlen_decode;
                }
            };
        }
        let mfast_localbits_on = !mfast_localbits_disabled();
        if mfast_localbits_on {
            // N2 LOCAL-BITS PATH: stack-local lb — compiler promotes
            // bitbuf/bitsleft/pos to registers (ring stores can't alias lb).
            // Test-only routing counter proves the arm is taken.
            #[cfg(test)]
            MFAST_LOCALBITS_ON_ITERS.with(|c| c.set(c.get() + 1));
            let mut lb = Bits {
                data: bits.data,
                pos: bits.pos,
                bitbuf: bits.bitbuf,
                bitsleft: bits.bitsleft,
            };
            mfast_lb_run!(
                lb,
                self.dist_hc.decode(&mut lb),
                {
                    // WINDOW-ABSENT CONVERGE (Lever A): the bottom-of-loop
                    // `$cur.refill()` runs immediately before this decode, so the
                    // backstop is dead — use `decode_prefilled` like the clean path.
                    // T3-ILP #2: finish from the entry pre-loaded before the
                    // refill (`spec_litlen`).
                    pre = self.asm.lut_litlen.decode_from_spec(spec_litlen, &lb);
                },
                {
                    bits.pos = lb.pos;
                    bits.bitbuf = lb.bitbuf;
                    bits.bitsleft = lb.bitsleft;
                }
            );
            // Sync lb → bits after natural 'mfast exit (break paths land here).
            bits.pos = lb.pos;
            bits.bitbuf = lb.bitbuf;
            bits.bitsleft = lb.bitsleft;
        } else {
            // TEST-OVERRIDE PATH (forced off): exact
            // pre-change struct-field path — bits.xxx throughout the loop.
            // Same-binary causal A/B arm for F-w1.
            mfast_lb_run!(
                bits,
                self.dist_hc.decode(bits),
                {
                    // WINDOW-ABSENT CONVERGE (Lever A): post-`$cur.refill()` site
                    // — backstop-free decode (kill-switch struct-field arm).
                    // T3-ILP #2: finish from the pre-refill-loaded entry.
                    pre = self.asm.lut_litlen.decode_from_spec(spec_litlen, bits);
                },
                {}
            );
        }
        // FALL THROUGH: every `break 'mfast` leaves a fresh un-consumed `pre`
        // and the synced `bits` cursor before it; write the live cursors back so
        // the careful tail resumes byte-exactly.
        *pos_io = pos;
        *emitted_io = emitted;
        *distance_marker_io = distance_marker;
        None
    }

    /// CLEAN-WINDOW SPECULATIVE FAST LOOP (Loop A) — mirror of vendor's
    /// clean (`!containsMarkerBytes`) `readInternalCompressedMultiCached`
    /// (gzip/deflate.hpp:1589-1666): the igzip-style software-pipelined multi-
    /// symbol loop on gzippy's wrapping u8 ring. Factored out of
    /// `read_internal_compressed_specialized::<false>` for structural
    /// convergence with rapidgzip; `#[inline(always)]` keeps codegen identical
    /// to the prior in-line block (pure code motion ⇒ byte-identical output).
    ///
    /// Returns `Some(result)` when the whole decode terminated here (EOB or an
    /// error — `self.ring.*` written back exactly as the in-line block did), or
    /// `None` when the fast loop broke on a guard/invalid-code: the bit cursor
    /// then sits before an un-consumed `pre`, and `pos`/`emitted` are written
    /// back through the `&mut` params so the careful tail resumes byte-exactly.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn decode_clean_fast_loop(
        &mut self,
        bits: &mut Bits,
        ring8: *mut u8,
        pos_io: &mut usize,
        emitted_io: &mut usize,
        distance_marker_io: &mut usize,
        n_max_to_decode: usize,
        drained: usize,
    ) -> Option<Result<usize, BlockError>> {
        const MAX_RUN_LENGTH: usize = 258;
        const MAX_LIT_LEN_SYM: u32 = 512;
        // Output headroom the fast loop reserves so it can over-write without a
        // per-symbol bounds check: 8 speculative literal bytes + a 258-byte
        // max-length back-ref + a 16-byte word-copy overshoot (igzip asm:511).
        const FAST_OUT_SLOP: usize = 8 + MAX_RUN_LENGTH + 16;
        // Input slop so a refill can always read an 8-byte word (igzip
        // IN_BUFFER_SLOP, asm:48).
        const FAST_IN_SLOP: usize = 8;
        let mut pos = *pos_io;
        let mut emitted = *emitted_io;
        let distance_marker = *distance_marker_io;
        let ring8_fast = ring8;
        let in_end = bits.data.len();
        // Preload (igzip pipeline): decode the first symbol before entering
        // the loop so the back-ref branch resolves against an already-fetched
        // next symbol. `pre` is ALWAYS a fresh, un-consumed decode at the top
        // of each iteration (we consume its bits inside, then preload again),
        // so on ANY break the bit cursor sits exactly before `pre`'s bits and
        // the careful loop re-decodes from the same position — never desyncs.
        bits.refill();
        let mut pre = self.asm.lut_litlen.decode(bits);
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
                    return Some(Ok(emitted));
                }
                if (code as u32) > MAX_LIT_LEN_SYM {
                    self.ring.pos = pos;
                    self.decoded_bytes += emitted;
                    self.ring.distance_to_last_marker = distance_marker;
                    return Some(Err(BlockError::InvalidHuffmanCode));
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
                            return Some(Err(BlockError::InvalidHuffmanCode));
                        }
                    };
                    if dsym as usize >= DISTANCE_BASE.len() {
                        self.ring.pos = pos;
                        self.decoded_bytes += emitted;
                        self.ring.distance_to_last_marker = distance_marker;
                        return Some(Err(BlockError::InvalidHuffmanCode));
                    }
                    let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                    let distance = if extra > 0 {
                        if bits.available() < extra {
                            bits.refill();
                            if bits.available() < extra {
                                self.ring.pos = pos;
                                self.decoded_bytes += emitted;
                                self.ring.distance_to_last_marker = distance_marker;
                                return Some(Err(BlockError::EndOfFile));
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
                        return Some(Err(BlockError::ExceededWindowRange));
                    }
                    if distance > self.decoded_bytes + emitted {
                        self.ring.pos = pos;
                        self.decoded_bytes += emitted;
                        self.ring.distance_to_last_marker = distance_marker;
                        return Some(Err(BlockError::ExceededWindowRange));
                    }
                    // Back-ref copy via the SAME production routine the
                    // careful loop uses — fully wrap-safe (its non-overlap /
                    // RLE / overlap arms each mask every index `% U8_RING_SIZE`
                    // and only take the fast word-copy when the rounded run
                    // fits without straddling). The dst word-copy headroom is
                    // guaranteed by the top guard (`dst_phys + FAST_OUT_SLOP
                    // <= U8_RING_SIZE`); the source straddle is handled inside.
                    unsafe {
                        emit_backref_ring_u8(ring8_fast, &mut pos, drained, distance, length);
                    }
                    self.record_backreference_for_sparsity(distance, length, emitted);
                    emitted += length;
                }
            }

            // Preload next symbol (pipeline). Refill first so the decode and
            // any subsequent dist-extra reads have headroom.
            bits.refill();
            pre = self.asm.lut_litlen.decode(bits);
        }
        // FALL THROUGH: `pre` was decoded but NOT consumed (we always break
        // with a fresh un-consumed `pre`), so the bit cursor is positioned
        // exactly before `pre`'s bits. The careful loop re-decodes from here —
        // no state carried, no desync. Write the live cursors back so the
        // caller's careful tail resumes byte-exactly.
        *pos_io = pos;
        *emitted_io = emitted;
        *distance_marker_io = distance_marker;
        None
    }

    /// CAREFUL PER-SYMBOL TAIL — the universal (both-specialization) per-symbol
    /// decode loop, factored out of `read_internal_compressed_specialized` for
    /// structural convergence with rapidgzip. Mirrors vendor's canonical
    /// per-symbol path `readInternalCompressed` (gzip/deflate.hpp:1612-1661) —
    /// the cache-miss fallback of `readInternalCompressedMultiCached`. It owns
    /// the wrap-straddle, the resumable `n_max_to_decode` boundary, and the
    /// block tail after the speculative fast loops (`decode_clean_fast_loop` /
    /// `decode_marker_fast_loop`) break. `#[inline(always)]`: this is pure code
    /// motion (the loop body is byte-for-byte the prior in-line tail), so the
    /// emitted machine code — and therefore the decoded bytes — is identical.
    ///
    /// State is threaded by `&mut` (`pos`/`emitted`/`distance_marker`); this
    /// loop always TERMINATES the decode (it never falls through), so it returns
    /// the final `Result` directly and the `commit!` macro performs the same
    /// `self.ring.*` write-back the in-line tail did.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn decode_careful_tail<const CONTAINS_MARKERS: bool>(
        &mut self,
        bits: &mut Bits,
        ring_ptr: *mut u16,
        ring8: *mut u8,
        pos_io: &mut usize,
        emitted_io: &mut usize,
        distance_marker_io: &mut usize,
        n_max_to_decode: usize,
        drained: usize,
        track_backref: bool,
    ) -> Result<usize, BlockError> {
        const MAX_LIT_LEN_SYM: u32 = 512;
        let mut pos = *pos_io;
        let mut emitted = *emitted_io;
        let mut distance_marker = *distance_marker_io;
        macro_rules! commit {
            ($result:expr) => {{
                self.ring.pos = pos;
                self.decoded_bytes += emitted;
                self.ring.distance_to_last_marker = distance_marker;
                return $result;
            }};
        }
        while emitted < n_max_to_decode {
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
                            drained,
                            distance,
                            length,
                            &mut distance_marker,
                        );
                    } else {
                        // Clean tail: u8-direct copy (half the traffic, no
                        // marker scan). Vendor readInternal<false> back-ref.
                        emit_backref_ring_u8(ring8, &mut pos, drained, distance, length);
                    }
                }
                if track_backref {
                    self.record_backreference_for_sparsity(distance, length, emitted);
                }
                emitted += length;
                break;
            }
        }
        self.ring.pos = pos;
        self.decoded_bytes += emitted;
        self.ring.distance_to_last_marker = distance_marker;
        Ok(emitted)
    }

    /// P3.1/P3.4 dist-table ensure, factored out of `decode_clean_into_contig`
    /// for rung (d) increment 1 (git history (campaign plan, removed) §4/N1): the
    /// libdeflate-shape `DistTable` build with the P3.4 amortization scheme —
    /// FixedHuffman + amortization ON never touches this cache (callers use the
    /// process-wide `fixed_dist_table()`); DynamicHuffman blocks memcmp the
    /// (≤30-byte) lens and REUSE the live table on a match, else rebuild in
    /// place. Latched per block by `dist_table_checked` (reset in
    /// `read_header`), so the contig clean loop and the marker fast loop can
    /// both call it and the work runs at most once per block.
    #[cfg(pure_inflate_decode)]
    fn ensure_dist_table(&mut self) {
        if self.dist_table_checked {
            return;
        }
        self.dist_table_checked = true;
        // ELEMENT A: the dist table lives INLINE in `self.asm.dist` and is
        // built IN PLACE (zero-copy). Both fixed and dynamic blocks build into
        // it (the old process-wide `fixed_dist_table()` static is no longer the
        // asm/contig source — the asm addresses ONLY `self.asm.dist` off its
        // single base). The lens-cache still skips redundant rebuilds, so a run
        // of fixed (constant lens) or repeated-dynamic blocks rebuilds at most
        // once. `dist_valid` replaces the old `Option::is_some()` signal.
        let lens: &[u8] = match self.compression_type {
            CompressionType::FixedHuffman => &FIXED_DIST_LENGTHS[..],
            CompressionType::DynamicHuffman => {
                let split = self.literal_code_count;
                let end = split + self.distance_code_count;
                &self.literal_cl[split..end]
            }
            _ => {
                self.dist_valid = false;
                return;
            }
        };
        let reusable = self.dist_valid && &self.dist_table_lens[..self.dist_table_nlens] == lens;
        if !reusable {
            let ok = self.asm.dist.rebuild(lens);
            self.dist_valid = ok;
            if ok {
                self.dist_table_lens[..lens.len()].copy_from_slice(lens);
                self.dist_table_nlens = lens.len();
            } else {
                self.dist_table_nlens = 0;
            }
        }
    }

    /// Build/cache the flat libdeflate-style litlen table (engine A) for the
    /// CLEAN contig fastloop wire-in. DynamicHuffman: build from
    /// `literal_cl[..literal_code_count]`, cached on `flat_litlen` keyed by
    /// `flat_litlen_lens` (rebuild only when lengths change — same amortization
    /// as `ensure_dist_table`). FixedHuffman: leaves `flat_litlen = None`; the
    /// caller uses the process-wide static `get_fixed_tables().0`.
    ///
    /// Returns `true` if a usable table is available (static-fixed or cached).
    #[cfg(all(
        pure_inflate_decode,
        not(all(feature = "asm-kernel", target_arch = "x86_64"))
    ))]
    fn ensure_flat_litlen(&mut self) -> bool {
        use crate::decompress::inflate::libdeflate_entry::LitLenTable;
        match self.compression_type {
            CompressionType::FixedHuffman => {
                // Static process-wide fixed table; nothing to build.
                self.flat_litlen = None;
                true
            }
            CompressionType::DynamicHuffman => {
                let split = self.literal_code_count;
                let lens = &self.literal_cl[..split];
                let reusable =
                    self.flat_litlen.is_some() && self.flat_litlen_lens.as_slice() == lens;
                if reusable {
                    return true;
                }
                match LitLenTable::build(lens) {
                    Some(t) => {
                        self.flat_litlen_lens.clear();
                        self.flat_litlen_lens.extend_from_slice(lens);
                        self.flat_litlen = Some(t);
                        true
                    }
                    None => {
                        self.flat_litlen = None;
                        self.flat_litlen_lens.clear();
                        false
                    }
                }
            }
            _ => false,
        }
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
    /// # Safety
    /// `base` must be a valid `*mut u8` for the range `[0, cap)`; `*pos <= cap`
    /// and the per-`read()` headroom contract documented inline (`out_room`)
    /// must hold so the word-copy overshoot stays in-bounds. `bits.data` must
    /// not alias the `[base, base+cap)` destination. (Marked `unsafe` because it
    /// dereferences the raw `base` pointer — clippy `not_unsafe_ptr_arg_deref`.)
    pub unsafe fn decode_clean_into_contig(
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
        // LAZY two-level LUT build (engine B). Validate the compression type
        // here, but DEFER `build_huffman_luts_for_block` (which builds the
        // engine-B `lut_litlen` table) until just before the engine-B fallback
        // loop below. Engine A (the production clean path) commits on EVERY exit
        // without ever touching `lut_litlen`, so on the clean path the engine-B
        // table is never built — retiring the clean-path double table-build. The
        // dynamic-header validity check that `build_huffman_luts_for_block`
        // performed is preserved: the flat path validates via
        // `LitLenTable::build` (ensure_flat_litlen → false on bad lengths), and
        // the engine-B fallback still calls `build_huffman_luts_for_block()?`.
        match self.compression_type {
            CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {}
            _ => return Err(BlockError::InvalidCompression),
        }
        // P3.1 lazy dist_table: only needed in the contig fast loop, so skip
        // the build for marker-mode blocks that never flip clean.
        // After build_huffman_luts_for_block, dist_hc holds the validated lens,
        // and literal_cl / FIXED_DIST_LENGTHS still hold the same raw lengths.
        //
        // P3.4 item 1 (build amortization — the P3.3b T16 +8.6ms): the table
        // is a pure function of the dist code lengths, so
        //   * FixedHuffman blocks share ONE process-wide static table
        //     (`fixed_dist_table()`) — zero per-block builds;
        //   * DynamicHuffman blocks memcmp the (≤30-byte) lens against the
        //     ones the live table was built from and REUSE it on a match;
        //   * a rebuild reuses the entries allocation (`DistTable::rebuild`).
        // Byte-exact by construction: identical lens ⇒ identical table ⇒
        // identical decodes (differentials: `dist_table_rebuild_matches_
        // fresh_build`, `contig_clean_stream_multi_block_dist_reuse`).
        #[cfg(pure_inflate_decode)]
        self.ensure_dist_table();
        // The contig loops decode distance via `asm.dist` (built above). They
        // only fall back to `dist_hc` when `asm.dist` declined the lengths
        // (`!dist_valid` — invalid dist code, unreachable on a valid stream).
        // Build `dist_hc` for that fallback ONLY in that case, so a clean block
        // with a valid dist table pays no redundant `dist_hc` build.
        #[cfg(pure_inflate_decode)]
        if !self.dist_valid {
            self.ensure_dist_hc()?;
        }

        const MAX_LIT_LEN_SYM: u32 = 512;
        const MAX_RUN_LENGTH: usize = 258;
        // Contiguous-buffer cap: leave room for one max back-ref + word overshoot.
        //
        // Headroom proof (advisor-required): one outer iteration writes EITHER
        // ≤3 packed literals, OR ≤ 1+LIT_CHAIN_MAX = 3 runtime-chained single
        // literals (P3.2), OR exactly one back-ref of length ≤ MAX_RUN_LENGTH,
        // NEVER both — the ISA-L LUT's multi-symbol packing (sym_count ∈ {1,2,3})
        // stops at any symbol ≥ 256, so a pair/triple slot is all literals and a
        // back-ref appears only at sym_count == 1 (igzip_inflate.c:473-476, see
        // the multi-symbol notes on read_internal_compressed_specialized); the
        // P3.2 literal chain emits only lone literals and CARRIES any other
        // packet un-consumed to the next iteration. The
        // outer guard keeps `*pos ≤ out_room - 1 = cap - (MAX_RUN_LENGTH+8) - 1`
        // at any back-ref, and `emit_backref_contig`'s word path touches up to
        // `*pos + ((MAX_RUN_LENGTH+7)&!7) = *pos + 264`, max end `= cap - 3 < cap`.
        // The literal-triple path (packed or chained) ends at `≤ cap - 264`. So
        // MAX_RUN_LENGTH+8 is
        // provably sufficient — but the proof DEPENDS on `MAX sym_count == 3`,
        // "multi-symbol ⇒ all literals", and `1 + LIT_CHAIN_MAX ≤ FAST_OUT_SLOP`
        // (compile-asserted below); a LUT change that packed a length code
        // into a multi-symbol slot would invalidate it.
        let out_room = cap.saturating_sub(MAX_RUN_LENGTH + 8);
        debug_assert!(*pos <= out_room, "decode_clean_into_contig: no spare");
        let local_cap = n_max_to_decode.min(out_room.saturating_sub(*pos));

        let mut emitted: usize = 0;

        macro_rules! commit {
            ($result:expr) => {{
                let __r: Result<usize, BlockError> = $result;
                self.decoded_bytes += emitted;
                return __r;
            }};
        }

        // ── ENGINE A (flat libdeflate-style) BULK FASTLOOP wire-in ───────────
        // STEP-0.5 verdict: the flat decoder (`decode_huffman_libdeflate_style`)
        // beats the two-level engine-B loop below 2.07–3.84× instr/B on gzippy's
        // own primitives. Run engine A's BOUNDED fastloop here to consume the
        // BULK of the block straight into the contiguous tail; it exits at a
        // clean SYMBOL boundary ≥ FASTLOOP_MARGIN (320) bytes before the budget,
        // hands a CANONICAL re-read `Bits` (the N32-safe reclass_reread seam —
        // `bit_position()` → `at_bit_offset`) to the engine-B careful tail below,
        // which finishes the <320-byte residual + the exact `n_max`/`out_room`
        // boundary + EOB (already resumable + byte-exact). So engine A never
        // lands on the resumable boundary itself — the seam risk stays with the
        // proven engine-B tail.
        //
        // Gated OFF whenever a measurement knob / oracle / backref-tracking is
        // active (those instruments must keep measuring the engine-B path, and
        // engine A implements none of them). Compiled out on x86 WITH the BMI2
        // asm kernel (design §3: x86 asm path untouched); present on aarch64 and
        // on the x86 asm-OFF cross-ISA-LAW arm.
        #[cfg(all(
            pure_inflate_decode,
            not(all(feature = "asm-kernel", target_arch = "x86_64"))
        ))]
        {
            use crate::decompress::inflate::consume_first_decode::{
                decode_huffman_fastloop_bounded_pipelined, FlatFastloopExit, FLAT_CONTIG_BYTES,
                FLAT_CONTIG_CALLS,
            };
            let flat_eligible = !self.track_backreferences && self.dist_valid && local_cap > 0;
            if flat_eligible && self.ensure_flat_litlen() {
                let out_fastloop_end = *pos + local_cap;
                // SAFETY: `base` is valid for `[0, cap)` (caller contract). The
                // fastloop bound is `out_fastloop_end = *pos + local_cap ≤
                // out_room = cap - (MAX_RUN_LENGTH+8)`, and a single iteration's
                // copy/8-literal overshoot past the bound is ≤ MAX_RUN_LENGTH+40
                // < the reserved 266 headroom, so every store stays `< cap`.
                let out_slice = unsafe { std::slice::from_raw_parts_mut(base, cap) };
                let fixed_tbl;
                let litlen_ref = match self.compression_type {
                    CompressionType::FixedHuffman => {
                        fixed_tbl =
                            crate::decompress::inflate::libdeflate_decode::get_fixed_tables();
                        &fixed_tbl.0
                    }
                    // ensure_flat_litlen() returned true ⇒ Some for Dynamic.
                    _ => self.flat_litlen.as_ref().unwrap(),
                };
                // ROUTE-B: production default = faithful libdeflate software-
                // pipelined fastloop port (gated aarch64 T1 win).
                let exit = decode_huffman_fastloop_bounded_pipelined(
                    bits,
                    out_slice,
                    *pos,
                    out_fastloop_end,
                    litlen_ref,
                    &self.asm.dist,
                );
                FLAT_CONTIG_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Canonical seam re-read: rebuild a pristine `Bits` from the
                // absolute bit position so the engine-B tail / next read_header
                // never sees engine A's internal bit representation.
                let reread = |bits: &mut Bits| {
                    let data = bits.data;
                    let bp = bits.bit_position();
                    *bits = Bits::at_bit_offset(data, bp);
                };
                match exit {
                    FlatFastloopExit::EndOfBlock(new_pos) => {
                        let delta = new_pos - *pos;
                        *pos = new_pos;
                        emitted += delta;
                        FLAT_CONTIG_BYTES
                            .fetch_add(delta as u64, std::sync::atomic::Ordering::Relaxed);
                        self.at_end_of_block = true;
                        reread(bits);
                        commit!(Ok(emitted));
                    }
                    FlatFastloopExit::Stopped(new_pos) => {
                        let delta = new_pos - *pos;
                        *pos = new_pos;
                        emitted += delta;
                        FLAT_CONTIG_BYTES
                            .fetch_add(delta as u64, std::sync::atomic::Ordering::Relaxed);
                        reread(bits);
                        // ENGINE-A FLAT CAREFUL TAIL (the convergence): finish the
                        // <FASTLOOP_MARGIN residual + the exact n_max/EOB boundary
                        // with engine A's own flat resumable careful loop, then
                        // COMMIT — the engine-B two-level speculative loop + careful
                        // tail below are NOT reached on the clean path, and the
                        // engine-B `lut_litlen` table is never built. `out_fastloop_end`
                        // (= *pos_initial + local_cap) is the absolute budget cap.
                        use crate::decompress::inflate::consume_first_decode::{
                            decode_clean_careful_flat, FlatCarefulExit,
                        };
                        let cexit = decode_clean_careful_flat(
                            bits,
                            out_slice,
                            *pos,
                            out_fastloop_end,
                            litlen_ref,
                            &self.asm.dist,
                        );
                        match cexit {
                            FlatCarefulExit::EndOfBlock(p) => {
                                let d = p - *pos;
                                *pos = p;
                                emitted += d;
                                FLAT_CONTIG_BYTES
                                    .fetch_add(d as u64, std::sync::atomic::Ordering::Relaxed);
                                self.at_end_of_block = true;
                                reread(bits);
                                commit!(Ok(emitted));
                            }
                            FlatCarefulExit::Stopped(p) => {
                                let d = p - *pos;
                                *pos = p;
                                emitted += d;
                                FLAT_CONTIG_BYTES
                                    .fetch_add(d as u64, std::sync::atomic::Ordering::Relaxed);
                                reread(bits);
                                commit!(Ok(emitted));
                            }
                            FlatCarefulExit::InvalidDistance(p) => {
                                let d = p - *pos;
                                *pos = p;
                                emitted += d;
                                FLAT_CONTIG_BYTES
                                    .fetch_add(d as u64, std::sync::atomic::Ordering::Relaxed);
                                reread(bits);
                                commit!(Err(BlockError::ExceededWindowRange));
                            }
                        }
                    }
                    FlatFastloopExit::InvalidDistance(new_pos) => {
                        let delta = new_pos - *pos;
                        *pos = new_pos;
                        emitted += delta;
                        FLAT_CONTIG_BYTES
                            .fetch_add(delta as u64, std::sync::atomic::Ordering::Relaxed);
                        reread(bits);
                        commit!(Err(BlockError::ExceededWindowRange));
                    }
                }
            }
        }

        // ── ENGINE-B fallback table-build (LAZY) ─────────────────────────────
        // Reached ONLY when engine A did not take the block: a measurement knob
        // forced engine B, the flat litlen build declined invalid lengths,
        // `local_cap == 0`, `!dist_valid`, or the x86 BMI2 asm kernel is compiled
        // in (engine A is cfg'd out there). Engine A commits on EVERY exit above,
        // so on the production clean path this is NEVER reached — the engine-B
        // two-level `lut_litlen` table is NOT built (the clean-path double-build
        // retired). `CLEAN_LUT_BUILDS` counts the fallbacks (0 on a pure-clean
        // run = Gate-0 non-inert proof). `build_huffman_luts_for_block` also
        // performs the dynamic-header validity check for the engine-B path.
        if !self.block_huffman_luts_ready {
            self.build_huffman_luts_for_block()?;
            self.block_huffman_luts_ready = true;
            crate::decompress::inflate::consume_first_decode::CLEAN_LUT_BUILDS
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        // Headroom: at any literal `*pos <= out_room - 1 = cap - (MAX_RUN_LENGTH+8)
        // - 1 = cap - 267`, so the speculative 8-byte store touches at most
        // `*pos + 8 <= cap - 259 < cap` — never OOB. The back-ref word-copy headroom
        // is the SAME MAX_RUN_LENGTH+8 the careful loop relies on. Input slop:
        // FAST_IN_SLOP bytes ahead of `bits.pos` so refills stay on the fast word
        // path; when input slop fails we break to the careful loop (which owns the
        // block tail / resumable boundary).
        const FAST_OUT_SLOP: usize = 8;
        const FAST_IN_SLOP: usize = 8;
        // P3.2: max EXTRA single-literal packets consumed inline after the lit1
        // arm's first literal (3 literals/iteration total — the wrapper's
        // measured 2-extra shape, resumable.rs:1305-1313 / vendor
        // decompress_template.h:381 "3 extras decreases performance").
        const LIT_CHAIN_MAX: usize = 2;
        // Headroom-proof dependency: the chain writes ≤ 1+LIT_CHAIN_MAX bytes
        // under the iteration-top `emitted + FAST_OUT_SLOP < local_cap` guard
        // (clippy int_plus_one shape of `1 + LIT_CHAIN_MAX <= FAST_OUT_SLOP`).
        const _: () = assert!(LIT_CHAIN_MAX < FAST_OUT_SLOP);
        {
            let in_end = bits.data.len();
            // P3.1 (T1 recovery, Lever-B1 class): mirror the bit-reader into a
            // STACK-LOCAL `Bits` for the duration of the fast loop and write it
            // back at every exit. Through `&mut Bits` the
            // bitbuf/bitsleft/pos round-trip memory each symbol (the raw
            // `base: *mut u8` output stores defeat LLVM's aliasing analysis in
            // practice — same finding as resumable.rs Lever B1, which lifted
            // the wrapper loop's reader into locals for exactly this reason);
            // a non-escaping local promotes them to registers. Decode order and
            // bit consumption are IDENTICAL — byte-exact by construction.
            let mut lb = Bits {
                data: bits.data,
                pos: bits.pos,
                bitbuf: bits.bitbuf,
                bitsleft: bits.bitsleft,
            };
            // Hoisted backref-tracking flag (the per-backref `&mut self` method
            // call re-loaded the flag from memory inside the hot loop; the flag
            // cannot change mid-decode).
            let track_backrefs = self.track_backreferences;
            // Hoisted single-lookup distance table (see `dist_table` field doc).
            // Field-disjoint from every `self` mutation inside the loop
            // (`at_end_of_block`, `backreferences`, `decoded_bytes`), so the
            // borrow is legal across them. FixedHuffman blocks borrow the
            // process-wide static table (P3.4 item 1) — same 5-bit lens every
            // time, so the table is identical to the per-block build it
            // replaces.
            // ELEMENT A: set the per-call header on the boxed AsmState BEFORE
            // taking the shared table borrows below — the asm reads
            // in_ptr/in_lim/out_lim/out_base via `[ctx+0/8/16/24]`. This is the
            // only &mut self.asm in the region; the loop then only reads it
            // (`run_contig(&self.asm,…)`, `decode_prefilled`, dist lookups all
            // SHARED — the anchor stores that needed &mut are gone).
            #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
            {
                self.asm.in_ptr = lb.data.as_ptr() as u64;
                self.asm.in_lim = in_end.saturating_sub(super::asm_kernel::IN_MARGIN) as u64;
                self.asm.out_lim = (base as u64)
                    .wrapping_add(*pos as u64)
                    .wrapping_add(local_cap as u64)
                    .wrapping_sub(FAST_OUT_SLOP as u64);
                self.asm.out_base = base as u64;
            }
            // Hoisted single-lookup distance table (inline in `self.asm.dist`).
            // Field-disjoint from every `self` mutation inside the loop
            // (`at_end_of_block`, `backreferences`, `decoded_bytes`), so the
            // SHARED borrow is legal across them and across `run_contig(&self.asm)`.
            #[cfg(pure_inflate_decode)]
            let dist_tbl: Option<
                &crate::decompress::inflate::libdeflate_entry::DistTable,
            > = if self.dist_valid {
                Some(&self.asm.dist)
            } else {
                None
            };
            #[cfg(not(pure_inflate_decode))]
            let dist_tbl: Option<
                &crate::decompress::inflate::libdeflate_entry::DistTable,
            > = None;
            // ── ASM-campaign rung (c) dispatch (asm_kernel.rs; charter §4) ──
            // One OnceLock env/CPU read + the pure knob-exclusion predicate
            // (charter §3.5 — any measurement knob forces the pure-Rust
            // loop). OFF ⇒ the loop below is the sole path; it is ALWAYS
            // compiled regardless.
            #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
            let mut asm_on: bool = super::asm_kernel::enabled()
                && super::asm_kernel::dispatch_allowed(
                    // Measurement knobs removed: these are always the OFF values,
                    // so they never force the pure-Rust loop.
                    false,
                    false,
                    false,
                    0,
                    0,
                    track_backrefs,
                    local_cap,
                    FAST_OUT_SLOP,
                )
                && dist_tbl.is_some();
            macro_rules! sync_local_bits {
                () => {{
                    bits.pos = lb.pos;
                    bits.bitbuf = lb.bitbuf;
                    bits.bitsleft = lb.bitsleft;
                }};
            }
            macro_rules! commit_fast {
                ($result:expr) => {{
                    sync_local_bits!();
                    let __r: Result<usize, BlockError> = $result;
                    self.decoded_bytes += emitted;
                    return __r;
                }};
            }
            lb.refill();
            // P3.5 c4: immediately after a refill — backstop-free decode
            // (byte-exact proof on `decode_prefilled`).
            let mut pre = self.asm.lut_litlen.decode_prefilled(&lb);
            // ── TWO LOOP VARIANTS (flip-precondition 4, campaign §9 gate) ──
            // The fast loop is instantiated twice from one macro source:
            // the `false` instantiation const-folds the asm dispatch away
            // (`if false && asm_on` is statically dead), so a binary built
            // WITH the asm-kernel feature but dispatch-disabled (kill-switch
            // / no BMI2 / knobs) runs a loop with ZERO asm-related code in
            // its body — the c3 OFF-vs-base ~+44 ms layout tax was the
            // per-iteration dispatch test + asm_ctx liveness in the shared
            // loop. The variant is selected ONCE per region run, before the
            // loop; the `true` variant keeps the runtime `asm_on` test so
            // an EXIT_BOUNDARY can still hand the tail to the Rust arms
            // (identical to the pre-split behavior).
            //
            // MEASURED VERDICT (frozen guest, §10): the split cuts the
            // disabled-arm tax 44 → 12 ms (1.0%) on T1 silesia and to a
            // TIE on model T8 (643 vs 645 ms). The residual is NOT in the
            // loop: the false variant's body contains zero asm code
            // (objdump: exactly ONE run_contig call site in this whole
            // function — the true variant's), and the per-call costs
            // (enabled() + dispatch_allowed + KernCtx init) are bounded
            // ≤ ~2 ms by the 24,128 region calls. What remains is
            // cross-binary code layout of the doubled function body —
            // the campaign's documented layout-phantom class, irreducible
            // here without outlining/PGO. Production-irrelevant post-flip:
            // the disabled arm exists only on pre-BMI2 x86 (pre-2013
            // Haswell class), where `enabled()` is false at runtime.
            macro_rules! fast_loop_run {
                ($use_asm:literal) => {
            'fast: loop {
                // ── ASM-campaign rung (c): the asm region owns the
                // per-symbol hot path; Rust handles boundary/rare packets
                // (asm_kernel.rs EXIT-STATE CONTRACT). Re-entry via
                // `continue 'fast`. `prof_on` true ⇒ `asm_on` false, so the
                // profiler block above never brackets asm iterations.
                #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
                if $use_asm && asm_on {
                    // SAFETY: contract E1-E6 hold here — this IS the Rust
                    // loop's iteration top (fresh un-consumed packet, clean
                    // bitsleft, lockstep cursors, validated tables, knobs
                    // excluded by dispatch).
                    let dst0 = unsafe { base.add(*pos) };
                    let (exit, dst1) =
                        unsafe { super::asm_kernel::run_contig(&self.asm, &mut lb, dst0) };
                    let delta = (dst1 as usize) - (dst0 as usize);
                    if delta != 0 {
                        *pos += delta;
                        emitted += delta;
                        // X5/X6: re-derive the carried packet from the
                        // identical cursor (decode purity + ≤21-bit backing
                        // — contract doc); skipped when nothing changed.
                        pre = self.asm.lut_litlen.decode_prefilled(&lb);
                    }
                    if exit == super::asm_kernel::EXIT_BOUNDARY {
                        // Monotone guard failure — the Rust loop owns the
                        // tail under its own (looser) guards.
                        asm_on = false;
                    }
                }
                // Out headroom: keep `*pos + FAST_OUT_SLOP` within the reserved
                // word-copy region (out_room already reserves MAX_RUN_LENGTH+8 so
                // both the 8-byte literal store and a max back-ref fit). In slop:
                // enough bytes ahead for the unchecked fast refill.
                let out_ok = emitted + FAST_OUT_SLOP < local_cap;
                let in_ok = lb.pos + FAST_IN_SLOP < in_end;
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
                lb.consume(bit_count0);

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
                // P3.5 c2: fused litlen→dist lookahead (lone-length arm).
                // Carries the dist short-LUT entry loaded EARLY — at the
                // point the litlen packet is known to be a lone non-literal,
                // before the EOB/MAX/length branch tree resolves — so the
                // dependent dist-table load issues sooner. Same index
                // (`lb.bitbuf` is post-consume and untouched until the
                // backref arm), same table ⇒ same value as the in-place
                // lookup it replaces; a wasted load on an EOB/invalid code
                // reads always-valid table memory. Byte-exact by
                // construction.
                let mut spec_dist: Option<crate::decompress::inflate::libdeflate_entry::DistEntry> =
                    None;
                if sym_count0 == 1 {
                    let code = (sym0 & 0xFFFF) as u16;
                    if code <= 255 {
                        unsafe {
                            base.add(*pos).write(code as u8);
                        }
                        *pos += 1;
                        emitted += 1;
                        // ── P3.2 RUNTIME LITERAL CHAIN ───────────────────────
                        // Wrapper analog (resumable.rs:1314-1356; vendor
                        // libdeflate decompress_template.h:354-381): after a
                        // lone literal, speculatively decode the next packet;
                        // while it is ALSO a lone literal, consume + emit it
                        // inline (≤ LIT_CHAIN_MAX extras); the final un-consumed
                        // packet becomes next iteration's `pre` (the carry that
                        // makes the speculative decode never wasted). This is
                        // the mechanism behind the wrapper's ~1.96 lits/iter:
                        // the ISA-L LUT's build-time packing needs the COMBINED
                        // codeword lengths to fit in the 12-bit short lookup
                        // (igzip_inflate.c:386-599), which 2×(7..9)-bit silesia
                        // literal codes almost never do — runtime chaining is
                        // length-independent.
                        //
                        // Byte-exact by construction: identical symbols decoded
                        // in identical order with identical bit consumption —
                        // only WHICH loop arm consumes them changes. Refills
                        // are append-only (Bits::refill ORs new bits above the
                        // existing low bits), so the carried `pre` stays valid
                        // across the invariant-restoring refill below.
                        //
                        // Consume gate `bit_count <= available()`: near the end
                        // of input a decode can be backed by fabricated zero
                        // bits past the real data (slow-path refill couldn't
                        // fill); the gate ensures we only CONSUME packets fully
                        // backed by real bits. A gate-failed packet is carried
                        // un-consumed — the next iteration-top `in_ok` check
                        // (pos+8 < in_end fails whenever a slow-path refill
                        // ran) breaks to the careful loop, which re-decodes
                        // from the same cursor. Same contract as the existing
                        // bottom-of-loop preload.
                        let mut chained = 0usize;
                        loop {
                            let nxt = self.asm.lut_litlen.decode(&mut lb);
                            let ncode = (nxt.symbol & 0xFFFF) as u16;
                            if chained < LIT_CHAIN_MAX
                                && nxt.sym_count == 1
                                && nxt.bit_count != 0
                                && ncode <= 255
                                && nxt.bit_count <= lb.available()
                            {
                                lb.consume(nxt.bit_count);
                                // SAFETY: iteration-top guard `emitted +
                                // FAST_OUT_SLOP < local_cap` + the compile
                                // assert `1+LIT_CHAIN_MAX <= FAST_OUT_SLOP`
                                // keep every chained write < out_room < cap.
                                unsafe {
                                    base.add(*pos).write(ncode as u8);
                                }
                                *pos += 1;
                                emitted += 1;
                                chained += 1;
                                continue;
                            }
                            pre = nxt; // carry un-consumed
                            break;
                        }
                        // Restore the ≥48-bit iteration-top invariant for the
                        // carried packet (mirrors the wrapper's carry-site
                        // refills, resumable.rs:1345-1348, and the bottom
                        // preload's threshold refill).
                        if (lb.bitsleft as u8) < 48 {
                            lb.refill();
                        }
                        continue 'fast;
                    } else {
                        trailing_code = code;
                        have_trailing = true;
                        // P3.5 c2: issue the dist load now (see decl above).
                        if let Some(dt) = dist_tbl {
                            spec_dist = Some(dt.lookup(lb.bitbuf));
                        }
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
                }

                if have_trailing {
                    let code = trailing_code;
                    if code == END_OF_BLOCK_SYMBOL {
                        self.at_end_of_block = true;
                        commit_fast!(Ok(emitted));
                    }
                    if (code as u32) > MAX_LIT_LEN_SYM {
                        commit_fast!(Err(BlockError::InvalidHuffmanCode));
                    }
                    let length = (code as usize).wrapping_sub(254);
                    if length != 0 {
                        let distance = if let Some(dt) = dist_tbl {
                            // Single-lookup path (wrapper analog,
                            // resumable.rs:1394-1405): code + extra decoded
                            // from the already-peeked word, no mid-dist
                            // refill. Bit budget proof: every iteration top
                            // has >= 48 bits (bottom threshold refill);
                            // litlen consumed <= 20 (12-bit packed slots or
                            // <= 15-bit code + <= 5 packed length-extra), so
                            // >= 28 remain >= 9 main + 6 subtable + 13 extra.
                            use crate::decompress::inflate::libdeflate_entry::DistTable;
                            // P3.5 c2: use the early-issued entry when the
                            // lone-length arm already loaded it (identical
                            // index — no consumes between the sites).
                            let mut dist_entry = match spec_dist {
                                Some(e) => e,
                                None => dt.lookup(lb.bitbuf),
                            };
                            if dist_entry.is_subtable_ptr() {
                                lb.consume(DistTable::TABLE_BITS as u32);
                                dist_entry = dt.lookup_subtable_direct(dist_entry, lb.bitbuf);
                            }
                            let dist_raw = dist_entry.raw();
                            // raw == 0 is the unassigned/invalid-code marker
                            // (codes 30/31 and incomplete-code holes) — the
                            // exact set `dist_hc.decode` returns `None` for.
                            if dist_raw == 0 {
                                commit_fast!(Err(BlockError::InvalidHuffmanCode));
                            }
                            let dist_extra_saved = lb.bitbuf;
                            lb.consume_entry(dist_raw);
                            dist_entry.decode_distance(dist_extra_saved) as usize
                        } else {
                            // Fallback: vendor-faithful dist_hc walk (kept
                            // verbatim; unreachable in production — the LUT
                            // build always populates `dist_table` for
                            // Huffman blocks — but it keeps the loop sound
                            // if a future builder declines a block).
                            let dsym = match self.dist_hc.decode(&mut lb) {
                                Some(d) => d,
                                None => commit_fast!(Err(BlockError::InvalidHuffmanCode)),
                            };
                            if dsym as usize >= DISTANCE_BASE.len() {
                                commit_fast!(Err(BlockError::InvalidHuffmanCode));
                            }
                            let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                            let d = if extra > 0 {
                                if lb.available() < extra {
                                    lb.refill();
                                    if lb.available() < extra {
                                        commit_fast!(Err(BlockError::EndOfFile));
                                    }
                                }
                                let mask = (1u64 << extra) - 1;
                                let v = (lb.peek() & mask) as usize;
                                lb.consume(extra);
                                DISTANCE_BASE[dsym as usize] as usize + v
                            } else {
                                DISTANCE_BASE[dsym as usize] as usize
                            };
                            d
                        };
                        if distance == 0 || distance > MAX_WINDOW_SIZE {
                            commit_fast!(Err(BlockError::ExceededWindowRange));
                        }
                        // Clean-mode range check — mirror of the careful loop
                        // (deflate.hpp:1652-1655): valid iff `distance <= *pos`.
                        if distance > *pos {
                            commit_fast!(Err(BlockError::ExceededWindowRange));
                        }
                        // ── P3.5 c1: NEXT-SYMBOL PRELOAD BEFORE THE COPY ─────
                        // libdeflate decompress_template.h:555-572 — "Before
                        // starting to issue the instructions to copy the match,
                        // refill the bitbuffer and preload the litlen decode
                        // table entry for the next loop iteration … allowing
                        // the latency of the match copy to overlap with these
                        // other operations." The refill + LUT load read ONLY
                        // input-side state (`lb`, `lut_litlen`); the copy and
                        // the sparsity bookkeeping below touch ONLY the output
                        // buffer and `self` fields — disjoint, so hoisting the
                        // bottom-of-loop preload above the copy is byte-exact
                        // by construction (identical refill threshold,
                        // identical decode, identical bit consumption; only
                        // instruction SCHEDULING changes). The iteration ends
                        // with `continue 'fast` so the bottom preload (which
                        // now serves the literal-pack arm only) is not re-run.
                        if (lb.bitsleft as u8) < 48 {
                            lb.refill();
                        }
                        // P3.5 c4: threshold-refill site — backstop-free
                        // decode (byte-exact proof on `decode_prefilled`).
                        pre = self.asm.lut_litlen.decode_prefilled(&lb);
                        // SAFETY: `distance <= *pos`; `*pos + ((length+7)&!7) <= cap`
                        // (out_room reserved MAX_RUN_LENGTH + 8 headroom).
                        unsafe {
                            emit_backref_contig(base, pos, distance, length);
                        }
                        if track_backrefs {
                            // Inlined `record_backreference_for_sparsity`
                            // body (field-disjoint from the `dist_tbl`
                            // borrow; a `&mut self` method call would not be).
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
                        emitted += length;
                        // P3.5 c1: `pre` already preloaded above (before the
                        // copy) — skip the bottom-of-loop preload.
                        continue 'fast;
                    }
                }

                // Preload next symbol (pipeline). Refill first so the decode and
                // any subsequent dist-extra reads have headroom. Threshold-gated
                // (one iteration consumes <= 48 bits: litlen <= 20 incl. packed
                // length-extra, dist code <= 15, dist extra <= 13 — the same
                // budget as resumable.rs REFILL_THRESHOLD); the LUT decodes keep
                // their own `available() < 32` backstops, and refills never
                // change decode results, so gating is byte-transparent.
                if (lb.bitsleft as u8) < 48 {
                    lb.refill();
                }
                // P3.5 c4: threshold-refill site — backstop-free decode
                // (byte-exact proof on `decode_prefilled`).
                pre = self.asm.lut_litlen.decode_prefilled(&lb);
            }
                };
            }
            // Variant selection — once per region run (see macro doc above).
            #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
            {
                if asm_on {
                    fast_loop_run!(true);
                } else {
                    fast_loop_run!(false);
                }
            }
            #[cfg(not(all(feature = "asm-kernel", target_arch = "x86_64")))]
            fast_loop_run!(false);
            // FALL THROUGH: `pre` was decoded but NOT consumed (every break leaves
            // a fresh un-consumed `pre`), so the bit cursor sits exactly before
            // `pre`'s bits. The careful loop below re-decodes from here — no state
            // carried, no desync. `*pos`/`emitted`/`at_end_of_block` are all live.
            sync_local_bits!();
        }

        while emitted < local_cap {
            bits.refill();
            let (symbol, sym_count, bit_count) = self.lut_litlen_decode(bits);
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
                // Distance via the SAME single libdeflate-shape `asm.dist`
                // `DistTable` the fast loop uses (converges on igzip's single
                // inline distance table; lets clean blocks skip the redundant
                // `dist_hc` build). Byte-identical to the `dist_hc` walk
                // (`dist_table_matches_dist_hc_differential`). `dist_hc` stays
                // as the `!dist_valid` fallback (unreachable on a valid stream).
                let distance = if self.dist_valid {
                    use crate::decompress::inflate::libdeflate_entry::DistTable;
                    let dt = &self.asm.dist;
                    bits.refill();
                    let mut dist_entry = dt.lookup(bits.bitbuf);
                    if dist_entry.is_subtable_ptr() {
                        bits.consume(DistTable::TABLE_BITS as u32);
                        dist_entry = dt.lookup_subtable_direct(dist_entry, bits.bitbuf);
                    }
                    let dist_raw = dist_entry.raw();
                    // raw == 0 is the unassigned/invalid-code marker (codes
                    // 30/31 + incomplete-code holes) — the exact set
                    // `dist_hc.decode` returns `None` for.
                    if dist_raw == 0 {
                        commit!(Err(BlockError::InvalidHuffmanCode));
                    }
                    let dist_extra_saved = bits.bitbuf;
                    bits.consume_entry(dist_raw);
                    dist_entry.decode_distance(dist_extra_saved) as usize
                } else {
                    let dsym = match self.dist_hc.decode(bits) {
                        Some(d) => d,
                        None => commit!(Err(BlockError::InvalidHuffmanCode)),
                    };
                    if dsym as usize >= DISTANCE_BASE.len() {
                        commit!(Err(BlockError::InvalidHuffmanCode));
                    }
                    let extra = DISTANCE_EXTRA[dsym as usize] as u32;
                    if extra > 0 {
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
                    }
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
    /// # Safety
    /// `base` must be valid for `[0, cap)` with `*pos <= cap`; `bits.data` must
    /// not alias the destination. Marked `unsafe` for the raw `base` deref
    /// (clippy `not_unsafe_ptr_arg_deref`).
    pub unsafe fn decode_clean_stored_into_contig(
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
                #[cfg(test)]
                STORED_CONTIG_BULK_TL.with(|c| c.set(c.get() + 1));
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

// ELEMENT A: the process-wide `fixed_dist_table()` static is REMOVED — the
// dist table (fixed and dynamic) is now built INLINE into `self.asm.dist`
// (the asm addresses only the single `ctx` base). `FIXED_DIST_LENGTHS` is
// still the source of the fixed lens fed to `DistTable::rebuild` in
// `ensure_dist_table`.

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
///
/// STAGE-c (perf/window-absent-inflate): forced `inline(always)` — the plain
/// `#[inline]` hint was being DECLINED by LLVM (the function stayed a standalone
/// symbol carrying ~18% of gz's retired instructions, with a per-backref
/// call prologue/epilogue rg does not pay because rg fuses the same copy into
/// `Block<false>::read`). The wall gap vs rg is an instruction-count gap (gz
/// 1.32× rg instructions at HIGHER IPC), so eliminating the per-backref call/
/// spill/return is an instruction-removal lever. Byte-exact: codegen-only.
#[inline(always)]
pub(crate) unsafe fn emit_backref_ring<const CONTAINS_MARKERS: bool>(
    ring_ptr: *mut u16,
    pos: &mut usize,
    drained: usize,
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
        //     `*pos`, and subsequent literals/copies overwrite those slots
        //     BEFORE the next drain — guaranteed by the `span_round_fits`
        //     guard below. The marker backward-scan below reads only the
        //     `length` bytes behind the new `*pos`.
        // `distance < 4` non-overlap (only length-3/distance-3) is rare and
        // would alias the word stride, so it keeps the exact element copy.
        //
        // P0 STORED-MARKER-CRC FIX (2026-06-12, mono-gnu9 single-byte
        // corruption at output 35,335,338): the old "subsequent writes
        // overwrite the overshoot" invariant is FALSE for the FINAL back-ref
        // of a maximally-full `read()` call. The per-call cap is `RING_SIZE -
        // MAX_RUN_LENGTH` but the final event may overshoot it by up to
        // `MAX_RUN_LENGTH - 1`, making the undrained span up to RING_SIZE-1
        // slots. The ≤3-slot rounding overshoot then wraps PHYSICALLY onto
        // the OLDEST UNDRAINED slot(s) (logical `pos + length + k - RING_SIZE
        // >= drained`), which the end-of-call drain ships to the caller —
        // one clobbered u16 in `data_with_markers`. Vendor never overshoots
        // (exact `memcpy(length * 2)`, deflate.hpp:1376); gzippy keeps the
        // word-copy DEVIATION but only when the rounded run also fits within
        // the undrained window: `(*pos - drained) + rounded <= RING_SIZE`.
        // Otherwise fall to the exact-length arms below (byte-identical).
        // 16-byte (8-u16) WIDE path (2026-06-16): the measured back-ref length
        // distribution is mean ≈ 6.3 u16, 98.6% < 16, 77% length 4-7 — so a
        // SINGLE 16-byte unaligned store (`movups` on x86_64 SSE2, one `str q`
        // on aarch64) finalises length ≤ 8 in ONE store where the 8-byte arm
        // below needs two. This ~halves backref store instructions on the
        // window-absent path. REQUIRES `distance >= 8` u16 so the 16-byte
        // stride never aliases (src and dst are ≥ 16 bytes apart): each chunk
        // read sees only ALREADY-FINALISED bytes — the exact `dist >= WORDBYTES`
        // invariant `copy_match_fast` relies on, now at u128 width. The
        // overshoot (≤ 7 u16) is bounded by the SAME three guards as the 8-byte
        // arm, just rounded to a multiple of 8 u16: it stays inside the
        // physical buffer AND inside the undrained window (the mono-gnu9 P0
        // contract — see below), else we fall to the 8-byte / exact arms which
        // are byte-identical to vendor's `memcpy(length*2)`.
        let rounded16 = (length + 7) & !7;
        let src_r16_fits = src_phys + rounded16 <= RING_SIZE;
        let dst_r16_fits = dst_phys + rounded16 <= RING_SIZE;
        let span_r16_fits = (*pos - drained) + rounded16 <= RING_SIZE;
        let rounded = (length + 3) & !3;
        let src_round_fits = src_phys + rounded <= RING_SIZE;
        let dst_round_fits = dst_phys + rounded <= RING_SIZE;
        let span_round_fits = (*pos - drained) + rounded <= RING_SIZE;
        if distance >= 8 && src_r16_fits && dst_r16_fits && span_r16_fits {
            // One 16-byte store covers length ≤ 8; the loop handles the rare
            // long tail (≥ 9 u16). `u128` write_unaligned lowers to a single
            // unaligned 128-bit vector store on x86_64 (SSE2 baseline) and
            // aarch64 — portable, no arch-gated intrinsic needed.
            let s = ring_ptr.add(src_phys) as *const u8;
            let d = ring_ptr.add(dst_phys) as *mut u8;
            (d as *mut u128).write_unaligned((s as *const u128).read_unaligned());
            if length > 8 {
                let mut s = s.add(16);
                let mut d = d.add(16);
                let dend = (ring_ptr.add(dst_phys + rounded16)) as *mut u8;
                while d < dend {
                    (d as *mut u128).write_unaligned((s as *const u128).read_unaligned());
                    s = s.add(16);
                    d = d.add(16);
                }
            }
        } else if distance >= 4 && src_round_fits && dst_round_fits && span_round_fits {
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
            // Branchless backward marker scan — PROVABLY EQUIVALENT to the
            // scalar `slice.iter().rev()` it replaces. A u16 is a marker iff its
            // high bit is set (`v >= MARKER_U16`, and `MARKER_U16 == 0x8000` is
            // exactly the u16 sign bit), so 4 contiguous u16 pack into one
            // unaligned u64 whose per-lane high bits are isolated by the mask
            // `0x8000_8000_8000_8000`. We walk the just-written region from the
            // HIGH (latest) end in 4-u16 (u64) strides; the FIRST stride with any
            // high bit set holds the LAST (highest-index) marker — `from_le`
            // normalises lane order so lane 0 is the lowest index, and
            // `63 - leading_zeros()` picks the highest set lane (= highest index
            // = the marker the rev-scan would have stopped on). `distance_marker`
            // is then `length - 1 - global_index` (matching the scalar `k`, since
            // there `index == length - 1 - k`); the sub-4 low-end remainder and
            // the no-marker `+= length` fall through exactly as before.
            //
            // This collapses a chain of up-to-`length` dependent single-u16
            // loads/compares into one u64 load + mask + lzcnt per 4 elements. AVX2
            // (`_mm256`) was deliberately NOT used: 98.6% of back-refs have
            // length < 16 (measured distribution above) and the scan early-exits,
            // so vector setup + GPR-transfer cost would dominate (external review
            // concurred) — a GPR-width bitmask is the right granularity here.
            const MARKER_LANES: u64 = 0x8000_8000_8000_8000;
            let base = ring_ptr.add(dst_phys);
            let mut hi = length;
            while hi >= 4 {
                let chunk = hi - 4;
                let w = u64::from_le((base.add(chunk) as *const u64).read_unaligned());
                let m = w & MARKER_LANES;
                if m != 0 {
                    let lane = (63 - m.leading_zeros()) as usize / 16;
                    *distance_marker = length - 1 - (chunk + lane);
                    return;
                }
                hi = chunk;
            }
            // Low-end remainder (< 4 u16), scanned high → low.
            while hi > 0 {
                hi -= 1;
                if *base.add(hi) >= MARKER_U16 {
                    *distance_marker = length - 1 - hi;
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
    // P3.4 item 2 (backref-arm polish): libdeflate fastloop copy shape, ported
    // from `copy_match_fast` (consume_first_decode.rs:479-553, itself the
    // libdeflate decompress_template.h match copy). Three arms by DISTANCE
    // only — the old `distance >= length` non-overlap pre-branch is gone:
    //
    //  * `distance >= 8`: unconditional 5-word (40-byte) burst, then a stride-8
    //    word loop. Correct for ANY length INCLUDING overlap (`length >
    //    distance`): every 8-byte load reads bytes that sit >= 8 positions
    //    behind the store frontier, hence already FINAL. (The old shape sent
    //    dist>=8 overlap — e.g. dist=10,len=100 — to a PER-BYTE loop.)
    //  * `distance == 1`: RLE broadcast-word fill (vendor memset arm,
    //    deflate.hpp:1393-1398, as words instead of a libc call).
    //  * `2 <= distance <= 7`: stride-`distance` word trick (libdeflate's
    //    small-offset arm): each 8-byte store advances by `distance`; its
    //    first `distance` bytes are correct (loaded from the final region one
    //    period behind) and its garbage tail is overwritten by the NEXT store;
    //    the final store's first `distance` bytes reach `end`. The old shape
    //    per-byte-copied every small-distance backref.
    //
    // OVERSHOOT ENVELOPE (headroom proof dependency): worst touch is
    // `*pos + max(40, ((3*7)+8), length + 7) = *pos + max(40, length+7)
    // <= *pos + 265` — strictly inside the `MAX_RUN_LENGTH + 8 = 266` byte
    // reservation BOTH contig loops already hold below `cap` (the same
    // envelope the old word path used: `*pos + 264`). Overshoot bytes lie
    // ahead of the advanced `*pos`, are never sourced by a later back-ref,
    // and get overwritten — invisible. Byte-exactness of `[*pos, *pos+length)`
    // is locked by the permanent `emit_backref_contig_differential` test
    // (dist 1..258 x len 3..258 x alignments vs a sequential-copy reference).
    let mut dst = base.add(*pos);
    let mut src = base.add(*pos - distance);
    let end = dst.add(length);
    // P3.4 item 3 (variant B): long matches walk well past the head of the
    // source — prefetch one line ahead before the copy loop reaches it
    // (libdeflate analog, copy_match_fast:430-433).
    #[cfg(target_arch = "x86_64")]
    if length > 40 {
        std::arch::x86_64::_mm_prefetch(src.add(40) as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    if distance >= 8 {
        // 5-word unconditional burst (covers the common short match without a
        // length branch), then the stride-8 loop for long matches.
        (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
        (dst.add(8) as *mut u64).write_unaligned((src.add(8) as *const u64).read_unaligned());
        (dst.add(16) as *mut u64).write_unaligned((src.add(16) as *const u64).read_unaligned());
        (dst.add(24) as *mut u64).write_unaligned((src.add(24) as *const u64).read_unaligned());
        (dst.add(32) as *mut u64).write_unaligned((src.add(32) as *const u64).read_unaligned());
        if length > 40 {
            src = src.add(40);
            dst = dst.add(40);
            while dst < end {
                (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                src = src.add(8);
                dst = dst.add(8);
            }
        }
    } else if distance == 1 {
        // RLE: broadcast the byte into a word and store words to `end`
        // (<= 7-byte overshoot).
        let v = 0x0101_0101_0101_0101u64.wrapping_mul(*src as u64);
        while dst < end {
            (dst as *mut u64).write_unaligned(v);
            dst = dst.add(8);
        }
    } else {
        // 2 <= distance <= 7: stride-`distance` word stores. 4 unconditional
        // (touch <= dst + 3*7 + 8 = dst+29), then loop to `end`.
        (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
        src = src.add(distance);
        dst = dst.add(distance);
        (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
        src = src.add(distance);
        dst = dst.add(distance);
        (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
        src = src.add(distance);
        dst = dst.add(distance);
        (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
        src = src.add(distance);
        dst = dst.add(distance);
        while dst < end {
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(distance);
            dst = dst.add(distance);
        }
    }
    *pos += length;
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
    drained: usize,
    distance: usize,
    length: usize,
) {
    let src_phys = (*pos + U8_RING_SIZE - distance) % U8_RING_SIZE;
    let dst_phys = *pos % U8_RING_SIZE;

    if distance >= length {
        // Non-overlap. Vendor: `memcpy(&window[wp], &window[off], length)`
        // (deflate.hpp:1376, sizeof==1). Word = 8 u8; round run up to 8.
        //
        // P0 STORED-MARKER-CRC FIX (2026-06-12): same undrained-span guard as
        // `emit_backref_ring` (see the vendor-cited comment there). The u8
        // twin has the identical latent clobber: a maximally-full `read()`
        // call (cap `U8_RING_SIZE - MAX_RUN_LENGTH`, final event overshoots
        // by up to 257) leaves an undrained span of up to U8_RING_SIZE-1
        // bytes; the ≤7-byte rounding overshoot then wraps onto the oldest
        // undrained byte(s), which the end-of-call drain ships to the caller.
        let rounded = (length + 7) & !7;
        let src_round_fits = src_phys + rounded <= U8_RING_SIZE;
        let dst_round_fits = dst_phys + rounded <= U8_RING_SIZE;
        let span_round_fits = (*pos - drained) + rounded <= U8_RING_SIZE;
        if distance >= 8 && src_round_fits && dst_round_fits && span_round_fits {
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

    /// ONE shared lock for ALL same-binary kill-switch differential tests
    /// (`marker_dist_lut_diff`, `mfast_local_bits_diff`, `stored_flip`). Each
    /// of those tests flips a PROCESS-WIDE override atomic
    /// (`MARKER_DIST_LUT_OVERRIDE` / `MFAST_LOCALBITS_OVERRIDE` /
    /// `STORED_FLIP_OVERRIDE`) and then compares two decode arms expecting
    /// byte- AND cursor-identical results. Per-module locks were NOT enough:
    /// while test A held its own lock comparing its two arms, test B (a
    /// different module, different lock) could flip a DIFFERENT global override
    /// mid-comparison, so A's two arms decoded under different background-switch
    /// state and diverged (observed as a spurious "cursor/state diverged" on
    /// the rand/stored case under parallel test scheduling — never in
    /// isolation). A single shared lock makes the three mutually exclusive so no
    /// override is mutated inside another's comparison window. (Non-differential
    /// tests only READ the overrides at their -1 default, so they cannot
    /// corrupt a comparison and need not take this lock.)
    pub(super) static DIFFERENTIAL_OVERRIDE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        decode_contig_clean_with_state(deflate_bytes, window, cap, per_call).0
    }

    /// `decode_contig_clean` + MULTI-BLOCK support (loops `read_header`
    /// until BFINAL, routes stored blocks through
    /// `decode_clean_stored_into_contig` — the `contig_multi` driver shape)
    /// + the FINAL bit-cursor state `(pos, bitbuf, bitsleft)`. The asm-ON/OFF
    /// fuzz net (flip-precondition 3) asserts state equality, not just byte
    /// equality. Single-block streams behave exactly as the old single-header
    /// driver (the loop exits after the first block's EOB).
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    fn decode_contig_clean_with_state(
        deflate_bytes: &[u8],
        window: &[u8],
        cap: usize,
        per_call: usize,
    ) -> (Vec<u8>, (usize, u64, u32)) {
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
        loop {
            b.read_header(&mut bits, false).unwrap();
            while !b.eob() {
                let before = pos;
                // SAFETY: test-local `base`/`cap` from a contiguous buffer.
                let n = if b.compression_type() == CompressionType::Uncompressed {
                    unsafe {
                        b.decode_clean_stored_into_contig(&mut bits, base, cap, &mut pos, per_call)
                    }
                    .unwrap()
                } else {
                    unsafe { b.decode_clean_into_contig(&mut bits, base, cap, &mut pos, per_call) }
                        .unwrap()
                };
                assert_eq!(n, pos - before, "contig: emitted != pos delta");
                if n == 0 && !b.eob() {
                    panic!("contig decode stalled before EOB (cap too small?)");
                }
            }
            if b.is_last_block() {
                break;
            }
        }
        (
            buf[window.len()..pos].to_vec(),
            (bits.pos, bits.bitbuf, bits.bitsleft),
        )
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

    /// Flip-precondition 3 (campaign §9 gate): the PERMANENT asm-ON-vs-OFF
    /// fuzz differential over random VALID GZIP MEMBERS — the stream-level
    /// net binding the asm kernel and the pure-Rust loop forever (charter
    /// §4: "bound by the differential suite FOREVER").
    ///
    /// Seeded + bounded: 96 members, payloads 1 KiB–128 KiB of mixed
    /// segments (uniform random → stored-prone; skewed random → dynamic
    /// all-literal; text → backrefs; RLE runs → dist==1; self-copies →
    /// long distances; zeros), flate2-gzip'd at levels {0 (stored), 1, 6, 9}
    /// with occasional mid-stream sync flushes (extra stored-block
    /// boundaries). Each member is validity-checked by a flate2 round-trip,
    /// then its DEFLATE body is decoded twice through the production
    /// contig-clean path — kernel force-ENABLED vs force-DISABLED (same
    /// binary; the in-process `TEST_FORCE` override stands in for the
    /// process-wide `enabled()` OnceLock) — asserting byte equality,
    /// equality to the payload, and final bit-cursor equality
    /// (pos/bitbuf/bitsleft — the X1 contract at stream end). ON-arm
    /// engagement is effect-verified via `TEST_RUN_CONTIG_CALLS`, never
    /// assumed. Skips gracefully without BMI2 (local Rosetta); the guest
    /// run is authoritative.
    #[cfg(all(pure_inflate_decode, feature = "asm-kernel", target_arch = "x86_64"))]
    #[test]
    fn asm_kernel_on_off_fuzz_random_gzip_members() {
        use crate::decompress::parallel::asm_kernel::{TEST_FORCE, TEST_RUN_CONTIG_CALLS};
        use std::io::{Read, Write};
        use std::sync::atomic::Ordering;
        if !std::arch::is_x86_feature_detected!("bmi2") {
            eprintln!("SKIP asm on/off fuzz: no BMI2 on this host (run on guest)");
            return;
        }
        // Restore production dispatch semantics for the rest of the test
        // process on every exit path (including assert panics).
        struct ForceGuard;
        impl Drop for ForceGuard {
            fn drop(&mut self) {
                TEST_FORCE.store(0, Ordering::Relaxed);
            }
        }
        let _guard = ForceGuard;

        let mut x: u64 = 0x5851_F42D_4C95_7F2D;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        };

        const TRIALS: usize = 96;
        let mut on_hits_total = 0u64;
        let mut huffman_trials = 0u64;
        for trial in 0..TRIALS {
            // ── payload: mixed segments until the target size ──
            let target = 1usize << (10 + (next() as usize % 8)); // 1 KiB..128 KiB
            let mut payload: Vec<u8> = Vec::with_capacity(target + 4096);
            while payload.len() < target {
                match next() % 6 {
                    0 => {
                        // uniform random (stored-prone at every level)
                        let n = 64 + (next() as usize % 4096);
                        for _ in 0..n {
                            payload.push(next() as u8);
                        }
                    }
                    1 => {
                        // skewed random (~6-bit entropy → dynamic all-literal)
                        let n = 64 + (next() as usize % 4096);
                        for _ in 0..n {
                            payload.push((next() as u8) & (next() as u8));
                        }
                    }
                    2 => {
                        // repeated text (in-block backrefs, length variety)
                        let reps = 4 + (next() as usize % 200);
                        payload
                            .extend(b"the quick brown fox jumps over the lazy dog. ".repeat(reps));
                    }
                    3 => {
                        // RLE run (dist==1 broadcast arm)
                        let b = next() as u8;
                        let n = 16 + (next() as usize % 2048);
                        payload.extend(std::iter::repeat(b).take(n));
                    }
                    4 => {
                        // self-copy of an earlier slice (long distances)
                        if payload.is_empty() {
                            payload.extend_from_slice(b"seed material for copies ");
                        } else {
                            let start = next() as usize % payload.len();
                            let len = (1 + (next() as usize % 8192)).min(payload.len() - start);
                            let slice = payload[start..start + len].to_vec();
                            payload.extend_from_slice(&slice);
                        }
                    }
                    _ => {
                        // zeros (max-length RLE backrefs)
                        let n = 16 + (next() as usize % 2048);
                        payload.extend(std::iter::repeat(0u8).take(n));
                    }
                }
            }
            // ── a VALID gzip member: mixed levels; occasional sync flush ──
            let level = match next() % 4 {
                0 => 0u32,
                1 => 1,
                2 => 6,
                _ => 9,
            };
            if level > 0 {
                huffman_trials += 1;
            }
            let mut enc =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
            if payload.len() > 8192 && next() % 2 == 0 {
                let cut = payload.len() / 2;
                enc.write_all(&payload[..cut]).unwrap();
                enc.flush().unwrap(); // sync flush: extra block boundary
                enc.write_all(&payload[cut..]).unwrap();
            } else {
                enc.write_all(&payload).unwrap();
            }
            let gz = enc.finish().unwrap();
            // gzip-member validity oracle: flate2's own decoder round-trips.
            let mut rt = Vec::new();
            flate2::read::GzDecoder::new(&gz[..])
                .read_to_end(&mut rt)
                .unwrap();
            assert_eq!(rt, payload, "trial {trial}: flate2 gzip round-trip");
            // The member's raw DEFLATE body (header is 10 bytes when FLG==0
            // — flate2 sets no name/extra/comment; trailer is CRC32+ISIZE).
            assert_eq!(&gz[..2], b"\x1f\x8b", "trial {trial}: gzip magic");
            assert_eq!(gz[3], 0, "trial {trial}: FLG != 0 (10-byte header)");
            let deflate = &gz[10..gz.len() - 8];

            let cap = payload.len() + 1024;
            let per_call = match next() % 3 {
                0 => cap,
                1 => 4096,
                _ => 65536,
            };

            // ── ON arm (engagement effect-verified) ──
            TEST_FORCE.store(2, Ordering::Relaxed);
            let h0 = TEST_RUN_CONTIG_CALLS.load(Ordering::Relaxed);
            let (on_bytes, on_state) = decode_contig_clean_with_state(deflate, &[], cap, per_call);
            on_hits_total += TEST_RUN_CONTIG_CALLS.load(Ordering::Relaxed) - h0;
            // ── OFF arm: every dispatch inside this decode reads the
            //    override, so the pure-Rust loop is the sole path ──
            TEST_FORCE.store(1, Ordering::Relaxed);
            let (off_bytes, off_state) =
                decode_contig_clean_with_state(deflate, &[], cap, per_call);
            TEST_FORCE.store(0, Ordering::Relaxed);

            assert_eq!(
                on_bytes, payload,
                "trial {trial}: ON output != payload (level {level})"
            );
            assert_eq!(
                off_bytes, payload,
                "trial {trial}: OFF output != payload (level {level})"
            );
            assert_eq!(
                on_state, off_state,
                "trial {trial}: final (pos, bitbuf, bitsleft) diverged ON vs OFF"
            );
        }
        // Engagement floor: Huffman-bearing members must drive the kernel.
        // (A level>0 member CAN still be all-stored if its payload segments
        // came out incompressible, so the floor is half the Huffman trials,
        // not all of them; seed-stable either way.)
        assert!(
            huffman_trials > 0,
            "seed produced no Huffman members — generator broken"
        );
        assert!(
            on_hits_total >= huffman_trials / 2,
            "asm engagement too low: {on_hits_total} run_contig calls across \
             {huffman_trials} Huffman members — ON arm not exercising the kernel"
        );
        eprintln!(
            "[fuzz] {TRIALS} gzip members ON==OFF==payload; \
             run_contig calls in ON windows: {on_hits_total} \
             ({huffman_trials} Huffman members)"
        );
    }

    /// P3.2 stream-level differential for the runtime literal chain: an
    /// incompressible (LCG) payload makes flate2 emit ~all-literal dynamic
    /// blocks with 8-9-bit literal codes — the regime where the ISA-L LUT's
    /// build-time multi-symbol packing ~never fires (pairs need ≤12 combined
    /// bits) and the lit1-arm runtime chain fires constantly. The text tail
    /// mixes in back-refs (chain→length-packet carries) and the small
    /// `per_call` run splits chains across resumable re-entries.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[test]
    fn contig_clean_matches_ring_clean_literal_heavy_chain() {
        // SKEWED random bytes: non-uniform distribution (AND of two LCG
        // bytes → ~6-bit entropy) with no repetition, so flate2 picks
        // dynamic-Huffman all-literal blocks (NOT stored — uniform random
        // would be stored and never exercise the Huffman loop). The payload
        // is large enough (~250 KiB) to span MULTIPLE deflate blocks, so
        // these drivers (unlike the single-block helpers above) loop
        // read_header until BFINAL.
        fn lcg_skewed(seed: u64, n: usize) -> Vec<u8> {
            let mut s = seed;
            let mut v = Vec::with_capacity(n);
            let next = |s: &mut u64| {
                *s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (*s >> 56) as u8
            };
            for _ in 0..n {
                let b = next(&mut s) & next(&mut s);
                v.push(b);
            }
            v
        }
        let window: Vec<u8> = (0u32..32768)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        let mut payload = lcg_skewed(0xfeed_beef_cafe_f00d, 200_000);
        payload.extend_from_slice(&b"the quick brown fox jumps over the lazy dog. ".repeat(1000));
        payload.extend_from_slice(&window[..2048]); // dict back-ref tail
        let deflate_bytes = deflate_with_dictionary(&payload, &window);

        fn ring_multi(deflate_bytes: &[u8], window: &[u8], n_max: usize) -> Vec<u8> {
            let boxed: &'static [u8] = Box::leak(deflate_bytes.to_vec().into_boxed_slice());
            let mut bits = Bits::new(boxed);
            let mut b = Block::new();
            let mut output: Vec<u16> = Vec::new();
            b.set_initial_window(&mut output, window).unwrap();
            loop {
                b.read_header(&mut bits, false).unwrap();
                while !b.eob() {
                    let n = b.read(&mut bits, &mut output, n_max).unwrap();
                    if n == 0 && !b.eob() {
                        panic!("ring decode stalled before EOB");
                    }
                }
                if b.is_last_block() {
                    break;
                }
            }
            output
                .iter()
                .map(|&v| {
                    assert!(v < 256, "clean decode must not emit markers");
                    v as u8
                })
                .collect()
        }

        fn contig_multi(
            deflate_bytes: &[u8],
            window: &[u8],
            cap: usize,
            per_call: usize,
        ) -> Vec<u8> {
            let boxed: &'static [u8] = Box::leak(deflate_bytes.to_vec().into_boxed_slice());
            let mut bits = Bits::new(boxed);
            let mut b = Block::new();
            let mut sink: Vec<u16> = Vec::new();
            b.set_initial_window(&mut sink, window).unwrap();
            let mut buf = vec![0u8; cap];
            buf[..window.len()].copy_from_slice(window);
            let base = buf.as_mut_ptr();
            let mut pos = window.len();
            loop {
                b.read_header(&mut bits, false).unwrap();
                while !b.eob() {
                    let before = pos;
                    // SAFETY: test-local `base`/`cap` from a contiguous buffer.
                    let n = if b.compression_type() == CompressionType::Uncompressed {
                        unsafe {
                            b.decode_clean_stored_into_contig(
                                &mut bits, base, cap, &mut pos, per_call,
                            )
                        }
                        .unwrap()
                    } else {
                        unsafe {
                            b.decode_clean_into_contig(&mut bits, base, cap, &mut pos, per_call)
                        }
                        .unwrap()
                    };
                    assert_eq!(n, pos - before, "contig: emitted != pos delta");
                    if n == 0 && !b.eob() {
                        panic!("contig decode stalled before EOB");
                    }
                }
                if b.is_last_block() {
                    break;
                }
            }
            buf[window.len()..pos].to_vec()
        }

        let cap = window.len() + payload.len() + 512;
        let ring = ring_multi(&deflate_bytes, &window, payload.len() + 16);
        assert_eq!(ring.len(), payload.len(), "ring decoded wrong length");
        assert_eq!(ring, payload, "ring decode != payload");
        let contig_full = contig_multi(&deflate_bytes, &window, cap, payload.len() + 16);
        assert_eq!(contig_full, ring, "literal-heavy contig (full) diverged");
        // Odd per-call cap → ~160 re-entries, each cutting a chain at an
        // arbitrary point (the carried packet is re-decoded on re-entry).
        let contig_split = contig_multi(&deflate_bytes, &window, cap, 1537);
        assert_eq!(contig_split, ring, "literal-heavy contig (split) diverged");
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
                block.ring.drained,
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

    /// P0 REGRESSION (2026-06-12, /tmp/mono-gnu9.tar.gz single-byte CRC
    /// corruption at output 35,335,338): direct contract test for the
    /// undrained-span guard in `emit_backref_ring`'s word-copy arm.
    ///
    /// Production worst case: a `read()` call's per-call cap is `RING_SIZE -
    /// MAX_RUN_LENGTH` (65278) but the FINAL back-ref of the call may
    /// overshoot it by up to 257, leaving an undrained span of up to
    /// RING_SIZE - 1 slots. The word copy rounds the run up to a multiple of
    /// 4 u16; the ≤3-slot rounding overshoot then wraps PHYSICALLY onto the
    /// OLDEST UNDRAINED slot, which the end-of-call drain ships to
    /// `data_with_markers` — one clobbered output u16. Vendor rapidgzip
    /// never overshoots (`std::memcpy(..., length * 2)`, deflate.hpp:1376).
    ///
    /// Geometry mirror of the mono-gnu9 failure: span before the final
    /// back-ref = 65277 (= cap - 1), final back-ref length 258, distance
    /// 1000 ⇒ rounded = 260, span + rounded = 65537 > RING_SIZE ⇒ the
    /// overshoot's last slot aliases the first undrained element.
    #[test]
    fn emit_backref_ring_overshoot_must_not_clobber_undrained() {
        let mut block = Block::new();
        // Deterministic fill so source reads are defined.
        for s in 0..RING_SIZE {
            block.ring.window16[s] = (s % 251) as u16;
        }
        let drained = 100_000usize; // arbitrary logical drain frontier
        let pos0 = drained + 65_277; // undrained span = cap - 1
        let oldest_slot = drained % RING_SIZE; // slot of the FIRST undrained element
        let sentinel = 0x002Eu16; // '.' — the byte the mono-gnu9 clobber destroyed
        block.ring.window16[oldest_slot] = sentinel;

        let ring_ptr = block.ring.window16.as_mut_ptr();
        let mut pos = pos0;
        let mut dm = 0usize;
        unsafe {
            emit_backref_ring::<true>(ring_ptr, &mut pos, drained, 1000, 258, &mut dm);
        }
        assert_eq!(pos, pos0 + 258);
        // The copy itself must be exact (vendor memcpy semantics).
        for k in 0..258usize {
            let dst = block.ring.window16[(pos0 + k) % RING_SIZE];
            let src_logical_slot = (pos0 + k - 1000) % RING_SIZE;
            assert_eq!(
                dst,
                (src_logical_slot % 251) as u16,
                "byte {k} of the back-ref copy is wrong"
            );
        }
        // THE P0 ASSERTION: the rounding overshoot must NOT touch the oldest
        // undrained slot (pre-fix it wrote the u16 at src+259 here — the 'L').
        assert_eq!(
            block.ring.window16[oldest_slot], sentinel,
            "word-copy rounding overshoot clobbered the oldest undrained ring slot \
             (the mono-gnu9 P0 single-byte corruption)"
        );
    }

    /// u8 (clean-mode) twin of the test above — `emit_backref_ring_u8` has the
    /// identical latent clobber (8-byte rounding, cap `U8_RING_SIZE -
    /// MAX_RUN_LENGTH`, ≤7-byte overshoot onto the oldest undrained bytes).
    #[test]
    fn emit_backref_ring_u8_overshoot_must_not_clobber_undrained() {
        let mut block = Block::new();
        let ring8_len = U8_RING_SIZE;
        // Fill the u8 view deterministically.
        {
            let ring8 = block.ring.window16.as_mut_ptr() as *mut u8;
            for s in 0..ring8_len {
                unsafe { ring8.add(s).write((s % 249) as u8) };
            }
        }
        let drained = 200_000usize;
        let pos0 = drained + 130_813; // undrained span = cap - 1 (U8 cap = 130814)
        let oldest = drained % U8_RING_SIZE;
        let ring8 = block.ring.window16.as_mut_ptr() as *mut u8;
        // Sentinels across the up-to-5 clobberable bytes.
        let sentinels: Vec<u8> = (0..5).map(|k| 0xA0 + k as u8).collect();
        for (k, &v) in sentinels.iter().enumerate() {
            unsafe { ring8.add((oldest + k) % U8_RING_SIZE).write(v) };
        }
        let mut pos = pos0;
        unsafe {
            emit_backref_ring_u8(ring8, &mut pos, drained, 1000, 258);
        }
        assert_eq!(pos, pos0 + 258);
        for (k, &v) in sentinels.iter().enumerate() {
            let got = unsafe { *ring8.add((oldest + k) % U8_RING_SIZE) };
            assert_eq!(
                got, v,
                "u8 word-copy rounding overshoot clobbered undrained byte {k} \
                 (clean-mode twin of the mono-gnu9 P0)"
            );
        }
    }

    /// DIFFERENTIAL PROPTEST for the wide (16-byte / u128) back-ref copy
    /// (2026-06-16). Drives `emit_backref_ring` directly across a broad random
    /// space of (ring contents, pos, drained, distance, length, prior
    /// distance_marker) and asserts byte-for-byte equality of the `length`
    /// copied slots AND the recomputed marker counter against a NAIVE scalar
    /// reference (sequential per-element ring copy = exact deflate semantics).
    /// Covers the new `distance >= 8` wide path, the 8-byte arm, distance
    /// 1/2/3 overlap + RLE, wrap-straddle, and the undrained overshoot guard.
    ///
    /// The ring is filled with FULL-range u16 (≈50% are >= MARKER_U16) so the
    /// backward marker scan and the `distance_marker >= distance` fast path are
    /// both exercised. `drained` is chosen behind `pos` with random headroom so
    /// both the overshoot-fits (wide/word) and overshoot-doesn't-fit (exact
    /// fallback) branches are hit.
    fn naive_backref_ref(
        ring: &[u16],
        pos: usize,
        distance: usize,
        length: usize,
        dm_in: usize,
    ) -> (Vec<u16>, usize) {
        const MARKER_U16: u16 = MAX_WINDOW_SIZE as u16;
        let mut r = ring.to_vec();
        let src_phys = (pos + RING_SIZE - distance) % RING_SIZE;
        let dst_phys = pos % RING_SIZE;
        for i in 0..length {
            let v = r[(src_phys + i) % RING_SIZE];
            r[(dst_phys + i) % RING_SIZE] = v;
        }
        // Counter: mirror emit_backref_ring's CONTAINS_MARKERS arm exactly.
        let dm_out = if dm_in >= distance {
            dm_in + length
        } else {
            let mut found = dm_in + length;
            for k in 0..length {
                let v = r[(pos + length - 1 - k) % RING_SIZE];
                if v >= MARKER_U16 {
                    found = k;
                    break;
                }
            }
            found
        };
        (r, dm_out)
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 4000,
            .. proptest::prelude::ProptestConfig::default()
        })]

        #[test]
        fn emit_backref_ring_wide_copy_matches_naive_reference(
            seed in proptest::prelude::any::<u64>(),
            pos in 0usize..RING_SIZE,
            headroom in 0usize..RING_SIZE,
            distance_raw in 1usize..=MAX_WINDOW_SIZE,
            length in 1usize..=258usize,
            dm_in in 0usize..(2 * MAX_WINDOW_SIZE),
        ) {
            // distance must be <= pos (a back-ref can't reach before the
            // stream start) for the logical ring math to be meaningful here;
            // clamp into [1, max(1, min(pos, MAX_WINDOW_SIZE))].
            let dmax = pos.min(MAX_WINDOW_SIZE).max(1);
            let distance = (distance_raw % dmax).max(1);
            let drained = pos.saturating_sub(headroom);

            // Random full-range ring (≈50% markers) from a cheap LCG seed.
            let mut ring = vec![0u16; RING_SIZE];
            let mut rng = seed | 1;
            for v in ring.iter_mut() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = (rng >> 33) as u16;
            }

            let (ref_ring, ref_dm) = naive_backref_ref(&ring, pos, distance, length, dm_in);

            // ── CONTAINS_MARKERS = true ──
            {
                let mut work = ring.clone();
                let mut p = pos;
                let mut dm = dm_in;
                unsafe {
                    emit_backref_ring::<true>(
                        work.as_mut_ptr(), &mut p, drained, distance, length, &mut dm,
                    );
                }
                proptest::prop_assert_eq!(p, pos + length, "pos advance");
                proptest::prop_assert_eq!(dm, ref_dm, "distance_marker counter");
                let dst_phys = pos % RING_SIZE;
                for i in 0..length {
                    let slot = (dst_phys + i) % RING_SIZE;
                    proptest::prop_assert_eq!(
                        work[slot], ref_ring[slot],
                        "copied slot {} (i={}) dist={} len={} pos={} drained={}",
                        slot, i, distance, length, pos, drained
                    );
                }
            }

            // ── CONTAINS_MARKERS = false (counter compile-eliminated; copy identical) ──
            {
                let mut work = ring.clone();
                let mut p = pos;
                let mut dm = dm_in;
                unsafe {
                    emit_backref_ring::<false>(
                        work.as_mut_ptr(), &mut p, drained, distance, length, &mut dm,
                    );
                }
                proptest::prop_assert_eq!(p, pos + length, "pos advance (false arm)");
                let dst_phys = pos % RING_SIZE;
                for i in 0..length {
                    let slot = (dst_phys + i) % RING_SIZE;
                    proptest::prop_assert_eq!(
                        work[slot], ref_ring[slot],
                        "copied slot {} (false arm) dist={} len={} pos={}",
                        slot, distance, length, pos
                    );
                }
            }
        }
    }

    /// Deterministic OFF-BY-ONE battery for the branchless u64 backward
    /// marker scan (2026-06-16). Drives `emit_backref_ring::<true>` through the
    /// non-wrap branch with EXACT marker placements — no markers, all markers,
    /// single marker at the first / last / each interior copied index — across
    /// every length boundary (1..=3 below the u64 stride, 4 exact stride, 5..=8,
    /// the 15/16/17 stride+tail seam, and 258 max) and verifies `distance_marker`
    /// equals the proven scalar `naive_backref_ref`. The off-by-one is the #1
    /// risk: `distance_marker == length - 1 - last_marker_index`.
    #[test]
    fn backref_marker_scan_offbyone_boundaries() {
        const MARKER: u16 = MAX_WINDOW_SIZE as u16; // 0x8000, a marker
        const CLEAN: u16 = 0x1234; // < MARKER_U16, not a marker
        let pos = 40_000usize; // mid-ring: non-wrap, src/dst both in [0, RING_SIZE)
        for &length in &[1usize, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 258] {
            let distance = length; // non-overlap, distance >= length
                                   // Enumerate marker patterns over the COPIED region [0, length):
                                   //   - all clean, all marker, and a single marker at each index.
            let mut patterns: Vec<Vec<bool>> = vec![vec![false; length], vec![true; length]];
            for one in 0..length {
                let mut p = vec![false; length];
                p[one] = true;
                patterns.push(p);
            }
            for pat in patterns {
                // The copied region [pos, pos+length) is sourced from
                // [pos-distance, pos). Place the desired markers THERE so the
                // copy reproduces `pat` at the destination.
                let mut ring = vec![CLEAN; RING_SIZE];
                let src_phys = pos - distance;
                for (i, &is_marker) in pat.iter().enumerate() {
                    ring[src_phys + i] = if is_marker { MARKER } else { CLEAN };
                }
                let dm_in = 0usize; // < distance ⇒ fast path NOT taken, scan runs
                let (_ref_ring, ref_dm) = naive_backref_ref(&ring, pos, distance, length, dm_in);
                let mut work = ring.clone();
                let mut p = pos;
                let mut dm = dm_in;
                unsafe {
                    emit_backref_ring::<true>(
                        work.as_mut_ptr(),
                        &mut p,
                        distance, /*drained*/
                        distance,
                        length,
                        &mut dm,
                    );
                }
                assert_eq!(p, pos + length, "pos advance len={}", length);
                assert_eq!(
                    dm, ref_dm,
                    "distance_marker len={} pattern={:?}",
                    length, pat
                );
            }
        }
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
                0,   // drained: chunk start, nothing drained yet
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

    // ── Rung (d) increment 1 differential (git history (campaign plan, removed) §5) ──
    //
    // The marker fast loop's DistTable distance decode (ON arm) vs the exact
    // pre-change dist_hc chain (OFF arm, via MARKER_DIST_LUT_OVERRIDE) on the
    // SAME window-absent streams: raw u16 output (markers included), final
    // bit cursor, decoded_bytes, and distance_to_last_marker must all be
    // equal, and the marker-resolved output must equal the original payload
    // (ground truth — catches both-arms-wrong). Arm engagement is proven by
    // the test-only MARKER_DIST_LUT_HITS counter (ON > 0; OFF window must add
    // ZERO process-wide — the override forces every thread off the arm).
    #[cfg(pure_inflate_decode)]
    mod marker_dist_lut_diff {
        use super::*;
        use std::sync::atomic::Ordering::Relaxed;

        struct OverrideGuard;
        impl Drop for OverrideGuard {
            fn drop(&mut self) {
                MARKER_DIST_LUT_OVERRIDE.store(-1, Relaxed);
            }
        }

        fn with_marker_dist_lut<T>(disabled: bool, f: impl FnOnce() -> T) -> T {
            let _g = super::DIFFERENTIAL_OVERRIDE_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            MARKER_DIST_LUT_OVERRIDE.store(if disabled { 1 } else { 0 }, Relaxed);
            let _restore = OverrideGuard;
            f()
        }

        fn xorshift(seed: &mut u64) -> u64 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        }

        fn make_rand(mut seed: u64, len: usize) -> Vec<u8> {
            (0..len).map(|_| xorshift(&mut seed) as u8).collect()
        }

        fn make_text(mut seed: u64, len: usize) -> Vec<u8> {
            const WORDS: &[&str] = &[
                "marker",
                "window",
                "deflate",
                "huffman",
                "distance",
                "table",
                "ring",
                "chunk",
                "decode",
                "bootstrap",
                "speculative",
                "clean",
            ];
            let mut out = Vec::with_capacity(len + 16);
            while out.len() < len {
                let w = WORDS[(xorshift(&mut seed) as usize) % WORDS.len()];
                out.extend_from_slice(w.as_bytes());
                out.push(b' ');
                if xorshift(&mut seed).is_multiple_of(13) {
                    out.push(b'0' + (xorshift(&mut seed) % 10) as u8);
                }
            }
            out.truncate(len);
            out
        }

        fn payload_rle(mut seed: u64, len: usize) -> Vec<u8> {
            // Long single-byte runs (dist==1 backrefs) separated by noise.
            let mut out = Vec::with_capacity(len + 512);
            while out.len() < len {
                let b = xorshift(&mut seed) as u8;
                let run = 20 + (xorshift(&mut seed) as usize) % 400;
                out.extend(std::iter::repeat_n(b, run));
                for _ in 0..4 {
                    out.push(xorshift(&mut seed) as u8);
                }
            }
            out.truncate(len);
            out
        }

        /// Payload that interleaves slices OF the dictionary with fresh
        /// separator bytes — the encoder emits distance-into-dictionary
        /// back-refs at the very start of the stream, i.e. cross-chunk
        /// MARKERS, exercising the marker-emission + marker-scan paths
        /// around the new dist decode.
        fn payload_dict_refs(dict: &[u8], len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len + 256);
            let mut seed = 0x5EED_0001u64;
            while out.len() < len {
                let off = (xorshift(&mut seed) as usize) % dict.len();
                let take = 32 + (xorshift(&mut seed) as usize) % 200;
                let end = (off + take).min(dict.len());
                out.extend_from_slice(&dict[off..end]);
                for _ in 0..8 {
                    out.push(xorshift(&mut seed) as u8);
                }
            }
            out.truncate(len);
            out
        }

        /// Window-absent (marker-mode) multi-block decode driver. Returns the
        /// raw u16 sink (markers included) + final state
        /// `(bits.pos, bitbuf, bitsleft, decoded_bytes, distance_to_last_marker)`.
        #[allow(clippy::type_complexity)]
        fn decode_marker_u16(
            deflate_bytes: &[u8],
            n_max: usize,
        ) -> (Vec<u16>, (usize, u64, u32, usize, usize)) {
            let boxed: &'static [u8] = Box::leak(deflate_bytes.to_vec().into_boxed_slice());
            let mut bits = Bits::new(boxed);
            let mut b = Block::new();
            let mut output: Vec<u16> = Vec::new();
            loop {
                b.read_header(&mut bits, false).expect("header");
                while !b.eob() {
                    let n = b.read(&mut bits, &mut output, n_max).expect("read");
                    if n == 0 && !b.eob() {
                        panic!("marker decode stalled before EOB");
                    }
                }
                if b.is_last_block() {
                    break;
                }
            }
            let dm = b.ring.distance_to_last_marker;
            (
                output,
                (bits.pos, bits.bitbuf, bits.bitsleft, b.decoded_bytes, dm),
            )
        }

        /// Resolve markers against the (≤32 KiB, right-aligned) dictionary
        /// window and assert byte equality with the original payload.
        fn resolve_and_check(out: &[u16], dict: &[u8], payload: &[u8], label: &str) {
            let mut w = vec![0u8; 32768];
            if !dict.is_empty() {
                let n = dict.len().min(32768);
                w[32768 - n..].copy_from_slice(&dict[dict.len() - n..]);
            }
            let mut data = out.to_vec();
            crate::decompress::parallel::replace_markers::replace_markers(&mut data, &w);
            assert_eq!(data.len(), payload.len(), "{label}: length mismatch");
            for (i, (&v, &p)) in data.iter().zip(payload).enumerate() {
                assert!(v < 256, "{label}: unresolved marker at {i}: {v:#x}");
                assert_eq!(v as u8, p, "{label}: byte {i} mismatch");
            }
        }

        #[test]
        fn marker_fast_loop_dist_table_matches_dist_hc_and_payload() {
            let dict_text = make_text(0xD1C7, 32768);
            let dict_rand = make_rand(0xFEED, 8192);
            let empty: Vec<u8> = Vec::new();

            // (payload, dict, must_engage): sizes cross the 64 Ki-slot ring
            // wrap, multi-block boundaries, and the mid-stream clean flip.
            let cases: [(&str, Vec<u8>, &[u8], bool); 5] = [
                (
                    "dict-text",
                    payload_dict_refs(&dict_text, 150_000),
                    &dict_text,
                    true,
                ),
                (
                    "dict-rand",
                    payload_dict_refs(&dict_rand, 90_000),
                    &dict_rand,
                    true,
                ),
                ("random", make_rand(0xABCD, 130_000), &empty, false),
                ("rle", payload_rle(0x42, 120_000), &empty, true),
                (
                    "text-nodict-refs",
                    make_text(0x7777, 140_000),
                    &dict_text,
                    true,
                ),
            ];

            for (label, payload, dict, must_engage) in &cases {
                let stream = deflate_with_dictionary(payload, dict);
                for &n_max in &[100_000usize, 1499] {
                    let tag = format!("{label} n_max={n_max}");
                    let (hits_on, on) = with_marker_dist_lut(false, || {
                        let h0 = MARKER_DIST_LUT_HITS.with(|c| c.get());
                        let r = decode_marker_u16(&stream, n_max);
                        (MARKER_DIST_LUT_HITS.with(|c| c.get()) - h0, r)
                    });
                    let (hits_off, off) = with_marker_dist_lut(true, || {
                        let h0 = MARKER_DIST_LUT_HITS.with(|c| c.get());
                        let r = decode_marker_u16(&stream, n_max);
                        (MARKER_DIST_LUT_HITS.with(|c| c.get()) - h0, r)
                    });
                    assert_eq!(on.0, off.0, "{tag}: u16 output diverged");
                    assert_eq!(on.1, off.1, "{tag}: cursor/state diverged");
                    assert_eq!(
                        hits_off, 0,
                        "{tag}: OFF arm must route zero decodes through the DistTable path"
                    );
                    if *must_engage {
                        assert!(
                            hits_on > 0,
                            "{tag}: ON arm never engaged the DistTable path"
                        );
                    }
                    resolve_and_check(&on.0, dict, payload, &tag);
                }
            }
        }
    }

    // ── ENGINE-W INC-1 / N2: local-Bits mirror kill-switch differential ──
    //
    // Verifies that the lb arm and the struct-field arm produce bit-identical
    // u16 output + final cursor state across the same marker-mode streams, and
    // that MFAST_LOCALBITS_ON_ITERS advances (ON) / stays zero (OFF).
    // Not gated on pure_inflate_decode — the lb mirror exists in both feature
    // sets; the MFAST_LOCALBITS_ON_ITERS counter is always available in tests.
    mod mfast_local_bits_diff {
        use super::*;
        use std::sync::atomic::Ordering::Relaxed;

        struct OverrideGuard;
        impl Drop for OverrideGuard {
            fn drop(&mut self) {
                MFAST_LOCALBITS_OVERRIDE.store(-1, Relaxed);
            }
        }

        fn with_localbits<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
            let _g = super::DIFFERENTIAL_OVERRIDE_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            // 0 = force ON; 1 = force OFF (kill-switch)
            MFAST_LOCALBITS_OVERRIDE.store(if enabled { 0 } else { 1 }, Relaxed);
            let _restore = OverrideGuard;
            f()
        }

        fn xorshift(seed: &mut u64) -> u64 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        }

        fn make_rand(mut seed: u64, len: usize) -> Vec<u8> {
            (0..len).map(|_| xorshift(&mut seed) as u8).collect()
        }

        fn make_text(mut seed: u64, len: usize) -> Vec<u8> {
            const WORDS: &[&str] = &[
                "local", "bits", "register", "mirror", "marker", "window", "deflate", "huffman",
                "distance", "table", "ring", "chunk",
            ];
            let mut out = Vec::with_capacity(len + 16);
            while out.len() < len {
                let w = WORDS[(xorshift(&mut seed) as usize) % WORDS.len()];
                out.extend_from_slice(w.as_bytes());
                out.push(b' ');
            }
            out.truncate(len);
            out
        }

        fn payload_rle(mut seed: u64, len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len + 512);
            while out.len() < len {
                let b = xorshift(&mut seed) as u8;
                let run = 20 + (xorshift(&mut seed) as usize) % 400;
                out.extend(std::iter::repeat_n(b, run));
                for _ in 0..4 {
                    out.push(xorshift(&mut seed) as u8);
                }
            }
            out.truncate(len);
            out
        }

        fn payload_dict_refs_local(dict: &[u8], len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len + 256);
            let mut seed = 0x5EED_0002u64;
            while out.len() < len {
                let off = (xorshift(&mut seed) as usize) % dict.len();
                let take = 32 + (xorshift(&mut seed) as usize) % 200;
                let end = (off + take).min(dict.len());
                out.extend_from_slice(&dict[off..end]);
                for _ in 0..8 {
                    out.push(xorshift(&mut seed) as u8);
                }
            }
            out.truncate(len);
            out
        }

        /// Decode deflate bytes in marker mode (no pre-existing window).
        /// Returns the raw u16 sink + final cursor state tuple.
        #[allow(clippy::type_complexity)]
        fn decode_marker_u16(
            deflate_bytes: &[u8],
            n_max: usize,
        ) -> (Vec<u16>, (usize, u64, u32, usize, usize)) {
            let boxed: &'static [u8] = Box::leak(deflate_bytes.to_vec().into_boxed_slice());
            let mut bits = Bits::new(boxed);
            let mut b = Block::new();
            let mut output: Vec<u16> = Vec::new();
            loop {
                b.read_header(&mut bits, false).expect("header");
                while !b.eob() {
                    let n = b.read(&mut bits, &mut output, n_max).expect("read");
                    if n == 0 && !b.eob() {
                        panic!("marker decode stalled before EOB");
                    }
                }
                if b.is_last_block() {
                    break;
                }
            }
            let dm = b.ring.distance_to_last_marker;
            (
                output,
                (bits.pos, bits.bitbuf, bits.bitsleft, b.decoded_bytes, dm),
            )
        }

        #[test]
        fn mfast_local_bits_matches_struct_bits() {
            let dict_text = make_text(0xB17B, 32768);
            let empty: Vec<u8> = Vec::new();

            // must_engage: if true, assert ON arm counter > 0 (the mfast Huffman
            // path must be reached). False for random data which deflates mostly
            // as STORED blocks (BTYPE=00), bypassing the Huffman mfast loop.
            let cases: [(&str, Vec<u8>, &[u8], bool); 4] = [
                (
                    "text-dict",
                    payload_dict_refs_local(&dict_text, 150_000),
                    &dict_text,
                    true,
                ),
                ("rle", payload_rle(0x99, 120_000), &empty, true),
                ("rand", make_rand(0xDEAD, 130_000), &empty, false),
                ("text-nodict", make_text(0x3A3A, 140_000), &empty, true),
            ];

            for (label, payload, dict, must_engage) in &cases {
                let stream = deflate_with_dictionary(payload, dict);
                for &n_max in &[100_000usize, 2048] {
                    let tag = format!("{label} n_max={n_max}");

                    // ON arm: lb path
                    let (iters_on, on) = with_localbits(true, || {
                        let i0 = MFAST_LOCALBITS_ON_ITERS.with(|c| c.get());
                        let r = decode_marker_u16(&stream, n_max);
                        (MFAST_LOCALBITS_ON_ITERS.with(|c| c.get()) - i0, r)
                    });

                    // OFF arm: struct-field path (kill-switch)
                    let (iters_off, off) = with_localbits(false, || {
                        let i0 = MFAST_LOCALBITS_ON_ITERS.with(|c| c.get());
                        let r = decode_marker_u16(&stream, n_max);
                        (MFAST_LOCALBITS_ON_ITERS.with(|c| c.get()) - i0, r)
                    });

                    // Byte-identical u16 output.
                    assert_eq!(
                        on.0, off.0,
                        "{tag}: u16 output diverged (lb vs struct-field)"
                    );
                    // Identical final cursor + decode state.
                    assert_eq!(
                        on.1, off.1,
                        "{tag}: cursor/state diverged (lb vs struct-field)"
                    );
                    // OFF arm must NEVER increment the lb counter.
                    assert_eq!(
                        iters_off, 0,
                        "{tag}: OFF (kill-switch) arm must not increment MFAST_LOCALBITS_ON_ITERS"
                    );
                    // ON arm must have exercised at least one block call if the
                    // stream produces Huffman-compressed blocks (must_engage=true).
                    if *must_engage {
                        assert!(
                            iters_on > 0,
                            "{tag}: ON arm never incremented MFAST_LOCALBITS_ON_ITERS — lb path not taken"
                        );
                    }
                }
            }
        }
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

        struct OverrideGuard;
        impl Drop for OverrideGuard {
            fn drop(&mut self) {
                STORED_FLIP_OVERRIDE.store(-1, Relaxed);
            }
        }

        fn with_stored_flip<T>(disabled: bool, f: impl FnOnce() -> T) -> T {
            let _g = super::DIFFERENTIAL_OVERRIDE_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            STORED_FLIP_OVERRIDE.store(if disabled { 1 } else { 0 }, Relaxed);
            let _restore = OverrideGuard;
            f()
        }

        // Reads the THREAD-LOCAL, test-only counters so the kill-switch-arm-
        // inert deltas are immune to concurrent decode on other test threads
        // (see the note at the counter definitions).
        fn counters() -> (u64, u64, u64, u64) {
            (
                STORED_FLIP_GE_WINDOW_TL.with(|c| c.get()),
                STORED_FLIP_CROSSING_TL.with(|c| c.get()),
                STORED_CLEAN_BULK_TL.with(|c| c.get()),
                STORED_CONTIG_BULK_TL.with(|c| c.get()),
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

            let enabled = with_stored_flip(false, || {
                let c0 = counters();
                let (sink, b) = decode_windowless(&stream);
                let c1 = counters();
                assert!(!b.contains_marker_bytes(), "must be clean at end");
                assert!(
                    c1.0 - c0.0 >= 1,
                    "case 1 must fire on a 40000-byte stored block"
                );
                sink
            });
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
        #[ignore = "shared process-global counter races under parallel cargo test; run serially with --ignored --test-threads=1"]
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
                // SAFETY: test-local `base`/`cap` from a contiguous buffer.
                let n1 = unsafe {
                    b.decode_clean_stored_into_contig(&mut bits, base, cap, &mut pos, 100)
                }
                .expect("partial");
                assert_eq!(n1, 100);
                assert!(!b.eob());
                let n2 = unsafe {
                    b.decode_clean_stored_into_contig(&mut bits, base, cap, &mut pos, usize::MAX)
                }
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

    /// Differential: for every distance code-length set that `dist_hc` accepts,
    /// `DistTable::build` must succeed, and for every random bit pattern the
    /// decoded distance symbol, codeword length, base, and extra bits must agree.
    ///
    /// Fixed sets: [5;30], single len-1, two len-1, deep ladder (forces
    /// subtable), incomplete/holes.  Then ~400 pseudo-random sets via seeded
    /// LCG; the test asserts ≥238 total accepted across all tries.
    #[test]
    #[cfg(pure_inflate_decode)]
    fn dist_table_matches_dist_hc_differential() {
        /// Run the dist_hc ↔ DistTable differential for one code-length set.
        /// Returns true iff dist_hc accepted the set (and all assertions passed).
        fn run_set(lens: &[u8], pattern_seed: u64) -> bool {
            use crate::decompress::inflate::consume_first_decode::Bits;
            use crate::decompress::inflate::libdeflate_entry::{DistTable, DISTANCE_TABLE};
            use crate::decompress::parallel::error::Error as HcError;
            use crate::decompress::parallel::huffman_short_bits_cached::DistanceShortBitsCached;

            // 1. Production invariant: dist_table is only built after dist_hc
            //    accepts the same lens.
            let mut dist_hc = DistanceShortBitsCached::<30>::new();
            if dist_hc.initialize_from_lengths(lens, false) != HcError::None {
                return false;
            }

            // 2. DistTable::build must succeed for any lens dist_hc accepted.
            let dt = DistTable::build(lens)
                .expect("DistTable::build must return Some when dist_hc accepts");

            // 3. ≥200k bit patterns via seeded LCG.
            let mut s = pattern_seed;
            for _ in 0..200_000u32 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits_val: u64 = s;

                // 16-byte stack buffer: two copies give ≥56 bits after Bits::new.
                let raw = bits_val.to_le_bytes();
                let buf: [u8; 16] = {
                    let mut b = [0u8; 16];
                    b[..8].copy_from_slice(&raw);
                    b[8..].copy_from_slice(&raw);
                    b
                };

                // --- dist_hc path (tracks codeword bits via bit_position delta) ---
                let mut br = Bits::new(&buf);
                let bp_before = br.bit_position();
                let hc_result = dist_hc.decode(&mut br);
                let hc_valid = hc_result.is_some();
                let hc_consumed = (br.bit_position() - bp_before) as u8;

                // --- DistTable path ---
                // For subtable entries +TABLE_BITS are consumed by main lookup.
                let main_entry = dt.lookup(bits_val);
                let (dt_entry, dt_consumed) = if main_entry.is_subtable_ptr() {
                    let sub = dt.lookup_subtable_direct(
                        main_entry,
                        bits_val >> (DistTable::TABLE_BITS as u32),
                    );
                    (sub, DistTable::TABLE_BITS + sub.codeword_bits())
                } else {
                    (main_entry, main_entry.codeword_bits())
                };

                // Invariant 1: validity must agree — raw()==0 is the
                // unassigned/invalid-code sentinel, mirrors dist_hc None.
                assert_eq!(
                    hc_valid,
                    dt_entry.raw() != 0,
                    "validity mismatch: bits_val={bits_val:#018x} hc={hc_result:?} \
                     dt_raw={:#010x}",
                    dt_entry.raw()
                );

                if hc_valid {
                    let sym = hc_result.unwrap() as usize;

                    // Invariant 2: codeword bits consumed agree.
                    assert_eq!(
                        hc_consumed, dt_consumed,
                        "codeword bits mismatch: sym={sym} bits_val={bits_val:#018x} \
                         hc={hc_consumed} dt={dt_consumed}"
                    );

                    // Invariant 3: distance_base matches RFC table.
                    assert_eq!(
                        DISTANCE_TABLE[sym].0,
                        dt_entry.distance_base(),
                        "distance_base mismatch: sym={sym}"
                    );

                    // Invariant 4: extra_bits matches RFC table
                    // (total_bits - codeword_bits for both direct and subtable entries).
                    assert_eq!(
                        DISTANCE_TABLE[sym].1,
                        dt_entry.total_bits() - dt_entry.codeword_bits(),
                        "extra_bits mismatch: sym={sym}"
                    );
                }
            }
            true
        }

        let mut accepted = 0u32;
        // Seed stream for run_set pattern seeds (independent of set-length LCG).
        let mut seed = 0xfeed_face_dead_beef_u64;
        let mut next_seed = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        // --- Fixed sets ---

        // Set 1: all lengths = 5 (Kraft sum = 30/32 < 1, nearly complete).
        if run_set(&[5u8; 30], next_seed()) {
            accepted += 1;
        }

        // Set 2: single symbol at length 1 (degenerate — only one valid code).
        let mut single_len1 = [0u8; 30];
        single_len1[0] = 1;
        if run_set(&single_len1, next_seed()) {
            accepted += 1;
        }

        // Set 3: two symbols at length 1 (Kraft-complete, codes 0 and 1).
        let mut two_len1 = [0u8; 30];
        two_len1[0] = 1;
        two_len1[1] = 1;
        if run_set(&two_len1, next_seed()) {
            accepted += 1;
        }

        // Set 4: deep ladder 1,2,...,13,14,14 — Kraft-complete, forces subtable
        // entries (TABLE_BITS=9; lengths 10-14 require subtable lookup).
        let mut deep_ladder = [0u8; 30];
        for i in 0..13usize {
            deep_ladder[i] = (i + 1) as u8;
        }
        deep_ladder[13] = 14;
        deep_ladder[14] = 14;
        if run_set(&deep_ladder, next_seed()) {
            accepted += 1;
        }

        // Set 5: incomplete/holes — sparse assignment, Kraft sum 0.25 < 1.
        let mut sparse_holes = [0u8; 30];
        sparse_holes[0] = 3;
        sparse_holes[5] = 4;
        sparse_holes[10] = 5;
        sparse_holes[15] = 5;
        if run_set(&sparse_holes, next_seed()) {
            accepted += 1;
        }

        // --- ~400 pseudo-random sets ---
        // Length generation: 30% skip (L=0), 70% uniform in {4..15}.
        // Expected Kraft sum ≈ 0.22 → acceptance rate ≈ 99%.
        let mut rng = 0x0123_4567_89ab_cdef_u64;
        for _ in 0..400u32 {
            let mut lens = [0u8; 30];
            for b in &mut lens {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let raw = (rng >> 48) as u8;
                // 77/256 ≈ 30% → L=0 (skip); rest → L in {4..15}.
                *b = if raw < 77 { 0 } else { 4 + (raw - 77) % 12 };
            }
            if run_set(&lens, next_seed()) {
                accepted += 1;
            }
        }

        // Sanity: verify the LCG produced a reasonable number of accepted sets.
        assert!(
            accepted >= 238,
            "expected ≥238 accepted code-length sets out of 405 tried, got {accepted}"
        );
    }

    /// P3.4 item 1 differential (permanent): `DistTable::rebuild` (allocation
    /// reuse) must produce a table BEHAVIORALLY IDENTICAL to a fresh
    /// `DistTable::build` for every code-length set, in any rebuild ORDER
    /// (shallow→deep grows the entries Vec, deep→shallow exercises the
    /// truncate-shrink with stale memory beyond the new length). The 15-bit
    /// pattern space is exhaustive: the main lookup reads the low 9 bits and a
    /// subtable lookup the next 6, so 2^15 patterns cover every reachable
    /// (main entry, final entry) pair.
    #[test]
    #[cfg(pure_inflate_decode)]
    fn dist_table_rebuild_matches_fresh_build() {
        use crate::decompress::inflate::libdeflate_entry::DistTable;

        fn assert_tables_equal(fresh: &DistTable, reused: &DistTable, tag: &str) {
            for p in 0..(1u64 << 15) {
                let mf = fresh.lookup(p);
                let mr = reused.lookup(p);
                assert_eq!(
                    mf.is_subtable_ptr(),
                    mr.is_subtable_ptr(),
                    "{tag}: subtable-ness diverged at pattern {p:#x}"
                );
                let (ff, fr) = if mf.is_subtable_ptr() {
                    let shifted = p >> (DistTable::TABLE_BITS as u32);
                    (
                        fresh.lookup_subtable_direct(mf, shifted).raw(),
                        reused.lookup_subtable_direct(mr, shifted).raw(),
                    )
                } else {
                    (mf.raw(), mr.raw())
                };
                assert_eq!(ff, fr, "{tag}: final entry diverged at pattern {p:#x}");
            }
        }

        // Set sequence: the same shapes the dist_hc differential uses, ordered
        // to force grow AND shrink transitions of the reused allocation.
        let mut sets: Vec<[u8; 30]> = Vec::new();
        sets.push([5u8; 30]); // flat (no subtables)
        let mut deep = [0u8; 30]; // deep ladder (subtables)
        for (i, d) in deep.iter_mut().enumerate().take(13) {
            *d = (i + 1) as u8;
        }
        deep[13] = 14;
        deep[14] = 14;
        sets.push(deep);
        sets.push(FIXED_DIST_LENGTHS); // back to shallow (shrink)
        let mut single = [0u8; 30];
        single[0] = 1;
        sets.push(single); // degenerate
        sets.push(deep); // grow again
                         // ~60 pseudo-random sets (same generator family as the dist_hc
                         // differential; unaccepted-by-dist_hc shapes are fine here — rebuild
                         // equality is a pure table property).
        let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..60 {
            let mut lens = [0u8; 30];
            for b in &mut lens {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let raw = (rng >> 48) as u8;
                *b = if raw < 77 { 0 } else { 4 + (raw - 77) % 12 };
            }
            sets.push(lens);
        }

        let mut reused = DistTable::build(&sets[0]).expect("first build");
        assert_tables_equal(
            &DistTable::build(&sets[0]).unwrap(),
            &reused,
            "set 0 (fresh==fresh sanity)",
        );
        for (i, lens) in sets.iter().enumerate().skip(1) {
            reused.rebuild(lens);
            let fresh = DistTable::build(lens).expect("fresh build");
            assert_tables_equal(&fresh, &reused, &format!("set {i}"));
        }
    }

    /// P3.4 item 1 stream-level differential (permanent): a MULTI-BLOCK raw
    /// DEFLATE stream decoded through ONE `Block` instance on the contig clean
    /// path must match the flate2 oracle byte-for-byte. Full-flush chunk
    /// boundaries force separate dynamic blocks; the repeated chunk content
    /// makes consecutive blocks carry IDENTICAL dist lens (exercising the
    /// same-lens REUSE arm) while the distinct chunks force REBUILDS, and the
    /// sync-flush padding blocks exercise the stored-block path between them.
    #[cfg(pure_inflate_decode)]
    #[test]
    fn contig_clean_stream_multi_block_dist_reuse() {
        use flate2::{Compress, Compression, FlushCompress, Status};

        // Three chunk contents: A, A again (same lens → reuse), B (rebuild),
        // then A once more (rebuild back). Sizes large enough for dynamic
        // blocks, small enough to stay single-block per chunk.
        let chunk_a: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .repeat(700)
            .to_vec();
        let chunk_b: Vec<u8> = (0u32..32000)
            .map(|i| ((i.wrapping_mul(2654435761) >> 13) & 0x3f) as u8 + 0x20)
            .collect();
        let plan: [&[u8]; 4] = [&chunk_a, &chunk_a, &chunk_b, &chunk_a];

        let mut c = Compress::new(Compression::default(), false);
        let mut deflate = Vec::new();
        let mut buf = vec![0u8; 256 * 1024];
        for (i, chunk) in plan.iter().enumerate() {
            let flush = if i + 1 == plan.len() {
                FlushCompress::Finish
            } else {
                FlushCompress::Full // resets window → blocks are independent
            };
            let before_in = c.total_in();
            let before_out = c.total_out() as usize;
            let status = c.compress(chunk, &mut buf, flush).expect("compress");
            assert_eq!(c.total_in() - before_in, chunk.len() as u64);
            if i + 1 == plan.len() {
                assert_eq!(status, Status::StreamEnd);
            }
            deflate.extend_from_slice(&buf[..c.total_out() as usize - before_out]);
        }
        let payload: Vec<u8> = plan.concat();

        // Oracle: flate2 raw-deflate decode of the same stream.
        let mut oracle = Vec::with_capacity(payload.len());
        {
            use flate2::read::DeflateDecoder;
            use std::io::Read;
            DeflateDecoder::new(&deflate[..])
                .read_to_end(&mut oracle)
                .expect("oracle inflate");
        }
        assert_eq!(oracle, payload, "oracle decode != payload");

        // Contig decode: ONE Block across all blocks (the dist_table
        // reuse/rebuild cache lives in the Block), empty-window clean seed.
        let boxed: &'static [u8] = Box::leak(deflate.clone().into_boxed_slice());
        let mut bits = Bits::new(boxed);
        let mut b = Block::new();
        let mut sink: Vec<u16> = Vec::new();
        b.set_initial_window(&mut sink, &[]).unwrap();
        let cap = payload.len() + 1024;
        let mut out = vec![0u8; cap];
        let base = out.as_mut_ptr();
        let mut pos = 0usize;
        let mut saw_dynamic = 0u32;
        loop {
            b.read_header(&mut bits, false).unwrap();
            match b.compression_type() {
                CompressionType::Uncompressed => {
                    // Sync/full-flush padding blocks (len may be 0).
                    // SAFETY: test-local `base`/`cap` from a contiguous buffer.
                    while !b.eob() {
                        unsafe {
                            b.decode_clean_stored_into_contig(&mut bits, base, cap, &mut pos, cap)
                        }
                        .unwrap();
                    }
                }
                CompressionType::FixedHuffman | CompressionType::DynamicHuffman => {
                    if b.compression_type() == CompressionType::DynamicHuffman {
                        saw_dynamic += 1;
                    }
                    while !b.eob() {
                        // Small per-call cap splits blocks across resumable
                        // re-entries (the lazy-build site runs once per block).
                        // SAFETY: test-local `base`/`cap` from a contiguous buffer.
                        let n = unsafe {
                            b.decode_clean_into_contig(&mut bits, base, cap, &mut pos, 8192)
                        }
                        .unwrap();
                        if n == 0 && !b.eob() {
                            panic!("contig stream decode stalled");
                        }
                    }
                }
                other => panic!("unexpected block type {other:?}"),
            }
            if b.is_last_block() {
                break;
            }
        }
        assert!(
            saw_dynamic >= 3,
            "expected ≥3 dynamic blocks (got {saw_dynamic}) — chunk sizes no longer force them?"
        );
        out.truncate(pos);
        assert_eq!(
            out, oracle,
            "multi-block contig decode diverged from flate2 oracle"
        );
    }

    /// P3.4 item 2 differential (permanent): `emit_backref_contig` (the
    /// libdeflate-shape copy: 5-word burst for dist>=8, broadcast RLE for
    /// dist==1, stride-dist word trick for dist 2..7) must write EXACTLY the
    /// sequential-copy reference bytes in `[*pos, *pos+length)` and touch
    /// NOTHING beyond the documented overshoot envelope
    /// (`*pos + max(40, length+7) <= *pos + 265`), for every overlap distance
    /// x length x alignment combination.
    #[cfg(any(
        all(
            feature = "isal-compression",
            not(feature = "pure-rust-inflate"),
            target_arch = "x86_64"
        ),
        pure_inflate_decode
    ))]
    #[test]
    fn emit_backref_contig_differential() {
        const HISTORY: usize = 300; // > max distance 258
        const ENVELOPE: usize = 265; // documented worst-case touch past *pos
        const TAIL: usize = ENVELOPE + 64; // envelope + canary slack
        const CANARY: u8 = 0xA5;

        // Lengths: all short (the burst/stride boundary cases) + stepped long.
        let lens: Vec<usize> = (3..=48).chain((49..=258).step_by(3)).chain([258]).collect();

        let mut buf = vec![0u8; HISTORY + 8 + 258 + TAIL];
        let mut reference = vec![0u8; buf.len()];
        for align in 0..8usize {
            let start = HISTORY + align;
            for distance in 1..=258usize {
                for &length in &lens {
                    // Pseudo-random history (deterministic per case).
                    let mut s = (align as u64) << 32 | (distance as u64) << 16 | length as u64;
                    for (i, b) in buf[..start].iter_mut().enumerate() {
                        s = s
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        *b = (s >> 56) as u8 ^ i as u8;
                    }
                    for b in buf[start..].iter_mut() {
                        *b = CANARY;
                    }
                    // Reference: strict sequential byte copy.
                    reference[..start].copy_from_slice(&buf[..start]);
                    for i in 0..length {
                        reference[start + i] = reference[start + i - distance];
                    }

                    let mut pos = start;
                    unsafe {
                        emit_backref_contig(buf.as_mut_ptr(), &mut pos, distance, length);
                    }
                    assert_eq!(pos, start + length, "pos advance d={distance} l={length}");
                    assert_eq!(
                        &buf[..start + length],
                        &reference[..start + length],
                        "bytes diverged: d={distance} l={length} align={align}"
                    );
                    // No touch beyond the envelope.
                    assert!(
                        buf[start + ENVELOPE..].iter().all(|&b| b == CANARY),
                        "wrote past the {ENVELOPE}-byte envelope: d={distance} l={length} align={align}"
                    );
                }
            }
        }
    }

    /// P3.2 differential (permanent, modeled on
    /// `dist_table_matches_dist_hc_differential`): the contig fast loop's
    /// RUNTIME LITERAL CHAIN emission (lit1 arm: consume up to LIT_CHAIN_MAX
    /// extra lone literals inline, carry the final un-consumed packet,
    /// threshold-refill at the carry site) must emit EXACTLY the bytes and
    /// consume EXACTLY the bits of the old one-packet-per-iteration emission
    /// (decode at the iteration top after a `<48 → refill`), and the carried
    /// packet must be refill-invariant — for fixed-Huffman, balanced,
    /// short-biased (forces build-time-packed sym_count>1 carries) and
    /// deep-biased (forces long-path/subtable codes) complete code sets, over
    /// random bit patterns INCLUDING short buffers that exhaust input
    /// mid-chain (the fabricated-bits regime the `bit_count <= available`
    /// consume gate exists for).
    #[test]
    fn lit_chain_emission_matches_unchained_differential() {
        use crate::decompress::inflate::consume_first_decode::Bits;
        use crate::decompress::parallel::lut_huffman::{DecodedSymbol, LutLitLenCode};

        // MUST mirror decode_clean_into_contig's LIT_CHAIN_MAX.
        const LIT_CHAIN_MAX: usize = 2;
        // Bound per-pattern emission in BOTH sims (same cut rule).
        const CAP: usize = 12;

        fn lcg(s: &mut u64) -> u64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s
        }

        // The production consume gate: lone literal, valid, fully backed by
        // real bits (post any internal decode refill).
        fn lone_backed_literal(p: &DecodedSymbol, b: &Bits) -> bool {
            p.sym_count == 1
                && p.bit_count != 0
                && (p.symbol & 0xFFFF) <= 255
                && p.bit_count <= b.available()
        }

        /// Returns (total literals emitted, patterns that emitted ≥3 — i.e.
        /// the NEW sim's chain demonstrably consumed inline).
        fn run_set(lut: &LutLitLenCode, pattern_seed: u64, n_patterns: u32) -> (u64, u64) {
            let mut total_lits = 0u64;
            let mut ge3_patterns = 0u64;
            let mut s = pattern_seed;
            for _ in 0..n_patterns {
                let mut bytes = [0u8; 64];
                for chunk in bytes.chunks_mut(8) {
                    chunk.copy_from_slice(&lcg(&mut s).to_le_bytes());
                }
                // Adversarial truncation 2..=64: short buffers exhaust input
                // mid-chain and exercise the consume gate + slow refills.
                let len = 2 + ((lcg(&mut s) >> 32) as usize % 63);
                let buf = &bytes[..len];

                // ── OLD emission: one packet per iteration top ──
                let mut old_lits: Vec<u8> = Vec::new();
                let mut ob = Bits::new(buf);
                let mut o_pre = lut.decode(&mut ob);
                let mut o_capped = false;
                loop {
                    if !lone_backed_literal(&o_pre, &ob) {
                        break;
                    }
                    ob.consume(o_pre.bit_count);
                    old_lits.push((o_pre.symbol & 0xFF) as u8);
                    if old_lits.len() >= CAP {
                        o_capped = true;
                        break;
                    }
                    if (ob.bitsleft as u8) < 48 {
                        ob.refill();
                    }
                    o_pre = lut.decode(&mut ob);
                }

                // ── NEW emission: the production chain shape (decode_clean_
                // into_contig lit1 arm, transcribed) ──
                let mut new_lits: Vec<u8> = Vec::new();
                let mut nb = Bits::new(buf);
                let mut n_pre = lut.decode(&mut nb);
                let mut n_capped = false;
                'outer: loop {
                    if !lone_backed_literal(&n_pre, &nb) {
                        break;
                    }
                    nb.consume(n_pre.bit_count);
                    new_lits.push((n_pre.symbol & 0xFF) as u8);
                    if new_lits.len() >= CAP {
                        n_capped = true;
                        break;
                    }
                    let mut chained = 0usize;
                    loop {
                        let nxt = lut.decode(&mut nb);
                        if chained < LIT_CHAIN_MAX && lone_backed_literal(&nxt, &nb) {
                            nb.consume(nxt.bit_count);
                            new_lits.push((nxt.symbol & 0xFF) as u8);
                            chained += 1;
                            if new_lits.len() >= CAP {
                                n_capped = true;
                                break 'outer;
                            }
                            continue;
                        }
                        n_pre = nxt; // carry un-consumed
                        break;
                    }
                    // Carry-site refill (the chain's ≥48 restoration).
                    if (nb.bitsleft as u8) < 48 {
                        nb.refill();
                    }
                }

                assert_eq!(
                    old_lits, new_lits,
                    "emitted literals diverge: buf={buf:02x?}"
                );
                assert_eq!(
                    ob.bit_position(),
                    nb.bit_position(),
                    "bit cursors diverge: buf={buf:02x?}"
                );
                assert_eq!(o_capped, n_capped, "cap states diverge: buf={buf:02x?}");
                if !o_capped {
                    // Both stopped on the same un-consumed carry. OLD decoded
                    // it post-refill, NEW (possibly) pre-refill — equality is
                    // the carry-across-refill invariance the production carry
                    // depends on (Bits::refill is append-only).
                    assert_eq!(o_pre, n_pre, "carried packet diverges: buf={buf:02x?}");
                }
                total_lits += new_lits.len() as u64;
                if new_lits.len() >= 3 {
                    ge3_patterns += 1;
                }
            }
            (total_lits, ge3_patterns)
        }

        // Valid-by-construction complete prefix codes via random binary
        // splitting: start at the root; each step replaces a random leaf
        // (len < 15) with two leaves of len+1 — Kraft stays exactly 1, so
        // the LUT builder must accept every generated set.
        // bias: -1 = split shallowest candidate (short codes → build-time
        // packing fires → sym_count>1 carries in the chain), +1 = split
        // deepest (long codes → long-path/subtable decodes), 0 = random.
        fn gen_complete_lens(n_syms: usize, n_codes: usize, bias: i8, rng: &mut u64) -> Vec<u8> {
            let mut leaves: Vec<u8> = vec![1, 1]; // first split done
            while leaves.len() < n_codes {
                let a = (lcg(rng) as usize) % leaves.len();
                let b = (lcg(rng) as usize) % leaves.len();
                let pick = |i: usize, j: usize| -> usize {
                    match bias {
                        -1 => {
                            if leaves[i] <= leaves[j] {
                                i
                            } else {
                                j
                            }
                        }
                        1 => {
                            if leaves[i] >= leaves[j] {
                                i
                            } else {
                                j
                            }
                        }
                        _ => i,
                    }
                };
                let mut idx = pick(a, b);
                if leaves[idx] >= 15 {
                    // Find any splittable leaf (always exists for ≤286 codes).
                    match leaves.iter().position(|&l| l < 15) {
                        Some(p) => idx = p,
                        None => break,
                    }
                }
                let l = leaves[idx] + 1;
                leaves[idx] = l;
                leaves.push(l);
            }
            // Assign the lengths to distinct random symbols (partial
            // Fisher-Yates over 0..n_syms).
            let mut symbols: Vec<usize> = (0..n_syms).collect();
            let mut lens = vec![0u8; n_syms];
            for (k, &l) in leaves.iter().enumerate() {
                let j = k + (lcg(rng) as usize) % (n_syms - k);
                symbols.swap(k, j);
                lens[symbols[k]] = l;
            }
            lens
        }

        let mut seed = 0xfeed_face_dead_beef_u64;

        // Fixed-Huffman lengths: the production static table (8-bit literals
        // 0..143, 9-bit 144..255 — chains fire, packing ~never).
        let mut fixed_lut = LutLitLenCode::new_empty();
        assert!(
            fixed_lut.rebuild_from(&FIXED_LIT_LEN_LENGTHS[..]),
            "fixed-Huffman lens must be accepted"
        );
        let (mut lits_total, mut ge3_total) = run_set(&fixed_lut, lcg(&mut seed), 20_000);

        // Generated complete sets: balanced / short-biased / deep-biased.
        let mut lut = LutLitLenCode::new_empty();
        for (n_sets, bias, n_codes_lo, n_codes_hi) in [
            (40usize, 0i8, 32usize, 286usize), // balanced
            (40, -1, 8, 64),                   // short-biased (packing-rich)
            (40, 1, 32, 286),                  // deep-biased (subtable/long)
            // Lit-rich regime (the production target: silesia-class ~8-9-bit
            // literal codes, pairs >12 bits → build-time packing ~never,
            // runtime chains dominate).
            (40, 0, 192, 286),
        ] {
            for _ in 0..n_sets {
                let n_codes = n_codes_lo + (lcg(&mut seed) as usize) % (n_codes_hi - n_codes_lo);
                let lens = gen_complete_lens(286, n_codes, bias, &mut seed);
                assert!(
                    lut.rebuild_from(&lens),
                    "complete code set must be accepted: bias={bias} n_codes={n_codes}"
                );
                let (l, g) = run_set(&lut, lcg(&mut seed), 5_000);
                lits_total += l;
                ge3_total += g;
            }
        }

        // Vacuity guards: the differential must actually exercise literal
        // emission AND multi-literal chains, not stop at packet 1 everywhere.
        // (Deterministic LCG seeds: measured 303,802 / 31,494 at authoring.)
        assert!(
            lits_total > 250_000,
            "differential too vacuous: only {lits_total} literals emitted"
        );
        assert!(
            ge3_total > 25_000,
            "chain never demonstrably fired: only {ge3_total} ≥3-literal patterns"
        );
    }
}
