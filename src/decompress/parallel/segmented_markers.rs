#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

//! Segmented `Vec<u16>` replacement for `ChunkData::data_with_markers`,
//! plus its IN-PLACE resolve-to-u8 step that eliminates the separate
//! `narrowed` buffer.
//!
//! ## Vendor blueprint
//!
//! Port of rapidgzip's `DecodedData::dataWithMarkers`
//! (`vendor/rapidgzip/src/rapidgzip/DecodedData.hpp:238-275`): a
//! `std::vector<MarkerVector>` where each inner `MarkerVector` is a
//! `FasterVector<uint16_t>` filled in [`SEGMENT_ELEMENTS`]-equal
//! chunks via `appendToEquallySizedChunks` (DecodedData.hpp:238-275).
//!
//! The marker resolution `applyWindow` (DecodedData.hpp:306-392)
//! resolves the u16 markers to u8 IN PLACE over each 128 KiB marker
//! chunk (`reinterpret_cast<uint8_t*>(chunk.data()); target[i] =
//! window[chunk[i]]`) then `std::swap(reusedDataBuffers,
//! dataWithMarkers)` to RETAIN the backing allocations and reinterpret
//! them as u8 views. Zero new allocation, one pass.
//!
//! ## Why this is the faithful port (and what gzippy replaced)
//!
//! gzippy previously stored `data_with_markers` as a MONOLITHIC
//! `std::vec::Vec<u16>` (glibc `System` allocator, NOT rpmalloc) that
//! grows to multiple MiB — well above rpmalloc's huge threshold, so on
//! the C side the analog would `munmap` on free and re-fault on the
//! next chunk. gzippy further used a SEPARATE rpmalloc `narrowed: U8`
//! 3rd buffer for the resolved bytes. This module ports BOTH vendor
//! properties:
//!   1. 128 KiB-capped u16 segments, rpmalloc-backed, warm-recycled via
//!      the per-segment pool (each segment < rpmalloc `LARGE_SIZE_LIMIT`
//!      = 2 MiB so it stays in the per-thread span cache).
//!   2. In-place resolve: u16 markers → u8 narrowed bytes are written
//!      into the SAME backing allocation (u8 output ≤ u16 input bytes,
//!      so it always fits), then exposed as u8 slices. No separate
//!      `narrowed` allocation.
//!
//! ### Safe-Rust deviation from the C++ `std::swap` reinterpret
//!
//! Vendor `std::swap`s the `Vec<u16>` storage into a `Vec<u8>` view
//! list and reads the same bytes back as u8. In Rust, transmuting a
//! `Vec<u16>` into a `Vec<u8>` is UB (the `Layout` passed to `dealloc`
//! would carry the wrong element size/align on free). We achieve the
//! SAME resident-footprint goal — zero new heap allocation, the
//! resolved u8 bytes living in the marker buffer's own pages — by:
//!   - keeping each segment a `Vec<u16>`,
//!   - resolving markers into the low half of that same segment's
//!     backing store via raw pointers (`u8` write at byte offset `i`
//!     reading `u16` at element `i`, strictly left-to-right so the
//!     read of element `i` precedes the write of byte `i ≤ 2i`),
//!   - exposing the resolved bytes as a `&[u8]` view over the segment's
//!     first `len` bytes.
//! The Vec is freed correctly as `Vec<u16>`; we never hand a
//! mis-Layout'd allocation to the allocator. Same residency, safe
//! mechanics. The deviation is documented per the inner-loop honesty
//! rule.

use crate::decompress::parallel::chunk_buffer_pool;
use crate::decompress::parallel::replace_markers::MARKER_BASE;
use crate::decompress::parallel::rpmalloc_alloc::types::U16;

thread_local! {
    /// Per post-process worker: literal iota + zero mid-range initialized once.
    static APPLY_WINDOW_LUT: std::cell::RefCell<Option<[u8; 65536]>> =
        std::cell::RefCell::new(None);
}

/// Allocate (or recycle) one 128 KiB marker segment. Sources from the
/// per-worker marker-segment pool so freed segments stay warm in
/// rpmalloc's per-thread span cache across chunks — the vendor
/// `FasterVector` recycle behavior (`core/FasterVector.hpp:120-128`).
#[inline]
fn new_segment() -> U16 {
    chunk_buffer_pool::take_marker_segment()
}

/// Vendor's `ALLOCATION_CHUNK_SIZE` in ELEMENTS for the u16 marker
/// buffer. Vendor (`DecodedData.hpp:241`) sizes the chunk in *bytes*
/// (128 KiB) for the byte buffer; for the u16 marker buffer the same
/// 128 KiB byte span holds 64 Ki u16 elements. We keep each segment's
/// byte footprint at 128 KiB (= 64 Ki u16) so the allocation lands in
/// the same rpmalloc span class as the clean-data segments.
pub const SEGMENT_ELEMENTS: usize = 64 * 1024;

/// Segmented `Vec<u16>` marker buffer. Each segment is a
/// `Vec<u16, RpmallocAlloc>` capped at [`SEGMENT_ELEMENTS`] elements
/// (= 128 KiB bytes). Append-only from the decoder's perspective
/// (the inner deflate decoder resolves all back-references from its
/// OWN 32 KiB `output_ring` and only `push_slice`s resolved runs into
/// this sink — there is no cross-segment back-reference into this
/// buffer, so the 128 KiB seam needs no special back-ref handling
/// here).
#[derive(Debug, Default, Clone)]
pub struct SegmentedU16 {
    /// Owned 64 Ki-element (128 KiB) inner buffers, append order.
    segments: Vec<U16>,
    /// Cached total element count (sum of segment lens). O(1) `len`.
    cached_len: usize,
}

impl SegmentedU16 {
    /// Total u16 element count across all segments. O(1).
    #[inline]
    pub fn len(&self) -> usize {
        self.cached_len
    }

    /// True iff no elements are stored. O(1).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }

    /// Number of populated segments. Diagnostic surface.
    #[allow(dead_code)]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Sum of segment capacities, in u16 elements. Recycler stats.
    #[allow(dead_code)]
    pub fn capacity_elements(&self) -> usize {
        self.segments.iter().map(|s| s.capacity()).sum()
    }

    /// Truncate every segment to zero length, retaining the
    /// allocations for reuse. Mirrors `Vec::clear`.
    #[allow(dead_code)] // Vec-surface parity; production uses take_segments
    pub fn clear(&mut self) {
        for seg in &mut self.segments {
            seg.clear();
        }
        self.cached_len = 0;
    }

    /// Pre-reserve at least `additional` more u16 elements of total
    /// capacity, allocated in [`SEGMENT_ELEMENTS`]-sized segments.
    /// Mirrors `Vec::reserve`; the underlying allocations are chunked.
    /// Used by the bootstrap to warm the first segment.
    pub fn reserve(&mut self, additional: usize) {
        let current_cap = self.capacity_elements();
        let needed = self.cached_len + additional;
        if needed <= current_cap {
            return;
        }
        let extra = needed - current_cap;
        let n_segments = extra.div_ceil(SEGMENT_ELEMENTS);
        for _ in 0..n_segments {
            self.segments.push(new_segment());
        }
    }

    /// Append a u16 slice, distributing across segments at the
    /// [`SEGMENT_ELEMENTS`] boundary — the LAST segment fills first,
    /// then a fresh segment is allocated for the overflow. Mirror of
    /// vendor's `appendToEquallySizedChunks` (DecodedData.hpp:243-258).
    ///
    /// This is the decoder's write path (via
    /// [`crate::decompress::parallel::marker_inflate::MarkerSink`]).
    pub fn push_slice(&mut self, values: &[u16]) {
        let mut src = values;
        while !src.is_empty() {
            let needs_new = match self.segments.last() {
                Some(seg) => seg.len() >= SEGMENT_ELEMENTS,
                None => true,
            };
            if needs_new {
                self.segments.push(new_segment());
            }
            let last = self.segments.last_mut().unwrap();
            let room = SEGMENT_ELEMENTS - last.len();
            let n = src.len().min(room);
            // `extend_from_slice` on `allocator_api2::Vec<u16, RpmallocAlloc>`
            // does NOT lower to a vectorized memcpy under LLVM — the custom
            // allocator breaks the specialization and emits a scalar u16-per-
            // element loop (~0.649B instructions, ~8-9% of decode total, shared
            // across both build flavors). Use `copy_nonoverlapping` directly.
            // SAFETY: `take_marker_segment` pre-allocates SEGMENT_ELEMENTS
            // capacity; `old_len + n <= SEGMENT_ELEMENTS <= capacity`; `src`
            // and `last` are distinct allocations — no overlap.
            let old_len = last.len();
            debug_assert!(
                last.capacity() >= old_len + n,
                "segment capacity invariant: cap={} old_len={} n={}",
                last.capacity(),
                old_len,
                n
            );
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), last.as_mut_ptr().add(old_len), n);
                last.set_len(old_len + n);
            }
            self.cached_len += n;
            src = &src[n..];
        }
    }

    /// Iterate over the segments as u16 slices, in append order.
    #[inline]
    pub fn segments(&self) -> impl Iterator<Item = &[u16]> {
        self.segments.iter().map(|v| v.as_slice())
    }

    /// Logical element iterator (slow; tests and debug asserts only).
    pub fn iter(&self) -> impl Iterator<Item = u16> + '_ {
        self.segments().flat_map(|s| s.iter().copied())
    }

    /// Copy the last `n` elements into a caller-provided buffer (which
    /// it clears + fills). Walks at most two trailing segments. Used
    /// by the bootstrap's clean-window extraction and per-block
    /// trailing-clean tracking, which only ever read a tail suffix
    /// (≤ 32 KiB window or one just-decoded block).
    ///
    /// Returns the number of elements copied (== `n.min(self.len())`).
    pub fn copy_last_n(&self, n: usize, out: &mut Vec<u16>) -> usize {
        out.clear();
        let n = n.min(self.cached_len);
        if n == 0 {
            return 0;
        }
        let start = self.cached_len - n;
        let mut elem_base = 0usize;
        out.reserve(n);
        for seg in &self.segments {
            let seg_len = seg.len();
            let seg_start = elem_base;
            let seg_end = elem_base + seg_len;
            if seg_end > start {
                let local_from = start.saturating_sub(seg_start);
                out.extend_from_slice(&seg[local_from..]);
            }
            elem_base = seg_end;
        }
        debug_assert_eq!(out.len(), n);
        n
    }

    /// If the last `n` elements are all CLEAN (`< MARKER_BASE`), append
    /// them as u8 bytes to `out` (NOT cleared — caller controls) and
    /// return `true`. If any of the last `n` is a marker, append nothing
    /// and return `false`. Used by `last_32kib_window{,_vec}` which need
    /// a marker-free tail to seed the successor's dict (otherwise the
    /// caller must wait for resolution).
    pub fn append_last_n_clean_bytes(&self, n: usize, out: &mut [u8]) -> bool {
        debug_assert!(n <= self.cached_len);
        debug_assert!(out.len() >= n);
        if n == 0 {
            return true;
        }
        let start = self.cached_len - n;
        // First pass: marker check over the suffix.
        let mut elem_base = 0usize;
        let mut write_pos = 0usize;
        for seg in &self.segments {
            let seg_len = seg.len();
            let seg_start = elem_base;
            let seg_end = elem_base + seg_len;
            if seg_end > start {
                let local_from = start.saturating_sub(seg_start);
                for &v in &seg[local_from..] {
                    if v >= MARKER_BASE {
                        return false;
                    }
                    out[write_pos] = v as u8;
                    write_pos += 1;
                }
            }
            elem_base = seg_end;
        }
        debug_assert_eq!(write_pos, n);
        true
    }

    /// Run-length of trailing CLEAN (`< MARKER_BASE`) values, capped at
    /// `cap`. Walks segments back-to-front, stopping at the first
    /// marker or at `cap`. Used by the bootstrap's per-block
    /// trailing-clean tracker (replaces a `block_slice.iter().rev()
    /// .take_while(...)` over a contiguous slice).
    pub fn trailing_clean_run(&self, cap: usize) -> usize {
        let mut run = 0usize;
        for seg in self.segments.iter().rev() {
            for &v in seg.iter().rev() {
                if v >= MARKER_BASE || run >= cap {
                    return run.min(cap);
                }
                run += 1;
            }
        }
        run.min(cap)
    }

    /// Position (element index) of the LAST marker (`>= MARKER_BASE`)
    /// across all segments, or `None` if the buffer is marker-free.
    /// Equivalent to `slice.iter().rposition(|&v| v >= MARKER_BASE)`.
    /// Used by `clean_unmarked_data`.
    pub fn rposition_last_marker(&self) -> Option<usize> {
        let mut elem_end = self.cached_len;
        for seg in self.segments.iter().rev() {
            let seg_len = seg.len();
            let seg_start = elem_end - seg_len;
            for (local_rev, &v) in seg.iter().rev().enumerate() {
                if v >= MARKER_BASE {
                    return Some(seg_start + (seg_len - 1 - local_rev));
                }
            }
            elem_end = seg_start;
        }
        None
    }

    /// Truncate the logical buffer to at most `new_len` u16 elements.
    /// Empties trailing segments first, then truncates the partial
    /// segment. Mirrors `Vec::truncate`. Used by `clean_unmarked_data`
    /// after migrating the marker-free tail into the clean `data`.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.cached_len {
            return;
        }
        let mut remaining = new_len;
        for seg in &mut self.segments {
            if remaining >= seg.len() {
                remaining -= seg.len();
            } else {
                seg.truncate(remaining);
                remaining = 0;
            }
        }
        self.cached_len = new_len;
    }

    /// Element access. O(segments) per call — avoid in tight loops.
    /// Used by `clean_unmarked_data`'s tail-migration narrow loop and
    /// the window-builders, which index a bounded suffix.
    pub fn get(&self, mut index: usize) -> Option<u16> {
        if index >= self.cached_len {
            return None;
        }
        for seg in &self.segments {
            if index < seg.len() {
                return Some(seg[index]);
            }
            index -= seg.len();
        }
        None
    }

    /// Copy the resolved-or-literal value at logical element range
    /// `[from, from+len)` into `out` (cleared first), resolving any
    /// marker against `window`. Used by the window-builders
    /// (`get_last_window`, `populate_subchunk_windows`) which need a
    /// resolved byte run from an arbitrary offset. Walks the relevant
    /// segments; `out` receives up to `len` u8 bytes (fewer if the
    /// range extends past the buffer end).
    pub fn resolve_range_into(&self, from: usize, len: usize, window: &[u8], out: &mut Vec<u8>) {
        out.clear();
        if len == 0 || from >= self.cached_len {
            return;
        }
        let end = (from + len).min(self.cached_len);
        let n = end - from;
        out.resize(n, 0);
        self.resolve_range_into_buf(from, n, window, out);
    }

    /// Segment-efficient resolve: copy logical range `[from, from+len)` into
    /// `out[..len]`, mapping markers through `window`. One segment walk —
    /// O(touched segments), not O(len × segments) like repeated [`Self::get`].
    pub fn resolve_range_into_buf(&self, from: usize, len: usize, window: &[u8], out: &mut [u8]) {
        debug_assert!(out.len() >= len);
        if len == 0 || from >= self.cached_len {
            return;
        }
        let end = (from + len).min(self.cached_len);
        let mut elem_base = 0usize;
        let mut out_off = 0usize;
        for seg in &self.segments {
            let seg_len = seg.len();
            let seg_start = elem_base;
            let seg_end = elem_base + seg_len;
            if seg_end > from && seg_start < end {
                let local_from = from.saturating_sub(seg_start);
                let local_to = (end - seg_start).min(seg_len);
                for &v in &seg[local_from..local_to] {
                    out[out_off] = if v >= MARKER_BASE {
                        window[(v - MARKER_BASE) as usize]
                    } else {
                        v as u8
                    };
                    out_off += 1;
                }
            }
            elem_base = seg_end;
            if elem_base >= end {
                break;
            }
        }
        debug_assert_eq!(out_off, end - from);
    }

    /// Construct from owned segments (recycler give-back).
    #[allow(dead_code)] // recycler-symmetry helper; tested
    pub fn from_segments(segments: Vec<U16>) -> Self {
        let cached_len = segments.iter().map(|s| s.len()).sum();
        Self {
            segments,
            cached_len,
        }
    }

    /// Release the owned segments to the recycler. Leaves `self`
    /// empty.
    pub fn take_segments(&mut self) -> Vec<U16> {
        self.cached_len = 0;
        std::mem::take(&mut self.segments)
    }

    /// Resolve every marker (`>= MARKER_BASE`) to its literal byte value
    /// (as a u16 `< 256`) IN PLACE, keeping the buffer a `SegmentedU16`.
    /// This is the u16→u16 form of vendor `applyWindow` used by the
    /// standalone `apply_window` helper + tests; the production path uses
    /// the u8-producing [`Self::resolve_in_place`]. Applies the same
    /// branchless LUT per segment.
    pub fn resolve_markers_u16(&mut self, window: &[u8]) {
        if self.cached_len == 0 {
            return;
        }
        debug_assert_eq!(window.len(), 32768);
        let mut lut = [0u16; 65536];
        for (i, slot) in lut[0..256].iter_mut().enumerate() {
            *slot = i as u16;
        }
        for (i, &b) in window.iter().enumerate() {
            lut[MARKER_BASE as usize + i] = b as u16;
        }
        for seg in &mut self.segments {
            for v in seg.iter_mut() {
                *v = lut[*v as usize];
            }
        }
    }

    /// Narrow already-resolved u16 (`< MARKER_BASE`) to u8 in place over
    /// each segment's backing store (low `len` bytes). Used by
    /// `ChunkData::narrow_markers_in_place` after `apply_window`.
    pub fn narrow_markers_to_u8_in_place(&mut self) {
        for seg in &mut self.segments {
            let n = seg.len();
            if n == 0 {
                continue;
            }
            let dst = seg.as_mut_ptr() as *mut u8;
            let src = seg.as_ptr();
            for i in 0..n {
                // SAFETY: read u16 at i, write u8 at i; `i < 2i` so no clobber.
                let v = unsafe { *src.add(i) };
                unsafe {
                    dst.add(i).write(v as u8);
                }
            }
        }
    }

    /// FUSED resolve + narrow in a SINGLE pass with a 64 KiB u8 LUT.
    /// Equivalent to `resolve_markers_u16(window)` followed by
    /// `narrow_markers_to_u8_in_place()` but in ONE pass over the data and
    /// with a 64 KiB u8 LUT instead of a 128 KiB u16 LUT — the same hot loop
    /// as the (unwired) [`Self::resolve_in_place`], kept type-preserving so the
    /// downstream narrowed readers (CRC, iovecs, subchunk windows) are
    /// unchanged. After this call the low `len` bytes of each segment hold the
    /// resolved u8 output; the buffer stays a `SegmentedU16`.
    ///
    /// SAFETY of the in-place u16→u8 overwrite: we read element `i` (u16 at
    /// byte offset `2i`) then write the resolved byte at offset `i`. Since
    /// `i <= 2i` and we iterate left-to-right, every write lands on a byte of
    /// an element already read this pass — never clobbers an unread element.
    pub fn resolve_and_narrow_in_place(&mut self, window: &[u8]) {
        if self.cached_len == 0 {
            return;
        }
        debug_assert_eq!(window.len(), 32768);
        // Reuse a per-thread LUT (see `APPLY_WINDOW_LUT` above).
        APPLY_WINDOW_LUT.with(|cell| {
            let mut opt = cell.borrow_mut();
            let lut = opt.get_or_insert_with(|| {
                let mut lut = [0u8; 65536];
                for (i, slot) in lut[0..256].iter_mut().enumerate() {
                    *slot = i as u8;
                }
                lut
            });
            lut[MARKER_BASE as usize..MARKER_BASE as usize + 32768].copy_from_slice(window);
            Self::resolve_and_narrow_segments_in_place(&mut self.segments, lut);
        });
    }

    fn resolve_and_narrow_segments_in_place(segments: &mut [U16], lut: &[u8; 65536]) {
        for seg in segments {
            let n = seg.len();
            if n == 0 {
                continue;
            }
            let base = seg.as_mut_ptr() as *mut u8;
            let src = seg.as_ptr();
            for i in 0..n {
                // SAFETY: read element i in [0,n); write byte i (< n <= 2n).
                let v = unsafe { *src.add(i) };
                unsafe {
                    base.add(i).write(lut[v as usize]);
                }
            }
        }
    }

    /// Copy `len` u8 bytes starting at logical narrowed offset `from` into
    /// `out` (must have room). Used by `populate_subchunk_windows`.
    pub fn copy_narrowed_u8_range_into(&self, from: usize, len: usize, out: &mut [u8]) {
        debug_assert!(out.len() >= len);
        if len == 0 {
            return;
        }
        let end = from + len;
        let mut elem_base = 0usize;
        let mut out_off = 0usize;
        for seg in &self.segments {
            let seg_len = seg.len();
            let seg_start = elem_base;
            let seg_end = elem_base + seg_len;
            if seg_end > from && seg_start < end {
                let local_from = from.saturating_sub(seg_start);
                let local_to = (end - seg_start).min(seg_len);
                let n = local_to - local_from;
                // SAFETY: `narrow_markers_to_u8_in_place` wrote u8 at byte
                // offsets `[0, seg_len)` in this segment's storage.
                let src = unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, seg_len) };
                out[out_off..out_off + n].copy_from_slice(&src[local_from..local_to]);
                out_off += n;
            }
            elem_base = seg_end;
            if elem_base >= end {
                break;
            }
        }
        debug_assert_eq!(out_off, len);
    }

    /// Append `narrowed_len` bytes (post in-place narrow) as iovecs.
    pub fn append_narrowed_iovecs<'a>(&'a self, narrowed_len: usize, out: &mut Vec<&'a [u8]>) {
        if narrowed_len == 0 {
            return;
        }
        let mut left = narrowed_len;
        for seg in &self.segments {
            if left == 0 {
                break;
            }
            let n = left.min(seg.len());
            // SAFETY: see `copy_narrowed_u8_range_into`.
            let sl = unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, n) };
            out.push(sl);
            left -= n;
        }
    }

    /// True iff every value is `< MARKER_BASE` (no unresolved markers).
    /// Debug-assert helper.
    pub fn all_resolved(&self) -> bool {
        self.segments
            .iter()
            .all(|seg| seg.iter().all(|&v| v < MARKER_BASE))
    }

    /// Read the value at logical index `i` (panics out of range). Used
    /// by tests asserting resolved values.
    #[allow(dead_code)]
    pub fn at(&self, i: usize) -> u16 {
        self.get(i).expect("index out of range")
    }

    /// IN-PLACE resolve: turn this segmented u16 marker buffer into a
    /// [`ResolvedMarkers`] u8 view list, resolving every marker against
    /// `window` and writing the resolved u8 bytes into the LOW HALF of
    /// each segment's OWN backing store (vendor `applyWindow`
    /// in-place, DecodedData.hpp:316-337). No separate `narrowed`
    /// allocation; the resolved bytes live in the marker buffer's
    /// pages. Returns a `ResolvedMarkers` that owns the (now u8-bearing)
    /// segments and exposes `&[u8]` views.
    ///
    /// Each segment is resolved with the branchless 64 KiB LUT
    /// (`window[chunk[i]]` for markers, identity for literals),
    /// matching vendor's hot loop. The LUT is built ONCE and reused
    /// across segments.
    ///
    /// `window.len()` must be 32 KiB. The result's total byte length
    /// equals `self.len()`.
    pub fn resolve_in_place(mut self, window: &[u8]) -> ResolvedMarkers {
        debug_assert_eq!(
            window.len(),
            32768,
            "in-place resolve requires 32 KiB window"
        );
        // Build the LUT once (vendor DecodedData.hpp:316-326).
        let mut lut = [0u8; 65536];
        for (i, slot) in lut[0..256].iter_mut().enumerate() {
            *slot = i as u8;
        }
        lut[MARKER_BASE as usize..MARKER_BASE as usize + 32768].copy_from_slice(window);

        let total = self.cached_len;
        let segments = std::mem::take(&mut self.segments);
        let mut byte_lens = Vec::with_capacity(segments.len());
        let mut u16_segments = Vec::with_capacity(segments.len());
        for mut seg in segments {
            let n = seg.len();
            byte_lens.push(n);
            // Resolve markers in place into the low half of the same
            // backing store. Read element `i` (u16 at byte offset 2i),
            // write byte at offset `i`. Since `i <= 2i`, the write
            // never clobbers an unread element going left-to-right.
            //
            // SAFETY: `seg` owns `n` initialized u16 (= 2n bytes) of
            // backing storage; we write `n` u8 into the first `n`
            // bytes of that same storage. Pointers stay within the
            // allocation. We keep `seg` (a `Vec<u16>`) alive in
            // `u16_segments` so it is freed with the correct u16
            // Layout — we never reinterpret-and-free.
            let base = seg.as_mut_ptr() as *mut u8;
            let src = seg.as_ptr();
            for i in 0..n {
                // SAFETY: read element i (in [0,n)), within the Vec.
                let v = unsafe { *src.add(i) };
                let b = lut[v as usize];
                // SAFETY: byte offset i < n <= 2n bytes owned.
                unsafe {
                    base.add(i).write(b);
                }
            }
            u16_segments.push(seg);
        }
        ResolvedMarkers {
            u16_segments,
            byte_lens,
            total_bytes: total,
        }
    }
}

/// The result of [`SegmentedU16::resolve_in_place`]: the marker buffer's
/// own segment allocations, now carrying the resolved u8 bytes in their
/// low halves. Exposes `&[u8]` views for the consumer write path + CRC,
/// and returns the underlying u16 segments to the recycler on
/// `into_segments`.
///
/// The u8 bytes live at `segment.as_ptr() as *const u8` for `byte_len`
/// bytes per segment. We retain the `Vec<u16>` so the allocation is
/// freed with the correct Layout.
#[derive(Debug, Default)]
pub struct ResolvedMarkers {
    u16_segments: Vec<U16>,
    // Manual `Clone` below (deep-copy): a u16 segment carries resolved u8
    // bytes in its low half; a naive derived Clone would copy the u16
    // logical contents, not the u8 reinterpretation. `ChunkData` derives
    // `Clone` for its rare cache-promote path, so we provide a faithful
    // deep clone that reconstructs the byte segments.
    /// Resolved u8 byte count per segment (== the segment's prior u16
    /// element count).
    byte_lens: Vec<usize>,
    total_bytes: usize,
}

impl Clone for ResolvedMarkers {
    fn clone(&self) -> Self {
        // Deep-copy each segment's resolved u8 bytes into a fresh
        // u16-backed segment's low half (so `byte_segments` reads them
        // back correctly). Rare path (ChunkData cache-promote clone).
        let mut u16_segments = Vec::with_capacity(self.u16_segments.len());
        for (seg, &blen) in self.u16_segments.iter().zip(self.byte_lens.iter()) {
            // SAFETY: source segment holds `blen` resolved u8 bytes in its
            // low half (invariant of `resolve_in_place`).
            let src_bytes = unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, blen) };
            let mut dst = new_segment();
            // Ensure capacity for blen u16 (= 2*blen bytes) then write the
            // u8 bytes into the low half via raw ptr.
            if dst.capacity() < blen {
                dst.reserve(blen - dst.capacity());
            }
            let base = dst.as_mut_ptr() as *mut u8;
            // SAFETY: dst owns >= blen u16 (= 2*blen bytes) capacity.
            unsafe {
                std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), base, blen);
                // Set logical u16 len so the allocation is freed correctly;
                // ceil(blen/2) u16 elements cover blen bytes.
                dst.set_len(blen.div_ceil(2));
            }
            u16_segments.push(dst);
        }
        Self {
            u16_segments,
            byte_lens: self.byte_lens.clone(),
            total_bytes: self.total_bytes,
        }
    }
}

impl ResolvedMarkers {
    #[inline]
    pub fn len(&self) -> usize {
        self.total_bytes
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_bytes == 0
    }

    /// Iterate the resolved u8 segments in append order. The consumer
    /// write path does `for s in r.byte_segments() { writer.write_all(s)?; }`.
    pub fn byte_segments(&self) -> impl Iterator<Item = &[u8]> {
        self.u16_segments
            .iter()
            .zip(self.byte_lens.iter())
            .map(|(seg, &blen)| {
                // SAFETY: `resolve_in_place` wrote `blen` resolved u8
                // bytes into the low `blen` bytes of this segment's
                // backing store; the segment owns >= blen bytes
                // (blen u16 = 2*blen bytes).
                unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, blen) }
            })
    }

    /// Return the underlying u16 segment allocations to the recycler.
    /// Clears each segment's logical length first (the recycler expects
    /// empty Vecs).
    pub fn into_segments(mut self) -> Vec<U16> {
        for seg in &mut self.u16_segments {
            seg.clear();
        }
        std::mem::take(&mut self.u16_segments)
    }
}

/// The bootstrap decodes DIRECTLY into a `SegmentedU16` via this impl —
/// `push_slice` is the append-only write path (the inner decoder's own
/// 32 KiB output_ring resolves all back-refs, so the sink never reads
/// itself), and the two tail accessors serve the clean-window arming.
impl crate::decompress::parallel::marker_inflate::MarkerSink for SegmentedU16 {
    #[inline]
    fn push_slice(&mut self, values: &[u16]) {
        SegmentedU16::push_slice(self, values);
    }
    #[inline]
    fn sink_len(&self) -> usize {
        self.cached_len
    }
    #[inline]
    fn as_slice(&self) -> &[u16] {
        // Bootstrap must use `trailing_clean_since` / `copy_last_n_clean_u8`
        // — segmented storage is not one contiguous slice.
        &[]
    }
    fn trailing_clean_since(&self, from: usize) -> usize {
        if from >= self.cached_len {
            return 0;
        }
        let mut run = 0usize;
        let mut i = self.cached_len;
        while i > from {
            i -= 1;
            match self.get(i) {
                Some(v) if v < MARKER_BASE => run += 1,
                _ => break,
            }
        }
        run
    }
    fn copy_last_n_clean_u8(&self, n: usize, out: &mut Vec<u8>) -> bool {
        out.clear();
        if n == 0 || n > self.cached_len {
            return false;
        }
        out.resize(n, 0);
        if !self.append_last_n_clean_bytes(n, out.as_mut_slice()) {
            out.clear();
            return false;
        }
        true
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(off: u16) -> u16 {
        MARKER_BASE + off
    }

    #[test]
    fn push_within_one_segment() {
        let mut b = SegmentedU16::default();
        b.push_slice(&[1, 2, 3]);
        assert_eq!(b.len(), 3);
        assert_eq!(b.segment_count(), 1);
        assert!(!b.is_empty());
    }

    #[test]
    fn push_crosses_segment_boundary() {
        let mut b = SegmentedU16::default();
        let a = vec![7u16; SEGMENT_ELEMENTS - 5];
        let c = vec![9u16; 20];
        b.push_slice(&a);
        b.push_slice(&c);
        assert_eq!(b.len(), SEGMENT_ELEMENTS + 15);
        assert_eq!(b.segment_count(), 2);
        let cat: Vec<u16> = b.segments().flatten().copied().collect();
        assert_eq!(cat[..a.len()], a[..]);
        assert_eq!(cat[a.len()..], c[..]);
    }

    #[test]
    fn push_one_huge_slice_fills_many_segments() {
        let mut b = SegmentedU16::default();
        let total = SEGMENT_ELEMENTS * 3 + 777;
        let big: Vec<u16> = (0..total).map(|i| (i & 0x7FFF) as u16).collect();
        b.push_slice(&big);
        assert_eq!(b.len(), total);
        assert_eq!(b.segment_count(), 4);
        let cat: Vec<u16> = b.segments().flatten().copied().collect();
        assert_eq!(cat, big);
    }

    #[test]
    fn copy_last_n_walks_two_segments() {
        let mut b = SegmentedU16::default();
        let total = SEGMENT_ELEMENTS + 100;
        let big: Vec<u16> = (0..total).map(|i| (i & 0x7FFF) as u16).collect();
        b.push_slice(&big);
        let mut out = Vec::new();
        let got = b.copy_last_n(300, &mut out);
        assert_eq!(got, 300);
        assert_eq!(out, big[total - 300..]);
    }

    #[test]
    fn copy_last_n_clamps_to_len() {
        let mut b = SegmentedU16::default();
        b.push_slice(&[1, 2, 3]);
        let mut out = Vec::new();
        let got = b.copy_last_n(10, &mut out);
        assert_eq!(got, 3);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn trailing_clean_run_counts_to_first_marker() {
        let mut b = SegmentedU16::default();
        let v = vec![1u16, 2, marker(5), 10, 11, 12];
        b.push_slice(&v);
        assert_eq!(b.trailing_clean_run(32768), 3);
        let all_clean = vec![1u16; 50];
        let mut b2 = SegmentedU16::default();
        b2.push_slice(&all_clean);
        assert_eq!(b2.trailing_clean_run(20), 20);
        assert_eq!(b2.trailing_clean_run(100), 50);
    }

    #[test]
    fn trailing_clean_run_crosses_segment_boundary() {
        let mut b = SegmentedU16::default();
        b.push_slice(&vec![1u16; SEGMENT_ELEMENTS]);
        b.push_slice(&[2u16; 100]);
        assert_eq!(
            b.trailing_clean_run(SEGMENT_ELEMENTS + 100),
            SEGMENT_ELEMENTS + 100
        );
        let mut b2 = SegmentedU16::default();
        let mut first = vec![1u16; SEGMENT_ELEMENTS];
        first[SEGMENT_ELEMENTS - 10] = marker(3);
        b2.push_slice(&first);
        b2.push_slice(&[2u16; 100]);
        // trailing clean = 9 (after the marker in seg1) + 100 in seg2 = 109
        assert_eq!(b2.trailing_clean_run(32768), 109);
    }

    #[test]
    fn rposition_last_marker_across_segments() {
        let mut b = SegmentedU16::default();
        let total = SEGMENT_ELEMENTS + 50;
        let mut v: Vec<u16> = vec![1u16; total];
        v[10] = marker(0);
        v[SEGMENT_ELEMENTS + 5] = marker(1);
        b.push_slice(&v);
        assert_eq!(b.rposition_last_marker(), Some(SEGMENT_ELEMENTS + 5));
        let mut b2 = SegmentedU16::default();
        b2.push_slice(&[1, 2, 3]);
        assert_eq!(b2.rposition_last_marker(), None);
    }

    #[test]
    fn truncate_drops_tail() {
        let mut b = SegmentedU16::default();
        let total = SEGMENT_ELEMENTS * 2 + 30;
        b.push_slice(&(0..total).map(|i| (i & 0x7FFF) as u16).collect::<Vec<_>>());
        b.truncate(SEGMENT_ELEMENTS + 7);
        assert_eq!(b.len(), SEGMENT_ELEMENTS + 7);
        let cat: Vec<u16> = b.segments().flatten().copied().collect();
        assert_eq!(cat.len(), SEGMENT_ELEMENTS + 7);
    }

    #[test]
    fn get_indexes_across_segments() {
        let mut b = SegmentedU16::default();
        let total = SEGMENT_ELEMENTS + 5;
        b.push_slice(&(0..total).map(|i| (i & 0x7FFF) as u16).collect::<Vec<_>>());
        assert_eq!(b.get(0), Some(0));
        assert_eq!(
            b.get(SEGMENT_ELEMENTS),
            Some((SEGMENT_ELEMENTS & 0x7FFF) as u16)
        );
        assert_eq!(b.get(total), None);
    }

    #[test]
    fn resolve_range_into_resolves_markers() {
        let mut b = SegmentedU16::default();
        let window: Vec<u8> = (0..32768).map(|i| (i & 0xFF) as u8).collect();
        b.push_slice(&[b'a' as u16, marker(0), marker(255), b'z' as u16]);
        let mut out = Vec::new();
        b.resolve_range_into(0, 4, &window, &mut out);
        assert_eq!(out, vec![b'a', 0, 255, b'z']);
        b.resolve_range_into(1, 2, &window, &mut out);
        assert_eq!(out, vec![0, 255]);
    }

    #[test]
    fn resolve_range_into_buf_matches_vec_form() {
        let mut b = SegmentedU16::default();
        let window: Vec<u8> = (0..32768).map(|i| (i & 0xFF) as u8).collect();
        let total = SEGMENT_ELEMENTS + 200;
        let mut input: Vec<u16> = Vec::with_capacity(total);
        for i in 0..total {
            if i % 5 == 0 {
                input.push(marker((i % 32768) as u16));
            } else {
                input.push((i & 0xFF) as u16);
            }
        }
        b.push_slice(&input);
        for (from, len) in [(0, 50), (SEGMENT_ELEMENTS - 10, 40), (total - 100, 100)] {
            let mut vec_out = Vec::new();
            b.resolve_range_into(from, len, &window, &mut vec_out);
            let mut buf_out = vec![0u8; len];
            b.resolve_range_into_buf(from, len, &window, &mut buf_out);
            assert_eq!(vec_out, &buf_out[..vec_out.len()]);
        }
    }

    #[test]
    fn resolve_in_place_produces_correct_bytes_and_reuses_storage() {
        let mut b = SegmentedU16::default();
        let window: Vec<u8> = (0..32768).map(|i| ((i * 7 + 1) & 0xFF) as u8).collect();
        let total = SEGMENT_ELEMENTS + 1000;
        let mut input: Vec<u16> = Vec::with_capacity(total);
        for i in 0..total {
            if i % 3 == 0 {
                input.push(marker((i % 32768) as u16));
            } else {
                input.push((i & 0xFF) as u16);
            }
        }
        b.push_slice(&input);
        let n_segments = b.segment_count();
        assert!(n_segments >= 2);

        let resolved = b.resolve_in_place(&window);
        assert_eq!(resolved.len(), total);
        let got: Vec<u8> = resolved.byte_segments().flatten().copied().collect();
        assert_eq!(got.len(), total);
        let expected: Vec<u8> = input
            .iter()
            .map(|&v| {
                if v >= MARKER_BASE {
                    window[(v - MARKER_BASE) as usize]
                } else {
                    v as u8
                }
            })
            .collect();
        assert_eq!(got, expected);
        let segs = resolved.into_segments();
        assert_eq!(segs.len(), n_segments);
        assert!(segs.iter().all(|s| s.is_empty()));
        assert!(segs.iter().all(|s| s.capacity() >= 1));
    }

    #[test]
    fn from_and_take_segments_round_trip() {
        let mut b = SegmentedU16::default();
        b.push_slice(&[1, 2, 3, 4]);
        let segs = b.take_segments();
        assert!(b.is_empty());
        let restored = SegmentedU16::from_segments(segs);
        assert_eq!(restored.len(), 4);
        let cat: Vec<u16> = restored.segments().flatten().copied().collect();
        assert_eq!(cat, vec![1, 2, 3, 4]);
    }
}
