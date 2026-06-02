#![cfg(parallel_sm)]

//! Segmented `Vec<u8>` replacement for `ChunkData::data`.
//!
//! Port of rapidgzip's `DecodedData` segmented-buffer pattern at
//! `vendor/.../rapidgzip/DecodedData.hpp:239-289`. Each `SegmentedU8`
//! is a `Vec<Vec<u8, RpmallocAlloc>>` where every inner Vec is sized
//! to [`ALLOCATION_CHUNK_SIZE`] (= 128 KiB) — strictly below
//! rpmalloc's `LARGE_SIZE_LIMIT` (= 2 MiB,
//! `rpmalloc-sys lib.rs:51-54`). This keeps every inner allocation
//! inside rpmalloc's per-thread span cache instead of routing through
//! the `huge_alloc` mmap path that bypasses the cache.
//!
//! Vendor's measurement (`ChunkData.hpp:35-65`) showed an
//! 8 → 21 GB/s speedup (2.6×) switching from 4 MiB granules to
//! 128 KiB-1 MiB granules. 128 KiB hit the bandwidth optimum in their
//! sweep.
//!
//! TMA (2026-05-29, this tree) confirms gzippy is memory-bound (26.2%
//! vs rapidgzip 15.6%) at parity core-bound — i.e. the gap is this
//! buffer model, not codegen.
//!
//! This module ships the u8 variant first; `data_with_markers` and
//! `narrowed` may be segmented in follow-ups if the lever proves out.

use crate::decompress::parallel::rpmalloc_alloc::types::U8;

/// Allocate (or recycle) one empty 128 KiB segment. Pulls from the current
/// worker's u8 pool (warm, pre-faulted pages) when available, falling back to
/// a fresh `Vec::with_capacity(ALLOCATION_CHUNK_SIZE)` on miss. Returned
/// segments are recycled to the same pool by `ChunkData::Drop`.
#[inline]
fn new_segment() -> U8 {
    use crate::decompress::parallel::chunk_buffer_pool;
    let mut seg = chunk_buffer_pool::take_u8(ALLOCATION_CHUNK_SIZE);
    seg.clear();
    if seg.capacity() < ALLOCATION_CHUNK_SIZE {
        seg.reserve_exact(ALLOCATION_CHUNK_SIZE - seg.capacity());
    }
    seg
}

/// Vendor's `ALLOCATION_CHUNK_SIZE` (`ChunkData.hpp:65`). Each
/// inner Vec is reserved to exactly this byte capacity. Sized to
/// fit comfortably below rpmalloc's `LARGE_SIZE_LIMIT = 2 MiB`
/// while staying small enough that the per-segment allocation
/// overhead (rpmalloc thread-cache pop/push) is amortized across
/// the bytes the segment holds.
pub const ALLOCATION_CHUNK_SIZE: usize = 128 * 1024;

/// Vec<u8>-shaped segmented buffer. Each segment is a
/// `Vec<u8, RpmallocAlloc>` capped at [`ALLOCATION_CHUNK_SIZE`].
///
/// Cloned via the derived impl: a deep clone of each segment. Used
/// by `ChunkData::Clone` in the cache-promote path; segment-level
/// clone is cheaper than monolithic 80 MiB clone because each
/// segment is a separate rpmalloc span-class allocation.
#[derive(Debug, Default, Clone)]
pub struct SegmentedU8 {
    /// Owned 128 KiB inner buffers. Append populates the last one
    /// until it hits ALLOCATION_CHUNK_SIZE, then allocates a new
    /// segment via `types::u8_with_capacity(ALLOCATION_CHUNK_SIZE)`.
    segments: Vec<U8>,
    /// Cached total byte count (sum of segment lens). Maintained by
    /// every mutating method so `len()` is O(1) and `is_empty()` is
    /// a one-comparison check rather than an iteration over
    /// `segments`. Vendor parity: `DecodedData::size()` returns the
    /// cached `m_size` member.
    cached_len: usize,
}

impl SegmentedU8 {
    /// Total byte count across all segments. O(1).
    #[inline]
    pub fn len(&self) -> usize {
        self.cached_len
    }

    /// True iff every segment is empty (or the buffer has no segments).
    /// O(1).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }

    /// True when all logical bytes still fit in segment 0 (A3 decode uses a
    /// contiguous view of that segment's backing allocation).
    #[inline]
    pub fn all_in_first_segment(&self) -> bool {
        self.segments.len() <= 1
    }

    /// Segment-native Option A3: install the predecessor's 32 KiB sliding window
    /// at the front of segment 0 (128 KiB capacity — room for prefill + decode).
    /// Must run before any decoded bytes are appended (`len() == 0`).
    pub fn prefill_window_prefix(&mut self, window: &[u8]) {
        debug_assert!(
            self.cached_len == 0,
            "prefill_window_prefix before any decoded bytes"
        );
        debug_assert!(
            window.len() <= ALLOCATION_CHUNK_SIZE,
            "window prefix exceeds one segment"
        );
        if window.is_empty() {
            return;
        }
        let mut seg = new_segment();
        seg.extend_from_slice(window);
        if seg.capacity() < ALLOCATION_CHUNK_SIZE {
            seg.reserve_exact(ALLOCATION_CHUNK_SIZE - seg.capacity());
        }
        self.segments.push(seg);
        self.cached_len = window.len();
    }

    /// Mutable `[0, capacity)` view of segment 0 for A3 `read_stream_starting_at`.
    /// The decoder writes at `out_pos_start = self.len()`; back-references into
    /// the prefilled prefix resolve via `output[..out_pos]`. Only valid while
    /// [`Self::all_in_first_segment`] holds.
    ///
    /// SAFETY contract: caller writes only at indices `>= len()` before `commit`;
    /// same as the monolithic `Vec` handoff this replaces.
    pub fn first_segment_a3_output(&mut self) -> &mut [u8] {
        debug_assert_eq!(
            self.segments.len(),
            1,
            "A3 contiguous view requires a single segment"
        );
        let seg = &mut self.segments[0];
        debug_assert!(
            seg.capacity() >= ALLOCATION_CHUNK_SIZE,
            "segment 0 must reserve {} bytes at prefill time",
            ALLOCATION_CHUNK_SIZE
        );
        let cap = seg.capacity();
        // SAFETY: `cap` bytes are allocated; bytes `[0, len())` are initialized
        // (prefill and/or prior commits); the tail is writable spare capacity.
        unsafe { std::slice::from_raw_parts_mut(seg.as_mut_ptr(), cap) }
    }

    /// Write every decoded payload byte, skipping the first `skip_prefix` logical
    /// bytes (the A3 window image at the front of segment 0).
    pub fn write_payload_skipping_prefix<W: std::io::Write>(
        &self,
        skip_prefix: usize,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let mut skip = skip_prefix;
        for seg in &self.segments {
            if skip >= seg.len() {
                skip -= seg.len();
                continue;
            }
            writer.write_all(&seg[skip..])?;
            skip = 0;
        }
        Ok(())
    }

    /// Truncate every segment to zero length, retaining the
    /// underlying allocations for reuse. Matches `Vec::clear` for
    /// the API surface ChunkData callers rely on (`chunk.data.clear()`
    /// in tests + recycler return paths). Allocations stay owned;
    /// next append reuses them.
    pub fn clear(&mut self) {
        for seg in &mut self.segments {
            seg.clear();
        }
        self.cached_len = 0;
    }

    /// Append a clean (u8, marker-free) byte slice. Distributes
    /// across segments at the [`ALLOCATION_CHUNK_SIZE`] boundary —
    /// the LAST segment fills first, then a new segment is allocated
    /// for the overflow. Mirror of vendor's
    /// `appendToEquallySizedChunks` at `DecodedData.hpp:243-258`.
    pub fn extend_from_slice(&mut self, bytes: &[u8]) {
        let mut src = bytes;
        while !src.is_empty() {
            // Ensure we have a tail segment with room.
            let needs_new = match self.segments.last() {
                Some(seg) => seg.len() >= seg.capacity() || seg.len() >= ALLOCATION_CHUNK_SIZE,
                None => true,
            };
            if needs_new {
                self.segments.push(new_segment());
            }
            // SAFETY: just ensured last segment exists and has room.
            let last = self.segments.last_mut().unwrap();
            let room = ALLOCATION_CHUNK_SIZE - last.len();
            let n = src.len().min(room);
            last.extend_from_slice(&src[..n]);
            self.cached_len += n;
            src = &src[n..];
        }
    }

    /// Zero-copy decode sink: return a writable slice into the tail
    /// segment's spare capacity for a decoder to write DIRECTLY into
    /// (no intermediate buffer, no copy). Allocates a fresh 128 KiB
    /// segment when the tail is full, so the returned slice is always
    /// `[0, ALLOCATION_CHUNK_SIZE - tail.len())` bytes — uninitialized
    /// but writable. After the decoder writes N bytes, call
    /// [`Self::commit`] to record them. This is how `ResumableInflate2`
    /// (resumable: stops when `output` fills) decodes a chunk one
    /// segment at a time — its 32 KiB window ring resolves back-refs
    /// across segment boundaries (deflate max distance 32 KiB ≤ the
    /// 128 KiB segment, so the ring always covers them).
    ///
    /// SAFETY contract: the caller must only WRITE into the returned
    /// slice (it aliases uninitialized spare capacity) and must call
    /// `commit(n)` with `n <= slice.len()` before the next mutating call.
    pub fn writable_tail(&mut self) -> &mut [u8] {
        let needs_new = match self.segments.last() {
            Some(seg) => seg.len() >= seg.capacity() || seg.len() >= ALLOCATION_CHUNK_SIZE,
            None => true,
        };
        if needs_new {
            self.segments.push(new_segment());
        }
        let last = self.segments.last_mut().unwrap();
        let len = last.len();
        let spare = last
            .capacity()
            .saturating_sub(len)
            .min(ALLOCATION_CHUNK_SIZE.saturating_sub(len));
        debug_assert!(
            spare > 0,
            "writable_tail: no spare capacity (len {} cap {})",
            len,
            last.capacity()
        );
        // SAFETY: `[len, len+spare)` lies within the segment's allocation;
        // the caller writes before any read, then calls `commit`.
        unsafe { std::slice::from_raw_parts_mut(last.as_mut_ptr().add(len), spare) }
    }

    /// Record `n` bytes written into the slice returned by
    /// [`Self::writable_tail`]. Bumps the tail segment's length and the
    /// cached total. Panics in debug if `n` exceeds the tail's spare.
    pub fn commit(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let last = self
            .segments
            .last_mut()
            .expect("commit without a preceding writable_tail");
        let new_len = last.len() + n;
        debug_assert!(
            new_len <= last.capacity(),
            "commit {n} overflows tail spare (len {} cap {})",
            last.len(),
            last.capacity()
        );
        // SAFETY: bytes `[old_len, old_len+n)` were just written by the
        // caller through the `writable_tail` slice; `new_len <= capacity`.
        unsafe {
            last.set_len(new_len);
        }
        self.cached_len += n;
    }

    /// Iterate over the segments as byte slices, in append order.
    /// Used by the consumer's write path:
    ///
    /// ```ignore
    /// for seg in chunk.data.segments() {
    ///     writer.write_all(seg)?;
    /// }
    /// ```
    ///
    /// Replaces the prior `writer.write_all(&chunk.data)` call site
    /// (which assumed a contiguous `&[u8]`).
    #[inline]
    pub fn segments(&self) -> impl Iterator<Item = &[u8]> {
        self.segments.iter().map(|v| v.as_slice())
    }

    /// Number of populated segments. Used by stats / diagnostics;
    /// not part of the Vec API surface.
    #[allow(dead_code)] // diagnostic surface
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Sum of capacities across segments. Used by recycler-return
    /// counters (`reused_MB`-style stats). Vendor parity: vendor's
    /// per-chunk `cleanDataCount` aggregates the same way across
    /// the chunked storage.
    pub fn capacity(&self) -> usize {
        self.segments.iter().map(|s| s.capacity()).sum()
    }

    /// Drop in-place return for the chunk-buffer pool. Recycler
    /// pushes each owned segment back into the worker's per-segment
    /// LIFO so subsequent chunks reuse 128 KiB chunks at a time
    /// (each within rpmalloc's span-cache scope).
    pub fn take_segments(&mut self) -> Vec<U8> {
        self.cached_len = 0;
        std::mem::take(&mut self.segments)
    }

    /// Construct from a Vec of owned segments. Used by the recycler
    /// to give chunks a head-start with pre-allocated segments.
    pub fn from_segments(segments: Vec<U8>) -> Self {
        let cached_len = segments.iter().map(|s| s.len()).sum();
        Self {
            segments,
            cached_len,
        }
    }

    /// Reserve at least `additional` more bytes of total capacity,
    /// allocated in [`ALLOCATION_CHUNK_SIZE`]-sized segments. Mirrors
    /// `Vec::reserve` but the underlying allocations are chunked.
    pub fn reserve(&mut self, additional: usize) {
        let current_cap = self.capacity();
        let needed = self.len() + additional;
        if needed <= current_cap {
            return;
        }
        let extra = needed - current_cap;
        let n_segments = extra.div_ceil(ALLOCATION_CHUNK_SIZE);
        for _ in 0..n_segments {
            self.segments.push(new_segment());
        }
    }

    /// Truncate the logical buffer to at most `new_len` bytes. Empties
    /// trailing segments first, then truncates the partial segment.
    /// Mirrors `Vec::truncate`.
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

    /// Copy the logical contents into a single contiguous `Vec<u8>`.
    /// O(n). Used by call sites that genuinely need a contiguous slice
    /// (test helpers, defensive fallbacks) — the hot write path uses
    /// `segments()` to write segment-by-segment without copying.
    #[allow(dead_code)] // call sites add later as needed
    pub fn to_contiguous(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.cached_len);
        for seg in &self.segments {
            out.extend_from_slice(seg);
        }
        out
    }

    /// In-place index access. O(segments) per call — avoid in tight
    /// loops; prefer `segments()` iteration. Returns `None` if out
    /// of bounds.
    pub fn get(&self, mut index: usize) -> Option<u8> {
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

    /// Mutable index access. Same O(segments) cost as `get`. Used
    /// by post-process / apply_window for in-place mutation of
    /// specific bytes (rare; most apply_window work is segment-major).
    #[allow(dead_code)] // wire as call sites land
    pub fn get_mut(&mut self, mut index: usize) -> Option<&mut u8> {
        if index >= self.cached_len {
            return None;
        }
        for seg in &mut self.segments {
            if index < seg.len() {
                return Some(&mut seg[index]);
            }
            index -= seg.len();
        }
        None
    }

    /// Iterate over each byte in append order. Used by callers that
    /// genuinely need byte-by-byte iteration (CRC paths can use
    /// `segments()` more efficiently — prefer that when possible).
    pub fn iter_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.segments.iter().flat_map(|s| s.iter().copied())
    }

    /// Copy the last `n` logical bytes into `out` (which must be exactly
    /// `n` long). Used to source a chunk's trailing 32 KiB sliding window
    /// (`last_32kib_window` / `get_last_window`). `n` must be ≤ `len()`.
    /// Walks segments from the back, filling `out` from its end.
    pub fn copy_last_into(&self, out: &mut [u8]) {
        let n = out.len();
        debug_assert!(
            n <= self.cached_len,
            "copy_last_into: n {n} > len {}",
            self.cached_len
        );
        let mut remaining = n; // bytes still to fill, from the end of `out`
        for seg in self.segments.iter().rev() {
            if remaining == 0 {
                break;
            }
            let take = remaining.min(seg.len());
            // bytes [seg.len()-take, seg.len()) of this segment land at
            // [remaining-take, remaining) of `out`.
            out[remaining - take..remaining].copy_from_slice(&seg[seg.len() - take..]);
            remaining -= take;
        }
        debug_assert_eq!(remaining, 0, "copy_last_into under-filled");
    }

    /// Prepend `bytes` as new segment(s) at the FRONT of the buffer (the
    /// bytes become the logical prefix, before all existing segments).
    /// Mirror of vendor's `dataBuffers.emplace(dataBuffers.begin(), ...)` in
    /// `cleanUnmarkedData` (DecodedData.hpp:502): when a marker chunk's
    /// trailing run turns out to be clean, those resolved bytes are moved to
    /// the front of `data`. `bytes` is ≤ one chunk's marker prefix; we pack
    /// it into 128 KiB front segments to keep the segment-size invariant.
    pub fn prepend_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // Build the new front segments (128 KiB each), then splice before the
        // existing segments.
        let mut front: Vec<U8> = Vec::new();
        let mut src = bytes;
        while !src.is_empty() {
            let n = src.len().min(ALLOCATION_CHUNK_SIZE);
            let mut seg = new_segment();
            seg.extend_from_slice(&src[..n]);
            front.push(seg);
            src = &src[n..];
        }
        front.append(&mut self.segments);
        self.segments = front;
        self.cached_len += bytes.len();
    }

    /// Copy the logical byte range `[start, start + out.len())` into `out`.
    /// Walks segments, skipping to `start` then filling `out` across segment
    /// boundaries. Used by the window-construction paths (`get_last_window`)
    /// which read an arbitrary sub-range of the clean decoded bytes.
    pub fn copy_range_into(&self, start: usize, out: &mut [u8]) {
        let n = out.len();
        debug_assert!(
            start + n <= self.cached_len,
            "copy_range_into: [{start}, {}) > len {}",
            start + n,
            self.cached_len
        );
        let mut skip = start;
        let mut written = 0usize;
        for seg in &self.segments {
            if written == n {
                break;
            }
            if skip >= seg.len() {
                skip -= seg.len();
                continue;
            }
            let avail = seg.len() - skip;
            let take = avail.min(n - written);
            out[written..written + take].copy_from_slice(&seg[skip..skip + take]);
            written += take;
            skip = 0;
        }
        debug_assert_eq!(written, n, "copy_range_into under-filled");
    }

    /// Truncate this buffer to the first `at` logical bytes; return the
    /// suffix as a new `SegmentedU8`. Used to insert bytes after the A3
    /// window prefix inside segment 0 (`clean_unmarked_data`).
    pub fn split_off(&mut self, at: usize) -> SegmentedU8 {
        debug_assert!(at <= self.cached_len);
        if at == 0 {
            let mut tail = SegmentedU8::default();
            std::mem::swap(&mut tail.segments, &mut self.segments);
            tail.cached_len = self.cached_len;
            self.cached_len = 0;
            return tail;
        }
        if at == self.cached_len {
            return SegmentedU8::default();
        }
        let mut tail = SegmentedU8::default();
        let mut logical = 0usize;
        let mut split_seg = 0usize;
        let mut split_off = 0usize;
        for (i, seg) in self.segments.iter().enumerate() {
            if logical + seg.len() > at {
                split_seg = i;
                split_off = at - logical;
                break;
            }
            logical += seg.len();
        }
        // Tail of the split segment goes to `tail`.
        let split_seg_len = self.segments[split_seg].len();
        if split_off < split_seg_len {
            let spill = self.segments[split_seg].split_off(split_off);
            let spill_len = spill.len();
            tail.segments.push(spill);
            tail.cached_len += spill_len;
        }
        for seg in self.segments.drain(split_seg + 1..) {
            tail.cached_len += seg.len();
            tail.segments.push(seg);
        }
        self.cached_len = at;
        tail
    }

    /// Insert `bytes` at logical offset `offset` (shifting the suffix right).
    pub fn insert_logical_at(&mut self, offset: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if offset == 0 {
            self.prepend_bytes(bytes);
            return;
        }
        if offset >= self.cached_len {
            self.extend_from_slice(bytes);
            return;
        }
        let mut tail = self.split_off(offset);
        self.extend_from_slice(bytes);
        self.append_segmented(&mut tail);
    }

    /// Append the entire logical contents of `other` onto `self`, moving
    /// `other`'s segments in WITHOUT copying their bytes when alignment
    /// permits (the common merge case: `self` empty → take all of
    /// `other`'s segments). Falls back to a byte copy of the spill when
    /// `self`'s tail segment is partially full. Mirror of the merge in
    /// `absorb_isal_tail`. Leaves `other` empty.
    pub fn append_segmented(&mut self, other: &mut SegmentedU8) {
        if other.cached_len == 0 {
            return;
        }
        if self.cached_len == 0 {
            // Zero-copy: adopt other's segments wholesale.
            self.segments = std::mem::take(&mut other.segments);
            self.cached_len = other.cached_len;
            other.cached_len = 0;
            return;
        }
        // General case: copy other's bytes in (rare on the hot path).
        for seg in other.segments.drain(..) {
            self.extend_from_slice(&seg);
        }
        other.cached_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_window_prefix_uses_segment_zero() {
        let mut buf = SegmentedU8::default();
        let window = vec![0xABu8; 32 * 1024];
        buf.prefill_window_prefix(&window);
        assert_eq!(buf.len(), window.len());
        assert_eq!(buf.segment_count(), 1);
        assert_eq!(buf.segments().next().unwrap(), window.as_slice());
    }

    #[test]
    fn extend_within_one_segment() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"hello");
        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        assert_eq!(buf.segment_count(), 1);
        let cat: Vec<u8> = buf.segments().flatten().copied().collect();
        assert_eq!(cat, b"hello");
    }

    #[test]
    fn extend_crosses_segment_boundary() {
        let mut buf = SegmentedU8::default();
        let chunk_a = vec![0xABu8; ALLOCATION_CHUNK_SIZE - 10];
        let chunk_b = vec![0xCDu8; 50];
        buf.extend_from_slice(&chunk_a);
        buf.extend_from_slice(&chunk_b);
        assert_eq!(buf.len(), ALLOCATION_CHUNK_SIZE + 40);
        assert_eq!(buf.segment_count(), 2);
        // Concatenating segments must reproduce the input.
        let cat: Vec<u8> = buf.segments().flatten().copied().collect();
        assert_eq!(cat[..chunk_a.len()], chunk_a[..]);
        assert_eq!(cat[chunk_a.len()..], chunk_b[..]);
    }

    #[test]
    fn extend_one_huge_slice_fills_many_segments() {
        let mut buf = SegmentedU8::default();
        let n_segments = 5;
        let total = ALLOCATION_CHUNK_SIZE * n_segments + 1234;
        let big: Vec<u8> = (0..total).map(|i| (i & 0xFF) as u8).collect();
        buf.extend_from_slice(&big);
        assert_eq!(buf.len(), total);
        assert_eq!(buf.segment_count(), n_segments + 1);
        let cat: Vec<u8> = buf.segments().flatten().copied().collect();
        assert_eq!(cat, big);
    }

    #[test]
    fn clear_preserves_segments() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(&vec![0u8; ALLOCATION_CHUNK_SIZE * 3]);
        assert_eq!(buf.segment_count(), 3);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        // Segments retained for reuse: capacity should still be > 0.
        assert!(buf.capacity() >= 3 * ALLOCATION_CHUNK_SIZE);
    }

    #[test]
    fn take_segments_releases_storage() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"abc");
        let segs = buf.take_segments();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].len(), 3);
        assert!(buf.is_empty());
        assert_eq!(buf.segment_count(), 0);
    }

    #[test]
    fn from_segments_round_trip() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"first batch");
        let segs = buf.take_segments();
        let restored = SegmentedU8::from_segments(segs);
        assert_eq!(restored.len(), 11);
        let cat: Vec<u8> = restored.segments().flatten().copied().collect();
        assert_eq!(cat, b"first batch");
    }

    #[test]
    fn writable_tail_commit_zero_copy_decode() {
        // Mimics a resumable decoder writing into segment spare capacity
        // directly, across the 128 KiB segment boundary.
        let mut buf = SegmentedU8::default();
        {
            let s = buf.writable_tail();
            assert_eq!(s.len(), ALLOCATION_CHUNK_SIZE);
            for b in s[..100 * 1024].iter_mut() {
                *b = 0x11;
            }
        }
        buf.commit(100 * 1024);
        assert_eq!(buf.len(), 100 * 1024);
        assert_eq!(buf.segment_count(), 1);

        let room;
        {
            let s = buf.writable_tail(); // same segment, 28 KiB spare left
            room = s.len();
            assert_eq!(room, ALLOCATION_CHUNK_SIZE - 100 * 1024);
            for b in s.iter_mut() {
                *b = 0x22;
            }
        }
        buf.commit(room);
        assert_eq!(buf.len(), ALLOCATION_CHUNK_SIZE);
        assert_eq!(buf.segment_count(), 1);

        {
            let s = buf.writable_tail(); // tail full -> fresh segment
            assert_eq!(s.len(), ALLOCATION_CHUNK_SIZE);
            s[0] = 0x33;
        }
        buf.commit(1);
        assert_eq!(buf.segment_count(), 2);
        assert_eq!(buf.len(), ALLOCATION_CHUNK_SIZE + 1);

        let cat = buf.to_contiguous();
        assert_eq!(cat[0], 0x11);
        assert_eq!(cat[100 * 1024 - 1], 0x11);
        assert_eq!(cat[100 * 1024], 0x22);
        assert_eq!(cat[ALLOCATION_CHUNK_SIZE], 0x33);
    }
}
