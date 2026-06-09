#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

//! Contiguous `Vec<u8>`-shaped buffer for `ChunkData::data` (the CLEAN,
//! marker-free decoded output).
//!
//! **Faithful port of rapidgzip's CLEAN-data storage** in
//! `DecodedData::append` (`vendor/.../rapidgzip/DecodedData.hpp:278-289`):
//! vendor stores the *marker* buffer (`dataWithMarkers`) in 128 KiB
//! equally-sized chunks via `appendToEquallySizedChunks` (243-258) — but
//! DELIBERATELY stores the clean `data` in single contiguous per-append
//! buffers. The comment at `DecodedData.hpp:278-281` states that forcing
//! the clean `dataBuffer` chunks to 128 KiB "makes no sense"; vendor does
//! `dataBuffers.emplace_back(); copied.reserve(buffers.dataSize()); …`.
//!
//! An earlier gzippy revision wrongly applied the 128 KiB-segmented
//! *marker* pattern (`Vec<Vec<u8>>`) to the *clean* buffer as well. That
//! measured 3.26x DTLB-walks / 1.42x cycles vs rapidgzip at equal
//! instruction count (1.01x) — the memory-bound gap. The clean bytes
//! belong in ONE contiguous allocation, grown by amortized reserve.
//!
//! The u16 marker buffer stays 128 KiB-segmented in
//! [`super::segmented_markers::SegmentedU16`] — that one MATCHES vendor's
//! `dataWithMarkers` and must NOT be changed.
//!
//! The public API is unchanged from the segmented version (same method
//! names/signatures) so the decode sink, A3 fast-path, marker-resolution,
//! window-construction and writev output paths are byte-for-byte
//! transparent to the switch — only the physical layout differs.

use super::segmented_markers::SegmentedU16;
use crate::decompress::parallel::rpmalloc_alloc::types::{self, U8};

/// Vendor's `ALLOCATION_CHUNK_SIZE` (`ChunkData.hpp:65`). Reused here as
/// the GROWTH GRANULARITY for the contiguous buffer's amortized reserve
/// and as the A3 single-shot decode window size — NOT a hard segment
/// boundary. The clean buffer is one contiguous allocation.
pub const ALLOCATION_CHUNK_SIZE: usize = 128 * 1024;

/// TEST-ONLY reserved-tail poison (OPT-IN via `GZIPPY_POISON_RESERVE`). The
/// clean-tail copy-free path writes u8 DIRECTLY into this reserved
/// (uninitialized) spare and back-refs may resolve from it; a read-BEFORE-write
/// bug (e.g. an off-by-one seam address, or a reserve-clamp that under-sizes the
/// contiguous tail) reads garbage. In a release build that garbage is allocator
/// memory — often nonzero, so a differential CATCHES it, but on freshly-zeroed
/// pages it can read as the correct byte and slip. Filling the spare with a
/// non-zero sentinel makes any read-before-write deterministically corrupt the
/// output, so the seam/diff nets fail every run rather than flakily
/// (advisor item (b)).
///
/// Compiled ONLY under `cfg(test)`, and even then gated behind the
/// `GZIPPY_POISON_RESERVE` env var so it is OPT-IN: the seam-correctness nets
/// (`seam_crossing`) set it; the default test suite (and any perf-timed test)
/// leaves the reserve memset-free so the poison's memset cost never perturbs a
/// timing gate. Production binaries (no `cfg(test)`) get a no-op that inlines
/// away — byte- and cost-transparent to the shipped decode.
#[inline(always)]
fn poison_reserved_tail(_spare: &mut [u8]) {
    #[cfg(test)]
    {
        thread_local! {
            static ENABLED: bool = std::env::var_os("GZIPPY_POISON_RESERVE").is_some();
        }
        if ENABLED.with(|e| *e) {
            // 0xCD: the classic "uninitialized" sentinel; any byte read before
            // being overwritten by a real decoded value surfaces as a miss.
            for b in _spare.iter_mut() {
                *b = 0xCD;
            }
        }
    }
}

/// Vec<u8>-shaped contiguous buffer. Backed by a single
/// `Vec<u8, RpmallocAlloc>` sourced from the worker buffer pool on first
/// write and grown by amortized `reserve`.
///
/// Cloned via the derived impl: a single contiguous deep clone (matches
/// vendor, whose `data` buffers are also contiguous per append). Used by
/// `ChunkData::Clone` in the cache-promote path.
#[derive(Debug, Clone)]
pub struct SegmentedU8 {
    /// Single contiguous backing store for all clean decoded bytes.
    /// `buf.len()` is the logical byte count; `buf.capacity() == 0`
    /// means "not yet sourced from the pool".
    buf: U8,
}

impl Default for SegmentedU8 {
    fn default() -> Self {
        Self {
            buf: types::u8_with_capacity(0),
        }
    }
}

impl SegmentedU8 {
    /// Lazily source the backing allocation from the current worker's u8
    /// pool (warm, pre-faulted pages) on first use, mirroring the old
    /// per-segment `new_segment()` pull. No-op once `buf` owns an
    /// allocation; `Vec`'s own amortized `reserve` handles later growth.
    #[inline]
    fn ensure_buf(&mut self, min_capacity: usize) {
        if self.buf.capacity() == 0 {
            use crate::decompress::parallel::chunk_buffer_pool;
            self.buf = chunk_buffer_pool::take_u8(min_capacity.max(ALLOCATION_CHUNK_SIZE));
        }
    }

    /// Total byte count. O(1).
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True iff no decoded bytes have been appended. O(1).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// True when the logical contents still fit in a single 128 KiB A3
    /// decode window. The backing store is ALWAYS contiguous now, but we
    /// preserve the prior A3-engagement gate so decode-path *selection*
    /// stays byte-identical to the segmented version — the only change is
    /// the physical layout the A3 view points into.
    #[inline]
    pub fn all_in_first_segment(&self) -> bool {
        self.buf.len() <= ALLOCATION_CHUNK_SIZE
    }

    /// Segment-native Option A3: install the predecessor's 32 KiB sliding
    /// window at the FRONT of the contiguous buffer. Must run before any
    /// decoded bytes are appended (`len() == 0`).
    pub fn prefill_window_prefix(&mut self, window: &[u8]) {
        debug_assert!(self.buf.is_empty(), "prefill before any decoded bytes");
        debug_assert!(
            window.len() <= ALLOCATION_CHUNK_SIZE,
            "window prefix exceeds one A3 window"
        );
        if window.is_empty() {
            return;
        }
        self.ensure_buf(ALLOCATION_CHUNK_SIZE);
        self.buf.extend_from_slice(window);
    }

    /// Mutable contiguous A3 decode window — a `[0, capacity)` view into
    /// the single backing allocation. The decoder writes at
    /// `out_pos_start = self.len()`; back-references into the prefilled
    /// prefix resolve via `output[..out_pos]`. Only meaningful while
    /// [`Self::all_in_first_segment`] holds (caller's gate).
    ///
    /// SAFETY contract: caller writes only at indices `>= len()` before
    /// `commit`; identical to the segmented version this replaces.
    pub fn first_segment_a3_output(&mut self) -> &mut [u8] {
        self.ensure_buf(ALLOCATION_CHUNK_SIZE);
        debug_assert!(self.buf.capacity() >= ALLOCATION_CHUNK_SIZE);
        let cap = self.buf.capacity();
        // SAFETY: `cap` bytes are allocated; `[0, len())` is initialized
        // (prefill and/or prior commits); the tail is writable spare.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr(), cap) }
    }

    /// Write every decoded payload byte, skipping the first `skip_prefix`
    /// logical bytes (the A3 window image at the front of the buffer).
    pub fn write_payload_skipping_prefix<W: std::io::Write>(
        &self,
        skip_prefix: usize,
        writer: &mut W,
    ) -> std::io::Result<()> {
        if skip_prefix >= self.buf.len() {
            return Ok(());
        }
        writer.write_all(&self.buf[skip_prefix..])
    }

    /// Collect payload slice refs for `writev` (skips `skip_prefix`
    /// logical bytes). One contiguous slice now.
    pub fn append_payload_iovecs<'a>(&'a self, skip_prefix: usize, out: &mut Vec<&'a [u8]>) {
        if skip_prefix >= self.buf.len() {
            return;
        }
        out.push(&self.buf[skip_prefix..]);
    }

    /// Truncate to zero length, retaining the allocation for reuse.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Append a clean (marker-free) byte slice. Contiguous, amortized
    /// growth. Mirror of vendor's contiguous clean-data append
    /// (`DecodedData.hpp:282-289`).
    pub fn extend_from_slice(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        self.ensure_buf(bytes.len());
        self.buf.extend_from_slice(bytes);
    }

    /// Zero-copy decode sink: return a writable window (up to
    /// [`ALLOCATION_CHUNK_SIZE`] bytes) of contiguous spare capacity at the
    /// tail for a decoder to write DIRECTLY into. After the decoder writes
    /// N bytes, call [`Self::commit`] to record them. The 128 KiB cap
    /// keeps the resumable decoder's per-call cadence (and its 32 KiB
    /// window ring) identical to the segmented version; the bytes are now
    /// physically contiguous with the prior tail.
    ///
    /// SAFETY contract: the caller must only WRITE into the returned slice
    /// (uninitialized spare) and call `commit(n)` with `n <= slice.len()`
    /// before the next mutating call.
    pub fn writable_tail(&mut self) -> &mut [u8] {
        self.ensure_buf(ALLOCATION_CHUNK_SIZE);
        let len = self.buf.len();
        if self.buf.capacity() == len {
            // Only grow once the contiguous spare is fully exhausted, so
            // the decoder's per-call window cadence matches the segmented
            // version (which handed out the current 128 KiB segment's
            // remainder, then a fresh segment). Amortized; may realloc +
            // move, but no live raw pointer spans this call — callers
            // re-fetch the tail each outer decode iteration.
            self.buf.reserve(ALLOCATION_CHUNK_SIZE);
        }
        let window = ALLOCATION_CHUNK_SIZE.min(self.buf.capacity() - len);
        debug_assert!(window > 0, "writable_tail: no spare capacity");
        // SAFETY: `[len, len+window)` lies within the allocation; the
        // caller writes before any read, then calls `commit`.
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(len), window) };
        poison_reserved_tail(slice);
        slice
    }

    /// Like [`Self::writable_tail`] but guarantees AT LEAST `min_spare` bytes of
    /// CONTIGUOUS spare capacity and returns the WHOLE spare region (not capped
    /// at [`ALLOCATION_CHUNK_SIZE`]). Used by the copy-free ISA-L oracle to give
    /// the FFI decoder one contiguous output buffer to write directly into, so
    /// no intermediate `Vec` + `copy_from_slice` confounds the WALL measurement.
    /// SAFETY contract identical to [`Self::writable_tail`]: write-then-`commit`.
    pub fn writable_tail_reserve(&mut self, min_spare: usize) -> &mut [u8] {
        self.ensure_buf(self.buf.len() + min_spare);
        let len = self.buf.len();
        if self.buf.capacity() - len < min_spare {
            self.buf.reserve(min_spare - (self.buf.capacity() - len));
        }
        let spare = self.buf.capacity() - len;
        debug_assert!(spare >= min_spare, "writable_tail_reserve: short spare");
        // SAFETY: `[len, len+spare)` lies within the allocation; caller writes
        // before any read, then calls `commit`.
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(len), spare) };
        poison_reserved_tail(slice);
        slice
    }

    /// Copy-free-to-final contig decode window (gzippy-native FOLD post-flip
    /// tail). Ensures at least `min_spare` bytes of CONTIGUOUS spare past the
    /// current logical length, then returns `(base, cap, len)` where `base` is
    /// the FULL backing pointer (offset 0), `cap` the allocation capacity, and
    /// `len` the current committed length (= the contig write head `*pos`). The
    /// decoder writes at `base.add(len..)` and resolves back-refs from
    /// `base[*pos - distance]` — the already-committed clean tail (the faithful
    /// vendor `setInitialWindow` prepend, where prior real output precedes new).
    ///
    /// The full base (not the tail) is returned BECAUSE `decode_clean_into_contig`
    /// addresses back-refs against `base[0..*pos)`, not a tail window.
    ///
    /// SAFETY contract: the caller writes only at indices `>= len` (uninitialized
    /// spare), within `[0, cap)`, then calls [`Self::commit`]. The returned
    /// `base` is INVALIDATED by any subsequent grow (`reserve`/`extend`/this
    /// method when it grows) — re-fetch every outer decode iteration (H4).
    pub fn contig_decode_window(&mut self, min_spare: usize) -> (*mut u8, usize, usize) {
        self.ensure_buf(self.buf.len() + min_spare);
        let len = self.buf.len();
        if self.buf.capacity() - len < min_spare {
            // `Vec::reserve(min_spare)` guarantees `capacity >= len + min_spare`
            // (it reserves min_spare MORE than the current length), so the spare
            // is at least `min_spare` after this call.
            self.buf.reserve(min_spare);
        }
        debug_assert!(
            self.buf.capacity() - len >= min_spare,
            "contig_decode_window: spare {} < min_spare {min_spare}",
            self.buf.capacity() - len
        );
        let cap = self.buf.capacity();
        // TEST-ONLY: poison the uninitialized contig spare `[len, cap)` so the
        // Stage-2 copy-free clean tail (which writes u8 DIRECTLY here and resolves
        // back-refs from the committed prefix) deterministically corrupts output
        // on any read-before-write seam/regrow bug, instead of flakily passing on
        // freshly-zeroed pages. No-op (inlines away) outside `cfg(test)` and
        // unless `GZIPPY_POISON_RESERVE` is set — byte/cost-transparent shipped.
        // SAFETY: `[len, cap)` is allocated-but-uninitialized backing memory; the
        // decoder overwrites it before any read (same contract as the return).
        let spare =
            unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(len), cap - len) };
        poison_reserved_tail(spare);
        (self.buf.as_mut_ptr(), cap, len)
    }

    /// Record `n` bytes written into the slice returned by
    /// [`Self::writable_tail`] (or the A3 window). Bumps the logical
    /// length. Panics in debug if `n` overflows spare capacity.
    pub fn commit(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let new_len = self.buf.len() + n;
        debug_assert!(
            new_len <= self.buf.capacity(),
            "commit {n} overflows spare (len {} cap {})",
            self.buf.len(),
            self.buf.capacity()
        );
        // SAFETY: bytes `[old_len, old_len+n)` were just written by the
        // caller through the `writable_tail`/A3 slice; `new_len <= cap`.
        unsafe {
            self.buf.set_len(new_len);
        }
    }

    /// Zero-copy view of the committed bytes `[start, start+len)`. Used by the
    /// copy-free ISA-L oracle to CRC the exact kept region without re-copying.
    #[inline]
    pub fn decoded_range(&self, start: usize, len: usize) -> &[u8] {
        &self.buf[start..start + len]
    }

    /// Iterate over the logical contents as byte slices (one contiguous
    /// slice; empty buffers yield nothing). Used by the consumer write
    /// path and CRC.
    #[inline]
    pub fn segments(&self) -> impl Iterator<Item = &[u8]> {
        std::iter::once(self.buf.as_slice()).filter(|s| !s.is_empty())
    }

    /// Number of populated backing buffers (0 or 1). Diagnostic surface.
    #[allow(dead_code)]
    pub fn segment_count(&self) -> usize {
        usize::from(!self.buf.is_empty())
    }

    /// Capacity of the contiguous backing store.
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Hand the backing allocation to the chunk-buffer pool recycler as a
    /// one-element `Vec<U8>` (the recycler returns each element to the
    /// owner worker's u8 pool). Leaves `self` empty.
    pub fn take_segments(&mut self) -> Vec<U8> {
        if self.buf.capacity() == 0 {
            return Vec::new();
        }
        vec![std::mem::replace(&mut self.buf, types::u8_with_capacity(0))]
    }

    /// Construct from owned backing buffer(s). The common case is the
    /// single buffer a prior `take_segments` produced; multiple buffers
    /// are concatenated into one contiguous allocation.
    pub fn from_segments(mut segments: Vec<U8>) -> Self {
        match segments.len() {
            0 => Self::default(),
            1 => Self {
                buf: segments.pop().unwrap(),
            },
            _ => {
                let total: usize = segments.iter().map(|s| s.len()).sum();
                let mut buf = types::u8_with_capacity(total.max(ALLOCATION_CHUNK_SIZE));
                for s in &segments {
                    buf.extend_from_slice(s);
                }
                Self { buf }
            }
        }
    }

    /// Reserve at least `additional` more bytes of contiguous capacity.
    pub fn reserve(&mut self, additional: usize) {
        if additional == 0 {
            return;
        }
        self.ensure_buf(self.buf.len() + additional);
        let need = self.buf.len() + additional;
        if self.buf.capacity() < need {
            self.buf.reserve(need - self.buf.len());
        }
    }

    /// Truncate the logical buffer to at most `new_len` bytes.
    pub fn truncate(&mut self, new_len: usize) {
        self.buf.truncate(new_len);
    }

    /// Copy the logical contents into a single contiguous `Vec<u8>`. O(n).
    #[allow(dead_code)]
    pub fn to_contiguous(&self) -> Vec<u8> {
        self.buf.as_slice().to_vec()
    }

    /// In-place index access. O(1). Returns `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<u8> {
        self.buf.get(index).copied()
    }

    /// Mutable index access. O(1).
    #[allow(dead_code)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut u8> {
        self.buf.get_mut(index)
    }

    /// Iterate over each byte in append order.
    pub fn iter_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.buf.iter().copied()
    }

    #[inline]
    fn tail_slice(&self, n: usize) -> &[u8] {
        debug_assert!(
            n <= self.buf.len(),
            "tail_slice: n {n} > len {}",
            self.buf.len()
        );
        &self.buf[self.buf.len() - n..]
    }

    /// Copy the last `n` logical bytes into `out` (exactly `n` long).
    /// Used to source a chunk's trailing 32 KiB sliding window.
    #[inline]
    pub fn copy_last_into(&self, out: &mut [u8]) {
        out.copy_from_slice(self.tail_slice(out.len()));
    }

    /// Specialized hot-path tail copy for the consumer publish chain.
    #[inline]
    pub fn copy_last_32k(&self, out: &mut [u8; 32768]) {
        out.copy_from_slice(self.tail_slice(32768));
    }

    /// Vec-producing twin of [`Self::copy_last_32k`]. One 32 KiB allocation,
    /// one memcpy from the contiguous tail.
    #[inline]
    pub fn copy_last_32k_vec(&self) -> Vec<u8> {
        self.tail_slice(32768).to_vec()
    }

    /// Prepend `bytes` as the new logical prefix. Mirror of vendor's
    /// `dataBuffers.emplace(dataBuffers.begin(), …)` in `cleanUnmarkedData`
    /// (`DecodedData.hpp:502`).
    pub fn prepend_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut nb =
            types::u8_with_capacity((bytes.len() + self.buf.len()).max(ALLOCATION_CHUNK_SIZE));
        nb.extend_from_slice(bytes);
        nb.extend_from_slice(&self.buf);
        self.buf = nb;
    }

    /// Prepend `n` in-place-narrowed marker bytes (u8 view over u16
    /// segments) without an intermediate `Vec`. Vendor `applyWindow` swap
    /// (`DecodedData.hpp:365-388`).
    pub fn prepend_narrowed_from_markers(&mut self, markers: &SegmentedU16, n: usize) {
        if n == 0 {
            return;
        }
        let mut nb = types::u8_with_capacity((n + self.buf.len()).max(ALLOCATION_CHUNK_SIZE));
        let mut left = n;
        for seg in markers.segments() {
            if left == 0 {
                break;
            }
            let take = left.min(seg.len());
            // SAFETY: `resolve_and_narrow_in_place` wrote u8 at byte offsets
            // `[0, seg.len())` in this segment's storage.
            let sl = unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, take) };
            nb.extend_from_slice(sl);
            left -= take;
        }
        debug_assert_eq!(
            left, 0,
            "prepend_narrowed_from_markers: short by {left} bytes"
        );
        nb.extend_from_slice(&self.buf);
        self.buf = nb;
    }

    /// Copy the logical byte range `[start, start + out.len())` into `out`.
    pub fn copy_range_into(&self, start: usize, out: &mut [u8]) {
        let n = out.len();
        debug_assert!(
            start + n <= self.buf.len(),
            "copy_range_into: [{start}, {}) > len {}",
            start + n,
            self.buf.len()
        );
        out.copy_from_slice(&self.buf[start..start + n]);
    }

    /// Truncate to the first `at` logical bytes; return the suffix as a
    /// new `SegmentedU8`. Used by `clean_unmarked_data`.
    pub fn split_off(&mut self, at: usize) -> SegmentedU8 {
        debug_assert!(at <= self.buf.len());
        let tail = self.buf.split_off(at);
        SegmentedU8 { buf: tail }
    }

    /// Insert `bytes` at logical offset `offset` (shifting the suffix
    /// right).
    pub fn insert_logical_at(&mut self, offset: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if offset == 0 {
            self.prepend_bytes(bytes);
            return;
        }
        if offset >= self.buf.len() {
            self.extend_from_slice(bytes);
            return;
        }
        let tail = self.buf.split_off(offset);
        self.buf.extend_from_slice(bytes);
        self.buf.extend_from_slice(&tail);
    }

    /// Append the entire logical contents of `other` onto `self`, moving
    /// `other`'s allocation in wholesale when `self` is empty (the common
    /// merge case). Leaves `other` empty.
    pub fn append_segmented(&mut self, other: &mut SegmentedU8) {
        if other.buf.is_empty() {
            return;
        }
        if self.buf.is_empty() {
            std::mem::swap(&mut self.buf, &mut other.buf);
            return;
        }
        self.buf.extend_from_slice(&other.buf);
        other.buf.clear();
    }
}

/// Sink for the incremental (growable) copy-free ISA-L decode. Grows the single
/// contiguous backing `Vec` on demand so the steady-state footprint tracks the
/// ACTUAL decoded size rather than an 8x-compressed-span over-reserve — the
/// faithful analogue of rapidgzip's fixed-`ALLOCATION_CHUNK_SIZE` segment append
/// (GzipChunk.hpp:309-379), done on one contiguous Vec.
#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
impl crate::backends::isal_decompress::IncrementalOutSink for SegmentedU8 {
    #[inline]
    fn commit_and_reserve(&mut self, just_written: usize, min_spare: usize) -> (*mut u8, usize) {
        self.commit(just_written);
        if min_spare == 0 {
            // Final flush: commit only; the returned region is unused.
            return (self.buf.as_mut_ptr(), 0);
        }
        let spare = self.writable_tail_reserve(min_spare);
        let len = spare.len();
        (spare.as_mut_ptr(), len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_window_prefix_fills_buffer() {
        let mut buf = SegmentedU8::default();
        let window = vec![0xABu8; 32 * 1024];
        buf.prefill_window_prefix(&window);
        assert_eq!(buf.len(), window.len());
        assert_eq!(buf.segment_count(), 1);
        assert_eq!(buf.segments().next().unwrap(), window.as_slice());
    }

    #[test]
    fn extend_within_capacity() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"hello");
        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        assert_eq!(buf.segment_count(), 1);
        let cat: Vec<u8> = buf.segments().flatten().copied().collect();
        assert_eq!(cat, b"hello");
    }

    #[test]
    fn extend_past_one_allocation_chunk_stays_contiguous() {
        let mut buf = SegmentedU8::default();
        let chunk_a = vec![0xABu8; ALLOCATION_CHUNK_SIZE - 10];
        let chunk_b = vec![0xCDu8; 50];
        buf.extend_from_slice(&chunk_a);
        buf.extend_from_slice(&chunk_b);
        assert_eq!(buf.len(), ALLOCATION_CHUNK_SIZE + 40);
        // Contiguous now — a single backing buffer regardless of size.
        assert_eq!(buf.segment_count(), 1);
        let cat: Vec<u8> = buf.segments().flatten().copied().collect();
        assert_eq!(cat[..chunk_a.len()], chunk_a[..]);
        assert_eq!(cat[chunk_a.len()..], chunk_b[..]);
    }

    #[test]
    fn extend_one_huge_slice_one_buffer() {
        let mut buf = SegmentedU8::default();
        let total = ALLOCATION_CHUNK_SIZE * 5 + 1234;
        let big: Vec<u8> = (0..total).map(|i| (i & 0xFF) as u8).collect();
        buf.extend_from_slice(&big);
        assert_eq!(buf.len(), total);
        assert_eq!(buf.segment_count(), 1);
        let cat: Vec<u8> = buf.segments().flatten().copied().collect();
        assert_eq!(cat, big);
    }

    #[test]
    fn clear_preserves_allocation() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(&vec![0u8; ALLOCATION_CHUNK_SIZE * 3]);
        assert_eq!(buf.segment_count(), 1);
        let cap_before = buf.capacity();
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), cap_before);
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
        // Mimics a resumable decoder writing into contiguous spare
        // capacity directly, across the 128 KiB growth granularity.
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
            let s = buf.writable_tail(); // 28 KiB contiguous spare left
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
            let s = buf.writable_tail(); // tail full -> grow, still contiguous
            assert_eq!(s.len(), ALLOCATION_CHUNK_SIZE);
            s[0] = 0x33;
        }
        buf.commit(1);
        assert_eq!(buf.segment_count(), 1);
        assert_eq!(buf.len(), ALLOCATION_CHUNK_SIZE + 1);

        let cat = buf.to_contiguous();
        assert_eq!(cat[0], 0x11);
        assert_eq!(cat[100 * 1024 - 1], 0x11);
        assert_eq!(cat[100 * 1024], 0x22);
        assert_eq!(cat[ALLOCATION_CHUNK_SIZE], 0x33);
    }

    #[test]
    fn split_off_and_prepend_and_range() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"abcdefgh");
        let tail = buf.split_off(3);
        assert_eq!(buf.len(), 3);
        assert_eq!(tail.len(), 5);
        let mut out = [0u8; 5];
        tail.copy_range_into(0, &mut out);
        assert_eq!(&out, b"defgh");

        buf.prepend_bytes(b"XY");
        assert_eq!(buf.len(), 5);
        let mut o2 = [0u8; 5];
        buf.copy_range_into(0, &mut o2);
        assert_eq!(&o2, b"XYabc");

        let mut last = [0u8; 2];
        buf.copy_last_into(&mut last);
        assert_eq!(&last, b"bc");
    }

    #[test]
    fn copy_last_32k_fast_path_matches_tail() {
        let mut buf = SegmentedU8::default();
        let total = 64 * 1024;
        let src: Vec<u8> = (0..total).map(|i| (i & 0xFF) as u8).collect();
        buf.extend_from_slice(&src);

        let mut out = [0u8; 32768];
        buf.copy_last_32k(&mut out);
        assert_eq!(&out[..], &src[src.len() - 32768..]);
        assert_eq!(buf.copy_last_32k_vec(), src[src.len() - 32768..].to_vec());
    }
}
