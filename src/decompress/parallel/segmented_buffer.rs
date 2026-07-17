#![cfg(parallel_sm)]
#![allow(dead_code)]
// Pre-existing parallel-module dead code, exposed by the default-feature flip; delete in a dedicated cleanup.

//! `ChunkData::data` storage: a contiguous decode bulk PLUS an O(1)-prepend
//! front-segment list — the faithful port of rapidgzip's
//! `DecodedData::data = std::vector<VectorView<uint8_t>>`
//! (`vendor/.../rapidgzip/DecodedData.hpp:234`).
//!
//! ## Why a view-list and not one contiguous `Vec`
//!
//! rapidgzip's `data` is a LIST of views. `cleanUnmarkedData`
//! (`DecodedData.hpp:492-516`) narrows the marker-free clean tail into a small
//! buffer and INSERTS A VIEW AT THE FRONT (`data.insert(data.begin(), …)`,
//! O(1), one narrowing copy, NO byte-copy of the existing payload).
//! `applyWindow` (`DecodedData.hpp:365-388`) resolves markers in place and
//! likewise PREPENDS views — never copying the bulk.
//!
//! An earlier gzippy revision modelled `data` as a single contiguous
//! `Vec<u8>` (faithful to vendor's *clean-append* storage `dataBuffers`, which
//! IS contiguous per append). That kept the big sequential decode write
//! DTLB-friendly, but it forced `cleanUnmarkedData`/`applyWindow` to
//! `prepend_bytes`/`insert_logical_at` by REALLOCATING + memmoving the whole
//! payload PER CHUNK (plus a temp-`Vec` narrow, so the clean tail was copied
//! TWICE).
//!
//! ## The hybrid that converges WITHOUT regressing decode locality
//!
//! The decode engine (clean `run_contig`, A3 single-shot, ISA-L copy-free)
//! requires ONE CONTIGUOUS buffer to write into and to resolve back-references
//! against (`output[..out_pos]`). So decode keeps writing into a single
//! contiguous bulk [`buf`]. Prepends — which only ever happen AFTER decode is
//! complete (`finalize_with_deflate::clean_unmarked_data`, and the
//! consumer-side `merge_resolved_markers_into_data` on the legacy/Folded
//! path) — push onto [`front`], a small ordered list (typically 0-1 entries)
//! whose first element is the logical start of the chunk. No bulk byte is ever
//! moved by a prepend; the clean tail is narrowed DIRECTLY into its destination
//! front segment (one copy, mirroring vendor's single `std::transform`).
//!
//! Logical content is `front[0] ‖ front[1] ‖ … ‖ buf`. The public API is
//! unchanged from the contiguous version (same method names/signatures) so the
//! decode sink, marker-resolution, window-construction and writev output paths
//! are transparent to the switch — only the prepend cost changes.
//!
//! INVARIANT: decode-sink methods (`extend_from_slice`, `writable_tail*`,
//! `contig_decode_window`, `commit`, `first_segment_a3_output`,
//! `prefill_window_prefix`, `reserve`, `decoded_range`) run BEFORE any prepend,
//! so they `debug_assert!(front.is_empty())` and operate on `buf` exactly as
//! the contiguous version did.
//!
//! The u16 marker buffer stays 128 KiB-segmented in
//! [`super::segmented_markers::SegmentedU16`] — that one MATCHES vendor's
//! `dataWithMarkers` and must NOT be changed.

use super::segmented_markers::SegmentedU16;
use crate::decompress::parallel::rpmalloc_alloc::types::{self, U8};

/// Vendor's `ALLOCATION_CHUNK_SIZE` (`ChunkData.hpp:65`). Reused here as
/// the GROWTH GRANULARITY for the contiguous bulk's amortized reserve
/// and as the A3 single-shot decode window size — NOT a hard segment
/// boundary. The decode bulk is one contiguous allocation.
pub const ALLOCATION_CHUNK_SIZE: usize = 128 * 1024;

/// memcpy-append `src` onto `dst`.
///
/// `allocator_api2::Vec::extend_from_slice` is `self.extend(other.iter().cloned())`,
/// and BOTH of allocator-api2 0.2.21's `Extend` impls (the generic `Extend<T>`
/// AND the `Extend<&T>` one whose doc-comment *claims* `copy_from_slice`) are in
/// fact SCALAR per-element `while next() { ptr::write; set_len(len+1) }` loops
/// (vec/mod.rs:2704 & :2788). rapidgzip's clean `DecodedData::append` is a `std::copy`/`insert`
/// memcpy (DecodedData.hpp:282-289). Mirror that with `copy_nonoverlapping`.
/// Byte-for-byte identical to the element loop.
#[inline]
pub(crate) fn buf_append_memcpy(dst: &mut U8, src: &[u8]) {
    if src.is_empty() {
        return;
    }
    dst.reserve(src.len());
    let len = dst.len();
    // SAFETY: `reserve` guarantees `capacity() >= len + src.len()`; `src` is an
    // external slice disjoint from `dst`'s allocation (non-overlapping copy); we
    // initialize exactly `src.len()` fresh bytes before bumping the length.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().add(len), src.len());
        dst.set_len(len + src.len());
    }
}

/// View-list-shaped clean-data buffer (faithful port of vendor
/// `std::vector<VectorView<uint8_t>>`). A contiguous decode bulk [`Self::buf`]
/// plus an O(1)-prepend ordered front-segment list [`Self::front`].
///
/// Cloned via the derived impl (deep clone of every owned segment) — used by
/// `ChunkData::Clone` in the cache-promote path. The byte count cloned is
/// identical to the contiguous version; only the physical layout differs.
#[derive(Debug, Clone)]
pub struct SegmentedU8 {
    /// O(1)-prepended front segments, in LOGICAL order: `front[0]` is the
    /// chunk's logical start. Populated only AFTER decode by
    /// `cleanUnmarkedData`/`applyWindow`-style prepends. Typically empty or
    /// a single entry. Vendor: the front of `data` (the prepended views).
    front: Vec<U8>,
    /// Single contiguous backing store for the decode bulk (clean
    /// `run_contig`/A3/ISA-L output, plus any window-image prefix). Logically
    /// FOLLOWS every front segment. `buf.capacity() == 0` means "not yet
    /// sourced from the pool".
    buf: U8,
}

impl Default for SegmentedU8 {
    fn default() -> Self {
        Self {
            front: Vec::new(),
            buf: types::u8_with_capacity(0),
        }
    }
}

impl SegmentedU8 {
    /// Sum of the front-segment lengths. O(front.len()) — front is tiny.
    #[inline]
    fn front_total(&self) -> usize {
        self.front.iter().map(|s| s.len()).sum()
    }

    /// All logical slices in order: front segments then the bulk. Includes
    /// the (possibly empty) bulk so index/copy walks always terminate on it.
    #[inline]
    fn ordered_slices(&self) -> impl Iterator<Item = &[u8]> {
        self.front
            .iter()
            .map(|s| s.as_slice())
            .chain(std::iter::once(self.buf.as_slice()))
    }

    /// Lazily source the backing allocation from the current worker's u8
    /// pool (warm, pre-faulted pages) on first use. No-op once `buf` owns an
    /// allocation; `Vec`'s own amortized `reserve` handles later growth.
    ///
    /// Under the T1 resident scope the first take is at the FULL pinned
    /// capacity (the buffer is about to be reserve-pinned there anyway by
    /// `compute_initial_reserve`'s resident arm): (a) one fewer realloc+copy
    /// per chunk, and (b) the buffer's very first allocation is HUGE, so the
    /// slab gate — and, for tiny decodes, `rpmalloc_alloc::SystemHugeScope` —
    /// governs its backend: a tiny thin-T1 decode never touches rpmalloc's
    /// small-allocation path (whose first call triggers rpmalloc process init).
    /// T>1 workers never have the scope set: their take stays byte-identical.
    #[inline]
    fn ensure_buf(&mut self, min_capacity: usize) {
        if self.buf.capacity() == 0 {
            use crate::decompress::parallel::chunk_buffer_pool;
            let floor = if chunk_buffer_pool::resident_output_pool_enabled() {
                // Pinned capacity PLUS one segment: `reserve_clean` asks for the
                // pinned reserve as ADDITIONAL bytes on top of the ≤32 KiB window
                // prefix already appended, so the first take must exceed the pin
                // by at least that margin or the very first reserve GROWS the
                // buffer — an out-of-place relocation that copies (and faults)
                // the full just-taken capacity. With the margin the
                // per-chunk reserve is a no-op and the pooled buffer is
                // steady-state for every subsequent chunk, exactly like the
                // baseline's grown-once buffer.
                chunk_buffer_pool::RESIDENT_PINNED_CAPACITY + ALLOCATION_CHUNK_SIZE
            } else {
                ALLOCATION_CHUNK_SIZE
            };
            self.buf = chunk_buffer_pool::take_u8(min_capacity.max(floor));
        }
    }

    /// Total logical byte count. O(front.len()).
    #[inline]
    pub fn len(&self) -> usize {
        self.front_total() + self.buf.len()
    }

    /// True iff no bytes have been appended or prepended.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() && self.front.iter().all(|s| s.is_empty())
    }

    /// True when the logical contents still fit in a single 128 KiB A3
    /// decode window. Decode-time gate (front is empty) so it preserves the
    /// segmented version's decode-path *selection* byte-for-byte.
    #[inline]
    pub fn all_in_first_segment(&self) -> bool {
        debug_assert!(self.front.is_empty(), "all_in_first_segment is decode-time");
        self.buf.len() <= ALLOCATION_CHUNK_SIZE
    }

    /// Segment-native Option A3: install the predecessor's 32 KiB sliding
    /// window at the FRONT of the contiguous bulk. Must run before any
    /// decoded bytes are appended (`len() == 0`).
    pub fn prefill_window_prefix(&mut self, window: &[u8]) {
        debug_assert!(self.front.is_empty(), "prefill before any prepend");
        debug_assert!(self.buf.is_empty(), "prefill before any decoded bytes");
        debug_assert!(
            window.len() <= ALLOCATION_CHUNK_SIZE,
            "window prefix exceeds one A3 window"
        );
        if window.is_empty() {
            return;
        }
        self.ensure_buf(ALLOCATION_CHUNK_SIZE);
        buf_append_memcpy(&mut self.buf, window);
    }

    /// Mutable contiguous A3 decode window — a `[0, capacity)` view into
    /// the single backing allocation. The decoder writes at
    /// `out_pos_start = self.len()`; back-references into the prefilled
    /// prefix resolve via `output[..out_pos]`. Only meaningful while
    /// [`Self::all_in_first_segment`] holds (caller's gate).
    ///
    /// SAFETY contract: caller writes only at indices `>= len()` before
    /// `commit`; identical to the contiguous version this replaces.
    pub fn first_segment_a3_output(&mut self) -> &mut [u8] {
        debug_assert!(self.front.is_empty(), "a3 output is decode-time");
        self.ensure_buf(ALLOCATION_CHUNK_SIZE);
        debug_assert!(self.buf.capacity() >= ALLOCATION_CHUNK_SIZE);
        let cap = self.buf.capacity();
        // SAFETY: `cap` bytes are allocated; `[0, len())` is initialized
        // (prefill and/or prior commits); the tail is writable spare.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr(), cap) }
    }

    /// Write every decoded payload byte, skipping the first `skip_prefix`
    /// logical bytes (the A3 window image at the front of the bulk).
    pub fn write_payload_skipping_prefix<W: std::io::Write>(
        &self,
        skip_prefix: usize,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let mut skip = skip_prefix;
        for seg in self.ordered_slices() {
            if seg.is_empty() {
                continue;
            }
            if skip >= seg.len() {
                skip -= seg.len();
                continue;
            }
            writer.write_all(&seg[skip..])?;
            skip = 0;
        }
        Ok(())
    }

    /// Collect payload slice refs for `writev` (skips `skip_prefix`
    /// logical bytes). Yields each non-empty logical segment in order.
    pub fn append_payload_iovecs<'a>(&'a self, skip_prefix: usize, out: &mut Vec<&'a [u8]>) {
        let mut skip = skip_prefix;
        for seg in self.ordered_slices() {
            if seg.is_empty() {
                continue;
            }
            if skip >= seg.len() {
                skip -= seg.len();
                continue;
            }
            out.push(&seg[skip..]);
            skip = 0;
        }
    }

    /// Truncate to zero length, retaining the bulk allocation for reuse and
    /// dropping front segments.
    pub fn clear(&mut self) {
        self.front.clear();
        self.buf.clear();
    }

    /// Append a clean (marker-free) byte slice to the bulk. Decode-time
    /// (front empty). Contiguous, amortized growth. Mirror of vendor's
    /// contiguous clean-data append (`DecodedData.hpp:282-289`).
    pub fn extend_from_slice(&mut self, bytes: &[u8]) {
        debug_assert!(self.front.is_empty(), "extend after prepend not supported");
        if bytes.is_empty() {
            return;
        }
        self.ensure_buf(bytes.len());
        buf_append_memcpy(&mut self.buf, bytes);
    }

    /// Zero-copy decode sink: return a writable window (up to
    /// [`ALLOCATION_CHUNK_SIZE`] bytes) of contiguous spare capacity at the
    /// tail of the bulk for a decoder to write DIRECTLY into. Decode-time
    /// (front empty). After the decoder writes N bytes, call [`Self::commit`].
    ///
    /// SAFETY contract: the caller must only WRITE into the returned slice
    /// (uninitialized spare) and call `commit(n)` with `n <= slice.len()`
    /// before the next mutating call.
    pub fn writable_tail(&mut self) -> &mut [u8] {
        debug_assert!(self.front.is_empty(), "writable_tail is decode-time");
        self.ensure_buf(ALLOCATION_CHUNK_SIZE);
        let len = self.buf.len();
        if self.buf.capacity() == len {
            // Only grow once the contiguous spare is fully exhausted, so
            // the decoder's per-call window cadence matches the segmented
            // version. Amortized; may realloc + move, but no live raw
            // pointer spans this call — callers re-fetch the tail each
            // outer decode iteration.
            self.buf.reserve(ALLOCATION_CHUNK_SIZE);
        }
        let window = ALLOCATION_CHUNK_SIZE.min(self.buf.capacity() - len);
        debug_assert!(window > 0, "writable_tail: no spare capacity");
        // SAFETY: `[len, len+window)` lies within the allocation; the
        // caller writes before any read, then calls `commit`.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(len), window) }
    }

    /// Like [`Self::writable_tail`] but guarantees AT LEAST `min_spare` bytes of
    /// CONTIGUOUS spare capacity and returns the WHOLE spare region (not capped
    /// at [`ALLOCATION_CHUNK_SIZE`]). Used by the copy-free ISA-L oracle.
    /// Decode-time (front empty).
    /// SAFETY contract identical to [`Self::writable_tail`]: write-then-`commit`.
    pub fn writable_tail_reserve(&mut self, min_spare: usize) -> &mut [u8] {
        debug_assert!(
            self.front.is_empty(),
            "writable_tail_reserve is decode-time"
        );
        self.ensure_buf(self.buf.len() + min_spare);
        let len = self.buf.len();
        if self.buf.capacity() - len < min_spare {
            // `Vec::reserve(additional)` guarantees `capacity >= len + additional`,
            // so to obtain `min_spare` bytes of spare PAST the current `len` we must
            // request `min_spare` directly — NOT `min_spare - (capacity - len)`.
            self.buf.reserve(min_spare);
        }
        let spare = self.buf.capacity() - len;
        debug_assert!(spare >= min_spare, "writable_tail_reserve: short spare");
        // SAFETY: `[len, len+spare)` lies within the allocation; caller writes
        // before any read, then calls `commit`.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(len), spare) }
    }

    /// Copy-free-to-final contig decode window (gzippy-native FOLD post-flip
    /// tail). Decode-time (front empty). Ensures at least `min_spare` bytes of
    /// CONTIGUOUS spare past the current logical length, then returns
    /// `(base, cap, len)` where `base` is the FULL backing pointer (offset 0),
    /// `cap` the allocation capacity, and `len` the current committed length.
    /// The decoder writes at `base.add(len..)` and resolves back-refs from
    /// `base[*pos - distance]` — the already-committed clean tail.
    ///
    /// SAFETY contract: the caller writes only at indices `>= len` (uninitialized
    /// spare), within `[0, cap)`, then calls [`Self::commit`]. The returned
    /// `base` is INVALIDATED by any subsequent grow — re-fetch every outer
    /// decode iteration (H4).
    pub fn contig_decode_window(&mut self, min_spare: usize) -> (*mut u8, usize, usize) {
        debug_assert!(self.front.is_empty(), "contig_decode_window is decode-time");
        self.ensure_buf(self.buf.len() + min_spare);
        let len = self.buf.len();
        if self.buf.capacity() - len < min_spare {
            self.buf.reserve(min_spare);
        }
        debug_assert!(
            self.buf.capacity() - len >= min_spare,
            "contig_decode_window: spare {} < min_spare {min_spare}",
            self.buf.capacity() - len
        );
        let cap = self.buf.capacity();
        (self.buf.as_mut_ptr(), cap, len)
    }

    /// Record `n` bytes written into the slice returned by
    /// [`Self::writable_tail`] (or the A3 window). Decode-time (front empty).
    /// Bumps the bulk's logical length. Panics in debug if `n` overflows spare.
    pub fn commit(&mut self, n: usize) {
        debug_assert!(self.front.is_empty(), "commit is decode-time");
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

    /// Zero-copy view of the committed bulk bytes `[start, start+len)`.
    /// Decode-time (front empty). Used by the copy-free ISA-L oracle to CRC
    /// the exact kept region without re-copying.
    #[inline]
    pub fn decoded_range(&self, start: usize, len: usize) -> &[u8] {
        debug_assert!(self.front.is_empty(), "decoded_range is decode-time");
        &self.buf[start..start + len]
    }

    /// Iterate over the logical contents as byte slices (front segments then
    /// the bulk; empty slices are skipped). Used by the consumer write path
    /// and CRC.
    #[inline]
    pub fn segments(&self) -> impl Iterator<Item = &[u8]> {
        self.ordered_slices().filter(|s| !s.is_empty())
    }

    /// Number of populated logical segments. Diagnostic surface.
    #[allow(dead_code)]
    pub fn segment_count(&self) -> usize {
        self.front.iter().filter(|s| !s.is_empty()).count() + usize::from(!self.buf.is_empty())
    }

    /// Total reserved capacity across the bulk and front segments.
    pub fn capacity(&self) -> usize {
        self.front.iter().map(|s| s.capacity()).sum::<usize>() + self.buf.capacity()
    }

    /// Hand every owned backing allocation (front segments then the bulk) to
    /// the chunk-buffer pool recycler. Leaves `self` empty.
    pub fn take_segments(&mut self) -> Vec<U8> {
        let mut out: Vec<U8> = Vec::with_capacity(self.front.len() + 1);
        out.append(&mut self.front);
        if self.buf.capacity() != 0 {
            out.push(std::mem::replace(&mut self.buf, types::u8_with_capacity(0)));
        }
        out
    }

    /// Construct from owned backing buffer(s) in logical order, concatenated
    /// into one contiguous bulk (front empty). The common case is the single
    /// buffer a prior `take_segments` produced.
    pub fn from_segments(mut segments: Vec<U8>) -> Self {
        match segments.len() {
            0 => Self::default(),
            1 => Self {
                front: Vec::new(),
                buf: segments.pop().unwrap(),
            },
            _ => {
                let total: usize = segments.iter().map(|s| s.len()).sum();
                let mut buf = types::u8_with_capacity(total.max(ALLOCATION_CHUNK_SIZE));
                for s in &segments {
                    buf_append_memcpy(&mut buf, s);
                }
                Self {
                    front: Vec::new(),
                    buf,
                }
            }
        }
    }

    /// Reserve at least `additional` more bytes of contiguous bulk capacity.
    /// Decode-time (front empty).
    pub fn reserve(&mut self, additional: usize) {
        debug_assert!(self.front.is_empty(), "reserve is decode-time");
        if additional == 0 {
            return;
        }
        self.ensure_buf(self.buf.len() + additional);
        let need = self.buf.len() + additional;
        if self.buf.capacity() < need {
            self.buf.reserve(need - self.buf.len());
        }
    }

    /// Flush helper: the decoded output bytes BEFORE the
    /// trailing `keep` bytes have already been written to the sink; drop them and
    /// shift the trailing `keep` bytes to the FRONT of the bulk so they remain
    /// available as the DEFLATE sliding-window history for subsequent
    /// back-references (max distance 32768). Decode-time only (front empty,
    /// single contiguous bulk). After this call `len() == keep`, `buf[0..keep)`
    /// are the most-recent `keep` decoded bytes, and the CAPACITY is unchanged
    /// (no realloc → the buffer stays resident).
    pub fn retain_tail(&mut self, keep: usize) {
        debug_assert!(self.front.is_empty(), "retain_tail is decode-time");
        let len = self.buf.len();
        debug_assert!(keep <= len, "retain_tail keep {keep} > len {len}");
        let start = len - keep;
        if start == 0 {
            return;
        }
        // memmove the trailing `keep` bytes to the front; capacity preserved.
        self.buf.copy_within(start..len, 0);
        self.buf.truncate(keep);
    }

    /// Truncate the logical buffer to at most `new_len` bytes.
    pub fn truncate(&mut self, new_len: usize) {
        let ft = self.front_total();
        if new_len >= ft {
            self.buf.truncate(new_len - ft);
            return;
        }
        // Truncation lands inside the front list (rare — front prepends happen
        // post-decode and truncate is a decode-time rollback, but handle it).
        self.buf.clear();
        let mut rem = new_len;
        let mut keep = 0usize;
        for (i, s) in self.front.iter_mut().enumerate() {
            if rem >= s.len() {
                rem -= s.len();
                keep = i + 1;
            } else {
                s.truncate(rem);
                keep = i + 1;
                break;
            }
        }
        self.front.truncate(keep);
    }

    /// Copy the logical contents into a single contiguous `Vec<u8>`. O(n).
    #[allow(dead_code)]
    pub fn to_contiguous(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.len());
        for s in self.ordered_slices() {
            v.extend_from_slice(s);
        }
        v
    }

    /// Logical index access. O(front.len()). Returns `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<u8> {
        let mut idx = index;
        for s in &self.front {
            if idx < s.len() {
                return Some(s[idx]);
            }
            idx -= s.len();
        }
        self.buf.get(idx).copied()
    }

    /// Mutable logical index access. O(front.len()).
    #[allow(dead_code)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut u8> {
        let mut idx = index;
        for s in &mut self.front {
            if idx < s.len() {
                return s.get_mut(idx);
            }
            idx -= s.len();
        }
        self.buf.get_mut(idx)
    }

    /// Iterate over each byte in logical order.
    pub fn iter_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.front
            .iter()
            .flat_map(|s| s.iter().copied())
            .chain(self.buf.iter().copied())
    }

    /// Copy the last `n` logical bytes into `out` (exactly `n` long).
    /// Used to source a chunk's trailing 32 KiB sliding window. May span the
    /// bulk and the front list.
    #[inline]
    pub fn copy_last_into(&self, out: &mut [u8]) {
        let n = out.len();
        let total = self.len();
        debug_assert!(n <= total, "copy_last_into: n {n} > len {total}");
        self.copy_range_into(total - n, out);
    }

    /// Specialized hot-path tail copy for the consumer publish chain.
    #[inline]
    pub fn copy_last_32k(&self, out: &mut [u8; 32768]) {
        self.copy_last_into(out);
    }

    /// Vec-producing twin of [`Self::copy_last_32k`].
    #[inline]
    pub fn copy_last_32k_vec(&self) -> Vec<u8> {
        let mut v = vec![0u8; 32768];
        self.copy_last_into(&mut v);
        v
    }

    /// Prepend `bytes` as a new front segment — O(1) view insert at the
    /// logical front (no bulk byte-copy). Mirror of vendor's
    /// `data.insert(data.begin(), …)` in `cleanUnmarkedData`
    /// (`DecodedData.hpp:503`).
    pub fn prepend_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut seg = types::u8_with_capacity(bytes.len());
        seg.extend_from_slice(bytes);
        self.front.insert(0, seg);
    }

    /// Prepend `n` in-place-narrowed marker bytes (u8 view over u16
    /// segments) as a new front segment. Vendor `applyWindow` swap
    /// (`DecodedData.hpp:365-388`). Legacy/Folded path — NOT production, which
    /// keeps the narrowed bytes in `data_with_markers` and emits them as
    /// zero-copy views via `append_narrowed_iovecs`.
    pub fn prepend_narrowed_from_markers(&mut self, markers: &SegmentedU16, n: usize) {
        if n == 0 {
            return;
        }
        let mut seg = types::u8_with_capacity(n);
        let mut left = n;
        for mseg in markers.segments() {
            if left == 0 {
                break;
            }
            let take = left.min(mseg.len());
            // SAFETY: `resolve_and_narrow_in_place` wrote u8 at byte offsets
            // `[0, mseg.len())` in this segment's storage.
            let sl = unsafe { std::slice::from_raw_parts(mseg.as_ptr() as *const u8, take) };
            seg.extend_from_slice(sl);
            left -= take;
        }
        debug_assert_eq!(left, 0, "prepend_narrowed_from_markers: short by {left}");
        self.front.insert(0, seg);
    }

    /// Prepend the marker-free clean tail `markers[split_at .. split_at+n]`
    /// (every value already `< 256` by construction) as a new front segment,
    /// narrowing u16→u8 DIRECTLY into the destination — ONE copy, no temp `Vec`,
    /// no bulk memmove. This is the faithful port of vendor `cleanUnmarkedData`'s
    /// single `std::transform(marker.base(), …, downcasted->begin(), to_u8)`
    /// into the front-inserted `dataBuffer` (`DecodedData.hpp:502-505`).
    ///
    /// `split_at` and `n` are in u16-ELEMENT units of `markers` (one element →
    /// one output byte). After this call, [`Self::first_front_bytes`] returns
    /// exactly the narrowed bytes (for CRC) without re-copying.
    pub fn prepend_narrowed_clean_tail(
        &mut self,
        markers: &SegmentedU16,
        split_at: usize,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let mut seg = types::u8_with_capacity(n);
        let mut skip = split_at;
        let mut left = n;
        for mseg in markers.segments() {
            if left == 0 {
                break;
            }
            if skip >= mseg.len() {
                skip -= mseg.len();
                continue;
            }
            let take = left.min(mseg.len() - skip);
            seg.extend(mseg[skip..skip + take].iter().map(|&v| v as u8));
            left -= take;
            skip = 0;
        }
        debug_assert_eq!(left, 0, "prepend_narrowed_clean_tail: short by {left}");
        self.front.insert(0, seg);
    }

    /// Bytes of the most-recently-prepended front segment (`front[0]`). Used
    /// to CRC the migrated clean tail straight out of its destination, with no
    /// re-copy. Returns `&[]` if there are no front segments.
    #[inline]
    pub fn first_front_bytes(&self) -> &[u8] {
        self.front.first().map(|s| s.as_slice()).unwrap_or(&[])
    }

    /// Copy the logical byte range `[start, start + out.len())` into `out`.
    pub fn copy_range_into(&self, start: usize, out: &mut [u8]) {
        let n = out.len();
        debug_assert!(
            start + n <= self.len(),
            "copy_range_into: [{start}, {}) > len {}",
            start + n,
            self.len()
        );
        if n == 0 {
            return;
        }
        let mut skip = start;
        let mut written = 0usize;
        for seg in self.ordered_slices() {
            if written >= n {
                break;
            }
            if skip >= seg.len() {
                skip -= seg.len();
                continue;
            }
            let take = (seg.len() - skip).min(n - written);
            out[written..written + take].copy_from_slice(&seg[skip..skip + take]);
            written += take;
            skip = 0;
        }
        debug_assert_eq!(written, n, "copy_range_into underran");
    }

    /// Truncate to the first `at` logical bytes; return the suffix as a new
    /// `SegmentedU8`. Decode-time helper (front empty).
    pub fn split_off(&mut self, at: usize) -> SegmentedU8 {
        debug_assert!(self.front.is_empty(), "split_off with front prepends");
        debug_assert!(at <= self.buf.len());
        let tail = self.buf.split_off(at);
        SegmentedU8 {
            front: Vec::new(),
            buf: tail,
        }
    }

    /// Insert `bytes` at logical offset `offset` (shifting the suffix right).
    /// `offset == 0` is an O(1) front prepend. `offset > 0` is the
    /// window-present (`data_prefix_len > 0`) path, which carries no front
    /// prepends, so it inserts into the contiguous bulk.
    pub fn insert_logical_at(&mut self, offset: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if offset == 0 {
            self.prepend_bytes(bytes);
            return;
        }
        debug_assert!(
            self.front.is_empty(),
            "insert_logical_at(offset>0) with front prepends"
        );
        if offset >= self.buf.len() {
            self.ensure_buf(bytes.len());
            self.buf.extend_from_slice(bytes);
            return;
        }
        let tail = self.buf.split_off(offset);
        buf_append_memcpy(&mut self.buf, bytes);
        buf_append_memcpy(&mut self.buf, &tail);
    }

    /// Append the entire logical contents of `other` onto `self`, moving
    /// `other`'s bulk allocation wholesale when `self` is empty. Leaves
    /// `other` empty.
    pub fn append_segmented(&mut self, other: &mut SegmentedU8) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            std::mem::swap(self, other);
            return;
        }
        // `other`'s logical content goes after self's. Flatten other into a
        // contiguous tail of self.buf (front of `other` would otherwise need to
        // interleave). Front of `self` is preserved as the logical prefix.
        // memcpy-append each slice (was scalar `extend_from_slice`;
        // `buf_append_memcpy` mirrors rapidgzip's `DecodedData::append` memcpy,
        // byte-for-byte identical to the element loop).
        for seg in other.ordered_slices() {
            if !seg.is_empty() {
                buf_append_memcpy(&mut self.buf, seg);
            }
        }
        other.clear();
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
        // Decode bulk stays one contiguous buffer regardless of size.
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

    // ---- front-list (view-list convergence) coverage ----

    #[test]
    fn prepend_view_no_bulk_copy_logical_order() {
        // bulk = "world", prepend "hello " at front (O(1) view insert).
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"world");
        buf.prepend_bytes(b"hello ");
        assert_eq!(buf.len(), 11);
        assert_eq!(buf.segment_count(), 2);
        assert_eq!(buf.to_contiguous(), b"hello world");
        // logical indexing across the boundary
        assert_eq!(buf.get(0), Some(b'h'));
        assert_eq!(buf.get(5), Some(b' '));
        assert_eq!(buf.get(6), Some(b'w'));
        assert_eq!(buf.get(10), Some(b'd'));
        assert_eq!(buf.get(11), None);
        // iter_bytes order
        let it: Vec<u8> = buf.iter_bytes().collect();
        assert_eq!(it, b"hello world");
    }

    #[test]
    fn two_prepends_logical_order_markers_then_clean_then_bulk() {
        // Mimic finalize (clean tail prepend) then applyWindow (markers prepend).
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"BULK"); // run_contig clean tail decoded post-flip
        buf.prepend_bytes(b"clean"); // clean_unmarked_data clean tail
        buf.prepend_bytes(b"MARK"); // applyWindow resolved markers (legacy path)
        assert_eq!(buf.to_contiguous(), b"MARKcleanBULK");
        assert_eq!(buf.segment_count(), 3);

        // copy_range_into spanning all three segments
        let mut mid = [0u8; 7];
        buf.copy_range_into(2, &mut mid); // "RKclea"... len 7 => "RKcleanB"? check
        assert_eq!(&mid, b"RKclean");

        // copy_last_into spanning bulk + a front seg
        let mut last = [0u8; 6];
        buf.copy_last_into(&mut last);
        assert_eq!(&last, b"anBULK");
    }

    #[test]
    fn prepend_narrowed_clean_tail_one_copy() {
        use crate::decompress::parallel::replace_markers::MARKER_BASE;
        // markers buffer: [m, m, 0x41('A'), 0x42('B'), 0x43('C')] where the
        // first two are real markers and the trailing 3 are the clean tail.
        let mut markers = SegmentedU16::default();
        markers.push_slice(&[MARKER_BASE, MARKER_BASE + 1, 0x41, 0x42, 0x43]);
        let split_at = 2; // first 2 stay markered
        let n = 3; // clean tail length
        let mut buf = SegmentedU8::default();
        buf.prepend_narrowed_clean_tail(&markers, split_at, n);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.first_front_bytes(), b"ABC");
        assert_eq!(buf.to_contiguous(), b"ABC");
    }

    #[test]
    fn prepend_narrowed_clean_tail_matches_temp_vec_path() {
        // Equivalence: direct-narrow front prepend == old narrow-into-temp + prepend_bytes.
        let mut markers = SegmentedU16::default();
        let vals: Vec<u16> = (0..500u16).map(|i| i % 256).collect();
        markers.push_slice(&vals);
        let split_at = 137;
        let n = vals.len() - split_at;

        let mut direct = SegmentedU8::default();
        direct.extend_from_slice(b"tail-bulk");
        direct.prepend_narrowed_clean_tail(&markers, split_at, n);

        let mut temp: Vec<u8> = Vec::with_capacity(n);
        temp.extend(vals[split_at..].iter().map(|&v| v as u8));
        let mut viaold = SegmentedU8::default();
        viaold.extend_from_slice(b"tail-bulk");
        viaold.prepend_bytes(&temp);

        assert_eq!(direct.to_contiguous(), viaold.to_contiguous());
        assert_eq!(direct.first_front_bytes(), &temp[..]);
    }

    #[test]
    fn append_payload_iovecs_skips_prefix_across_segments() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"BULK");
        buf.prepend_bytes(b"PREFIXdata");
        // skip 6 ("PREFIX") -> "dataBULK"
        let mut parts: Vec<&[u8]> = Vec::new();
        buf.append_payload_iovecs(6, &mut parts);
        let cat: Vec<u8> = parts.concat();
        assert_eq!(cat, b"dataBULK");
    }

    #[test]
    fn truncate_into_front_segment() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"BULK");
        buf.prepend_bytes(b"FRONT");
        assert_eq!(buf.len(), 9);
        buf.truncate(3); // inside front seg "FRONT" -> "FRO"
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.to_contiguous(), b"FRO");
        buf.truncate(0);
        assert!(buf.is_empty());
    }

    #[test]
    fn insert_logical_at_offset_zero_is_prepend() {
        let mut buf = SegmentedU8::default();
        buf.extend_from_slice(b"world");
        buf.insert_logical_at(0, b"hello ");
        assert_eq!(buf.to_contiguous(), b"hello world");
        assert_eq!(buf.segment_count(), 2);
    }
}
