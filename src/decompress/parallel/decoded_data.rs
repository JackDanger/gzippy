//! Literal port of `rapidgzip::deflate::DecodedDataView` and
//! `rapidgzip::deflate::DecodedData`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/DecodedDataView.hpp:15-46,
//!  vendor/rapidgzip/librapidarchive/src/rapidgzip/DecodedData.hpp:27-235).
//!
//! Per-chunk decoded data storage used by `GzipChunk` / `ParallelGzipReader`.
//! The struct holds two parallel pools:
//!
//! - `data_with_markers`: `Vec<u16>` chunks emitted while the cross-chunk
//!   window is still unknown — the upper byte may carry a "MapMarkers"
//!   reference into the previous chunk's last 32 KiB.
//! - `data`: `Vec<u8>` chunks emitted once the window is resolved (no
//!   markers possible).
//!
//! After [`DecodedData::apply_window`] runs, the markers in
//! `data_with_markers` are resolved against the supplied window and
//! re-interpreted as raw bytes via [`MapMarkers`] semantics
//! (`MARKER_BASE = 32768`, mirror of `MarkerReplacement.hpp::MapMarkers`).
//!
//! NOTE — gzippy's production pipeline uses raw `Vec<u8>` /  `Vec<u16>`
//! per chunk inside `parallel::single_member`; this struct is the
//! rapidgzip-shaped storage that will be wired into the new
//! `ParallelGzipReader` port. See `parallel/mod.rs` for the relationship.

#![allow(dead_code)]

/// Mirror of `rapidgzip::deflate::MAX_WINDOW_SIZE` (gzip/definitions.hpp).
/// Deflate window is exactly 32 KiB.
pub const MAX_WINDOW_SIZE: usize = 32 * 1024;

/// Mirror of `rapidgzip::deflate::MarkerVector` (DecodedData.hpp:23). A
/// chunk of decoded data that may still carry cross-chunk markers.
pub type MarkerVector = Vec<u16>;

/// Mirror of `rapidgzip::deflate::DecodedVector` (DecodedData.hpp:24). A
/// chunk of fully resolved decoded bytes.
pub type DecodedVector = Vec<u8>;

/// Mirror of `rapidgzip::deflate::DecodedDataView`
/// (DecodedDataView.hpp:15-46).
///
/// Borrowed view of the dual-pool layout: two slices of `u16` (still
/// carrying markers) plus two slices of `u8` (fully resolved). The
/// double-slice is a vestige of rapidgzip's ring-buffer architecture
/// where a single logical chunk may wrap; in gzippy we usually fill only
/// the first slot but keep both for API parity.
#[derive(Default, Clone, Copy)]
pub struct DecodedDataView<'a> {
    /// Mirror of `dataWithMarkers` (DecodedDataView.hpp:43).
    pub data_with_markers: [&'a [u16]; 2],
    /// Mirror of `data` (DecodedDataView.hpp:44).
    pub data: [&'a [u8]; 2],
}

impl<'a> DecodedDataView<'a> {
    /// Mirror of `DecodedDataView::size()` (DecodedDataView.hpp:19-22).
    #[inline]
    pub fn size(&self) -> usize {
        self.data_with_markers[0].len()
            + self.data_with_markers[1].len()
            + self.data[0].len()
            + self.data[1].len()
    }

    /// Mirror of `DecodedDataView::dataSize()` (DecodedDataView.hpp:25-28).
    #[inline]
    pub fn data_size(&self) -> usize {
        self.data[0].len() + self.data[1].len()
    }

    /// Mirror of `DecodedDataView::dataWithMarkersSize()`
    /// (DecodedDataView.hpp:31-34).
    #[inline]
    pub fn data_with_markers_size(&self) -> usize {
        self.data_with_markers[0].len() + self.data_with_markers[1].len()
    }

    /// Mirror of `DecodedDataView::containsMarkers()`
    /// (DecodedDataView.hpp:37-40).
    #[inline]
    pub fn contains_markers(&self) -> bool {
        !self.data_with_markers[0].is_empty() || !self.data_with_markers[1].is_empty()
    }
}

/// Mirror of `rapidgzip::deflate::DecodedData` (DecodedData.hpp:27-235).
///
/// Owning per-chunk storage. The internal invariant from the C++ comment
/// (DecodedData.hpp:224-227) is preserved:
/// `data_with_markers` precede `data` in logical-stream order, and a
/// chunk may not append markers after raw data (`append` will return
/// `Err` on that misuse — mirror of the C++ `throw std::invalid_argument`
/// at DecodedData.hpp:268-271).
#[derive(Default)]
pub struct DecodedData {
    /// Mirror of `dataWithMarkers` (DecodedData.hpp:231). Vector-of-vectors
    /// to avoid reallocation on growth.
    pub data_with_markers: Vec<MarkerVector>,
    /// Mirror of `reusedDataBuffers` (DecodedData.hpp:232). After
    /// [`apply_window`], the marker buffers are swapped here so their
    /// allocation can be reused.
    pub reused_data_buffers: Vec<MarkerVector>,
    /// Mirror of `dataBuffers` (DecodedData.hpp:233). The owning storage
    /// for entries in `data`.
    pub data_buffers: Vec<DecodedVector>,
}

impl DecodedData {
    /// Mirror of the `(DecodedVector&&)` overload of `append`
    /// (DecodedData.hpp:109-117). Non-empty buffers move into
    /// `data_buffers`; the equivalent of the C++ `data` views is
    /// reconstituted on demand from `data_buffers` (we don't need to keep
    /// duplicate ranges since we own the storage).
    pub fn append_decoded(&mut self, to_append: DecodedVector) {
        if to_append.is_empty() {
            return;
        }
        let mut v = to_append;
        v.shrink_to_fit();
        self.data_buffers.push(v);
    }

    /// Mirror of `DecodedData::append(DecodedDataView const&)`
    /// (DecodedData.hpp:238-290).
    ///
    /// Append the view's contents into our owning pools. Markers may only
    /// be appended while `data_buffers` is empty (rapidgzip enforces the
    /// same invariant at DecodedData.hpp:268-271).
    pub fn append_view(&mut self, view: &DecodedDataView<'_>) -> Result<(), AppendError> {
        if view.data_with_markers_size() > 0 {
            if !self.data_buffers.is_empty() {
                return Err(AppendError::MarkersAfterData);
            }
            for buf in &view.data_with_markers {
                if !buf.is_empty() {
                    append_to_equally_sized_chunks_u16(&mut self.data_with_markers, buf);
                }
            }
        }
        if view.data_size() > 0 {
            // Concatenate both data slices into a single owning vector
            // (DecodedData.hpp:282-289). gzippy's lower-level pipeline is
            // expected to consolidate small chunks before reaching us.
            let total = view.data_size();
            let mut copied = Vec::with_capacity(total);
            for buf in &view.data {
                copied.extend_from_slice(buf);
            }
            self.data_buffers.push(copied);
        }
        Ok(())
    }

    /// Mirror of `dataSize()` (DecodedData.hpp:122-127).
    pub fn data_size(&self) -> usize {
        self.data_buffers.iter().map(Vec::len).sum()
    }

    /// Mirror of `dataWithMarkersSize()` (DecodedData.hpp:129-134).
    pub fn data_with_markers_size(&self) -> usize {
        self.data_with_markers.iter().map(Vec::len).sum()
    }

    /// Mirror of `size()` (DecodedData.hpp:136-140).
    pub fn size(&self) -> usize {
        self.data_size() + self.data_with_markers_size()
    }

    /// Mirror of `sizeInBytes()` (DecodedData.hpp:142-146).
    pub fn size_in_bytes(&self) -> usize {
        self.data_size() * std::mem::size_of::<u8>()
            + self.data_with_markers_size() * std::mem::size_of::<u16>()
    }

    /// Mirror of `containsMarkers()` (DecodedData.hpp:154-158).
    pub fn contains_markers(&self) -> bool {
        !self.data_with_markers.is_empty()
    }

    /// Mirror of `countMarkerSymbols()` (DecodedData.hpp:293-302).
    ///
    /// Counts entries with `(symbol & 0xFF00) != 0` — i.e. those whose
    /// upper byte is non-zero and therefore reference the previous
    /// chunk's window. Symbols in `0..256` are immediate literals.
    pub fn count_marker_symbols(&self) -> usize {
        let mut total = 0usize;
        for chunk in &self.data_with_markers {
            total += chunk.iter().filter(|&&s| (s & 0xFF00) != 0).count();
        }
        total
    }

    /// Mirror of `DecodedData::applyWindow` (DecodedData.hpp:305-371).
    ///
    /// Resolve all marker symbols against `window`, then re-interpret
    /// `data_with_markers` storage as raw bytes (the C++ uses the same
    /// memory: 16-bit slot's low byte holds the resolved literal). The
    /// resulting `Vec<u8>` chunks are pushed into `data_buffers`.
    ///
    /// `MapMarkers` semantics: a symbol `s < 256` is the literal `s`;
    /// `s ≥ MARKER_BASE` (= 32768) is `window[s - MARKER_BASE]` where
    /// index 0 is the OLDEST window byte. See
    /// `vendor/rapidgzip/.../MarkerReplacement.hpp::MapMarkers`.
    pub fn apply_window(&mut self, window: &[u8]) -> Result<(), AppendError> {
        if !self.reused_data_buffers.is_empty() {
            return Err(AppendError::AlreadyAppliedMarkers);
        }
        if self.data_with_markers_size() == 0 {
            self.data_with_markers.clear();
            return Ok(());
        }

        for chunk in std::mem::take(&mut self.data_with_markers) {
            let mut resolved = Vec::with_capacity(chunk.len());
            for sym in &chunk {
                resolved.push(map_marker(*sym, window));
            }
            // Park the original u16 backing into reused_data_buffers so
            // a higher layer can recycle the allocation (mirror of
            // `std::swap(reusedDataBuffers, dataWithMarkers)` at
            // DecodedData.hpp:368).
            self.reused_data_buffers.push(chunk);
            self.data_buffers.push(resolved);
        }
        Ok(())
    }

    /// Mirror of `getLastWindow` (DecodedData.hpp:176-177 declaration).
    ///
    /// Returns the last [`MAX_WINDOW_SIZE`] = 32 KiB bytes of decoded
    /// output (after any [`apply_window`] has resolved markers). If the
    /// total decoded data is shorter than the window, fills the high
    /// bytes from `previous_window` (the dictionary used to seed this
    /// chunk).
    pub fn get_last_window(&self, previous_window: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(MAX_WINDOW_SIZE);
        // Pre-pend previous-window suffix only if needed at the end.
        let total: usize = self.data_buffers.iter().map(Vec::len).sum();
        if total >= MAX_WINDOW_SIZE {
            // Walk backwards through data_buffers, taking the last
            // MAX_WINDOW_SIZE bytes.
            let mut remaining = MAX_WINDOW_SIZE;
            let mut take_from = Vec::new();
            for buf in self.data_buffers.iter().rev() {
                let take = buf.len().min(remaining);
                let start = buf.len() - take;
                take_from.push(&buf[start..]);
                remaining -= take;
                if remaining == 0 {
                    break;
                }
            }
            for slice in take_from.iter().rev() {
                out.extend_from_slice(slice);
            }
        } else {
            // Need bytes from previous_window first, then all of our data.
            let from_prev = MAX_WINDOW_SIZE - total;
            if previous_window.len() >= from_prev {
                let start = previous_window.len() - from_prev;
                out.extend_from_slice(&previous_window[start..]);
            } else {
                out.extend_from_slice(previous_window);
            }
            for buf in &self.data_buffers {
                out.extend_from_slice(buf);
            }
        }
        out
    }

    /// Mirror of `shrinkToFit` (DecodedData.hpp:190-199).
    pub fn shrink_to_fit(&mut self) {
        for c in &mut self.data_buffers {
            c.shrink_to_fit();
        }
        for c in &mut self.data_with_markers {
            c.shrink_to_fit();
        }
    }
}

/// Error variants for [`DecodedData`] APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError {
    /// Tried to append marker-bearing data after raw data had already
    /// been appended (mirror of the `throw std::invalid_argument` at
    /// DecodedData.hpp:268-271).
    MarkersAfterData,
    /// [`DecodedData::apply_window`] was called twice without an
    /// intervening reset (mirror of the `throw std::logic_error` at
    /// DecodedData.hpp:366).
    AlreadyAppliedMarkers,
}

/// Mirror of the `MapMarkers` operator from
/// `vendor/rapidgzip/.../MarkerReplacement.hpp`. Symbols `0..256` are
/// raw literals; symbols `≥ MARKER_BASE = 32768` index into `window`,
/// with 0 = oldest byte.
#[inline]
fn map_marker(symbol: u16, window: &[u8]) -> u8 {
    if symbol < 256 {
        return symbol as u8;
    }
    let marker_base: u16 = 32768;
    if symbol >= marker_base {
        let idx = (symbol - marker_base) as usize;
        if idx < window.len() {
            return window[idx];
        }
    }
    // Defensive: unmapped symbol resolves to zero (matches C++ behavior
    // when the window is undersized: it reads garbage; we return zero).
    0
}

/// Mirror of the lambda `appendToEquallySizedChunks` at
/// DecodedData.hpp:243-265, specialized for `u16` (the marker pool).
fn append_to_equally_sized_chunks_u16(target_chunks: &mut Vec<MarkerVector>, buffer: &[u16]) {
    /// Mirror of `ALLOCATION_CHUNK_SIZE = 128_Ki` (DecodedData.hpp:241).
    const ALLOCATION_CHUNK_SIZE: usize = 128 * 1024;
    let element_count = ALLOCATION_CHUNK_SIZE / std::mem::size_of::<u16>();

    if target_chunks.is_empty() {
        target_chunks.push(MarkerVector::with_capacity(element_count));
    }

    let mut copied = 0;
    while copied < buffer.len() {
        let last_cap = target_chunks.last().unwrap().capacity();
        let last_len = target_chunks.last().unwrap().len();
        let n_free = last_cap - last_len;
        if n_free == 0 {
            target_chunks.push(MarkerVector::with_capacity(element_count));
            continue;
        }
        let n_to_copy = n_free.min(buffer.len() - copied);
        target_chunks
            .last_mut()
            .unwrap()
            .extend_from_slice(&buffer[copied..copied + n_to_copy]);
        copied += n_to_copy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_size_accumulates() {
        let a = [0u16; 4];
        let b = [0u16; 8];
        let c = [0u8; 16];
        let view = DecodedDataView {
            data_with_markers: [&a, &b],
            data: [&c, &[]],
        };
        assert_eq!(view.data_with_markers_size(), 12);
        assert_eq!(view.data_size(), 16);
        assert_eq!(view.size(), 28);
        assert!(view.contains_markers());
    }

    #[test]
    fn append_decoded_skips_empty() {
        let mut dd = DecodedData::default();
        dd.append_decoded(Vec::new());
        assert_eq!(dd.size(), 0);
        assert!(dd.data_buffers.is_empty());
        dd.append_decoded(vec![1u8, 2, 3]);
        assert_eq!(dd.size(), 3);
        assert_eq!(dd.data_buffers.len(), 1);
    }

    #[test]
    fn append_view_rejects_markers_after_data() {
        let mut dd = DecodedData::default();
        let raw = [0u8; 4];
        dd.append_view(&DecodedDataView {
            data_with_markers: [&[], &[]],
            data: [&raw, &[]],
        })
        .unwrap();
        let marker = [32768u16];
        let err = dd
            .append_view(&DecodedDataView {
                data_with_markers: [&marker, &[]],
                data: [&[], &[]],
            })
            .unwrap_err();
        assert_eq!(err, AppendError::MarkersAfterData);
    }

    #[test]
    fn apply_window_resolves_literals_and_markers() {
        let mut dd = DecodedData::default();
        // Three markers: literal 'A', then a window reference at offset 0,
        // then a window reference at offset 1.
        dd.data_with_markers.push(vec![b'A' as u16, 32768, 32769]);
        let window = [b'X', b'Y'];
        dd.apply_window(&window).unwrap();
        assert_eq!(dd.data_buffers.len(), 1);
        assert_eq!(dd.data_buffers[0], vec![b'A', b'X', b'Y']);
        assert_eq!(dd.data_with_markers_size(), 0);
        // The freed buffer is parked in reused_data_buffers for recycling
        // (DecodedData.hpp:368).
        assert_eq!(dd.reused_data_buffers.len(), 1);
    }

    #[test]
    fn count_marker_symbols_counts_only_high_byte() {
        let mut dd = DecodedData::default();
        dd.data_with_markers
            .push(vec![0u16, 1, 255, 256, 32768, 65535]);
        assert_eq!(dd.count_marker_symbols(), 3); // 256, 32768, 65535
    }

    #[test]
    fn get_last_window_uses_previous_when_short() {
        let mut dd = DecodedData::default();
        dd.append_decoded(vec![0xAA; 100]);
        let prev = vec![0x55u8; MAX_WINDOW_SIZE];
        let win = dd.get_last_window(&prev);
        assert_eq!(win.len(), MAX_WINDOW_SIZE);
        // Last 100 bytes are our data.
        assert!(win[MAX_WINDOW_SIZE - 100..].iter().all(|&b| b == 0xAA));
        // Preceding bytes from `prev`.
        assert!(win[..MAX_WINDOW_SIZE - 100].iter().all(|&b| b == 0x55));
    }
}
