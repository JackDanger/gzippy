#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup
//! Async raw block-finder coordinator — slim port of vendor
//! `core/BlockFinder<RawFinder>` (BlockFinder.hpp:35-218).
//!
//! Vendor's async loop calls `m_rawBlockFinder->find()` repeatedly and
//! `m_blockOffsets.push(offset)` (`BlockFinder.hpp:186-192`). gzippy's
//! analogue runs a **single background thread** that scans 8 KiB-bit
//! windows sequentially (first-candidate-wins semantics — the consumer
//! stops as soon as one offset decodes) and streams monotonically
//! increasing offsets through [`StreamedResults`].
//!
//! Production uses [`Self::with_scoped_boundary_search`] so the finder
//! borrows `&[u8]` with zero copy (vendor passes a raw pointer; we use
//! `thread::scope` instead of `Arc::from` which would clone the slice).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::blockfinder_validation::{BlockBoundary, DeflateBlockValidator};
use super::streamed_results::{StreamedGetReturnCode, StreamedResults};

// B-instrumentation: per-spawn cost breakdown for the BlockFinder
// coordinator. Each `with_scoped_boundary_search` spawns one thread,
// so 22 calls per silesia means 22 spawns. We want to know if the
// 7.5B CPU samples are in scan (push_boundary_candidates) or in the
// consumer (try-each-candidate loop) or in spawn/sync overhead.

/// Window size for incremental boundary search — matches
/// `speculative_decode_find_boundary`'s scan window.
const CHUNK_SIZE_BITS: usize = 8 * 1024 * 8;

/// Push block-boundary candidates from `[start_bit, max_end)` into
/// `results`, then finalize. Shared by sync and scoped-async paths.
fn push_boundary_candidates(
    results: &StreamedResults<usize>,
    data: &[u8],
    start_bit: usize,
    max_end: usize,
    cancel: Option<&AtomicBool>,
) {
    let finder = DeflateBlockValidator::new(data);
    let mut chunk_begin = start_bit;
    while chunk_begin < max_end {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            break;
        }
        let chunk_end = (chunk_begin + CHUNK_SIZE_BITS).min(max_end);
        for block in finder.find_blocks(chunk_begin, chunk_end) {
            if block.valid && block.bit_offset >= chunk_begin && block.bit_offset < chunk_end {
                results.push(block.bit_offset);
            }
        }
        chunk_begin = chunk_end;
        // Test-only deterministic cancel seam: once the finder has produced
        // enough boundaries for the consumer to read, hold here until the
        // consumer's cancel lands, so `cancel stops the scan early` is
        // observable without racing the (very fast) branchless scan. Bounded
        // by wall-clock so a stray concurrent test's finder can never
        // deadlock — it resumes the full scan if no cancel arrives. Compiles
        // out entirely in non-test builds.
        #[cfg(test)]
        test_cancel_seam::maybe_hold_for_cancel(results, cancel);
    }
    results.finalize();
}

/// Test-only synchronization seam for [`scoped_cancel_stops_early_without_full_scan`].
/// Armed per-test via [`Guard`]; a no-op unless armed. See the call site in
/// [`push_boundary_candidates`].
#[cfg(test)]
mod test_cancel_seam {
    use super::{AtomicBool, Ordering, StreamedResults};
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    static ACTIVE: AtomicBool = AtomicBool::new(false);
    static PAUSE_AFTER: AtomicUsize = AtomicUsize::new(usize::MAX);

    /// Arm the seam for the lifetime of the returned guard: the finder will
    /// hold after emitting `pause_after` boundaries until its cancel flag is
    /// set. Disarms on drop (including on panic).
    pub(super) struct Guard;

    pub(super) fn arm(pause_after: usize) -> Guard {
        PAUSE_AFTER.store(pause_after, Ordering::Relaxed);
        ACTIVE.store(true, Ordering::Relaxed);
        Guard
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE.store(false, Ordering::Relaxed);
            PAUSE_AFTER.store(usize::MAX, Ordering::Relaxed);
        }
    }

    pub(super) fn maybe_hold_for_cancel(
        results: &StreamedResults<usize>,
        cancel: Option<&AtomicBool>,
    ) {
        if !ACTIVE.load(Ordering::Relaxed) || results.len() < PAUSE_AFTER.load(Ordering::Relaxed) {
            return;
        }
        // The target consumer sets its cancel within microseconds (it reads
        // already-buffered offsets); this deadline is only a no-deadlock
        // backstop for a concurrent test's finder that happens to observe the
        // armed flag and will never be cancelled.
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                break;
            }
            std::thread::yield_now();
        }
    }
}

/// Coordinator wrapping a [`StreamedResults`] queue fed by one finder thread.
pub struct RawBlockFinderCoordinator {
    results: Arc<StreamedResults<usize>>,
    cancel: Arc<AtomicBool>,
}

impl RawBlockFinderCoordinator {
    pub fn new() -> Self {
        Self {
            results: Arc::new(StreamedResults::new()),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn results(&self) -> &Arc<StreamedResults<usize>> {
        &self.results
    }

    /// Signal the scoped finder thread to stop scanning after the consumer
    /// finds a valid trial-decode (vendor stops waiting once decode succeeds).
    pub fn cancel_search(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Run a synchronous scan over `[start_bit, max_end)` (unit tests).
    #[cfg(test)]
    pub fn run_sync_boundary_search(&mut self, data: &[u8], start_bit: usize, max_end: usize) {
        push_boundary_candidates(&self.results, data, start_bit, max_end, None);
    }

    /// Synchronous chunk-by-chunk boundary search. Scans
    /// `[start_bit, max_end)` in `CHUNK_SIZE_BITS` windows on the
    /// caller's thread (no `std::thread::scope`, no `clone3`
    /// syscall, no fresh 8 MiB stack allocation), invokes
    /// `try_candidate(bit_offset)` for each valid boundary found,
    /// and returns the FIRST `Some(R)` the callback yields.
    ///
    /// Replaces `with_scoped_boundary_search` for callers that
    /// don't actually benefit from streaming overlap. Per the
    /// bench-sm verbose stats (post-absorb-isal-tail commit
    /// 9053895): `Slow-path decode: ok=61 fail=0 no_candidate=0`,
    /// `BlockFinder coordinator spawns: 33`, `total_ms=3010 /
    /// scan_ms=702 / avg_total_us=91236` — i.e., the actual
    /// BlockFinder scan work was 23% of the elapsed
    /// `with_scoped_boundary_search` time. The other 77% was
    /// thread spawn + page-fault for fresh stack + scheduling +
    /// join overhead. Sync runs the scan inline at full speed,
    /// trades the ~5 ms "streaming overlap" for the ~70 ms
    /// spawn-cleanup tax, and stops at the first successful
    /// candidate.
    pub fn with_sync_boundary_search<R>(
        data: &[u8],
        start_bit: usize,
        max_end: usize,
        mut try_candidate: impl FnMut(BlockBoundary) -> Option<R>,
    ) -> Option<R> {
        let finder = DeflateBlockValidator::new(data);
        let mut chunk_begin = start_bit;
        while chunk_begin < max_end {
            let chunk_end = (chunk_begin + CHUNK_SIZE_BITS).min(max_end);
            let mut next_dynamic = finder.find_next_dynamic_block(chunk_begin, chunk_end);
            let mut next_uncompressed = finder.find_next_uncompressed_block(chunk_begin, chunk_end);

            while next_dynamic.is_some() || next_uncompressed.is_some() {
                let use_dynamic = match (&next_dynamic, &next_uncompressed) {
                    (Some(dynamic), Some(uncompressed)) => {
                        (dynamic.bit_offset, dynamic.seek_bit)
                            <= (uncompressed.bit_offset, uncompressed.seek_bit)
                    }
                    (Some(_), None) => true,
                    (None, Some(_)) => false,
                    (None, None) => break,
                };

                let block = if use_dynamic {
                    let block = next_dynamic.take().unwrap();
                    // Advance from the seek position we just consumed. For
                    // stored blocks `seek_bit` is the byte-aligned header
                    // (`offset.second`), which skips re-yielding the same pair.
                    next_dynamic =
                        finder.find_next_dynamic_block(block.seek_bit.saturating_add(1), chunk_end);
                    block
                } else {
                    let block = next_uncompressed.take().unwrap();
                    next_uncompressed = finder
                        .find_next_uncompressed_block(block.seek_bit.saturating_add(1), chunk_end);
                    block
                };

                if block.valid && block.bit_offset >= chunk_begin && block.bit_offset < chunk_end {
                    if let Some(result) = try_candidate(block) {
                        return Some(result);
                    }
                }
            }
            chunk_begin = chunk_end;
        }
        None
    }

    /// Zero-copy async boundary search: spawns one scoped finder thread
    /// that borrows `data`, runs `consumer` on the caller thread, then
    /// joins the finder when the scope ends.
    #[allow(dead_code)] // retained for tests / parallel callers
    pub fn with_scoped_boundary_search<R>(
        data: &[u8],
        start_bit: usize,
        max_end: usize,
        consumer: impl FnOnce(&Self) -> R,
    ) -> R {
        let coord = Self::new();
        let results = Arc::clone(&coord.results);
        let cancel = Arc::clone(&coord.cancel);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                push_boundary_candidates(&results, data, start_bit, max_end, Some(&cancel));
            });
            consumer(&coord)
        })
    }

    pub fn get_offset(
        &self,
        index: usize,
        timeout: Duration,
    ) -> (Option<usize>, StreamedGetReturnCode) {
        self.results.get(index, timeout)
    }
}

impl Default for RawBlockFinderCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{write::DeflateEncoder, Compression};
    use std::io::Write;

    fn deflate_payload(data: &[u8]) -> Vec<u8> {
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    fn make_deflate_payload(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let mut rng: u64 = 0xdeadbeef;
        while data.len() < size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((rng >> 16) as u8);
        }
        deflate_payload(&data)
    }

    #[test]
    fn sync_boundary_search_finds_at_least_one_boundary() {
        let deflate = make_deflate_payload(256 * 1024);
        let end = deflate.len() * 8;
        let mut coord = RawBlockFinderCoordinator::new();
        coord.run_sync_boundary_search(&deflate, 0, end);
        assert!(!coord.results().is_empty(), "expected ≥1 block candidate");
        let (off, code) = coord.get_offset(0, Duration::ZERO);
        assert_eq!(code, StreamedGetReturnCode::Success);
        assert!(off.is_some());
    }

    #[test]
    fn scoped_boundary_search_matches_sync() {
        let deflate = make_deflate_payload(256 * 1024);
        let end = deflate.len() * 8;

        let mut sync = RawBlockFinderCoordinator::new();
        sync.run_sync_boundary_search(&deflate, 0, end);
        let sync_count = sync.results().len();

        let scoped_count =
            RawBlockFinderCoordinator::with_scoped_boundary_search(&deflate, 0, end, |coord| {
                while !coord.results().finalized() {
                    std::thread::yield_now();
                }
                coord.results().len()
            });
        assert_eq!(scoped_count, sync_count);
    }

    #[test]
    fn scoped_cancel_stops_early_without_full_scan() {
        let deflate = make_deflate_payload(512 * 1024);
        let end = deflate.len() * 8;
        let mut sync_full = RawBlockFinderCoordinator::new();
        sync_full.run_sync_boundary_search(&deflate, 0, end);
        let full_count = sync_full.results().len();

        // Deterministically hold the finder after it has produced the 3
        // boundaries the consumer reads, so the cancel below is guaranteed to
        // land before the (sub-millisecond) branchless scan completes.
        let _seam = test_cancel_seam::arm(3);
        let cancelled_count =
            RawBlockFinderCoordinator::with_scoped_boundary_search(&deflate, 0, end, |coord| {
                let mut count = 0usize;
                while count < full_count {
                    let (off, code) = coord.get_offset(count, Duration::from_millis(100));
                    if code != StreamedGetReturnCode::Success {
                        break;
                    }
                    if off.is_some() {
                        count += 1;
                    }
                    if count >= 3 {
                        coord.cancel_search();
                        break;
                    }
                }
                while !coord.results().finalized() {
                    std::thread::yield_now();
                }
                coord.results().len()
            });
        assert!(
            cancelled_count < full_count,
            "cancel should stop finder before full scan: cancelled={cancelled_count} full={full_count}"
        );
        assert!(
            cancelled_count >= 3,
            "consumer should have read at least 3 offsets"
        );
    }

    #[test]
    fn scoped_offsets_are_monotonic() {
        let deflate = make_deflate_payload(512 * 1024);
        let end = deflate.len() * 8;
        RawBlockFinderCoordinator::with_scoped_boundary_search(&deflate, 0, end, |coord| {
            while !coord.results().finalized() {
                std::thread::yield_now();
            }
            let mut prev = 0usize;
            for i in 0..coord.results().len() {
                let (off, code) = coord.get_offset(i, Duration::ZERO);
                assert_eq!(code, StreamedGetReturnCode::Success);
                let bit = off.expect("missing offset");
                assert!(bit >= prev, "offsets must be monotonic: {bit} < {prev}");
                prev = bit;
            }
        });
    }
}
