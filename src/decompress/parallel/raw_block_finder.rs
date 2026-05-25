#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]
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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::block_finder::BlockFinder;
use super::streamed_results::{StreamedGetReturnCode, StreamedResults};

// B-instrumentation: per-spawn cost breakdown for the BlockFinder
// coordinator. Each `with_scoped_boundary_search` spawns one thread,
// so 22 calls per silesia means 22 spawns. We want to know if the
// 7.5B CPU samples are in scan (push_boundary_candidates) or in the
// consumer (try-each-candidate loop) or in spawn/sync overhead.
pub static BOUNDARY_SEARCH_CALLS: AtomicU64 = AtomicU64::new(0);
pub static BOUNDARY_SEARCH_TOTAL_US: AtomicU64 = AtomicU64::new(0);
pub static BOUNDARY_SEARCH_SCAN_US: AtomicU64 = AtomicU64::new(0);
pub static CONSUMER_TIME_US: AtomicU64 = AtomicU64::new(0);

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
    let finder = BlockFinder::new(data);
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
    }
    results.finalize();
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

    /// Zero-copy async boundary search: spawns one scoped finder thread
    /// that borrows `data`, runs `consumer` on the caller thread, then
    /// joins the finder when the scope ends.
    pub fn with_scoped_boundary_search<R>(
        data: &[u8],
        start_bit: usize,
        max_end: usize,
        consumer: impl FnOnce(&Self) -> R,
    ) -> R {
        let t_total = std::time::Instant::now();
        let coord = Self::new();
        let results = Arc::clone(&coord.results);
        let cancel = Arc::clone(&coord.cancel);
        let scan_us = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let scan_us_inner = std::sync::Arc::clone(&scan_us);
        let r = std::thread::scope(|scope| {
            scope.spawn(move || {
                let t_scan = std::time::Instant::now();
                push_boundary_candidates(&results, data, start_bit, max_end, Some(&cancel));
                scan_us_inner.store(
                    t_scan.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            });
            let t_consumer = std::time::Instant::now();
            let result = consumer(&coord);
            CONSUMER_TIME_US.fetch_add(
                t_consumer.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            result
        });
        let total_us = t_total.elapsed().as_micros() as u64;
        let scan_us = scan_us.load(std::sync::atomic::Ordering::Relaxed);
        BOUNDARY_SEARCH_TOTAL_US.fetch_add(total_us, std::sync::atomic::Ordering::Relaxed);
        BOUNDARY_SEARCH_SCAN_US.fetch_add(scan_us, std::sync::atomic::Ordering::Relaxed);
        BOUNDARY_SEARCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        r
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
