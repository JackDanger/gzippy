//! The parallel scheduler's output memory must be bounded by THREAD COUNT, not by
//! input size.
//!
//! ⭐ OWNER REVIEW, 2026-08-23:
//!
//!   "The parallel scheduler preallocates an output Vec for every chunk and never
//!    releases one after writing it. Memory is O(input + compressed output), not
//!    bounded by worker count. For incompressible data, slot capacity alone is
//!    roughly 1.1x input. This contradicts the claimed 128 MiB in-flight cap.
//!    Use a bounded reorder window / reusable slot pool."
//!
//! Confirmed before the fix (peak RSS, this box):
//!
//!     movie.mp4     12.3 MiB in  ->   56.2 MiB  (4.55x)
//!     monorepo.tar  48.6 MiB in  ->   97.6 MiB  (2.01x)
//!     weights       86.7 MiB in  ->  226.1 MiB  (2.61x, -p8)
//!
//! This test does not measure RSS (too noisy to gate on). It asserts the STRUCTURAL
//! property that made RSS unbounded: the number of live output slots. A regression
//! here is someone reintroducing one-slot-per-block.

use gzippy::infra::scheduler::reorder_window_for;

/// The window is a function of THREADS, never of block count.
#[test]
fn reorder_window_does_not_grow_with_input() {
    // Same thread count, wildly different input sizes -> identical window.
    let small = reorder_window_for(4, 8);
    let huge = reorder_window_for(4, 1_000_000);
    assert_eq!(
        small, huge,
        "reorder window grew with block count ({small} -> {huge}): output memory is \
         O(input) again. The scheduler must reuse a fixed pool of slots, not \
         preallocate one per block."
    );

    // And it does scale with threads, which is the intended bound.
    assert!(
        reorder_window_for(16, 1_000_000) > reorder_window_for(2, 1_000_000),
        "window must scale with thread count — that is the whole bound"
    );
}

/// A tiny input must not allocate more slots than it has blocks.
#[test]
fn window_never_exceeds_block_count() {
    for blocks in 1..=8usize {
        for threads in [1usize, 4, 16] {
            let w = reorder_window_for(threads, blocks);
            assert!(
                w <= blocks,
                "window {w} > {blocks} blocks (threads={threads}): allocating slots \
                 for blocks that do not exist"
            );
        }
    }
}

/// The window must leave every worker able to make progress, or the scheduler
/// deadlocks: the worker holding the lowest unwritten block must never be gated.
#[test]
fn window_is_large_enough_to_keep_every_worker_fed() {
    for threads in [1usize, 2, 4, 8, 16, 64] {
        let w = reorder_window_for(threads, usize::MAX);
        assert!(
            w > threads,
            "window {w} <= {threads} threads: with every worker holding a slot the \
             writer can be starved of its next block — deadlock"
        );
    }
}
