//! Counting allocator for test builds.
//!
//! Wraps `std::alloc::System` and tallies every allocation. Used by
//! `alloc_budget_tests` to assert that hot paths don't acquire unexpected
//! heap allocations.
//!
//! Only active in `#[cfg(test)]` — production binary uses the default
//! system allocator with zero overhead.

#[cfg(test)]
pub use counter::*;

#[cfg(test)]
mod counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub struct CountingAllocator;

    static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
    static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

    impl CountingAllocator {
        /// Reset counters. Call immediately before the operation under test.
        pub fn reset() {
            ALLOC_COUNT.store(0, Ordering::SeqCst);
            ALLOC_BYTES.store(0, Ordering::SeqCst);
        }

        /// Total number of `alloc` calls since last `reset()`.
        pub fn count() -> u64 {
            ALLOC_COUNT.load(Ordering::SeqCst)
        }

        /// Total bytes allocated since last `reset()`.
        pub fn bytes() -> u64 {
            ALLOC_BYTES.load(Ordering::SeqCst)
        }
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            // SAFETY: delegating to System allocator
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: delegating to System allocator
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            // SAFETY: delegating to System allocator
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // Count realloc as an allocation event (growth)
            if new_size > layout.size() {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
                ALLOC_BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
            }
            // SAFETY: delegating to System allocator
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    // Activate only in test builds
    #[cfg(test)]
    #[global_allocator]
    static COUNTING_ALLOC: CountingAllocator = CountingAllocator;
}
