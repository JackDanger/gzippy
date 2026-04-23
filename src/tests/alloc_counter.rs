//! Counting allocator for test builds.
//!
//! Wraps `std::alloc::System` and tallies allocations per-thread via
//! thread-local `Cell` storage. This isolates `reset()`/`count()` from
//! concurrent test threads that run in parallel.
//!
//! Only active in `#[cfg(test)]` — production binary uses the default
//! system allocator with zero overhead.

#[cfg(test)]
pub use counter::*;

#[cfg(test)]
mod counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    pub struct CountingAllocator;

    thread_local! {
        // enabled: true between reset() and count() on this thread
        static ENABLED: Cell<bool>  = const { Cell::new(false) };
        static COUNT:   Cell<u64>   = const { Cell::new(0) };
        static BYTES:   Cell<u64>   = const { Cell::new(0) };
    }

    impl CountingAllocator {
        /// Arm the counter for the current thread and zero it.
        /// Call immediately before the operation under test.
        pub fn reset() {
            COUNT.with(|c| c.set(0));
            BYTES.with(|c| c.set(0));
            ENABLED.with(|c| c.set(true));
        }

        /// Total `alloc` calls on this thread since `reset()`.
        pub fn count() -> u64 {
            COUNT.with(|c| c.get())
        }

        /// Total bytes allocated on this thread since `reset()`.
        pub fn bytes() -> u64 {
            BYTES.with(|c| c.get())
        }
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // Thread-local Cell::with never allocates for Cell<primitive>.
            ENABLED.with(|e| {
                if e.get() {
                    COUNT.with(|c| c.set(c.get() + 1));
                    BYTES.with(|b| b.set(b.get() + layout.size() as u64));
                }
            });
            // SAFETY: delegating to System allocator
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: delegating to System allocator
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ENABLED.with(|e| {
                if e.get() {
                    COUNT.with(|c| c.set(c.get() + 1));
                    BYTES.with(|b| b.set(b.get() + layout.size() as u64));
                }
            });
            // SAFETY: delegating to System allocator
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if new_size > layout.size() {
                ENABLED.with(|e| {
                    if e.get() {
                        COUNT.with(|c| c.set(c.get() + 1));
                        BYTES.with(|b| b.set(b.get() + (new_size - layout.size()) as u64));
                    }
                });
            }
            // SAFETY: delegating to System allocator
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[cfg(test)]
    #[global_allocator]
    static COUNTING_ALLOC: CountingAllocator = CountingAllocator;
}
