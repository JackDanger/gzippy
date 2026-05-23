#![cfg(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
))]

//! Port of `rapidgzip::RpmallocAllocator` / `rpmalloc_ensuring_initialization`
//! (`vendor/.../core/FasterVector.hpp:46-113`).
//!
//! Provides `RpmallocAlloc` for `allocator_api2::vec::Vec` on the two SM hot-path
//! buffers (`ChunkData::data`, `data_with_markers`). Process-wide init mirrors
//! vendor's `RpmallocInit`; per-thread init mirrors `RpmallocThreadInit`.

#[cfg(feature = "arena-allocator")]
mod arena {
    use std::ptr::NonNull;
    use std::sync::Once;

    use allocator_api2::alloc::{AllocError, Allocator, Layout};

    /// Stateless rpmalloc-backed allocator (vendor `RpmallocAllocator<T>`).
    #[derive(Copy, Clone, Debug, Default)]
    pub struct RpmallocAlloc;

    static PROCESS_INIT: Once = Once::new();

    fn ensure_process_initialized() {
        PROCESS_INIT.call_once(|| unsafe {
            rpmalloc_sys::rpmalloc_initialize();
        });
    }

    struct RpmallocThreadInit;

    impl Drop for RpmallocThreadInit {
        fn drop(&mut self) {
            unsafe {
                rpmalloc_sys::rpmalloc_thread_finalize();
            }
        }
    }

    thread_local! {
        static THREAD_INIT: RpmallocThreadInit = {
            ensure_process_initialized();
            unsafe {
                rpmalloc_sys::rpmalloc_thread_initialize();
            }
            RpmallocThreadInit
        };
    }

    fn ensure_thread_initialized() {
        THREAD_INIT.with(|_| ());
    }

    unsafe impl Allocator for RpmallocAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            ensure_thread_initialized();
            if layout.size() == 0 {
                return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
            }

            let ptr = if layout.align() <= 16 {
                unsafe { rpmalloc_sys::rpmalloc(layout.size()) }
            } else {
                unsafe { rpmalloc_sys::rpaligned_alloc(layout.align(), layout.size()) }
            };

            if ptr.is_null() {
                return Err(AllocError);
            }

            Ok(NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(ptr.cast()) },
                layout.size(),
            ))
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            if layout.size() == 0 {
                return;
            }
            ensure_thread_initialized();
            unsafe {
                rpmalloc_sys::rpfree(ptr.as_ptr().cast());
            }
        }
    }
}

#[cfg(feature = "arena-allocator")]
pub use arena::RpmallocAlloc;

/// `ChunkData::data` / pool `Vec<u8>` type — rpmalloc when `arena-allocator` is on.
pub mod types {
    #[cfg(feature = "arena-allocator")]
    pub type U8 = allocator_api2::vec::Vec<u8, super::RpmallocAlloc>;
    #[cfg(not(feature = "arena-allocator"))]
    pub type U8 = std::vec::Vec<u8>;

    #[cfg(feature = "arena-allocator")]
    pub type U16 = allocator_api2::vec::Vec<u16, super::RpmallocAlloc>;
    #[cfg(not(feature = "arena-allocator"))]
    pub type U16 = std::vec::Vec<u16>;

    pub fn u8_with_capacity(cap: usize) -> U8 {
        #[cfg(feature = "arena-allocator")]
        {
            allocator_api2::vec::Vec::with_capacity_in(cap, super::RpmallocAlloc)
        }
        #[cfg(not(feature = "arena-allocator"))]
        {
            std::vec::Vec::with_capacity(cap)
        }
    }

    pub fn u16_with_capacity(cap: usize) -> U16 {
        #[cfg(feature = "arena-allocator")]
        {
            allocator_api2::vec::Vec::with_capacity_in(cap, super::RpmallocAlloc)
        }
        #[cfg(not(feature = "arena-allocator"))]
        {
            std::vec::Vec::with_capacity(cap)
        }
    }

    pub fn u8_empty() -> U8 {
        u8_with_capacity(0)
    }

    pub fn u16_empty() -> U16 {
        u16_with_capacity(0)
    }
}
