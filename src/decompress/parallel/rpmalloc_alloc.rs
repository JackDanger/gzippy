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
    use std::collections::HashMap;
    use std::ptr::NonNull;
    use std::sync::{Mutex, Once, OnceLock};

    use allocator_api2::alloc::{AllocError, Allocator, Layout};

    /// Stateless rpmalloc-backed allocator (vendor `RpmallocAllocator<T>`).
    ///
    /// When `GZIPPY_SLAB_ALLOC` is set, huge allocations route through
    /// [`SlabAlloc`] (retain resident blocks instead of munmapping) — the
    /// T4-16 page-fault fix; default OFF for controlled rollout.
    #[derive(Copy, Clone, Debug, Default)]
    pub struct RpmallocAlloc;

    // ── Slab retain-allocator (the T4-16 page-fault fix) ──────────────────
    //
    // rpmalloc munmaps allocations above its ~3.94 MiB "huge" threshold on
    // free, so the ~12 MiB `ChunkData` buffers re-fault (first-touch zeroing)
    // on every reuse — measured 12.4% of CPU in page faults at T8 vs
    // rapidgzip's 1.2% (the dominant T4-16 gap). rapidgzip avoids it via 128
    // KiB sub-buffers that stay under the threshold in rpmalloc's per-thread
    // free list. `SlabAlloc` gets the same residency for the MONOLITHIC
    // buffer: it keeps freed huge blocks resident in a capped free-list keyed
    // by TRUE block size and reuses them, instead of munmapping.
    //
    // CRITICAL correctness (advisor 2026-05-29): the live side-table keys
    // dealloc/grow/shrink on the BLOCK, not the caller's `layout.size()` —
    // `Allocator::grow` calls `deallocate` with the OLD (smaller) layout, so
    // keying a free-list on `layout.size()` files blocks under the wrong size
    // and later hands out an undersized block → heap corruption. (That was
    // the bug in the earlier reverted retain-list that broke multimember.)
    // `grow`/`shrink` are overridden so a resident block whose true size
    // already fits is reused in place (no copy/realloc churn).
    const SLAB_THRESHOLD: usize = 3 * 1024 * 1024;
    /// True block sizes are rounded up to this granularity so a few size
    /// classes cover the (near-constant) chunk-buffer sizes for high reuse.
    const SLAB_GRANULARITY: usize = 1024 * 1024;

    fn slab_enabled() -> bool {
        static EN: OnceLock<bool> = OnceLock::new();
        *EN.get_or_init(|| std::env::var_os("GZIPPY_SLAB_ALLOC").is_some())
    }

    /// Max resident free blocks retained globally before excess is released
    /// (munmapped). Sized to the in-flight working set (~depth) to avoid the
    /// T16 cache/TLB regression seen when over-retaining (MAX_POOLED 8→32).
    fn slab_cap() -> usize {
        static CAP: OnceLock<usize> = OnceLock::new();
        *CAP.get_or_init(|| {
            std::env::var("GZIPPY_SLAB_CAP")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(48)
        })
    }

    #[derive(Default)]
    struct SlabState {
        /// resident, currently-free blocks: (ptr, true_block_size, align)
        free: Vec<(usize, usize, usize)>,
        /// currently-allocated slab blocks: ptr -> (true_block_size, align)
        live: HashMap<usize, (usize, usize)>,
    }

    fn slab_state() -> &'static Mutex<SlabState> {
        static S: OnceLock<Mutex<SlabState>> = OnceLock::new();
        S.get_or_init(|| Mutex::new(SlabState::default()))
    }

    #[inline]
    fn raw_alloc(size: usize, align: usize) -> *mut u8 {
        if align <= 16 {
            unsafe { rpmalloc_sys::rpmalloc(size) as *mut u8 }
        } else {
            unsafe { rpmalloc_sys::rpaligned_alloc(align, size) as *mut u8 }
        }
    }

    #[inline]
    unsafe fn raw_free(ptr: *mut u8) {
        unsafe { rpmalloc_sys::rpfree(ptr.cast()) }
    }

    /// Slab allocator for huge buffers — keeps freed blocks resident.
    /// `RpmallocAlloc` delegates to this when `GZIPPY_SLAB_ALLOC` is set;
    /// the fuzz test exercises it directly (unconditional).
    #[derive(Copy, Clone, Debug, Default)]
    pub struct SlabAlloc;

    impl SlabAlloc {
        fn alloc_huge(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let bs = layout.size().div_ceil(SLAB_GRANULARITY) * SLAB_GRANULARITY;
            let align = layout.align().max(16);
            {
                let mut s = slab_state().lock().unwrap();
                if let Some(pos) = s.free.iter().position(|&(_, b, a)| b == bs && a >= align) {
                    let (p, b, a) = s.free.swap_remove(pos);
                    s.live.insert(p, (b, a));
                    return Ok(NonNull::slice_from_raw_parts(
                        unsafe { NonNull::new_unchecked(p as *mut u8) },
                        layout.size(),
                    ));
                }
            }
            // miss: fresh backing (retained on free)
            ensure_thread_initialized();
            let p = raw_alloc(bs, align);
            if p.is_null() {
                return Err(AllocError);
            }
            slab_state()
                .lock()
                .unwrap()
                .live
                .insert(p as usize, (bs, align));
            Ok(NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(p) },
                layout.size(),
            ))
        }

        /// Returns true if `ptr` was a live slab block (now retained/released).
        /// Keys on the side-table, NOT the caller layout (grow passes OLD).
        unsafe fn free_if_slab(&self, ptr: NonNull<u8>) -> bool {
            let key = ptr.as_ptr() as usize;
            let mut s = slab_state().lock().unwrap();
            match s.live.remove(&key) {
                Some((bs, a)) => {
                    if s.free.len() < slab_cap() {
                        s.free.push((key, bs, a));
                    } else {
                        drop(s);
                        ensure_thread_initialized();
                        unsafe { raw_free(ptr.as_ptr()) };
                    }
                    true
                }
                None => false,
            }
        }

        /// True block size of a live slab block, if `ptr` is one.
        fn live_block(&self, ptr: NonNull<u8>) -> Option<(usize, usize)> {
            slab_state()
                .lock()
                .unwrap()
                .live
                .get(&(ptr.as_ptr() as usize))
                .copied()
        }
    }

    unsafe impl Allocator for SlabAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            if layout.size() == 0 {
                return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
            }
            if layout.size() >= SLAB_THRESHOLD {
                return self.alloc_huge(layout);
            }
            ensure_thread_initialized();
            let ptr = raw_alloc(layout.size(), layout.align());
            if ptr.is_null() {
                return Err(AllocError);
            }
            Ok(NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(ptr) },
                layout.size(),
            ))
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            if layout.size() == 0 {
                return;
            }
            if unsafe { self.free_if_slab(ptr) } {
                return;
            }
            ensure_thread_initialized();
            unsafe { raw_free(ptr.as_ptr()) };
        }

        unsafe fn grow(
            &self,
            ptr: NonNull<u8>,
            old_layout: Layout,
            new_layout: Layout,
        ) -> Result<NonNull<[u8]>, AllocError> {
            // Resident slab block already big enough → reuse in place.
            if let Some((bs, a)) = self.live_block(ptr) {
                if bs >= new_layout.size() && a >= new_layout.align() {
                    return Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()));
                }
            }
            // Otherwise: allocate new + copy + free old (routes consistently).
            let new = self.allocate(new_layout)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new.as_ptr() as *mut u8,
                    old_layout.size(),
                );
                self.deallocate(ptr, old_layout);
            }
            Ok(new)
        }

        unsafe fn shrink(
            &self,
            ptr: NonNull<u8>,
            old_layout: Layout,
            new_layout: Layout,
        ) -> Result<NonNull<[u8]>, AllocError> {
            // Keep the larger resident slab block; just report the new len.
            if self.live_block(ptr).is_some() {
                return Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()));
            }
            let new = self.allocate(new_layout)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new.as_ptr() as *mut u8,
                    new_layout.size(),
                );
                self.deallocate(ptr, old_layout);
            }
            Ok(new)
        }
    }

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
            if slab_enabled() {
                return SlabAlloc.allocate(layout);
            }
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
            if slab_enabled() {
                return unsafe { SlabAlloc.deallocate(ptr, layout) };
            }
            if layout.size() == 0 {
                return;
            }
            ensure_thread_initialized();
            unsafe {
                rpmalloc_sys::rpfree(ptr.as_ptr().cast());
            }
        }

        unsafe fn grow(
            &self,
            ptr: NonNull<u8>,
            old_layout: Layout,
            new_layout: Layout,
        ) -> Result<NonNull<[u8]>, AllocError> {
            if slab_enabled() {
                return unsafe { SlabAlloc.grow(ptr, old_layout, new_layout) };
            }
            // default: allocate new + copy + free old
            let new = self.allocate(new_layout)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new.as_ptr() as *mut u8,
                    old_layout.size(),
                );
                self.deallocate(ptr, old_layout);
            }
            Ok(new)
        }

        unsafe fn shrink(
            &self,
            ptr: NonNull<u8>,
            old_layout: Layout,
            new_layout: Layout,
        ) -> Result<NonNull<[u8]>, AllocError> {
            if slab_enabled() {
                return unsafe { SlabAlloc.shrink(ptr, old_layout, new_layout) };
            }
            let new = self.allocate(new_layout)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new.as_ptr() as *mut u8,
                    new_layout.size(),
                );
                self.deallocate(ptr, old_layout);
            }
            Ok(new)
        }
    }

    #[cfg(test)]
    mod slab_fuzz {
        use super::*;
        use allocator_api2::alloc::Allocator;

        /// Isolation fuzz for `SlabAlloc`: randomized allocate/grow/shrink/
        /// deallocate of mixed small+huge sizes, with a UNIQUE per-allocation
        /// tag written to each block's head and verified on a random live
        /// block every iteration. If the allocator ever hands the same memory
        /// to two live allocations (the multimember-corruption class), one
        /// write clobbers the other's tag → assert fires. Catches aliasing
        /// without ASAN/miri.
        #[test]
        fn slab_alloc_fuzz_aliasing_and_sizes() {
            let a = SlabAlloc;
            let mut st = 0x9e3779b97f4a7c15u64;
            let mut rng = || {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                st >> 33
            };
            struct Live {
                ptr: NonNull<u8>,
                layout: Layout,
                size: usize,
                tag: u64,
            }
            let mut live: Vec<Live> = Vec::new();
            let mut next_tag: u64 = 1;

            let write_tag = |ptr: NonNull<u8>, tag: u64| unsafe {
                std::ptr::write_unaligned(ptr.as_ptr() as *mut u64, tag);
            };
            let read_tag =
                |ptr: NonNull<u8>| unsafe { std::ptr::read_unaligned(ptr.as_ptr() as *const u64) };

            for _ in 0..5000 {
                // verify a random live block's tag intact (aliasing detector)
                if !live.is_empty() {
                    let i = (rng() as usize) % live.len();
                    assert_eq!(
                        read_tag(live[i].ptr),
                        live[i].tag,
                        "live block tag clobbered → allocator aliased two live allocations"
                    );
                }
                match rng() % 4 {
                    0 => {
                        // allocate (mix huge >=3MiB and small)
                        let size = if rng() % 2 == 0 {
                            (rng() as usize % (10 * 1024 * 1024)) + 8
                        } else {
                            (rng() as usize % 8192) + 8
                        };
                        let layout = Layout::from_size_align(size, 16).unwrap();
                        if let Ok(p) = a.allocate(layout) {
                            let ptr = NonNull::new(p.as_ptr() as *mut u8).unwrap();
                            assert!(p.len() >= size, "returned slice shorter than requested");
                            let tag = next_tag;
                            next_tag += 1;
                            write_tag(ptr, tag);
                            live.push(Live {
                                ptr,
                                layout,
                                size,
                                tag,
                            });
                        }
                    }
                    1 if !live.is_empty() => {
                        let i = (rng() as usize) % live.len();
                        let old = &live[i];
                        let newsize = old.size + (rng() as usize % (3 * 1024 * 1024)) + 1;
                        let nl = Layout::from_size_align(newsize, 16).unwrap();
                        let r = unsafe { a.grow(old.ptr, old.layout, nl) };
                        if let Ok(p) = r {
                            let np = NonNull::new(p.as_ptr() as *mut u8).unwrap();
                            assert!(p.len() >= newsize);
                            // grow preserves old.size bytes incl. the tag
                            assert_eq!(read_tag(np), old.tag, "grow lost the tag (bad copy)");
                            let tag = old.tag;
                            write_tag(np, tag);
                            live[i] = Live {
                                ptr: np,
                                layout: nl,
                                size: newsize,
                                tag,
                            };
                        }
                    }
                    2 if !live.is_empty() => {
                        let i = (rng() as usize) % live.len();
                        let old = &live[i];
                        if old.size > 16 {
                            let newsize = (rng() as usize % old.size).max(8);
                            let nl = Layout::from_size_align(newsize, 16).unwrap();
                            let r = unsafe { a.shrink(old.ptr, old.layout, nl) };
                            if let Ok(p) = r {
                                let np = NonNull::new(p.as_ptr() as *mut u8).unwrap();
                                assert!(p.len() >= newsize);
                                let tag = old.tag;
                                write_tag(np, tag);
                                live[i] = Live {
                                    ptr: np,
                                    layout: nl,
                                    size: newsize,
                                    tag,
                                };
                            }
                        }
                    }
                    _ if !live.is_empty() => {
                        let i = (rng() as usize) % live.len();
                        let l = live.swap_remove(i);
                        unsafe { a.deallocate(l.ptr, l.layout) };
                    }
                    _ => {}
                }
            }
            for l in live {
                unsafe { a.deallocate(l.ptr, l.layout) };
            }
        }
    }
}

#[cfg(feature = "arena-allocator")]
pub use arena::RpmallocAlloc;

/// Allocator-visibility tool — the "don't guess" instrument for the span-cache
/// work. Prints rpmalloc's process-wide span-map stats. `mapped_total` (total
/// OS memory mapped since init) is the page-fault proxy: if 128 KiB segments
/// are warm-reused via the thread/global span cache it stays near the live
/// working set; if every segment re-maps it balloons toward bytes-touched.
/// `huge_alloc_peak` exposes the >2 MiB monolithic buffers routing through
/// `huge_alloc` (the cold path). Counters are nonzero only with the
/// `rpmalloc-stats` feature (rpmalloc `ENABLE_STATISTICS`); runtime-gated by
/// `GZIPPY_RPMALLOC_STATS` so normal runs pay nothing.
#[cfg(feature = "arena-allocator")]
pub fn dump_global_stats(tag: &str) {
    if std::env::var_os("GZIPPY_RPMALLOC_STATS").is_none() {
        return;
    }
    let mib = |b: usize| b as f64 / 1_048_576.0;
    // SAFETY: rpmalloc_global_statistics fills a zeroed POD struct; the FFI is
    // always bound (values populate only under ENABLE_STATISTICS).
    let s = unsafe {
        let mut s: rpmalloc_sys::rpmalloc_global_statistics_t = std::mem::zeroed();
        rpmalloc_sys::rpmalloc_global_statistics(&mut s);
        s
    };
    eprintln!(
        "[rpmalloc {tag}] mapped_peak={:.0}M mapped_total={:.0}M unmapped_total={:.0}M cached={:.1}M huge_alloc_peak={:.0}M",
        mib(s.mapped_peak),
        mib(s.mapped_total),
        mib(s.unmapped_total),
        mib(s.cached),
        mib(s.huge_alloc_peak),
    );
}

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
