#![cfg(parallel_sm)]
#![allow(dead_code)]
// task #8: pre-existing parallel-module dead code, exposed by default-feature flip; delete in a dedicated cleanup

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
    // MEASUREMENT-ONLY override (GZIPPY_SLAB_THRESHOLD_KIB): lower the slab
    // threshold so the 128 KiB u16 marker segments (the dominant page-fault
    // site, SegmentedU16::push_slice = 44% of faults) ALSO get the resident-
    // retain treatment instead of rpmalloc cross-thread-free + re-fault. Used
    // by the clean page-warmth oracle (advisor pass 2): drive marker faults to
    // ~rg's floor with a never-munmap resident slab and watch the matched wall.
    fn slab_threshold() -> usize {
        static T: OnceLock<usize> = OnceLock::new();
        *T.get_or_init(|| {
            std::env::var("GZIPPY_SLAB_THRESHOLD_KIB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .map(|kib| kib * 1024)
                .unwrap_or(3 * 1024 * 1024)
        })
    }
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
            // Sub-MiB blocks (the 128 KiB marker/data segments under a lowered
            // threshold) round to 128 KiB so they don't waste 8x via the 1 MiB
            // granularity (which would itself inflate faults).
            let gran = if layout.size() < SLAB_GRANULARITY {
                128 * 1024
            } else {
                SLAB_GRANULARITY
            };
            let bs = layout.size().div_ceil(gran) * gran;
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
            if layout.size() >= slab_threshold() {
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

    // ── Per-worker huge-block retention (GZIPPY_PW_RETAIN, default ON) ────
    //
    // DESIGN (2026-06-10) — written before implementation, per task contract.
    //
    // MECHANISM (box-proven this campaign): every per-chunk decode output is
    // ONE contiguous multi-MB Vec<u8, RpmallocAlloc>. rpmalloc munmaps
    // huge-class (>3.94 MiB) frees UNCONDITIONALLY (rpmalloc.c:2470-2491 —
    // no cache feature reaches them; measured TIE), so each free → re-alloc
    // cycle re-faults every page: ~1.25 M pgfaults/run, 31% of T8 per-chunk
    // decode, the dominant T4-T8 band gap vs rapidgzip (cells 0.79-0.97).
    // Existence proof: the process-global slab (GZIPPY_SLAB_ALLOC=1, above)
    // wins +13% wall / -72% pgfaults on the model corpus at T8.
    //
    // FALSIFIER FENCE (violate none — each bought with a measurement):
    //   * NO process-global byte budget: falsified at EVERY size (B8 = no-op
    //     because one model chunk is 12-25 MiB > budget; B16/B32 = silesia
    //     RSS +24%/+15% vs the <= +10% criterion).
    //   * NO mutex-guarded pools: GZIPPY_MANUAL_BUFFER_POOL measured WORSE
    //     (model +24%).
    //   * Naive thread_local slot is WRONG: buffers are ALLOCATED on worker
    //     threads (ChunkData::new in decode tasks) but FREED on the consumer
    //     thread (post-writev recycle) — a thread_local would accumulate on
    //     the consumer and never hit on workers.
    //
    // SHAPE: retention bounded at ONE huge block per worker slot, sized by
    // what that worker last used → retained ≈ live working set → RSS-neutral
    // by construction.
    //
    //   * Slots: fixed array of PW_SLOT_COUNT AtomicPtr<u8> single-block
    //     slots — one per ThreadPool worker index (bind_worker_pool_index,
    //     reusing chunk_buffer_pool's index plumbing) + one shared slot for
    //     unbound threads (consumer-side trial decodes). Cap is structural:
    //     a slot is one pointer, so retention can never exceed
    //     PW_SLOT_COUNT blocks ⇒ provably non-accumulating.
    //   * Owner identity TRAVELS WITH THE BLOCK: each huge allocation is
    //     over-allocated by an alignment-preserving header gap and a
    //     PwHeader{magic, usable, hdr_offset, owner} is written at
    //     user_ptr - 64. The consumer-thread free reads the header and
    //     installs the block into the OWNER's slot — no side-table, no lock.
    //   * Hot path cost: alloc = one swap(null) on own slot (+ header read
    //     on hit); free = one swap(install) on owner slot. ≤ 2 atomic ops,
    //     uncontended in steady state (each worker touches only its slot;
    //     the consumer touches it only at install time).
    //   * Policy: free KEEPS THE NEWEST block (swap-install; the displaced
    //     older resident is released) and alloc RELEASES A MISFIT (block too
    //     small / under-aligned for the request) instead of reinstalling —
    //     both make the slot track the worker's CURRENT working-set size.
    //   * TRIM-ON-INSTALL (v2; measured): reuse-if-fits lets small requests
    //     ride big blocks, so a slot's RESIDENT size drifted to the LARGEST
    //     block the worker ever saw (silesia T8 RSS +52% — failed the <=+10%
    //     gate). On install, MADV_DONTNEED the pages beyond the freeing
    //     layout's size: retained residency == what the worker LAST USED
    //     (the pinned sizing clause), while `usable` keeps the true capacity
    //     so bigger requests still fit and only re-fault their tail.
    //   * Steady state on the doubling growth pattern (128 KiB → … → 16 MiB
    //     via Vec::reserve): the first huge request (~4 MiB) reuses the
    //     retained full-size block from the previous chunk, `grow` then
    //     resolves IN PLACE against the block's true `usable` size (header,
    //     not caller layout — the slab's grow-keying lesson), so a warm
    //     worker performs ZERO huge mmap/munmap per chunk.
    //
    // BYTE-TRANSPARENCY: a reused block satisfies the requested layout
    // (usable >= size, hdr_offset >= align ⇒ user pointer alignment holds);
    // contents are NEVER reinterpreted; reuse hands back DIRTY pages —
    // identical to the slab's contract and to plain rpmalloc (`allocate` has
    // no zeroing guarantee; `allocate_zeroed`'s default impl memsets after
    // allocate). The decoder writes before reading (poison_reserved_tail
    // test net covers read-before-write).
    //
    // DISPATCH SAFETY: routing is by layout.size() >= PW_RETAIN_THRESHOLD,
    // so every transition that changes which side of the threshold a LIVE
    // block's current layout is on must also move the block:
    //   * grow small→huge / shrink huge→small: allocate-new + copy + free
    //     old (each side routed by its own size) — never in place.
    //   * grow/shrink huge→huge: in place only when the header's true
    //     `usable`/`hdr_offset` fit the NEW layout; header stays valid.
    // The always-on magic check turns any routing bug into a loud assert
    // instead of silent heap corruption.
    //
    // KNOBS: GZIPPY_PW_RETAIN=0 disables (default ON). GZIPPY_SLAB_ALLOC is
    // ORTHOGONAL and untouched: the slab branch is checked first in every
    // RpmallocAlloc method, exactly as before, so the legacy knob's behavior
    // is bit-identical and pw-retention never sees slab traffic.

    /// Huge-class threshold: at or above this size rpmalloc munmaps on free
    /// (its large-class ceiling is ~3.94 MiB). Matches `slab_threshold()`'s
    /// default so both retainers agree on what "huge" means.
    pub const PW_RETAIN_THRESHOLD: usize = 3 * 1024 * 1024;

    /// Fixed distance from the USER pointer back to the `PwHeader`. The
    /// header is always at `user - PW_HEADER_GAP` regardless of alignment
    /// padding, because `hdr_offset >= PW_HEADER_GAP` always.
    const PW_HEADER_GAP: usize = 64;

    /// Mirrors `chunk_buffer_pool::MAX_WORKERS`.
    const PW_WORKER_SLOTS: usize = 64;
    /// +1 shared slot for unbound (non-worker) threads.
    const PW_SLOT_COUNT: usize = PW_WORKER_SLOTS + 1;

    const PW_MAGIC: u64 = 0x675a_5057_5245_5431; // "gZPWRET1"

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct PwHeader {
        magic: u64,
        /// Usable bytes at the user pointer — the size requested at FRESH
        /// allocation, i.e. the block's true reuse capacity. NEVER rewritten
        /// on reuse/grow/shrink (the slab lesson: key on the block's truth,
        /// not the caller's layout).
        usable: usize,
        /// `user_ptr - hdr_offset` is the start of the rpmalloc allocation;
        /// also the effective alignment the block was allocated with.
        hdr_offset: usize,
        /// Owner slot index (the allocating worker); frees install here.
        owner: u32,
        _pad: u32,
    }

    const _: () = assert!(std::mem::size_of::<PwHeader>() <= PW_HEADER_GAP);

    use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

    static PW_SLOTS: [AtomicPtr<u8>; PW_SLOT_COUNT] =
        [const { AtomicPtr::new(std::ptr::null_mut()) }; PW_SLOT_COUNT];

    pub static PW_HITS: AtomicU64 = AtomicU64::new(0);
    pub static PW_MISSES: AtomicU64 = AtomicU64::new(0);
    pub static PW_MISFIT_RELEASED: AtomicU64 = AtomicU64::new(0);
    pub static PW_INSTALLS: AtomicU64 = AtomicU64::new(0);
    pub static PW_DISPLACED: AtomicU64 = AtomicU64::new(0);
    pub static PW_GROW_IN_PLACE: AtomicU64 = AtomicU64::new(0);
    pub static PW_TRIMMED_BYTES: AtomicU64 = AtomicU64::new(0);

    fn pw_retain_enabled() -> bool {
        static EN: OnceLock<bool> = OnceLock::new();
        *EN.get_or_init(|| {
            std::env::var("GZIPPY_PW_RETAIN")
                .map(|v| v != "0")
                .unwrap_or(true)
        })
    }

    /// Measurement-only (GZIPPY_PW_STATS): per-event log of huge alloc/free
    /// sizes, dumped as a histogram line. Mutex is fine — stats-gated only.
    fn pw_size_log() -> &'static Mutex<Vec<(char, usize)>> {
        static L: OnceLock<Mutex<Vec<(char, usize)>>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn pw_stats_enabled() -> bool {
        static EN: OnceLock<bool> = OnceLock::new();
        *EN.get_or_init(|| std::env::var_os("GZIPPY_PW_STATS").is_some())
    }

    fn pw_log_size(kind: char, size: usize) {
        if pw_stats_enabled() {
            pw_size_log().lock().unwrap().push((kind, size));
        }
    }

    /// Stats line for the measurement loop (`GZIPPY_PW_STATS=1`).
    pub fn dump_pw_stats(tag: &str) {
        if std::env::var_os("GZIPPY_PW_STATS").is_none() {
            return;
        }
        {
            let log = pw_size_log().lock().unwrap();
            let mut hist: std::collections::BTreeMap<(char, usize), usize> =
                std::collections::BTreeMap::new();
            for &(k, s) in log.iter() {
                *hist.entry((k, s.div_ceil(1_048_576))).or_default() += 1;
            }
            let line: Vec<String> = hist
                .iter()
                .map(|(&(k, mib), &n)| format!("{k}{mib}M:{n}"))
                .collect();
            eprintln!("[pw-retain {tag}] sizes {}", line.join(" "));
        }
        eprintln!(
            "[pw-retain {tag}] hits={} misses={} misfit_released={} installs={} displaced={} grow_in_place={} trimmed_mib={:.1}",
            PW_HITS.load(Ordering::Relaxed),
            PW_MISSES.load(Ordering::Relaxed),
            PW_MISFIT_RELEASED.load(Ordering::Relaxed),
            PW_INSTALLS.load(Ordering::Relaxed),
            PW_DISPLACED.load(Ordering::Relaxed),
            PW_GROW_IN_PLACE.load(Ordering::Relaxed),
            PW_TRIMMED_BYTES.load(Ordering::Relaxed) as f64 / 1_048_576.0,
        );
    }

    /// Slot for the CURRENT thread: the bound ThreadPool worker index, or
    /// the shared unbound slot (consumer thread / tests without a binding).
    #[inline]
    fn pw_my_slot() -> usize {
        crate::decompress::parallel::chunk_buffer_pool::current_worker_pool_index()
            .map(|i| i.min(PW_WORKER_SLOTS - 1))
            .unwrap_or(PW_WORKER_SLOTS)
    }

    /// Read the header of a pw block. SAFETY: `user` must have been returned
    /// by `PwRetainAlloc`'s huge path (header written at fresh allocation and
    /// never touched since — the caller only writes at/after `user`).
    #[inline]
    unsafe fn pw_read_header(user: *mut u8) -> PwHeader {
        unsafe { std::ptr::read_unaligned(user.sub(PW_HEADER_GAP) as *const PwHeader) }
    }

    /// Release a pw block's backing memory to rpmalloc (the munmap path).
    unsafe fn pw_release(user: *mut u8) {
        let h = unsafe { pw_read_header(user) };
        assert_eq!(
            h.magic, PW_MAGIC,
            "pw_retain: header corrupted or non-pw block routed to pw_release"
        );
        ensure_thread_initialized();
        unsafe { raw_free(user.sub(h.hdr_offset)) };
    }

    /// Per-worker huge-block retention allocator. `RpmallocAlloc` delegates
    /// to this when `GZIPPY_PW_RETAIN` != "0" (and the slab is off); the unit
    /// tests exercise it directly (unconditional). Sub-threshold allocations
    /// route to plain rpmalloc exactly like `RpmallocAlloc`'s raw path.
    #[derive(Copy, Clone, Debug, Default)]
    pub struct PwRetainAlloc;

    impl PwRetainAlloc {
        fn alloc_huge(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let slot = pw_my_slot();
            let taken = PW_SLOTS[slot].swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !taken.is_null() {
                // SAFETY: only pw blocks are ever installed into PW_SLOTS.
                let h = unsafe { pw_read_header(taken) };
                assert_eq!(
                    h.magic, PW_MAGIC,
                    "pw_retain: retained block header corrupted"
                );
                if h.usable >= layout.size() && h.hdr_offset >= layout.align() {
                    // user ptr is `hdr_offset`-aligned; hdr_offset >= align
                    // (both powers of two) ⇒ requested alignment holds.
                    PW_HITS.fetch_add(1, Ordering::Relaxed);
                    pw_log_size('h', layout.size());
                    return Ok(NonNull::slice_from_raw_parts(
                        unsafe { NonNull::new_unchecked(taken) },
                        layout.size(),
                    ));
                }
                // Misfit: release rather than reinstall, so the slot
                // self-corrects to the worker's current working-set size.
                PW_MISFIT_RELEASED.fetch_add(1, Ordering::Relaxed);
                unsafe { pw_release(taken) };
            }
            PW_MISSES.fetch_add(1, Ordering::Relaxed);
            pw_log_size('m', layout.size());
            ensure_thread_initialized();
            let ea = layout.align().max(PW_HEADER_GAP); // power of two
            let total = ea.checked_add(layout.size()).ok_or(AllocError)?;
            let orig = unsafe { rpmalloc_sys::rpaligned_alloc(ea, total) as *mut u8 };
            if orig.is_null() {
                return Err(AllocError);
            }
            // SAFETY: `orig` is `ea`-aligned with `total = ea + size` bytes;
            // user = orig + ea keeps `size` usable bytes and the header gap
            // [user-64, user) lies inside the allocation (ea >= 64).
            let user = unsafe { orig.add(ea) };
            let h = PwHeader {
                magic: PW_MAGIC,
                usable: layout.size(),
                hdr_offset: ea,
                owner: slot as u32,
                _pad: 0,
            };
            unsafe { std::ptr::write_unaligned(user.sub(PW_HEADER_GAP) as *mut PwHeader, h) };
            Ok(NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(user) },
                layout.size(),
            ))
        }

        /// SAFETY: `ptr` must be a live pw huge block whose current layout is
        /// `layout` (the deallocation layout — its size is the Vec's final
        /// capacity, i.e. what this use-cycle actually touched at most).
        unsafe fn free_huge(ptr: NonNull<u8>, layout: Layout) {
            let user = ptr.as_ptr();
            let h = unsafe { pw_read_header(user) };
            assert_eq!(
                h.magic, PW_MAGIC,
                "pw_retain: freeing a non-pw block through the huge path"
            );
            pw_log_size('f', layout.size());
            pw_log_size('U', h.usable);
            // Retained-resident budget (GZIPPY_PW_KEEP_MIB, default 6): the
            // freeing layout is the Vec's CAPACITY, and the per-chunk buffers
            // are RESERVED far above their decoded size (silesia: 16-32 MiB
            // reservations for ~4.4 MiB chunks — measured histogram
            // f16M:30 f32M:4), so trim-to-layout still retained ~16 MiB
            // resident per slot (silesia T8 RSS +33%, gate is <=+10%). Cap
            // the retained WARM PREFIX at this budget; the trimmed tail
            // re-faults on reuse, but the VMA survives — no mmap/munmap +
            // cross-core TLB-shootdown churn, which is the other half of the
            // slab's measured win. Measured knee (guest, interleaved
            // median-of-9, model T8 wall vs OFF / silesia T8 RSS vs the
            // LOWEST observed OFF baseline): keep=3 +4.7% wall (misses the
            // >=+5% bar); keep=4 +7.5% wall, RSS +8.1%; keep=5 +8% wall,
            // RSS +10.2% (straddles the <=+10% gate); keep=6 +11% wall,
            // RSS +9..13.7% (straddles); uncapped +10% wall, RSS +30%
            // (fails). keep=4 is past the wall knee AND clears the RSS gate
            // with margin on every cell — the default.
            fn pw_keep_budget() -> usize {
                static B: OnceLock<usize> = OnceLock::new();
                *B.get_or_init(|| {
                    std::env::var("GZIPPY_PW_KEEP_MIB")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|mib| mib * 1024 * 1024)
                        .unwrap_or(4 * 1024 * 1024)
                })
            }
            // TRIM-ON-INSTALL (the "sized by what that worker last used"
            // clause, made literal — v1 retained at LARGEST-EVER capacity and
            // failed the silesia RSS gate +52%/+34%): drop residency of pages
            // beyond this cycle's used capacity via MADV_DONTNEED. The VMA
            // and `header.usable` are untouched, so a later BIGGER request
            // still fits and merely re-faults its tail; the retained
            // footprint is capped at the worker's live working-set size.
            // Correctness: the trimmed range is strictly inside OUR live
            // allocation [user, user+usable); DONTNEED'd anonymous pages read
            // back as zeros on next touch — within the allocator's
            // uninitialized-memory contract (callers must write before read,
            // same as any fresh allocation). Done BEFORE the slot install so
            // no concurrent taker can be writing while we trim.
            #[cfg(target_os = "linux")]
            {
                let keep = layout.size().min(pw_keep_budget());
                if h.usable > keep {
                    const PAGE: usize = 4096;
                    // madvise needs page-aligned addr: `user` is only 64-aligned.
                    let start = (user as usize + keep).div_ceil(PAGE) * PAGE;
                    let end = (user as usize + h.usable) & !(PAGE - 1);
                    if end > start {
                        PW_TRIMMED_BYTES.fetch_add((end - start) as u64, Ordering::Relaxed);
                        unsafe {
                            libc::madvise(
                                start as *mut libc::c_void,
                                end - start,
                                libc::MADV_DONTNEED,
                            );
                        }
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            let _ = layout;
            let slot = (h.owner as usize).min(PW_SLOT_COUNT - 1);
            // Keep-NEWEST: install this block into the OWNER's slot; release
            // any displaced older resident. One xchg on the hot path.
            let prev = PW_SLOTS[slot].swap(user, Ordering::AcqRel);
            PW_INSTALLS.fetch_add(1, Ordering::Relaxed);
            if !prev.is_null() {
                PW_DISPLACED.fetch_add(1, Ordering::Relaxed);
                unsafe { pw_release(prev) };
            }
        }
    }

    unsafe impl Allocator for PwRetainAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            if layout.size() == 0 {
                return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
            }
            if layout.size() >= PW_RETAIN_THRESHOLD {
                return Self::alloc_huge(layout);
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
            if layout.size() >= PW_RETAIN_THRESHOLD {
                return unsafe { Self::free_huge(ptr, layout) };
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
            if old_layout.size() >= PW_RETAIN_THRESHOLD {
                // The block carries its TRUE capacity in the header — resolve
                // grow against that, never the caller's layout (slab lesson).
                let h = unsafe { pw_read_header(ptr.as_ptr()) };
                assert_eq!(h.magic, PW_MAGIC, "pw_retain: grow on corrupted block");
                if h.usable >= new_layout.size() && h.hdr_offset >= new_layout.align() {
                    PW_GROW_IN_PLACE.fetch_add(1, Ordering::Relaxed);
                    return Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()));
                }
            }
            // Move: each side routes by its own size, so threshold crossings
            // (small→huge) always re-allocate through the right path.
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
            // In place only when BOTH sides stay huge (dispatch safety): a
            // huge→small in-place shrink would later route the small-layout
            // deallocate to rpfree(user_ptr) — the wrong pointer.
            if old_layout.size() >= PW_RETAIN_THRESHOLD && new_layout.size() >= PW_RETAIN_THRESHOLD
            {
                let h = unsafe { pw_read_header(ptr.as_ptr()) };
                assert_eq!(h.magic, PW_MAGIC, "pw_retain: shrink on corrupted block");
                if h.hdr_offset >= new_layout.align() {
                    return Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()));
                }
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
            // memlife: component-agnostic allocator total (the closure anchor).
            #[cfg(parallel_sm)]
            crate::decompress::parallel::memlife::allocator_total(layout.size());
            if pw_retain_enabled() {
                return PwRetainAlloc.allocate(layout);
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
            if pw_retain_enabled() {
                return unsafe { PwRetainAlloc.deallocate(ptr, layout) };
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
            if pw_retain_enabled() {
                return unsafe { PwRetainAlloc.grow(ptr, old_layout, new_layout) };
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
            if pw_retain_enabled() {
                return unsafe { PwRetainAlloc.shrink(ptr, old_layout, new_layout) };
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

    #[cfg(test)]
    mod pw_retain_tests {
        use super::*;
        use crate::decompress::parallel::chunk_buffer_pool::bind_worker_pool_index;
        use allocator_api2::alloc::Allocator;

        // Each test binds a DISTINCT high worker slot (58-63) so concurrently
        // running tests (and any production-path test using worker indexes
        // 0..T) can never share a retention slot with these assertions.

        const MIB: usize = 1024 * 1024;

        fn layout(size: usize, align: usize) -> Layout {
            Layout::from_size_align(size, align).unwrap()
        }

        fn write_tag(p: NonNull<[u8]>, tag: u64) {
            unsafe { std::ptr::write_unaligned(p.as_ptr() as *mut u64, tag) };
        }

        fn read_tag_raw(p: *mut u8) -> u64 {
            unsafe { std::ptr::read_unaligned(p as *const u64) }
        }

        fn user_ptr(p: NonNull<[u8]>) -> NonNull<u8> {
            NonNull::new(p.as_ptr() as *mut u8).unwrap()
        }

        /// Hit = same pointer AND dirty contents intact (pages never left the
        /// process, so the pre-free tag survives — a munmap/mmap cycle at the
        /// same address would read back zeros).
        #[test]
        fn pw_reuse_same_worker_and_misfit_rejection() {
            bind_worker_pool_index(58);
            let a = PwRetainAlloc;
            let l8 = layout(8 * MIB, 16);
            let p1 = a.allocate(l8).unwrap();
            assert!(p1.len() >= 8 * MIB);
            write_tag(p1, 0xDEAD_BEEF_0058_0001);
            let p1_addr = user_ptr(p1);
            unsafe { a.deallocate(p1_addr, l8) };

            // Same-size re-alloc: must reuse the retained block, dirty.
            let p2 = a.allocate(l8).unwrap();
            assert_eq!(
                user_ptr(p2),
                p1_addr,
                "same-worker re-alloc must hit the slot"
            );
            assert_eq!(
                read_tag_raw(p2.as_ptr() as *mut u8),
                0xDEAD_BEEF_0058_0001,
                "retained block must be the SAME resident pages (dirty reuse)"
            );
            unsafe { a.deallocate(user_ptr(p2), l8) };

            // Larger request: 8 MiB resident block must be REJECTED (layout
            // fit), released, and a fresh fitting block returned.
            let l16 = layout(16 * MIB, 16);
            let p3 = a.allocate(l16).unwrap();
            assert!(p3.len() >= 16 * MIB);
            write_tag(p3, 0xDEAD_BEEF_0058_0003);
            // Whole block writable end to end.
            unsafe { *(p3.as_ptr() as *mut u8).add(16 * MIB - 1) = 0x5A };
            let p3_addr = user_ptr(p3);
            unsafe { a.deallocate(p3_addr, l16) };

            // Smaller request FITS inside the retained 16 MiB block (true
            // block size from the header, not the caller layout).
            let p4 = a.allocate(l8).unwrap();
            assert_eq!(
                user_ptr(p4),
                p3_addr,
                "smaller request must reuse the bigger retained block"
            );
            assert_eq!(read_tag_raw(p4.as_ptr() as *mut u8), 0xDEAD_BEEF_0058_0003);
            unsafe { a.deallocate(user_ptr(p4), l8) };
        }

        /// The affinity contract: blocks are allocated on a worker but freed
        /// on the consumer thread — the free must install into the OWNER's
        /// slot (header `owner`), not the freeing thread's slot.
        #[test]
        fn pw_cross_thread_free_returns_to_owner_slot() {
            bind_worker_pool_index(59);
            let a = PwRetainAlloc;
            let l = layout(6 * MIB, 16);
            let p = a.allocate(l).unwrap();
            write_tag(p, 0xDEAD_BEEF_0059_0001);
            let addr = user_ptr(p).as_ptr() as usize;

            // Free on a DIFFERENT, unbound thread (the consumer model).
            std::thread::spawn(move || {
                let ptr = NonNull::new(addr as *mut u8).unwrap();
                unsafe { PwRetainAlloc.deallocate(ptr, layout(6 * MIB, 16)) };
            })
            .join()
            .unwrap();

            // The OWNER (this thread, slot 59) must hit on re-alloc.
            let p2 = a.allocate(l).unwrap();
            assert_eq!(
                p2.as_ptr() as *mut u8 as usize,
                addr,
                "cross-thread free must return the block to the OWNER's slot"
            );
            assert_eq!(read_tag_raw(p2.as_ptr() as *mut u8), 0xDEAD_BEEF_0059_0001);
            unsafe { a.deallocate(user_ptr(p2), l) };
        }

        /// Slot eviction policy: one block per slot, keep-NEWEST — the
        /// displaced older resident is released, never accumulated.
        #[test]
        fn pw_slot_eviction_keeps_newest() {
            bind_worker_pool_index(61);
            let a = PwRetainAlloc;
            let l = layout(5 * MIB, 16);
            let pa = a.allocate(l).unwrap();
            let pb = a.allocate(l).unwrap();
            write_tag(pa, 0xAAAA_0061_0001);
            write_tag(pb, 0xBBBB_0061_0002);
            let addr_b = user_ptr(pb);
            unsafe { a.deallocate(user_ptr(pa), l) }; // installs A
            unsafe { a.deallocate(addr_b, l) }; // displaces A, keeps B

            let p = a.allocate(l).unwrap();
            assert_eq!(user_ptr(p), addr_b, "slot must hold the NEWEST freed block");
            assert_eq!(read_tag_raw(p.as_ptr() as *mut u8), 0xBBBB_0061_0002);
            unsafe { a.deallocate(user_ptr(p), l) };
        }

        /// grow resolves against the block's TRUE capacity (header), in
        /// place when it fits; shrink stays in place only huge→huge.
        #[test]
        fn pw_grow_shrink_in_place_and_threshold_crossing() {
            bind_worker_pool_index(62);
            let a = PwRetainAlloc;
            // Seed the slot with a 16 MiB block.
            let l16 = layout(16 * MIB, 16);
            let p = a.allocate(l16).unwrap();
            let seed_addr = user_ptr(p);
            unsafe { a.deallocate(seed_addr, l16) };

            // 4 MiB request reuses the 16 MiB block...
            let l4 = layout(4 * MIB, 16);
            let q = a.allocate(l4).unwrap();
            assert_eq!(user_ptr(q), seed_addr);
            write_tag(q, 0xCAFE_0062_0001);
            // ...and grow to 12 MiB resolves IN PLACE (true usable = 16 MiB).
            let l12 = layout(12 * MIB, 16);
            let q2 = unsafe { a.grow(user_ptr(q), l4, l12) }.unwrap();
            assert_eq!(
                user_ptr(q2),
                seed_addr,
                "grow within true capacity must be in place"
            );
            assert_eq!(read_tag_raw(q2.as_ptr() as *mut u8), 0xCAFE_0062_0001);

            // Grow PAST true capacity: must move and preserve contents.
            let l32 = layout(32 * MIB, 16);
            let q3 = unsafe { a.grow(user_ptr(q2), l12, l32) }.unwrap();
            assert!(q3.len() >= 32 * MIB);
            assert_eq!(
                read_tag_raw(q3.as_ptr() as *mut u8),
                0xCAFE_0062_0001,
                "moved grow must copy old contents"
            );

            // Shrink huge→huge: in place.
            let l8 = layout(8 * MIB, 16);
            let q4 = unsafe { a.shrink(user_ptr(q3), l32, l8) }.unwrap();
            assert_eq!(
                user_ptr(q4),
                user_ptr(q3),
                "huge→huge shrink stays in place"
            );

            // Shrink across the threshold: must MOVE (dispatch safety) and
            // the resulting small block must free cleanly on the raw path.
            let l1 = layout(MIB, 16);
            let q5 = unsafe { a.shrink(user_ptr(q4), l8, l1) }.unwrap();
            assert_eq!(read_tag_raw(q5.as_ptr() as *mut u8), 0xCAFE_0062_0001);
            unsafe { a.deallocate(user_ptr(q5), l1) };
        }

        /// Alignment is part of the fit check: an under-aligned resident
        /// block must be rejected; fresh blocks honor the requested align.
        #[test]
        fn pw_alignment_fit_rejection() {
            bind_worker_pool_index(60);
            let a = PwRetainAlloc;
            let l128 = layout(5 * MIB, 128);
            let p = a.allocate(l128).unwrap();
            assert_eq!(p.as_ptr() as *mut u8 as usize % 128, 0);
            let addr = user_ptr(p);
            unsafe { a.deallocate(addr, l128) };

            // Same layout: hit, still aligned.
            let p2 = a.allocate(l128).unwrap();
            assert_eq!(user_ptr(p2), addr);
            unsafe { a.deallocate(user_ptr(p2), l128) };

            // Stricter alignment than the resident block carries: reject.
            let l4k = layout(5 * MIB, 4096);
            let p3 = a.allocate(l4k).unwrap();
            assert_eq!(p3.as_ptr() as *mut u8 as usize % 4096, 0);
            unsafe { a.deallocate(user_ptr(p3), l4k) };
        }

        /// Aliasing fuzz, modeled on `slab_alloc_fuzz_aliasing_and_sizes`:
        /// randomized allocate/grow/shrink/deallocate of mixed small+huge
        /// sizes with per-allocation tags. If the retention layer ever hands
        /// the same memory to two live allocations, a tag gets clobbered.
        #[test]
        fn pw_alloc_fuzz_aliasing_and_sizes() {
            bind_worker_pool_index(63);
            let a = PwRetainAlloc;
            let mut st = 0x243f_6a88_85a3_08d3u64;
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

        /// Trim-on-install: after a block is freed under a SMALLER layout
        /// than its true capacity, the tail beyond that layout is dropped
        /// from residency (reads back zero on Linux), while the warm prefix
        /// and the block's reusability for big requests are preserved.
        #[cfg(target_os = "linux")]
        #[test]
        fn pw_trim_on_install_drops_tail_keeps_prefix() {
            bind_worker_pool_index(57);
            let a = PwRetainAlloc;
            let l16 = layout(16 * MIB, 16);
            let p = a.allocate(l16).unwrap();
            let addr = user_ptr(p);
            // Marker deep in the tail (8 MiB) + prefix tag.
            unsafe {
                std::ptr::write_unaligned(
                    (p.as_ptr() as *mut u8).add(8 * MIB) as *mut u64,
                    0x7A11_0057_0001,
                )
            };
            unsafe { a.deallocate(addr, l16) }; // full-size free: no trim

            // Reuse at 4 MiB, then free under the 4 MiB layout → tail >4 MiB trimmed.
            let l4 = layout(4 * MIB, 16);
            let q = a.allocate(l4).unwrap();
            assert_eq!(user_ptr(q), addr);
            write_tag(q, 0x7A11_0057_0002);
            unsafe { a.deallocate(user_ptr(q), l4) };

            // Big request still fits (true capacity in header)...
            let r = a.allocate(l16).unwrap();
            assert_eq!(
                user_ptr(r),
                addr,
                "trimmed block must still satisfy its full capacity"
            );
            // ...prefix stayed warm, tail was dropped (zero-filled on touch).
            assert_eq!(read_tag_raw(r.as_ptr() as *mut u8), 0x7A11_0057_0002);
            let tail = unsafe {
                std::ptr::read_unaligned((r.as_ptr() as *mut u8).add(8 * MIB) as *const u64)
            };
            assert_eq!(tail, 0, "tail beyond the freeing layout must be DONTNEED'd");
            unsafe { a.deallocate(user_ptr(r), l16) };
        }

        /// Zero-size allocations stay on the dangling path.
        #[test]
        fn pw_zero_size() {
            let a = PwRetainAlloc;
            let l = layout(0, 16);
            let p = a.allocate(l).unwrap();
            assert_eq!(p.len(), 0);
            unsafe { a.deallocate(NonNull::new(p.as_ptr() as *mut u8).unwrap(), l) };
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
    arena::dump_pw_stats(tag);
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
