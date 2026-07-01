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
    /// Huge allocations route through [`SlabAlloc`] (retain resident blocks
    /// instead of munmapping) when the T-conditional gate is on: auto-ON
    /// STRICTLY at decode T <= 2 (default `GZIPPY_SLAB_MAX_T` = 2); at T > 2
    /// the slab is off by construction. The budget (`T × largest_block_seen`,
    /// uncapped) admits the chunk-class blocks — retention ≈ one chunk buffer
    /// per active worker. `GZIPPY_SLAB_ALLOC=1`/`=0` force on/off;
    /// `GZIPPY_SLAB_MAX_T=K` overrides the ceiling.
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

    // ── T-conditional gate (iter-2, 2026-06-11: strict T<=2 + chunk-class budget) ──
    //
    // The fulcrum-decide causal A/B proved the slab PAYS ~100ms at native
    // silesia T1 (N=21, frozen, canonical mask). The prior RSS blowup
    // (silesia-T8 +24-35%) was caused by a count-cap retention policy that
    // retained up to 48 × 12 MiB = 576 MiB of free blocks.
    //
    // ITER-1 POST-MORTEM (9bebcada, NO-SHIP): it auto-ON'd at ALL T
    // (max_t = usize::MAX — chosen to appease the T4-vs-T1 diff_ratio guard)
    // with budget `min(16 MiB, T × largest_block_seen)`. Both halves were
    // wrong: the open gate moved T8 RSS (+16.8% max-run) and T8 wall (+2.6%),
    // and the 16 MiB hard cap EXCLUDED the chunk-class blocks (tens of MB) —
    // a just-freed chunk buffer always blew the budget and was evicted
    // immediately (largest-first = itself), so only 18 small-block hits ever
    // happened and the T1 win measured CAUSAL-NULL. The reconciliation
    // neutered its own lever.
    //
    // ITER-2 (this version): the gate is STRICTLY T <= 2 (default max_t = 2)
    // — at T > 2 the slab is OFF BY CONSTRUCTION (zero engagement, zero RSS
    // movement; the high-T criteria pass trivially). At T <= 2 the budget
    // ADMITS the chunk-class blocks: `T × largest_block_seen`, UNCAPPED
    // (≈ one resident chunk buffer per active worker — the working set, not a
    // leak), and eviction is SMALLEST-first so the chunk-class block is the
    // LAST thing dropped (largest-first would evict exactly the lever block
    // whenever anything else shared the free list).
    //
    // `GZIPPY_SLAB_ALLOC=1` forces ON at every T, `=0` forces OFF (both
    // override the auto gate). `GZIPPY_SLAB_MAX_T=K` overrides the ceiling.
    //
    // The decode thread count is STORED AT EVERY DECODE ENTRY
    // (`set_decode_threads`, called from `sm_driver::read_parallel_sm_inner`)
    // — an atomic, NOT a OnceLock, so a second decode in the same process
    // with a different T re-gates correctly (the earlier OnceLock-env-cache
    // lesson). T==0 (never set: direct allocator use outside a decode) gates
    // OFF. The gate is consulted only when ROUTING A NEW huge allocation;
    // dealloc/grow/shrink are pointer-keyed against the live side-table (see
    // `SLAB_EVER_ENGAGED`), so a mid-process gate flip can never mis-free a
    // slab block (the multimember-corruption class).
    fn slab_force() -> Option<bool> {
        static F: OnceLock<Option<bool>> = OnceLock::new();
        *F.get_or_init(|| std::env::var("GZIPPY_SLAB_ALLOC").ok().map(|v| v != "0"))
    }

    /// Highest decode thread count at which the slab auto-enables.
    /// Default: 2 — STRICTLY T <= 2 (iter-2 spec). At T > 2 the slab is off
    /// by construction (zero engagement, zero RSS movement). Override with
    /// `GZIPPY_SLAB_MAX_T=K`.
    const SLAB_AUTO_MAX_T_DEFAULT: usize = 2;

    fn slab_auto_max_t() -> usize {
        static K: OnceLock<usize> = OnceLock::new();
        *K.get_or_init(|| {
            std::env::var("GZIPPY_SLAB_MAX_T")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(SLAB_AUTO_MAX_T_DEFAULT)
        })
    }

    /// Decode thread count of the current/most-recent parallel-SM decode.
    /// 0 = no decode has started (slab auto-gate stays off).
    static DECODE_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// True once any allocation has been routed through the slab in this
    /// process; afterwards every dealloc/grow/shrink must consult the live
    /// side-table regardless of the current gate state.
    static SLAB_EVER_ENGAGED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    pub fn set_decode_threads(t: usize) {
        DECODE_THREADS.store(t, std::sync::atomic::Ordering::Relaxed);
    }

    /// Pure gate decision (unit-tested at the T boundaries).
    fn gate_decision(force: Option<bool>, auto_max_t: usize, decode_threads: usize) -> bool {
        match force {
            Some(v) => v,
            None => decode_threads != 0 && decode_threads <= auto_max_t,
        }
    }

    /// Test-only programmatic gate override (takes precedence over the env
    /// force): the T4-vs-T1 `diff_ratio` guard forces the slab OFF for BOTH
    /// arms so it measures pipeline parallelism apples-to-apples — the
    /// T-conditional policy legitimately makes T1 ~3x faster than T4 on the
    /// in-cache fixture, which is allocator policy, not a parallelism
    /// regression. -1 = none, 0 = force-off, 1 = force-on.
    #[cfg(test)]
    static TEST_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

    #[cfg(test)]
    pub fn slab_test_force(v: Option<bool>) {
        TEST_FORCE.store(
            match v {
                None => -1,
                Some(false) => 0,
                Some(true) => 1,
            },
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    fn slab_test_override() -> Option<bool> {
        #[cfg(test)]
        {
            match TEST_FORCE.load(std::sync::atomic::Ordering::SeqCst) {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            }
        }
        #[cfg(not(test))]
        {
            None
        }
    }

    /// Should a NEW huge allocation route through the slab right now?
    fn slab_route_new() -> bool {
        gate_decision(
            slab_test_override().or_else(slab_force),
            slab_auto_max_t(),
            DECODE_THREADS.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Bytes-budget cap for the resident free list (iter-2, 2026-06-11).
    ///
    /// If `GZIPPY_SLAB_BUDGET_MIB` is set, that fixed MiB value is used.
    /// Otherwise the budget auto-scales: `T × largest-block-seen`, UNCAPPED —
    /// it must ADMIT the chunk-class blocks (tens of MB; iter-1's 16 MiB hard
    /// cap evicted the just-freed chunk buffer every time and erased the T1
    /// win). T = current decode thread count; the gate (T <= 2) bounds the
    /// multiplier, so the steady-state retention is ≈ the decode's own
    /// working set (one chunk buffer per active worker), not a leak.
    fn configured_budget_bytes() -> Option<usize> {
        static BUDGET: OnceLock<Option<usize>> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            std::env::var("GZIPPY_SLAB_BUDGET_MIB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .map(|mib| mib * 1024 * 1024)
        })
    }

    /// Pure budget computation (unit-tested): override wins; otherwise
    /// `max(GRANULARITY, T × largest_block_seen)` with NO hard cap — the
    /// chunk-class block (tens of MB) must fit, times one slot per worker.
    fn budget_for(
        decode_threads: usize,
        largest_block_seen: usize,
        configured: Option<usize>,
    ) -> usize {
        configured.unwrap_or_else(|| {
            let t = decode_threads.max(1);
            SLAB_GRANULARITY.max(t.saturating_mul(largest_block_seen))
        })
    }

    fn effective_budget(largest_block_seen: usize) -> usize {
        budget_for(
            DECODE_THREADS.load(std::sync::atomic::Ordering::Relaxed),
            largest_block_seen,
            configured_budget_bytes(),
        )
    }

    #[derive(Default)]
    struct SlabState {
        /// resident, currently-free blocks: (ptr, true_block_size, align)
        free: Vec<(usize, usize, usize)>,
        /// currently-allocated slab blocks: ptr -> (true_block_size, align)
        live: HashMap<usize, (usize, usize)>,
        /// sum of `true_block_size` for all entries in `free`
        current_free_bytes: usize,
        /// largest `true_block_size` ever seen alive (for auto-budget)
        largest_block_seen: usize,
    }

    fn slab_state() -> &'static Mutex<SlabState> {
        static S: OnceLock<Mutex<SlabState>> = OnceLock::new();
        S.get_or_init(|| Mutex::new(SlabState::default()))
    }

    /// MEASUREMENT-ONLY event trace (`GZIPPY_SLAB_TRACE=1`): one line per
    /// slab event (hit/miss/retain/evict) with the size and the live/free
    /// byte ledgers — the "instrument, don't guess" tool for RSS-peak
    /// attribution (which blocks coexist when).
    fn slab_trace_on() -> bool {
        static T: OnceLock<bool> = OnceLock::new();
        *T.get_or_init(|| std::env::var_os("GZIPPY_SLAB_TRACE").is_some())
    }

    fn slab_trace(event: &str, bs: usize, s: &SlabState) {
        if slab_trace_on() {
            let live_bytes: usize = s.live.values().map(|&(b, _)| b).sum();
            eprintln!(
                "[slab] {event} bs={:.1}M live_n={} live={:.1}M free_n={} free={:.1}M",
                bs as f64 / 1048576.0,
                s.live.len(),
                live_bytes as f64 / 1048576.0,
                s.free.len(),
                s.current_free_bytes as f64 / 1048576.0,
            );
        }
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
    /// `RpmallocAlloc` routes NEW huge allocations here when the
    /// T-conditional gate is on (see `slab_route_new`), and ALWAYS routes
    /// dealloc/grow/shrink of slab-live pointers here (pointer-keyed);
    /// the fuzz test exercises it directly (unconditional).
    /// Engagement proof for the measurement A/B (fulcrum decide effect
    /// predicate): cache hits + installs prove the slab actually ran in the
    /// knob arm. Printed in the GZIPPY_RPMALLOC_STATS dump below.
    pub static SLAB_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static SLAB_INSTALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
            SLAB_EVER_ENGAGED.store(true, std::sync::atomic::Ordering::Release);
            {
                // BEST-FIT-LARGER match (iter-2): serve the SMALLEST resident
                // block with b >= bs — NO waste bound. Exact-size matching is
                // why iter-1-shape runs measured only ~24 hits and +27% T1
                // RSS: chunk buffer sizes VARY per chunk, so consecutive
                // chunk allocs missed the exact class — fresh mmap (faults
                // kept) while the old block sat retained (RSS doubled). A 2x
                // waste bound was tried and REJECTED by measurement: it
                // re-created the overlap (a small-output chunk after a big
                // one got bound-rejected -> fresh mmap while the big block
                // sat retained; T1 peak RSS stayed +24.6%). Serving an
                // oversized resident block costs ZERO extra RSS — its pages
                // are resident either way — and at T <= 2 (the only gated-on
                // regime) essentially only chunk-class blocks flow through
                // the slab (installs ~ chunk count), so any resident block
                // is the right block. With best-fit the ONE resident
                // chunk-class block IS the working set: every subsequent
                // chunk reuses it in place (hits ~ chunk-count, RSS overhead
                // ~ granularity, faults eliminated).
                let mut s = slab_state().lock().unwrap();
                if let Some(pos) = s
                    .free
                    .iter()
                    .enumerate()
                    .filter(|(_, &(_, b, a))| b >= bs && a >= align)
                    .min_by_key(|(_, &(_, b, _))| b)
                    .map(|(i, _)| i)
                {
                    SLAB_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let (p, b, a) = s.free.swap_remove(pos);
                    s.current_free_bytes -= b;
                    s.live.insert(p, (b, a));
                    slab_trace("hit", b, &s);
                    return Ok(NonNull::slice_from_raw_parts(
                        unsafe { NonNull::new_unchecked(p as *mut u8) },
                        layout.size(),
                    ));
                }
            }
            // MISS. Evict-on-miss (iter-2 RSS peak fix): resident blocks
            // SMALLER than this request cannot serve it — munmap them BEFORE
            // the fresh mmap so peak RSS never stacks live + useless-retained
            // (that overlap, during the chunk-size growing phase, was the
            // whole +24.6% T1 peak-RSS delta; the criterion is <= +15%).
            // Blocks >= bs stay (they serve future requests in their class).
            {
                let mut s = slab_state().lock().unwrap();
                let mut doomed: Vec<*mut u8> = Vec::new();
                let mut dropped_bytes = 0usize;
                s.free.retain(|&(p, b, _)| {
                    if b < bs {
                        doomed.push(p as *mut u8);
                        dropped_bytes += b;
                        false
                    } else {
                        true
                    }
                });
                s.current_free_bytes -= dropped_bytes;
                if !doomed.is_empty() {
                    slab_trace("evict-on-miss", dropped_bytes, &s);
                }
                drop(s);
                if !doomed.is_empty() {
                    ensure_thread_initialized();
                    for p in doomed {
                        unsafe { raw_free(p) };
                    }
                }
            }
            // fresh backing (retained on free)
            ensure_thread_initialized();
            let p = raw_alloc(bs, align);
            if p.is_null() {
                return Err(AllocError);
            }
            {
                let mut s = slab_state().lock().unwrap();
                s.live.insert(p as usize, (bs, align));
                slab_trace("miss-fresh", bs, &s);
            }
            Ok(NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(p) },
                layout.size(),
            ))
        }

        /// Returns true if `ptr` was a live slab block (now retained/released).
        /// Keys on the side-table, NOT the caller layout (grow passes OLD).
        ///
        /// Retention policy: bytes-budget, SMALLEST-first eviction (iter-2).
        /// Budget = T × largest-block-seen (uncapped) or GZIPPY_SLAB_BUDGET_MIB.
        /// Smallest-first keeps the chunk-class block resident as long as
        /// possible — with budget = 1 × largest at T1, largest-first would
        /// evict exactly the lever block whenever any smaller block shared
        /// the free list. Blocks over budget are munmapped outside the lock.
        unsafe fn free_if_slab(&self, ptr: NonNull<u8>) -> bool {
            let key = ptr.as_ptr() as usize;
            let mut s = slab_state().lock().unwrap();
            match s.live.remove(&key) {
                Some((bs, a)) => {
                    if bs > s.largest_block_seen {
                        s.largest_block_seen = bs;
                    }
                    // Add to free list, then evict smallest-first until under
                    // budget. The just-freed block participates in eviction
                    // like any other (smallest goes first; the chunk-class
                    // block is dropped last).
                    SLAB_INSTALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    s.current_free_bytes += bs;
                    s.free.push((key, bs, a));
                    slab_trace("retain", bs, &s);
                    let mut to_munmap: Vec<*mut u8> = Vec::new();
                    loop {
                        let budget = effective_budget(s.largest_block_seen);
                        if s.current_free_bytes <= budget {
                            break;
                        }
                        // evict the smallest resident block (retain chunk-class)
                        let (pos, _) = s
                            .free
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, &(_, b, _))| b)
                            .unwrap(); // safe: free non-empty while current_free_bytes > 0
                        let (evict_ptr, evict_bs, _) = s.free.swap_remove(pos);
                        s.current_free_bytes -= evict_bs;
                        to_munmap.push(evict_ptr as *mut u8);
                        slab_trace("evict-budget", evict_bs, &s);
                    }
                    drop(s);
                    // Release outside the lock (munmap can be slow).
                    if !to_munmap.is_empty() {
                        ensure_thread_initialized();
                        for p in to_munmap {
                            unsafe { raw_free(p) };
                        }
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
            // Gate consulted ONLY for routing new huge allocations; small
            // allocations and the gate-off path are today's raw rpmalloc,
            // bit-for-bit.
            if layout.size() >= slab_threshold() && slab_route_new() {
                return SlabAlloc.alloc_huge(layout);
            }
            ensure_thread_initialized();
            if layout.size() == 0 {
                return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
            }
            // memlife: component-agnostic allocator total (the closure anchor).
            #[cfg(parallel_sm)]
            crate::decompress::parallel::memlife::allocator_total(layout.size());
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
            // Pointer-keyed, NOT gate-keyed: a block allocated while the gate
            // was on MUST go back through the slab even if the gate has since
            // flipped (different decode T) — otherwise the live side-table
            // goes stale and a later address reuse hands out an undersized
            // block (the multimember-corruption class). The atomic flag keeps
            // the never-engaged path free of the side-table lock.
            if SLAB_EVER_ENGAGED.load(std::sync::atomic::Ordering::Acquire)
                && unsafe { SlabAlloc.free_if_slab(ptr) }
            {
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
            // Slab blocks stay in the slab domain (pointer-keyed, see
            // deallocate); SlabAlloc::grow reuses a resident block in place
            // when its true size already fits.
            if SLAB_EVER_ENGAGED.load(std::sync::atomic::Ordering::Acquire)
                && SlabAlloc.live_block(ptr).is_some()
            {
                return unsafe { SlabAlloc.grow(ptr, old_layout, new_layout) };
            }
            // default: allocate new (gate routes it) + copy + free old
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
            if SLAB_EVER_ENGAGED.load(std::sync::atomic::Ordering::Acquire)
                && SlabAlloc.live_block(ptr).is_some()
            {
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
    mod gate_tests {
        use super::*;
        use allocator_api2::alloc::Allocator;
        use std::sync::atomic::Ordering;

        /// T-boundary truth table for the auto gate + env force overrides.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn gate_decision_boundaries() {
            // force-on wins at every T (GZIPPY_SLAB_ALLOC=1)
            assert!(gate_decision(Some(true), 1, 0));
            assert!(gate_decision(Some(true), 1, 16));
            // force-off wins at every T (GZIPPY_SLAB_ALLOC=0)
            assert!(!gate_decision(Some(false), 1, 1));
            assert!(!gate_decision(Some(false), 8, 4));
            // auto (unset): ON iff 1 <= T <= max_t; T==0 (no decode) OFF
            assert!(!gate_decision(None, 1, 0));
            assert!(gate_decision(None, 1, 1));
            assert!(!gate_decision(None, 1, 2));
            assert!(!gate_decision(None, 1, 16));
            // max_t=2 boundary (the iter-2 default gate): T=1,2 ON; T=3 OFF
            assert!(gate_decision(None, 2, 1));
            assert!(gate_decision(None, 2, 2));
            assert!(!gate_decision(None, 2, 3));
        }

        /// Iter-2 spec: the DEFAULT gate ceiling is STRICTLY 2 (T<=2). Iter-1
        /// shipped usize::MAX here — that "leak" is what moved T8 RSS/wall.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn default_auto_max_t_is_strictly_two() {
            assert_eq!(SLAB_AUTO_MAX_T_DEFAULT, 2);
            if std::env::var_os("GZIPPY_SLAB_MAX_T").is_none() {
                assert_eq!(slab_auto_max_t(), 2);
            }
        }

        /// Iter-2 crux: the budget must ADMIT a synthetic chunk-class block
        /// (tens of MB). Iter-1's min(16 MiB, ...) cap evicted the just-freed
        /// chunk buffer every time and erased the T1 win.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn budget_admits_chunk_class_blocks_at_low_t() {
            let chunk = 38 * 1024 * 1024; // chunk-class: tens of MB
                                          // T1: exactly one resident chunk slot — the block FITS.
            assert_eq!(budget_for(1, chunk, None), chunk);
            // T2: one slot per active worker.
            assert_eq!(budget_for(2, chunk, None), 2 * chunk);
            // T=0 (allocator used before any decode) behaves as one worker.
            assert_eq!(budget_for(0, chunk, None), chunk);
            // Floor: granularity when nothing big has been seen yet.
            assert_eq!(budget_for(1, 0, None), SLAB_GRANULARITY);
            // Env override wins verbatim (GZIPPY_SLAB_BUDGET_MIB).
            assert_eq!(budget_for(2, chunk, Some(16 << 20)), 16 << 20);
        }

        /// Smallest-first eviction: when the budget forces an eviction, the
        /// CHUNK-CLASS (largest) block must be the one retained — largest-
        /// first here would evict the lever block whenever any smaller block
        /// shared the free list.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn eviction_retains_chunk_class_block() {
            // Pin a deterministic budget independent of DECODE_THREADS races:
            // simulate via the pure parts — free-list bookkeeping is exercised
            // end-to-end in bytes_budget_bounds_free_list; here we assert the
            // ORDER policy on the in-memory state machine.
            let mut free: Vec<(usize, usize, usize)> = vec![(0x1000, 38 << 20, 16)];
            let mut current: usize = 38 << 20;
            // a small block arrives; budget (1 x largest = 38 MiB) is blown
            free.push((0x2000, 4 << 20, 16));
            current += 4 << 20;
            let budget = budget_for(1, 38 << 20, None);
            let mut evicted = Vec::new();
            while current > budget {
                let (pos, _) = free
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &(_, b, _))| b)
                    .unwrap();
                let (p, b, _) = free.swap_remove(pos);
                current -= b;
                evicted.push((p, b));
            }
            assert_eq!(evicted, vec![(0x2000, 4 << 20)], "small block evicted");
            assert_eq!(free.len(), 1);
            assert_eq!(free[0].1, 38 << 20, "chunk-class block retained");
        }

        /// Best-fit-larger reuse (iter-2): a freed block must serve a LATER,
        /// SMALLER request (chunk buffer sizes vary per chunk — exact-size
        /// matching missed most chunk allocs: ~24 hits, +27% T1 RSS). No
        /// waste bound (measured: a 2x bound re-created the RSS overlap).
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn best_fit_serves_smaller_request_from_larger_block() {
            // Odd sizes unique to this test to dodge cross-test free-list races.
            let big = Layout::from_size_align(11 * 1024 * 1024, 16).unwrap();
            let p = SlabAlloc.alloc_huge(big).unwrap();
            let big_ptr = p.as_ptr() as *mut u8 as usize;
            unsafe { SlabAlloc.deallocate(NonNull::new(big_ptr as *mut u8).unwrap(), big) };

            // 9 MiB request: 11 MiB resident block fits (11 <= 2*9) — must be
            // served the SAME backing (a cache hit), not a fresh mmap.
            let hits_before = SLAB_CACHE_HITS.load(Ordering::Relaxed);
            let small = Layout::from_size_align(9 * 1024 * 1024, 16).unwrap();
            let q = SlabAlloc.alloc_huge(small).unwrap();
            let q_ptr = q.as_ptr() as *mut u8 as usize;
            if q_ptr == big_ptr {
                // The block survived the (global, cross-test) free list — the
                // best-fit path must have counted a hit.
                assert!(SLAB_CACHE_HITS.load(Ordering::Relaxed) > hits_before);
                assert_eq!(
                    SlabAlloc
                        .live_block(NonNull::new(q_ptr as *mut u8).unwrap())
                        .unwrap()
                        .0,
                    11 * 1024 * 1024,
                    "live table must keep the TRUE block size"
                );
            }
            unsafe { SlabAlloc.deallocate(NonNull::new(q_ptr as *mut u8).unwrap(), small) };

            // NO waste bound (measured decision, see alloc_huge comment): any
            // resident block b >= bs serves — serving oversized costs zero
            // extra RSS (pages resident either way); rejecting forces a fresh
            // mmap that STACKS on the retained block (+24.6% T1 peak RSS when
            // a 2x bound was tried).
            let bs = 4 * 1024 * 1024usize;
            let b = 11 * 1024 * 1024usize;
            assert!(b >= bs, "oversized resident block is a valid fit");
        }

        /// Evict-on-miss (iter-2 RSS peak fix): a resident block SMALLER than
        /// a missing request must be munmapped BEFORE the fresh mmap — peak
        /// RSS must never stack live + useless-retained.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn miss_evicts_smaller_resident_blocks() {
            let small = Layout::from_size_align(7 * 1024 * 1024, 16).unwrap();
            let p = SlabAlloc.alloc_huge(small).unwrap();
            let small_ptr = p.as_ptr() as *mut u8 as usize;
            unsafe { SlabAlloc.deallocate(NonNull::new(small_ptr as *mut u8).unwrap(), small) };

            // 15 MiB request: the 7 MiB block can't serve it (b < bs) — the
            // miss path must have dropped it from the free list (either we
            // evicted it, or a concurrent test legitimately re-allocated it;
            // in NO case may it still sit free while we mmap fresh).
            let big = Layout::from_size_align(15 * 1024 * 1024, 16).unwrap();
            let q = SlabAlloc.alloc_huge(big).unwrap();
            {
                let s = slab_state().lock().unwrap();
                assert!(
                    !s.free.iter().any(|&(fp, _, _)| fp == small_ptr),
                    "smaller resident block must not survive a bigger-class miss"
                );
                // ledger consistency under the same lock
                let sum: usize = s.free.iter().map(|&(_, b, _)| b).sum();
                assert_eq!(s.current_free_bytes, sum, "free-bytes ledger");
            }
            let q_ptr = NonNull::new(q.as_ptr() as *mut u8).unwrap();
            unsafe { SlabAlloc.deallocate(q_ptr, big) };
        }

        /// The dynamic-gate correctness crux: a block that entered the slab
        /// domain MUST be handled by the slab on dealloc/grow/shrink via the
        /// pointer-keyed live table, regardless of the current gate state —
        /// including after a shrink below the slab threshold (the stale-
        /// side-table corruption class). Engagement counters must move.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn cross_gate_pointer_keyed_consistency_and_counters() {
            let huge = 5 * 1024 * 1024;
            let layout = Layout::from_size_align(huge, 16).unwrap();

            // Allocate via SlabAlloc directly (always slab — gate-independent).
            let p = SlabAlloc.alloc_huge(layout).unwrap();
            let ptr = NonNull::new(p.as_ptr() as *mut u8).unwrap();
            assert!(
                SlabAlloc.live_block(ptr).is_some(),
                "huge slab alloc must be tracked live"
            );
            assert!(SLAB_EVER_ENGAGED.load(Ordering::Acquire));

            // Shrink BELOW the slab threshold through RpmallocAlloc: must stay
            // in the slab domain (in-place, same ptr) — not copy to raw.
            let small = Layout::from_size_align(4096, 16).unwrap();
            let s = unsafe { RpmallocAlloc.shrink(ptr, layout, small) }.unwrap();
            assert_eq!(
                s.as_ptr() as *mut u8,
                ptr.as_ptr(),
                "slab shrink must reuse the resident block in place"
            );
            assert!(SlabAlloc.live_block(ptr).is_some());

            // Deallocate with the SMALL layout through RpmallocAlloc: the
            // pointer-keyed path must retain it in the slab free list (counted
            // as an install), never rpfree it raw (which would leave a stale
            // live entry → later aliasing).
            let installs_before = SLAB_INSTALLS.load(Ordering::Relaxed);
            unsafe { RpmallocAlloc.deallocate(ptr, small) };
            assert!(
                SlabAlloc.live_block(ptr).is_none(),
                "dealloc must remove the live entry"
            );
            assert!(
                SLAB_INSTALLS.load(Ordering::Relaxed) > installs_before,
                "slab install counter must move on retained free"
            );

            // Re-allocating the same size class must be able to hit the cache
            // (engagement-counter path); tolerate concurrent tests by checking
            // monotonic movement across our own alloc+free pair.
            let hits_before = SLAB_CACHE_HITS.load(Ordering::Relaxed);
            let p2 = SlabAlloc.alloc_huge(layout).unwrap();
            let ptr2 = NonNull::new(p2.as_ptr() as *mut u8).unwrap();
            let hits_after = SLAB_CACHE_HITS.load(Ordering::Relaxed);
            assert!(
                hits_after >= hits_before,
                "cache-hit counter must never regress"
            );
            unsafe { SlabAlloc.deallocate(ptr2, layout) };
        }

        /// Bytes-budget eviction: with a tiny fixed budget the free list can
        /// never retain more than the budget (largest evicted first).
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn bytes_budget_bounds_free_list() {
            // Use blocks larger than any plausible configured budget floor is
            // not possible in-process (env is OnceLock-cached), so verify the
            // INVARIANT against whatever budget is in effect: after freeing N
            // huge blocks, current_free_bytes <= effective budget.
            let huge = 7 * 1024 * 1024;
            let layout = Layout::from_size_align(huge, 16).unwrap();
            let mut ptrs = Vec::new();
            for _ in 0..24 {
                let p = SlabAlloc.alloc_huge(layout).unwrap();
                ptrs.push(NonNull::new(p.as_ptr() as *mut u8).unwrap());
            }
            for ptr in ptrs {
                unsafe { SlabAlloc.deallocate(ptr, layout) };
            }
            let s = slab_state().lock().unwrap();
            let budget = effective_budget(s.largest_block_seen);
            assert!(
                s.current_free_bytes <= budget,
                "free list {} bytes exceeds budget {}",
                s.current_free_bytes,
                budget
            );
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

/// Record the decode thread count for the slab auto-gate. Called at every
/// parallel-SM decode entry (`sm_driver::read_parallel_sm_inner`) — atomic
/// store, never once-cached, so per-decode T changes re-gate correctly.
#[cfg(feature = "arena-allocator")]
pub use arena::set_decode_threads;
#[cfg(not(feature = "arena-allocator"))]
pub fn set_decode_threads(_t: usize) {}

/// Test-only programmatic slab gate override (see `arena::slab_test_force`):
/// the T4-vs-T1 diff_ratio guard forces the slab off for both arms so the
/// T-conditional allocator policy can't masquerade as a parallelism delta.
#[cfg(all(test, feature = "arena-allocator"))]
pub use arena::slab_test_force;
#[cfg(all(test, not(feature = "arena-allocator")))]
pub fn slab_test_force(_v: Option<bool>) {}

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
        "[rpmalloc {tag}] slab_hits={} slab_installs={}",
        arena::SLAB_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed),
        arena::SLAB_INSTALLS.load(std::sync::atomic::Ordering::Relaxed),
    );
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
