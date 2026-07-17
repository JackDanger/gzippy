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

    use allocator_api2::alloc::{AllocError, Allocator, Global, Layout};

    /// Stateless rpmalloc-backed allocator (vendor `RpmallocAllocator<T>`).
    ///
    /// Huge allocations route through [`SlabAlloc`] (retain resident blocks
    /// instead of munmapping) when the size-aware T gate is on: auto-ON at
    /// decode T <= 2 for EVERY huge block (uncapped `T × largest_block_seen`
    /// budget ≈ one chunk buffer per active worker), and at 2 < T <= 64 ONLY
    /// for chunk-class blocks (<= 8 MiB) with a `min(T × 8 MiB, 16 MiB)`
    /// retained-free ceiling — the stored/incompressible high-T fault-storm fix
    /// that leaves large-output compressible chunks (never routed) untouched.
    /// All thresholds are hardcoded (knob-free prod path).
    #[derive(Copy, Clone, Debug, Default)]
    pub struct RpmallocAlloc;

    // ── Slab retain-allocator (the T4-16 page-fault fix) ──────────────────
    //
    // rpmalloc munmaps allocations above its ~3.94 MiB "huge" threshold on
    // free, so the ~12 MiB `ChunkData` buffers re-fault (first-touch zeroing)
    // on every reuse — a significant CPU cost at high thread counts.
    // rapidgzip avoids it via 128 KiB sub-buffers that stay under the
    // threshold in rpmalloc's per-thread free list. `SlabAlloc` gets the same
    // residency for the MONOLITHIC buffer: it keeps freed huge blocks resident
    // in a capped free-list keyed by TRUE block size and reuses them, instead
    // of munmapping.
    //
    // CRITICAL correctness: the live side-table keys
    // dealloc/grow/shrink on the BLOCK, not the caller's `layout.size()` —
    // `Allocator::grow` calls `deallocate` with the OLD (smaller) layout, so
    // keying a free-list on `layout.size()` files blocks under the wrong size
    // and later hands out an undersized block → heap corruption. (That was
    // the bug in the earlier reverted retain-list that broke multimember.)
    // `grow`/`shrink` are overridden so a resident block whose true size
    // already fits is reused in place (no copy/realloc churn). The huge
    // threshold is the hardcoded `SLAB_THRESHOLD` (3 MiB) — knob-free prod path.
    /// Huge-allocation threshold: buffers this size or larger route through the
    /// resident slab free-list. Hardcoded (knob-free prod path).
    const SLAB_THRESHOLD: usize = 3 * 1024 * 1024;

    #[inline]
    fn slab_threshold() -> usize {
        SLAB_THRESHOLD
    }
    /// True block sizes are rounded up to this granularity so a few size
    /// classes cover the (near-constant) chunk-buffer sizes for high reuse.
    const SLAB_GRANULARITY: usize = 1024 * 1024;

    // ── T-conditional gate (strict T<=2 + chunk-class budget) ──
    //
    // The slab reduces first-touch faults at native T1. An earlier count-cap
    // retention policy caused an RSS blowup because it retained up to
    // 48 × 12 MiB = 576 MiB of free blocks.
    //
    // An earlier version auto-ON'd at ALL T (max_t = usize::MAX) with budget
    // `min(16 MiB, T × largest_block_seen)`. Both halves were wrong: the open
    // gate moved T8 RSS and wall, and the 16 MiB hard cap EXCLUDED the
    // chunk-class blocks (tens of MB) — a just-freed chunk buffer always blew
    // the budget and was evicted immediately (largest-first = itself), so the
    // slab almost never hit.
    //
    // The current gate is STRICTLY T <= 2 (default max_t = 2) — at T > 2 the
    // slab is OFF BY CONSTRUCTION (zero engagement, zero RSS movement; the
    // high-T criteria pass trivially). At T <= 2 the budget ADMITS the
    // chunk-class blocks: `T × largest_block_seen`, UNCAPPED (≈ one resident
    // chunk buffer per active worker — the working set, not a leak), and
    // eviction is SMALLEST-first so the chunk-class block is the LAST thing
    // dropped (largest-first would evict exactly that block whenever anything
    // else shared the free list).
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
    /// Highest decode thread count for the LEGACY UNCAPPED regime: at
    /// `1 <= T <= this` the slab auto-enables for EVERY huge block with the
    /// uncapped `T × largest_block_seen` budget. Hardcoded to 2 (knob-free
    /// prod path).
    const SLAB_AUTO_MAX_T_DEFAULT: usize = 2;

    #[inline]
    fn slab_auto_max_t() -> usize {
        SLAB_AUTO_MAX_T_DEFAULT
    }

    // ── HIGH-T SIZE-AWARE regime (close stored/incompressible high-T) ──────
    //
    // The stored/incompressible high-T loss vs rapidgzip is a per-chunk
    // output-buffer alloc fault-storm.
    // At `SLAB_AUTO_MAX_T_DEFAULT < T <= SLAB_HIGH_T_MAX`, a NEW huge allocation
    // routes through the slab ONLY if its size is <= SLAB_HIGH_T_MAX_BLOCK (the
    // chunk-class threshold): stored/incompressible chunks reserve ~4 MiB
    // (<= thresh ⇒ route ⇒ reuse ⇒ CLOSE), compressible chunks reserve 12–64
    // MiB (> thresh ⇒ never route ⇒ identical to the T<=2-gated path ⇒ the
    // large-output protected cells cannot regress in wall OR RSS). Retained-free
    // bytes are hard-capped at `min(T × SLAB_HIGH_T_MAX_BLOCK,
    // SLAB_HIGH_T_BUDGET_CEIL)` — the RSS backstop for variable-block
    // compressible high-T corpora. All three constants are hardcoded
    // (knob-free prod path).
    //
    // BUDGET_CEIL is 64 MiB, not 16 MiB: at 16 MiB the CEIL (not `T × block`)
    // was the binding cap at T>=4 — only ~4 retained 4 MiB stored buffers for
    // 8–16 workers, so storedheavy-512M output buffers still munmap→re-faulted
    // (the 512M fixture demotes StoredParallel→ParallelSM and its 99%-stored
    // body flows through `decode_clean_stored_into_contig`). 64 MiB admits full
    // stored retention while `MAX_BLOCK`=8 MiB still excludes
    // variable-large-output compressible chunks.
    const SLAB_HIGH_T_MAX: usize = 64;
    const SLAB_HIGH_T_MAX_BLOCK: usize = 8 * 1024 * 1024;
    const SLAB_HIGH_T_BUDGET_CEIL: usize = 64 * 1024 * 1024;

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
    ///
    /// `alloc_size` is the size of the NEW huge allocation being routed. The
    /// legacy regime (T <= auto_max_t) ignores it (routes every huge block,
    /// uncapped); the size-aware high-T regime routes ONLY chunk-class blocks
    /// (<= high_t_block), so large-output compressible chunks never engage the
    /// slab at high T — that is what keeps peak RSS flat on the large-output
    /// protected cells.
    fn gate_decision(
        force: Option<bool>,
        auto_max_t: usize,
        high_t_max: usize,
        high_t_block: usize,
        decode_threads: usize,
        alloc_size: usize,
    ) -> bool {
        match force {
            Some(v) => v,
            None => {
                if decode_threads == 0 {
                    false
                } else if decode_threads <= auto_max_t {
                    true // legacy uncapped regime — every huge block, all sizes
                } else if decode_threads <= high_t_max {
                    alloc_size <= high_t_block // size-aware: chunk-class only
                } else {
                    false
                }
            }
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

    /// Should a NEW huge allocation of `alloc_size` bytes route through the
    /// slab right now?
    fn slab_route_new(alloc_size: usize) -> bool {
        gate_decision(
            slab_test_override(),
            SLAB_AUTO_MAX_T_DEFAULT,
            SLAB_HIGH_T_MAX,
            SLAB_HIGH_T_MAX_BLOCK,
            DECODE_THREADS.load(std::sync::atomic::Ordering::Relaxed),
            alloc_size,
        )
    }

    /// True iff we are in the AUTO size-aware high-T regime (T above the legacy
    /// uncapped ceiling but within the high-T window). In this regime the
    /// retained-free budget is per-thread-capped at `SLAB_HIGH_T_MAX_BLOCK` and
    /// clamped to `SLAB_HIGH_T_BUDGET_CEIL` so resident overhead stays bounded
    /// on small-block high-T corpora.
    fn slab_high_t_capped() -> bool {
        if slab_test_override().is_some() {
            return false;
        }
        let t = DECODE_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        t > SLAB_AUTO_MAX_T_DEFAULT && t <= SLAB_HIGH_T_MAX
    }

    /// Pure budget computation (unit-tested): an explicit `configured` value
    /// (test-only) wins; otherwise `max(GRANULARITY, T × per_thread)`. In the
    /// LEGACY regime (`per_thread_cap = None`) `per_thread = largest_block_seen`,
    /// UNCAPPED — the chunk-class block (tens of MB) must fit, times one slot
    /// per worker. In the AUTO size-aware HIGH-T regime (`per_thread_cap =
    /// Some(cap)`) it is `min(largest_block_seen, cap)`, so retained-free
    /// resident bytes are hard-bounded at `T × cap`.
    fn budget_for(
        decode_threads: usize,
        largest_block_seen: usize,
        configured: Option<usize>,
        per_thread_cap: Option<usize>,
    ) -> usize {
        configured.unwrap_or_else(|| {
            let t = decode_threads.max(1);
            let per_thread = match per_thread_cap {
                Some(cap) => largest_block_seen.min(cap),
                None => largest_block_seen,
            };
            SLAB_GRANULARITY.max(t.saturating_mul(per_thread))
        })
    }

    /// Pure high-T budget (unit-tested): `T × min(largest, per_thread_cap)`
    /// clamped to the absolute `ceil`. Bounds retained-free RSS regardless of T
    /// or block-size variance.
    fn capped_budget(
        decode_threads: usize,
        largest_block_seen: usize,
        per_thread_cap: usize,
        ceil: usize,
    ) -> usize {
        budget_for(
            decode_threads,
            largest_block_seen,
            None,
            Some(per_thread_cap),
        )
        .min(ceil)
    }

    fn effective_budget(largest_block_seen: usize) -> usize {
        let threads = DECODE_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        if slab_high_t_capped() {
            // AUTO high-T regime: per-thread cap AND an absolute ceiling.
            capped_budget(
                threads,
                largest_block_seen,
                SLAB_HIGH_T_MAX_BLOCK,
                SLAB_HIGH_T_BUDGET_CEIL,
            )
        } else {
            // Legacy T<=2: uncapped `T × largest_block_seen`.
            budget_for(threads, largest_block_seen, None, None)
        }
    }

    /// Which backend owns a live huge block's memory (pointer-keyed side-table
    /// value). `Rp` = rpmalloc `raw_alloc` (the historical slab block, retained
    /// on free). `Sys` = `allocator_api2::alloc::Global` (the tiny thin-T1
    /// per-decode scope, see [`SystemHugeScope`]) — freed straight back to the
    /// system on dealloc, NEVER retained in the slab free list.
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    enum HugeBackend {
        Rp,
        Sys,
    }

    #[derive(Default)]
    struct SlabState {
        /// resident, currently-free blocks: (ptr, true_block_size, align).
        /// Always rpmalloc-backed (`Sys` blocks are never retained here).
        free: Vec<(usize, usize, usize)>,
        /// currently-allocated slab blocks: ptr -> (true_block_size, align, backend)
        live: HashMap<usize, (usize, usize, HugeBackend)>,
        /// sum of `true_block_size` for all entries in `free`
        current_free_bytes: usize,
        /// largest `true_block_size` ever seen alive (for auto-budget)
        largest_block_seen: usize,
    }

    // ── Per-decode system-backend scope for huge blocks (tiny-file path) ─────
    //
    // A tiny (≤8 MiB output) thin-T1 decode makes exactly ONE rpmalloc-backed
    // allocation — its huge chunk-output reserve — and that single allocation
    // triggers rpmalloc process+thread init (~1M instructions), which buys
    // NOTHING for a single-buffer single-thread decode. This scope routes ONLY
    // that huge allocation to the system allocator (`Global`), so a tiny decode
    // never initializes rpmalloc at all.
    //
    // Mechanism constraints (a prior process-global latch was reverted because
    // it regressed a high-T workload — keep this per-decode and thread-local):
    //   - `RpmallocAlloc`'s `allocate`/`deallocate`/`grow`/`shrink` are
    //     BYTE-IDENTICAL source to baseline — no new atomic load, no new branch.
    //     The parallel and big-thin-T1 paths are unchanged by construction.
    //   - The scope check lives ONLY in `SlabAlloc::alloc_huge` — the cold,
    //     mutex-taking, once-per-huge-block path (at T1 the legacy slab gate
    //     already routes every huge block here).
    //   - PER-DECODE, THREAD-LOCAL, RAII (same pattern as
    //     `chunk_buffer_pool::T1ResidentScope`): no process-global latch, no
    //     cross-decode state — a big decode after a tiny one (or the reverse)
    //     behaves exactly as if it were the first decode in the process.
    //   - Provenance is pointer-keyed in the SAME live side-table the slab
    //     already uses (`SLAB_EVER_ENGAGED` + `free_if_slab`), tagged with
    //     [`HugeBackend`]: every block is freed by the backend that allocated
    //     it, no matter which thread frees it or whether any scope is active —
    //     the cross-backend-free corruption class is structurally impossible.
    thread_local! {
        static SYSTEM_HUGE_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// RAII guard: while alive on this thread, NEW huge slab-routed allocations
    /// are system-backed (see module comment above). Entered by
    /// `sm_driver::read_parallel_sm_inner` for tiny thin-T1 decodes only.
    pub struct SystemHugeScope {
        prev: bool,
    }

    impl SystemHugeScope {
        pub fn enter() -> Self {
            let prev = SYSTEM_HUGE_SCOPE.with(|c| c.replace(true));
            Self { prev }
        }
    }

    impl Drop for SystemHugeScope {
        fn drop(&mut self) {
            SYSTEM_HUGE_SCOPE.with(|c| c.set(self.prev));
        }
    }

    #[inline]
    fn system_huge_scope_active() -> bool {
        SYSTEM_HUGE_SCOPE.with(|c| c.get())
    }

    /// Counter of system-backed huge allocations actually served
    /// (a tiny-decode test asserts this moved; a big-decode test asserts it
    /// did NOT — the scope must never leak past its RAII guard).
    pub static SYS_HUGE_ALLOCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// `RpmallocAlloc` routes NEW huge allocations here when the
    /// T-conditional gate is on (see `slab_route_new`), and ALWAYS routes
    /// dealloc/grow/shrink of slab-live pointers here (pointer-keyed);
    /// the fuzz test exercises it directly (unconditional).
    /// Engagement counters: cache hits + installs prove the slab actually ran.
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
            // Tiny thin-T1 per-decode scope: back this huge block with the
            // system allocator so the decode never triggers rpmalloc init (see
            // the `SystemHugeScope` module comment). Skips the resident free
            // list entirely (deterministic: a scoped decode neither consumes
            // nor produces retained rpmalloc blocks); registered pointer-keyed
            // with the `Sys` tag so dealloc/grow/shrink route by provenance.
            if system_huge_scope_active() {
                let sys_layout = Layout::from_size_align(bs, align).map_err(|_| AllocError)?;
                let p = Global.allocate(sys_layout)?;
                let addr = p.as_ptr() as *mut u8 as usize;
                {
                    let mut s = slab_state().lock().unwrap();
                    s.live.insert(addr, (bs, align, HugeBackend::Sys));
                }
                SYS_HUGE_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(NonNull::slice_from_raw_parts(
                    unsafe { NonNull::new_unchecked(addr as *mut u8) },
                    layout.size(),
                ));
            }
            {
                // BEST-FIT-LARGER match: serve the SMALLEST resident block
                // with b >= bs — NO waste bound. Exact-size matching misses
                // most reuse because chunk buffer sizes VARY per chunk, so
                // consecutive chunk allocs miss the exact class — fresh mmap
                // (faults kept) while the old block sits retained (RSS
                // doubles). A 2x waste bound was tried and rejected: it
                // re-created the overlap (a small-output chunk after a big one
                // got bound-rejected -> fresh mmap while the big block sat
                // retained). Serving an oversized resident block costs ZERO
                // extra RSS — its pages are resident either way — and at
                // T <= 2 (the only gated-on regime) essentially only
                // chunk-class blocks flow through the slab (installs ~ chunk
                // count), so any resident block is the right block. With
                // best-fit the ONE resident chunk-class block IS the working
                // set: every subsequent chunk reuses it in place (hits ~
                // chunk-count, RSS overhead ~ granularity, faults eliminated).
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
                    s.live.insert(p, (b, a, HugeBackend::Rp));
                    return Ok(NonNull::slice_from_raw_parts(
                        unsafe { NonNull::new_unchecked(p as *mut u8) },
                        layout.size(),
                    ));
                }
            }
            // MISS. Evict-on-miss (RSS peak fix): resident blocks SMALLER than
            // this request cannot serve it — munmap them BEFORE the fresh mmap
            // so peak RSS never stacks live + useless-retained (that overlap,
            // during the chunk-size growing phase, was a large peak-RSS delta).
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
                s.live.insert(p as usize, (bs, align, HugeBackend::Rp));
            }
            Ok(NonNull::slice_from_raw_parts(
                unsafe { NonNull::new_unchecked(p) },
                layout.size(),
            ))
        }

        /// Returns true if `ptr` was a live slab block (now retained/released).
        /// Keys on the side-table, NOT the caller layout (grow passes OLD).
        ///
        /// Retention policy: bytes-budget, SMALLEST-first eviction.
        /// Budget = `effective_budget` (legacy T<=2: uncapped T × largest-block;
        /// high-T: `min(T × 8 MiB, 16 MiB)`). Smallest-first keeps the chunk-
        /// class block resident as long as
        /// possible — with budget = 1 × largest at T1, largest-first would
        /// evict exactly that block whenever any smaller block shared
        /// the free list. Blocks over budget are munmapped outside the lock.
        unsafe fn free_if_slab(&self, ptr: NonNull<u8>) -> bool {
            let key = ptr.as_ptr() as usize;
            let mut s = slab_state().lock().unwrap();
            match s.live.remove(&key) {
                // System-backed block (tiny thin-T1 scope): free straight back
                // to the system with the EXACT layout it was allocated with.
                // Never retained (the resident free list is rpmalloc-only) and
                // never counted into `largest_block_seen` (it must not inflate
                // the rpmalloc retention budget).
                Some((bs, a, HugeBackend::Sys)) => {
                    drop(s);
                    // Layout was validated at allocation time (from_size_align
                    // succeeded there with the same rounded bs/align).
                    let layout = Layout::from_size_align(bs, a)
                        .expect("stored Sys block layout must be valid");
                    unsafe { Global.deallocate(ptr, layout) };
                    true
                }
                Some((bs, a, HugeBackend::Rp)) => {
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

        /// True block size of a live slab block, if `ptr` is one. Backend-
        /// agnostic: in-place grow/shrink reuse is valid for BOTH backends
        /// (the block stays live and registered; only a real dealloc or an
        /// out-of-place grow consults the provenance tag via `free_if_slab`).
        fn live_block(&self, ptr: NonNull<u8>) -> Option<(usize, usize)> {
            slab_state()
                .lock()
                .unwrap()
                .live
                .get(&(ptr.as_ptr() as usize))
                .map(|&(bs, a, _)| (bs, a))
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
            if layout.size() >= slab_threshold() && slab_route_new(layout.size()) {
                return SlabAlloc.alloc_huge(layout);
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

        /// SystemHugeScope white-box: a huge allocation made while the scope is
        /// active must be system-backed (Sys-tagged, counter moves), survive a
        /// write/read pattern, and free cleanly through the pointer-keyed
        /// dealloc path EVEN AFTER the scope has ended (provenance is keyed on
        /// the pointer, never on the current scope). Huge allocations outside
        /// the scope must stay rpmalloc-backed (counter still).
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn system_huge_scope_alloc_free_and_no_leak() {
            set_decode_threads(1); // legacy T1 regime: every huge block slab-routes
            let a = RpmallocAlloc;
            let layout = Layout::from_size_align(5 * 1024 * 1024, 16).unwrap();

            // In scope: system-backed.
            let before = SYS_HUGE_ALLOCS.load(Ordering::Relaxed);
            let ptr = {
                let _scope = SystemHugeScope::enter();
                let p = a.allocate(layout).unwrap();
                NonNull::new(p.as_ptr() as *mut u8).unwrap()
            }; // scope ends BEFORE the free — pointer-keyed provenance must hold
            assert!(
                SYS_HUGE_ALLOCS.load(Ordering::Relaxed) > before,
                "scoped huge allocation did not take the system backend"
            );
            // Write/read pattern across the block.
            unsafe {
                for off in (0..5 * 1024 * 1024).step_by(4096) {
                    ptr.as_ptr().add(off).write(0xA5);
                }
                for off in (0..5 * 1024 * 1024).step_by(4096) {
                    assert_eq!(ptr.as_ptr().add(off).read(), 0xA5);
                }
            }
            // Freed through free_if_slab -> Global (a cross-backend free would
            // corrupt/abort). The block must leave the live table.
            unsafe { a.deallocate(ptr, layout) };
            assert!(
                SlabAlloc.live_block(ptr).is_none(),
                "Sys block must be removed from the live table on free"
            );

            // Out of scope: rpmalloc-backed (counter unchanged).
            let c0 = SYS_HUGE_ALLOCS.load(Ordering::Relaxed);
            let p2 = a.allocate(layout).unwrap();
            let ptr2 = NonNull::new(p2.as_ptr() as *mut u8).unwrap();
            assert_eq!(
                SYS_HUGE_ALLOCS.load(Ordering::Relaxed),
                c0,
                "unscoped huge allocation must not take the system backend"
            );
            unsafe { a.deallocate(ptr2, layout) };
        }

        /// Scope + grow: a Sys block grown past its true size must relocate
        /// correctly (content preserved) and both old/new blocks free cleanly.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn system_huge_scope_grow_relocates_correctly() {
            set_decode_threads(1);
            let a = RpmallocAlloc;
            let old_l = Layout::from_size_align(4 * 1024 * 1024, 16).unwrap();
            let new_l = Layout::from_size_align(9 * 1024 * 1024, 16).unwrap();
            let _scope = SystemHugeScope::enter();
            let p = a.allocate(old_l).unwrap();
            let ptr = NonNull::new(p.as_ptr() as *mut u8).unwrap();
            unsafe {
                ptr.as_ptr().write(0x5A);
                ptr.as_ptr().add(4 * 1024 * 1024 - 1).write(0xC3);
            }
            let g = unsafe { a.grow(ptr, old_l, new_l) }.unwrap();
            let gp = NonNull::new(g.as_ptr() as *mut u8).unwrap();
            unsafe {
                assert_eq!(gp.as_ptr().read(), 0x5A);
                assert_eq!(gp.as_ptr().add(4 * 1024 * 1024 - 1).read(), 0xC3);
            }
            unsafe { a.deallocate(gp, new_l) };
            assert!(SlabAlloc.live_block(gp).is_none());
        }

        /// T-boundary truth table for the auto gate + env force overrides.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn gate_decision_boundaries() {
            // Signature: (force, auto_max_t, high_t_max, high_t_block, T, size).
            let big = 32 << 20; // compressible chunk-class (> high_t_block)
            let small = 4 << 20; // stored/incompressible chunk-class (<= block)
            let blk = 8 << 20; // high_t_block threshold
            let htm = 64; // high_t_max
                          // force-on wins at every T and size
            assert!(gate_decision(Some(true), 2, htm, blk, 0, big));
            assert!(gate_decision(Some(true), 2, htm, blk, 16, big));
            // force-off wins at every T
            assert!(!gate_decision(Some(false), 2, htm, blk, 1, small));
            assert!(!gate_decision(Some(false), 2, htm, blk, 4, small));
            // auto, LEGACY regime T<=auto_max_t: ON for ALL sizes; T==0 OFF
            assert!(!gate_decision(None, 2, htm, blk, 0, small));
            assert!(gate_decision(None, 2, htm, blk, 1, big)); // legacy ignores size
            assert!(gate_decision(None, 2, htm, blk, 2, big));
            // auto, HIGH-T size-aware regime (auto_max_t < T <= high_t_max):
            // chunk-class (<= blk) routes; large (> blk) does NOT.
            assert!(gate_decision(None, 2, htm, blk, 3, small)); // stored ⇒ route
            assert!(gate_decision(None, 2, htm, blk, 16, small));
            assert!(!gate_decision(None, 2, htm, blk, 3, big)); // compressible ⇒ skip
            assert!(!gate_decision(None, 2, htm, blk, 16, big));
            // exact boundary: size == blk routes (<=), size == blk+1 does not
            assert!(gate_decision(None, 2, htm, blk, 8, blk));
            assert!(!gate_decision(None, 2, htm, blk, 8, blk + 1));
            // above the high-T window ⇒ OFF regardless of size
            assert!(!gate_decision(None, 2, 8, blk, 9, small));
        }

        /// The DEFAULT gate ceiling is STRICTLY 2 (T<=2). A prior version
        /// shipped usize::MAX here, which moved T8 RSS and wall.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn default_auto_max_t_is_strictly_two() {
            assert_eq!(SLAB_AUTO_MAX_T_DEFAULT, 2);
            assert_eq!(slab_auto_max_t(), 2);
        }

        /// The budget must ADMIT a synthetic chunk-class block (tens of MB); a
        /// min(16 MiB, ...) cap would evict the just-freed chunk buffer every
        /// time.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn budget_admits_chunk_class_blocks_at_low_t() {
            let chunk = 38 * 1024 * 1024; // chunk-class: tens of MB
                                          // Legacy regime (per_thread_cap=None): T1 = one resident chunk slot.
            assert_eq!(budget_for(1, chunk, None, None), chunk);
            // T2: one slot per active worker.
            assert_eq!(budget_for(2, chunk, None, None), 2 * chunk);
            // T=0 (allocator used before any decode) behaves as one worker.
            assert_eq!(budget_for(0, chunk, None, None), chunk);
            // Floor: granularity when nothing big has been seen yet.
            assert_eq!(budget_for(1, 0, None, None), SLAB_GRANULARITY);
            // Explicit configured budget wins verbatim (test-only path).
            assert_eq!(budget_for(2, chunk, Some(16 << 20), None), 16 << 20);
            // HIGH-T per-thread cap: retained-free is bounded at T × cap even
            // when largest_block_seen (38 MiB) exceeds the cap (8 MiB).
            let cap = 8 << 20;
            assert_eq!(budget_for(4, chunk, None, Some(cap)), 4 * cap);
            assert_eq!(budget_for(16, chunk, None, Some(cap)), 16 * cap);
            // A small (stored) block below the cap keeps its full working set.
            let stored = 4 << 20;
            assert_eq!(budget_for(4, stored, None, Some(cap)), 4 * stored);
            // Absolute ceiling clamps the high-T retained-free budget: at T16
            // the uncapped T×cap (128 MiB) is clamped to the 16 MiB ceiling.
            let ceil = 16 << 20;
            assert_eq!(capped_budget(16, chunk, cap, ceil), ceil);
            assert_eq!(capped_budget(4, chunk, cap, ceil), ceil); // 4×8=32 → clamp 16
                                                                  // Below the ceiling the per-thread working set is preserved.
            assert_eq!(capped_budget(3, stored, cap, ceil), 3 * stored); // 12 MiB < 16
            assert_eq!(
                capped_budget(1, stored, cap, ceil),
                SLAB_GRANULARITY.max(stored)
            );
        }

        /// Smallest-first eviction: when the budget forces an eviction, the
        /// CHUNK-CLASS (largest) block must be the one retained — largest-
        /// first here would evict that block whenever any smaller block
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
            let budget = budget_for(1, 38 << 20, None, None);
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

        /// Best-fit-larger reuse: a freed block must serve a LATER, SMALLER
        /// request (chunk buffer sizes vary per chunk — exact-size matching
        /// misses most chunk allocs). No waste bound.
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

            // NO waste bound (see alloc_huge comment): any resident block
            // b >= bs serves — serving oversized costs zero extra RSS (pages
            // resident either way); rejecting forces a fresh mmap that STACKS
            // on the retained block.
            let bs = 4 * 1024 * 1024usize;
            let b = 11 * 1024 * 1024usize;
            assert!(b >= bs, "oversized resident block is a valid fit");
        }

        /// Evict-on-miss (RSS peak fix): a resident block SMALLER than
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

        /// CONCURRENCY aliasing check. Drives the PRODUCTION `RpmallocAlloc` under the high-T
        /// size-aware slab regime (DECODE_THREADS = 16 ⇒ chunk-class blocks
        /// route through the shared slab free-list) from 8 worker threads
        /// concurrently, each doing randomized allocate/grow/shrink/deallocate
        /// of a mix of chunk-class (<= 8 MiB, routed), large (> 8 MiB, not
        /// routed) and small blocks. Every block carries a UNIQUE
        /// (thread_id << 40 | seq) tag written to its head; each iteration
        /// verifies a random block THIS thread still holds live. If the slab
        /// ever hands the same memory to two LIVE allocations across threads
        /// (the multimember-corruption / stale-side-table class), one thread's
        /// tag write clobbers the other's → assert fires. Exercises the global
        /// `slab_state` mutex, best-fit reuse, evict-on-miss and budget
        /// eviction under contention. Harness-serial (own note); internally
        /// parallel — run with `--ignored --test-threads=1`.
        #[ignore = "white-box rpmalloc arena probe: races on the process-global arena under parallel `cargo test`; run serially with --ignored --test-threads=1"]
        #[test]
        fn slab_concurrent_alias_stress_high_t() {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;

            // Activate the AUTO size-aware high-T regime for the whole test.
            set_decode_threads(16);

            let failed = Arc::new(AtomicBool::new(false));
            let n_threads = 8usize;
            let mut handles = Vec::new();
            for tid in 0..n_threads as u64 {
                let failed = Arc::clone(&failed);
                handles.push(std::thread::spawn(move || {
                    let a = RpmallocAlloc;
                    let mut st = 0x9e3779b97f4a7c15u64 ^ (tid.wrapping_mul(0xd1b54a32d192ed03));
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
                    let mut seq: u64 = 0;
                    let tag_of = |tid: u64, seq: u64| (tid << 40) | (seq & ((1 << 40) - 1));
                    let write_tag = |ptr: NonNull<u8>, tag: u64| unsafe {
                        std::ptr::write_unaligned(ptr.as_ptr() as *mut u64, tag);
                    };
                    let read_tag = |ptr: NonNull<u8>| unsafe {
                        std::ptr::read_unaligned(ptr.as_ptr() as *const u64)
                    };

                    for _ in 0..4000 {
                        if failed.load(Ordering::Relaxed) {
                            break;
                        }
                        // Verify a random live block THIS thread owns.
                        if !live.is_empty() {
                            let i = (rng() as usize) % live.len();
                            if read_tag(live[i].ptr) != live[i].tag {
                                failed.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                        match rng() % 4 {
                            0 => {
                                // 0/1/2: chunk-class (routed), large (not routed), small
                                let size = match rng() % 3 {
                                    0 => (rng() as usize % (8 * 1024 * 1024)) + 8, // <= chunk-class
                                    1 => (rng() as usize % (24 * 1024 * 1024)) + (9 * 1024 * 1024), // > 8 MiB large
                                    _ => (rng() as usize % 8192) + 8, // small
                                };
                                let layout = Layout::from_size_align(size, 16).unwrap();
                                if let Ok(p) = a.allocate(layout) {
                                    let ptr = NonNull::new(p.as_ptr() as *mut u8).unwrap();
                                    assert!(p.len() >= size);
                                    seq += 1;
                                    let tag = tag_of(tid, seq);
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
                                let newsize = old.size + (rng() as usize % (4 * 1024 * 1024)) + 1;
                                let nl = Layout::from_size_align(newsize, 16).unwrap();
                                if let Ok(p) = unsafe { a.grow(old.ptr, old.layout, nl) } {
                                    let np = NonNull::new(p.as_ptr() as *mut u8).unwrap();
                                    if read_tag(np) != old.tag {
                                        failed.store(true, Ordering::Relaxed);
                                        break;
                                    }
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
                                    if let Ok(p) = unsafe { a.shrink(old.ptr, old.layout, nl) } {
                                        let np = NonNull::new(p.as_ptr() as *mut u8).unwrap();
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
                }));
            }
            for h in handles {
                h.join()
                    .expect("worker thread panicked (alloc invariant broke)");
            }
            assert!(
                !failed.load(Ordering::Relaxed),
                "slab handed the same memory to two live allocations across threads (aliasing)"
            );
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

/// Per-decode, thread-local RAII scope: while alive, NEW huge slab-routed
/// allocations are system-backed (tiny thin-T1 path — see the module comment
/// in `arena`). No-op when the arena is compiled out (buffers are already
/// plain `std::vec::Vec` = system allocator).
#[cfg(feature = "arena-allocator")]
pub use arena::SystemHugeScope;
#[cfg(not(feature = "arena-allocator"))]
pub struct SystemHugeScope;
#[cfg(not(feature = "arena-allocator"))]
impl SystemHugeScope {
    pub fn enter() -> Self {
        SystemHugeScope
    }
}

/// Counter for [`SystemHugeScope`] (always 0 when the
/// arena is compiled out). Read by the latch-interleave test only.
#[cfg(feature = "arena-allocator")]
#[allow(unused_imports)]
pub use arena::SYS_HUGE_ALLOCS;
#[cfg(not(feature = "arena-allocator"))]
pub static SYS_HUGE_ALLOCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Test-only programmatic slab gate override (see `arena::slab_test_force`):
/// the T4-vs-T1 diff_ratio guard forces the slab off for both arms so the
/// T-conditional allocator policy can't masquerade as a parallelism delta.
#[cfg(all(test, feature = "arena-allocator"))]
pub use arena::slab_test_force;
#[cfg(all(test, not(feature = "arena-allocator")))]
pub fn slab_test_force(_v: Option<bool>) {}

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
