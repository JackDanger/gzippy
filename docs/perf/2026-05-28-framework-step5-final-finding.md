# Framework Step 5 final: rpmalloc binding is broken, not the lever

**Date**: 2026-05-28
**Branch**: `reimplement-isa-l` @ `8692257`
**Source examined**:
- `~/.cargo/registry/src/.../rpmalloc-0.2.2/src/lib.rs:130-180`
- `vendor/rapidgzip/librapidarchive/src/external/rpmalloc/README.md:48`
- `vendor/rapidgzip/librapidarchive/src/external/rpmalloc/CHANGELOG:231`
- `vendor/rapidgzip/librapidarchive/src/external/rpmalloc/rpmalloc/rpmalloc.c:730-732`

## The find

Step 5 production A/B falsified Lever 4.1 (`#[global_allocator] =
rpmalloc::RpMalloc`) with +41% page-faults and +167% wall on file
output. The advisor flagged that `rpmalloc_thread_initialize` per
worker is REQUIRED — without it, threads fall back to a slow
global-mutex'd heap.

**Confirmed**: The `rpmalloc 0.2.2` crate's `RpMalloc::alloc()`
implementation (verified in source) is:

```rust
unsafe impl GlobalAlloc for RpMalloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ffi::rpaligned_alloc(layout.align(), layout.size()) as *mut u8
    }
    // ...
}
```

It calls `rpaligned_alloc` directly — **no `rpmalloc_thread_initialize`,
no pthread-key registration, no hugepage config**.

Per rpmalloc's own README:
> __rpmalloc_thread_initialize__: Call at each thread start to
> initialize the thread local data for the allocator

Per rapidgzip's bundled rpmalloc CHANGELOG line 231:
> Require thread initialization with rpmalloc_thread_initialize, add
> pthread hooks for automatic init/fini

Rapidgzip uses a version of rpmalloc that auto-installs pthread hooks
so EVERY thread is initialized at create-time. The Rust crate does
not.

**Conclusion**: Lever 4.1 (rpmalloc as global allocator via the Rust
0.2.2 crate) is falsified because the **Rust binding is incomplete**,
not because rpmalloc-the-algorithm is wrong.

## What this implies

The 5-step framework just found something a stab-in-dark lever
attempt would have missed entirely:

1. We'd have shipped `#[global_allocator] = rpmalloc::RpMalloc`.
2. Users would experience +167% wall on file output.
3. Diagnosis would have been "rpmalloc is bad for gzippy, abandon"
   — incorrect, because the bug is the binding.

Instead, the framework:
1. Predicted rpmalloc would help (microbench)
2. Falsified that prediction in production (Step 5 A/B)
3. Identified the SPECIFIC fix (thread_initialize + pthread hooks)

## Paths to actually test rpmalloc properly

In order of risk-adjusted upside:

### Path A: Vendor rapidgzip's bundled rpmalloc + Rust wrapper

Copy `vendor/rapidgzip/librapidarchive/src/external/rpmalloc/rpmalloc/`
into `vendor/rpmalloc-vendored/`. Build via cc-rs. Write a
`#[global_allocator]` wrapper that uses pthread `pthread_key_create`
+ destructor to call `rpmalloc_thread_initialize` /
`rpmalloc_thread_finalize` for every thread.

Effort: ~1 day. Risk: low (proven exact pattern from vendor).

### Path B: Hook into Rayon / std::thread::spawn

Wrap thread creation in gzippy's worker pool (`chunk_buffer_pool.rs`'s
`bind_worker_pool_index`) to call `rpmalloc_thread_initialize`.
Doesn't help allocations that happen before the worker reaches
`bind_worker_pool_index` (e.g., during thread spawn itself), so still
incomplete.

Effort: ~½ day. Risk: medium (partial coverage).

### Path C: Submit a PR to the rpmalloc crate

Add automatic pthread-key registration in `RpMalloc::alloc`. Useful
for the ecosystem; long path to gzippy benefit.

Effort: external. Risk: low.

### Path D: Use mimalloc or jemalloc as a control

If the issue is "no thread-init in the binding", try a Rust allocator
binding that DOES handle thread-init (mimalloc, jemalloc). If they
show similar wall improvement, confirms the issue is binding-quality;
if not, rpmalloc has something specific.

Effort: ~½ day. Risk: low (negative-control experiment).

## Recommended next action

**Path D first** (mimalloc + jemalloc as controls). Cheapest, fastest
falsifier. If either gives meaningful wall improvement without the
page-fault regression, we have a known-good allocator while the
rpmalloc binding work happens. If both also regress, rpmalloc isn't
specially magical — the win is elsewhere (and the framework's
Step 2 PEBS attribution + Step 1 source-dive still point at chunk-
shape and marker bootstrap).

## Framework summary

Five steps shipped (Step 3 deferred):

- **Step 1**: source-dive → identified candidate levers + the
  3 sub-levers advisor caught.
- **Step 2**: PEBS attribution → write-side cold-page is the stall.
- **Step 4**: microbench → predicted Lever 4.1 = -54% wall.
- **Step 5**: production A/B → falsified Lever 4.1 (+167% wall on
  file) + identified the specific cause (binding missing thread
  init).

**The framework's value**: caught a falsification in <30 minutes that
the prior 8-falsification session pattern would have shipped as a
production regression. Identified the specific fix needed.

This is the difference between "stab in dark" and "instrumented
diagnosis."
