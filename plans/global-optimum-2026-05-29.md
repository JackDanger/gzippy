# Global-Optimum Plan: closing the parallel single-member gap (2026-05-29)

**Directive (user):** Stop hill-climbing isolated levers. Treat the decoder
as ONE system; reason about how the measured effects interact; plan to a
GLOBAL optimum; if it's a two-month project, do it — but de-risk first.

**Status of the lever campaign:** ~16 isolated levers FALSIFIED (copy-elim,
allocator-swap ×2, prefetch-depth reduction, slab, buffer right-size,
MAX_POOLED, shared pool, vendor `nextNthEviction` guard). All hill-climbed
within a basin defined by two coupled architecture decisions. See
[[project_sm_bootstrap_overshoot_2026_05_29]] for the falsification record.

## The measurements (frozen, P-core-pinned i7-13700T, best-of-N + perf)

- T8: gzippy 1539–1879 MB/s vs rapidgzip 2571 (gap ~1.4–1.7×; GROWS at T16).
- TMA T8: retiring 23%, **bad-spec 18%**, frontend 11%, backend 33.7%
  (**DRAM 17.8%**, L3 6.1%, L2 1.2%, L1/store ~8.6%), core 15%.
  IPC gzippy 1.08 vs rapidgzip 1.42.
- Page faults: gzippy 12.4% of CPU (clear_page) vs rapidgzip 1.2%.
- Working set: ~27 live chunks at T8 (~324 MB). Depth is LOAD-BEARING:
  27=1826, 23=1421, 19=1285, 17=1023 MB/s (monotonic-worse shrinking).
- Engine: ~31% of silesia output (156 MB) decoded on the SLOW pure-Rust
  resumable marker path; inner loop 2.07× libdeflate instructions
  (1.65× resumable-contract tax + 1.25× algorithm).
- ~985 MB extra memmove (≈2× output) from marker/copy passes, on worker
  threads with slack (shared DRAM bandwidth though).

## Systemic synthesis (initial — then REFUTED, see below)

Two coupled decisions, not independent gaps:
- **D1 monolithic ~12 MB per-chunk buffers** → fault churn + DRAM +
  forced-deep-pipeline.
- **D2 slow pure-Rust marker-speculation engine** → bad-spec + instruction
  tax + 2× marker traffic + copy passes.
Markers (D2 output) live in buffers (D1 structure) → isolated levers fail.

## Adversarial refutation (Opus advisor, verified against vendor source)

The synthesis above is **motivated by the failed levers, not the TMA
budget.** Corrections:

1. **Bad-spec 18% is inner-loop branch-mispredict — D1 moves ZERO of it.**
   It's nearly as large as the whole DRAM stall and is the dominant IPC gap
   (1.08 vs 1.42). Burying it inside D2 was a category error.
2. **D1 alone is likely NEUTRAL and cannot be banked first.** The deep
   27-pipeline exists *to hide the very fault/DRAM latency D1 removes*; freed
   slack gets reabsorbed (this is exactly why copy-elim measured neutral).
   D1 only pays if D2 is *also* fixed so the pipeline no longer needs depth
   27 to hide engine latency. Coupled OPPOSITE to the initial assumption.
3. **Segmenting the CLEAN buffer is a vendor NO-OP.** `DecodedData.hpp:278-283`
   explicitly refuses to segment `dataBuffers` ("makes no sense"); only
   `dataWithMarkers` is 128 KiB-segmented. silesia is mostly clean
   (fused-LUT) → a faithful segmented port barely touches silesia faults.
   The real fault tax is **over-allocation** (`chunk_data.rs` `take_u8(
   max_decoded_chunk_size)` reserves ~12 MB regardless of actual decoded
   size) **+ the broken rpmalloc recycler** (Rust binding missing
   `rpmalloc_thread_initialize`; see [[project_allocator_framework_2026_05_28]]),
   NOT buffer shape.
4. **The bigger missed lever:** marker machinery solves *random-seek*
   uncertainty. The CLI does *whole-file sequential* decode, where chunk N's
   window is deterministically chunk N−1's last 32 KiB. The 156 MB slow
   marker path may be **self-inflicted** by porting a random-access
   architecture onto a sequential workload.

## THE PLAN — kill-test first, then branch

### Step 0 (DAYS, do FIRST): the discriminating oracle

Build a measurement-only oracle that removes the marker engine entirely:
1. Sequential fast pre-pass: decode the file once (ISA-L/libdeflate
   streaming) recording, at each ~4 MB-compressed chunk boundary, the
   (bit-offset, 32 KiB window) pair. This builds a window index. (Cost is a
   known, amortizable quantity — it is NOT what we time.)
2. Parallel second pass (THIS is timed): spawn T workers; each ISA-L-decodes
   its chunk with the KNOWN predecessor window into its buffer; consumer
   concatenates in order. Zero markers, zero `apply_window`, clean path only.
   (This is rapidgzip's "decode with existing index" mode.)
3. Measure MB/s + TMA of the timed second pass at T8/T16.

**Discriminates both theses at once:**
- **≈2571 MB/s** → the **engine/markers** was the whole game. The segmented
  buffer project (D1) is DEAD. Pursue: (a) known-window fast decode for the
  sequential CLI path (drop speculation where the window is deterministic),
  (b) inner-loop bad-spec reduction (the 18%). Two months saved.
- **still ~1.4× off** → the gap is the **data-model / allocator / pipeline**.
  THEN justify D1 — but target the **rpmalloc recycler fix**
  (`rpmalloc_thread_initialize` per worker) + **right-size the clean buffer
  to actual `dataSize()`** FIRST (both days-scale), and only segment
  `data_with_markers` (never the clean buffer) if faults persist.

Entry points: production driver `sm_driver::read_parallel_sm` →
`chunk_fetcher::drive`; per-chunk decode
`gzip_chunk::decode_chunk_marker_bootstrap_then_isal`; window flow via
`window_map`. The oracle is a parallel sibling that skips the bootstrap and
feeds known windows.

### Step 1+ (branch on Step 0's result)

- **Engine branch:** known-window decode + inner-loop bad-spec work
  (the inner Huffman loop is explicitly open territory per CLAUDE.md).
- **Data-model branch:** recycler fix → clean-buffer right-size →
  `data_with_markers` segmentation (vendor-faithful: clean stays monolithic).

## Guardrails

- Correctness non-negotiable: silesia md5 `c070ed84…` + multi-member
  differential on every step.
- Measure frozen (LXC 111/105 cgroup.freeze, <30 min) + P-core-pinned,
  best-of-N, same-session A/B. Wall numbers shift with box state; only
  same-session A/B counts.
- No isolated lever ships without same-session A/B; record falsifications.
