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

### Step 0 RESULT (2026-05-29) — D1 is DEAD, D2 is the entire game

Cheap proxy for the oracle: gzippy's **BGZF parallel path is already a
markers-free parallel clean decode** (independent 64 KiB blocks, libdeflate
FFI, zero markers, zero apply_window). Measured frozen/pinned, best-of-7,
silesia-bgzf (212 MB) vs silesia-large single-member (503 MB):

| T  | BGZF-clean | single-member (markers) |
|----|-----------|--------------------------|
| 1  | 1124      | 1080  |
| 4  | 3867      | 1367  |
| 8  | **6100**  | 1837  |
| 16 | **7029**  | 1734  |

**CORRECTION (2026-05-29, user caught the overclaim):** This is NOT a clean
kill-test and "D1 is dead" is RETRACTED. BGZF parallel uses **libdeflate
(C) FFI** (`bgzf.rs:473`), a **separate pipeline** (`bgzf.rs`, not
`single_member.rs`), on **independent 64 KiB blocks** (empty window by
construction). The single-member 1837 uses a **different backend** (ISA-L C
for clean tails + gzippy **pure-Rust ResumableInflate2** for the 156 MB
window-absent bootstrap) AND window-dependent 4 MB chunks. So 6100 vs 1837
moves FOUR variables at once: (a) markers vs none, (b) libdeflate vs
ISA-L+Rust, (c) bgzf pipeline vs single-member pipeline, (d) independent vs
window-dependent blocks. It CANNOT attribute the gap to the engine alone.

What 6100 honestly establishes: the hardware/memory subsystem + gzippy's
process can sustain ~6 GB/s with libdeflate on independent blocks → 1837 is
NOT a hardware/IO cap. It does NOT isolate the single-member data model from
the engine. The CLEAN discriminator is still the advisor's actual oracle:
disable markers INSIDE the single-member pipeline (same buffer pool, same
consumer, same ISA-L backend, known windows) so only the engine variable
moves. D1 vs D2 remains OPEN until that runs.

### REAL three-way numbers + corrected attribution (2026-05-29)

Audit finding (verified file:line): the SHIPPED x86_64 binary AND all prior
session benchmarks (`--features isal-compression`) use **C ISA-L FFI** for
clean chunks; gzippy's OWN pure-Rust `ResumableInflate2` is reachable only
under `--features pure-rust-inflate` (ships nowhere). So prior "single-member"
numbers measured a C library, not gzippy's decoder, for clean bytes.

Built both binaries; frozen, P-pinned, best-of-7, silesia-large (503 MB),
gzippy-PURE byte-exact (c070ed84):

| T  | gzippy-PURE (own Rust) | gzippy-ISAL (C) | rapidgzip |
|----|------------------------|------------------|-----------|
| 1  | 898  | 1083 | 884  |
| 4  | 1150 | 1378 | 1726 |
| 8  | 1623 | 1835 | 2235 |
| 16 | 1211 | 1477 | 2340 |

Corrected attribution:
- pure-Rust engine ~12% behind C ISA-L (achievement, unshipped).
- **T8 gap to rapidgzip (22%) is MOSTLY ENGINE, not pipeline.** gzippy-ISAL
  IPC 1.44 ≈ rapidgzip 1.42 → at equal IPC, rapidgzip executes fewer
  instructions/byte → the 156 MB pure-Rust **bootstrap overshoot** is the
  extra work. (My "22% is pipeline" was wrong — re-derived the retracted
  BGZF conclusion.)
- **T16 "collapse" is an HT/topology artifact, not a pipeline bug.** perf
  T8→T16: instructions FLAT (10.94B→11.10B), page-faults +9% only, IPC
  1.44→0.92 (16 threads on 8 physical P-cores contend for ports). Box has
  8 fast P-cores; gzippy is execution-port-bound so HT doesn't help;
  rapidgzip is memory-bound so it does. Not a fixable gzippy bug.

HARD-RESET DECISION (advisor): **NO.** The inner-loop commits are correct,
measured-positive, on-goal, and the bootstrap path is pure-Rust in the
shipped binary too. Falsified pipeline levers already reverted/gated-OFF.

### THE TARGET (revised again): the 156 MB bootstrap overshoot is ENGINE work

Root cause (per [[project_sm_bootstrap_overshoot_2026_05_29]]): silesia uses
large deflate blocks (2–14 MB). gzippy can't hand off from the slow pure-Rust
marker decode to fast ISA-L until a clean 32 KiB boundary appears, so it
decodes WHOLE multi-MB blocks (156 MB, ~31% of output) on the slow path. The
window is actually DETERMINISTIC for the sequential CLI (chunk N's window =
chunk N−1's last 32 KiB) — only the first ≤32 KiB of each chunk can reference
the unknown predecessor window; the rest is decodable clean.

Two complementary directions (both D2, both authorized open-territory per
CLAUDE.md inner-loop rule):
1. **Hand off to the clean fast path EARLY.** Decode only the chunk prefix
   that can reference the unknown window via markers; switch to ISA-L for the
   remainder even mid-large-block. Shrinks the 156 MB slow path drastically.
2. **Make the window-absent decode itself fast.** Close the inner-loop
   2.07× instruction gap (1.65× resumable-contract tax already partly
   addressed in f01eb74; 1.25× algorithm remains) and the **bad-spec 18%**
   (inner-loop branch mispredict) — the dominant IPC gap (1.08 vs 1.42).

Ceiling proof: 6100 MB/s shows there is no data-model ceiling below ~6 GB/s;
a clean-enough engine could in principle blow past rapidgzip. Even halving
the slow-path share should move T8 materially toward 2571+.

### Step 1+ (superseded by Step 0 result — D1 branch removed)

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
