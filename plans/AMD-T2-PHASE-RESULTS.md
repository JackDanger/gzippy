# AMD-T2 PHASE LOCATE — RESULTS (2026-06-22)

Box: solvency AMD EPYC 7282 Zen2, `root@10.0.2.240`, FROZEN gov=performance/boost=0,
cores 8,10,12,14 pinned away from a HOT roaming `llama-server` (load avg 11–16 the
whole session — see CAVEAT). Subject = `kernel-converge-A` @ `fc661a58` (phase-timing
instrument incl. main_start/main_end PRE/POST marks), built native+pure on box in
/dev/shm. Gate-4: `path=ParallelSM`, sha==zcat (monorepo). Instrument Gate-0:
conservation gap=0.000ms, monotonic, all 9 marks fired (PASS) on every run.

## TL;DR VERDICT
The AMD-T2 gz-vs-rg excess is **NOT a fat decode phase and NOT a fat serial wrapper
INSIDE the decode**. The instrument (Gate-0 self-validated) localizes it to a serial
region **OUTSIDE the userspace decode**: process exec/runtime-init + **kernel
address-space teardown of gz's elevated peak RSS** (`process::exit` skips Rust
destructors → the kernel unmaps the whole RSS at `_exit`). gz's userspace decode span
is at **rg-parity**. The prior-locate hypothesis ("~13ms serial wrapper INSIDE the
drive, needs a phase instrument") is REFUTED by this better instrument: the in-`main`
wrapper (PRE+POST) is ~5ms; the gz-specific excess is the process-lifecycle teardown.

## Phase split (instrument, monorepo-T2, medians N=11–13, load-contended)
PRE  main_start->decode_entry : ~4.0 ms    (CLI parse / file open / mmap / route)
     decode_entry->envelope    : ~0.01 ms
     envelope->scaffold_built  : ~0.05 ms  (block-finder + pool spawn — CHEAP)
     scaffold->first_output    : ~73 ms    (first-chunk decode latency)
     first_output->consumer    : ~115 ms   (steady-state parallel decode)
     consumer->finalize        : ~0.8 ms
     finalize->crc_verified    : ~0.01 ms
POST crc_verified->main_end    : ~0.9 ms
=> serial userspace wrapper (PRE + setup + finalize + POST) ≈ 5 ms. The wall is the
   two parallel-decode phases. (envelope+scaffold ~0.06ms KILLS the "block-finder
   bootstrap dominates" sub-hypothesis.)

## Consolidated interleaved (monorepo-T2, N=13, A/A-gated)
gz process-wall 209.9ms ; gz A/A 205.4ms (A/A ratio ~1.00–1.02) ; rg process-wall 192.6ms
gz/rg wall = 1.090, excess = **17.3 ms** (> A/A spread ~4.5ms → real, Gate-1).
gz instrument-wall (main_start->main_end, = PRE+decode+POST) = **190.4 ms < rg total 192.6 ms**.
=> gz's entire userspace decode is at/below rg's TOTAL wall. The 17.3ms excess matches
   the (process-wall − instrument-wall) gap = **process exec + kernel teardown**.

## Gate-2 (causal): teardown scales with RSS (perturbation knob = output/RSS size)
Measured WITHOUT perf (`/usr/bin/time`), teardown = full-wall − instrument-wall:
  monorepo : teardown 15.5 ms  @ peak RSS  99,072 KB (≈97 MB)
  silesia  : teardown 23.0 ms  @ peak RSS 242,964 KB (≈237 MB)
  => +140 MB RSS → +7.5 ms teardown (≈ +0.054 ms/MB). MONOTONIC + PROPORTIONAL
     ⇒ kernel address-space teardown of RSS is on the critical serial wall.
Peak RSS gz vs rg (monorepo-T2): **gz 99 MB vs rg 59 MB = 1.65×** (load-independent fact).
Fixed exec floor (tiny file, gz pw−iw): ~5.8 ms (with perf) — exec/runtime init.

## Convergence target (faithful-rg)
Reduce gz's peak RSS toward rg's (rg 59MB vs gz 97MB for the same 50MB output). rg
keeps RSS ≈ output-sized via ChunkData slab reuse (`RpmallocAllocator`,
`ChunkData.hpp`) + windowMap/chunk-cache eviction during the parallel phase; gz's
u16-marker chunk buffers + resident chunk_buffer_pool retain ~2× the footprint, and
`process::exit` (`src/main.rs:138`) skips destructors so the full RSS is torn down by
the kernel at exit. Options: (a) shrink peak RSS (recycle/free chunk + u16 marker
buffers earlier, smaller resident pool); (b) free the big allocations BEFORE the
parallel region drains so teardown overlaps useful work / drops RSS before `_exit`.

## CAVEAT (scope)
Absolute ms are LOAD-INFLATED (llama at load 11–16 all session; could not quiesce —
not our box to kill). The ROBUST, load-independent / interleaved facts: gz RSS 1.65×
rg; gz iw ≈ rg total (decode parity); teardown ∝ RSS. The teardown MAGNITUDE (15–23ms)
needs a llama-free reconfirm before sizing the prize. Single-arch AMD (Zen2). The
fixed ~10ms intercept of the 2-point teardown↔RSS fit conflates exec floor with a
baseline and is NOT separately trustworthy.
