# ENGINE CAMPAIGN CHARTER (funded by user 2026-06-10)

The standing gated decision is RESOLVED: **the native pure-Rust engine rewrite is funded.**
Goal: gzippy-native >= 0.99x rapidgzip at EVERY thread count (the same bar), no C-FFI in its
decode graph. gzippy-isal keeps benefiting from every shared-pipeline win.

## Starting scorecard (official, b8e4fe58, frozen N=9 + N=21 resolution)
- native FAILS: silesia T1 0.609 (THE engine cell), T4 0.790, T8 0.952, T16 0.863 (but ties rg in
  1/21 runs — the speed state EXISTS, scheduling reaches it 5%); model 0.577-0.588; bignasa T8
  0.951 (N=21 resolved); nasa T4 RESOLVED-PASS 0.9998(!).
- native PASSES: nasa T16 1.142, ghcn T8 1.036, nasa T4 1.000, small everywhere.

## Direction (governed by banked memories — binding)
1. **u8-faithful architecture FIRST** (project_engine_plateau_pure_rust): the 0.667x asm plateau
   was measured on the WRONG u16-ring arch; rapidgzip's bulk decode is u8-direct. Build the
   ONE-engine u8-flip-in-place (project_faithful_unified_decoder): MarkerRing flips u16->u8 WIDTH
   in place when the window goes marker-free; clean bulk decodes u8-DIRECT. No second engine.
2. THEN re-attempt the inner-loop arsenal (multi-literal packed writes, BMI2 PEXT/BZHI dispatch,
   table prefetch, asm hot loops) — prior falsifications are NON-BINDING per CLAUDE.md (they
   predate PRELOAD/BMI2 and were on the wrong arch).
3. The engine SHAPE finding (2026-06-10 masked decomposition): rg marker-decodes whole chunks
   with its fast engine then ONE cache-friendly u8 LUT apply pass; gzippy's flip-hybrid
   (slow marker loop -> flip -> ISA-L/fold tail) loses on marker-heavy spans. The u8 rewrite
   should evaluate adopting rg's decode-all-then-apply shape vs the in-place flip — vendor
   deflate.hpp/ChunkData.hpp is the blueprint; cite file:line for every ported behavior.

## Process (the rules that survived this campaign)
- Every A/B masked (taskset 0,2,4,6,8,10,12,14) + canonical spine + frozen for bankable cells;
  N=21 for close calls. Pre-registered falsifiers. Kill-switches verified to disable fully.
- Byte-exactness: silesia/model/bignasa/nasa at T1/2/4/8/16, sha-pinned, every change.
- Workers: Sonnet for specified tasks, Fable for complicated; supervisor gates everything
  (Opus/Fable disproof gate before merge). Suites run natively on LXC 199.
- plans/orchestrator-status.md = the per-turn record; this file = the campaign spine.

## Phase plan
- P1: two-column map (vendor deflate.hpp Block u8 bulk + ChunkData apply ↔ gzippy marker_inflate
  u16 ring + flip sites), then the u8-flip-in-place design doc, then implementation skeleton
  with byte-exact tests. NO perf claims in P1.
- P2: land u8-direct clean bulk; masked A/B native T1/T4/T8 (the cells that isolate the engine).
- P3: inner-loop arsenal on the new arch (libdeflate/ISA-L techniques, then asm if still short).
- P4: bar matrix both builds; close-band cleanup (bignasa 0.951, silesia T16 scheduling-state).
