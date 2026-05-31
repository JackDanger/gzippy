# x86 high-thread decode gap — falsification ledger

**Purpose:** every lever attacked against the x86 high-thread single-member decode gap vs
native rapidgzip, with the *validated* measurement and the verdict. Do NOT re-test a row
marked DEAD without new evidence that the prior measurement was wrong. Each row names the
measurement so its instrument can be re-validated, not just trusted.

## The gap (frozen ground truth, 2026-05-30, host-frozen, best-of-7, pinned, sha-verified)
gzippy-prod 1684 | gzippy-clean-oracle 1938 | rapidgzip 2770 MB/s (silesia-large, T8).
Decomposition: bootstrap (marker speculation) ≈1.15–1.66× (load-sensitive); clean-pipeline
residual ≈1.17–1.41× = pure-Rust-vs-ISA-L inner-loop dispatch floor. **Both at hardware floors.**

## DEAD levers (do not reopen without falsifying the listed measurement)

| Lever | Result | Measurement (the instrument to re-validate) |
|---|---|---|
| Bootstrap overshoot (post-flip clean bytes) | DEAD | post_flip = 2.7% of bootstrap body (GZIPPY_VERBOSE) |
| Marker-decode = 82% of wall | DEAD (denominator) | 82% of decode-CPU; decode-CPU = 28% of WALL |
| Copy-elimination (drain/append/absorb) | DEAD wall-neutral | agent A/B: removing one copy = −0.0% wall; cache-misses flat 37→36M |
| Drain ring→Vec | DEAD | single-thread 1.06×; 8-thread test was CONFOUNDED (drain+flip), discarded |
| Page-faults (3.2× more) | DEAD (symptom) | faults/sec EQUAL across engines (523K≈534K<677K isal); prewarm/MADV = −300ms |
| Bootstrap distance-LUT (sub-lever) | small (+2%) | distance isn't the sub-bottleneck — NOT proof of a bandwidth floor (was mis-read as DEAD) |
| ~~Bootstrap = bandwidth-bound DEAD~~ | **RETRACTED** | agent a1e4b175: two pure-Rust decoders (deflate_block bootstrap vs clean) differ **2-6× on identical hw/data ⇒ COMPUTE-bound, real headroom**. The 69% LLC-miss is deflate_block's poor ring/u16 access pattern, fixable by a better decoder, not a DRAM floor. |
| Chunk-size — bigger | DEAD | regresses (parallelism loss); 4MB optimal |
| Chunk-size — smaller | DEAD | regresses (per-chunk overhead); oracle 1912→1027 |
| NT-stores / memory-locality | DEAD | gzippy moves 5.3× FEWER LLC-misses than rapidgzip; less memory-bound |
| Inner-loop branch-mispredict | DEAD wall-neutral | −13.3% branches byte-exact, wall in noise; TopdownL1 co-limited (BE31/Ret28/BS26) |
| Marker-VOLUME reduction (mid-decode window re-check) | DEAD | kill-test: 0% of 156MB is post-predecessor-window (validated probe: control 100% vs real 0%) |
| u8 / sparse-journal markers | DEAD | −60% (dense propagation → O(n²) journal) |
| rpmalloc-global / Z-prewarm / segmented buffer | DEAD | +167% / −15% / regress (binding bug + resident-set growth) |

## CLOSED (fixed to parity/win)
- **Incompressible (random100)**: was framed 1.56×, REAL gap ~13% (the "2299" was a measurement
  artifact, not reproducible). Closed to parity (+0.6–0.7%) — fold parallel prefix CRC + overlap
  the single-threaded Huffman tail with the parallel prefix copy. Landed `52e2361`/`9724b7f`.
  Residual = single-threaded Huffman tail (Amdahl), hidden not eliminated.
- **arm64 all threads**: gzippy fastest (rapidgzip lacks ISA-L there). Shipped.

## CORRECTED LOCALIZATION (agent a1e4b175, measure.sh + window-export, sha-verified)
The PIPELINE is EXCELLENT, not the gap: gzippy clean-pipeline ceiling = **4577 MB/s = 2.1× rapidgzip**,
near-linear scaling; production driver already uses rapidgzip's continuous prefetch pool (no wave-barrier;
the work-stealing +28% was an oracle/import-path artifact). **The ENTIRE x86 T8 gap (1.732×, measure.sh) is
the BOOTSTRAP marker decoder**: ~8-10 chunks run wholly through the slow pure-Rust `deflate_block`
(70-250 MB/s) and head-of-line-block the in-order consumer (fetcher_get = 77% of consumer wall). It is
COMPUTE-bound (see retracted row). So the gap is ONE thing, gated YES (it's ~the whole gap), and the
decoder slice that's DEAD is the CLEAN inner loop — NOT the bootstrap decoder.

## BOOTSTRAP DECODE-RATE = WALL-DEAD (agent a10edf55, RESOLVED — discipline worked)
Built FastBootstrap (clean libdeflate-style u16 hot loop, flat buffer): discriminator PASS = **1.72-1.89×
faster decode than deflate_block, byte-identical** (u16 store does NOT cap — a0f2ce23 falsified; deflate_block
slow = 128KiB ring + per-symbol single-slot writes + backward-scan + drain + Vec::push). BUT production WALL
A/B (measure.sh N=11, 4 rounds, 3 host-frozen) = **TIE (1.04-1.10×, in noise)**: the bootstrap decode is
PIPELINE-OVERLAPPED — speeding it removes worker-thread slack but does NOT shorten the in-order CONSUMER
critical path. **This FALSIFIES "the entire gap is the bootstrap decoder" (a1e4b175 mis-localized).** Decode
RATE (bootstrap + clean inner) is now wall-DEAD entirely. Landed default-OFF behind GZIPPY_FAST_BOOTSTRAP
(deflate_block kept), branch bootstrap-u16-discriminator @5514453, NOT merged (a correct simplification to keep).

## THE OPEN LEVER (Amdahl-gated YES) — STRUCTURAL CONSUMER / PIPELINE, the ~86% slice
The gap is the in-order CONSUMER's serial critical path (window-dependency resolution chain: resolve chunk N-1's
markers → publish N's window → process N — a serial LATENCY chain the worker decode can't shorten) + buffer-
lifecycle. Clean-import (free windows, no chain) = 4577 = 2.1× rapidgzip; production (with the chain) = 1244-1684.
The lever is to make gzippy's consumer/window-resolution as lean as rapidgzip's (faithful structural port — the
thing CLAUDE.md wrongly rescinded on the broken-oracle "pipeline at parity"). ANALYZE via CONSUMER CRITICAL-PATH /
dependency-graph (trace_v2 + timeline_analyze), NOT decode asm (wall-dead). Pin the serial bottleneck before porting.

## OPEN / lower-priority
- Single-member-T2: same low-thread bootstrap; lowering T≥4 gate REGRESSES (tested).
- Heavily-mixed streams: single-threaded Huffman tail (window-map parallelization).

## Methodology rules (violated repeatedly today — enforce)
1. ABSOLUTES on the shared box are NOISE. Only interleaved (jitter-immune) or host-frozen relatives count.
2. Every measurement instrument needs a POSITIVE-CONTROL self-test before its output is trusted
   (the broken oracle had none for a multi-session-corrupting bug).
3. on-critical-path ≠ total-CPU. The wall A/B (output-verified) is the only lever verdict; a
   perf-counter correlation is a hypothesis.
4. Isolate ONE variable per benchmark; if a result is physically impossible (drained > no-drain),
   the bench is confounded — discard, don't interpret.
5. Re-verify cell framings in the SAME window before attacking (the "1.56× incompressible" was 13%).
