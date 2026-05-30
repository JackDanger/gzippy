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
| Bootstrap decode-SPEED (distance LUT, unify) | DEAD (bandwidth-bound) | real distance-LUT speedup moved body_rate only 2%; IPC 0.97, 69% LLC-miss |
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

## OPEN / not-yet-exhausted
- Single-member-T2: same low-thread marker-bootstrap ceiling; lowering the T≥4 gate REGRESSES (tested).
- Heavily-mixed streams: the single-threaded Huffman tail (window-map parallelization) — out of
  scope of the stored fix; would need to parallelize the tail decode itself.

## Methodology rules (violated repeatedly today — enforce)
1. ABSOLUTES on the shared box are NOISE. Only interleaved (jitter-immune) or host-frozen relatives count.
2. Every measurement instrument needs a POSITIVE-CONTROL self-test before its output is trusted
   (the broken oracle had none for a multi-session-corrupting bug).
3. on-critical-path ≠ total-CPU. The wall A/B (output-verified) is the only lever verdict; a
   perf-counter correlation is a hypothesis.
4. Isolate ONE variable per benchmark; if a result is physically impossible (drained > no-drain),
   the bench is confounded — discard, don't interpret.
5. Re-verify cell framings in the SAME window before attacking (the "1.56× incompressible" was 13%).
