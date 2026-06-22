# T2 MARKER-FREE — PRE-REGISTERED FALSIFIER

Branch: kernel-converge-A (worktree gzippy-amd-t2t4). Base HEAD: 8d0d24bd.
Primary metric: PEAK RSS (load-immune; macOS aarch64 `/usr/bin/time -l` maximum
resident set size; Linux x86_64 `/usr/bin/time -v` Maximum resident set size;
run-to-run spread ≤ 0.02 MiB per the prior locate). Pre-registered BEFORE any code.

## GATED STARTING POINT (inherited, Gate-2 causal, load-immune)
- gz monorepo-T2 peak RSS = 96.5 MiB vs rapidgzip 59 MiB → ~37 MiB gap.
- Gate-2 (GZIPPY_CHUNK_KIB sweep 256→4096 = 38.5→96.8 MiB) proved peak RSS ∝
  in-flight decoded chunk bytes. monorepo decoded = 50.9 MiB; 96.8 ≈ 2× output.

## HYPOTHESIS (unvalidated) being tested
The window-absent T2 chunks decode into a u16 `data_with_markers` buffer
(2 bytes/symbol). After `apply_window`/`resolve_and_narrow_markers_in_place`,
the narrowed u8 bytes occupy the LOW half of the same 2× allocation, which is
held (per-chunk) until consumer writev + `recycle_decoded_buffers`. The
hypothesis: this unfreed 2× u16 buffer is the dominant ~37 MiB of the gz-vs-rg gap.

COUNTER-EVIDENCE noted at registration (source-read = HYPOTHESIS, not verdict):
vendor `DecodedData::applyWindow` (DecodedData.hpp:374-379) explicitly does NOT
shrink the marker buffer ("leaves half of the chunk space unused... shrink_to_fit
would be expensive" — a `@todo`). So rg also holds a 2× buffer in-flight; the gap
may instead be the NUMBER of simultaneously-live chunks (cache_capacity=max(16,pool)
holds ~all chunks of a small file) rather than the per-chunk 2×. This is why an
empirical RSS perturbation — not the source read — is the verdict.

## ISOLATION MEASUREMENT (the gating prerequisite)
Causal RSS perturbation: env-gated `GZIPPY_FREE_MARKERS=1` frees the u16
`data_with_markers` allocation immediately after markers are resolved (move the
narrowed bytes into the u8 `data` buffer via the tested `merge_resolved_markers_into_data`
+ release the u16 segments), so the 2× buffer is not held for the chunk's life.
Measure monorepo-T2 peak RSS with the flag OFF (baseline) vs ON.
Non-inert proof: a counter (MARKER_FREE_FIRED) > 0 and freed-bytes > 0.

- ISOLATION **CONFIRMED** iff: flipping the flag drops monorepo-T2 peak RSS by
  > ~18.5 MiB (>50% of the ~37 MiB gap), toward rg's ~59 MiB — load-immune.
- ISOLATION **REFUTED** iff: < ~18.5 MiB drop (the 2× u16 buffer is NOT the
  dominant gap source). Then report the actual dominant source (per the Gate-2
  locate: in-flight chunk count × chunk size) and STOP before shipping a fix.

## FIX (only if isolation CONFIRMED)
Make the free faithful + default; byte-exact is non-negotiable. Prefer
shrink/reuse over free-then-realloc if alloc churn appears.
- FIX **CONFIRMED** iff: byte-exact (sha == zcat, silesia+monorepo differential
  multi-chunk-size) AND monorepo-T2 peak RSS drops materially toward rg's ~59 MiB
  (load-immune), with NO throughput regression on T2/T4/T8 (T1/clean path untouched).
- FIX **FALSIFIED**: no RSS drop, OR any byte mismatch, OR a throughput regression
  → revert.

## ARCHES
RSS is arch-portable: replicate on macOS aarch64 (local) AND a linux x86_64 box
(neurotic/solvency — no freeze needed for RSS). The 96.5 baseline is x86 monorepo.
