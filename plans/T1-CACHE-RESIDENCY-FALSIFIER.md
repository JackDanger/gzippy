# T1-CACHE-RESIDENCY — PRE-REGISTERED FALSIFIER (frozen before building)

Branch `t1-cache-residency` off `kernel-converge-A` (@7e6f37cb). This doc is
written and committed BEFORE any code/measurement so the thresholds cannot move.

## Gated starting point (cross-arch LAW, Intel+AMD; see project_x86_t1_monolith / project_x86_prod_path_locate)

- The x86 T1 gap to igzip is the per-chunk DRIVER cost, NOT the kernel
  (native = parity-or-faster than ISA-L under the same driver, both arches).
- The T1-MONOLITH (one big reused ISIZE buffer) was FALSIFIED: materializing the
  full output fault-storms (silesia 68,950 minor-faults vs igzip 17,092 vs
  thin-T1 23,040). The lever is therefore CACHE RESIDENCY / WORKING-SET: igzip
  wins T1 by STREAMING through a small cache-warm REUSED buffer, never
  materializing the whole output. gzippy's thin-T1 already approximates this and
  BEATS rapidgzip −P1 at T1; the residual is vs igzip.
- Current gated ratios (thin-T1 prod / igzip-ISA-L bar, streaming_thin, best-of-15):
  Intel silesia 1.213 / nasa 1.260 / monorepo 1.353 / squishy 1.178;
  AMD   silesia 1.201 / nasa 1.248 / monorepo 1.392 / squishy 1.157.

## Levers under test (apply to the CHUNKED STREAMING thin-T1 path, KEEP its small reused buffer)

- (a) 2 MiB T1 chunk (chunk-size U-curve, T1 default 1 MiB suboptimal; T1-GATED —
  chunk size is T>1-load-bearing). Oracle knob: `GZIPPY_CHUNK_KIB=2048`.
- (b) output-buffer RECYCLING POOL — the thin-T1 serial loop currently lets each
  chunk's `SegmentedU8` buffer drop + re-allocate fresh per chunk (no recycling
  unless the global manual-pool flag is set). Reuse one warm, resident buffer
  across chunks. Oracle knobs: `GZIPPY_MANUAL_BUFFER_POOL=1` (recycle, keep
  ratio-reserve) and `GZIPPY_RESIDENT_OUTPUT_POOL=1` (recycle + pin reserve to a
  single fixed 64 MiB size).
- (c) shed per-chunk WINDOW CLONE + per-block BOUNDARY RECORD at T==1 (oracle-
  classified PURE-T1 cost; the monolith cycle built a byte-transparent
  `record_boundaries=false` shed — reuse on the chunked path).

## MECHANISM CHECK (Gate-2; required — a wall move must be EXPLAINED by the mechanism)

AMD bare-metal `perf stat -e minor-faults,page-faults,instructions,cycles`. A
wall drop is only banked as the cache-residency mechanism if it is accompanied by
a minor-faults DROP toward igzip's level. A wall move with faults moving the
wrong way is NOT this mechanism — report and drop that lever.

## PRE-REGISTERED VERDICTS (do not move goalposts)

- CONFIRMED iff: gzippy-NATIVE prod/igzip drops from the current ~1.16–1.43 to
  **<= 1.10 on all 8 cells (Intel+AMD)**, byte-exact, AND T4/T8 do NOT regress vs
  the current build and vs rapidgzip, AND the wall drop is accompanied by a
  minor-faults drop toward igzip's level (mechanism confirmed).
- PARTIAL: a real gated drop that does not reach 1.10 — report the new gated
  ratios + remaining residual; do NOT narrate as "closed"; identify the next
  lever (likely per-symbol kernel codegen, re-measured below).
- FALSIFIED for a given lever iff: it does not move the wall beyond spread, OR it
  moves faults the wrong way — report and drop that lever, keep the ones that pay.
- ALSO re-measure the native-vs-igzip KERNEL ceiling AFTER the driver is
  tightened (the kernel may RE-EMERGE as dominant once the driver is shed — if it
  does, that is the next real lever per no-phases; report it).

## Measurement discipline (enforced)

- Gate-0: sha==zcat all arms; A/A ≪ Δ; /dev/null both arms; comparator self-test ~1.0.
- Gate-1: interleaved best-of-N>=15 (taskset cpu4, unfreezable LXC on Intel) /
  best-of-N>=7 frozen on AMD; report Δ vs spread; label TIEs.
- Gate-3: Intel (neurotic/trainer 10.30.0.199) AND AMD (solvency 10.0.2.240);
  T1 AND T4/T8.
- Gate-4: GZIPPY_DEBUG routing (thin-T1 fires) + feature fingerprint
  (gzippy-native = no isal_clean_tail; bar = ISA-L).

## Method (measure the oracle knobs FIRST, then bake the winner)

The levers (a)/(b) are already reachable as byte-transparent env oracles on the
current code (`GZIPPY_CHUNK_KIB`, `GZIPPY_MANUAL_BUFFER_POOL`,
`GZIPPY_RESIDENT_OUTPUT_POOL`). Measure those arms on both boxes to find which
mechanism actually pays BEFORE writing production code (no code from an
unconfirmed model). Then implement the winning mechanism as the thin-T1
production DEFAULT — LOCAL to the T1 serial path, never flipping a global default
that would touch the T>1 parallel pipeline — byte-exact, with a correctness
differential (flate2 + libdeflate, multiple chunk sizes, multi-member resume) in
the same commit. Re-measure the production default vs igzip + T4/T8 no-regress.
