# SINGLE-SHOT / PER-CHUNK PIPELINE CAPSTONE — Opus disproof verdict (DIS-15)

Role: independent read-only disproof advisor. Method: first-hand source verification
of every load-bearing claim + audit of the harness gating logic. I did NOT re-execute
the frozen-guest bench (no box access from the advisor env); the wall numbers are the
owner's reported guest measurement (bin 9c466f67, N=15, frozen, bench-lock). My verdicts
attack the INFERENCE and the FAIRNESS of the comparison, which is where a removal oracle
goes wrong, not the stopwatch.

Bottom line up front: **the pivotal result is SOUND at T1** — the per-chunk ParallelSM
pipeline is a real, sized (247 ms / ~24% of T1 wall), gzippy-specific, removable cost,
NOT a floor; the single-shot 1.197x win is a fair, byte-verified removal oracle; real
production `-p1` IS ParallelSM-with-no-floor so the 247 ms is a genuine production T1
deficit. **Three honest temperings**: (a) the cheap FIX (route T1→single-shot ISA-L) is
an isal-build-only path-fork that DIVERGES from the "ONE PRODUCTION PATH / pure-Rust SOLE
decode path" goal and does nothing for gzippy-native; the FAITHFUL fix (lean the consumer
toward rg's pipeline) is UNSIZED. (b) ~30 ms of the 247 ms is a one-off pure-Rust fallback
re-decode (a chunking *consequence*, fairly attributed to the pipeline, but not steady
per-chunk handoff). (c) this completes the **T1** attribution only; T4 0.911x is a
separate, still-open parallel-scheduling gap. None of these threatens "pipeline is the
lever, not a floor."

---

## CLAIM 1 — single-shot 1.197x SOUND / not cheating? → **SOUND**

Verified first-hand:
- The oracle calls `isal_decompress::decompress_gzip_stream` (isal_decompress.rs:25-89)
  with `state.crc_flag = isal_raw::IGZIP_GZIP` (line 34). IGZIP_GZIP makes ISA-L parse
  the gzip header AND compute CRC32 (vpclmulqdq) AND validate CRC32 + ISIZE against the
  trailer internally; `isal_inflate` returns non-zero on a checksum/length mismatch and
  the oracle then returns `None`→`Err`. So the single-shot arm pays the SAME correctness
  work production pays (header parse, CRC32, ISIZE/trailer verify). It does NOT skip CRC,
  window, or trailer verification.
- Output goes to the SAME writer/sink the harness passes; harness writes to a /dev/shm
  REGULAR file (asserted not-symlink/not-fifo, `_perchunk_guest.sh:149`), same-sink for
  gz-prod, gz-singleshot, and rg-file.
- Byte-exactness is GATED, not merely asserted once: every interleaved iteration
  sha256sums the sink and compares to `REF_SHA = gzip -dc(corpus)` (the true decoded
  bytes); any divergence on a byte-exact contender VOIDs the whole run
  (`_perchunk_guest.sh:170-177`). gz-singleshot is registered with SHACK=1 (sha-checked).
  sha=028bd002… on every arm ⇒ the oracle decodes byte-identical output.
- Direction of any residual unfairness is CONSERVATIVE (favors rg, not single-shot):
  single-shot decodes into a reused 1 MiB buffer then `write_all` (an extra ISA-L→buf→
  kernel memcpy) whereas production writev's; if anything single-shot does one MORE copy.
  Its small reused buffer gives it a genuine cache/working-set advantage, but that is an
  intrinsic property of the no-pipeline architecture (exactly what the oracle is meant to
  expose), not a measurement artifact — DIS-14 already showed page-faults track written
  bytes, not buffer capacity.

No way found for the single-shot arm to cheat. 1.197x is a fair, byte-verified number.

## CLAIM 2 — real `-p1` is ParallelSM-no-floor (247 ms is a real production deficit)? → **SOUND**

Verified first-hand in `src/decompress/mod.rs:147-197` (`classify_gzip`):
- Single-member, under `#[cfg(parallel_sm)]`, returns `DecodePath::ParallelSM` with **no
  thread/size floor** — `num_threads` is consulted ONLY in the multi-member branch
  (:152), never in the single-member branch. The sole single-member exception is
  `StoredParallel` for stored-DOMINATED streams (`parallel_sm_unprofitable && first_block_
  is_stored`, :182-185); silesia is highly compressible ⇒ not stored-dominated ⇒
  `ParallelSM`. So real production `-p1` on silesia routes to ParallelSM. The comment at
  :170-188 is explicit and matches the code.
- `GZIPPY_FORCE_PARALLEL_SM` is a **dead no-op**: `grep` finds ZERO consumers in `src/`
  at HEAD and in the worktree src (only bench scripts and a `parallel_sm` cfg comment in
  CLAUDE.md still mention it). It cannot change the worker count or routing. Therefore
  gz-prod "T1" (`-p1` + FORCE=1) is byte-identical in behavior to real `-p1`: ParallelSM
  with `num_threads=1` (one worker). `-p1` sets `args.processes=1` (cli.rs:286-287) which
  flows as `num_threads`.

⇒ FORCE_PARALLEL_SM cannot differ from real `-p1` for silesia. The 247 ms IS a real
production T1 deficit, not a FORCE artifact. The caveat the owner flagged in the ledger
("whether real `-p1` forces ParallelSM is a routing question to confirm") is hereby
RESOLVED in favor of the result.

## CLAIM 3 — is 247 ms the pipeline lifecycle, or does the 1 fallback chunk dominate? → **SOUND with refinement (fallback ≈ 30 ms, ~13%, does NOT dominate)**

- `isal_fallbacks=1` at T1: one of 16 chunks (~13.25 MB output, silesia 211,968,000 B /
  16) decodes via the pure-Rust marker engine instead of ISA-L. Sizing it: ISA-L here ran
  at 277 MB/s ⇒ ~48 ms for that chunk; pure-Rust best (~164 MB/s, VAR_VIII) ⇒ ~81 ms ⇒
  **~30–33 ms EXTRA**, i.e. ~13% of the 247 ms. Real, acknowledged in the orchestrator
  note, but NOT dominant; ~210 ms remains genuine pipeline/serialization.
- The fallback is fairly attributed to "the pipeline": single-shot has no chunking ⇒ no
  boundary-misalignment ⇒ no fallback, so removing the chunking removes it. Mechanistically
  it is a chunking-INDUCED engine penalty, not handoff/CRC overhead — the verdict's prose
  should say "247 ms = per-chunk lifecycle + T1 serialization + a ~30 ms one-off chunking
  fallback," not fold all 247 ms into "lifecycle."
- Independent corroboration that the pipeline term is large irrespective of the fallback:
  the structural-residual run (orchestrator §STRUCTURAL-RESIDUAL, bin 2d317027) measured
  the gzippy-SPECIFIC excess at T1 ENGINE-MATCHED and OUTPUT-REMOVED on BOTH sides
  (gz-skipwritev 932 ms vs rg-null 808 ms) = **124 ms / 0.867x** — a second, methodologically
  independent sizing of "gzippy's pipeline is heavier than rg's." The two numbers answer
  different questions (247 ms = gzippy-pipeline ABSOLUTE vs no-pipeline; 124 ms = gzippy-
  pipeline excess vs rg's pipeline) and are mutually consistent.

## CLAIM 4 — lever + cheap fix right? faithful? closeable? → **FIX-NEEDED (lever SOUND; realization caveated)**

- **The LEVER is SOUND, real, sized.** Per-chunk pipeline = 247 ms absolute / 124 ms vs
  rg; at T1 markers are ZERO (flip_to_clean=0, finished_no_flip=0) so the T1 gap is purely
  pipeline (+ the one fallback), with no marker-bootstrap confound. Two existence proofs
  (rg's leaner consumer; gzippy's own single-shot) show it is not irreducible.
- **The cheap FIX (route T1→single-shot ISA-L) is MEASURED-to-win but goal-divergent.**
  It re-introduces the single-shot ISA-L path the campaign deleted in 5e563dc, and creates
  a path FORK (T1→single-shot, T>1→ParallelSM) against Rule 1 "ONE PRODUCTION PATH." More
  importantly it is **isal-build-only**: it does NOTHING for gzippy-native, which is the
  build the architectural goal (BAR-3: pure-Rust SOLE decode path, C-FFI off the decode
  graph) actually cares about. So as a *production* change it optimizes the build that is
  already off-charter (gzippy-isal keeps ISA-L FFI by design, DIS-12) while leaving the
  on-charter build untouched. Sensible ONLY if the user accepts an isal-only T1 fast-path
  as a pragmatic cell-win; it is NOT a step toward the faithful-port done-criterion.
- **The FAITHFUL fix (lean the consumer toward rg's per-chunk pipeline) is UNSIZED.** rg
  reaches 0.916 s at T1 WITH a chunked pipeline; gzippy's single-shot (no pipeline) is
  0.766 s. So gzippy's ISA-L engine raw rate is FASTER than rg's whole wall — but rg's OWN
  T1 pipeline overhead is UNMEASURED. Back-of-envelope: if rg's engine-raw ≈ gzippy's
  (~0.76 s) then rg's pipeline overhead ≈ 0.156 s, and a gzippy pipeline leaned to that
  same overhead would land 0.766+0.156 ≈ 0.92 s ≈ rg-parity ≈ 0.90–0.99x — i.e. a faithful
  lean might only TIE, not beat, and might still miss BAR-1's 0.99 at T1. The owed
  measurement to size the faithful ceiling is **rg single-shot vs rg chunked at T1** (rg's
  own pipeline overhead), which has not been run.
- **The T4 0.911x is a SEPARATE blocker, not addressed here.** Single-shot at T4 is
  single-threaded (0.651x) — useless; the pipeline is net-POSITIVE +216 ms at T4 (it buys
  the parallelism). So T4 0.911x is a parallel-SCHEDULING gap (entangled with the shared
  marker-bootstrap per OPEN-1), owed its own oracle. Do not let the T1 result imply T4.

## CLAIM 5 — completes the isal low-T attribution? → **PARTIAL (T1 complete; T4 still open)**

- At **T1** the attribution is now CLEAN and COMPLETE: markers=0 (no bootstrap), output is
  a shared floor (structural-residual §b: gz 86 ms vs rg 85 ms), D1/glue refuted (DIS-14),
  so the ENTIRE T1 gap is the per-chunk pipeline (+ one fallback) — sized here and shown
  recoverable (single-shot beats rg). "isal low-T is a proved FLOOR / the in-process ISA-L
  call" (DIS-15 null branch) is correctly REFUTED: single-shot uses the SAME igzip kernel
  and is the fastest arm.
- At **T4** it does NOT complete the picture: T4 carries the (shared) marker-bootstrap +
  the net-positive pipeline + a scheduling residual; single-shot can't probe it. So
  "COMPLETES the isal low-T attribution" is accurate for T1 and OVERSTATED if read as all
  low-T. Recommend the ledger phrase it "completes the T1 attribution; T4 scheduling gap
  remains OPEN."

---

## PER-CLAIM VERDICTS
1. single-shot 1.197x fair / not cheating: **SOUND** (crc_flag=IGZIP_GZIP ⇒ CRC+ISIZE
   verified; sha-gated every iteration vs gzip-dc reference; same-sink; copy-asymmetry
   favors rg).
2. real `-p1` = ParallelSM-no-floor, 247 ms is real production: **SOUND** (classify_gzip
   no floor; FORCE_PARALLEL_SM dead no-op — caveat resolved).
3. 247 ms = pipeline, not the fallback: **SOUND w/ refinement** (~30 ms / ~13% is the
   one-off fallback, fairly attributed to chunking; ~210 ms genuine pipeline; corroborated
   by the independent 124 ms structural-residual sizing).
4. lever + cheap fix: **FIX-NEEDED** (lever real+sized; T1→single-shot routing is an
   isal-only path-fork divergent from the pure-Rust/ONE-PATH goal; faithful lean is
   UNSIZED; T4 separate).
5. completes isal low-T attribution: **PARTIAL** (T1 complete; T4 open).

## IS THE PER-CHUNK PIPELINE A REAL, SIZED, CLOSEABLE LEVER?
YES — real (causally removed by single-shot), sized (247 ms abs / 124 ms vs rg at T1), and
closeable at T1 (two existence proofs). NOT a floor.

## IS THE T1 SINGLE-SHOT WIN REAL?
YES — byte-verified, tight spread (0.6–1.0%), real-production routing, conservative copy
asymmetry. 1.197x stands.

## HONEST NEXT MOVE
1. State the result precisely: "At T1 the gzippy-ISAL low-T gap is 100% the per-chunk
   ParallelSM pipeline (sized 247 ms abs / 124 ms vs rg), removable, NOT a floor; gzippy's
   ISA-L engine raw rate already beats rg." That is bankable.
2. Before banking a FIX, decide the fork explicitly with the user: (a) pragmatic isal-only
   T1→single-shot routing (measured win, but a path-fork off the pure-Rust/ONE-PATH goal,
   helps neither native nor the faithful-port criterion); vs (b) the faithful "lean the
   per-chunk consumer toward rg's pipeline" port — which FIRST needs its ceiling sized by
   the owed **rg single-shot vs rg chunked T1** measurement, because a faithful lean may
   only TIE rg (~0.90–0.99x), not beat it.
3. T4 0.911x is the live BAR-1 blocker that single-shot cannot touch — owe it its own
   parallel-scheduling oracle (entangled with OPEN-1's shared marker-bootstrap). Do not
   let the T1 capstone imply T4.
4. Keep gzippy-native scoped out (separate 0.667x engine floor, VAR_VIII).
