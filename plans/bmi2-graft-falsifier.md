# BMI2 / hot-technique graft falsifier (PRE-REGISTERED 2026-06-06, leader)

Pre-registered BEFORE any optimization work, per CLAUDE.md rule 3 + the process
retrospective. The verdict on each technique is the **causal perturbation /
measured locked-harness delta** registered here, NOT post-hoc attribution.

## HEAD / invariants
- HEAD: `8cfad3a` (flip-in-place one-engine fold; advisor-CONFIRMED faithful +
  byte-exact). branch `reimplement-isa-l`.
- BYTE-EXACT ABSOLUTE: every variant must emit silesia sha256
  `028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f`
  via `path=ParallelSM` (DUAL-SHA: gzippy-native arm64 + gzippy-isal x86_64/Rosetta).
- Numbers ONLY from the locked harness (`scripts/bench/run_locked_fulcrum.sh`,
  neurotic, T8 silesia, interleaved best-of-N≥7, sha-verified). Local arm64 is
  for byte-exact iteration + the in-process byte-accounting (RSS) instrument only.
- Cache mandate held throughout: ONE shared decode-table copy across threads
  (FIXED_TABLES OnceLock already shared; dynamic tables are inherently per-block,
  must NOT be duplicated per thread); no large per-thread buffers; RSS ~flat as T.
- finish_decode_chunk_impl preserved (gzippy-isal Design-A insertion point).

## CEILING (the existence proofs — what room exists)
- ISA-L (the floor target): d_c ~44ms (compute), d_w ~67ms (with window apply).
- rapidgzip wall ~461ms (the parity target the campaign must TIE-or-beat).
- The graft can only help up to the next binding component. **Slow-down slope ≠
  speed-up ceiling** (CLAUDE.md PROCESS #3): to bound a technique's speed-up
  ceiling we REMOVE/zero the region (oracle) and measure — never extrapolate a
  slow-injection slope through an unlocated knee.

### Ceiling-bounding step (DONE FIRST, before committing to any graft)
The decode hot loop (`marker_inflate.rs::read_internal_compressed_canonical_
specialized` → `run_multi_cached_loop`) must be confirmed ON the critical path
AND its share bounded BEFORE grafting. Two instruments:
1. In-process byte accounting (`GZIPPY_MEM_STATS=1`, mem_stats.rs) — confirms the
   shared-table / per-thread-working-set state the graft must NOT regress.
   Validate via the positive control (`GZIPPY_MEM_BALLAST_MIB=N` recovers slope ~N)
   BEFORE trusting any reading.
2. Locked harness perturbation: the bootstrap/decode slow-injection
   (`GZIPPY_SLOW_*`) already on record moved the wall ~proportionally
   (survived frequency-neutral disproof) ⇒ decode is on the critical path.
   To BOUND the speed-up ceiling, an oracle that ZEROES the per-symbol decode
   cost (or substitutes a no-op LUT lookup) sets the max wall reachable by a
   faster inner loop. If that oracle wall ≈ current wall ⇒ the inner loop is NOT
   the binding component and the graft is wall-neutral by construction (record as
   such; do NOT proceed to graft on extrapolation).

## PRIOR STATE (read before claiming a technique is "new")
The production dynamic-Huffman path ALREADY implements libdeflate hot techniques
(`run_multi_cached_loop`, marker_inflate.rs:1630+):
- **Multi-symbol LUT**: PRESENT — 2-/3-literal speculative chain
  (marker_inflate.rs:1782-1841), speculative next-entry carry-forward.
- **Lean refill**: PRESENT — branchless `refill_fast!` with bounds elided under
  the FASTLOOP guard (marker_inflate.rs:1715-1731), `REFILL_THRESHOLD=48`.
- **BMI2 PEXT/BZHI**: OFF (Generic profile `HAS_BMI2=false`, unified.rs:122; no
  BMI2 in the marker hot loop). **THE leading candidate** — the one genuinely
  un-grafted technique.
So techniques (b) multi-symbol LUT and (c) lean refill are RE-VALIDATION /
sharpening passes (re-attempt the ca52389-class SIMD multi-literal with fresh
measurement on the post-fold loop — KNOWN HAZARD, measure don't assume), while
(a) BMI2 is the genuinely-new graft.

## PER-TECHNIQUE FALSIFIERS

### (a) BMI2 PEXT/BZHI runtime dispatch  [PRIMARY]
- HYPOTHESIS: BZHI replaces the `peek & mask` + shift in extra-bits extraction
  and the table-index mask; PEXT collapses the reversed-bit table index. Expected
  to shave per-symbol ALU on the dynamic-Huffman FASTLOOP.
- MECHANISM vs rapidgzip (CLAUDE.md rule 7a): rapidgzip/ISA-L use BMI2 in the
  inner extract; gzippy currently does not — so a structural reason exists for a
  gap IF the inner loop binds the wall.
- EXPECTED DELTA: a measurable wall reduction on the x86_64 locked harness at T8,
  bounded above by the decode-zero oracle ceiling. Must be `Δ > inter-run spread`.
- MUST: runtime dispatch (`is_x86_feature_detected!("bmi2")`), NOT a target-feature
  build — the shipped binary must run on non-BMI2 x86. SHARED single table copy
  (no per-thread table dup). aarch64 path unchanged (NEON/scalar).
- TIE/REGRESSION JUDGMENT: `Δ ≤ spread` ⇒ TIE on this run. Per CLAUDE.md rule 7,
  a TIE alone does NOT reject the direction — but a TIE is KEPT only if byte-exact
  AND it does not regress RSS/working-set. A measured wall REGRESSION (Δ negative
  beyond spread) OR an RSS/per-thread-working-set increase ⇒ REVERT.

### (b) multi-symbol LUT  [RE-VALIDATION — largely present]
- Already 3-literal chain. Falsifier: widen to a 4-literal path OR a packed
  2-symbol LUT entry. HAZARD: ca52389 SIMD multi-literal regressed on the
  PRE-PRELOAD/PRE-BMI2 loop — measure on the post-fold loop, do not assume.
- EXPECTED: marginal; literal-heavy runs benefit, branch-misprediction on the
  chain cap may eat it. KEEP only on measured win or byte-exact TIE w/o RSS regress.

### (c) lean refill  [RE-VALIDATION — present]
- Already branchless bounds-elided. Falsifier: a wider 256-bit shift-register
  refill (only meaningful paired with BMI2 X86_64Bmi2 BITBUF_BITS=128/256).
  Bound by the same decode-zero oracle. Sequence AFTER (a) since it's coupled.

## SEQUENCE
1. Bound the ceiling (byte-accounting positive-control validate → decode-zero
   oracle on locked harness). If inner loop is wall-neutral ⇒ STOP, record.
2. (a) BMI2 — graft, DUAL-SHA, locked-harness measure. Keep/revert per above.
3. (b) then (c) only if (a) shows the inner loop has headroom.
Each step: ONE cargo at a time via scripts/cargo-lock.sh; outputs captured to
plans/orchestrator-status.md (the board).

## VERDICT (2026-06-06, ceiling-bound run on guest 199)
- (a) BMI2 BZHI: **REJECTED with mechanism (TIE)**. The production build already
  compiles BZHI (target-cpu=native ⇒ target_feature=bmi2). A/B ON vs OFF on the
  locked-harness corpus (silesia-large, T8, best-of-9, byte-exact both arms):
  median 0.6485 vs 0.6484 (-0.02%), Δ << within-arm spread (63-108ms). Even
  FORCING BZHI off is a wall TIE — the per-packet extract is invisible against a
  memory-bound ~600ms wall. Runtime dispatch only helps a portable binary (not the
  perf target). See plans/orchestrator-status.md "BMI2 CEILING-BOUND A/B — DONE".
- (b)/(c): already present in run_multi_cached_loop AND are per-symbol-ALU
  techniques the same memory-bound argument covers — not re-attempted for the wall
  (no mechanism by which they'd move a memory-bound wall the BZHI A/B just showed
  is ALU-insensitive). The lever is the cache mandate (MPKI), not the ALU graft.
- Repro: scripts/bench/bmi2_ceiling_ab.sh (BRANCH/REPO/CORPUS/T/N/MASK env).

## ADVISOR
Consequential design choices (esp. the runtime-dispatch shape, any "lever worked"
claim) advisor-corroborated before finalizing (supervisor runs advisors the
leader can't — flag on the board). FLAGGED: the (a) rejection-with-mechanism +
the redirect to the cache/MPKI surface need advisor corroboration (board entry).
