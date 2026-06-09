# HANDOFF.md independent review (read-only Opus, 2026-06-09)

Cross-checked HANDOFF.md against disproof-ledger.md (DIS-1..28), corpus-reframe-gate.md,
var8-gate-verdict.md, job2-reserve-fix-gate-verdict.md, and source/vendor first-hand.
Overall the document is high-quality and substantially accurate; the DEAD list, the
neurotic topology section, and the Fulcrum/process-discipline sections are correct as
written (spot-checks below). The fixes below are required before it goes to a fresh model.

---

## REQUIRED FIXES

### R1. "DIS-1..29" is wrong — the ledger ends at DIS-28.
Lines 5 and 161 say "DIS-1..29". There is no DIS-29 anywhere in the repo (`grep -rn DIS-29`
= empty; highest ledger header is `## DIS-28`). The corpus-generality entry IS DIS-28, and
its gate file is literally titled "DIS-28 corpus-generality reframe."
**Fix:** change both occurrences to **DIS-1..28**.

### R2. MISSING, important: HEAD d56cb0f5 is UNPUSHED / local-only.
Line 3 says "HEAD `d56cb0f5`. This is the authoritative handoff" with no caveat. Verified:
`git branch -r --contains d56cb0f5` = EMPTY; the only remote tip of `origin/reimplement-isa-l`
is **7bf26096**, which is d56cb0f5's ANCESTOR. Every owner branch/worktree the HANDOFF lists
(`owner/t1-singleshot-route`, `owner/isal-incremental-growth`, etc.) is also local-only and
uncommitted-by-policy. The ledger even records that the VAR_VIII bench had to be BUILT at
7bf26096 "because d56cb0f5 is LOCAL-ONLY/unpushed" (ledger line 299).
**Fix:** add a sentence to §1 or §6: "HEAD d56cb0f5 is LOCAL-ONLY/unpushed — the remote tip of
`origin/reimplement-isa-l` is its ancestor 7bf26096. A fresh clone gets 7bf26096, NOT d56cb0f5;
the authoritative working tree (and all owner/* branches + worktrees) exists only on this
machine. Do not assume a remote checkout reproduces the banked binaries."

### R3. Central finding §2 conflates two distinct engines and overstates "the one real lever."
var8-gate-verdict.md Claim 3 is explicitly **FIX-NEEDED** on exactly this framing. Two errors:
- VAR_VIII's **0.667x** bounds the **clean-tail inner-Huffman kernel** (the native build's clean
  path) — NOT the **u16-marker bootstrap**. The gate: "It is not a CLEAN-TAIL-engine problem —
  the thing VAR_VIII accelerates … But it is partly an engine problem on a *different* engine: the
  window-absent u16 marker-resolution prefix." HANDOFF line 27 lumps "u16-marker / inner-Huffman
  symbol-rate" under one 0.667x bound; the marker-prefix rate was never measured/bounded by VAR_VIII.
- "The one real lever is the ENGINE-W" overstates. A **perfect** engine — gzippy-isal already runs
  REAL ISA-L on clean chunks — **still loses T1 0.899x AND T4 0.906x** (LEV-1, OPEN-1). The gate's
  load-bearing conclusion: even engine-perfect AND placement-perfect loses low-T, so there is an
  **UNSIZED low-T structural residual** = {u16-marker bootstrap structural position + slow compute,
  serial-output floor, chunk-0 bootstrap}, owed a **removal oracle** (OPEN-1, still open). And
  "native ≥0.99-every-T unreachable" is the **leading hypothesis, NOT a removal-proved floor.**
**Fix (suggested §2.2):** "The clean-tail inner-Huffman kernel is asm-bounded at ~0.667x ISA-L
(VAR_VIII; failed its 0.85 bar) — this is the native-build clean-path ceiling. SEPARATELY, even a
perfect engine (gzippy-isal = real ISA-L) loses T1 0.899x / T4 0.906x, so a still-UNSIZED low-T
structural residual (u16-marker bootstrap position+compute, serial-output floor, chunk-0 bootstrap)
remains — owed the OPEN-1 removal oracle. 'native ≥0.99 at every T is unreachable' is the leading
hypothesis, not yet a proven floor. The asm rewrite + bar revisit are the user's gated calls."

### R4. Scorecard "ratio" column is mislabeled (collides with the header definition).
Line 33 defines "ratio = rg_wall/gz_wall, >1 = gz WINS," but the column literally headed **ratio**
holds **compression ratios** (10x / 9.9x / 7.8x / 3.1x / 1.26x), while the wall ratios are the
T1/T8/T16 columns. A fresh model can read "small 10x" as a 10x wall win.
**Fix:** rename that column header to `compr`/`expand` (compression factor) and keep "wall
ratio = rg/gz, >1=gz wins" attached to the T1/T8/T16 columns.

### R5. Scorecard omits T4 — and gzippy-isal FAILS T4 under BAR-1.
BAR-1 is ">= 0.99x at EVERY thread count." The table shows only T1/T8/T16. gzippy-isal loses **T4
~0.90–0.906x** (LEV-1 ocl_cf 0.899x; OPEN-1 measured 0.906x; DIS-16 0.892–0.902x) — a BAR-1-failing
cell that the T1/T8/T16-only table hides, making isal look closer to passing than it is.
**Fix:** add a T4 column (silesia ~0.90 L; model ~0.74-class) or a one-line note: "gzippy-isal also
loses T4 (~0.90x, LEV-1/OPEN-1) — a BAR-1 fail not shown in the table."

### R6. §4(i) "the fix already exists (incremental-growth branch)" overstates the storm fix.
corpus-reframe-gate.md (the authoritative DIS-28 storm gate) §b/§4 says the blessed fix is
**retry-on-None or auto-size EXPAND_FACTOR** and explicitly "No readStream-coalesce port needed" —
a *different, simpler* mechanism than the incremental-growth branch. The incremental-growth branch
(DIS-23) was measured/gated for **FOOTPRINT (dTLB/RSS) on silesia**, where there is NO storm
(isal_chunks=14/0); it has **NOT** been verified to dissolve the storm on nasa/small — that is the
HANDOFF's own in-flight, unverified work. The gate also warns the storm's **parallel-T wall payoff
is UNPROVEN** and must be bounded on a **>=200MB, >8x corpus** before ranking it above the user-gated
work (the HANDOFF notes this in §4 OWED but not in (i)).
**Fix:** soften (i) to: "A candidate mechanism exists (the growable sink on `owner/isal-incremental-
growth`), but the gate's blessed fix is retry-on-None / auto-size EXPAND_FACTOR (no port needed).
The storm fix is NOT yet verified on nasa/small, and its parallel-T payoff is unproven — bound it on
a >=200MB >8x corpus before treating the storm as closed."

---

## OPTIONAL IMPROVEMENTS

- **O1 (line 43):** "The deficit scales with compressed-size -> chunk-count" is stated as fact; the
  gate (a) downgraded it to "PLAUSIBLE, not established — a 2-point trend; keep as hypothesis." Mark
  it as a hypothesis.
- **O2 (VAR_VIII salvage):** the HANDOFF presents VAR_VIII only as a failure ("FAILED its 0.85 bar").
  The gate (Claim 2) says the +14.6%-over-VAR_VI is a **genuine, copy-confound-free, native-only
  engine win** to BENCH-BANK and salvage native-only via a Stage-1 byte-exact + Stage-2 whole-system
  perturbation. Worth a half-line so the next model doesn't discard it outright.
- **O3 (JOB-2 precision):** the gated-PASS is specifically the `writable_tail_reserve` under-sizing
  fix (commit b50c0d23, job2-reserve-fix-gate-verdict.md = KEEP/MERGE). The OTHER JOB-2 piece — the
  `until_exact` EXACT-match coverage relax for stored/fixed (OPEN-3) — is still OPEN/gated, NOT done.
  The HANDOFF's "JOB-2 SYNC_FLUSH reserve fix … gated-PASS" reads as the whole of JOB-2 being closed;
  clarify the coverage relax remains OPEN-3.
- **O4 (minor):** native stub line range — HANDOFF says `:390-400`; the fn is :390-399 (`Ok(false)`
  at 399), DIS-13 cites :390-408. Cosmetic; the load-bearing fact (increment at :386, stub never
  increments) is correct.

---

## SPOT-CHECKS THAT PASSED (no change needed)

- **DEAD list — all attributions correct.** de-frag PHANTOM (DIS-21, vendor re-derivation
  deflate.hpp:805 ring / :1319 appendToWindow / :1376 memcpy — verified the citations match the
  re-derivation, not the misread :926); chunk-count (DIS-25, rg 66 vs gz 34, scales with -P at
  ParallelGzipReader.hpp:294-306 — verified the formula uses `parallelization` at those lines);
  E-core feeding (DIS-27, E-cores 72% busy); marker-fraction (DIS-19, ledger range 31–34.5%);
  out-of-order publish UNFAITHFUL (rg consumer in-order). flip-to-clean: verified first-hand —
  GzipChunk.hpp:520-525 `if (cleanDataCount >= MAX_WINDOW_SIZE) finishDecodeChunkWithInexactOffset
  <IsalInflateWrapper>` — rg DOES flip the clean tail to ISA-L at 32KiB. Accurate.
- **Scorecard cell values** match DIS-28 / corpus-reframe-gate (model 0.89/0.685/0.677; nasa
  0.57/1.04/1.05; ghcn 0.95/1.01/0.96; silesia 0.90/1.02/0.92; small 1.89/1.86/1.74). The
  "wins small/compressible, ties text, loses large-incompressible" summary matches the gate bottom
  line. The "T1 = ParallelSM not single-shot on HEAD" caveat (line 46-47) is correct (DIS-28 caveat
  c; single-shot is the unmerged owner/t1-singleshot-route, 1.197x silesia per DIS-15/DIS-22).
- **Storm root** (reserve under-sizing `compressed_span * EXPAND_FACTOR=8`, NOT EOB-stop/JOB-2):
  CORRECT per corpus-reframe-gate §b — verified EXPAND_FACTOR=8 at gzip_chunk.rs:265, None-return at
  isal_decompress.rs. The "NOT JOB-2 / SYNC_FLUSH" distinction is right.
- **neurotic §6:** topology (i7-13700T, 8P-SMT 0-15 / 8E 16-23 / 24 logical, cpu_core+cpu_atom PMUs),
  cores:16 → `pct set 199 --cores 24` + restore (DIS-26/27 confirm), bench-lock.sh freeze,
  proof-of-binary (isal_chunks>=14, increment at gzip_chunk.rs:386 vs native stub Ok(false)),
  worktree-submodule-empty + `rsync --delete` wipe gotchas — all accurate.
- **§7 Fulcrum / §8 process discipline:** accurate — hypothesis-generator-not-verdict, rule-3
  (slow-slope ≠ speed-up ceiling, removal oracle required), perturbation-is-the-verdict,
  gate-every-number, synchronous-not-background-yield. No issues.

## BOTTOM LINE
Not accurate as-is. Six required fixes (R1 factual off-by-one; R2 missing unpushed-HEAD gotcha;
R3 two-engine conflation + overstated lever per the var8 gate; R4 mislabeled ratio column;
R5 omitted/failing T4 cell; R6 overstated storm fix per the corpus-reframe gate). The DEAD list,
neurotic, and process sections are sound and need no change.
