# T≥2 LOCATE PLAN — the confound-proof measurement path

**Author:** path-plotting advisor (Opus). **Date:** 2026-06-20. **Scope:** the T≥2
pure-Rust-native deficit vs rapidgzip. **Status of this doc:** a measurement DESIGN, not
a finding. Every number it asks for must pass Gate-0/1/2 (CLAUDE.md PROTOCOL) before it
counts.

---

## 0. STOP — READ THIS BEFORE EXECUTING. The brief is anchored on STALE facts.

The brief asks for a path "from `T≥2 loses ~0.77` to `the gap is LOCATED in stage X`."
That framing is the **2026-06-16 two-binary-matrix state**. The on-disk memory record
contains a stack of **NEWER gated results (2026-06-17/18)** that already walked most of
this path. Re-plotting a locate-from-0.77 path without first re-checking would be
**BIAS #2 (disproof amnesia across restarts)** — the single costliest bias in
`plans/BIAS-FORENSICS.md`. The newer record (project_two_binary_matrix_2026_06_16.md,
late entries; project_settled_rg_gap_2026_06_15.md) says, in order:

1. The T≥2 gap was localized to **gz's window-absent (markered) inflate doing ~28% more
   decode CPU than rg on the SAME chunks** (DECODE-WAIT dominates; consumer/overlap/
   apply_window/serial all ruled OFF the critical path by removal-oracle + decompose).
2. The root was the **`SegmentedU8` contiguous-Vec data plane** doing per-chunk
   realloc+memmove+clone at flip-to-clean, vs rg's `vector<VectorView>` O(1)-prepend.
3. Porting rg's view-list data plane (branch `reimplement-isa-l` @ `300e772b`) hit
   **whole-program INSTRUCTION PARITY at T4** on Intel (gz/rg instr 1.268→1.013 silesia,
   1.197→0.975 squishy — gz *beats* rg on squishy).
4. The residual **silesia-T4 wall ~1.15** was located as **tail load-imbalance** (few
   large late chunks, in-order consumer wait), and the prefetch-ordering fix was
   **FALSIFIED at the wall** → declared "at the structural in-order-tail floor."
5. **Campaign status 2026-06-17: parallel-parity essentially MET on Intel**, blocked only
   on **AMD/solvency for LAW**.

**Therefore the honest job is NOT "locate the gap from scratch."** It is:
**(A) re-establish ground truth at a pinned current binary** (because HEAD has moved and
the win is on a *different branch*), then **(B) branch**: either confirm-and-pay-the-LAW-
debt, or — only if the deficit reappears — run the locate tree. This respects the brief's
real intent (a confound-proof deterministic path to a verdict) while refusing to re-walk a
path the record says is already walked.

### The branch hazard that makes Step 0 non-optional
- Current branch is **`kernel-converge-A` @ `c91aa125`** (T1 inner-kernel "night31/32"
  work). The T≥2 view-list convergence landed on **`reimplement-isa-l` @ `300e772b`**.
  **If the subject binary is built from a branch that does NOT contain `300e772b`, the
  ~0.77 deficit will REAPPEAR — and that is a missing-merge problem, not a new lever.**
  The first thing Step 0 must nail down is *which commit the binary contains*, asserted by
  sha, not by branch name.

---

## 1. THE PATH (decision tree)

```
S0  Gate-0 the rig + pin the binary       ─┐
S1  Re-establish the T≥2 matrix at HEAD    ─┴─> the FORK
         │
         ├─ FORK-A: T≥2 ~parity (view-list present) ──> the gap is LOCATED+CLOSED on Intel
         │        A1 re-confirm silesia-T4 "at-floor" (effcores/tail)   [the one residual]
         │        A2 pay the AMD/Zen2 LAW debt  [converts the stack to LAW]
         │        A3 (optional) zero-copy-resolved-markers increment  [marginal]
         │
         └─ FORK-B: deficit persists (binary lacks 300e772b, OR a regression) ──> LOCATE
                  L0  is the binary missing the win? (sha-diff vs 300e772b) → if yes, MERGE, re-run S1
                  L1  instruction-vs-stall split (cyc/B + instr/B, freq-invariant)
                       ├─ instruction-bound → L2a  seed-windows removal-oracle (marker machinery vs clean)
                       └─ stall/serial-bound → L2b consumer-decompose + effcores (decode-wait vs tail vs serial)
                  L3  causal confirm (frequency-neutral slow-inject into the located stage) + removal-oracle ceiling
```

Each node below: **instrument · what it perturbs · Gate-0 self-validation · Gate-1/2
verdict criterion · branch on outcome.**

---

### S0 — Gate-0 the rig + pin the binary (BLOCKING; no number exists until this passes)

- **Instrument:** `scripts/bench/decide.sh` + `_decide_guest.sh` (the matrix runner) on
  guest `10.30.0.199` (double-hop `ssh -J neurotic root@10.30.0.199`, see `guest.env` —
  but **treat `guest.env` paths as STALE**: it pins `b22e1b14`/`/dev/shm/gz-b22-target`,
  which predates everything; re-pin to a fresh build of the subject commit).
- **What it perturbs:** nothing yet — this is the self-test arm.
- **Gate-0 it must pass (ALL, loud):**
  - (a) **rg present + self-tests to 1.0±spread.** Run rg-vs-rg interleaved best-of-N on
    one cell; if ratio ≠ 1.0±spread or the binary is absent, STOP (the "no rg on solvency"
    trap). Confirm it is the **native ELF v0.16.0**, not the Python wheel.
  - (b) **Both arms same `/dev/null` sink** (SINK LAW; tmpfs "negligible" must be *measured*
    on one cell, not asserted — the late memory flags this).
  - (c) **`GZIPPY_DEBUG=1` → `path=ParallelSM`** on the subject, with
    `GZIPPY_FORCE_PARALLEL_SM=1`; sha-verified output == gzip reference.
  - (d) **Both binaries built identically** (same `RUSTFLAGS="-C target-cpu=native"`,
    fresh, NOT a pinned `/dev/shm` artifact — the banked **binary-drift** hazard: a stale
    pin retired ~5% more instr and invalidated an A/B).
  - (e) **Record the subject sha** and `git merge-base --is-ancestor 300e772b HEAD` →
    write the answer into the run log (this is the FORK-B/L0 discriminator).
- **Verdict:** the rig is trustworthy iff all of (a)–(e) pass. Else fix the rig first.

### S1 — Re-establish the T≥2 matrix at the pinned binary

- **Instrument:** `decide.sh` matrix, **N≥13 interleaved**, masks = T distinct P-cores at
  T1/T2/T4/T7, corpora silesia + squishy + nasa + monorepo, `/dev/null` both arms.
- **What it perturbs:** nothing — it measures the current gz/rg wall ratio.
- **Gate-1 verdict:** per cell, Δ vs inter-run spread; report `rg_wall/gz_wall` with spread.
  `<1−spread` = real loss; `≥1−spread` = TIE-or-win.
- **Branch:**
  - **All T≥2 cells ≥ ~0.97 except possibly silesia-T4 → FORK-A.** The view-list win is in
    the binary; the gap is located+closed on Intel. Do **A1/A2**, do NOT run the locate tree.
  - **Any T≥2 cell still ~0.77–0.86 (the old deficit) → FORK-B.** Go to **L0** first.

---

## FORK-A — confirm + pay the LAW debt (the most likely real state)

### A1 — re-confirm the silesia-T4 "at-floor" verdict (don't trust the prose)
- **Instrument:** the effcores/tail decompose — `effcores = total_cyc/wall` (avg busy
  cores) + the consumer decompose (`scripts/consumer_decompose.py` lineage; if absent on
  the pinned tree, rebuild from `GZIPPY_TIMELINE` trace + `scripts/parallel_sm_log_summary.py`).
  Self-validate: span conservation (busy+idle==span; consumer.iter == chunks+stale-skips).
- **What it perturbs:** nothing (attribution-tier) → this only *re-confirms* the located
  cause; the *causal* arm is the already-run, already-FALSIFIED prefetch-depth A/B
  (memory: silesia-T4 wall flat at prefetch-depth 0→3). **Do not re-run that A/B as if
  new** (BIAS #2). A1's only job: confirm effcores still ~0.92 and tail-imbalance still the
  shape, i.e. nothing regressed. If effcores is now ~1.0, silesia-T4 closed on its own —
  bank and move on.
- **Verdict:** matches the banked at-floor picture ⇒ silesia-T4 is a known, gated-falsified
  residual; stop poking it. Diverges ⇒ a regression appeared → FORK-B/L1.

### A2 — pay the AMD/Zen2 LAW debt (THE blocking gate; defer only if box offline)
- Every T≥2 win is **Intel-only NOT-YET-LAW**. One AMD/solvency (`root@192.168.7.222`,
  Zen2) interleaved matrix run at the same pinned shas converts the whole stack to LAW.
- **AMD-specific confound to pre-register:** the marker/window-absent path is the most
  likely **PEXT/PDEP** user, which is **microcoded on Zen2** → a win could shrink/invert.
  The late memory claims the data-plane fix is memmove+CRC/pclmulqdq with no PEXT/PDEP
  (low inversion risk) — *verify by grepping the hot path for `_pext_u64`/`_pdep_u64`
  before trusting that*, then measure.
- **This is the single highest-VoI action in FORK-A.** If solvency is offline, this is the
  one DEFER — flag it, don't block A1.

### A3 — (optional, marginal) zero-copy resolved markers
- The data plane has one more rg-faithful increment: eliminate the remaining narrow copy
  via rg's `reusedDataBuffers` swap (`DecodedData.hpp`). **Prize ≡ measured whole-program
  Δ**, never the gap-to-rg. Low priority; only after A2.

---

## FORK-B — the LOCATE tree (run ONLY if S1 shows the deficit persists)

### L0 — is the binary simply missing the win? (cheapest possible explanation first)
- `git merge-base --is-ancestor 300e772b <subject-sha>`. If **false**, the subject lacks
  the view-list convergence → the deficit is EXPECTED, not a new lever. **Action: merge/
  rebase `300e772b` into the subject branch, rebuild, re-run S1.** Do not start locating a
  gap that a known, pushed win already closes.
- If **true** (win present) and the deficit still persists → a genuine regression or an
  un-replicated prior result → continue to L1.

### L1 — instruction-vs-stall split (the first real fork inside locate)
- **Instrument:** `perf stat` on the production decode (taskset P-cores), gz vs rg, same
  cell: **cyc/byte AND instr/byte AND IPC**. cyc/byte is **frequency-invariant** (works on
  the loaded LXC; the box is base-pinned per the freeze-validity result). Scripts:
  `_wa_perf_guest.sh` / `_perf_attr_guest.sh` lineage.
- **Gate-0:** counters fired (cyc>0, instr>0); `path=ParallelSM`; sha==gzip; both arms
  fresh-built identically (binary-drift trap).
- **Verdict (Gate-1, Δ vs spread):**
  - **gz instr/byte ≫ rg AND wall≈instr ratio (IPC equal) → INSTRUCTION-BOUND** → L2a. The
    lever is work-volume in the window-absent inflate (the historically-correct branch).
  - **gz instr/byte ≈ rg but cyc/byte/wall ≫ rg (IPC lower, or wall ≫ cyc i.e. effcores
    low) → STALL/SERIALIZATION-BOUND** → L2b. The lever is overlap/tail/scheduling.
- **Confound red-team:** instr/byte is an *attribution*, not a verdict — it tells you which
  perturbation to run next, never "the lever is X." The verdict is L2/L3.

### L2a — INSTRUCTION-BOUND branch: the seed-windows removal-oracle
- **Instrument:** the **REPAIRED** `seed_windows` oracle (`instruments/seed_windows.rs`,
  knobs `GZIPPY_SEED_WINDOWS_CAPTURE=<file>` then `GZIPPY_SEED_WINDOWS=<file>`, plus
  decompose knobs `GZIPPY_SEED_NO_WINDOWS=1` / `GZIPPY_SEED_NO_BOUNDARIES=1`). 4-wall:
  production / seed-no-windows / seeded-clean (all marker machinery removed) / rg.
- **What it perturbs (ONE thing):** removes the *window-dependency* so chunks decode clean
  instead of markered — isolating "marker machinery cost" from "clean decode cost."
- **Gate-0 (the historically-broken part — assert all):** seeded run **BYTE-EXACT**
  (sha==gzip); `seed_hits>0` and `loaded==captured N>0` (the old `=1`-is-a-file-path inert
  trap, and the 0-windows-captured trap — both must be proven dead this run); replay
  hits>0 / misses==0; conservation holds.
- **Verdict (removal-tier, STRONG):** seeded-clean wall vs production wall = the marker-
  machinery share; vs rg = whether removing it overshoots. **Branch:** marker machinery
  owns the gap (the banked answer) → the lever is making gz's marker machinery as cheap as
  rg's (the data-plane / window-absent inflate), confirm via L3. Clean owns it → kernel
  front (out of T≥2 scope; that is the T1/igzip front).
- **Confound red-team:** seeded-clean **CHEATS** (knows windows from a T1 capture; prod
  can't) → it is a **CEILING**, never the prize. Do NOT budget work at "gap-to-seeded-
  clean" or "gap-to-rg" (the banked **ceiling-as-prize = slope-as-prize** catch). It also
  removes BOTH markered-decode AND the consumer-lag/cold-redecode dependency at once, so it
  cannot by itself separate decode-CPU from overlap — that separation is L2b's job. The
  prize is always the **measured Δ of ONE concrete byte-exact cheapening A/B** (L3).

### L2b — STALL/SERIAL branch: consumer-decompose + effcores
- **Instrument:** consumer decompose (DECODE-WAIT / APPLY-WAIT / SERIAL bookkeeping) +
  `effcores=cyc/wall` + overlap% (in-flight-join vs cold-redecode), from `GZIPPY_TIMELINE`.
- **Gate-0:** span conservation (busy+idle==span); consumer.iter reconciles to chunk count;
  sink-law (a pipe-sink violation was caught mid-run before — watch for it).
- **Verdict:** DECODE-WAIT dominates ⇒ folds back to L2a (engine is the lever, not the
  pipeline — the banked outcome). TAIL/effcores-low with few-large-late chunks ⇒ tail
  imbalance (but note: prefetch-ordering already FALSIFIED at the wall → likely at-floor,
  see A1). SERIAL dominates ⇒ lean-consumer port (cite `GzipChunkFetcher.hpp:264-288`,
  off-in-order post-processing) — but memory shows SERIAL was only 3-4%, so this is a
  low-prior branch.
- **Confound red-team:** decompose is attribution-tier — it RANKS, the verdict is L3's
  perturbation. The chunk-policy rival (does rg do less work via smarter chunking?) was
  decorrelated-resolved (both 17 chunks, pool ~91%) — **do not re-open it as a chunk-size
  sweep** (see graveyard).

### L3 — causal confirm of the located stage (the ONLY verdict)
- **Instrument:** the `slow_knob` family (`GZIPPY_SLOW_DECODE` / `GZIPPY_SLOW_STORE` /
  `GZIPPY_SLOW_MARKER_MODE`, with `GZIPPY_SLOW_KIND=sleep` for the frequency-neutral arm
  and the `GZIPPY_SLOW_*_HITS=1` counters), AND the `removal_oracle`
  (`GZIPPY_ORACLE_NODECODE`/`NOSTORE`) for the removal ceiling.
- **What it perturbs:** ONE located stage's time by a known factor (≥2 magnitudes for
  slow-inject; full removal for the ceiling).
- **Gate-0:** the relevant HITS counter fired and scales with dose (non-inert), proving the
  injection lands in the intended stage and nowhere else; sha unchanged at dose-0.
- **Gate-2 verdict:** interleaved wall response monotonic+proportional ⇒ stage is on the
  critical path; flat ⇒ slack. **Run the frequency-neutral sleep control** — if Δ survives
  sleep (not just busy-spin), it is real, not a turbo-depression artifact.
- **Confound red-team:** slow-inject gives a **SLOPE (on/off-path), never a prize SIZE**
  (the banked marker-decode-loop over-read: a slope was narrated as prize, then the
  MFAST_DISABLE removal A/B capped the real prize at ≤2.5%). To BOUND the prize you must
  REMOVE the stage (oracle) or A/B a concrete cheapening — never extrapolate the slope.

---

## 2. THE FIRST STEP — leader brief for tomorrow (S0 + S1)

**Goal of the day:** produce one trustworthy `gz/rg` wall matrix at a sha-pinned current
binary, and the single bit `is-ancestor 300e772b HEAD`. That bit + the matrix decide the
entire rest of the tree. **No optimization, no locating, no perturbation today.**

**Build:**
1. Pick the subject commit. Default: the tip of `reimplement-isa-l` (it contains `300e772b`
   — the view-list win). ALSO build current `kernel-converge-A` HEAD `c91aa125` as a second
   subject IF you want to know whether the T≥2 win is present on the active T1 branch.
2. On the guest, fresh-build both gz flavors AND the comparator path identically:
   `--no-default-features --features pure-rust-inflate`, `RUSTFLAGS="-C target-cpu=native"`,
   into a fresh target dir (NOT a reused `/dev/shm` pin). Record each binary's sha.
3. Confirm `rapidgzip` native ELF v0.16.0 is on the box and runnable.

**Measure (`decide.sh`, N≥13 interleaved, /dev/null both arms, P-core masks T1/T2/T4/T7,
corpora silesia+squishy+nasa+monorepo):**

**Pass/fail gates (BLOCKING, in order):**
- G0-a rg-vs-rg self-test == 1.0 ± spread on one cell (else: rig dead, stop).
- G0-b `GZIPPY_DEBUG=1` → `path=ParallelSM`; `GZIPPY_FORCE_PARALLEL_SM=1`; sha==gzip.
- G0-c both arms `/dev/null`; tmpfs/sink equivalence measured on one cell, not assumed.
- G0-d both binaries fresh + identical flags (binary-drift killer); shas logged.
- G0-e `git merge-base --is-ancestor 300e772b <subject-sha>` result logged.
- G1 each cell: Δ vs inter-run spread reported; `<1−spread` = loss, else TIE/win.

**Decision output the leader returns:**
- The 4×4 ratio+spread matrix, the two shas, and the is-ancestor bit.
- Which FORK (A or B) S1 selects, per §1.

**Confound red-team for THIS step:**
- *Stale-pin trap:* `guest.env` points at `b22e1b14`/`/dev/shm/gz-b22-target` — do NOT
  measure that; it predates every win. Re-pin to the fresh subject sha.
- *Branch-name ≠ code:* never infer "the win is present" from being on a branch; the
  is-ancestor bit on the actual built sha is the only proof.
- *Binary-drift A/B trap:* a reused `/dev/shm` artifact retired ~5% more instr than a fresh
  build — both arms must be fresh and identically-flagged or the ratio is meaningless.
- *Sink phantom:* a file/pipe sink penalizes the faster arm and manufactures sign-flips;
  /dev/null both arms, and prove tmpfs≈/dev/null on one cell rather than asserting it.
- *Single-cell over-read:* report ALL of T1/T2/T4/T7 × 4 corpora — the deficit is corpus-
  and T-specific (silesia-T4 differs from squishy-T4 differs from nasa which often BEATS).

---

## 3. RED-TEAM OF THE WHOLE PLAN (per-perturbation confound + how the design rules it out)

| Step | The confound that could make it another chunk-sweep / N37 | How this design rules it out |
|---|---|---|
| **S1 matrix** | Measuring a stale/drifted binary, or a binary on a branch lacking the win → "deficit" is an artifact of the wrong subject. | G0-d fresh identical builds + G0-e is-ancestor bit + sha logging. One-thing: only the wall ratio varies. |
| **L1 split** | Treating instr/byte attribution as the verdict ("the lever is instruction count") — the campaign did this and reversed. | L1 only *routes* to L2; the verdict is L2/L3's perturbation, explicitly. |
| **L2a seed-oracle** | (i) inert oracle (the banked `=1`-is-a-file-path + 0-windows-captured traps) silently measures production; (ii) seeded-clean "CHEATS" → ceiling mistaken for prize; (iii) removes marker-dep AND overlap-dep at once → can't separate them. | (i) Gate-0 asserts seed_hits>0, loaded==captured>0, byte-exact, replay misses==0. (ii) ceiling labeled ceiling; prize ≡ measured Δ of a concrete A/B only. (iii) the decode-vs-overlap separation is delegated to L2b, not claimed here. |
| **L2b decompose** | Span-labeling error (the banked "267ms scan_candidate" was 91% marker-decode mislabeled as block-finding) → blames the wrong stage. | Gate-0 conservation + iter-reconcile; decompose only RANKS, L3 perturbation is the verdict. |
| **L3 slow-inject** | Slope-as-prize (narrate on-path slope as prize size — the marker-loop over-read, capped at ≤2.5% by the removal A/B); turbo-depression from busy-spin faking a delta. | Slope only proves on/off-path; prize bounded by REMOVAL oracle or concrete A/B; frequency-neutral SLEEP control required, Δ must survive it. |
| **Any chunk-size move** | The graveyard: chunk size co-varies output-buffer cache residency → slope sums scaffold-shed + cache-penalty. | The plan NEVER sweeps chunk size to localize. The one place granularity appears (A1/L2b tail) it is a *diagnostic of tail-fill*, gated against squishy as a saturated control, and was already falsified as a *shippable* lever — not re-opened as a locate tool. |
| **Whole-engine swap** | The N37 trap: swapping rg's full streaming engine drags in set_dict + staging glue → bounds nothing. | No whole-engine swaps. Every perturbation is an in-place env-gated knob changing ONE stage with a non-inert counter. |

**Cross-cutting:** every step carries the linguistic tripwire — if a draft says "finally /
the lever is / this proves," that is BIAS #1/#3 firing; rewrite as `HYPOTHESIS
(unvalidated)` + the exact next perturbation. Advisor-before-banking on any FORK decision.

---

## 4. WHAT rapidgzip's SOURCE SHOWS THE SOTA PIPELINE DOES (the structural target is knowable)

Cited from `vendor/rapidgzip/` so "match the SOTA" is a blueprint, not a guess:

- **In-flight / prefetch depth = `3 × hardware_concurrency`** —
  `core/BlockFinder.hpp:212` (`m_prefetchCount = 3ULL * std::thread::hardware_concurrency()`)
  and the prefetch gate `core/BlockFinder.hpp:173`. This is rg's answer to tail-imbalance:
  keep ≫T chunks in flight so late-large chunks don't strand cores. gz's depth = P (vendor
  `BlockFetcher.hpp:467` per memory) — **a candidate structural divergence IF FORK-B/L2b
  shows tail under-fill** (but note: the prefetch-DEPTH bump A/B was already FALSIFIED at
  the silesia-T4 wall, so this is a re-check, not a fresh lever).
- **The data plane is a list of VIEWS, not a contiguous buffer** —
  `ChunkData.hpp:418 finalize` → `cleanUnmarkedData()`; `applyWindow` at `ChunkData.hpp:247,302`;
  the view-list narrow + in-place resolve at `DecodedData.hpp:305-516`. rg's `finalize` is
  ~0.1% of program; gz's was ~10% before the port. **This is the divergence the view-list
  convergence (`300e772b`) already ported — FORK-B/L0 must confirm the subject binary
  actually contains it.**
- **One unified decoder for clean+markered**, window width the only difference —
  `gzip/deflate.hpp:513-1156` (`deflate::Block`), fused read+backref at
  `deflate.hpp:1589-1666` (`readInternalCompressedMultiCached`). gz historically split this
  into `read_internal_compressed` + `emit_backref_ring`; the 5 banked instruction-count wins
  closed most of the divergence (now ~1.20→parity instr).
- **applyWindow is a single ~0.113s pass** (memory, traced rg) — gz's marker resolution
  already BEATS this (0.49× per the divergence map), so apply_window is NOT a target.
- **Buffer REUSE/RECYCLING across chunks** (`reusedDataBuffers`, `core/FasterVector.hpp` +
  `RpmallocAllocator`) — rg faults ~1/MiB vs gz ~480/MiB, but the fault gap was removal-
  tested OFF the critical path (overlapped) → not a T≥2 lever.

The blueprint says the SOTA pipeline = view-list data plane + unified decoder + deep
prefetch. The record says gz has converged on the first two (Intel instr-parity). The only
structurally-open SOTA item is prefetch depth, and that was wall-falsified for silesia-T4.

---

## 5. WHAT NEEDS AMD FOR LAW (defer, don't block)

- **Everything.** All T≥2 results — the matrix, the view-list instruction parity, the
  silesia-T4 at-floor verdict, every cyc/byte win — are **Intel i7-13700T single-arch ⇒
  NOT-YET-LAW** (Gate-3). solvency (`root@192.168.7.222`, Zen2) is the replication box.
- **AMD-specific pre-registered confound:** PEXT/PDEP are microcoded on Zen2. Grep the
  window-absent / marker hot path for `_pext_u64`/`_pdep_u64`/`bzhi` before trusting the
  "low inversion risk" claim; the marker path is the most likely user. A win that depends on
  PEXT could shrink or invert on Zen2 — so **AMD is BLOCKING for banking the marker/data-
  plane prize specifically**, not merely "owed later."
- **Defer rule:** if solvency is offline (it has been, repeatedly — "192.168.7 segment
  dark"), this is the ONE step that waits on user infra. It does not block S0/S1/FORK-A1 or
  any FORK-B locate step — those run on Intel and stay NOT-YET-LAW until AMD pays the debt.

---

## TL;DR for the leader
1. **Do NOT locate from 0.77.** That fact is stale; the gap was located (window-absent
   inflate → `SegmentedU8` contiguous data plane) and closed to Intel instruction-parity by
   the view-list port `300e772b` on `reimplement-isa-l`.
2. **Day 1 = S0+S1 only:** Gate-0 the rig, build a fresh sha-pinned subject, log the
   `is-ancestor 300e772b` bit, run the 4T×4corpus interleaved matrix /dev/null.
3. **FORK-A (likely):** re-confirm silesia-T4 at-floor (don't re-run the falsified prefetch
   A/B), then **pay the AMD/Zen2 LAW debt** — highest VoI.
4. **FORK-B (only if deficit reappears):** check L0 (missing merge) first; then the
   instruction-vs-stall split → seed-oracle / consumer-decompose → frequency-neutral
   slow-inject. Prize ≡ measured Δ of one concrete A/B, never the gap-to-rg.
