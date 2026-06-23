# SOBER-STATE OF THE CAMPAIGN — 2026-06-21

**Author:** sober-state advisor (grounded in the on-disk record, not prose-from-memory).
**Branch:** kernel-converge-A @ `462b0549` (HEAD = the just-landed pin DELETE).
**Stamp on every perf claim below:** see the per-item arch/freeze stamp. The headline
matrix is FROZEN-Intel + quiet-mac, **NOT-YET-LAW** (AMD/Zen2 owed). A fresh full
matrix is being measured separately — claims that depend on it are flagged `⟂MATRIX`.

---

## 0. THE "CIRCLING" CORRECTION (the user's pushback is CORRECT)

The "we've been circling the rewrite-vs-accept fork" framing is wrong as a description
of the WORK. What circled was the *prose attribution layer* — the 74 distinct "the
lever is X" claims, reversals 2:1 over conclusions (`plans/BIAS-FORENSICS.md` §HEADLINE,
mined from 764 MB / 1,509 .jsonl). That is the documented bias, and it is real. But
underneath it, **a sequence of large, byte-exact structural rewrites actually LANDED on
the branch** and were gated. The git log is the receipt. Section 1 is that record.

---

## 1. SIGNIFICANT WORK THAT LANDED (the "we did real rewrites" record)

Each row: what it was → gated status. Commits are on `kernel-converge-A` unless noted.

| # | Rewrite / port | Commit(s) | Status |
|---|---|---|---|
| 1 | **Data-plane port to rapidgzip `vector<VectorView>`** — ChunkData::data converged onto rg's segmented view list (O(1) front-prepend + direct narrow), replacing the contiguous-growing Vec | `300e772b` (ancestor of HEAD, verified) | **KEPT** — byte-exact, **−18–20% program instr** |
| 2 | **Engine A (flat libdeflate-style decoder) wired into the clean contig path**, then clean path converged to **100% engine A**, retiring engine-B's two-level careful loop + its `lut_litlen` build on the clean path | `f6b1bcbe` → `834ba516` → `773ad62c` → `cffa61ee` | **KEPT** — byte-exact x86 + cross-ISA; engine-A beat engine-B 2.07–3.84× in the S0.5 A/B (`f6b1bcbe`) |
| 3 | **SOLE-PATH clean convergence validated** on x86 (asm-off), −2% double-build drop replicated, engine-A win unchanged/better | `39160e00`, `c4c3cc97` | **KEPT** (D2/D3 cross-validation) |
| 4 | **NIGHT40 heroic `run_contig` restructure** — hoist the D-1 un-consume {d0} anchor off the hot literal path toward igzip `_04`'s integrated loop | `46f74d69` | **KEPT** — byte-exact + CI-disjoint win BOTH corpora (silesia −0.0169, monorepo −0.0229 cyc/B); closed ~1.4–1.5% of the igzip gap |
| 5 | **Table-build convergence toward igzip** — drop redundant per-block litlen double-clear (kept); per-symbol speculative self-guard gating (reverted, see §2) | `de05fc80` (kept), `fef1f8c7`→`6dc67b1a` (reverted) | MIXED — one KEPT, one REVERTED-on-measurement |
| 6 | **DELETE speculative decode-worker pinning** (gzippy had added `with_pinning_for_capacity` rg never had; on SMT it packed 2 workers onto sibling cores) → default unpinned, faithful to rg `BlockFetcher.hpp:185` | `c9dceb68`→`225a7369`→`462b0549` | **KEPT (just landed)** — FROZEN-Intel N=13; the silesia-T4 +18–20% loss was THIS pin |
| 7 | **Measurement/freeze tooling** — `bench-lock.sh` (one host-freeze lifecycle, triple-restore), `kernel_gate.sh` (Gate-0 FFI-off proof + Gate-1 Wilcoxon/bootstrap), `standing.sh` (one-command gated matrix vs rg+igzip), distpreload cyc/B single-core gated-on-a-loaded-box, the 11 env-gated byte-transparent instruments, `parallel_sm_tail_metric.py` conservation-gated | many; catalogued in `plans/TOOL-INVENTORY-2026-06-21.md` | **LIVE** — this is the durable apparatus |

This is not circling. It is ~6 byte-exact structural landings plus a measurement rig,
each carrying a gated verdict. The reversible/reverted ones (5, parts of others) are
the SYSTEM WORKING (TIE-or-better keep bar enforced), not wheel-spinning.

---

## 2. THE HONEST GATED LEDGER (current truth)

### KEPT (gated wins — frozen-Intel BOX-VALID or quiet-mac; all NOT-YET-LAW)
- **Pin DELETE** — silesia-T4 unpinned 1.028 vs pinned 1.198 vs rg; FROZEN Intel N=13;
  /proc witness proved the OS spreads workers to 4 distinct physical cores. (`225a7369`)
- **Data-plane viewlist port** — −18–20% program instr, byte-exact. (`300e772b`)
- **Engine-A clean-path unification** — byte-exact, cross-ISA, win unchanged/better. (`cffa61ee`/`39160e00`)
- **NIGHT40 anchor hoist** — CI-disjoint cyc/B win both corpora; ~1.4–1.5% of igzip gap. (`46f74d69`)
- Earlier banked gated-KEPT T1 wins (per `project_beat_igzip_t1_campaign.md`):
  dist-preload, flag-bit, ratio-reserve, T1-depth-1 (nasa −22.9%), chunk-size-T1=1MiB.
  *(These predate this branch's head; re-verify at HEAD before citing as current.)*

### DISPROVEN / DEAD (FALSIFY entries — do NOT re-derive; these are the real "dead ends")
- **"Marker loop is a 1.5–2.3× gz-vs-rg gap; port rg's leaner marker loop"** — DISPROVEN
  (`plans/MARKER-TAX-DISPROOF-2026-06-21.md`, `dc3158e6`). The 2.3× is a ratio-of-ratios
  artifact of gz's light T1 base; absolute marker work is ≤ rg's where markers dominate
  (nasa); the gz-vs-rg gap **anti-correlates** with marker fraction (Intel wall r=−0.999,
  mac instr r=−0.767). A leaner marker loop cannot be the silesia lever.
- **Consumer writev / zero-copy consumer at silT4/T8** — SLACK, STOP (`9f2abf18`).
- **Clean-path speculative table-build guard** is the igzip lever — REVERTED, slope TIE,
  guards were dead code (`322e8725`/`6dc67b1a`).
- **"silesia-T4 idle is in-decode-stall/drain-tail"** — DISPROVEN; it was OVERSUBSCRIPTION
  (the pin), not stall (`c488c448`→ pin delete confirmed it).
- **Hot 5th consumer thread** as the +18% — DISPROVEN; consumer is COLD (duty 0.043),
  the +18% was SMT co-location of WORKERS (`f86da806`/`38dbbb81`).
- Older falsified levers (cautionary, `MEMORY.md` disproven section): buffer-warmth,
  key-pin, prefetch-depth, decode-volume, emission-port, footprint-align/reservation,
  consumer-coherence. **74 total — none citable as current.**

### REFINED (claim core survives, label corrected)
- **"The gz-vs-rg gap = x86 ISA-L clean INNER-KERNEL"** → NEEDS-REFINEMENT
  (`plans/CLEANKERNEL-CLAIM-DISPROOF-2026-06-21.md`, `50ebadd1`). Survives: the gap is
  real (silesia-T4 +16% reproduced, A/A=1.003), tracks the CLEAN fraction, x86-specific.
  Refined: it is the **clean per-symbol PATH (copy/store + decode-loop throughput)**,
  NOT the Huffman arithmetic (that is ≤12% of the clean wall, ≤6% of instr on both
  arches; CRC ≤0.02%). WHICH ISA-L technique closes it is OWED.

### OPEN (live, not yet gated to a verdict)
- The **clean per-symbol path** sub-component (AVX2 copy vs multi-symbol decode-loop
  packing) — needs an ISA-L-vs-gz isolation (rg --verbose folds them). `⟂MATRIX`-adjacent.
- The **marker-machinery absolute instruction** reduction (it adds 6.9–14.3 instr/B; on
  aarch64 gz is still lighter absolute than rg, so it's a generic-efficiency target of
  UNPROVEN wall sign — `PIPELINE-PORT-DESIGN.md` STEP-0 discriminator is the gate).
- **Low-T pipeline fixed-overhead** (+2.70 cyc/B mac silesia 4MiB) makes T2<T1 on cheap
  corpora; cross-arch mechanism but not yet Gate-2 perturbed (`PARALLEL-SCALING-2026-06-20.md`).

### Trust tiers
- **FROZEN-Intel BOX-VALID:** pin-delete (N=13 frozen). 
- **NOT-YET-LAW (Intel unfrozen / quiet-mac, AMD owed):** the STANDING-MATRIX, engine-A,
  NIGHT40, marker/cleankernel disproofs, scaling.
- **AMD/Zen2:** OFFLINE per user this campaign → every above row OWES AMD for LAW.

---

## 3. WHERE WE ACTUALLY STAND vs THE GOAL (`plans/STANDING-MATRIX-2026-06-20.md`)

Goal: gzippy-native (pure-Rust, FFI-off) ≥ parity with every gzip tool incl rapidgzip +
igzip, at every T. Honest per-arch read (NOT-YET-LAW; `⟂MATRIX` to confirm):

**Intel x86 (asm-kernel build, vs rg [T>1 SOTA] + igzip [T1 SOTA]):**
- vs **rapidgzip**: nasa WINS/ties every T (T1 −22.9%); monorepo WINS T1 / ties; silesia
  is the **standing loss** (T1 +6.7%, **T4 +16.1%**, T2/T8 TIE). The +16% silesia-T4
  reproduces (W3) — it was NOT the pin alone; the pin delete fixed an oversubscription
  loss that the standing matrix (pre-delete) had folded in. **After the pin delete, T≥2
  silesia moves toward parity — but the standing-matrix numbers PRE-DATE the delete; the
  fresh matrix must confirm whether silesia-T4 is now at-parity or still carries a
  clean-path residual.** `⟂MATRIX` — this is the single most important number to re-read.
- vs **igzip**: gz beats ig at T≥4 (parallel) but **loses single-core T1** (ig = ISA-L
  hand-asm SOTA): silesia +24%, monorepo +40%, nasa is the exception (gz WINS T1 −33%).

**macOS aarch64 (pure-Rust engine A):**
- vs **rapidgzip**: gz WINS every cell on instr/cyc/wall (rg has NO ISA-L on aarch64 →
  runs ~3× heavier portable inflate). **This does NOT transfer to x86.**
- vs **libdeflate** (T1 serial ref): gz costs ~24–29% more cyc/B & instr/B on silesia.
- The transferable signal: gz's pipeline inflates instr T1→T by 1.8–4.1× vs rg's 1.2–1.8×
  (a coordination tax differential) — but absolute gz is lighter on aarch64.

**Net:** at T≥2 gzippy-native is **near-parity-to-winning vs rapidgzip** on most corpora
(silesia-T4 the open question post-pin-delete `⟂MATRIX`); the **durable deficit is T1
single-core vs igzip (~25%)** — the ISA-L hand-asm serial kernel.

---

## 4. THE ACTUAL NEXT GOAL — RANKED CANDIDATES

Ranked by (goal-impact × tractability). Each carries evidence-tier + honest ROI.

### A. CONFIRM silesia-T4 post-pin-delete + convert the matrix to LAW (cheapest, highest-VoI)
- **Why first:** the pin delete (`462b0549`) just changed the single worst gz-vs-rg cell.
  The STANDING-MATRIX is now STALE for silesia/monorepo T≥2. We do not actually KNOW
  where we stand vs rg at T≥2 until the fresh matrix lands. **Re-running the gated matrix
  at HEAD is the highest-information, lowest-cost action.** `⟂MATRIX` (already in flight).
- **AMD/Zen2 LAW conversion** stacks here: when solvency returns, replay the KEPT stack
  (pin-delete, engine-A, data-plane, NIGHT40) through `kernel_gate.sh` on Zen2. Cheapest
  path to flip "NOT-YET-LAW" → "LAW" for the whole banked stack at once.
- Evidence-tier: STRONG (it's measurement, not a lever). ROI: high — it either banks a
  parity win or re-localizes the residual. Risk: none (measurement only).

### B. The T1 inner-kernel codegen front vs igzip (~25% — the BIG remaining gap)
- **The honest ROI tension (a user R3, do NOT resolve here):**
  - The banked `BEAT-IGZIP-T1` assessment (`plans/BEAT-IGZIP-T1-STATE.md` lines
    1363–1369, 1543–1563): removal-oracle ceiling = +0.908 cyc/B silesia but **DIFFUSE**
    across refill/classify/loop-overhead + LUT-build; only a fraction is capturable per
    technique (flag-bit got ~0.10, NIGHT40 got ~0.017). Realistic = **~3–6% WALL over
    MULTIPLE sessions, HIGH byte-exact risk** → advisor's LEAD rec = **BANK + ACCEPT**
    the narrowed gap (silesia +28%, nasa +30%, down from +39.5%/+116.5%).
  - The removal-oracle (`NIGHT18`, line 370) is STRONG-tier: removing the resumable/marker
    glue closes only **16–27% of instr gap, ~3–6% of WALL** → a dedicated stateless T1
    kernel sheds ~¼ the instructions but ~5% of the wall. NOT a path to parity.
  - **Counter-stance (user):** "no floor / accept until igzip parity, heroic rewrites
    funded." The CLAUDE.md goal is literal parity with igzip. Under that stance the
    ~3–6% poor-ROI estimate is not a stop sign; it's the price.
  - **Unexhausted lever the record flags:** `5526585e` redirected the port target — ~44%
    of the +3.22 instr/B STEP-0 gap lives in the `decode_clean_into_contig` SCAFFOLD
    *outside* run_contig, not the hot loop. The scaffold (not the asm loop) is the
    next-attack surface, and it's a less-explored, less-heroic target than deeper asm.
- Evidence-tier: STRONG (oracle-bounded). ROI: poor-per-technique but the ONLY route to
  the literal igzip-parity goal. Risk: HIGH byte-exact, multi-session. **This is the R3.**

### C. The clean per-symbol PATH on x86 (the refined silesia-T4 rg gap, IF it survives the matrix)
- From CLEANKERNEL-DISPROOF: the silesia-T4 rg loss is the clean copy/store + decode-loop
  throughput where rg uses ISA-L. **Conditional on §A:** if the pin delete already closed
  silesia-T4, this front evaporates; if a residual remains, this is the rg-convergence
  target (an ISA-L AVX2 copy / multi-symbol packing port). `⟂MATRIX` gates whether this
  is even live. Evidence-tier: STRONG-attribution (needs ISA-L-vs-gz isolation for causal).

### D. Sole-path collapse / dead-code removal / cleanup (the end-state simplification)
- The surface is large and tangled: **57 .rs in `src/decompress/parallel/`**, 11
  instruments, 85 inflate/decompress building blocks, and **148 remote branches** (most
  forgotten: jit/cranelift/llvm-ir/bytecode, dozens of feat/* oracles). The
  TOOL-INVENTORY flags LOST tools (built in cleaned worktrees, gone from the tree).
- `MEMORY.md` `feedback_stop_circling_converge_and_clean`: the user's own directive is
  faithful STRUCTURAL convergence on rg + a **clarity refactor (rename files/dirs/fns,
  fix docs) so the real gap becomes visible** — "the code's tangle is why we keep
  re-deriving the same dead fork." This is endorsed work, not gold-plating.
- ROI: doesn't move the wall, but it's the goal-completion criterion ("pure-Rust decoder
  is the SOLE decode path, C-FFI off the decode graph") and it lowers the cost of every
  future measurement. Evidence-tier: N/A (mechanical, byte-exact). Risk: low if byte-exact.

### Ranking
1. **A** (confirm matrix + AMD-LAW) — must happen first; everything else is conditional on it.
2. **D** (cleanup / sole-path collapse) — endorsed, low-risk, makes the real gap visible.
3. **B vs C** — the actual perf fork, and a genuine **R3 for the user** (accept the
   narrowed gap vs fund the heroic kernel rewrite). C only exists if A shows a residual.

---

## 5. ONE-PARAGRAPH SOBER VERDICT

The campaign has NOT been circling — six byte-exact structural rewrites LANDED and were
gated on `kernel-converge-A` (data-plane viewlist port −18–20% instr; engine-A clean-path
unification; NIGHT40 anchor hoist; and the just-landed worker-pin DELETE that fixed the
worst gz-vs-rg cell), on top of a real measurement apparatus (bench-lock, kernel_gate,
standing rig, distpreload, 11 instruments). What circled was the prose attribution layer
(74 reversed "the lever is X" claims) — and the response has been the right one: a
disproof discipline that this very session used to KILL the marker-loop-port direction
and REFINE the clean-kernel claim. Where we stand (frozen-Intel/quiet-mac, NOT-YET-LAW):
gzippy-native is near-parity-to-winning vs rapidgzip at T≥2 on most corpora, with
silesia-T4 the one open cell now that the pin (its +16–18% cause) is deleted — **the
fresh matrix must confirm this**. The one durable, large deficit is **T1 single-core vs
igzip (~25%)**, which the oracle bounds as only ~3–6% capturable by deep asm at high
byte-exact risk. **The actual next goal, in order: (1) re-measure the matrix at HEAD to
learn where the pin-delete left us vs rg + convert the KEPT stack to AMD-LAW; (2) the
sole-path/cleanup collapse the user already directed; (3) put the real strategic fork to
the user as an R3 — ACCEPT the narrowed igzip-T1 gap (advisor-recommended, poor ROI) vs
FUND the heroic inner-kernel/scaffold rewrite toward literal igzip parity (the stated
goal, multi-session, high risk).** The tension between the banked "~3–6%, accept" estimate
and the user's "no floor until igzip parity" stance is not ours to resolve — it is the
decision to surface.

---

### Claims that DEPEND on the in-flight fresh matrix (`⟂MATRIX` — confirm before banking)
- silesia-T4 gz-vs-rg AFTER the pin delete (is it now at-parity or residual-bearing?).
- Whether candidate C (clean per-symbol x86 path) is still a live front.
- The current per-T gz/rg ratios on silesia/monorepo (STANDING-MATRIX is pre-pin-delete).
- The exact igzip-T1 gap at HEAD (cited ~25% / +28% silesia is from BEAT-IGZIP-T1-STATE,
  pre-NIGHT40-and-later landings).
