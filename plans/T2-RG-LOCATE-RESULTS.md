# T≥2 vs rapidgzip — RE-BASELINE + FRESH LOCATE: RESULTS

**Date:** 2026-06-22. **Branch:** `t2-rg-locate` → pushed to `origin/kernel-converge-A`.
**Subject sha:** `42b8c869` (= `kernel-converge-A` HEAD + this cycle's docs; code identical
to `036b835d`). **Falsifier:** `plans/T2-RG-LOCATE-FALSIFIER.md` (pre-registered, committed
before any number). Grading every claim against the gates in CLAUDE.md.

---

## Gate-0 — COMPARATOR CONFIRMED on BOTH boxes (the "no rg on solvency" trap checked)

| box | arch | rapidgzip | self-test (A/A rg-vs-rg) | corpora | toolchain |
|---|---|---|---|---|---|
| neurotic | Intel i7-13700T | `/root/oracle_c/rapidgzip-native` v0.16.0 ELF | rg A/A ≤ 3.7% across cells (tol 5%) → OK | silesia/monorepo/nasa/squishy present | cargo 1.96 |
| solvency | AMD EPYC 7282 Zen2 | `/root/gz-base/vendor/rapidgzip/librapidarchive/build/src/tools/rapidgzip` v0.16.0 ELF (functional: silesia → 211,968,000 bytes) | (matrix in progress) | silesia/monorepo/nasa/squishy present | cargo 1.96 (via /root/.cargo/bin, symlinked into /usr/local/bin) |

- Both rapidgzip are the **native ELF v0.16.0**, NOT the pip wheel.
- Solvency rg comparator was ABSENT at the memoryed `/root/oracle_c/` path but a built
  native ELF exists in the vendor build dir; confirmed functional. boxes.sh solvency entry
  updated with live paths (`root@10.0.2.240`, BOX_RG, BOX_SRC=/root/gz-head, corpora /root).
- `is-ancestor 300e772b HEAD` = **TRUE** — the view-list data-plane convergence IS in the
  subject binary (the FORK-B/L0 "missing merge" explanation is ruled out).
- Each arm sha-verified == zcat; `path=ParallelSM`; build-flavor=parallel-sm+pure; same
  /dev/null sink; fresh identical builds (`-C target-cpu=native`). All Gate-0/Gate-4 PASS.

---

## RE-BASELINE — Intel matrix (neurotic, N=13 interleaved, GZ/GZ2/RG/RG2/IG arms)

Ratio = `gz_wall / rg_wall`; `<1.0` = gz faster. Verdict applies the per-cell spread gate
(Δ<spread ⇒ TIE). Box: turbo-on/powersave but base-pinned & frequency-stable (GHzσ ≤0.4%),
A/A self-tests all OK ⇒ ratios trusted. load_start 8.32 (loaded LXC; interleave cancels).

| corpus | T1 | T2 | T4 | T8 |
|---|---|---|---|---|
| monorepo | 0.920 WIN | 1.028 **TIE**(±3.8%) | 0.966 TIE | 0.963 TIE |
| nasa | 0.774 WIN | 0.968 TIE | **0.889 WIN** | **0.871 WIN** |
| silesia | 1.032 LOSS* | 1.029 **TIE**(±3.4%) | **1.036 LOSS** | 0.988 TIE |
| squishy | 1.001 TIE | 0.969 TIE | 1.027 **TIE**(±5.8%) | 0.945 TIE |

\* silesia-T1 (1.032) is a real loss but T1 = inner-kernel front (igzip/native single-stream),
OUT of the T≥2 scope of this cycle.

### The ONLY statistically-significant T≥2 LOSS on Intel = **silesia-T4: gz/rg 1.036 (3.6%)**
- spread gz 1.9% / rg 1.6% ⇒ Δ 3.6% > spread ⇒ REAL (not a TIE).
- cyc/B: gz 9.61 vs rg 9.37 ⇒ **cyc ratio 1.026** (near-parity WORK). wall ratio 1.036.
  wall ≈ cyc ratio ⇒ the residual is now MOSTLY a small work-excess plus a small
  serialization sliver — NOT the old "cyc-parity-but-wall-1.17 tail-imbalance" shape.
- **The old matrix's "silesia-T4 ~1.165" is STALE.** At HEAD it is ~1.036. The view-list
  convergence + subsequent work shrank it ~4.5×.

### H-BASELINE verdict (Intel)
Largely **FALSIFIED**: the broad T≥2 gz<rg deficit is CLOSED at HEAD on Intel. gz BEATS rg
on nasa T2/T4/T8 and ties/wins monorepo+squishy at all T≥2. The lone real residual is
**silesia-T4 (3.6%)** — a small, single-cell, single-corpus tax. (AMD replication pending.)

---

## RE-BASELINE — AMD matrix (solvency, Zen2 EPYC 7282, FROZEN gov=performance/boost=0, N=15, A/A-gated)

Box restored to default (gov=ondemand, boost=1) after the run; watchdog killed; the
user's `llama-completion` workload (1 roaming core) was untouched — it forced the T8
cells UNTRUSTED (A/A>5%). T1–T4 cells passed A/A. Ratio = `gz_wall/rg_wall`.

| corpus | T1 | T2 | T4 | T8 |
|---|---|---|---|---|
| monorepo | 0.815 WIN | **1.095 LOSS (9.5%)** | 0.989 TIE | 0.957 (untrusted) |
| nasa | 0.649 WIN | 1.005 TIE | 0.969 TIE | 0.886 (untrusted) |
| silesia | 0.982 TIE | **1.070 LOSS (7.0%)** | **1.075 LOSS (7.5%)** | 0.946 (untrusted) |
| squishy | 0.945 WIN | 1.007 TIE | **1.075 LOSS (7.5%)** | 1.067 (untrusted) |

### GATE-3 REPLICATION FAILS — the Intel "T≥2 closed" picture does NOT hold on AMD
- AMD has BROADER, LARGER real T≥2 losses than Intel: monorepo-T2 (9.5%), silesia-T2
  (7.0%), silesia-T4 (7.5%), squishy-T4 (7.5%) — vs Intel's lone silesia-T4 (3.6%).
- **The whole Intel near-parity stack is therefore NOT-YET-LAW and, where measured,
  REFUTED on AMD.** gz still WINS T1 broadly on both arches (isal/native single-stream).

### AMD loss shape (cyc/B vs wall — the instruction-vs-stall split, freq-pinned)
- **T4 losses are WORK-bound:** silesia-T4 cyc 11.72 vs rg 10.84 = **cyc ratio 1.081 ≈ wall
  1.075**; squishy-T4 cyc 1.087 ≈ wall 1.075. gz retires ~8% more cycles on the SAME chunks.
  This work-excess is LARGER on AMD (cyc 1.08) than Intel (cyc 1.03).
- **T2 losses have a SERIALIZATION component:** monorepo-T2 cyc 1.017 but **wall 1.095**
  (wall ≫ cyc) ⇒ a parallel-pipeline/serialization tax at low T on AMD; silesia-T2 cyc
  1.038 / wall 1.070 (mixed).

---

## LOCATE of the losing-cell tax — fresh, this cycle

### (1) Intel silesia-T4 (the one Intel residual) — WORK, not tail, not marker
- **effcores/tail decompose** (GZIPPY_TIMELINE, Gate-0 PASS, span conservation held): the
  4 workers are **95–98% busy** (478–502 ms of 511 ms life), well-balanced; tid1 is the
  in-order consumer (mostly waiting — expected). effcores 3.65–3.80/4, tail/global 1.26.
  ⇒ NOT a scheduling/tail/serialization gap.
- cyc/B ratio (1.02–1.03) ≈ wall ratio ⇒ the residual is a small distributed WORK excess
  in the saturated workers' decode, NOT coordination.

### (2) marker machinery contribution — REMOVAL-ORACLE BROKEN at HEAD; bounded by other arms
- **The `seed_windows` removal-oracle is INERT on `kernel-converge-A`** (Gate-0 fails non-
  inert): `record_clean_window` fires only on the `window_at_offset.is_some()` natural-clean
  branch, which the speculative pipeline rarely takes at T>1 (captures **0 windows** at
  p2/p4 → seeding hits=0 = measuring production); and the historic "capture at p1" fix no
  longer reaches `drive_impl` because p1 now routes to the thin-T1 driver
  (`chunk_fetcher.rs` recent T1-cache work). So NO clean Gate-2 removal verdict on the
  marker share this cycle. (Repair = a separate instrument task; deferred — see NO-GO below
  for why it isn't load-bearing.)
- **Non-inert marker-fraction comparison (fresh, GZIPPY_VERBOSE vs rg --verbose, silesia-T4,
  same 17 chunks):** rg replaces **34.5%** marker symbols (73,124,965), apply-window 0.116 s.
  gz processes the same 17 chunks (flip_to_clean 13/17), window-absent body 74.9 MB (~35% of
  the 212 MB output); within it ~34.5% of symbols are markers — **gz's marker VOLUME ≈ rg's.**
  Banked + consistent: gz's marker resolution (apply_window) already BEATS rg (≈0.49×).
- **PEXT/PDEP confound RULED OUT (objdump, definitive):** gz binary = **0** PEXT/PDEP,
  753 fast-BMI2 (bzhi/shrx/shlx, all 1-cycle on Zen2); rg = 0 PEXT/PDEP. So the larger AMD
  cyc/B is NOT a microcoded-BMI2 artifact — it is genuine relative codegen efficiency.

---

## VERDICT vs the PRE-REGISTERED FALSIFIER

### NO-GO on the marker-structure port — CONFIRMED (both arches)
Per the pre-registered NO-GO, the marker port is forbidden because MULTIPLE NO-GO criteria
are met:
- **gz's marker path already MATCHES rg's structure** (marker volume gz ≈ rg; gz's
  apply_window already BEATS rg) — there is no "gz does MORE" divergence to converge. This
  alone triggers NO-GO independent of the (broken) removal-oracle.
- **The marker machinery is NOT the dominant bucket of the losing-cell tax.** Intel
  silesia-T4 is a small work-excess (cyc 1.03) in saturated workers; AMD T4 is an ~8%
  WORK (cyc) excess that tracks the wall — both point at the inner window-absent DECODE
  KERNEL codegen, not the speculative marker/apply-window/replace-markers STRUCTURE.
- rg carries the SAME u16 marker machinery (34.5% replaced markers here) — deleting/altering
  the structure is not faithful convergence.

### The ACTUAL dominant bucket (ranked) and the real frontier
1. **AMD T4 (silesia, squishy): inner window-absent DECODE-KERNEL cyc/B excess on Zen2
   (~8%)** — the largest, cleanest losing bucket. WORK-bound (cyc≈wall), no PEXT/PDEP, gz's
   asm `run_contig` kernel is relatively less Zen2-efficient than rg's `Block::read`. This is
   the **inner-Huffman kernel front** (CLAUDE.md "open territory"), the SAME front as the
   T1-native deficit — NOT a parallel-architecture/marker convergence target.
2. **AMD T2 (monorepo, silesia): a parallel-pipeline SERIALIZATION component** (wall ≫ cyc).
   A distinct, un-located low-T AMD behavior.
3. **Intel silesia-T4 (~3.6%, borderline TIE):** a small distributed work-excess; same
   inner-kernel front, at-or-near floor on Intel.

### Recommended next lever (do NOT start the marker port)
- **Highest VoI: an AMD-side instruction-mix / perf-annotate locate of gz `run_contig` vs rg
  `Block::read` on Zen2** (HYPOTHESIS: the inner window-absent kernel is the dominant AMD T4
  bucket; confirm with a Zen2 perf-annotate + a Gate-2 perturbation, prize ≡ measured Δ).
  The inner-Huffman loop is open territory; this is where the cross-arch (LAW) gap actually
  lives — not the marker structure.
- **Second: locate the AMD T2 serialization** (effcores/consumer-decompose on AMD
  monorepo/silesia T2 — the standing rig's EFFCORES mode now works on AMD after the
  perf-events fix).
- **Repair the `seed_windows` oracle** (make `record_clean_window` capture via a sequential
  full-window pass that reaches `drive_impl`, or capture at the speculative-resolve point) so
  a clean Gate-2 marker-vs-clean removal verdict is available next cycle — though the NO-GO
  above does not depend on it.

### Instrument/rig fixes landed this cycle (banked)
- `_standing_guest.sh`: arch-conditional perf EVENTS (the Intel hybrid `cpu_core/<ev>/`
  syntax errored on every AMD run → zero samples → a misleading "no trusted cells — box too
  loaded"). AMD now produces a trusted matrix.
- `boxes.sh`: solvency live paths confirmed (root@10.0.2.240, BOX_RG = vendor build ELF,
  BOX_SRC=/root/gz-head, corpora /root); cargo/rustc symlinked into /usr/local/bin for
  non-interactive ssh.
- `_seed_silt4_guest.sh`: seed-windows removal-oracle harness (revealed the inert oracle).

