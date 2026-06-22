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

## LOCATE of the silesia-T4 residual — (in progress; Gate-2 removal-oracle is the verdict)

(filled in below as the locate runs)
