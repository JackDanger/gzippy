# PIN-DISCRIMINATOR — silesia-T4 fork SETTLED (2026-06-21, FROZEN-Intel, gated, NOT-YET-LAW)

**Question (Opus-advisor-decided):** is the faithful structural fix for gzippy's
silesia-T4 loss to **DELETE the decode-worker pinning** (let the OS schedule, like
rapidgzip `BlockFetcher.hpp:185` = empty map) or a **portable topology-aware
distinct-physical pin**? RISK to settle: gzippy may have added pinning to escape a
restricted container cpuset (commit 38dbbb81 "escape inherited 1-core mask") — if
the LXC launches with a constrained mask, plain no-pin could leave workers stuck.

**VERDICT: DELETE THE PINNING.** Ship arm B (the empty-map ctor = `GZIPPY_NO_PIN`),
faithful to rapidgzip, portable by construction, removes code. PHYS_PIN discarded.

## The measurement
- Box: neurotic Intel i7-13700T LXC (guest 199), **FROZEN** (no_turbo=1, gov=performance,
  8 noisy guests frozen, watchdog armed; BENCH_LOCK=quiet runnable_avg=1.75), **UNPINNED**
  (no external taskset — the OS/container decides placement, which is the whole question).
- 3 byte-exact arms of the SAME gz-native binary (sha c9dceb68), env-selected:
  A = HEAD default (`with_pinning_for_capacity` — SMT-packing pin),
  B = `GZIPPY_NO_PIN=1` (empty `ThreadPinning` → pin:None per worker — rg-faithful, candidate FIX),
  C = `GZIPPY_PHYS_PIN=1` (distinct-physical {0,2,4,8} diagnostic).
- vs `/root/oracle_c/rapidgzip-native` 0.16.0. silesia + monorepo × T2/T4/T8, interleaved
  best-of-N=13 (n=26 w/ A/A replicate), /dev/null both arms, perf cyc/B + GHz.
- Gate-0: build-flavor=parallel-sm+pure, path=ParallelSM, all 3 arms + rg sha==zcat
  (pinning is byte-transparent), A/A self-tests all PASS ≤1.03 (one C-T8 CHECK 1.06).

## THE 3-arm table (best wall ms / gz-arm-over-rg best ratio / cyc/B)
| cell        | A (HEAD pin)            | B (NO_PIN)              | C (PHYS_PIN)            | RG     |
|-------------|-------------------------|-------------------------|-------------------------|--------|
| silesia T2  | 721.8 / 1.020 TIE /9.21 | 726.6 / 1.026 TIE /9.26 | 729.0 / 1.030 TIE /9.19 | 708.0  |
| **silesia T4** | **467.0 / 1.198 LOSS /11.23** | **400.7 / 1.028 TIE /9.80** | 404.4 / 1.037 TIE /9.65 | 389.8 |
| silesia T8  | 320.1 / 1.178 LOSS /11.50 | 272.2 / 1.002 TIE /10.57 | 287.6 / 1.059 TIE /9.07 | 271.7 |
| monorepo T2 | 212.6 / 1.039 TIE /10.25| 215.0 / 1.051 TIE /10.54| 219.2 / 1.072 TIE /10.26| 204.6  |
| monorepo T4 | 144.8 / 1.081 TIE /13.22| 130.1 / 0.972 TIE /11.85| 136.9 / 1.023 TIE /11.43| 133.9  |
| monorepo T8 |  91.2 / 1.002 TIE /14.80|  87.5 / 0.961 TIE /13.95|  98.8 / 1.085 TIE /11.30|  91.1  |

Gate-1 TIE band = max relative inter-run spread of the two arms. silesia-T4/T8
arm A is the ONLY non-tie: A/rg 1.198 (Δ0.198 ≫ band 0.084) and 1.178 (Δ0.178 ≫ 0.138).
**B/rg silesia-T4 = 1.028 TIE; B≈C (0.991 TIE).** B fixes the loss; ties rg.

## THE NON-INERT /proc CORE-PLACEMENT WITNESS (silesia T4, unpinned) — settles the fork
- **A:** workers `allowed=[2]`,`[3]`,`[4]` — cpu2/cpu3 are SMT siblings (both phys=[2]).
  → **4 workers on 3 distinct physical cores, collision on phys 2.** The bug, reproduced.
- **B:** every worker `allowed=[0,2-5,8,10,12-17,19-21]` (the FULL container cpuset, NOT a
  constrained 1-core mask); OS spread them to dom cpus 8/10/2/4 → **4 distinct physical
  cores, no collision.**
- **C:** workers locked to {0,2,4,8} → 4 distinct physical, box-specific.

**The container-constrained-cpuset RISK is FALSIFIED.** Unpinned workers inherit the full
cpuset and the OS spreads them across distinct physical cores. The 1-core mask only
appeared in A/C because gzippy's OWN pinning (`set_for_current`) narrowed it.

## Why DELETE (B) over the portable pin (C)
1. B ≈ C ≈ rg at the contested cell (B/rg 1.028, C/rg 1.037, B/C 0.991 — all TIE) AND
   B's workers land on 4 distinct physical cores unpinned → the OS already does the right thing.
2. C is box-specific ({0,2,4,8}) and at high T is WORSE: T8 C GHz throttles to 1.20/1.16,
   C/rg 1.059 (silesia) / 1.085 (monorepo, A/A CHECK 1.06) — the hardcoded map collides the
   consumer/workers at T8. B stays clean (T8 B/rg 1.002 silesia / 0.961 monorepo).
3. B removes code (the whole `with_pinning_for_capacity` + `pinning_for_capacity` +
   `distinct_physical_core_ids` machinery) and matches rg `BlockFetcher.hpp:185` by construction.

## Committed (gated, reversible) — c9dceb68 on origin/kernel-converge-A
- `thread_pool.rs`: `no_pin_enabled()` (`GZIPPY_NO_PIN=1`, byte-transparent, default OFF).
- `chunk_fetcher.rs:766`: arm B builds the decode pool with an EMPTY `ThreadPinning`.
- The rig: `scripts/bench/kernel-ab/pin_discriminator_{guest.sh,placement.py,report.py}`.

## Scope / owed
- FROZEN-Intel BOX-VALID, single-arch → **NOT-YET-LAW** (AMD/Zen2 replication owed; SMT
  topology + cpuset layout are box-specific so the placement witness must replicate).
- **R3 (user strategic decision):** promoting B to the DEFAULT (delete the pin in production)
  changes thread placement on ALL boxes/arches — Intel-only evidence here. The fix is COMMITTED
  as a default-OFF env knob; flip-to-default is the R3 call (recommended: DELETE the pin =
  default-on B / remove the pinning code, since it's faithful to rg and removes box-specific code).
- Caveat: bench-lock's uncore-freq wrmsr segfaulted (line 114) so uncore stayed at baseline;
  no_turbo=1 + gov=performance held and GHz was stable 1.39 across the trusted arms, so the
  walls are trustworthy (the only GHz anomaly is C's real high-T co-location throttle).
