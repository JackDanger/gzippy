# scripts/bench/deprecated/ — retired one-off drivers

These are the per-turn one-off bench drivers that the **driver explosion**
produced. Every CAPABILITY they had is preserved by the consolidated spine
(`scripts/bench/parity.sh` + `scripts/bench/oracle.sh`); they are kept here (not
deleted) so banked numbers / transcript references still resolve. **Do not write
new drivers here — add a `--kind` to `oracle.sh` instead.**

## Replacement map

| retired driver(s) | replaced by | notes |
|-------------------|-------------|-------|
| `guest_ceiling.sh` + (its run pair) | `oracle.sh --kind ceiling` | decode-removed FLOOR via bypass replay (byte-exact). |
| `guest_clean_only.sh` + `run_clean_only.sh` | `oracle.sh --kind clean-only` | SEEDED clean-engine ceiling (masks-binder; SHA-NOT-CHECKED). |
| `guest_engine_isolation.sh` + `run_engine_isolation.sh` | `oracle.sh --kind engine-isolation` | ISA-L engine oracle ("ocl_cf"); byte-exact; fallbacks==0 asserted. |
| `guest_same_sink_floor.sh` + `run_same_sink_floor.sh` | `oracle.sh --kind same-sink` | production-knob byte-exact control / floor. |
| `guest_fulcrum_capture.sh` + `run_locked_fulcrum.sh` | `parity.sh --fulcrum` → `fulcrum_total_capture.sh` → `fulcrum_total.py` | trace + counter-sidecar capture, window-absent-preserving. |
| `guest_step0.sh` + `run_step0.sh` | (probe; not yet a `--kind`) | STALL_RESIDENCY_PROBE + consumer-block decompose. The probe knobs (`GZIPPY_STALL_RESIDENCY_PROBE`, `GZIPPY_SEED_NO_*`) still exist; re-add as `oracle.sh --kind step0` if needed. Analyzer: `consumer_block_decompose.py` (here). |
| `bmi2_ceiling_ab.sh` | `oracle.sh --kind perturb --slow …` | BMI2 A/B is a behavior/perturbation knob run. |
| `rss_vs_t.sh` + `_rss_regression.py` | (RSS-vs-T sweep) | resident-set sweep; re-add as a `--kind` if revisited. |
| `lag_causality_sweep.sh` | `oracle.sh --kind perturb` | lag-causality is a slow-injection sweep. |
| `_parity_baseline_capture.sh` | `parity.sh` | superseded baseline capture. |
| `_c2_advisor_owed.sh`, `_oclcf_overlap_bound.sh` | `oracle.sh` (`--kind perturb` / `GZIPPY_PERFECT_OVERLAP`) | one-off advisor scratch drivers. |
| analyzers `_plateau_median.py`, `project_wall.py` | (kept here) | only referenced by the retired drivers above. |

## Why consolidated

Each retired `run_*.sh` froze the host by hand and each missed a *different*
noisy neighbor — the root cause of load-artifact numbers. The spine now brackets
EVERY run with the single host freeze (`scripts/bench/host/bench-lock.sh`, which
freezes ALL noisy LXCs and verifies quiet) via `lib_hostlock.sh`, so freezing is
no longer a per-driver responsibility that can be forgotten or done partially.
