# Wall-parity scoreboard — the ONLY progress metric

**Goal:** gzippy wall == rapidgzip wall (ratio **1.0×**) on the workload matrix.
**Metric:** `scripts/whole_view.sh` section 1 — sha-verified, interleaved, jitter-immune.
**Progress = the ratio drops.** Everything else (IPC, cycle-%, cache-misses) is a *hypothesis*, never a verdict.

## Trajectory (silesia-large, T8)

| date | commit / branch | wall ratio | Δ | what moved it (or why TIE) |
|---|---|---|---|---|
| 2026-05-30 | baseline (frozen) | **1.537×** | — | session start |
| 2026-05-30 | `69202e4` back-ref inline | **1.39×** | **−0.15  WIN (+13.7%)** | inlined the per-back-ref libc memcpy (short copies were paying call dispatch) |
| 2026-05-30 | copy-collapse (reverted) | 1.40× | 0  TIE | overlapped → wall-neutral; reverted |
| 2026-05-30 | `feat/consumer-postprocess-pump` | _TBD_ | ? | porting rapidgzip's eager successor `apply_window` pump (the holistically-confirmed lever) |

## Search state (the second progress axis)
- **~15 levers pruned** (see `plans/x86-falsification-ledger.md`) — each a validated dead-end, not wasted.
- **Converged:** "diffuse gap" → "the WALL is the in-order consumer critical path; worker side is ~37× overlapped slack" (trace + symbolized PEBS + discriminator). Current candidate is specific: the missing consumer post-process pump.

## How to read this (stalled vs progress)
- **Progress (green):** the ratio above dropped since last check, OR a lever was *validly* refuted and the next candidate got *more specific*.
- **Search (yellow):** measuring/analyzing, no wall move yet, but converging on ONE named lever with a wall-A/B planned.
- **STALLED (red) — any of these:**
  1. re-attacking a ledger-DEAD lever;
  2. optimizing a part without a `whole_view` check (the part-bias);
  3. >1 work-stretch with no sha-verified wall-A/B;
  4. "it's a floor" without a floor proof.

## Update rule
Every landed change OR refuted lever → add a row (commit, ratio, Δ, what/why). The ratio column is the trajectory; a flat ratio after a *landed* change = stalled.
