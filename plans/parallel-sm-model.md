# Parallel single-member decode — the quantitative model (advisor-validated 2026-05-31)

The model gzippy's wall obeys, so we stop twiddling knobs blind. Validated by
adversarial advisor ac87bcff; two earlier hand-built versions were wrong (see
"Killed premises").

## System
In-order parallel pipeline. `N` chunks (~13 MB each, silesia 503 MB ⇒ N≈40).
`T` workers decode in parallel; ONE in-order consumer publishes each chunk's
32 KB tail-window + writes output.

A chunk decodes **CLEAN** (fast windowed ISA-L) iff its predecessor's tail-window
is **PUBLISHED** when a worker STARTS it; else **WINDOW-ABSENT** (slow bootstrap →
u16 markers → later resolve). Mode is set at decode-START and locked.

## Two frontiers (over chunk-index vs wall-time)
- **Q(t)** dispatch frontier: chunks workers have STARTED. Rate ≈ `T / d_w`.
- **P(t)** publish frontier: chunks whose tail-window is published — advanced by
  the in-order consumer, and it is a **SERIAL DEPENDENCY CHAIN**: publishing
  chunk i needs chunk i RESOLVED, which needs i−1's resolved tail-window.

## Clean predicate (corrected)
`chunk i clean ⟺ P(t_dispatch(i)) ≥ i−1`. **NOT** `depth = Q−P ≤ 1` — depth is the
wrong frontier (it counts dispatched-but-unpublished; cleanness is about P alone).

## The wall
```
wall ≈ max( worker-bound:  frontier + (N/T)·d_w_eff ,
            publish-chain: frontier + N·L_resolve  )  + tail
```
- `d_w_eff` = decode time/chunk, weighted: d_w for window-absent (~31%), d_c for clean.
- **`L_resolve`** = critical-path latency of ONE publish link = consumer slice
  `ρ_link` (output-write + narrow) **PLUS the serial portion of marker-resolve /
  apply_window that gates the successor.** This is the parameter we have NOT
  measured — and the whole game.
- Catch-up term is **latency `1/L_resolve`**, NOT throughput `1/ρ_link`.

## The worker-bound KNEE (why FastBootstrap TIE'd)
Cutting `L_resolve` lowers the wall ONLY until the worker-bound term becomes the
max. Past that knee, faster resolve stops paying. The model PREDICTS the
FastBootstrap TIE (decode sped, wall flat) instead of being refuted by it — the
prior "slow-down slope = speed-up ceiling" view could not.

## Killed premises (do not resurrect)
- ✗ "clean iff depth ≤ 1" — wrong frontier; use P.
- ✗ "depth ≈ T unconditionally ⇒ 100% window-absent" — depth is regime-specific
  (it's the inter-server buffer occupancy = min(buffer_cap, ∫(μ_w−μ_c))), not a law.
- ✗ "rapidgzip is 88% clean / P≈Q / depth≈0" — **FALSE.** rapidgzip reports
  31.25% replaced-marker symbols; gzippy 31.97% — they MATCH. rapidgzip is ALSO
  mostly window-absent-armed. So "move publish to the worker to get depth≈0" is
  NOT what rapidgzip does and is NOT the lever. This fact refutes the
  bound-depth/eager-publish framing.
- ✗ catch-up inequality with `1/ρ_link` (throughput) — must be `1/L_resolve` (chain latency).

## The lever (what the model + the matched 31% fraction imply)
Since BOTH tools are ~31% window-absent, rapidgzip wins by a **smaller
`L_resolve`** — its unified decoder resolves markers→bytes at ISA-L class vs
gzippy's slow pure-Rust window-absent path. Lever = **reduce `L_resolve`**
(window-absent resolve latency), bounded by the worker-bound knee. Consistent
with the +100%-bootstrap → +30%-wall causal result (resolve latency is ON the
publish chain) and with [[project_t8_saturated_pool_diag_2026_05_30]].

## Parameters to MEASURE (per tool, x86 production path, CPU-pinned, clean)
| param | meaning | measurement |
|---|---|---|
| N | chunk count | counter / chunk-index range |
| chunk bytes | for d_c,d_w | ISIZE/N |
| d_w | window-absent decode latency/chunk | `worker.bootstrap` span median |
| d_c | clean decode latency/chunk | `worker.decode_chunk` span on clean chunks |
| **L_resolve** | per-link publish latency (THE missing one) | **inter-publish gap `t_publish(i)−t_publish(i−1)`** via a new per-chunk `window.published` span on the consumer — overlap/serialism already baked in |
| ρ_link | consumer-only slice (diagnosis) | already 4.5ms (output-write+narrow) |
| T | workers | config |
| frontier | startup before steady state | `t_publish(0)` |
| tail | drain after last dispatch | last `t_publish` → EOF |

**Predictive check:** compute wall_pred = max(worker-bound, publish-chain) + tail;
must match observed wall within noise. The discriminator is whether
`N·L_resolve ≈ wall` (publish-chain binds → L_resolve is the lever) or the
worker-bound term binds (→ more workers / faster bulk d_w).

**The point of copying rapidgzip:** measure ITS {d_w, d_c, L_resolve, frontier,
tail} with the SAME Fulcrum instrument, so the gzippy-vs-rapidgzip parameter
DELTA names the lever and its magnitude — not a guess.
