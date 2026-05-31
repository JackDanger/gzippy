# Wall-parity scoreboard — the trustworthy progress signal

**Goal:** gzippy wall == rapidgzip wall (ratio **1.0×**) on the workload matrix.
**Instrument:** `scripts/whole_view.sh` section 1 — sha-verified, interleaved, best-of-7, self-tested.

## THE verdict rule (one line; the advisor-hardened version)
> **Progress ⟺ the sha-verified gzippy *ABSOLUTE* MB/s rose AND the interleaved best-of-7 ratio fell by MORE than its sample-spread, on a frozen host.**
> Everything else — refuted levers, "specificity", IPC, "it's a floor" — is *search*, not progress.

Why absolute too, not just ratio: a ratio can fall because **rapidgzip got slower** (box load), logging a fake gzippy "win" — the exact silent-instrument failure that burned this project. A row is **WIN** only if `gzippy_mbps ↑` AND `ratio ↓ > spread`. If ratio fell but gzippy abs is flat → **"rival regressed (load)" — NOT progress.**

## Row protocol (a row is INCONCLUSIVE unless all hold)
1. interleaved A/B (both binaries alternating, one run) — kills per-run jitter.
2. best-of-N, **N ≥ 7** (min, not mean — load only *adds* time).
3. sha-verified output on both sides.
4. **frozen host** if the claimed Δ < the inter-run spread (noise floor here ≈ **15%**, the 1.39–1.62 swing).
5. log the **sample spread**; Δ < spread ⇒ TIE, full stop.
6. self-test: a binary vs *itself* must read 1.0 ± spread, else the harness is lying → void.

## Trajectory (silesia-large, T8)

| date | commit / branch | gzippy MB/s | rapidgzip MB/s | ratio | spread | verdict |
|---|---|---|---|---|---|---|
| 05-30 | baseline (frozen) | ~1370 | ~2105 | **1.537×** | — | start |
| 05-30 | `69202e4` back-ref inline | **↑ (+13.7% same-run A/B)** | ~flat | **1.39×** | ~frozen | **WIN** (gzippy abs ↑ AND ratio ↓; controlled OLD-vs-NEW same run) |
| 05-30 | copy-collapse (reverted) | flat | flat | 1.40× | — | TIE (Δ<spread) |
| 05-30 | `feat/consumer-postprocess-pump` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _pending wall A/B_ |

## Convergence axis (replaces the gameable "kill count")
- Every candidate carries, **on paper before attack**, a `predicted_wall_ceiling ≥ remaining_gap` with its assumed fraction-on-critical-path. A lever that can't clear its own Amdahl bound is dead before it costs a work-stretch. (This would have pre-killed "marker-decode 82%" — it's 28% of WALL, ceiling < the 1.5× gap.)
- Convergence = the **measured critical-path bound is trending down across rows**, not the kill count.

## STALL tripwire (the one thing to watch)
> **2 consecutive *effortful* wall-A/Bs (build+measure spent) with Δ < spread ⇒ STALLED ⇒ mandatory RE-LOCALIZE (re-measure where the wall actually is) before attacking a 3rd lever.**
Unit = effortful work-stretches, NOT levers (a cheap paper-Amdahl kill doesn't tick it) and NOT wall-clock hours.

## Current candidate — reconciled with the ledger (advisor flag)
`feat/consumer-postprocess-pump` is a **consumer-SCHEDULING** lever (eager successor `apply_window` pumped to workers DURING the in-order stall, so the consumer's serial wait shrinks) — i.e. the ledger's structural-consumer OPEN LEVER (lines 58-64), **not** decode-mechanism work (which is wall-dead). Discriminator confirmed ~100ms of the consumer's 166ms stall is ready-work-going-unused. Gated on the wall A/B; trip-wire armed if `wait.block_fetcher_get` drops but the wall doesn't.
