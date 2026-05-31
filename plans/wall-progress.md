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
| 05-30 | `feat/consumer-postprocess-pump` | flat | flat | 1.39× | in-noise | **TIE** — pump enqueued 0 tasks; premise falsified (stalls are on-demand decode of guard-rejected prefetches, not unpumped post-processing). 2nd TIE in a row ⇒ tripwire fired ⇒ re-localized. |

## Convergence axis (replaces the gameable "kill count")
- Every candidate carries, **on paper before attack**, a `predicted_wall_ceiling ≥ remaining_gap` with its assumed fraction-on-critical-path. A lever that can't clear its own Amdahl bound is dead before it costs a work-stretch. (This would have pre-killed "marker-decode 82%" — it's 28% of WALL, ceiling < the 1.5× gap.)
- Convergence = the **measured critical-path bound is trending down across rows**, not the kill count.

## STALL tripwire (the one thing to watch)
> **2 consecutive *effortful* wall-A/Bs (build+measure spent) with Δ < spread ⇒ STALLED ⇒ mandatory RE-LOCALIZE (re-measure where the wall actually is) before attacking a 3rd lever.**
Unit = effortful work-stretches, NOT levers (a cheap paper-Amdahl kill doesn't tick it) and NOT wall-clock hours.

## Current candidate (advisor-validated, src-grounded) — CLOSE THE CHAIN INVARIANT
The pump TIE re-localized the wall: the consumer eats **on-demand heavy-chunk decode** because it **throws away range-valid prefetches** — `chunk_fetcher.rs:1152` demands exact equality (`max_acceptable_start_bit == next_block_offset`) where rapidgzip accepts a range (`encoded ≤ offset ≤ max`). The missing **chain invariant** blocks BOTH the prefetch-accept AND the pump's window cascade-publish (one gap, two dead levers — ledger OPEN LEVER 58-64). Lever: anchor `max_acceptable_start_bit` at the real found boundary on the fast path + guarantee the predecessor's end lands in `[encoded, max]`.
**BEFORE coding:** one traced run, confirm `PREFETCH_REJECT_BY_GUARD ≈ 4` (vs `on_demand`/`is_speculative` = late-prefetch instead). Reach ≈ 100–130ms (frontier/first-chunk irreducible). See `plans/refreshed-plan.md`.
