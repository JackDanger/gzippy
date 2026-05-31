# Refreshed plan — closing the gzippy → rapidgzip x86 single-member wall gap

_2026-05-30, after the consumer-pump TIE and TWO rounds of advisor-validated re-localization
(my "scheduling-bound" read was wrong; the agent's "32-MiB-drift" read was also wrong; the
source-grounded diagnosis below is §1.3-1.5)._

## 0. Goal + where we are
- **Goal:** gzippy wall == rapidgzip (ratio **1.0×**) across the matrix; pure-Rust the sole decode path.
- **Now:** x86 T8 single-member = **1.39×** (from 1.537×, one win). **arm64 = fastest decoder every thread count (shipped).** Multi-member, BGZF T≥2, single-T1, incompressible — parity/win. Open cell: **x86 single-member T2–T16.**
- **Progress metric (only verdict):** `whole_view.sh` §1 — sha-verified, interleaved best-of-7, gzippy **absolute** MB/s + spread + ratio (`plans/wall-progress.md`). WIN ⟺ gzippy abs ↑ AND ratio ↓ > spread.

## 1. THE DIAGNOSIS (source-grounded, advisor-validated)
1. **Pure memory-stall IPC** — instructions equal, parallelism equal, IPC 1.60 vs 2.13.
2. **The wall is the in-order CONSUMER's serial path.** Workers ~8.5 worker-sec / 231ms wall = ~37× overlapped slack → aggregate decode-rate AND the 30%-of-cycles memmove are **wall-DEAD by construction.** Wall ≈ consumer tid-1 (busy 340ms + wait 166ms).
3. **The 166ms wait = the consumer eating ON-DEMAND heavy-chunk decode** (4 stalls, ~44–65ms). The chunk's own decode runs start-to-end inside the stall.
4. **WHY (the real divergence, src-confirmed): the consumer THROWS AWAY range-valid prefetches.** rapidgzip accepts a partition-seeded prefetch by **range** (`encoded ≤ offset ≤ max`, `ChunkData.hpp:397-402`, `GzipChunk.hpp:712-734`). gzippy has the identical range type (`chunk_data.rs:336-339`) but `chunk_fetcher.rs:1152` requires **exact equality** `max_acceptable_start_bit == next_block_offset` — comment: *"gzippy doesn't yet enforce that chain."* So a prefetch that DID decode the right bytes is refused → on-demand re-decode. (The "fixed-32-MiB vs block-finder seeding" story is FALSE — both seed at the same partition grid.) _Confirm via `PREFETCH_REJECT_BY_GUARD ≈ 4` — see §5 measurement._
5. **So the pipeline is NOT "fully ported."** The missing piece is the **chain invariant** (range-acceptance + window cascade-publish) — exactly the ledger's OPEN LEVER (`x86-falsification-ledger.md:58-64`). The same missing invariant starved the consumer-pump (0 tasks). One gap, two failed levers.

## 2. WHAT WE TRIED (falsification record — do not re-attack without new evidence)
**WINS (banked):** back-ref-copy inline (+13.7%, 1.537→1.39×, `69202e4`). · arm64 enablement (fastest, shipped). · incompressible → parity.
**TIE / DEAD (with the measurement):**
- **Consumer post-process pump** (`906908e`) = **TIE** (wall in-noise; 0 tasks; blocked by the missing chain invariant, §1.5). Kept the opportunistic reaping (−15ms `wait.future_recv`, real but wall-neutral).
- FastBootstrap (faster decode) = wall-TIE (overlapped). · branch-mispredict = noise. · copy-collapse/zero-copy/copy-elim = wall-neutral. · buffer-reuse/prewarm/rpmalloc/segmented = regress. · NT-stores = rejected. · overshoot = 2.7%. · chunk-data single-buffer = premise false (both ring). · ring warm-reuse = both alloc fresh. · **inner-loop prefetch/multi-literal = VETOED** (FastBootstrap corpse; the part-bias caught 3×).
- **Range-widen the acceptance guard (naive)** = corrupted output / premature EOF (`chunk_data.rs:1206-1214`) — because the fast path collapses the range (`max=encoded`) and the window chain doesn't guarantee decoded bytes start at `max`. So it needs the FULL invariant, not a guard tweak.

## 3. TOOLS that got us here (reuse)
`whole_view.sh` (self-testing full-system budget + trajectory CSV). · `trace_v2`/`timeline_analyze.py` (consumer critical path). · **the per-chunk stall discriminator + already-instrumented counters** (`on_demand_fetch_count`, `PREFETCH_REJECT_BY_GUARD`, `prefetch_cache_misses`, `is_speculative_prefetch` span flag) — the right way to pin prefetch-miss vs late-prefetch vs guard-reject. · symbolized + cross-tool PEBS. · falsification ledger + wall-progress scoreboard + the Opus advisor (caught the part-bias 3×, the broken oracle, and BOTH wrong re-localizations).

## 4. LESSONS (durable)
- **The part-bias is the recurring failure**; lead via holistic measurements, not inner-loop asm.
- **A discriminator can be wrong (twice this session)** — match the SPECIFIC awaited chunk AND read provenance counters; "starts at stall-start" is tautological for on-demand submits. Validate every re-localization (advisor + source).
- on-critical-path ≠ total-CPU; the sha-verified wall A/B is the only verdict. · Amdahl gate before building. · stall tripwire (2 no-move A/Bs ⇒ re-localize) — it fired correctly.

## 5. THE REMAINING SPACE (re-ranked, advisor-validated)
**FIRST — confirm the cause (one traced run, ZERO new code, before any prefetch work):** read the 4 counters against the 4 `wait.block_fetcher_get` events —
- `PREFETCH_REJECT_BY_GUARD ≈ 4` ⇒ **chain-invariant acceptance miss confirmed** → lever A.
- `on_demand ≈ 4`, reject≈0, `cache_misses≈4` ⇒ keying/dispatch coverage → check the saturation gate (`block_fetcher.rs:692`).
- waited-on spans `is_speculative=true`, start < stall-start ⇒ **late prefetch** → lever A.3 (depth), not the invariant.

**A. Close the CHAIN INVARIANT (first-order; subsumes the dead pump):** anchor chunk N's `max_acceptable_start_bit` at the real found boundary even on the fast path, and guarantee the predecessor's confirmed end lands in `[encoded, max]` so re-anchoring drops no bytes. Then the consumer **accepts range-valid prefetches** (kills the on-demand re-decode) AND can **cascade-publish windows** (resurrects the pump). Real work in the consumer/window-resolution chain. _Kill: `whole_view` §1 gzippy-abs ↑ AND ratio ↓ > spread; `wait.block_fetcher_get` drops._
  - A.2 predict/prioritize heavy chunks; A.3 deeper prefetch window — only if the measurement shows *late prefetch*, not guard-reject; gate on the diagnostic.
**B. Clean-residual (endgame, ~1.17–1.41×):** pure-Rust-vs-ISA-L inner-loop dispatch floor; inline-asm authorized. After A.
**C. Heavy-chunk decode speed:** near-dead (FastBootstrap proved decode-rate overlapped); prefer A (avoid the on-demand decode).
**D. Roofline (CP_res):** academic now; the one GATE still owed.

**Amdahl reality (advisor):** 166ms/231ms ≈ 72% touchable ≫ the gap, BUT irreducible: the first chunk(s) (no predecessor window), the frontier chunk (always genuinely speculative), and the serial window-dependency latency. Realistic reach ≈ **100–130ms**, targeting the ~3-of-4 behind-frontier guard-rejects. Clears the gate; won't reach a perfect 1.0× alone — B is the remainder.

## 6. METHOD going forward
Lead with `whole_view` + the trace + the provenance counters (the WHOLE). Every candidate: paper Amdahl ceiling ≥ gap BEFORE building. Verdict only on §1. Advisor for every judgement call; 2 no-move A/Bs ⇒ re-localize. Update `wall-progress.md` per lever.
