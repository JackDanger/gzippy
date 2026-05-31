# Complete learnings — gzippy → rapidgzip parity campaign (x86 single-member decode)

_2026-05-30. The full knowledge body. Companions: `refreshed-plan.md` (forward plan),
`x86-falsification-ledger.md` (the kills), `wall-progress.md` (the trajectory + verdict rule)._

---

## 0. Competitive position (ground truth, per cell)
- **arm64: gzippy is THE fastest gzip decoder, every thread count** (1.2–2.8× faster than rapidgzip T1–T8). Shipped (`089e1d3`). The pivot that won it: ENABLE the pure-Rust parallel path on arm64 (143 x86-gates → build.rs cfg aliases), not inner-loop tuning.
- **x86 multi-member, BGZF T≥2, single-member T1, incompressible:** parity or win.
- **THE OPEN CELL: x86 single-member T2–T16 = 1.39× slower** than native rapidgzip (down from 1.537× via one win).
- **Production decode path (x86):** ISA-L (C) for clean chunks + pure-Rust `deflate_block` for the window-absent bootstrap, inside the parallel single-member pipeline (prefetcher → worker pool → in-order consumer → window map). arm64 = pure-Rust throughout.

## 1. THE DIAGNOSIS — the full causal chain (every step measured, not assumed)
The single most important result of the campaign: **we traced the 1.39× from symptom to root cause, and every intermediate "it's X" was tested.**

1. **The gap is pure memory-STALL IPC, not work or parallelism.** `whole_view.sh` + `perf stat`: instructions EQUAL (gzippy 11.27B vs rapidgzip 11.09B = 1.02×), parallelism EQUAL (CPUs-used 6.30 vs 6.47), **IPC 1.60 vs 2.13.** ⇒ gzippy issues the same instructions, equally parallel, but stalls ~1.48× more. _Killed: "gzippy does more work", "gzippy is less parallel."_
2. **The stalls are LATENCY, not bandwidth.** Demand-DRAM ≈ 540 MB/s ≪ roofline; `tma_memory_bound` 30.6% vs 15.1%; **11× more demand-L3 misses** (2.36M vs 0.21M), amplifying down the hierarchy (1.5×→6×→11×). _Killed: NT-stores / any traffic-reduction lever._
3. **The WALL is the in-order CONSUMER's serial critical path; the worker side is overlapped SLACK.** The trace: workers do **~8.5 worker-seconds in a 231ms wall ≈ 37× oversubscribed.** Therefore aggregate decode-rate AND the 30%-of-cycles memmove are **wall-DEAD by construction** — a sink can be 33% of CPU and 0% of wall. Wall ≈ consumer tid-1 (busy 340ms + wait 166ms). _This is THE reframe. It explains why every "make the worker faster / move fewer bytes" lever TIE'd, and it's the lesson the user had to force three times: optimize the WHOLE, the consumer chain — not parts of the overlapped worker._
4. **The consumer's 166ms wait = ON-DEMAND decode of 4 chunks** (not unpumped post-processing — my first discriminator was tautological and WRONG). Verbose: `Fetched On-demand: 4`.
5. **3 of those 4 are GUARD-REJECTS** — the prefetch decoded the right bytes and the consumer THREW IT AWAY. `Prefetch guard-rejects: 3`. The acceptance guard (`chunk_fetcher.rs:1152`) demands `max_acceptable_start_bit == next_block_offset` (exact); rapidgzip accepts a RANGE (`encoded ≤ offset ≤ max`). gzippy is **missing the CHAIN INVARIANT** (comment in source: "gzippy doesn't yet enforce that chain"). The 4th is the irreducible frontier/first chunk.

**Root cause, one sentence:** same work, equally parallel, latency-stalled; the wall is the in-order consumer re-decoding on demand the chunks whose range-valid prefetch it refuses for lack of a chain invariant.

## 2. THE INSTRUMENTS we built (and exactly what each revealed)
- **`whole_view.sh`** — full-system budget in one command (wall + IPC/instructions/parallelism + both tools' cycle budgets + gzippy phases), **self-testing** (timer-brackets-process; binary-vs-itself ≈1.0 or it declares the box too noisy), emits an append-only trajectory CSV, states the verdict rule. _Revealed the IPC decomposition + the back-ref memmove anomaly._ It was MIS-FRAMED at first (section 3 ranks CPU-time, which ≠ wall on an overlapped pipeline — the advisor caught it); now banners the regime and labels CPU-time as hypotheses.
- **`trace_v2` (GZIPPY_TIMELINE) + `timeline_analyze.py`** — per-thread busy/wait, the consumer critical path (CP_dep), span call-tree with self-times. _Located the wall on tid 1 (consumer wait 166ms) and proved the worker side is overlapped slack._
- **The per-chunk stall discriminator + provenance counters** (`on_demand_fetch_count`, `PREFETCH_REJECT_BY_GUARD`, `prefetch_cache_misses`, the `is_speculative_prefetch` span flag) — match each `wait.block_fetcher_get` to its SPECIFIC awaited chunk's decode, and read why it wasn't a cache hit. _Pinned the guard-reject (3 of 4). NOTE: the aggregate-count version of this discriminator was WRONG (read "scheduling-bound"); the per-chunk + counter version is right._
- **Symbolized PEBS** (`-C strip=none -C force-frame-pointers=yes`, `perf --call-graph fp`) + **cross-tool PEBS** (gzippy vs rapidgzip side-by-side) — _found the back-ref memmove (gzippy 16.7% vs rapidgzip 3.95%) and bootstrap_with_deflate_block_inner = 28.69%._
- **Falsification ledger** — every kill with the validated measurement, so corpses stay buried.
- **wall-progress scoreboard** — the trajectory + the advisor-hardened verdict rule (WIN ⟺ gzippy ABS ↑ AND ratio ↓ > spread; ratio-only can log "rival got slower" as a fake win) + the stall tripwire (2 effortful no-move A/Bs ⇒ re-localize).
- **The Opus advisor** — caught the part-bias 3×, the broken oracle, the gameable progress metric, AND both wrong re-localizations. The single highest-leverage process tool.
- **FULCRUM** (public JackDanger/fulcrum) — the systemic diffuse-gap tooling that drove the arm64 win.

## 3. EVERY LEVER, in detail
### Wins (banked)
- **Back-ref-copy inline** (`69202e4`, +13.7%, 1.537→1.39×): `emit_backref_ring` called libc `memcpy` (`copy_nonoverlapping`, runtime length) per back-reference; back-refs are short (3–258 u16) so the memcpy size-class DISPATCH dominated (16.7% of cycles vs rapidgzip 3.95% — it inlines into `Block::read`). Replaced with compile-time-const 16-byte chunks + scalar tail. On-wall because it shortened a critical chunk's decode.
- **arm64 enablement** (`089e1d3`): see §0.
- **Incompressible/random100**: closed to parity (`52e2361`/`9724b7f`) — fold parallel-prefix CRC + overlap the single-threaded Huffman tail with the prefix copy.

### Dead (wall-neutral or regressed — with the measurement that killed it; DO NOT re-attack)
- **Consumer post-process pump** (`906908e`) — TIE; enqueued 0 tasks (structurally starved — same missing chain invariant). The opportunistic future-reaping it added IS a real −15ms `wait.future_recv` (kept), but wall-neutral.
- **FastBootstrap** (faster window-absent decoder, banked `5514453`) — 1.7–1.9× faster decode, byte-identical, **wall-TIE** (decode is pipeline-overlapped). The canonical proof that decode-rate is wall-dead.
- **Inner-loop branch-mispredict** (−13% branches) — wall-noise.
- **Copy-collapse / zero-copy drain / copy-elim** — wall-neutral; miss-count stayed flat (37→36M) when a 156 MB/chunk copy was removed (the copies are off-critical-path).
- **Buffer-reuse / prewarm (MADV_POPULATE) / rpmalloc-global / Z-prewarm / segmented buffers** — −300ms / +167% / regress. Faults/sec are EQUAL across engines (so fault-total is the shadow of runtime, not a cause); prewarm serialized faults that previously parallelized.
- **NT-stores** — rejected: the resolve is sequential/bandwidth-shaped and we're 5× under bandwidth (latency-bound), and the prior "5.3× fewer misses" rationale was a different config (clean-import vs production).
- **Bootstrap overshoot** (post-flip clean bytes decoded as markers) — only 2.7% of body.
- **Chunk-data single-buffer port** — premise FALSE: rapidgzip ALSO uses a 128 KiB u16 ring + drain + backward-scan + separate apply pass; gzippy's `deflate_block` is a faithful file:line port of it.
- **Ring lifecycle (warm-reuse)** — both gzippy and rapidgzip alloc the ring fresh per chunk.
- **Inner-loop prefetch-pipeline / multi-literal packed writes** — VETOED before building (the FastBootstrap corpse re-dressed; the part-bias).
- **Naive range-widen of the acceptance guard** — corrupts output (premature EOF) because the fast path collapses the range to `max=encoded` and the window chain doesn't guarantee bytes start at `max`. ⇒ needs the FULL chain invariant, not a guard tweak.

## 4. THE METHODOLOGY / DISCIPLINE (what finally worked)
- **The sha-verified wall A/B is the ONLY verdict.** Every CPU-%, IPC, cache-miss number is a HYPOTHESIS. (The session's worst hours were spent treating CPU-fraction as wall-fraction.)
- **Lead with the WHOLE** (`whole_view` + the trace + provenance counters), never a single part's CPU-%.
- **Amdahl gate:** a lever may be attacked only if its predicted wall-ceiling ≥ the remaining gap, on paper, before building. (Would have pre-killed "marker-decode is 82%" — it's 28% of WALL.)
- **Validate the instrument first** (positive-control) — the broken clean-window oracle had none and corrupted multiple sessions.
- **Measure interleaved + best-of-N + frozen-if-Δ<spread** — the shared box swings the ratio ~15% (1.39–1.62); a Δ below that is a TIE.
- **Stall tripwire:** 2 effortful no-move wall-A/Bs ⇒ STOP and re-localize before a 3rd lever. (It fired correctly after copy-collapse + pump.)
- **Advisor for every judgement call** — it caught what I missed every single time it mattered.

## 5. THE HONEST RECORD OF BEING WRONG (the corrections — the most valuable part)
- **The broken clean-window oracle** silently re-ran the bootstrap it claimed to skip, corrupting "markers vs scaling" across MULTIPLE sessions and CLAUDE.md's decision to rescind the structural port. Found only by an over-falsification audit. Fixed `64eb6df`.
- **The part-optimization bias** — optimizing the inner loop (back-ref copy, prefetch, multi-literal) when the wall is the consumer chain. The user flagged it THREE times before it stuck. The fix: lead via holistic measurements.
- **TWO wrong re-localizations in one day:** (a) my "scheduling-bound" read of the consumer stall — an aggregate decode count that's tautological; (b) the agent's "32-MiB-partition-seed drift" cause — false, both tools seed at the same partition grid. Both were caught only by source-grounded advisor validation. **A discriminator can be confidently wrong; match the SPECIFIC chunk and read provenance counters.**
- **Premature "irreducible floor" declared 3–4 times**, each overturned by the next clean measurement.
- **A gameable progress metric** ("kills + specificity") that would have scored this 15-kills-1-win session as "converging." Replaced by the absolute-MB/s verdict + the Amdahl gate.

## 6. THE REMAINING SPACE (ranked)
**A. Close the CHAIN INVARIANT (first-order, IN FLIGHT `feat/chain-invariant`):** anchor chunk N's `max_acceptable_start_bit` at the real found boundary on the fast path + guarantee the predecessor's end ∈ `[encoded, max]`; then accept the range (`chunk_fetcher.rs:1152`). Kills the 3 guard-reject on-demand re-decodes AND resurrects the pump's window cascade-publish. Reach ≈ 100–130ms of the 166ms (the frontier/first chunk is irreducible). _Verdict pending; confirmed cause (`PREFETCH_REJECT_BY_GUARD = 3`)._
**B. Clean-residual (endgame, ~1.17–1.41×):** the pure-Rust-vs-ISA-L inner-loop DISPATCH floor on the clean path — genuinely hand-asm-vs-Rust; CLAUDE.md authorizes inline-asm. This is the slice that keeps "exact parity in pure Rust" honest, and the remainder after A.
**C. Heavy-chunk decode speed:** near-dead (FastBootstrap proved decode-rate overlapped). Prefer A (avoid the on-demand decode) over speeding it.
**D. Roofline (CP_res):** the never-collected IMC/STREAM bandwidth-vs-latency disambiguation — academic now that the consumer chain owns the wall; the one GATE still formally owed.

## 7. THE DURABLE META-LESSONS
1. **When a working reference implementation exists (rapidgzip's source is in `vendor/`), the gap is a list of structural choices to converge — not a mystery to decompose.** Decomposition was the rabbit hole.
2. **on-critical-path ≠ total-CPU.** On a 37×-overlapped parallel pipeline, the wall is one serial thread; everything else is free. This single fact retro-explains ~15 TIE'd levers.
3. **The wall is the only verdict; the instrument must self-test; the human's "this feels stalled" is usually right.**
4. **The advisor + source-grounded validation catches confident errors that measurement alone reproduces.** Used it for every judgement call.
5. **The campaign's actual output is not (yet) parity — it's a fully-diagnosed, validated frontier: one banked win (1.39×), every dead end mapped, and the exact next lever (the chain invariant) confirmed and in flight.**
