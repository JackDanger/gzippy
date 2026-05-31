# Complete learnings — gzippy → rapidgzip parity campaign (x86 single-member decode)

> **⚠️ CORRECTION BANNER (2026-05-31, Opus meta-audit). Read before trusting any
> lever below.** Two ideas this campaign repeatedly entertained are MEASURED-WRONG:
> (1) **"delete the marker machinery / converge by removing deflate_block+apply_window"**
> — FALSE: traced rapidgzip carries the SAME machinery (its own counters: 31.25%
> replaced-marker symbols, 0.113s apply-window) and gzippy's arming is a byte-for-byte
> vendor port; deleting it DIVERGES. (2) **"speed the inner Huffman / window-absent
> marker decode to win the wall"** — DEAD: FastBootstrap sped it 1.7–1.9× → wall TIE
> (frozen, N=11); decoder slice ceiling ~14% (`lever-selection-gate.md`). The wall is
> the **structural/scheduling slice (~86%)**: dispatch, prefetch feeding, in-order
> consumer, thread-scaling fill-factor. VERDICT instrument is **`fulcrum coz` (causal)**,
> not biasable `critpath`/`flow --whatif`; gate every lever on a frozen causal run +
> production-path assertion + ledger check BEFORE coding. See the CORRECTION in
> `CLAUDE.md` Goal section and the RETRACTION at the top of `wall-progress.md`.

_2026-05-31. The full knowledge body, advisor-audited for completeness. Companions:
`refreshed-plan.md`, `x86-falsification-ledger.md`, `wall-progress.md`, and the memory notes
(`project_t8_saturated_pool_diag_2026_05_30`, `project_wall_is_consumer_critical_path`,
`feedback_perf_measurement_discipline`, `project_copies_wall_neutral`)._

> **READ FIRST — the central unresolved tension.** This campaign produced TWO competing
> decompositions of the x86 gap that are **not yet reconciled** (§1.6). The most rigorous
> (host-frozen) measurement says the **clean pipeline** is the dominant 1.43× (bootstrap only
> 1.15×); the latest trace says the wall is the **in-order consumer chain** (a prefetch the
> consumer throws away). The chain-invariant lever now in flight is the discriminating
> experiment. Do not read §6.A as settled — it is the test, not the answer.

---

## 0. Competitive position (ground truth, per cell)
- **arm64: gzippy is THE fastest gzip decoder, every thread count** (1.2–2.8× faster than rapidgzip, T1–T8). Shipped (`089e1d3`). The winning move was ENABLEMENT (pure-Rust parallel path on arm64; 143 x86-gates → build.rs cfg aliases), not inner-loop tuning.
- **x86 multi-member, BGZF T≥2, single-member T1, incompressible:** parity or win.
- **THE OPEN CELL: x86 single-member T2–T16.** At T8 ≈ **1.399×** slower than native rapidgzip (`69202e4`, down from 1.537×). T2 also lost; **lowering the T≥4 parallel gate was tried and REGRESSES** (ledger H4) — so T2 is its own sub-problem, not a gate tweak.
- **Production decode path (x86):** ISA-L (C) for clean chunks + pure-Rust `deflate_block` for the window-absent bootstrap, inside the parallel single-member pipeline (prefetcher → worker pool → in-order consumer → window map). arm64 = pure-Rust throughout.

## 1. THE DIAGNOSIS — the causal chain, with measurements AND the contradiction it must own
Every step below names the run that established it. **Caveat the whole section:** on the shared box, absolutes swing ~2× with load and can **flip the sign of an attribution** (same binary read 1217 vs 1712 MB/s under different load) — so only host-frozen or interleaved-relative numbers are load-safe, and several of this campaign's reversals were box-load artifacts (§4).

1. **The gap is memory-STALL IPC, not work/parallelism** — _source: `whole_view.sh` + `perf stat`, THIS session (unfrozen, best-of-7 interleaved; NOT the frozen run in §1.6)._ instructions ≈ equal (11.27B vs 11.09B), parallelism ≈ equal (CPUs-used 6.30 vs 6.47), **IPC 1.60 vs 2.13.** _(The frozen run §1.6 reads slightly different absolutes — 11.82B/10.97B, IPC 1.36/2.09 — same conclusion, different load.)_ Killed: "gzippy does more work," "gzippy is less parallel."
2. **Latency, not bandwidth** — _whole_view production run._ demand-DRAM ≈ 540 MB/s ≪ roofline; `tma_memory_bound` 30.6% vs 15.1%. **L3-miss ratios are CONFIG-SPECIFIC and even reverse:** production-vs-production demand-L3 = **11× MORE** (gzippy 2.36M vs 0.21M); production LLC-loads = 3.08× more; but clean-import-vs-rapidgzip-index (window-export tool) = **5.3× FEWER**. The NT-store kill rests on the latency fact (sequential resolve, 5× under bandwidth) — _note the ledger states the SAME kill using the opposite "5.3× fewer" config number; both are true, scoped to different configs._
3. **The WALL is the in-order CONSUMER's serial path; the worker side is overlapped SLACK** — _source: `trace_v2`/`timeline_analyze.py`, span-sum._ total worker-busy ≈ 8.5 s vs a 0.231 s wall ⇒ **worker-busy is ~37× the wall** (a slack ratio; per-worker busy/wall ≈ 4.6×). ⇒ aggregate decode-rate AND the 30%-of-cycles memmove are **wall-DEAD by construction.** Wall ≈ consumer tid-1 (busy 340ms + wait 166ms). _This is THE reframe — it retro-explains ~15 TIE'd levers, and it's the lesson the user forced three times._
4. **The 166ms consumer wait = on-demand decode of 4 chunks** — _source: `GZIPPY_VERBOSE` `Fetched On-demand: 4` + the per-chunk discriminator (NOT the aggregate-count one, which was tautological and read "scheduling-bound" — wrong)._
5. **3 of those 4 are GUARD-REJECTS** — _`GZIPPY_VERBOSE` `Prefetch guard-rejects: 3`._ The prefetch decoded the right bytes; the consumer REFUSED it because `chunk_fetcher.rs:1152` demands exact equality (`max_acceptable_start_bit == next_block_offset`) where rapidgzip accepts a RANGE (`encoded ≤ offset ≤ max`). gzippy is missing the **chain invariant** ("gzippy doesn't yet enforce that chain", in source). The 4th is the irreducible frontier/first chunk. _This is the confirmed CAUSE; the FIX is unbuilt (§6.A)._
6. **THE UNRECONCILED CONTRADICTION (own this).** The host-FROZEN ground truth (cgroup-frozen, load 0.2, best-of-7, pinned — the campaign's most rigorous run; `project_t8_saturated_pool_diag` 200–214, `ledger:10-11`): gzippy-prod **1684** | gzippy-clean-oracle (bootstrap removed) **1938** | rapidgzip **2770** MB/s ⇒ **bootstrap costs only 1.15×; the CLEAN PIPELINE is the dominant 1.43×**, attributed there to the "pure-Rust-vs-ISA-L inner-loop **dispatch floor**." That attribution (clean-DECODE floor) and this session's (consumer-CHAIN) are NOT the same lever. They CAN be consistent (the consumer chain is inside the clean pipeline), but they disagree on WHERE the 1.43× lives — decode dispatch vs prefetch-acceptance. **The chain invariant in flight is the discriminator:** a ~100–130ms wall drop ⇒ consumer-chain was dominant (this session right); a TIE ⇒ the clean-decode dispatch floor (frozen ledger) is the real story and B (§6) becomes first-order.

## 2. THE INSTRUMENTS we built (and what each revealed)
- **`scripts/whole_view.sh`** — one-command full-system budget (wall + IPC/instr/parallelism + both tools' cycle budgets + gzippy phases), self-testing (timer-brackets-process; binary-vs-itself ≈1.0 or it declares the box too noisy), append-only trajectory CSV, the verdict rule. Revealed §1.1–1.2 and the back-ref memmove. _Was first MIS-FRAMED (section 3 ranks CPU-time ≠ wall on an overlapped pipeline; advisor caught it); now banners the regime._
- **`scripts/measure.sh`** — the canonical interleaved, jitter-immune relative harness (the discipline memory's named tool: "never quote a bare absolute"). The basis for every wall A/B.
- **`trace_v2` (`GZIPPY_TIMELINE`) + `timeline_analyze.py`** — per-thread busy/wait, the consumer critical path, span call-tree with self-times. Located the wall on tid 1.
- **The per-chunk stall discriminator + provenance counters** (`on_demand_fetch_count`, `PREFETCH_REJECT_BY_GUARD`, `prefetch_cache_misses`, the `is_speculative_prefetch` span flag, `GZIPPY_VERBOSE`) — pinned the guard-reject (3 of 4). _The AGGREGATE-count version of this was WRONG; the per-chunk + counter version is right._
- **The FIXED clean-window oracle** (`GZIPPY_CLEAN_WINDOW_ORACLE`) + **`GZIPPY_WINDOW_EXPORT/IMPORT`** (branch `perf/clean-pipeline-window-export`) — the way to measure the clean pipeline in isolation (the frozen ground truth §1.6 and "the cleanest measurement of the session" came from these). _The earlier version of the oracle was SILENTLY BROKEN (§5) — these are the repaired/replacement instruments._
- **Symbolized PEBS** (`-C strip=none -C force-frame-pointers=yes`, `perf --call-graph fp`) + **cross-tool PEBS** — found the back-ref memmove (gzippy 16.7% vs rapidgzip 3.95%) and `bootstrap_with_deflate_block_inner` = 28.69%.
- **FULCRUM** (public JackDanger/fulcrum, PR #117) — a causal-mechanistic pipeline profiler (Coz virtual-speedup + critical-path + PEBS), 4/4 ground-truth self-checks; used on BOTH arches (drove the arm64 win AND ranked bootstrap as the x86 lever, score 0.330, 91% on critical path).
- **Falsification ledger**, **wall-progress scoreboard** (verdict rule + stall tripwire), and **the Opus advisor** — the advisor caught the part-bias 3×, the broken oracle, the gameable metric, both wrong re-localizations, AND this doc's frozen-truth contradiction.

## 3. EVERY LEVER (wins + dead, each with its killing measurement — do not re-attack)
### Wins (banked)
- **Back-ref-copy inline** (`69202e4`, +13.7%, 1.537→**1.399×**): `emit_backref_ring` paid a libc `memcpy` size-class DISPATCH per short back-ref (16.7% of cycles vs rapidgzip 3.95%); replaced with inlined const-chunk copy. On-wall (shortened a critical chunk's decode).
- **arm64 enablement** (`089e1d3`); **incompressible/random100** → parity (`52e2361`/`9724b7f`).
### Dead / TIE (the measurement that killed each)
- **Consumer post-process pump** (`906908e`) — TIE; 0 tasks (starved by the same missing chain invariant). Kept its opportunistic reaping (−15ms `wait.future_recv`, wall-neutral).
- **FastBootstrap** (`5514453`) — 1.7–1.9× faster window-absent decode, byte-identical, **wall-TIE**. The canonical proof decode-rate is wall-dead.
- **Inner-loop branch-mispredict** (agent ad456145, `ledger:27`) — −13.3% branches, −3.8% insns, byte-exact, **wall-noise**; TopdownL1 co-limited (BE31/Ret28/BadSpec26). **Two side-findings to keep:** (a) a work-stealing driver lifted clean 2065→~2654 MB/s (clean residual shrank to ~1.17×); (b) in-process clean decode 4174 MB/s vs full-process 2265 ⇒ **~half the full-process wall is non-decode reassembly/output-write** (a structural target the consumer-chain story is consistent with).
- **Marker-VOLUME reduction / mid-decode window re-check** (agent a038087b, `ledger:28`) — kill-test: **0% of the 156MB bootstrap body is post-predecessor-window** (validated probe control 100% vs real 0%). Distinct from the 2.7% overshoot below; the more decisive volume kill.
- **Bootstrap overshoot** (post-flip clean bytes as markers) — 2.7% of body. **Bootstrap distance-LUT / IsalDistCodePure** (`ledger:22`) — +2% only, plus a latent crash-fix worth landing.
- **Copy-collapse / zero-copy drain / copy-elim** — wall-neutral; miss-count flat (37→36M) when a 156MB/chunk copy was removed. **Drain ring→Vec** as its own row: single-thread 1.06×, but the **8-thread A/B was CONFOUNDED** (drain+flip in one change) and discarded — a methodology data point.
- **u8 / sparse-journal markers** (`feat/unify-bootstrap-fast-decoder`, a major multi-agent build) — **−60% REGRESSION**; FALSIFIED the "u16-WIDTH is THE lever" thesis with a mechanism: taint_journal 21.25M entries, dirty_intervals ~735K/chunk (O(n²) dense propagation), slow-path bytes DOUBLED 156→296MB.
- **Buffer-reuse / prewarm / rpmalloc / segmented** — −300ms / +167% / regress; faults/sec EQUAL across engines (fault-total is the shadow of runtime), prewarm serialized faults that previously parallelized.
- **NT-stores** — rejected (latency-bound, 5× under bandwidth).
- **Chunk-data single-buffer port** — premise FALSE (rapidgzip ALSO uses a 128KiB u16 ring + drain + backward-scan + separate apply; gzippy is a faithful file:line port). **Ring warm-reuse** — both alloc fresh.
- **scan_candidate full-decode-per-candidate** (t8 note audit) — flagged as a LARGE worker-CPU sink (~1600ms summed): gzippy validates speculative block boundaries by full-decoding each candidate; a cheap boundary validator is a NAMED forward lever (but worker-side ⇒ likely overlapped; gate on a wall A/B).
- **Inner-loop prefetch-pipeline / multi-literal** — VETOED before building (FastBootstrap corpse). **Naive range-widen of the acceptance guard** — corrupts output (premature EOF); needs the FULL chain invariant.

## 4. THE METHODOLOGY / MEASUREMENT ENVIRONMENT (the discipline that finally worked)
- **The sha-verified wall A/B is the ONLY verdict.** CPU-%, IPC, cache-miss are HYPOTHESES.
- **The shared box (`ssh -J neurotic root@REDACTED_IP`) is dangerous:** absolutes swing ~2× with load and **FLIP attributions** (prod 1217 vs 1712, same binary). Use `measure.sh` (interleaved) or **host-freeze** (`cgroup.freeze` lxc/105 plex + 111 frigate, pre-armed unfreeze, <30min, user-authorized) for clean absolutes. **Pinning confound:** `taskset 0,2,4,6…` is 1 thread/P-core; "T16 there" is 2× oversubscribed — it faked a T16 plateau.
- **Build-config trap:** debug-info-in-release Cargo.toml CONTAMINATED an entire prior session (made the gap read 41% when a clean rebuild showed 10%). Always confirm the bench build's profile.
- **Lead with the WHOLE** (whole_view + trace + provenance counters), never a part's CPU-%. **Amdahl gate** (ceiling ≥ gap, on paper, before building). **Validate the instrument first** (positive-control). **Stall tripwire** (2 effortful no-move A/Bs ⇒ re-localize). **Advisor for every judgement call.**

## 5. THE HONEST RECORD OF BEING WRONG (the campaign's defining feature)
**The diagnosis flip-flopped 6+ times** (the discipline memory's #1 lesson): **markers → pipeline → bootstrap → memory(allocator) → branch-mispredict → non-decode-overhead → consumer-chain**, each asserted then refuted on careful (often host-frozen) measurement. The clean linear chain in §1 is the *current* synthesis, not how it was found. Specific corrections:
- **The broken clean-window oracle** silently re-ran the bootstrap it claimed to skip, corrupting "markers vs scaling" across MULTIPLE sessions (and CLAUDE.md's rescind of the structural port). Found only by an over-falsification audit. _Fix commit: `64eb6df` per the memory / `b757038` per the t8 note — the two sources disagree; both may be un-merged. PIN THIS._ A separate confound: the oracle ran 45% SLOWER via a two-pass CLEAN_WINDOW_ORACLE (361 vs 663).
- **The part-optimization bias** — optimizing the inner loop when the wall is the consumer chain; the user flagged it THREE times.
- **TWO wrong re-localizations in one day:** (a) my aggregate-count "scheduling-bound" read; (b) the agent's "32-MiB-partition-drift" cause (false — both tools seed at the same grid). Both caught only by source-grounded advisor validation.
- **Premature "irreducible floor" declared 3–4 times**, each overturned by the next clean run.
- **A gameable progress metric** ("kills + specificity") that would have scored this 15-kills-1-win session "converging"; replaced by the absolute-MB/s verdict + Amdahl gate.

## 6. THE REMAINING SPACE (ranked; A and B are both live until the chain-invariant test resolves §1.6)
**A. Close the CHAIN INVARIANT (IN FLIGHT `feat/chain-invariant`; PROJECTED, UNBUILT):** anchor chunk N's `max_acceptable_start_bit` at the real found boundary on the fast path + guarantee predecessor end ∈ `[encoded, max]`; then accept the range. Confirmed CAUSE (`PREFETCH_REJECT_BY_GUARD=3`); the FIX is unmeasured. **Projected** reach ≈ 100–130ms of the 166ms (frontier/first chunk irreducible) — this is a paper-Amdahl projection, not a banked result.
**B. Clean-pipeline / clean-decode residual (the FROZEN ground truth's dominant 1.43×):** if A ties, THIS is first-order — the pure-Rust-vs-ISA-L inner-loop DISPATCH floor on the clean path, plus the "half the full-process wall is non-decode reassembly/output" finding (§3). Inline-asm authorized. The slice that keeps "exact parity in pure Rust" honest.
**C. scan_candidate cheap-boundary-validator + T2-specific sub-problem.** **D. Roofline (CP_res):** the never-collected IMC/STREAM bandwidth-vs-latency GATE — academic unless A and B both stall.

## 7. CORRECTNESS / TESTING DISCIPLINE (what gated every "byte-identical" claim)
Every lever was gated on: the **byte-exact silesia differential** (single + multi-member + gzip-9) against libdeflate/flate2 oracles; **811+ lib tests**; the **marker-propagation differential** (a lasting artifact); **fuzz diverged=0** over N interleaved runs; and the rule that **new inner-Huffman levers MUST ship the real-corpus silesia differential in the same commit** (`feedback_real_corpus_test_with_lever` — synthetic 729-case differentials over-trust; a T3 bug shipped despite passing them). `whole_view.sh` re-checks output sha every run.

## 8. THE DURABLE META-LESSONS
1. **A working reference (rapidgzip's source is in `vendor/`) makes the gap a list of structural choices to converge — not a mystery to decompose.** Decomposition was the rabbit hole.
2. **on-critical-path ≠ total-CPU.** On a ~37×-slack parallel pipeline the wall is one serial thread; everything else is free. Retro-explains ~15 TIE'd levers.
3. **The shared box flips attributions; freeze or go interleaved-relative.** Most flip-flops were measurement, not ideas.
4. **The wall is the only verdict; the instrument must self-test; "this feels stalled" is usually right.**
5. **The advisor + source-grounded validation catches confident errors that measurement alone reproduces** — including the contradiction in THIS document.
6. **The campaign's output is not (yet) parity — it is a fully-mapped frontier:** one banked win (1.399×), every dead end recorded with its kill, the two competing decompositions named, and the discriminating experiment (the chain invariant) in flight.
