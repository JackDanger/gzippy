# TOOL INVENTORY — 2026-06-21 (mined from .claude transcripts + committed tree)

Catalog of every measurement / freeze / causality tool the campaign built or used.
Categories: **A** = measurement, **B** = freeze/quiesce, **C** = causality.
STATUS legend: **LIVE** = committed in current `kernel-converge-A` tree; **LOST** =
referenced in transcripts but absent from current `src/` / `scripts/` (built in a
cleaned worktree); **SUPERSEDED** = replaced by a newer committed tool.

Cross-check method: `git ls-files scripts src` + per-knob `grep -rl "$k" src/`
(0 src files for a knob heavily used in transcripts ⇒ LOST). Transcript dir:
`/home/user/.claude/projects/-Users-jackdanger-www-gzippy-reimplement-isal/`
(607 MB, 1541 .jsonl).

---

## ★ MOST RELEVANT TO CURRENT WORK (silesia-T4 parallel UTILIZATION / per-thread idle / scheduling causality)

### LIVE — committed, ready to use NOW
| tool | cat | what / the discriminator | path | invoke |
|---|---|---|---|---|
| `parallel_sm_tail_metric.py` | A | **THE H-TAIL vs H-KERNEL discriminator.** Reduces a GZIPPY_TIMELINE trace to: `effcores`=Σ(worker decode dur)/drive_wall (avg busy cores; <<T ⇒ schedule slack, ~T ⇒ CPU-bound); `tail_idle`=Σ(wait.future_recv)/wall (consumer blocked on not-ready chunk); `last_wave`; `decode_var` (per-chunk heterogeneity); `order_inv` (out-of-order completion → writer waits). Has BLOCKING Gate-0 self-validation. | `scripts/parallel_sm_tail_metric.py` | `GZIPPY_TIMELINE=/t.json gzippy -dc …` then `python3 …tail_metric.py /t.json` |
| `timeline_analyze.py` | A | Per-thread UTILIZATION summary; critical-path estimate via `wait_for` edges; lock-contention sum per `args.lock`; alloc hotspots; **diff mode gz-vs-rg** ("where does gzippy spend N% more"). Consumes trace_timeline.rs (`trace_v2`) schema. | `scripts/timeline_analyze.py` | `python3 …timeline_analyze.py gz.json [rg.json]` |
| `trace_timeline` instrument | A | Span/timeline Chrome-trace emitter (per-tid B/E spans, `wait.*`, `wait_for` edges) — feeds the two analyzers above. | `src/decompress/parallel/instruments/trace_timeline.rs` | `GZIPPY_TIMELINE=/t.json` (+`GZIPPY_TRACE_DETAIL`, `GZIPPY_PID`) |
| `trace_jsonl` instrument | A | Per-event JSONL trace (phase wall, partition outcomes) — feeds `parallel_sm_log_summary.py` / `analyze_sm_logs.py`. | `src/decompress/parallel/instruments/trace_jsonl.rs` | `GZIPPY_LOG_FILE=/log.jsonl` |
| `parallel_sm_log_summary.py` / `analyze_sm_logs.py` | A | Aggregates the JSONL log: per-event duration pct, phase wall, partition outcome; `analyze_sm_logs` adds **decode-NOT-STARTED stall analysis** (per-stage). | `scripts/parallel_sm_log_summary.py`, `scripts/analyze_sm_logs.py` | `python3 … /log.jsonl` |
| `tracev2_span_sum.py` | A | Per-span total duration from a trace_v2 JSON (count, sum_ms, p50/p95/max). | `scripts/tracev2_span_sum.py` | `python3 … t.json` |
| `stall_residency` instrument | A/C | STEP-0 stall/occupancy discriminator (per-thread stall residency). | `src/decompress/parallel/instruments/stall_residency.rs` | `GZIPPY_STALL_RESIDENCY_PROBE=1` |
| `perfect_overlap` instrument | C | Warm-chunk prefetch-overlap oracle/stats (decode-ahead so consumer hits only applyWindow). | `src/decompress/parallel/instruments/perfect_overlap.rs` | `GZIPPY_PERFECT_OVERLAP=1` (+`GZIPPY_SEED_WINDOWS`) |
| `slow_knob` instrument | C | **Sleep-vs-spin freq-neutral causal perturbation** of decode / store / marker phases (the Gate-2 pre-gate). Has HITS counters (non-inert proof). | `src/decompress/parallel/instruments/slow_knob.rs` | `GZIPPY_SLOW_MODE`,`GZIPPY_SLOW_KIND`,`GZIPPY_SLOW_DECODE`,`GZIPPY_SLOW_STORE`,`GZIPPY_SLOW_MARKER_MODE`,`GZIPPY_SLOW_HITS`,`GZIPPY_SLOW_MARKER_HITS` |
| `removal_oracle` instrument | C | STORE-removal + symbol-stream NODECODE replay ceiling (removes a region to BOUND the speed-up — the only valid ceiling method). | `src/decompress/parallel/instruments/removal_oracle.rs` | `GZIPPY_ORACLE_NODECODE`,`GZIPPY_ORACLE_NOSTORE`,`GZIPPY_ORACLE_RECORD` |
| `decode_bypass` instrument | C | Whole-chunk replay / sleep-decode bypass oracle. | `src/decompress/parallel/instruments/decode_bypass.rs` | `GZIPPY_BYPASS_DECODE`,`GZIPPY_BYPASS_CAPTURE`,`GZIPPY_BYPASS_REBUILD`,`GZIPPY_BYPASS_META_ONLY`,`GZIPPY_BYPASS_FORCE_CLEAN`,`GZIPPY_SLEEP_DECODE_NS` |
| `seed_windows` instrument | C | Clean-only engine oracle via captured predecessor windows (CAPTURE/SEED/NO_WINDOWS + HITS). | `src/decompress/parallel/instruments/seed_windows.rs` | `GZIPPY_SEED_WINDOWS`,`GZIPPY_SEED_WINDOWS_CAPTURE`,`GZIPPY_SEED_NO_WINDOWS`,`GZIPPY_SEED_NO_BOUNDARIES` |
| `GZIPPY_OVERLAP_WRITER` | A/C | Dedicated overlap WRITER thread (O(1) combine; OFF==inline writev, byte-identical; Linux+regular-fd). Still LIVE (1 src file). | `src/decompress/parallel/…` | `GZIPPY_OVERLAP_WRITER=1` |
| `parallel_scaling_intel.py` / `parallel_scaling_mac.py` | A | Thread-scaling curve (does T2<T1 reproduce uncontended?) interleaved, sha-checked, per corpus. | `scripts/bench/standing/parallel_scaling_{intel,mac}.py` | `python3 …` on guest |
| `scaling_ab.sh` | A | Thread-scaling interleaved A/B gz-purerust vs gz-isal vs rapidgzip, P-core pinned. | `scripts/scaling_ab.sh` | `bash scaling_ab.sh [N]` (in LXC) |
| `prefetch_lifecycle_diff.py` | A | Diffs gz-vs-rg PREFETCH-LIFECYCLE traces (block_fetcher prefetch_new_blocks + vendor patch). | `scripts/prefetch_lifecycle_diff.py` | `python3 …` |

### ⚠ LOST — built then cleaned; DIRECTLY relevant to current scheduling/consumer-pacing causality. RECOVER THESE before rebuilding.
| tool | cat | what it did | recover from (VERIFIED in git) |
|---|---|---|---|
| `GZIPPY_NULL_CONSUMER_WORK` | C | Positive-controlled REMOVAL oracle nulling consumer thread's removable serial compute (`write_all` output + `total_crc.append`) in `drain_one_pending`; default OFF byte-identical. **This oracle already produced the campaign's FIRST positive-controlled GO** (see below). | **commit `593819d7`, branch `origin/consumer-null-oracle`** — `git show 593819d7` |
| `GZIPPY_SLOW_CONSUMER` (+`GZIPPY_CHUNK_PHASE`) | C | Slow-injection of consumer self-time (Task-3 falsifier; causal perturbation, not attribution). | **commit `80a6670d`** (`instrument(probe/parallel-eff): Task-1 gap classification + Task-3 SLOW_CONSUMER falsifier`) |
| `GZIPPY_SLOW_PUBLISH_MODE=N` + `GZIPPY_SLOW_DISPATCH_MODE=N` | C | Two OnceLock SLEEP-kind knobs: inject N ms before each predecessor-window publish (`publish_end_window_before_post_process`, chunk_fetcher.rs:2840) / before dispatch (block_fetcher.rs:950). **Tests whether window-publish / dispatch timing gates the decode wall — the exact consumer-pacing perturbation for the current work.** | **commit `5cfb3345` (`probe(sched-phase0)`)** — slow_knob.rs + chunk_fetcher.rs + block_fetcher.rs |
| `GZIPPY_FRONTIER_PREFETCH_ORACLE` | C | Decode the in-order FRONTIER chunk AHEAD at priority −1 so the consumer arrives to only applyWindow remaining (placement oracle, same decode rate). Ran interleaved gz-BASELINE / gz-ORACLE / rg. **Directly tests "is the T4 gap PLACEMENT/scheduling vs kernel".** | **commit `9ae90482` (`feat(oracle): frontier-placement prefetch oracle (rule-3 removal test)`)** |
| `GZIPPY_SAT_PAR=N` | C | In-flight-depth / saturation-parallelization probe in `block_fetcher` (overrides saturation gate / pool depth) — tests whether prefetch depth starves workers. | **transcript-only (`3ed8f25e`, 2026-05-29) — NOT in any git branch; truly LOST, rebuild** |
| `GZIPPY_PIN_WORKERS` | C | Pin worker threads to cores. | in git history (docs); locate impl via `git log --all -S GZIPPY_PIN_WORKERS -- 'src/*'` |
| `GZIPPY_NO_PREFETCH` / `GZIPPY_MAX_PREFETCH_DEPTH` / `GZIPPY_BURST_PREFETCH` / `GZIPPY_PREFETCH_CACHE_CAP` | C | Prefetch on/off + depth + burst + cache cap (scheduling/in-flight perturbations). `NO_PREFETCH` still LIVE (1 src file); the depth/burst/cap knobs appear LOST. | various | verify each; rebuild the LOST ones |

---

## A — MEASUREMENT (general)

| tool | what | path | STATUS |
|---|---|---|---|
| `fulcrum` (mac-side analyzer) | The SOLE oracle binary — analyzer/gate engine that renders `fulcrum vs/flow/critpath/causal/score/decide/locate`. Runs ON MAC; walls run on guest via decide.sh shipped scripts. | `~/www/fulcrum/target/release/fulcrum` (separate repo) | LIVE (external) |
| `decide.sh` + `_decide_guest.sh` + `lib_decide_guest.sh` | One command: freeze box → run each cell's wall interleave + trace/prof + in-tree kill-switch knob A/Bs on guest → pull artifacts → render `fulcrum decide` ranked table. | `scripts/bench/decide.sh`, `_decide_guest.sh`, `lib_decide_guest.sh` | LIVE |
| `standing.sh` + `standing_report.py` | THE one-command ground-truth rig: build gzippy-native at pinned sha on guest → Gate-0 self-validate → interleaved best-of-N matrix vs rapidgzip (T>1 SOTA) + igzip (T1 SOTA) → ONE gated table + FORK verdict. | `scripts/bench/standing/standing.sh` (+`_standing_guest.sh`, `standing_report.py`) | LIVE |
| `standing_mac.sh` / `mac_rg_gap.sh` / `localize_mac.sh` / `mac_pipeline_components.sh` | Mac-side (M-series) standing + rg-gap + localization + pipeline-component rigs. | `scripts/bench/standing/*` | LIVE |
| `measure.sh` | THE trustworthy compare: interleaved RELATIVE delta only, sha-verify every run, Δ<spread⇒TIE, flags contended box. | `scripts/measure.sh` | LIVE |
| `interleaved_ab.sh` | The only valid wall metric on neurotic (interleaved, jitter-immune). | `scripts/interleaved_ab.sh` | LIVE |
| `kernel_gate.sh` + `_kernel_gate_guest.sh` + `_kernel_gate_analyze.py` | **The gated A/B kernel rig** — Gate-0 (build-flavor FFI-off proof both arms, non-inert .o-diff, routing+sha both arms, A/A self-test, GHz spread, LLC-miss%) + Gate-1 (paired Wilcoxon p<0.01 + bootstrap CI + Δ-vs-spread). Verdict KEEP/TIE. | `scripts/bench/kernel-ab/*` | LIVE |
| distpreload cyc/byte single-core instrument | taskset-isolate one P-core + interleave + best-of-N + GHz-stability self-test → real gated T1 cyc/byte even on a loaded LXC. T1 inner-kernel ONLY (not the T4/T8 wall). | `scripts/bench/kernel-ab/_distpreload_cycbyte_{guest.sh,analyze.py}`, `_distpreload_paired_*` | LIVE |
| `intel_xval_*` | Intel cross-validation build/perf/converge/d1grid harness. | `scripts/bench/kernel-ab/intel_xval_*` | LIVE |
| `matrix_report.py` / `timeit.py` | Matrix renderer / timing helper. | `scripts/bench/*` | LIVE |
| `cleankernel_*` / `kernel_ab_aarch64.py` / `selftime.py` / `clean_core_decomp_mac.py` | Clean-kernel silesia-T4 capture, aarch64 kernel A/B, self-time decomposition, mac clean-core decomp. | `scripts/bench/standing/*` | LIVE |
| perf-stat recipes (cpu_core PMU, IPC, LLC-miss, instructions-load-immune) | Embedded in the guest scripts below. | `scripts/profile_single_member_decompression_x86_64.sh`, `guest_ceiling_bench.sh`, `whole_view.sh`, `alloc_ab_harness.sh`, `intel_xval_perf.sh`, `_distpreload_paired_guest.sh` | LIVE |
| `/usr/bin/time -l` M-series PMU + rdtsc | Mac PMU + rdtsc cross-checks. | `m2_perf.sh`, `m2_perf_fine.sh`, `m2_grid.sh`, `contig_prof.rs` (rdtsc) | LIVE |
| `neurotic_sm_dataplane_bench.sh` | Wall + trace + fulcrum on guest 199 via jump. | `scripts/neurotic_sm_dataplane_bench.sh` | LIVE |
| `_byfn` by-function decomposition | by-function CPU decomposition. | (referenced in transcripts) | check `scripts/bench/deprecated/consumer_block_decompose.py` / `project_wall.py` — DEPRECATED |
| `memlife` instrument | Allocation lifetime / component accounting. | `src/decompress/parallel/instruments/memlife.rs` | LIVE (`GZIPPY_MEMLIFE`) |
| `contig_prof` instrument | contig clean-loop rdtsc class profiler. | `src/decompress/parallel/instruments/contig_prof.rs` | LIVE (`GZIPPY_CONTIG_PROF`,`GZIPPY_SEEDED_BLOCK`) |

## B — FREEZE / QUIESCE

| tool | what | path | STATUS |
|---|---|---|---|
| `bench-lock.sh` | THE single host (neurotic) freeze lifecycle: `acquire [TTL]`/`release`/`status`/`verify`. ALLOWLIST-freezes EVERY LXC except access-needed ones (fixes the old clock_freeze that froze only plex/frigate), no_turbo/uncore/governor lock, SETTLE, VERIFY QUIET (loadavg readback → `BENCH_LOCK=quiet|loaded`). Triple-guaranteed restore (release / systemd watchdog TTL / volatile-sysfs reboot). **Consolidates the 4 old scripts.** | repo mirror `scripts/bench/host/bench-lock.sh`; deployed `/root/bench-lock.sh` on neurotic | LIVE |
| `lib_hostlock.sh` | Laptop-side bracket: `hostlock_acquire`/`verify`/`release` over the -J jump, trap-released always. Knobs `HOSTLOCK_TTL`, `QUIET_MAX_LOADAVG`. | `scripts/bench/lib_hostlock.sh` | LIVE |
| `lib_state.sh` / `gzippy_bench_restore.sh` / `lib_gate.sh` | Host plumbing reused by bench-lock: atomic baseline capture / idempotent restore / **aperf-mperf freq-stability gate**. | `/root/gzippy-bench/*` on neurotic | GUEST-ONLY (host) |
| `clock_freeze.sh` / `host_freeze_watchdog.sh` / `freeze_wrapper.sh` / `freeze_profile.sh` | The 4 original host freeze scripts. | `scripts/host_freeze_watchdog.sh`, `scripts/freeze_wrapper.sh` (repo copies remain) | SUPERSEDED by bench-lock.sh |
| `leader-lock.sh` | Single-leader work lock (operating-model: one leader does the work). | `scripts/leader-lock.sh` | LIVE |
| taskset / cpuset single-core isolation + GHz-stability gate | P-core pinning (CPUS mask) + aperf/mperf GHz spread check; the resolution-floor banner ("BOX NOT FROZEN ⇒ sub-spread deltas read TIE"). | inside `measure.sh`, `scaling_ab.sh`, `_distpreload_cycbyte_guest.sh`, `_kernel_gate_guest.sh` | LIVE |
| boxes.sh / await.sh / guest-status.sh / orphan-check.sh / ensure-corpus.sh | Box registry / wait-for-lock / guest status / orphan check / corpus fetch. | `scripts/bench/*` | LIVE |

## C — CAUSALITY (see also the ★ section above for the scheduling-specific knobs)

| tool | what | path / knob | STATUS |
|---|---|---|---|
| `GZIPPY_KERNEL_INJECT` (+`_MODE`) | Perturbation injector into the inner kernel (asm_kernel/marker_inflate). | `src/decompress/parallel/{asm_kernel,marker_inflate}.rs` | LIVE |
| `GZIPPY_TBUILD_MULT` | Multiply table-build cost (Gate-2 perturbation of table build). | `src/…` (1 file) | LIVE |
| `GZIPPY_THIN_T1_ORACLE` | Thin-T1 removal oracle (bound T1 scaffold-free kernel). | `src/…` (2 files) | LIVE |
| `GZIPPY_FLAT_CLEAN` / `GZIPPY_STATELESS_KERNEL` | Flatten clean path / stateless kernel oracle. | `src/…` (1 file each) | LIVE |
| `GZIPPY_CHUNK_KIB` | Chunk-size knob (scheduling granularity). | `src/…` | LIVE |
| `GZIPPY_NO_HIT_DRIVE` | Disable hit-drive. | `src/…` (1 file) | LIVE |
| `GZIPPY_VII_COVERAGE` | asm coverage counter (decode_one_symbol re-entry % bytes-in-asm). | `benches/engine_isolation.rs` (VAR_VII) | LIVE (bench) |
| `lag_causality_sweep.sh` / `consumer_block_decompose.py` / `project_wall.py` | Lag-causality sweep / consumer-block decompose / project-wall. | `scripts/bench/deprecated/*` | DEPRECATED |

---

## Reusable Gate-0 / self-validation patterns worth copying into the next instrument

1. **Build-flavor (FFI-off) proof on BOTH arms** before any number — `_kernel_gate_guest.sh:108` GATE0a verifies the feature-set fingerprint of each binary (kills the "native-as-isal mislabel" bombshell).
2. **Non-inert proof / .o-diff** — GATE0b disassembles the perturbed symbol and asserts it DIFFERS from baseline; runtime knobs carry HITS counters (`GZIPPY_SLOW_HITS`, `GZIPPY_*_HITS`, seed-windows HITS) asserted `>0` / `==expected` so an inert no-op can't silently measure the normal path.
3. **Routing + correctness both arms** — GATE0c asserts `path=ParallelSM` + `sha==zcat` per corpus per binary.
4. **A/A self-test** — `_kernel_gate_analyze.py:14` requires median(A2−A1)≈0 with CI including 0, else the cell is UNTRUSTED (licenses trusting the box's resolution even on a loaded LXC).
5. **GHz spread + LLC-miss% confounder readback** — reported every run; the "BOX NOT FROZEN ⇒ sub-spread deltas read TIE" banner instead of a false win.
6. **Gate-1 significance** — paired Wilcoxon p<0.01 AND bootstrap CI excludes 0 AND |Δ|>inter-run spread ⇒ KEEP; else TIE (`_kernel_gate_analyze.py:19-21`).
7. **Same /dev/null sink both arms + interleaved best-of-N** — `measure.sh` / `interleaved_ab.sh` (file-sink penalizes the faster arm; absolutes are load artifacts).
8. **tail_metric.py Gate-0** — refuses to print a metric if conservation (busy+idle==span, buckets==chunk count) fails.

## ⚑ ALREADY-FOUND LEVER the supervisor must not forget (relevant to current work)
`git log` shows commit **`2ba5cf4b`**: the `GZIPPY_NULL_CONSUMER_WORK` oracle (#1 below)
produced the campaign's **FIRST positive-controlled GO** — "lever is the consumer
single-core **503 MB OUTPUT-WRITE memcpy** (NOT crc, NOT generic compute), ceiling
**+27% at T8 = ~half the rapidgzip gap**, mechanism pinned by 3 adversarial checks.
Necessary-not-sufficient (ON still 1.32×). Next: get the output write off the in-order
consumer critical path." This is a STRONG-tier (removal-oracle) result aimed squarely at
the current parallel-utilization question — re-verify it at HEAD (`git diff 593819d..HEAD`)
before re-deriving it; the cherry-pickable knob + the GZIPPY_OVERLAP_WRITER tool (LIVE)
are the path to test the "off the critical path" fix.

## How to recover the LOST tools (priority order, all VERIFIED in git unless noted)
1. `git show 593819d7` / `git checkout origin/consumer-null-oracle` → `GZIPPY_NULL_CONSUMER_WORK` (consumer removal oracle; the GO lever above).
2. `git cherry-pick`/port from **`80a6670d`** → `GZIPPY_SLOW_CONSUMER` (+`GZIPPY_CHUNK_PHASE`).
3. From **`5cfb3345`** → `GZIPPY_SLOW_PUBLISH_MODE` + `GZIPPY_SLOW_DISPATCH_MODE` (window-publish / dispatch pacing perturbations).
4. From **`9ae90482`** → `GZIPPY_FRONTIER_PREFETCH_ORACLE` (placement vs kernel oracle).
5. Rebuild `GZIPPY_SAT_PAR` (transcript-only, not in git) + locate `GZIPPY_PIN_WORKERS` impl via `git log --all -S`.
