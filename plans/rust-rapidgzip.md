# Closing the gap to rapidgzip — clean state & lever sequence

*Rewritten clean 2026-05-28. The prior 485-line layered-corrections version is
in git history (`git log -- plans/rust-rapidgzip.md`). Everything below is
current truth, measured on neurotic LXC 199 (i7-13700T, 8P+8E) with the
methodology in §4.*

## 0. TL;DR

The real end-to-end gap to rapidgzip on silesia at T=16 is **~1.7× wall**, and
it is **NOT in the inner Huffman loop** — that is at the Rust-vs-hand-C floor.
The gap is, in descending order of impact:

1. **Parallel scaling** — gzippy plateaus at T=8 while rapidgzip scales to T=16
   (a ~2× *multiplier*, the dominant end-to-end gap). [**CURRENT FOCUS**]
2. **The resumable-contract tax (1.33×)** — [SUBSTANTIVELY CLOSED 2026-05-28]
   the tractable win (inline-match-copy, f01eb74, +5% T16) landed; the rest is
   the silesia floor (avg_batch 1.32 — short literal runs; batching more is
   falsified) + a small-ceiling packed-store (also falsified). See §2.
3. **Allocator / page-faults (~33% CPU)** — partly shared with rapidgzip; the
   easy levers (global rpmalloc, prewarm, hugepage) are falsified. [later]

Strategy (user directive): start in the tight inner loop (done), work outward —
flushing out problems (e.g. the pipe-deadlock test, the bench probe bug) as we
go. So the order is inner → resumable-contract (§2) → parallel scaling (§3) →
allocator.

## 1. Measured baseline (clean, pinned, probe-free)

Per-decode inner-loop instructions (encapsulated `examples/inner_bench.rs`,
single-thread, silesia, instructions exact):

| decoder | instructions | vs libdeflate |
|---|---|---|
| libdeflate (C) | 1.81B | 1.00× |
| our `consume_first` (libdeflate-style scalar) | 2.37B | 1.31× |
| our `ResumableInflate2` (production) | 3.14B | 1.74× (= 1.31× algo × 1.33× contract) |

End-to-end wall, frozen-neighbor pinned A/B (T=16, silesia-large):
rapidgzip ≈ 1.6–1.8× our throughput; the ratio **grows** with thread count
(1.64× at T=4 → 2.12× at T=16) — the signature of a serial bottleneck, not a
per-symbol cost.

## 2. [CLOSED] the resumable-contract tax (1.33×)

`ResumableInflate2` does 1.33× the instructions of the same-repo non-resumable
`consume_first` decoder — the cost of the yield-on-output-fill contract
(pending_match state, FASTLOOP/SAFELOOP alternation, per-iteration yield
bounds). This is real, it ships, and it has a **proven-tractable lever**: the
landed inline-match-copy (commit f01eb74) already cut −22.6% of the resumable
inner instructions and +5.2% T=16 wall by removing the per-match
`copy_match_windowed` call.

CLOSED 2026-05-28 after investigation (asmlens diff + a batch-ratio probe):
- The unroll / FASTLOOP_MARGIN yield-elide is DEAD — the fastloop already elides
  per-symbol yield checks, and the loop condition fires once per
  `decode_one_symbol!` invocation (continue-batching), with the 320 B margin too
  small to amortize further.
- Batch-ratio probe: **avg_batch = 1.32** symbols/invocation on silesia (~68%
  single-symbol) = the silesia floor (short literal runs). Batching more is
  falsified (de-ladder). The residual delta is a small-ceiling packed-store
  (also falsified, +0.4%). (A) loop-cond / (B) speculative-miss / (C) yield
  machinery all ruled out as the dominant tax.
- Net: the tractable part was the landed inline-match-copy. No remaining
  tractable lever here → §3 is the focus. (See project_inner_loop_resumable_tax.)

## 3. CURRENT FOCUS — parallel scaling (T=8 → T=16), the multiplier

gzippy plateaus at T=8; rapidgzip scales to T=16. Suspected serial bottleneck:
the single consumer thread (reorder + CRC + `apply_window`/`replace_markers`
marker resolution) and/or prefetch over-emit (memory note: 1.66× emit ratio)
and buffer-return latency (chunks held to `ChunkData::Drop` in the reorder
buffer — `MAX_LIVE_CHUNKS` instrumentation shows in-flight depth 16/28/45 at
T=4/8/16, tracking the page-fault churn). This is a *multiplier* and the real
reason rapidgzip wins end-to-end — but it's outward of the decoder, so it
follows §2 per the "work outward" plan.

## 4. Methodology (hard-won — do not skip)

- **The shared box can't be quieted below load 4** (noisy neighbors: frigate
  cameras LXC 111, plex 105). For clean wall numbers, **freeze the neighbors**:
  `echo 1 > /sys/fs/cgroup/lxc/{111,105}/cgroup.freeze` (user-authorized for
  tests < 30 min) — see `scripts/freeze_wrapper.sh` (always thaws + watchdog).
  ALWAYS verify they thaw afterward (frozen cameras = down).
- **Instruction/cycle counts are load-independent** (per-process HW counters) —
  use them, not wall, when the box is busy. Pin with `taskset -c 0,2,4,6` to
  P-cores; cpu_atom cycles must read 0 (no E-core spill).
- **Tooling:** `examples/inner_bench.rs` (single-thread inner-decode µbench, no
  probe), `tools/asmlens` (multi-arch x86_64/aarch64 machine-code analyzer:
  disasm + loops + perf dynamic-weight + DWARF + two-symbol diff — built for the
  future arm64/M4 work too), `scripts/{asm_compare,scaling_ab,freeze_wrapper}.sh`.
- **Every claimed win → adversarial Opus advisor review** before finalizing
  (user process rule). Several measurement traps (E-core-spill fake-GHz,
  circular page-faults, a libdeflate sizing-probe contaminating 16% of every
  bench, an un-normalized "0.94×") were caught only this way.
- **Run the test suite on a quiet box, single `cargo test`, under `timeout`**
  (a pipe-deadlock test was fixed 2026-05-28 but the suite is heavy).

## 5. Falsification index (don't re-walk)

Inner-loop levers, all measured neutral/worse — the inner loop is at the floor:
de-ladder 8→3 (+1.3%), u32 packed-store (+0.4%), direct-store-3 (+29%), explicit
BMI2 wrapper (regress; rustc already emits BZHI), scalar dynasm (parity),
TABLE_BITS=13 (flat), AOT fingerprints (0% hit on silesia), noalias (already
correct + exploited), copy_match word-copy (already done for short matches).

Pipeline/allocator levers falsified: global rpmalloc (+167% wall), pool prewarm
(−15%), MADV_HUGEPAGE (−38%), GZIPPY_SHARED_POOL (no measurable effect),
GZIPPY_CACHE_CAP (LRU cap doesn't bound in-flight depth).

## 6. Done-when (the prime directive)

Beat rapidgzip on representative workloads. Concretely: pure-rust within 1pp of
rapidgzip end-to-end on neurotic silesia at T=16 (20-trial paired median,
load<4 or frozen neighbors), byte-identical, 3-oracle fuzz ≥72h clean. The
multiplicative path runs through §3 (parallel scaling); §2 (resumable contract)
is the next tractable step outward from the now-floored inner loop.
