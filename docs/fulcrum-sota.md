# Fulcrum — State of the Art

**Public repo:** https://github.com/JackDanger/fulcrum

Fulcrum is a causal-mechanistic pipeline profiler. It ingests the Chrome-trace
timeline that gzippy emits when `FULCRUM_TRACE=/path.json` is set, plus
optional Coz profiles and Linux `perf` captures, and produces views that fuse
three analysis layers over one span-and-dependency graph: causal wall-elasticity
(Coz virtual speedup), critical-path attribution (wPerf-style consumer-anchored
wait), and hardware mechanism (TMA top-down / PEBS / c2c).

**Role in the gzippy campaign (from CLAUDE.md):** Fulcrum views are
_hypothesis generators, never the verdict_. The verdict is always a causal
perturbation — a `GZIPPY_SLOW_BOOTSTRAP=N` slow-inject or an oracle run that
removes a region entirely — and it must survive a frequency-neutral (sleep-not-
spin) control before being recorded as real. Fulcrum's job is to narrow the
search space and surface the next candidate region for that perturbation.


## Subcommands (verified against the binary at `/Users/jackdanger/www/fulcrum/target/release/fulcrum`)

The complete list, in the order the binary reports them, plus the three-part
`memlife` family. All take a Chrome-trace `trace.json` unless noted otherwise.

### `fulcrum critpath <trace.json> [--heavy-ms 30] [--config profile.json]`

Critical-path analysis anchored on the in-order consumer thread. Walks the
span graph backwards from the consumer's last event, attributing on-path time
to the code regions that gate it. Outputs a ranked attribution table (on-path
share, max span), a consumer BUSY/WAIT split, and a list of heavy long-pole
blockers — the individual chunk decodes whose duration gates the consumer stall.

`--heavy-ms` sets the threshold for "heavy blocker" (default 30 ms).
`--config` selects a vocabulary profile so span names are classified into
regions (`gzippy`, `demo`, or `generic`).

### `fulcrum flow <trace.json> [--whatif stage:factor] [--config profile]`

Multi-stage pipeline flow report. For each configured stage: WALL-CRITICAL time
(on the in-order consumer path), total BUSY time across all threads, SLACK
(busy − wall-critical), thread count, occupancy %, and flags:

- `SERIAL` — the stage runs single-threaded
- `STARVED` — the stage is under-supplied by its upstream
- `wall-dead` — wall-critical share < 3%; speeding this stage cannot move wall

`--whatif stage:factor` projects the wall saving if a named stage ran `factor`×
faster — an upper-bound estimate (critical-path, not a full simulation).

A large busy/slack ratio means the stage is off the critical path; wall
attribution is not the same as wall elasticity.

### `fulcrum causal <trace.json> [--timeline N] [--static-fraction P]`

The speculation-interconnectedness view. Reconstructs each chunk's lifecycle
from `causal.*` instant events and reports four sections:

1. **Runtime window-absent fraction** vs the static boundary fraction
   (`--static-fraction`, default 31%). Δ > 0 means gzippy speculates more than
   the data layout forces; Δ ≈ 0 means speculation is set by the data's block
   structure, not by late publishing.
2. **Window-publish latency distribution** — the decode_start − predecessor_publish
   gap per chunk. Negative values are causally forced speculation (started before
   the window existed). Reports key-mismatch window-absent (the partition-seed
   structural cause) separately from timing-induced speculation.
3. **Dependency timeline** — per-chunk in pipeline order: decode_start, mode
   (clean/ABSENT), publish site and timestamp, consume timestamp. `--timeline N`
   controls how many chunks are shown (default 24).
4. **Data-model tax** — the three-pass cost a window-absent chunk pays but a
   clean chunk never does: decode to u16 write, resolve (replace_markers), narrow
   u16→u8. Reports each pass's total time and the fraction of wall.

### `fulcrum consumer <trace.json> [trace2.json ...]`

Consumer-span decomposition. Computes EXCLUSIVE per-span self-time on the
in-order consumer thread via a proper B/E stack (no nested same-name
double-count). Classifies each span as:

- **WAIT** — blocked on a producer (decode-wait, fetch, prefetch)
- **COMPUTE** — consumer's own serial CPU (narrow, resolve, CRC)
- **OUTPUT** — materializing decompressed bytes to the writer (the floor)
- **IDLE** — loop-umbrella self-time: uninstrumented gap

Emits an IDLE-GAP = span − Σbusy and ASSERTS that busy + idle == span
(surfacing any reconciliation miss as a non-zero exit rather than hiding it).
Pass multiple traces for a per-thread-count table side-by-side.

This is the fix for the `combine_crc` phantom: that 62 ms "serial CRC" reading
was a nested-span double-count. The proper B/E stack shows it is an O(1) CRC
combine of worker-computed values, negligible in self-time.

### `fulcrum schedule <trace.json>`

S1 — the PLACEMENT-vs-RATE arbiter. Classifies every consumer stall
(`wait.block_fetcher_get`) as:

- **PLACEMENT** — an idle worker existed while the frontier chunk was undecoded;
  ready capacity went unused. Lever: queuePrefetchedChunkPostProcessing (eagerly
  hand the next chunk to the first idle worker).
- **RATE** — the frontier chunk was genuinely not decoded; all workers were busy.
  Lever: decode speed (~15% bounded by the inner-loop oracle).
- **SPECULATION-INVALID** — the stall arose from a speculative chunk that was
  later invalidated.

Reports the time-fraction for each verdict and prints the dominant one. The
2026-06-01 clean run found RATE-100% at T8, which closed the placement lever.

### `fulcrum decompose <trace.json> [--config profile]`

Names the model residual. wall = Σ(named consumer regions) + NAMED RESIDUAL.
The residual comes from `getrusage` + `schedstat` counters gzippy emits at
region boundaries (minor faults, major faults, voluntary context switches,
involuntary context switches, runnable time, RSS delta). Reports each residual
component with its magnitude so the unexplained wall time has a name rather
than being absorbed silently.

### `fulcrum model <trace.json> [trace2.json] [--workers T] [--labels A,B]`

Populates the parallel-SM wall-model parameter table from a trace:

- N (chunks), T (workers), f (window-absent fraction)
- d_c (clean decode), d_w (window-absent decode), d_w_eff = f×d_w + (1−f)×d_c
- L_resolve (mean/median/p95 window-publish-to-consume latency) — the central
  model parameter
- frontier (startup ramp), tail (drain)

Predicts: wall = max(worker-bound, publish-chain) + tail, reports the residual
vs observed. With two traces, prints the A−B parameter delta and names the
implied lever with its magnitude.

### `fulcrum vs <A-trace.json> <B-trace.json> [--labels a,b] [--config profile]`

Side-by-side per-span comparison between two tools at the same thread count.
Shows, for each span name shared by both: busy time in A, busy time in B, the
wall-critical delta, and which tool gates the wall more in that span. Spans are
classified by the vocabulary in `--config`. Ranked by wall-critical divergence.

Usage example:

```
FULCRUM_TRACE=/tmp/gz.json gzippy -d -T8 -c corpus.gz > /dev/null
FULCRUM_TRACE=/tmp/rg.json rapidgzip -d -P8 -c corpus.gz > /dev/null
fulcrum vs /tmp/gz.json /tmp/rg.json --labels gzippy,rapidgzip --config gzippy
```

### `fulcrum vs-sweep --at T:a.json:b.json [--at ...] [--labels a,b] [--config c.json]`

Cross-tool divergence report across multiple thread counts. For each T supplied
via `--at T:gzippy.json:rapidgzip.json`, produces the per-role (dispatch /
decode / resolve / consumer-wait / write) busy + wall-critical breakdown ranked
by wall-critical divergence, with a top-line lever per T and a cross-T scaling
matrix. Repeat `--at` once per thread count to populate the full matrix.

Both traces must share the parallel-SM span vocabulary.

### `fulcrum rank <trace.json> [profile.coz] [perf_report.txt] [--config profile.json] [--topdown td.txt]`

Fuses the three analysis layers into one ranked lever list:

- Critical-path attribution from the trace
- Coz wall-elasticity curves (if `profile.coz` supplied)
- Linux `perf` function-level cycles% (if `perf_report.txt` supplied)
- TMA top-down bound (if `--topdown td.txt` supplied — a `perf stat --topdown`
  capture)

The combined rank is by on-path share × Coz elasticity × mechanism severity.
A region that scores high on all three is the candidate for the next causal
perturbation.

### `fulcrum region-hw <trace.json> <perf_script_mem.txt> [perf_stat_intervals.csv] [--config c.json] [--topdown td.txt]`

Joins PEBS memory-load samples from `perf mem record` into the trace's region
time windows, producing a per-region hardware table: LLC-miss rate, DRAM-load
fraction, bandwidth. Optional `perf stat --A` interval CSV adds a time-series
view. Reconciles against the run-level TMA if `--topdown` is supplied, flagging
divergence between per-region and global accounting.

Capture recipe:

```bash
FULCRUM_TRACE=/tmp/tl.json FULCRUM_TRACE_CLOCK=monotonic <bin> <args>
perf mem record -k CLOCK_MONOTONIC -o /tmp/mem.data -- <bin> <args>
perf script -i /tmp/mem.data -F time,data_src > /tmp/mem.txt
fulcrum region-hw /tmp/tl.json /tmp/mem.txt --config gzippy
```

The trace must be captured with `FULCRUM_TRACE_CLOCK=monotonic` or the PEBS
join is invalid (timestamps on different clocks).

### `fulcrum coz-parse <profile.coz> [--config profile.json]`

Parses a Coz causal profile into per-region wall-elasticity curves and a ranked
per-line table. Elasticity = ∂(program-speedup) / ∂(region-speedup); values
near 0 mean speeding the region has no wall effect; values near 1 mean it is
the sole bottleneck. Reports median, IQR proxy, and the single highest-
confidence line (the peak-line) for each region.

### `fulcrum coz-jsonl <profile.jsonl> [...]`

Ingests modern Coz `profile.jsonl` format (pass multiple runs for statistical
power) and reports per-region causal impact folded by source file. Folds
`path/to/file.rs:line` → `file.rs` (line-level Coz is too noisy to trust
individually).

### `fulcrum sweep capture --spec s.json --out DIR` / `fulcrum sweep mine DIR [--config region.json]`

Exhaustive thread-count causal sweep. `capture` runs the tool under Coz across
all thread counts in the spec and saves one `profile.coz` per T into DIR.
`mine` post-processes the directory offline (re-runnable), producing a per-T
per-region elasticity matrix.

### `fulcrum mech-report <perf_report.txt>`

Parses a `perf report --stdio -n` text file and prints the top-25 functions by
cycles%. Standalone; does not need a trace.

### `fulcrum xtool --input <name> --tool name:topdown.txt:report.txt[:mbps] [--tool ...]`

Folds per-tool `perf stat --topdown` + `perf report` captures into one
comparable cross-tool accounting on the same input. Each `--tool` spec is
`name:topdown.txt:report.txt` with an optional `:mbps` throughput figure.
Produces a side-by-side TMA table plus the top functions per tool.

### `fulcrum compare --spec compare.json [--samples 5] [--strict-contention] [--timeout-s 120]`

Fair cross-tool benchmark from a generic `--spec` JSON (no tool names baked in
to the binary). Verifies every output's sha256 against a reference, detects
interpreter-wrapped binaries and subtracts per-invocation startup overhead,
uses each tool's documented best configuration, interleaves best-of-N runs with
contention detection, and sweeps corpus × thread-count cells.

### `fulcrum audit --spec compare.json --claim "<stated perf claim>" [--samples 5]`

Runs `compare` and then validates a stated performance claim against the
resulting matrix. Outputs one of:

- **SURVIVES** — the claim holds across the measured cells (exit 0)
- **NARROWS-TO-SCOPE** — the claim holds in a subset of the cells but not all (exit 1)
- **FALSE** — the claim does not hold in any cell (exit 1)

### `fulcrum validate <trace.json> [profile.coz] [--config profile.json]`

Checks the trace + Coz profile against configured ground truth. The gate: known
facts about the system (e.g. "the frontier-placement oracle ties wall") are
stated in the config, and `validate` asserts they hold in the captured data.
FAILS loudly if they diverge so a broken instrument is not trusted silently.

### `fulcrum mech-caps`

Reports this host's hardware-counter availability — TMA events, PEBS
capabilities, c2c support. Cross-arch (never x86-only on arm). Run once on a
new bench host to know which `perf` captures are possible.

### `fulcrum plan --bin <path> [--args "..."] [--scope %/src/%] [--cpus 0,2,4,6] [--iters 200]`

Prints the complete Coz + perf capture workflow for a binary: the exact
`FULCRUM_TRACE=`, `coz run`, `perf stat --topdown`, `perf record`, `perf
report`, and `fulcrum rank` invocations, parameterized with the binary, args,
CPU pin set, and source scope.

### `fulcrum memlife <run.json>`
### `fulcrum memlife vs <A.json> <B.json>`
### `fulcrum memlife growth <T1.json> <T8.json>`

Cross-tool, per-buffer ATTRIBUTED memory-lifecycle breakdown. Introduced on the
`feat/memlife-attribution` branch and now available on `consolidate/campaign`.

The view answers the question the page-fault correlation could not: _which
specific buffer's traffic is the excess, tied to which code site?_ Three modes:

- **Single-run** (`fulcrum memlife <run.json>`): per-component table showing
  alloc / written / read / copied bytes normalized per MB decoded, plus the
  alloc-path split (rpmalloc-span / rpmalloc-huge / glibc / pool-hit) and the
  closure check.
- **Cross-tool** (`fulcrum memlife vs A.json B.json`): side-by-side A vs B
  per-MB delta for each component and each phase (alloc, written, read, copied).
  Positive Δ means A (gzippy) is heavier in that component. Each component row
  is tied to a `file:line` code site.
- **Growth** (`fulcrum memlife growth T1.json T8.json`): one tool at two thread
  counts — the written bytes/MB delta per component shows which buffers grow
  with T (the destructive-contention smoking gun).

The data source on the gzippy side is
`src/decompress/parallel/memlife.rs` (activated by `GZIPPY_MEMLIFE=/path.json`).
The module records exact byte totals as process-global atomics keyed by a fixed
component enum; it is inert unless the env var is set (one relaxed-atomic load
on the hot path). Components:

| Component | gzippy buffer | rapidgzip counterpart |
|---|---|---|
| `DataWithMarkers` | `ChunkData::data_with_markers` (Vec\<u16\>) | `DecodedData::dataWithMarkers` |
| `Data` | `ChunkData::data` (u8, clean bulk) | `DecodedData::data` |
| `Narrowed` | `ChunkData::narrowed` (u8, resolve output) | NONE — rapidgzip resolves in-place |
| `Window` | 32 KiB tail + WindowMap storage | `WindowMap` (compressed) |
| `OutputWrite` | bytes streamed to writer | `toIoVec` writev gather |

The `Narrowed` component is the in-place-resolve target: gzippy allocates +
writes a fresh u8 buffer during resolve; rapidgzip reinterprets its marker
buffer in place (no alloc, no second buffer). The DELTA on `Narrowed`
alloc+write is the exact byte-traffic cost of that structural divergence.

**Closure check** — the honesty gate. `Σ(fresh rpmalloc component alloc)` is
compared against the independently measured allocator total (every
`rpmalloc_alloc` call). Residual > 15% flags the attribution as untrustworthy.
A getrusage minflt sanity ratio is reported separately. The closure check is
mandatory because two instruments in this campaign were silently broken: a
clean-window oracle that re-ran the bootstrap, and a trace that emitted empty
output.


## Configuration profiles

`--config` accepts a JSON file path or a built-in profile name:

- `gzippy` — the worked-example vocabulary for gzippy's parallel-SM span names
- `demo` — matches `examples/toy_pipeline.rs` (the default when no --config is given)
- `generic` — no vocabulary; classifies spans via the universal wait convention
  (any span named `wait.*` is WAIT, anything else is COMPUTE)

The consumer, flow, and vs views classify spans entirely from the config, so
they run on any pipeline's span vocabulary without changes to the binary.


## The measurement process

The following rules govern how Fulcrum output is used — they distill CLAUDE.md's
Measurement PROCESS section and the MEMORY feedback files.

**Perturb, do not attribute.** Fulcrum tells you where time goes; it does not
tell you whether speeding that region moves the wall. The only verdict is a
causal perturbation: change region R's time by a known factor via
`GZIPPY_SLOW_BOOTSTRAP=N` (spin or sleep N% of R's own measured time) and
measure the interleaved wall response. A monotonic proportional response means
R is on the critical path; a flat response means R is slack. A region can
consume large CPU and be 100% wall-neutral — the 212 ms `absorb_isal_tail` copy
is the confirmed example (eliminating it moved wall 0%).

**Always run a frequency-neutral control.** A busy-spin depresses all-core
turbo and can inflate the measured delta. Re-run the perturbation with a sleep
variant; if the delta survives, the criticality is real.

**Slope ≠ ceiling.** Slowing a critical region always adds wall; speeding it
helps only until the next component binds. To bound a speed-up lever you must
REMOVE the region (an oracle) and measure. Never extrapolate the slow-down slope
through an unlocated knee. The page-fault result is the confirmed example:
gzippy faults 2.55× more than rapidgzip, and three removal oracles at T4/8/16
showed 0% wall savings — the faults are overlapped slack, not on the critical
path.

**Validate the instrument first.** Run a positive control (binary vs itself
must read 1.0 ± spread) and a negative control before trusting any number from
a new instrument or a new capture. Two instruments in this campaign were
silently broken before the validation rule was enforced.

**Disproof-driven.** State the claim, then actively try to break it. Δ < inter-
run spread ⇒ TIE on that run. A TIE is not a refutation of the direction; a
correct change is kept and layered even on a TIE. To reject a lever you need
either (a) a concrete mechanism of what goes wrong AND how rapidgzip avoids it,
or (b) a specific measurement showing it makes a named metric worse.

**The frozen-clock clean-bench harness.** All authoritative numbers come from
`scripts/bench/clean_bench.sh` on neurotic: frozen host clock (host cgroups via
Proxmox), uncore-lock, watchdog-restore, interleaved best-of-N ≥ 7 with
sha-verified output. The local `make` is for iteration only; `make ship` is the
verdict. Hand scripts produce phantoms (the `combine_crc` 62 ms reading was a
nested-span double-count in a hand script; `fulcrum consumer` showed it is O(1)
in self-time).

**Numbers from the fullest Fulcrum test.** Extend the instrument before trusting
a partial one. A correct, validated, complete instrument beats a fast partial
one. The reconciliation assert in `fulcrum consumer` (busy + idle == span, hard
failure if not) is the enforcement mechanism.


## How we drove the campaign with Fulcrum

Fulcrum generated three classes of hypothesis during the campaign; each was
either confirmed or refuted by a causal perturbation.

**Three-bucket TMA (instructions / IPC / memory stalls).** The 2026-05-28
three-way comparison (`fulcrum xtool`) showed gzippy and rapidgzip execute
essentially equal instruction counts (0.98× ratio) but gzippy's IPC is 1.08 vs
rapidgzip's 1.42 and it has 2.55× more page faults. This pointed at memory
stalls, not instruction-count excess, as the gap source. The page-fault removal
oracles (three independent measurements at T4/8/16) then refuted the fault
_cost_ hypothesis: the faults are a correlate of gzippy's allocation pattern,
not themselves wall-rate-limiting. The memory-stall hypothesis redirected to the
in-order consumer window-resolution chain (L_resolve in the model).

**Schedule RATE vs PLACEMENT (`fulcrum schedule`).** The 2026-06-01 clean run
classified every consumer stall at T8 as RATE-100% (frontier not decoded, all
workers busy). This killed the placement lever (eager successor hand-off) and
bounded the decode-speed lever at ~15% of wall via the inner-loop oracle.
The fill lever (consumer-wait 167 ms vs rapidgzip 18 ms) was found already
shipped (priority -1 in `chunk_fetcher.rs`) and oracle-tied — closing that
direction.

**Memlife per-buffer attribution (`fulcrum memlife`).** The `Narrowed` component
delta (gzippy allocates + writes a fresh u8 resolve buffer; rapidgzip resolves
in-place) localized the structural divergence to the data-plane output tail.
This is the live target of the `reimplement-isa-l` branch work: port
rapidgzip's in-place resolve + writev output to eliminate the `Narrowed` buffer
and the extra write pass.

The ledger of what Fulcrum hypotheses were confirmed and which were refuted is
maintained in `docs/perf/` (per-session measurement notes) and the MEMORY index
at `~/.claude/projects/.../memory/MEMORY.md`.


## Discrepancies from CLAUDE.md's subcommand list

CLAUDE.md lists four views: `fulcrum vs A B`, `fulcrum flow`, `fulcrum
critpath`, and `fulcrum causal`. The binary has all four plus sixteen additional
subcommands: `consumer`, `schedule`, `memlife`, `decompose`, `model`,
`vs-sweep`, `coz-parse`, `coz-jsonl`, `sweep`, `mech-report`, `rank`,
`region-hw`, `xtool`, `compare`, `audit`, `validate`, `mech-caps`, and `plan`.
The CLAUDE.md list predates these additions; the authoritative list is the
binary's `--help` output reproduced in the Subcommands section above.
