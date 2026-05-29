# gzippy decoder plan

Built for an executor with effectively infinite implementation and maintenance
capacity. Duplication, multiple parallel variants, and "try everything" are
free; the ONLY scarce things are (a) physical truth (measurement validity,
hardware/memory limits) and (b) correctness on untrusted input. The plan
optimizes against those two and ignores human-effort/DRY/maintenance reflexes.

## Deliverable

gzippy decodes every gzip-family input with a pure-Rust DEFLATE engine as the
sole production path — byte-exact under continuous fuzzing, as fast as the
fastest-correct loops we can produce, with `isal-rs`/`libdeflate-sys`/`zlib-ng`
removed from each platform's production dep graph (kept behind a `dev`/`oracle`
feature as permanent CI differential oracles).

**"One engine" is NOT a goal.** Minimizing loop count / line count is a
maintenance reflex and maintenance is free. The goal is the fastest-correct set
of decode loops, however many. If a standalone hand-tuned Clean loop and a
separate hand-tuned Marker loop each beat a shared abstraction, ship both.

## The laws (physical / correctness constraints — these are the real budget)

These are not human-effort rules; infinite capacity means run them MORE, on more
variants, more often. Nothing below is negotiable.

- **Wall perf = interleaved-delta A/B only**, n≥5+ (free → use more). Both
  binaries in one trial loop sharing contention; report the DELTA. Absolute
  numbers are meaningless: the box has ~15-20% jitter. Freeze LXC 111+105, pin
  physical P-cores `0,2,4,6,8,10,12,14`, turbo ON. A delta within jitter is
  NOISE, not a win — banked numbers must clear jitter.
- **Inner-loop perf = deterministic `inner_bench` instructions/byte** (jitter-
  immune): modes `pure`/`consume_first`/`libdeflate`/`bootstrap`/`markers`.
- **Codegen referee:** disassemble each Clean-path FASTLOOP variant; assert zero
  `call` in the hot region + instruction count within ±2% of the best variant.
  This is the RACE REFEREE (not a guard for one chosen design): any variant that
  deopts loses.
- **Dual-arch CI gate:** every commit builds AND runs the corpus on x86_64 AND
  aarch64 (`--no-default-features --features pure-rust-inflate`). The cfg trap
  (x86-gated code silently compiling out on arm64) becomes a loud failure.
- **Continuous dual-oracle fuzz:** cargo-fuzz, both flate2 AND libdeflate
  oracles, adversarial seed corpus (marker/clean straddle, max-distance
  back-refs, degenerate Huffman, stored-block boundaries, empty dynamic blocks,
  multi-member, truncation). Runs forever in CI. A bug is a bug regardless of
  author; exhaustive automated verification replaces human caution.
- **Golden vectors snapshotted from any decoder BEFORE it is deleted/replaced.**
- **Irreversible deletes (a decoder, a C dep) require:** ≥parity proven on that
  platform, fuzz+corpus green, and a SECOND independent measurement run
  (no same-session banking).
- **One gated, bisectable commit per semantic change** (so a regression bisects
  cleanly). This is correctness hygiene, not pacing.
- **Physics that no implementation capacity changes (M0.3, 2026-05-29, interleaved
  best-of-15 T8): the rapidgzip T4–T8 gap is 85.9% STRUCTURAL** — pure 1709 /
  isal 1818 / rapidgzip 2483; only 14.1% (pure→isal) is inner-loop/decoder-
  closable. The decode engine CANNOT beat rapidgzip at T4–T8; that gap is
  pipeline/memory/parallel-scaling.

## Process — how to work (the practices that earned every real result here)

This is the most transferable part of the plan. Follow it; it is why the few
real wins were real and the many dead ends were caught cheaply.

**1. Map the conceptual space before designing.** Spin up subagents to produce
exhaustive maps of the whole surface BEFORE committing to a design — every
decoder, every caller, every `cfg` gate, the duplication, the dead code, the
routing, who-invokes-what. This campaign's design only became sound after two
full code-audit maps (all pure-Rust decoders; the orchestration/routing) replaced
assumptions with file:line ground truth. Never design on a mental model of code
you haven't had mapped.

**2. Measure before you step — full profiles and traces, not point pokes.** Use
TMA top-down (memory- vs core- vs bad-spec-bound), `perf record`/`annotate`
(a flat profile = a diffuse cost, not a hot spot), copy/memmove attribution
(`mmtrace`), decode-phase traces (`GZIPPY_VERBOSE`/`GZIPPY_LOG_FILE`), and
deterministic `inner_bench` ins/byte. **Instrument to SIZE THE PRIZE before
building the fix** — the single highest-ROI move here was adding a counter that
showed a planned lever's prize was 2.7%, not ~50%, killing a multi-day build
before it started. Operational rule: write the lever's expected prize as a
number BEFORE building; if the instrumented prize is below ~2× the box's wall
jitter floor (i.e. not reliably measurable as a win) OR below the threshold you
pre-registered, do not build it — get an advisor to confirm the sizing and the
go/no-go. "Build is free" does NOT mean "build un-sized"; the waste is compute
spent on a fix whose prize you never measured.

**3. Validate the measurement itself — assume it's lying until proven.** More
real progress came from catching bad measurements than from optimizing. Every
recurring trap actually happened here: a +8.6% "win" that was pure ~15-20% box
jitter (fixed by interleaved-delta A/B); a baseline that spanned 3 commits
(fixed by re-baselining to HEAD~1); benchmarking a C library while believing it
was our Rust (fixed by auditing which feature/binary runs); x86-gated code
silently compiling out on arm64 (fixed by dual-arch builds). For ANY surprising
or favorable result, ask "noise? confounded? wrong binary? wrong arch?" and have
an advisor pressure-test it before banking. A within-jitter delta is not a win.

**4. Use Opus advisors liberally and in distinct roles** (Agent, subagent_type
`claude`). They are the cheapest way to clear the path and avoid expensive
mistakes:
- *Clear-the-way:* "what is the highest-leverage next step given X?"
- *Explain surprises:* on ANY unexpected result, consult BEFORE acting on it.
- *Judgement on forks:* design choices, scope, reset/keep decisions.
- *Multi-angle for plan-level decisions:* run critique + premortem + full
  architectural-planning passes in parallel and synthesize (this is how this
  plan was hardened and de-biased).
- *Adversarial sign-off (required, not optional):* before any irreversible
  delete (a decoder, a C dep) and before granting any new structural-cell
  waiver. The author never self-approves these.
- Advisors are adversaries, not cheerleaders: prompt them to REFUTE, and when
  they refute the plan, update the plan — several headline conclusions here were
  advisor refutations that turned out correct (e.g. a wrong lever, a noise win,
  a human-scarcity bias).

**5. Falsification honesty.** Record what did NOT work and why, in the commit
message and in durable memory — negative results are first-class. ~16 levers
were falsified here; writing each down prevented re-attempting them and kept the
perf claims trustworthy. Never present planning, diagnosis, or a within-noise
result as a shipped win.

## Done-test (binary, no self-judged waivers)

1. **Correct:** lib tests + continuous dual-oracle fuzz + silesia md5
   `c070ed84` + multimember/bgzf/>1GiB-stream/seekable-index all byte-exact.
2. **Sole path, per platform:** every format on a flipped platform routes to the
   pure engine; no FFI inflate on its production path.
3. **C gone, per platform:** prod dep graph drops the C libs on a platform only
   after that platform shows measured ≥parity (kept behind `dev`/`oracle`).
4. **Fast, against a PRE-CLASSIFIED cell list:**
   - **Closable (must hit literal ≥parity):** single-member T1, single-member
     T2–T8 (vs libdeflate/ISA-L/zlib-ng), multi-member all-T, bgzf all-T,
     incompressible, L9.
   - **Structural (waiver requires a written measured TMA justification; a NEW
     waiver needs adversarial-advisor sign-off, never self):** tiny-file (~300 µs
     irreducible Rust process startup); single-member T9–T16 on an 8-P-core box;
     single-member T4–T8 vs rapidgzip (M0.3-confirmed 85.9% structural — the
     decoder closes the 14.1% inner-loop slice to reach ISA-L parity, which is
     what enables C deletion; beating rapidgzip is the concurrent
     parallel-pipeline workstream below, not the decoder).

## Method: build all variants, race them, ship the fastest-correct

At every design fork, do not choose-then-defend. Build all credible variants in
parallel and let the referee infra (codegen + inner_bench + interleaved-delta +
fuzz) pick the winner empirically. Specifically the decode-loop race:

- **Clean FASTLOOP:** build a STANDALONE hand-tuned u8 loop (no trait, no
  `S::Marker`, no `const ACTIVE`, no inline-split). It is the default for the hot
  common path (the closable cells). A shared/trait version must EARN a merge by
  proving ±2% identical codegen on the referee — the burden is on the
  abstraction, not on the standalone loop.
- **Marker FASTLOOP:** build a standalone hand-tuned u16 loop AND a shared/trait
  version. Expectation: they converge, because the marker cost (u16 width = 2×
  store/drain bandwidth, ring-modulo addressing, pre-init marker zone, backward
  marker scan) is INHERENT to marker correctness, not abstraction overhead — no
  design removes it. Keep whichever has cleaner codegen; the win here is small
  (M0.3: inner-loop is 14% of the structural gap) so don't over-invest, but
  building both is free.
- Keep the current two-loop implementation as the control baseline until a
  variant beats it correct.

## Work (a DAG, not phases — one gate, then everything parallel)

**GATE (must precede any rewrite/delete — genuine technical dependency).** Each
item tagged with its real status; do NOT proceed to deletes until all are `done`:
- `inner_bench` deterministic ins/byte — **done** (`examples/inner_bench.rs`,
  modes `pure|consume_first|libdeflate|bootstrap`); add a `markers` mode — **TODO**.
- Interleaved-delta wall A/B — **PATTERN established, not yet a committed script
  (TODO):** commit it as `scripts/interleaved_ab.sh` (see Tooling for the exact
  procedure; `scripts/scaling_ab.sh`/`alloc_ab_harness.sh` are starting points
  but verify they interleave per-trial before trusting them).
- Codegen referee — **TODO:** wire `scripts/asm_compare.sh` + `tools/asmlens/`
  into a `make` check that fails if a Clean FASTLOOP variant has any `call` in
  the hot region or is >±2% of the best variant's instruction count.
- Dual-arch CI gate — **TODO (currently UNGUARDED — the marquee trap is live):**
  `.github/workflows/ci.yml` builds aarch64 but runs the pure-rust corpus test
  x86_64-only. Make it BUILD AND RUN the corpus on both arches every commit.
- Continuous dual-oracle cargo-fuzz + adversarial seeds + golden snapshots —
  **TODO:** the 200-case marker fuzz `e630f01` (`src/tests/three_oracle_diff.rs::
  deflate_block_marker_decoder_fuzz`) is the seed; make it continuous cargo-fuzz,
  dual-oracle (flate2 AND libdeflate), with the adversarial seed corpus.
- M0.3 rapidgzip-gap attribution — **done** (classified structural; reproduce via
  the interleaved 3-way A/B in Tooling).

**Then all of these run concurrently (no inter-ordering except the bisectable-
commit rule and per-platform parity gates):**
- **Dead-code deletion:** `double_literal` → bgzf dead decode path → orphaned LUT
  modules (`combined_lut`/`packed_lut`/`simd_huffman`/`two_level_table`/
  `ultra_fast_inflate`). Independent of everything else once the fuzz/golden gate
  exists.
- **The decode-loop race** (above): standalone Clean, standalone Marker, shared
  trait; race; ship fastest-correct. Each variant is its OWN bisectable commit;
  the race/selection is a separate commit — "build all in parallel" never means
  one mega-commit. Wire the bootstrap (marker) + clean paths to the winners;
  delete the loops the race retires — `deflate_block::Block`
  (`src/decompress/parallel/deflate_block.rs`), the decode loops in
  `src/decompress/inflate/consume_first_decode.rs`, and `isal_lut_bulk` — only
  after golden snapshot + a SECOND independent measurement (a fresh freeze window
  / fresh process invocation, not the same run that produced the win).
- **All-platform drivers:** route single-member T1/streaming/>1GiB, multi-member,
  bgzf, and arm64 through the pure engine via a NEW `src/decompress/drivers.rs`
  (or extend `decompress/mod.rs::decompress_gzip_libdeflate`). arm64 scalar
  correctness ships immediately (shared scalar body); the AVX `copy_match` leaf
  (x86) and NEON `copy_match` leaf (arm64) are two independent perf-leaf builds.
- **Per-format / per-platform C deletion:** flip each format's default to the
  pure engine the moment it hits parity on a platform, and drop that platform's C
  dep. "C gone" is the deliverable; it does not wait on perf polish.
- **Algorithmic-gap polish:** close the residual pure→libdeflate inner-loop gap
  (~13→8.6 ins/byte) + the resumable-contract tax. Concurrent; try every lever;
  the only constraint is measurement validity (clear `inner_bench` + interleaved-
  delta; reject within-jitter wins). Bank nothing un-validated.
- **Concurrent structural workstream (NOT the decoder, NOT deferred):** the
  parallel-pipeline / buffer-lifecycle / allocator re-port that addresses the
  85.9% structural rapidgzip gap. Out of the decoder's done-test (it's not a
  decoder problem) but spun up now in its own module — infinite capacity means no
  reason to defer it to "later."

## Current state + where the durable record lives (read FIRST)
**Before picking any lever, read the project memory index** at
`~/.claude/projects/-Users-jackdanger-www-gzippy/memory/MEMORY.md` and the
`project_*` notes it links — they hold the ~16 FALSIFIED levers (copy-elim,
allocator swaps, depth reduction, slab, right-size, prefetch nextNthEviction,
the B1 collapse-prize refutation, etc.) and current-lever status. Re-attempting
a documented falsification is the most common avoidable waste.

Committed on `reimplement-isa-l`: 0c consolidate 2 FASTLOOPs→1 (`d15e35e`, +7%);
A2 subtract-elim (`d4c9294`, −2.5% ins/byte); bootstrap instrumentation refuting
B1 (`ed4ca13`); `inner_bench bootstrap` (`e68be90`, marker path 27.4 ins/byte);
marker-fuzz seed (`e630f01`); M0.3 structural classification; `vector_huffman`
deleted. `inner_bench` live; interleaved-delta is a documented pattern (not yet
a committed script — see GATE).

## Tooling — how to run the laws (so this plan is executable as handed off)

- **The bench box:** x86_64 i7-13700T (8 P-cores + 8 E-cores), inside Proxmox
  LXC 199, reached `ssh -J neurotic root@10.30.0.199`. SECURITY: `root@neurotic`
  is root over the whole Proxmox host — be extremely careful; only freeze
  guests, never reconfigure them. The box checkout is `/root/gzippy`; sync with
  `git fetch && git reset --hard origin/<branch>`.
- **Builds:** pure engine = `--no-default-features --features pure-rust-inflate`;
  C-ISA-L build = `--features isal-compression`; both with
  `RUSTFLAGS="-C target-cpu=native"` on the box. The parallel-SM + `deflate_block`
  code is `cfg(target_arch="x86_64")`-gated — it COMPILES OUT on arm64/Mac, so
  validate those paths on the box, not locally.
- **Interleaved-delta wall A/B (the only valid wall metric):** freeze neighbors
  via `cgroup.freeze` on LXC 111 (frigate) + 105 (plex) — user-authorized for
  <30 min, ALWAYS thaw (trap EXIT + a detached timeout that thaws). Pin physical
  P-cores `taskset -c 0,2,4,6,8,10,12,14`. Run BOTH binaries in ONE loop
  alternating per trial, take best-of-N (min time) each, report the DELTA. The
  absolute swings ~15-20%; the delta is stable to <1%.
- **Deterministic inner-loop perf:** `examples/inner_bench.rs` modes
  `pure|consume_first|libdeflate|bootstrap` (+ add `markers`). `perf stat -e
  instructions -- inner_bench <mode> 5 <gz>`; ins/byte = instructions / (stdout
  total bytes). Jitter-immune; no freeze needed. (Parse gotcha: extract the
  count with `grep instructions | grep -oE "^[ ]*[0-9,]+" | tr -d ", "`.)
- **Profiles/traces:** `perf record`/`report`/`annotate` (build with
  `CARGO_PROFILE_RELEASE_DEBUG=2 -C force-frame-pointers=yes`); TMA via `perf
  stat -M tma_*`; copy/memmove attribution via `tools/bench/mmtrace.c` (LD_PRELOAD);
  decode-phase counters via `GZIPPY_VERBOSE=1` (bootstrap stats) and chunk traces
  via `GZIPPY_LOG_FILE=` + `scripts/parallel_sm_log_summary.py`.
- **Competitive matrix:** FIRST run `tools/bench/build_competitors.sh` (builds
  `ld_gunzip`/`zng_gunzip` from the `.c` sources, inits `vendor/libdeflate` +
  `vendor/zlib-ng` submodules, installs rapidgzip) — they do NOT exist until
  built. Then `tools/bench/matrix.sh` (gzippy vs rapidgzip vs libdeflate vs
  zlib-ng) across archive × thread-count cells.
- **Reproduce the M0.3 structural finding:** interleaved 3-way A/B at T8 of
  pure vs isal vs rapidgzip on `silesia-large.gz` (frozen, pinned); pure→isal is
  the inner-loop (decoder-closable) slice, isal→rapidgzip the structural slice.
- **Correctness oracles:** `src/tests/three_oracle_diff.rs` (gzippy vs libdeflate
  vs zlib-ng + fuzz; `deflate_block_marker_decoder_fuzz` is the marker-path one).
  Prime invariant: `gzippy -d -c silesia-large.gz | md5sum` == `c070ed84…` (this
  is an MD5 of the decoded output, NOT a git commit). Keep the C oracles forever
  behind a `dev`/`oracle` feature (currently `libdeflate-sys`/`libz-ng-sys` are
  plain deps + an `oracle` Cargo feature exists — wiring the full graph behind it
  is part of the C-deletion work, not yet done).

## Prime directive
Correctness over speed, always — enforced by exhaustive automated verification
(continuous dual-oracle fuzz, dual-arch CI, codegen referee) and by the process
above (map → measure-the-prize → validate-the-measurement → advisor-check), not
by going slow.
