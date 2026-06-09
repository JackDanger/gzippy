# gzippy decode campaign — HANDOFF (2026-06-09)

Branch `reimplement-isa-l`, HEAD `d56cb0f5`. **GOTCHA: `d56cb0f5` is LOCAL-ONLY / UNPUSHED** —
`origin/reimplement-isa-l` points at the ancestor `7bf26096`, so a fresh clone gets the WRONG
(older) tree. Work from the existing local checkout (`/Users/jackdanger/www/gzippy-reimplement-isal`);
push d56cb0f5 before any clone-based work. This is the authoritative handoff. The exhaustive
record is `plans/disproof-ledger.md` (DIS-1..29) + `plans/orchestrator-status.md`; this file is
the curated, gated synthesis. Every claim below survived an independent Opus disproof gate
(verdicts in `plans/*-gate*.md`); claims that did NOT survive are listed under DEAD so you don't
re-chase them.

## 1. MISSION
Two flag-gated parallel-single-member gzip decode builds on x86_64, to parity with rapidgzip:
- **gzippy-isal** — faithful port of rapidgzip's WITH_ISAL pipeline; hands the clean tail to
  real ISA-L via C-FFI "at the right spot." The reference/at-parity build.
- **gzippy-native** — same job, PURE-RUST, NO C-FFI; steals ISA-L's techniques (inline asm OK).
  Plus a design thesis: small, shared, hot-in-cache memory so 8/16+ threads stay in cache.
- **BAR (binding, user-set):** a TIE = wall ratio **>= 0.99x rapidgzip at EVERY thread count**,
  not "within spread" / "ties at T8." Both builds, the parity corpus, interleaved + sha-verified.
- Cargo: `gzippy-native = [pure-rust-inflate]`; `gzippy-isal = [pure-rust-inflate, isal-compression]`.
  `parallel_sm` cfg = the production pure-Rust pipeline; `isal_clean_tail` cfg (x86_64+isal) =
  routes the clean tail to real ISA-L FFI.

## 2. THE CENTRAL FINDING (gated, the whole campaign in three lines)
1. **gzippy is STRUCTURALLY FAITHFUL to rapidgzip.** The ISA-L engine kernel is BYTE-IDENTICAL
   AVX2 nasm in both binaries (disasm-proven, DIS-17). Every machinery-defect hypothesis was
   refuted WITH A MECHANISM (see DEAD). The machinery is sound — sometimes *better* than vendor.
2. **The dominant lever is the ENGINE-W** = gzippy's pure-Rust Huffman symbol-rate (~+40%
   instructions vs rg's ISA-L-as-primitive). PRECISION (var8 gate): the full-kernel inline-asm
   prototype VAR_VIII plateaued at ~0.667x ISA-L for the **CLEAN-TAIL inner-Huffman kernel** — that
   bound is NOT directly the u16-MARKER bootstrap (a separate primitive); both are pure-Rust-vs-ISA-L
   symbol-rate, but only the clean-tail's asm ceiling is measured. Closing the engine is a
   multi-session inline-asm rewrite that FAILED its own 0.85 isolation bar (VAR_VIII was a real
   +14.6% over LLVM, salvageable, but sub-ISA-L). **USER's gated decision: fund it or accept.**
   CAVEAT — engine is necessary but NOT sufficient at low-T: even a PERFECT engine (gzippy-isal =
   real ISA-L) still LOSES T1 0.899x / T4 0.906x on silesia. That low-T deficit is an UNSIZED
   structural residual (owed: the OPEN-1 per-chunk/ParallelSM removal oracle), so "native pure-Rust
   >=0.99-every-T is unreachable" is the LEADING HYPOTHESIS, not a proven floor.
3. **Performance is CORPUS-DEPENDENT** (silesia, the campaign's old monoculture, is rg's tuning
   corpus and was misleading). Real scorecard below.

## 3. REAL-WORLD SCORECARD (gated, DIS-28). The T1/T4/T8/T16 cells = WALL ratio rg_wall/gz_wall
(>1 = gz WINS, bar TIE>=0.99). The `compress` column = the corpus's compression ratio (NOT a wall
ratio). DIS-28 measured T1/T8/T16 only; T4 from DIS-24/25 (silesia) — per-corpus T4 unmeasured (—).
| corpus | compress | T1 | T4 | T8 | T16 | note |
|--------|----------|----|----|----|-----|------|
| small (6MB) | 10x | 1.89 W | — | 1.86 W | 1.74 W | startup-dominated WIN |
| nasa (web log) | 9.9x | 0.57 L* | — | 1.04 W | 1.05 W | *T1 hit by the fallback storm (DIS-29) |
| ghcn (numeric CSV) | 7.8x | 0.95 | — | 1.01 T | 0.96 ~L | mixed |
| silesia (rg home) | 3.1x | 0.90 | **0.906 L** | 1.02 W | 0.92 L | T4 is the ONLY-at-T8-win window's neighbor LOSS |
| model (safetensors) | 1.26x | 0.89 | — | 0.685 L | 0.677 L | WORST — large near-incompressible |
- Under BAR-1 (>=0.99 at EVERY thread count) gzippy-isal does NOT pass silesia (T1 0.90, T4 0.906
  both LOSE; only ~T7-T8 wins; T9-T32 lose). The bar is unmet even on rg's home corpus.
- gz **WINS** small + large-compressible@high-T; **TIES** mixed text; **LOSES** large
  near-incompressible (`model`). The deficit scales with **compressed-size -> chunk-count**
  (more chunks = more per-chunk engine-W), all-T (not a T16-scheduler artifact).
- gzippy-NATIVE additionally carries its own **0.667x clean-path engine floor** on top.
- Two T1 numbers exist: production `-p1` on isal routes to **single-shot ISA-L** (a WIN, 1.20x on
  silesia) — but that route lives on UNMERGED branch `owner/t1-singleshot-route`; HEAD d56cb0f5
  still routes T1 to ParallelSM (the 0.90 numbers).

## 3b. GZIPPY-NATIVE — state + development path (UNDER-DEVELOPED vs isal; the next model's focus)
The campaign spent most effort DIAGNOSING gzippy-isal (why it loses to rg). gzippy-native was
developed but NOT pushed to parity. Its state:
- **Shares the ParallelSM pipeline with isal** (so every isal machinery finding/DEAD-lever applies).
  The ONLY production difference: native's clean tail is the pure-Rust FOLD
  (`marker_inflate.rs decode_clean_into_contig`) where isal hands off to ISA-L FFI. Marker bootstrap
  is pure-Rust u16 on BOTH builds.
- **Scorecard (silesia):** T1 0.608x / T4 0.761x / T8 0.915x — further behind isal at every cell,
  by the clean-tail engine gap (isal's clean tail is real ISA-L; native's is pure-Rust).
- **The native blocker = the engine floor.** The full-kernel inline-asm prototype VAR_VIII (native's
  clean tail in `core::arch::asm!`, register-pinned, byte-exact) reached only **0.667x ISA-L** — a
  REAL +14.6% over LLVM's best (salvageable), but sub-ISA-L. The prior per-symbol transliteration was
  a worse NO-GO (re-entry spills). So pure-Rust+asm cannot match ISA-L's hand-tuned AVX2 Huffman; the
  clean-tail (and the marker-bootstrap, same primitive) symbol rate is the floor.
- **Cache-locality / small-memory THESIS (the design goal "small shared hot-in-cache so 8/16+ threads
  stay in cache") — HALF MET:** the dTLB half is CLOSED (incremental-growth, gz dTLB MPKI now BELOW
  rg, DIS-23); the RSS/working-set half is NOT (gz peak RSS +21-25% vs rg — the *touched* working set:
  u16 marker buffers + per-chunk pipeline, NOT the over-reserve [refuted]). Data-cache MPKI is already
  BETTER than rg. So the thesis's PURPOSE (hot-in-cache) is largely achieved; "small footprint" is the
  remaining design clause.
- **Native development path (for the next model):** (a) the engine — salvage VAR_VIII's +14.6% and/or
  the marker-bootstrap asm; this is the only path toward native parity but is plateau-bounded ~0.667x
  ISA-L (so native >=0.99-every-T is leading-hypothesis-UNREACHABLE in pure-Rust+asm — a genuine
  finding to put to the user: relax no-C-FFI, accept "faithful-but-slower," or revisit the bar);
  (b) shrink the touched working set toward rg's (the RSS clause); (c) the OPEN-1 low-T per-chunk
  residual + the OPEN T8+ seeding bug (DIS-29) help BOTH builds.

## 4. THE PLAN (ranked; how we're proceeding)
- **(i) DONE/NEXT — fallback-storm fix (DIS-29, owner-turnable, cheap).** Root: on >8x-compressible
  corpora gzippy reserves `compressed_span * 8` UPFRONT; >8x expansion overflows -> `isal_inflate`
  fills the buffer -> `decompress_deflate_from_bit_into` returns None -> ALL chunks fall back to
  pure-Rust (~7.5x tax; crushes T1-nasa to 0.57). NOT the EOB/JOB-2 contract — reserve under-sizing
  (storm threshold is exactly 8x: ghcn 7.8x no storm, nasa 9.9x storm). VERIFIED (DIS-29): the
  incremental-growth change (branch `owner/isal-incremental-growth`) FIXES the storm BYTE-EXACT at
  low-T (nasa fb 5->1, bignasa 20->1; T1 +20-30%) AND satisfies DIS-23's owed force-regrow byte-exact
  gate (factor=1, ~12 regrows/chunk, byte-exact). BUT do **NOT** flip the always-small factor-4 knob
  default-ON: it REGRESSES sub-8x corpora at low-T (ghcn T1 -18%, regrow churn with no storm to fix)
  and buys ~0 at parallel-T (T8/T16 OFF==ON — high-T chunks are clean continuations that bypass the
  storm-prone path). **PREFERRED production form (owner-turnable): retry-on-`None`-with-growth** —
  keep the 8x upfront reserve as the first attempt (sub-8x pays zero regrow churn) and, on buffer-full
  `None`, retry via the growable path instead of the 7.5x pure-Rust fallback. Strictly dominates both
  arms. (NOTE: the DIS-29 worktree has env-gated `GZIPPY_STORM_DIAG` instrumentation — STRIP before
  merge.) SEPARATE owed bug (DIS-29): at T8+ on compressible corpora, some speculative-seed chunks
  hit ISA-L `ret=-1/-2` (INVALID_BLOCK/SYMBOL) and fall back — a seeding bug independent of reserve
  sizing, caps the high-T compressible engine rate.
- **(ii) MERGE the banked, gated wins** (correctness/perf, ready): T1 single-shot
  (`owner/t1-singleshot-route`, isal beats rg at T1), the dTLB/incremental-growth footprint win
  (`owner/isal-incremental-growth`, dTLB MPKI now below rg, byte-exact), the build.rs comment fix,
  and the JOB-2 SYNC_FLUSH reserve fix (`isal-resync-stored-fixed`, gated-PASS). All on branches,
  uncommitted by policy (commit-when-asked); merge deliberately, one at a time, build-verify each.
- **(iii) USER DECISION — the engine-W asm.** The only lever for the large-near-incompressible
  loss (model 0.68) + gzippy-native's floor. Full-kernel inline-asm rewrite, multi-session,
  bounded at 0.667x ISA-L (failed its 0.85 bar). Fund it or accept the plateau.
- **(iv) USER DECISION — the >=0.99-every-T BAR.** Given the asm plateau, low/high-T parity for
  native pure-Rust is likely unreachable; isal is corpus-dependent. Revisit the bar or scope it.
- **OWED measurements (small):** native-T1 per-chunk/ParallelSM removal oracle (var8-gate, LIVE
  for native, moot for isal-production); a >=200MB >8x corpus to bound the storm's parallel-T cost.

## 5. DEAD — refuted with mechanism; do NOT re-chase (ledger refs)
- u16-output DE-FRAG / flat-buffer (DIS-21): PHANTOM — rg's `m_window16` IS the same ring + drain
  + 128KiB segments (`deflate.hpp:805/1319/1376`); nothing to converge.
- "gz over-partitions vs rg" chunk-count (DIS-25): rg partitions MORE (66 vs 34) and wins; rg's
  chunk count also scales with -P (`ParallelGzipReader.hpp:294-306`).
- E-core UNDER-FEEDING / work-distribution (DIS-27): gz's E-cores are 72% BUSY (busier than its own
  P-cores), not starved — the high-T loss is engine-W amplified through the (shared, in-order)
  pipeline, not a feeding defect.
- Placement/offset-supply, consumer-prefetch (DIS-6); false-sharing/cache/lock contention (DIS-17,
  perf c2c noise floor); consumer-lean D2-D6 (DIS-16 null, bounded <=23ms); marker-FRACTION (DIS-19,
  rg marker-decodes the same ~34.5%); window-present-fraction (gz ahead); out-of-order publish
  (UNFAITHFUL — rg's consumer is strictly in-order, `ParallelGzipReader.hpp:575-628`).
- "flip-to-clean is a divergence" (REFUTED — rg ALSO flips clean tail to ISA-L at 32KiB,
  `GzipChunk.hpp:520-525`). "gzippy-isal clean tail is pure-Rust" (REFUTED — it IS real ISA-L FFI
  in production). serial writev OUTPUT = shared floor (rg pays it too), not a lever.

## 6. HOW TO USE neurotic (the measurement box)
- **Topology:** neurotic = Proxmox host (homelab, reach via `ssh neurotic`). Bench guest = LXC 199
  ("perception"/"trainer", 10.30.0.199), reach via `ssh -J neurotic root@10.30.0.199` (double-hop).
  CPU = **i7-13700T: 8 P-cores w/ SMT (logical 0-15) + 8 E-cores (16-23) = 24 logical.** Two PMUs:
  `cpu_core` = P-cores, `cpu_atom` = E-cores (use the right one per core type).
- **E-core gotcha:** the container is `cores:16` (cpuset 0-15 = P-core SMT threads only; E-cores
  EXCLUDED — `taskset -c 16` fails). To measure E-cores: `pct set 199 --cores 24` (live, no restart)
  then **RESTORE to `cores:16` after** (verify `taskset -c 16` fails + nproc=16). The old "T16 mask
  0-15" is SMT-oversubscribed on 8 P-cores, NOT 16 physical — control for this.
- **Freeze (MANDATORY for wall numbers):** `/root/bench-lock.sh acquire [TTL]` freezes Plex + all
  noisy LXCs (allowlist), sets no_turbo=1 + governor=performance + uncore-pin, verifies quiet via
  INSTANTANEOUS `procs_running` (NOT lagging loadavg), arms a watchdog that auto-restores after TTL.
  `release` thaws all + restores. `status` / `verify`. NEVER bank a number from a thawed/loaded box
  (procs_running gate). The parity spine brackets runs with this automatically.
- **Build:** the repo is pinned at guest `/root/gzippy-bench` (synced) + the corpus `silesia.gz`
  (compressed ~68 MB -> 212 MB). `RUSTFLAGS="-C target-cpu=native"`. Serialize builds with
  `scripts/cargo-lock.sh` (never concurrent cargo). `df -h` around builds (corpora + targets fill
  the volume — a full disk silently breaks builds). Locally you can build+run the x86_64 isal path
  via Rosetta 2 (`--target x86_64-apple-darwin`) for byte-exact iteration without the box.
- **GIT-WORKTREE GOTCHA:** `git worktree add` leaves submodules EMPTY (`vendor/isa-l`,
  `vendor/isal-rs`, zlib-ng) -> isal build fails ("isal-sys source missing"). Fix:
  `git submodule update --init vendor/isa-l vendor/isal-rs`. And an `rsync --delete` FROM an
  empty-submodule worktree WIPES the guest's vendor trees (happened repeatedly) — repopulate from
  the main checkout + rebuild.
- **Measurement spine (use these, don't hand-roll):** `scripts/bench/parity.sh` + `_parity_guest.sh`
  (matched same-sink = both tools -> regular file on /dev/shm [NOT a pipe — pipe backpressure is a
  phantom], env-scrub allowlist, content-fingerprint stale-binary guard, host-freeze HARD-FAIL on
  thaw, interleaved best-of-N>=9 min, sha-verify EVERY run, `path=ParallelSM`/`IsalSingleShot`
  assert, in-script `isal_chunks`/`isal_fallbacks==0` readback). Derived sweeps:
  `oracle.sh` (engine-isolation: GZIPPY_ISAL_ENGINE_ORACLE etc.), `hicurve.sh` (thread-count curve
  + topology pinning), `corpusgen.sh` (multi-corpus). Comparator: rapidgzip 0.16.0 (the `.so` is
  symboled; its CLI is a python wrapper).
- **Proof-of-binary (do this EVERY isal measurement):** env-unset, `GZIPPY_VERBOSE`, assert
  `isal_chunks >= 14` @T4/T8 — it increments ONLY in the real-ISA-L cfg (`gzip_chunk.rs:386`); the
  native stub (:390-400) returns Ok(false) and never increments. A native binary mislabeled as
  isal once produced a false "ISA-L dormant" bombshell — verify the binary, don't trust the label.

## 7. HOW TO USE Fulcrum (the analysis tooling — HYPOTHESIS GENERATOR, never the verdict)
- `scripts/fulcrum_total.py` — the trustworthy whole-system analyzer: no-double-count SELF time,
  asserts busy+idle==span, classifies wait-vs-compute-vs-output, REFUSES if the trace isn't
  window-absent-preserving, has a self-test that must read 1.0 +- spread. Use it for the per-stage
  picture (consumer DECODE-WAIT vs serial OUTPUT vs post-process).
- `fulcrum vs A B` (cross-tool per-span busy + wall-critical), `fulcrum flow` (per-stage
  slack/serial/starved), `fulcrum critpath`, `fulcrum causal` (speculation decode->publish chain +
  window-absent fraction). Capture a trace with `GZIPPY_LOG_FILE=/tmp/sm.log`, summarize via
  `scripts/parallel_sm_log_summary.py` or fulcrum_total.
- **THE RULE (hard-won):** Fulcrum/attribution GENERATES hypotheses; the VERDICT is always a CAUSAL
  PERTURBATION (slow-injection + a frequency-neutral SLEEP control) or a REMOVAL ORACLE measured at
  the interleaved WALL. Instruction-count / busy-time / latency-share are all analyst-biasable and
  have manufactured phantom levers repeatedly. Rule 3: a slow-down slope does NOT bound a speed-up
  ceiling — to bound a speed-up you must REMOVE the region and measure, never extrapolate the slope.

## 8. PROCESS DISCIPLINE (what kept the conclusions honest — keep doing it)
- **GATE EVERY consequential number.** In this env, background owner agents CANNOT spawn sub-agents
  (no Agent/Task tool) — so the SUPERVISOR runs the independent Opus disproof + bankability gate on
  every owner result before trusting/escalating/banking it. This caught (this campaign) two
  machinery-phantoms, a measurement confound, a premature convergence, a mechanism-misattribution,
  and a mislabeled-binary bombshell. Do NOT skip it.
- **Owner agents must run SYNCHRONOUSLY** (block in-turn with `timeout`; partial-then-STOP). The
  recurring death is background-and-yield (kick off a build + a Monitor, then END expecting an
  auto-resume that never comes) — multiple owners died this way. NO Monitor for "tell me when the
  build finishes"; NO yielding mid-work. Reap orphans (the `~/.dotfiles/bin/timeout` wrapper sleeps
  its FULL duration + leaks; the parallel-SM lib tests DEADLOCK multi-threaded — use
  `--test-threads=1` + `timeout`). pgrep-clean local+guest+neurotic before finishing; release the
  box freeze.
- **Governing frames (user-set):** (a) the wall delta vs rg IS a compiled-code/instruction delta —
  find the specific divergence, CONVERGE to rg's behavior. (b) NO work-displacement (don't make one
  region fast by pushing its work elsewhere). (c) faithful-port: a change that makes gzippy UNLIKE
  rapidgzip is forbidden even if it helps the wall (mind: rg is the existence proof — re-read vendor
  source first-hand; the campaign mis-cited vendor TWICE and built phantom levers on it).
- **Corpus:** diversify via **https://squishy.jackdanger.com/** (the canonical source) — silesia
  alone is rg's tuning corpus and misleads.

## 9. KEY ARTIFACTS
- `plans/disproof-ledger.md` (DIS-1..29) — the spine, every finding + refutation with mechanism.
- `plans/orchestrator-status.md` — per-turn checkpoints (newest at top; many marked SUPERSEDED).
- `plans/*-gate*.md` / `plans/*-verdict.md` — the independent Opus gates per claim.
- `plans/STATE.md`, `plans/MORNING-BRIEF.md` — prior synthesis snapshots (this HANDOFF supersedes).
- Branches (gated, unmerged): `owner/t1-singleshot-route`, `owner/isal-incremental-growth`,
  `isal-resync-stored-fixed` (JOB-2), the build.rs comment fix; plus measurement-only worktrees
  (curve/corpus/oracle tooling) to flag to a Steward for merge.
- `scripts/bench/` (parity/oracle/hicurve/corpusgen + cargo-lock + host/bench-lock),
  `scripts/analysis/` (disasm_*, sw_fit), `scripts/fulcrum_total.py`.
