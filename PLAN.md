# gzippy decompression — handoff plan (2026-07-07)

**One-line goal:** make gzippy the strictly-fastest gzip **decompressor** per
(arch × threadcount × archive-type) vs rapidgzip / igzip / libdeflate / zlib-ng /
pigz — with **zero env/config to get optimal perf** and **one production decode
path (no split paths)**. Progress is measured only by `fulcrum score`.

Branch: **`reimplement-isa-l`** (this branch). Tip when this was written:
**`dfa67ccc`**. PR #116 (`reimplement-isa-l` → `main`) is open.

---

## 0. THE LAW (non-negotiable — this project has been burned ~11 times by ignoring it)

- **Only a `fulcrum score` (or equivalent gated) measurement is a finding.** Prose,
  reasoning, source-reads, and single-run numbers are HYPOTHESES, never conclusions.
- A win/loss/tie is valid only if: interleaved best-of-N (N≥15), Δ > inter-run spread,
  **replicated on both arches**, `/dev/null` sink both arms, sha-verified output.
- **Gate-2 perturbation must change ONE variable.** (This session's Lever B failed
  because its "locate" compared *different data on different paths* — a confounded
  perturbation. The perf gate caught the regression before it shipped. Do not repeat:
  when you locate, change exactly one thing on the exact same corpus.)
- Byte-exactness is absolute. Every code change must produce output byte-identical to
  `gzip -dc` (differential across corpora + adversarial fixtures) before it can be
  perf-gated. A fast wrong answer is a total failure.
- No env knobs in the production path. No corpus-keyed split paths. A lever integrates
  as the default or it does not ship.

---

## 1. WHAT IS DONE + VERIFIED (do not redo)

**Env-var / config cleanup — COMPLETE, cross-arch verified, on `reimplement-isa-l`:**
- 11 byte-transparent commits (`git log dfa67ccc`), **−10,950 net lines**, the entire
  `src/decompress/parallel/instruments/` directory + `mem_stats` + all
  trace/stats/oracle/monolith/hugepage/prefault/kill-switch/knob/ISA-L-oracle
  scaffolding removed. Binary shrank **−21%** (2.38MB→1.88MB).
- **Live production env reads: 137 → 4** (all intentional keeps: `GZIPPY_DEBUG`,
  `GZIPPY_MIN_BYTES_PER_THREAD` [the one live perf lever], `GZIPPY_PROBE_FILE` +
  `GZIPPY_REGEN_FIXTURES` [test-infra]). The production decode path is config-free.
- Verified on BOTH arches: byte-differential sha==`gzip -dc` (incl. the x86-only
  `asm-kernel` hot loop that M1 can't compile — checked on the AMD box with a 53MB
  real-data differential in asm-active AND asm-fallback modes), full suite 0-failed,
  and a paired N=15 wall-regression smoke = **no regression** (some cells trended
  slightly faster).

**Verified perf levers already landed earlier (in `git log`):** T1
MultiMemberSeq→ParallelSM (`decompress_multi_member_seq_fast`), the serial member-scan
elimination (scan-fix), the T-blind cap deletion, the slab allocator default. Do not
re-derive; they're in.

---

## 2. THE MEASUREMENT SETUP (how to score — this IS the work)

**Tool:** `fulcrum score` (per-cell: gzippy-native vs rapidgzip; ratio = rg_wall /
gz_wall; **≥0.99 = PASS** = gz at-or-faster). Source: `~/www/fulcrum` (branch
`feat/scaling-load-immune`), with the committed fix making `--isal` **optional**
(gzippy is native-only decode now) on branch `score-isal-optional` (`b038641`).
**Deployed and built on the AMD box at `/root/fulcrum`** (the mac-path optional deps
`in-process-gzippy`/`gzippy`/`critpath-libdeflate` are commented out there so it builds).

Run template (on the box where the binaries live; fulcrum score needs Linux `taskset`,
so it does NOT run natively on macOS — for M1 use a hand ns-timer, see §5):
```
/root/fulcrum/target/release/fulcrum score \
  --arch-os amd-zen2-linux --threads <T> --mask <cpu-mask> \
  --corpus <name> --corpus-path <path> \
  --corpus-pin <sha256 of .gz> --decomp-pin <sha256 of `gzip -dc`> \
  --native <gzippy binary> --rg /root/rapidgzip-native \
  --box solvency-amd-zen2 --freeze-method external --freeze-acknowledged \
  --samples 15 --src-sha <sha7> --out-dir /root/score-<tag>
```
It emits a `SCORE:` line per cell with `native=<ratio> PASS|FAIL` and a distribution
tag `RESOLVED | BIMODAL | NOISY`. **Treat BIMODAL/NOISY cells as unreliable** (llama
contention) — re-run them on a quiet box (llama is pausable, see §6).

**Boxes / topology:**
- **AMD** `10.0.2.240` — Zen2 EPYC 7282, 32 threads, 62G RAM, 1.4T disk. Runs the
  user's llama (pausable under the no-orphan protocol; for correctness runs leave it
  up, for clean timing quiesce it). Fully staged: `/root/fulcrum`, `/root/gzippy-cyc`
  (gzippy-native @ dfa67ccc), `/root/rapidgzip-native` (native ELF, NOT the pip wheel).
- **Intel** `trainer` = `10.30.0.199` via `ssh -J neurotic` — 16-core LXC, **8G RAM /
  ~2.6G disk (FRAGILE — do not co-schedule build+test; watch ENOSPC)**. NOT staged:
  no rapidgzip-native, no gzippy@tip build, different corpora. Staging it is owed.
- **M1** local (Apple Silicon) — `rapidgzip 0.16` at `/opt/homebrew/bin/rapidgzip`,
  `libdeflate-gunzip`, `pigz` present; **no igzip / no ISA-L on aarch64** (this is why
  gz dominates rapidgzip on M1 — see §3).

**Corpora (AMD, with pins the tool verifies):** `/root/silesia.gz`, `/root/nasa.gz`,
`/root/archive/storedheavy.gz` (100MB, ratio-0.99 incompressible), `/root/…weights…`,
logs. Preserved score artifacts + M1 numbers: **`scratchpad/handoff-measurements/`**
in this repo.

---

## 3. THE LOSS SURFACE (fulcrum score, N=15, AMD-Zen2, subject dfa67ccc)

gzippy **wins or ties rapidgzip across the large majority of cells.** Full matrix in
`scratchpad/handoff-measurements/amd-baseline-dfa67ccc-SCORELINES.txt`. Summary:

| corpus | T1 | T4 | T8 | T16 | note |
|---|---|---|---|---|---|
| silesia | 1.17 PASS | 0.99 (BIMODAL) | 0.98 (NOISY) | 1.07 PASS | T4/8 noise, ~TIE |
| nasa | 1.52 | 1.04 | 1.02 | 1.00 | all PASS |
| weights | 1.05 | 0.95 (NOISY→flips PASS) | 0.99 (NOISY) | 0.99 (RESOLVED, Δ=1ms TIE) | noise/TIE |
| logs | 1.24 | 0.98 (rg-spread ≫ gap = TIE) | 1.04 | 1.06 | PASS/TIE |
| **storedheavy** | **1.15 PASS** | **0.94 FAIL** | **0.96 FAIL** | **0.94 FAIL** | **the one solid front (RESOLVED, re-confirmed)** |

**M1 (clean ns-timer, N=15, byte-exact):** gz **WINS** storedheavy every T —
ratio(rg/gz) T1=4.05, T4=2.87, T8=3.10. → **the storedheavy loss is AMD-Zen2-SPECIFIC.**
(rapidgzip uses x86 **ISA-L** on incompressible data; it has no ISA-L on aarch64, so gz
dominates M1.) Numbers: `scratchpad/handoff-measurements/m1-storedheavy-result.txt`.

---

## 4. THE ONE OPEN PERF FRONT: storedheavy T4/8/16 (AMD-Zen2 only)

**Confirmed mechanism (Gate-2 perturbation on the real corpus):** gz decodes this
mostly-Huffman-of-incompressible data through the ParallelSM chunk grid (correctly —
see the falsification below). At T>1 the grid allocates a fresh `vec![0u8; chunk]`
output buffer PER CHUNK; on incompressible data (large chunks) the **kernel first-touch
page-zeroing** (`do_anonymous_page`/`clear_page_rep`, ~18% of perf samples) is the
4-6% deficit. rapidgzip reuses a small bounded per-worker buffer, so its pages stay
faulted once. gz WINS storedheavy T1 (1.15) → this is **parallel overhead, not a kernel
deficit**.

### Falsification ledger (DO NOT re-attempt these):
- **Lever B — reroute storedheavy to the stored-stream path** (`perf-storedheavy-islands`,
  now deleted): **FALSIFIED** — regressed every cell (T1 1.15→0.72). storedheavy is NOT
  stored-throughout; it's an 8.6% stored *prefix* then mostly Huffman blocks. **The
  baseline demote-to-grid is CORRECT.** (This is where the confounded-locate lesson came
  from.)
- **Pinned resident buffer pool at T>1** (`7065e1e8`): **FALSIFIED** earlier — +5-16%
  wall / +9-76% RSS. Do NOT pin a large reserve.

### IMMEDIATE NEXT STEP → perf-gate Lever C
**`Lever C` is BUILT + byte-verified but NOT YET PERF-GATED.** Branch
**`origin/perf-storedheavy-poolreuse`** (commit `3219f0f5`, patch
`scratchpad/leverC-poolreuse.patch`). It reuses **small UNPINNED** per-worker output
buffers at T>1 (rapidgzip's model, distinct from the pinned form that regressed in
7065e1e8) so pages stay faulted. Its own report: byte-differential passed (incl.
incompressible fixtures T1/4/8/16), suite green both harnesses, no env knob, no split
path.

**Do this first:**
1. On the AMD box, build `3219f0f5`, byte-check it decodes `/root/archive/storedheavy.gz`
   == `gzip -dc`, then `fulcrum score` it vs rapidgzip on **storedheavy T4/T8/T16**
   (must move toward ≥0.99) AND **silesia-T4 + nasa-T4 + storedheavy-T1** (must NOT
   regress — this is the 7065e1e8 trap; a pool that helps incompressible can hurt
   compressible).
2. Also re-verify byte-exactness independently (differential + the parallel test harness
   — reuse bugs are scheduling-nondeterministic).
3. **If PASS both arches + byte-exact + no win-cell regression** → fast-forward merge
   `3219f0f5` into `reimplement-isa-l`, re-score the full matrix, done with this front.
4. **If it regresses the win cells** (7065e1e8 repeat) → FALSIFY it (delete the branch,
   bank the falsification). Then storedheavy-AMD is an **accepted narrow arch-residual**:
   gz wins it 3-4× on M1 and wins/ties everything else; matching x86 ISA-L's
   incompressible-data throughput in pure-Rust is the hard "copy-floor" front and is not
   worth a regression to the majority.

---

## 5. THEN (to fully close "win every cell")

- **De-noise the 5 NOISY/BIMODAL AMD cells** (silesia T4/T8, weights T4/T8, logs T8):
  re-score on a quiet box (quiesce llama, §6). They were all near-1.0 PASS/TIE, so this
  is confirmation, not expected to reveal a new front.
- **Score the Intel matrix** (stage rapidgzip-native + gzippy@tip + matching corpora on
  the fragile trainer LXC — mind the 2.6G disk) and a full **M1 matrix**. Any new
  RESOLVED FAIL → clean Gate-2 locate (one variable, same corpus) → lever → gate.
- Rebuild `fulcrum score` on Intel + M1 from `score-isal-optional` (the box builds need
  the mac-path optional deps commented out, as done on AMD).

---

## 6. QUIESCING llama for clean timing (allowed by the user, 2026-07-05)

AMD box runs `llama-swap` + `llama-server`. You MAY pause them for certified timing
under the **no-orphan protocol**: synchronous `kill -STOP` → measure → `kill -CONT`,
with a ≤600s watchdog + a trap that CONTs on EVERY exit path, and **verify `ps` STAT
returns to running (not `T`) after every pausing block**. Never leave llama stopped.
For correctness (byte/sha) runs, leave it up — contention is common-mode.

---

## 7. OTHER OPEN ITEMS (non-perf)

- **PR #116** (`reimplement-isa-l` → `main`): MERGEABLE but CI-BLOCKED. The blocker is a
  **pre-existing CI-environment issue**, NOT the cleanup: the `--features
  isal-compression` build fails fast in CI (the `isal-rs` submodule / `ISAL_SOURCE` the
  `isal-sys` build needs), while the exact CI command **builds clean in 45s on the AMD
  box**. Needs the actual "Build Tools" step log (GitHub Actions-read scope) or a CI
  infra fix (ensure the submodule/ISAL_SOURCE before the cargo build). `CodeQL` also
  fails in 3s (likely config). Most other CI cells pass.
- Pre-existing x86 clippy nits: 2× `manual_repeat_n` in `marker_inflate.rs:~5169/5185`
  (asm-kernel-cfg-gated, so M1-invisible) + a bench import — fix before CI green.

---

## 8. KEY FILES

| file | role for this work |
|---|---|
| `src/decompress/parallel/stored_split.rs` | stored-block decode + demote gate (Lever B territory — reverted) |
| `src/decompress/parallel/chunk_buffer_pool.rs` | the buffer pool (Lever C: unpinned reuse at T>1) |
| `src/decompress/parallel/chunk_decode.rs` | grid per-chunk decode + output buffer alloc (the alloc-storm) |
| `src/decompress/parallel/single_member.rs` | routing/selector (`effective_parallel_threads_with`), the one live lever `GZIPPY_MIN_BYTES_PER_THREAD` |
| `src/decompress/mod.rs` | format detect + DecodePath routing |
| `docs/parallel-decode-architecture.md` | the gz→rapidgzip module role-map |

## 9. THE GATE (every lever, no exceptions)
byte-transparent (sha==`gzip -dc` + adversarial fixtures) **AND** `fulcrum score` ≥0.99
on the target cells **both arches** **AND** no regression on the cells gz already wins
**AND** no env knob **AND** no corpus-keyed split path. Miss any → it does not ship.
